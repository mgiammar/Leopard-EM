"""Storage of sparse particle detections in a 2DTM search."""

import h5py
import numpy as np
import pandas as pd
import torch

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


def derive_orientation_grid_from_full_angles(
    euler_angles: torch.Tensor,
) -> tuple[list[tuple[float, float]] | None, list[float] | None]:
    """Split a grid of Euler angles into its out-of-plane and in-plane axes.

    A search grid is a Cartesian product of (phi, theta) pairs and psi values, but
    which of the two varies fastest depends on how the grid was built. ``torch_so3``
    emits *psi-outer* order -- every (phi, theta) pair for the first psi, then every
    pair for the next psi -- whereas a hand-written grid is more often *psi-inner*.
    Both layouts are detected here rather than assumed, because assuming the wrong
    one silently returns nonsense, notably a ``psi_angles`` list of identical values.

    These two axes are descriptive metadata only. The authoritative record of a
    search is the full ``euler_angles`` array, which :class:`CorrelationTable` stores
    verbatim; a ``search_index`` is resolved by indexing into that array, exactly as
    :func:`leopard_em.backend.process_results.decode_global_search_index` does, and
    so does not depend on the layout at all.

    Parameters
    ----------
    euler_angles : torch.Tensor
        All Euler angles used in the search, shape (num_orientations, 3), in ZYZ
        convention (degrees).

    Returns
    -------
    tuple[list[tuple[float, float]] | None, list[float] | None]
        - ``phi_theta_angles``: unique (phi, theta) pairs, one per out-of-plane
          orientation, in the order they appear in the search.
        - ``psi_angles``: unique psi values used in the search.

        Both are ``None`` when ``euler_angles`` does not factor into these two axes
        -- a constrained or subset search, for instance, or a grid in neither of the
        two recognised orders. There is no separable pair of axes to report then, and
        the full angle list is the only faithful description of the search.
    """
    phi_theta, _ = torch.unique(euler_angles[:, :2], dim=0, return_inverse=True)
    psi = torch.unique(euler_angles[:, 2])
    n_phi_theta = int(phi_theta.shape[0])
    n_psi = int(psi.shape[0])

    # Every (phi, theta) x psi combination must be present exactly once.
    if n_phi_theta * n_psi != euler_angles.shape[0]:
        return None, None

    # psi held constant across the first n_phi_theta rows means (phi, theta) is the
    # fast axis, i.e. the grid is psi-outer.
    if bool(torch.all(euler_angles[:n_phi_theta, 2] == euler_angles[0, 2])):
        phi_theta_rows = euler_angles[:n_phi_theta, :2]
        psi_values = euler_angles[::n_phi_theta, 2]
        expected = torch.cat(
            [
                phi_theta_rows.repeat(n_psi, 1),
                psi_values.repeat_interleave(n_phi_theta).unsqueeze(1),
            ],
            dim=1,
        )
    else:
        phi_theta_rows = euler_angles[::n_psi, :2]
        psi_values = euler_angles[:n_psi, 2]
        expected = torch.cat(
            [
                phi_theta_rows.repeat_interleave(n_psi, dim=0),
                psi_values.repeat(n_phi_theta).unsqueeze(1),
            ],
            dim=1,
        )

    # Confirm the guess reproduces the grid, rather than trusting the first rows.
    if not torch.equal(expected, euler_angles):
        return None, None

    phi_theta_angles = [(float(row[0]), float(row[1])) for row in phi_theta_rows]

    return phi_theta_angles, psi_values.tolist()


class CorrelationTable(BaseModel2DTM):
    """Correlation table data structure storing possible detections along a 2DTM search.

    Attributes
    ----------
    correlation_threshold : float
        Pre-defined threshold a cross-correlation value must surpass to be included
        in the correlation table.
    num_observations : int
        Total number of detections in the correlation table (number of search indices
        which surpassed the correlation threshold).
    defocus_offsets : list[float]
        List of defocus offsets (in Angstroms) used in the search.
    euler_angles : list[tuple[float, float, float]] | None
        Every orientation searched, shape (num_orientations, 3), as ZYZ Euler angles
        in degrees and in the exact order the search used them. This is what makes
        `search_index` decodable, and it is written for every new table; it is `None`
        only for tables read back from files written before it existed.
    phi_theta_angles : list[tuple[float, float]] | None
        Out-of-plane rotation angles (in degrees, Euler angles phi and theta, in ZYZ
        convention) used in the search. Descriptive summary of one axis of the grid,
        derived from `euler_angles`; `None` when the search space does not factor
        into separable (phi, theta) and psi axes.
    psi_angles : list[float] | None
        In-plane rotation angles (in degrees, Euler angle psi, in ZYZ convention)
        used in the search. Descriptive, and `None`, under the same terms as
        `phi_theta_angles`.
    search_index : list[int]
        Global search index identifying the defocus offset and orientation of each
        detection, as `defocus_index * num_orientations + orientation_index`. Length
        will be equal to `num_observations`.

        Decode it by indexing `euler_angles` directly -- `search_index %
        num_orientations` gives the row -- and *not* by combining `phi_theta_angles`
        with `psi_angles`. Which of those two axes varies fastest depends on how the
        grid was generated and is not recoverable from the axes alone.
    x : list[int]
        List of x-coordinates (in pixels) of the detections in the micrograph.
    y : list[int]
        List of y-coordinates (in pixels) of the detections in the micrograph.
    correlation_value : list[float]
        List of cross-correlation values for each detection.
    correlation_mean : list[float]
        List of mean cross-correlation values for each detection, calculated across all
        search indices for the same x/y coordinates.
    correlation_variance : list[float]
        List of variance of cross-correlation values for each detection, calculated
        across all search indices for the same x/y coordinates.

    Methods
    -------
    to_dataframe() -> pd.DataFrame
    from_dataframe(df: pd.DataFrame) -> CorrelationTable
    to_hdf5(file_path: str)
    from_hdf5(file_path: str) -> CorrelationTable
    from_match_template_results(...) -> CorrelationTable
    """

    correlation_threshold: float
    num_observations: int

    # Defining and indexing search space
    defocus_offsets: list[float]
    # Authoritative orientation axis; None only for tables from older files.
    euler_angles: list[tuple[float, float, float]] | None = None
    # Descriptive summaries of the grid; None when it does not factor.
    phi_theta_angles: list[tuple[float, float]] | None = None
    psi_angles: list[float] | None = None
    # defocus_index * num_orientations + orientation_index, length == num_observations
    search_index: list[int]

    # Other detection attributes
    x: list[int]
    y: list[int]
    correlation_value: list[float]
    correlation_mean: list[float]
    correlation_variance: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-detection data to a DataFrame.

        Search-space metadata is stored in ``df.attrs`` so that
        ``from_dataframe`` can reconstruct the full object.

        Returns
        -------
        pd.DataFrame
            One row per detection with columns: search_index, x, y,
            correlation_value, correlation_mean, correlation_variance.
        """
        df = pd.DataFrame(
            {
                "search_index": self.search_index,
                "x": self.x,
                "y": self.y,
                "correlation_value": self.correlation_value,
                "correlation_mean": self.correlation_mean,
                "correlation_variance": self.correlation_variance,
            }
        )
        df.attrs["correlation_threshold"] = self.correlation_threshold
        df.attrs["num_observations"] = self.num_observations
        df.attrs["defocus_offsets"] = self.defocus_offsets
        df.attrs["phi_theta_angles"] = self.phi_theta_angles
        df.attrs["psi_angles"] = self.psi_angles
        df.attrs["euler_angles"] = self.euler_angles
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CorrelationTable":
        """Reconstruct a CorrelationTable from a DataFrame produced by ``to_dataframe``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with detection columns and search-space metadata in
            ``df.attrs``.

        Returns
        -------
        CorrelationTable
        """
        return cls(
            correlation_threshold=float(df.attrs["correlation_threshold"]),
            num_observations=int(df.attrs["num_observations"]),
            defocus_offsets=list(df.attrs["defocus_offsets"]),
            phi_theta_angles=df.attrs.get("phi_theta_angles"),
            psi_angles=df.attrs.get("psi_angles"),
            euler_angles=df.attrs.get("euler_angles"),
            search_index=df["search_index"].tolist(),
            x=df["x"].tolist(),
            y=df["y"].tolist(),
            correlation_value=df["correlation_value"].tolist(),
            correlation_mean=df["correlation_mean"].tolist(),
            correlation_variance=df["correlation_variance"].tolist(),
        )

    def to_hdf5(self, file_path: str) -> None:
        """Write this CorrelationTable to an HDF5 file.

        Layout::

            /metadata              (attrs: correlation_threshold, num_observations)
            /search_space/
                defocus_offsets    float32 1-D
                euler_angles       float32 (num_orientations, 3)
                phi_theta_angles   float32 (n, 2)   [omitted if the grid does not
                psi_angles         float32 1-D       factor into separable axes]
            /detections/
                search_index       int32 1-D
                x                  int32 1-D
                y                  int32 1-D
                correlation_value  float32 1-D
                correlation_mean   float32 1-D
                correlation_variance float32 1-D

        Parameters
        ----------
        file_path : str
            Destination HDF5 file path.
        """
        with h5py.File(file_path, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["correlation_threshold"] = self.correlation_threshold
            meta.attrs["num_observations"] = self.num_observations

            search_space = f.create_group("search_space")
            search_space.create_dataset(
                "defocus_offsets",
                data=np.array(self.defocus_offsets, dtype=np.float32),
            )
            if self.euler_angles is not None:
                search_space.create_dataset(
                    "euler_angles",
                    data=np.array(self.euler_angles, dtype=np.float32),
                )
            # Written alongside, not instead of, the full angle list: readers that
            # only want the grid summary keep working, and no reader has to pick.
            if self.phi_theta_angles is not None:
                search_space.create_dataset(
                    "phi_theta_angles",
                    data=np.array(self.phi_theta_angles, dtype=np.float32),
                )
            if self.psi_angles is not None:
                search_space.create_dataset(
                    "psi_angles",
                    data=np.array(self.psi_angles, dtype=np.float32),
                )

            detections = f.create_group("detections")
            detections.create_dataset(
                "search_index",
                data=np.array(self.search_index, dtype=np.int32),
            )
            detections.create_dataset("x", data=np.array(self.x, dtype=np.int32))
            detections.create_dataset("y", data=np.array(self.y, dtype=np.int32))
            detections.create_dataset(
                "correlation_value",
                data=np.array(self.correlation_value, dtype=np.float32),
            )
            detections.create_dataset(
                "correlation_mean",
                data=np.array(self.correlation_mean, dtype=np.float32),
            )
            detections.create_dataset(
                "correlation_variance",
                data=np.array(self.correlation_variance, dtype=np.float32),
            )

    @classmethod
    def from_hdf5(cls, file_path: str) -> "CorrelationTable":
        """Load a CorrelationTable from an HDF5 file written by ``to_hdf5``.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file.

        Returns
        -------
        CorrelationTable
        """
        with h5py.File(file_path, "r") as f:
            correlation_threshold = float(f["metadata"].attrs["correlation_threshold"])
            num_observations = int(f["metadata"].attrs["num_observations"])

            defocus_offsets = f["search_space/defocus_offsets"][:].tolist()
            search_space = f["search_space"]
            full_angles = phi_theta_angles = psi_angles = None
            if "euler_angles" in search_space:
                full_angles = search_space["euler_angles"][:].tolist()
            if "phi_theta_angles" in search_space:
                phi_theta_angles = search_space["phi_theta_angles"][:].tolist()
            if "psi_angles" in search_space:
                psi_angles = search_space["psi_angles"][:].tolist()

            search_index = f["detections/search_index"][:].tolist()
            x = f["detections/x"][:].tolist()
            y = f["detections/y"][:].tolist()
            correlation_value = f["detections/correlation_value"][:].tolist()
            correlation_mean = f["detections/correlation_mean"][:].tolist()
            correlation_variance = f["detections/correlation_variance"][:].tolist()

        return cls(
            correlation_threshold=correlation_threshold,
            num_observations=num_observations,
            defocus_offsets=defocus_offsets,
            phi_theta_angles=phi_theta_angles,
            psi_angles=psi_angles,
            euler_angles=full_angles,
            search_index=search_index,
            x=x,
            y=y,
            correlation_value=correlation_value,
            correlation_mean=correlation_mean,
            correlation_variance=correlation_variance,
        )

    @classmethod
    def from_match_template_results(
        cls,
        processed_correlation_table: dict,
        defocus_values: torch.Tensor,
        euler_angles: torch.Tensor,
        correlation_average: torch.Tensor,
        correlation_variance_map: torch.Tensor,
    ) -> "CorrelationTable":
        """Construct a CorrelationTable from backend outputs.

        Parameters
        ----------
        processed_correlation_table : dict
            Output of ``process_correlation_table`` with an additional ``global_idx``
            key (list[int]). Expected keys: ``threshold``, ``global_idx``, ``x``,
            ``y``, ``correlation``.
        defocus_values : torch.Tensor
            Defocus offsets used in the search. Shape (num_defocus,).
        euler_angles : torch.Tensor
            All Euler angles used in the search, shape (num_orientations, 3), in ZYZ
            convention (degrees), in the order the search used them. Any order is
            accepted -- it is stored verbatim, and ``search_index`` is resolved
            against it.
        correlation_average : torch.Tensor
            Per-pixel mean cross-correlation, shape (H, W).
        correlation_variance_map : torch.Tensor
            Per-pixel standard deviation of cross-correlation, shape (H, W).

        Returns
        -------
        CorrelationTable
        """
        threshold = processed_correlation_table["threshold"]
        global_idx = processed_correlation_table["global_idx"]  # list[int]
        pos_x = processed_correlation_table["x"]  # list[int]
        pos_y = processed_correlation_table["y"]  # list[int]
        corr_values = processed_correlation_table["correlation"]  # list[float]

        # The factored axes are a summary derived from the orientations, and are
        # simply absent for a search space that does not factor.
        phi_theta_angles, psi_angles = derive_orientation_grid_from_full_angles(
            euler_angles
        )

        search_index = (
            list(global_idx) if isinstance(global_idx, list) else global_idx.tolist()
        )

        # Look up per-detection statistics from the pre-computed statistics tensors
        num_observations = len(pos_x)
        if num_observations > 0:
            x_tensor = torch.tensor(pos_x, dtype=torch.long)
            y_tensor = torch.tensor(pos_y, dtype=torch.long)
            det_mean = correlation_average[y_tensor, x_tensor].tolist()
            det_variance = correlation_variance_map[y_tensor, x_tensor].tolist()
        else:
            det_mean = []
            det_variance = []

        return cls(
            correlation_threshold=float(threshold),
            num_observations=num_observations,
            defocus_offsets=defocus_values.tolist(),
            phi_theta_angles=phi_theta_angles,
            psi_angles=psi_angles,
            # Recorded verbatim: this is what makes search_index decodable.
            euler_angles=[(float(p), float(t), float(s)) for p, t, s in euler_angles],
            search_index=search_index,
            x=list(pos_x),
            y=list(pos_y),
            correlation_value=list(corr_values),
            correlation_mean=det_mean,
            correlation_variance=det_variance,
        )
