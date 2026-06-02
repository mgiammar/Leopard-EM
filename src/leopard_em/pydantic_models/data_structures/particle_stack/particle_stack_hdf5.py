"""HDF5-backed ParticleStack implementation."""

import os
from pathlib import Path

import h5py
import torch
from pydantic import model_validator
from typing_extensions import Self

from .base import _ParticleStackBase
from .utils import (
    _HDF5_IMAGE_STACK_DATASET,
    _HDF5_LOCAL_STATS_GROUP,
    _read_df_from_hdf5_group,
    _write_df_to_hdf5_group,
)


class ParticleStackHDF5(_ParticleStackBase):
    """Particle stack stored entirely within a single HDF5 file.

    The particle table, optional image stack, and optional per-particle local
    correlation statistics are all held in one ``.h5`` file. Two loading modes are
    supported, but they cannot be mixed.

    - **Load from referenced files**: ``image_stack`` and ``local_stats`` are computed
      from the paths stored in the particle table. The HDF5 file stores only the
      particle table (``image_stack_stored=False``).
    - **Load from HDF5**: ``image_stack`` and ``local_stats`` are read
      directly from the HDF5 datasets (``image_stack_stored=True`` and/or
      ``local_stats_stored=True``).

    HDF5 file layout
    ----------------

    ::

        / (root)
        │  attrs: leopard_em_version, extracted_box_size, original_template_size,
        │         image_stack_stored, local_stats_stored,
        │         global_whitening_applied, local_whitening_applied,
        │         global_normalization_applied, local_normalization_applied
        ├─ particles/
        │      particle_id            (N,)   variable-length str  "{mic_stem}_{idx:05d}"
        │      <column>               (N,)   float64 or variable-length str
        │      ...
        ├─ image_stack                (N, box_h, box_w)             float32  [optional]
        └─ local_stats/                                                      [optional]
               correlation_average    (N, valid_h, valid_w)         float32
               correlation_variance   (N, valid_h, valid_w)         float32

    where ``valid_h = extracted_box_size[0] - original_template_size[0] + 1``
    and   ``valid_w = extracted_box_size[1] - original_template_size[1] + 1``.

    Attributes
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    allow_file_overwrite : bool
        Whether to permit overwriting an existing file, by default False.
    image_stack_stored : bool
        True when ``/image_stack`` is present in the HDF5 file.
    local_stats_stored : bool
        True when ``/local_stats`` group is present in the HDF5 file.
    """

    hdf5_path: str
    allow_file_overwrite: bool = False
    image_stack_stored: bool = False
    local_stats_stored: bool = False

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator(mode="after")  # type: ignore
    def validate_hdf5_path(self) -> Self:
        """Validate that the HDF5 path is writable and the overwrite policy is met.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If the path is not writable or the file exists and overwrite is disabled.
        """
        directory = str(Path(self.hdf5_path).parent)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if directory and not os.access(directory, os.W_OK):
            raise ValueError(
                f"Directory '{directory}' does not permit writing to "
                f"'{self.hdf5_path}'."
            )

        if not self.allow_file_overwrite and os.path.exists(self.hdf5_path):
            raise ValueError(
                f"File '{self.hdf5_path}' already exists but "
                f"'allow_file_overwrite' is False."
            )

        return self

    ###########################
    ### Data loading        ###
    ###########################

    def load_df(self) -> None:
        """Load the particle DataFrame from the HDF5 file at ``hdf5_path``.

        Raises
        ------
        FileNotFoundError
            If ``hdf5_path`` does not exist.
        """
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"HDF5 file '{self.hdf5_path}' does not exist. "
                "Pass skip_df_load=True if you intend to write a new file."
            )

        with h5py.File(self.hdf5_path, "r") as f:
            self._df = _read_df_from_hdf5_group(f)

    ###########################
    ### I/O methods         ###
    ###########################

    def to_hdf5(
        self,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> None:
        """Write the particle table and optional tensors to ``hdf5_path``.

        Parameters
        ----------
        include_image_stack : bool, optional
            Write ``image_stack`` to ``/image_stack``, by default False.
            Raises ``ValueError`` if ``image_stack`` is None.
        include_local_stats : bool, optional
            Write per-particle correlation stats to ``/local_stats``, by
            default False.  Raises ``ValueError`` if either stats tensor is
            None.
        """
        with h5py.File(self.hdf5_path, "w") as f:
            # Root attributes — metadata
            f.attrs["leopard_em_version"] = self.leopard_em_version
            f.attrs["extracted_box_size"] = list(self.extracted_box_size)
            f.attrs["original_template_size"] = list(self.original_template_size)
            f.attrs["global_whitening_applied"] = self.global_whitening_applied
            f.attrs["local_whitening_applied"] = self.local_whitening_applied
            f.attrs["global_normalization_applied"] = self.global_normalization_applied
            f.attrs["local_normalization_applied"] = self.local_normalization_applied

            # Particle table
            _write_df_to_hdf5_group(f, self._df)

            # Optional image stack
            if include_image_stack:
                if self.image_stack is None:
                    raise ValueError(
                        "image_stack is None; cannot write to HDF5. "
                        "Call construct_image_stack() first."
                    )
                f.create_dataset(
                    _HDF5_IMAGE_STACK_DATASET,
                    data=self.image_stack.cpu().to(torch.float32).numpy(),
                )
                self.image_stack_stored = True

            f.attrs["image_stack_stored"] = self.image_stack_stored

            # Optional per-particle local stats
            if include_local_stats:
                if (
                    self.local_stats_correlation_average is None
                    or self.local_stats_correlation_variance is None
                ):
                    raise ValueError(
                        "local_stats tensors are None; cannot write to HDF5."
                    )
                local_grp = f.create_group(_HDF5_LOCAL_STATS_GROUP)
                local_grp.create_dataset(
                    "correlation_average",
                    data=self.local_stats_correlation_average.cpu()
                    .to(torch.float32)
                    .numpy(),
                )
                local_grp.create_dataset(
                    "correlation_variance",
                    data=self.local_stats_correlation_variance.cpu()
                    .to(torch.float32)
                    .numpy(),
                )
                self.local_stats_stored = True

            f.attrs["local_stats_stored"] = self.local_stats_stored

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        allow_file_overwrite: bool = True,
    ) -> "ParticleStackHDF5":
        """Load a ``ParticleStackHDF5`` from an existing HDF5 file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file written by ``to_hdf5``.
        allow_file_overwrite : bool, optional
            Passed to the constructor so that the model validator does not
            reject the path of the file being loaded, by default True.

        Returns
        -------
        ParticleStackHDF5
        """
        with h5py.File(path, "r") as f:
            leopard_em_version = str(f.attrs.get("leopard_em_version", "unknown"))
            extracted_box_size = tuple(int(v) for v in f.attrs["extracted_box_size"])
            original_template_size = tuple(
                int(v) for v in f.attrs["original_template_size"]
            )
            global_whitening_applied = bool(
                f.attrs.get("global_whitening_applied", False)
            )
            local_whitening_applied = bool(
                f.attrs.get("local_whitening_applied", False)
            )
            global_normalization_applied = bool(
                f.attrs.get("global_normalization_applied", False)
            )
            local_normalization_applied = bool(
                f.attrs.get("local_normalization_applied", False)
            )
            image_stack_stored = bool(f.attrs.get("image_stack_stored", False))
            local_stats_stored = bool(f.attrs.get("local_stats_stored", False))

            df = _read_df_from_hdf5_group(f)

            image_stack: torch.Tensor | None = None
            if image_stack_stored:
                if _HDF5_IMAGE_STACK_DATASET not in f:
                    raise ValueError(
                        f"'image_stack_stored' is True but dataset "
                        f"'{_HDF5_IMAGE_STACK_DATASET}' is absent in '{path}'."
                    )
                image_stack = torch.from_numpy(f[_HDF5_IMAGE_STACK_DATASET][:])

            local_avg: torch.Tensor | None = None
            local_var: torch.Tensor | None = None
            if local_stats_stored:
                if _HDF5_LOCAL_STATS_GROUP not in f:
                    raise ValueError(
                        f"'local_stats_stored' is True but group "
                        f"'{_HDF5_LOCAL_STATS_GROUP}' is absent in '{path}'."
                    )
                local_grp = f[_HDF5_LOCAL_STATS_GROUP]
                local_avg = torch.from_numpy(local_grp["correlation_average"][:])
                local_var = torch.from_numpy(local_grp["correlation_variance"][:])

        instance = cls(
            hdf5_path=str(path),
            allow_file_overwrite=allow_file_overwrite,
            extracted_box_size=extracted_box_size,
            original_template_size=original_template_size,
            leopard_em_version=leopard_em_version,
            global_whitening_applied=global_whitening_applied,
            local_whitening_applied=local_whitening_applied,
            global_normalization_applied=global_normalization_applied,
            local_normalization_applied=local_normalization_applied,
            image_stack_stored=image_stack_stored,
            local_stats_stored=local_stats_stored,
            image_stack=image_stack,
            local_stats_correlation_average=local_avg,
            local_stats_correlation_variance=local_var,
            skip_df_load=True,
        )
        instance._df = df

        return instance
