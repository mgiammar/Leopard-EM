"""Base class for all particle stack Pydantic models."""

import warnings
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict, Field
from torch.utils.checkpoint import checkpoint
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import coordinate_grid
from torch_motion_correction.correct_motion import get_pixel_shifts

from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.utils.image_processing import (
    apply_image_filtering,
    dose_weight_movie_to_micrograph,
    volume_to_rfft_fourier_slice,
)

from .utils import (
    _any_nan_or_inf,
    _leopard_em_version,
    _load_image_2d,
    _parse_json_series_value,
    _parse_mag_matrix_value_to_tensor,
    get_cropped_image_regions,
)


# pylint: disable=too-many-instance-attributes
class _ParticleStackBase(BaseModel2DTM):
    """Base class holding particle stack data, preprocessing state, and compute methods.

    Not intended to be instantiated directly — use ``ParticleStackCSV`` or
    ``ParticleStackHDF5`` depending on the desired storage back-end.

    Attributes
    ----------
    leopard_em_version : str
        Version of Leopard-EM that created this particle stack.  Auto-populated from
        installed package metadata; preserved as-recorded when loading from a file.
    extracted_box_size : tuple[int, int]
        Size of extracted particle boxes in pixels (height, width).
    original_template_size : tuple[int, int]
        Size of the template used during template matching (height, width). Must be
        smaller than or equal to ``extracted_box_size``.
    global_whitening_applied : bool
        True if whitening was computed from and applied to the full micrograph before
        particle extraction.
    local_whitening_applied : bool
        True if whitening was computed from and applied to each individual extracted
        particle box.
    global_normalization_applied : bool
        True if normalization was computed from the full micrograph before extraction.
    local_normalization_applied : bool
        True if normalization was computed from and applied to each extracted particle
        box.
    image_stack : ExcludedTensor
        Stack of extracted particle images, shape ``(N, box_h, box_w)``.
        Not serialized to YAML/JSON.
    local_stats_correlation_average : ExcludedTensor
        Per-particle local mean of the cross-correlation map, extracted from the valid
        cross-correlation region around each particle center. Shape
        ``(N, valid_h, valid_w)`` where
        ``valid_h = extracted_box_size[0] - original_template_size[0] + 1`` and
        ``valid_w = extracted_box_size[1] - original_template_size[1] + 1``.
        Not serialized to YAML/JSON.
    local_stats_correlation_variance : ExcludedTensor
        Per-particle local variance of the cross-correlation map.  Same shape as
        ``local_stats_correlation_average``.  Not serialized to YAML/JSON.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    leopard_em_version: str = Field(default_factory=_leopard_em_version)
    extracted_box_size: tuple[int, int]
    original_template_size: tuple[int, int]

    # Pre-processing state flags
    global_whitening_applied: bool = False
    local_whitening_applied: bool = False
    global_normalization_applied: bool = False
    local_normalization_applied: bool = False

    # Private: tabular data (not part of Pydantic schema)
    # TODO: Move away from having a df-backed implementation in favor of either
    #       getter/setter methods OR private fields for the relevant data.
    _df: pd.DataFrame

    # Image and statistics tensors (excluded from YAML/JSON serialization)
    image_stack: ExcludedTensor
    local_stats_correlation_average: ExcludedTensor
    local_stats_correlation_variance: ExcludedTensor

    def __init__(self, skip_df_load: bool = False, **data: Any):
        """Initialize the particle stack.

        Parameters
        ----------
        skip_df_load : bool, optional
            When True the subclass ``load_df`` is not called automatically.
            Use this when constructing an empty instance before populating
            ``_df`` manually (e.g., during ``from_hdf5``).
        data : dict[str, Any]
            Fields forwarded to the Pydantic constructor.
        """
        super().__init__(**data)
        if not skip_df_load:
            self.load_df()

    def load_df(self) -> None:
        """Load the particle DataFrame from the backing store.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement load_df()")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_position_reference_columns(self) -> tuple[str, str]:
        """Return the y/x position column names to use (refined preferred)."""
        y_col = "refined_pos_y" if "refined_pos_y" in self._df.columns else "pos_y"
        x_col = "refined_pos_x" if "refined_pos_x" in self._df.columns else "pos_x"
        return y_col, x_col

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df_columns(self) -> list[str]:
        """Column names of the underlying DataFrame."""
        return list(self._df.columns.tolist())

    @property
    def num_particles(self) -> int:
        """Number of particles in the stack."""
        return len(self._df)

    # ------------------------------------------------------------------
    # DataFrame accessor / mutator helpers
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Get a column from the underlying DataFrame."""
        try:
            return self._df[key]
        except KeyError as err:
            raise KeyError(f"Key '{key}' not found in underlying DataFrame.") from err

    def set_column(self, column_name: str, value: Any) -> None:
        """Set a column in the underlying DataFrame.

        Parameters
        ----------
        column_name : str
            The column name to set.
        value : Any
            The value(s) to assign.
        """
        self._df.loc[:, column_name] = value

    def get_dataframe_copy(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self._df.copy()

    def get_ctf_kwargs(self, template_shape: tuple[int, int]) -> dict[str, Any]:
        """Assemble CTF keyword arguments from this particle stack's per-particle data.

        Note
        ----
        All particles must share the same voltage, spherical aberration, amplitude
        contrast ratio, phase shift, CTF B-factor, mag_matrix, and Zernike coefficients;
        this is checked internally.

        Parameters
        ----------
        template_shape : tuple[int, int]
            Real-space shape ``(h, w)`` of the template for CTF calculation.

        Returns
        -------
        dict[str, Any]
            Dictionary suitable for passing as ``ctf_kwargs`` to the backend correlation
            functions.
        """
        single_value_columns = [
            "voltage",
            "spherical_aberration",
            "amplitude_contrast_ratio",
            "ctf_B_factor",
            "phase_shift",
            "pixel_size",
            "mag_matrix",
            "even_zernikes",
            "odd_zernikes",
        ]
        for col in single_value_columns:
            if self[col].nunique() > 1:
                raise ValueError(
                    f"Column '{col}' has multiple unique values across particles. "
                    "Currently only a single value across all particles is supported."
                )

        # Columns "mag_matrix", "even_zernikes", and "odd_zernikes" parsed separately
        single_value_columns.remove("mag_matrix")
        single_value_columns.remove("even_zernikes")
        single_value_columns.remove("odd_zernikes")

        # Initialize result dict with values from scalar columns
        result = {col: self[col].iloc[0].item() for col in single_value_columns}

        mag_matrix_tensor = _parse_mag_matrix_value_to_tensor(
            self["mag_matrix"].iloc[0]
        )
        even_zernikes_dict = _parse_json_series_value(self["even_zernikes"].iloc[0])
        odd_zernikes_dict = _parse_json_series_value(self["odd_zernikes"].iloc[0])

        if even_zernikes_dict is not None:
            even_zernikes_dict = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in even_zernikes_dict.items()
            }
        if odd_zernikes_dict is not None:
            odd_zernikes_dict = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in odd_zernikes_dict.items()
            }

        result["mag_matrix"] = mag_matrix_tensor
        result["even_zernikes"] = even_zernikes_dict
        result["odd_zernikes"] = odd_zernikes_dict
        result["template_shape"] = template_shape

        return result

    def build_refined_dataframe(
        self,
        result: dict[str, np.ndarray],
        column_order: list[str],
        prefer_refined_positions: bool = True,
    ) -> pd.DataFrame:
        """Build a refined-results DataFrame from a backend result dict.

        Parameters
        ----------
        result : dict[str, np.ndarray]
            Backend result dictionary with keys: ``refined_pos_y``, ``refined_pos_x``,
            ``refined_euler_angles`` (shape ``(N, 3)``), ``refined_defocus_offset``,
            ``refined_pixel_size_offset``, ``refined_cross_correlation``,
            ``refined_z_score``.
        column_order : list[str]
            Column ordering applied via ``reindex`` before returning.
        prefer_refined_positions : bool, optional
            When True and refined position columns already exist in the DataFrame,
            apply offsets relative to them.  Defaults to True.

        Returns
        -------
        pd.DataFrame
            Refined DataFrame, column-ordered but not written to disk.
        """
        df = self.get_dataframe_copy()

        pos_offset_y = result["refined_pos_y"]
        pos_offset_x = result["refined_pos_x"]
        pos_offset_y_ang = pos_offset_y * df["pixel_size"]
        pos_offset_x_ang = pos_offset_x * df["pixel_size"]

        if prefer_refined_positions and self.get_position_reference_columns() == (
            "refined_pos_y",
            "refined_pos_x",
        ):
            pos_y_col = "refined_pos_y"
            pos_x_col = "refined_pos_x"
            pos_y_col_img = "refined_pos_y_img"
            pos_x_col_img = "refined_pos_x_img"
            pos_y_col_img_angstrom = "refined_pos_y_img_angstrom"
            pos_x_col_img_angstrom = "refined_pos_x_img_angstrom"
        else:
            pos_y_col = "pos_y"
            pos_x_col = "pos_x"
            pos_y_col_img = "pos_y_img"
            pos_x_col_img = "pos_x_img"
            pos_y_col_img_angstrom = "pos_y_img_angstrom"
            pos_x_col_img_angstrom = "pos_x_img_angstrom"

        df["refined_pos_y"] = pos_offset_y + df[pos_y_col]
        df["refined_pos_x"] = pos_offset_x + df[pos_x_col]
        df["refined_pos_y_img"] = pos_offset_y + df[pos_y_col_img]
        df["refined_pos_x_img"] = pos_offset_x + df[pos_x_col_img]
        df["refined_pos_y_img_angstrom"] = pos_offset_y_ang + df[pos_y_col_img_angstrom]
        df["refined_pos_x_img_angstrom"] = pos_offset_x_ang + df[pos_x_col_img_angstrom]

        df["refined_psi"] = result["refined_euler_angles"][:, 2]
        df["refined_theta"] = result["refined_euler_angles"][:, 1]
        df["refined_phi"] = result["refined_euler_angles"][:, 0]

        df["refined_relative_defocus"] = (
            result["refined_defocus_offset"] + self.get_relative_defocus().cpu().numpy()
        )
        df["refined_pixel_size"] = (
            result["refined_pixel_size_offset"] + self.get_pixel_size().cpu().numpy()
        )

        df["refined_mip"] = result["refined_cross_correlation"]
        df["refined_scaled_mip"] = result["refined_z_score"]

        return df.reindex(columns=column_order)

    def get_correlation_stacks(
        self,
        extracted_box_size: tuple[int, int],
        device: torch.device,
        mean_stack: torch.Tensor | None = None,
        std_stack: torch.Tensor | None = None,
        particle_indices: list[pd.Index] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract per-particle correlation mean and variance sub-images.

        Parameters
        ----------
        extracted_box_size : tuple[int, int]
            Size of the sub-region to extract for each particle.
        device : torch.device
            Target device for returned tensors.
        mean_stack : torch.Tensor | None
            Pre-loaded correlation mean maps (full micrograph size).  If None,
            loads from ``correlation_average_path`` column.
        std_stack : torch.Tensor | None
            Pre-loaded correlation variance maps.  If None, loads from
            ``correlation_variance_path`` column.
        particle_indices : list[pd.Index] | None
            Row indices matching pre-loaded ``mean_stack`` / ``std_stack``
            to particles; required when either stack is provided.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(corr_mean, corr_variance)`` sub-image stacks, on ``device``.
        """
        if mean_stack is None:
            corr_avg_images, corr_avg_indexes = self.load_images_grouped_by_column(
                "correlation_average_path"
            )
        else:
            if particle_indices is None:
                raise ValueError(
                    "particle_indices must be provided when mean_stack is given."
                )
            corr_avg_images, corr_avg_indexes = mean_stack, particle_indices

        corr_mean = self.construct_image_stack(
            images=corr_avg_images,
            indices=corr_avg_indexes,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=0.0,
        ).to(device)

        if std_stack is None:
            corr_var_images, corr_var_indexes = self.load_images_grouped_by_column(
                "correlation_variance_path"
            )
        else:
            if particle_indices is None:
                raise ValueError(
                    "particle_indices must be provided when std_stack is given."
                )
            corr_var_images, corr_var_indexes = std_stack, particle_indices

        corr_variance = self.construct_image_stack(
            images=corr_var_images,
            indices=corr_var_indexes,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=1e10,
        ).to(device)

        return corr_mean, corr_variance

    def _load_micrograph_images(
        self,
        micrograph_images: torch.Tensor | None,
        micrograph_indices: "list[pd.Index] | None",
        device: torch.device,
    ) -> tuple[torch.Tensor, list[pd.Index]]:
        """Return micrograph images and matching particle indices.

        Uses pre-provided images when given; otherwise loads from the
        ``micrograph_path`` column.

        Parameters
        ----------
        micrograph_images : torch.Tensor | None
            Pre-loaded micrograph stack, or None to load from disk.
        micrograph_indices : list[pd.Index] | None
            Particle row indices corresponding to ``micrograph_images``;
            required when ``micrograph_images`` is provided.
        device : torch.device
            Target device; images loaded from disk are moved here.

        Returns
        -------
        tuple[torch.Tensor, list[pd.Index]]
            ``(micrograph_images, micrograph_indices)``
        """
        if micrograph_images is not None:
            if micrograph_indices is None:
                raise ValueError(
                    "micrograph_indices must be provided when "
                    "micrograph_images is given."
                )
            return micrograph_images, micrograph_indices

        micrograph_images, micrograph_indices = self.load_images_grouped_by_column(
            "micrograph_path"
        )

        return micrograph_images.to(device), micrograph_indices

    def _apply_global_micrograph_filtering(
        self,
        micrograph_images: torch.Tensor,
        template: torch.Tensor,
        preprocessing_filters: PreprocessingFilters,
        micrograph_indices: list[pd.Index],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply global preprocessing filters to micrograph images.

        Computes per-micrograph projective filters, applies them in Fourier
        space, and returns the filtered micrographs in real space.

        Parameters
        ----------
        micrograph_images : torch.Tensor
            Micrograph image stack ``(N, H, W)``.
        template : torch.Tensor
            Template volume used for filter output shape and device.
        preprocessing_filters : PreprocessingFilters
            Filter configuration.
        micrograph_indices : list[pd.Index]
            Particle row indices per micrograph; forwarded to
            ``construct_projective_filters``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(filtered_micrograph_images, projective_filters)``
        """
        device = template.device
        box_h, box_w = self.extracted_box_size
        img_h, img_w = micrograph_images.shape[-2:]
        filter_output_shape = (template.shape[-2], template.shape[-1] // 2 + 1)

        micrograph_images_dft = torch.fft.rfftn(micrograph_images, dim=(-2, -1))  # pylint: disable=not-callable
        if micrograph_images.requires_grad:
            micrograph_images_dft = micrograph_images_dft.clone()
        micrograph_images_dft[..., 0, 0] = 0.0 + 0.0j

        with torch.no_grad():
            projective_filters = self.construct_projective_filters(
                preprocessing_filters,
                output_shape=filter_output_shape,
                images_dft=micrograph_images_dft.detach(),
                indices=micrograph_indices,
            ).to(device)

        micrograph_images_dft = apply_image_filtering(
            self,
            preprocessing_filters,
            micrograph_images_dft,
            full_image_shape=(img_h, img_w),
            extracted_box_shape=(box_h + 1, box_w + 1),
        )
        filtered_micrograph_images = torch.fft.irfftn(
            micrograph_images_dft, dim=(-2, -1)
        )  # pylint: disable=not-callable

        return filtered_micrograph_images, projective_filters

    def _extract_particle_images(
        self,
        micrograph_images: torch.Tensor,
        micrograph_indices: list[pd.Index],
        movie: torch.Tensor | None,
        deformation_field: CubicCatmullRomGrid3d | None,
        particle_shifts: torch.Tensor | None,
        pre_exposure: float,
        fluence_per_frame: float,
    ) -> torch.Tensor:
        """Extract particle images from micrographs or a motion-corrected movie.

        Parameters
        ----------
        micrograph_images : torch.Tensor
            Micrograph image stack ``(N, H, W)`` after any global filtering.
        micrograph_indices : list[pd.Index]
            Particle row indices per micrograph.
        movie : torch.Tensor | None
            Movie tensor; triggers movie-based extraction when provided
            together with ``deformation_field`` or ``particle_shifts``.
        deformation_field : CubicCatmullRomGrid3d | None
            Continuous deformation field for motion correction.
        particle_shifts : torch.Tensor | None
            Per-particle per-frame shifts ``(T, N, 2)``.
        pre_exposure : float
            Pre-exposure fluence in e-/A^2.
        fluence_per_frame : float
            Fluence per frame in e-/A^2.

        Returns
        -------
        torch.Tensor
            Extracted particle image stack ``(N, box_h, box_w)``.
        """
        if movie is not None and (
            deformation_field is not None or particle_shifts is not None
        ):
            return self.construct_image_stack_from_movie(
                movie=movie,
                deformation_field=deformation_field,
                particle_shifts=particle_shifts,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="reflect",
                padding_value=0.0,
                pre_exposure=pre_exposure,
                fluence_per_frame=fluence_per_frame,
            )
        return self.construct_image_stack(
            images=micrograph_images,
            indices=micrograph_indices,
            extraction_size=self.extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="reflect",
            padding_value=0.0,
        )

    def prepare_images_and_filters(
        self,
        template: torch.Tensor,
        preprocessing_filters: PreprocessingFilters,
        *,
        apply_global_filtering: bool,
        particle_images: torch.Tensor | None = None,
        micrograph_images: torch.Tensor | None = None,
        micrograph_indices: "list[pd.Index] | None" = None,
        movie: torch.Tensor | None = None,
        deformation_field: CubicCatmullRomGrid3d | None = None,
        particle_shifts: torch.Tensor | None = None,
        pre_exposure: float = 0.0,
        fluence_per_frame: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract particle images and compute FFT + preprocessing filters.

        Either pass ``particle_images`` (already extracted) or let the method load/
        extract them from micrographs or a movie.

        Parameters
        ----------
        template : torch.Tensor
            3-D template volume used for shape and device information.
        preprocessing_filters : PreprocessingFilters
            Filter configuration applied to the particle images.
        apply_global_filtering : bool
            When True, filters are derived from the full micrograph and applied
            before particle extraction.  When False, each particle image is
            filtered independently after extraction.
        particle_images : torch.Tensor | None
            Pre-extracted particle image stack.  When provided, micrograph
            loading and extraction are skipped.
        micrograph_images : torch.Tensor | None
            Pre-loaded micrograph images.  When None, micrographs are loaded
            from the ``micrograph_path`` column.
        micrograph_indices : list[pd.Index] | None
            Row indices matching ``micrograph_images`` to particles; required
            when ``micrograph_images`` is provided.
        movie : torch.Tensor | None
            Movie tensor for beam-induced motion correction.
        deformation_field : CubicCatmullRomGrid3d | None
            Continuous deformation field for motion correction.
        particle_shifts : torch.Tensor | None
            Per-particle per-frame shifts, shape ``(T, N, 2)``.
        pre_exposure : float
            Pre-exposure fluence in e-/A^2.
        fluence_per_frame : float
            Fluence per frame in e-/A^2.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(particle_images_dft, template_dft, projective_filters)``
        """
        device = template.device
        box_h, box_w = self.extracted_box_size
        filter_output_shape = (template.shape[-2], template.shape[-1] // 2 + 1)
        projective_filters = None

        if particle_images is not None:
            self.image_stack = particle_images
        else:
            micrograph_images, micrograph_indices = self._load_micrograph_images(
                micrograph_images, micrograph_indices, device
            )
            if apply_global_filtering:
                micrograph_images, projective_filters = (
                    self._apply_global_micrograph_filtering(
                        micrograph_images,
                        template,
                        preprocessing_filters,
                        micrograph_indices,
                    )
                )
            particle_images = self._extract_particle_images(
                micrograph_images,
                micrograph_indices,
                movie,
                deformation_field,
                particle_shifts,
                pre_exposure,
                fluence_per_frame,
            ).to(device)

        if not apply_global_filtering:
            particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))  # pylint: disable=not-callable
            particle_images_dft[..., 0, 0] = 0.0 + 0.0j
            with torch.no_grad():
                projective_filters = self.construct_image_filters(
                    preprocessing_filters,
                    output_shape=filter_output_shape,
                    images_dft=particle_images_dft.detach(),
                ).to(device)
            particle_images_dft = apply_image_filtering(
                self,
                preprocessing_filters,
                particle_images_dft,
                full_image_shape=(box_h, box_w),
                extracted_box_shape=(box_h, box_w),
            )
        else:
            particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))  # pylint: disable=not-callable

        template_dft = volume_to_rfft_fourier_slice(template)

        return particle_images_dft, template_dft, projective_filters

    # ------------------------------------------------------------------
    # CTF / orientation accessors
    # ------------------------------------------------------------------

    def get_relative_defocus(self, prefer_refined_defocus: bool = True) -> torch.Tensor:
        """Get the relative defocus values for each particle.

        Parameters
        ----------
        prefer_refined_defocus : bool, optional
            Whether to use the refined defocus values, by default True.

        Returns
        -------
        torch.Tensor
        """
        if "refined_relative_defocus" not in self._df.columns:
            prefer_refined_defocus = False

        defocus_col = (
            "refined_relative_defocus" if prefer_refined_defocus else "relative_defocus"
        )
        defocus_values = self._df[defocus_col].to_numpy().copy()

        if prefer_refined_defocus and _any_nan_or_inf(defocus_values):
            warnings.warn(
                "Refined defocus values contain NaN or inf values, using original "
                "defocus values...",
                stacklevel=2,
            )
            defocus_values = self._df["relative_defocus"].to_numpy().copy()

        return torch.tensor(defocus_values)

    def get_absolute_defocus(
        self, prefer_refined_defocus: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the absolute defocus (u, v) values for each particle.

        Parameters
        ----------
        prefer_refined_defocus : bool, optional
            Whether to use refined defocus, by default True.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(defocus_u, defocus_v)`` tensors in Angstroms.
        """
        particle_defocus = self.get_relative_defocus(prefer_refined_defocus)

        defocus_u = torch.tensor(self["defocus_u"].to_numpy().copy())
        defocus_v = torch.tensor(self["defocus_v"].to_numpy().copy())
        defocus_u = defocus_u + particle_defocus
        defocus_v = defocus_v + particle_defocus

        return defocus_u, defocus_v

    def get_pixel_size(self, prefer_refined_pixel_size: bool = True) -> torch.Tensor:
        """Get the pixel size for each particle.

        Parameters
        ----------
        prefer_refined_pixel_size : bool, optional
            Whether to use the refined pixel size, by default True.

        Returns
        -------
        torch.Tensor
        """
        if "refined_pixel_size" not in self._df.columns:
            prefer_refined_pixel_size = False

        pixel_size_col = (
            "refined_pixel_size" if prefer_refined_pixel_size else "pixel_size"
        )
        pixel_size_values = self._df[pixel_size_col].to_numpy().copy()

        if prefer_refined_pixel_size and _any_nan_or_inf(pixel_size_values):
            warnings.warn(
                "Refined pixel size values contain NaN or inf values, using original "
                "pixel size values...",
                stacklevel=2,
            )
            pixel_size_values = self._df["pixel_size"].to_numpy().copy()

        return torch.tensor(pixel_size_values)

    def get_euler_angles(self, prefer_refined_angles: bool = True) -> torch.Tensor:
        """Return the Euler angles (phi, theta, psi) of all particles as a tensor.

        Parameters
        ----------
        prefer_refined_angles : bool, optional
            When true, refined angles are used if present, by default True.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3)`` — columns correspond to (phi, theta, psi) in ZYZ.
        """
        phi_col = "phi"
        theta_col = "theta"
        psi_col = "psi"

        if prefer_refined_angles:
            if not all(
                x in self._df.columns
                for x in ["refined_phi", "refined_theta", "refined_psi"]
            ):
                warnings.warn(
                    "Refined angles not found in DataFrame, using original angles...",
                    stacklevel=2,
                )
            else:
                phi_col = "refined_phi"
                theta_col = "refined_theta"
                psi_col = "refined_psi"

        phi = torch.tensor(self[phi_col].to_numpy().copy())
        theta = torch.tensor(self[theta_col].to_numpy().copy())
        psi = torch.tensor(self[psi_col].to_numpy().copy())

        return torch.stack((phi, theta, psi), dim=-1)

    # ------------------------------------------------------------------
    # Image-stack construction
    # ------------------------------------------------------------------

    def load_images_grouped_by_column(
        self, column_name: str
    ) -> tuple[torch.Tensor, list[pd.Index]]:
        """Load images grouped by a column and return images as a tensor with indexes.

        Parameters
        ----------
        column_name : str
            The column name to group by (e.g., "micrograph_path" or "mip_path").

        Returns
        -------
        tuple[torch.Tensor, list[pd.Index]]
            A tuple containing:
            - A tensor of loaded images with shape (N, H, W) where N is the number of
              unique images and (H, W) is the image size
            - A list of pandas Index objects containing the row indexes for particles
              from each corresponding image
        """
        if column_name not in self._df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        images_list = []
        indices = []
        image_index_groups = self._df.groupby(column_name).groups
        for img_path, indexes in image_index_groups.items():
            img = _load_image_2d(img_path, column_name)
            images_list.append(img)
            indices.append(indexes)

        if not images_list:  # Empty case
            return torch.empty((0, 0, 0)), []

        images_tensor = torch.stack(images_list, dim=0)

        return images_tensor, indices

    def construct_image_stack(
        self,
        images: torch.Tensor,
        indices: list[pd.Index],
        extraction_size: tuple[int, int],
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Construct stack of images from the DataFrame (updates image_stack in-place).

        This method preferentially selects refined position columns by default
        (refined_pos_x, refined_pos_y) if they are present in the DataFrame, falling
        back to unrefined positions (pos_x, pos_y) otherwise.

        This method uses columns pos_x and pos_y (or refined_pos_x and refined_pos_y if
        available) to extract the boxes from the images. When using top-left reference
        position, the boxes are extracted as follows, where the dots represent the
        actual particle in the image

        Example:
            :                +----------------------------------+
            :                |                                  |
            :                |                                  |
            :                |     (x, y) *=== box_w ===+       |
            :                |            |             |       |
            :                |            |     ....  box_h     |
            :           img_height        |    ......   |       |
            :                |            |     ....    |       |
            :                |            |             |       |
            :                |            +=============+       |
            :                |                                  |
            :                +------------ img_width -----------+

        When center reference is used, then the position columns in the DataFrame are
        interpreted as the center of the particle, and the boxes are extracted around
        this x and y position as follows:

        Example:
            :                +----------------------------------+
            :                |                                  |
            :                |                                  |
            :                |            +=== box_w ===+       |
            :                |            |             |       |
            :                |            |     ....    |       |
            :           img_height        |(x, y).*.. box_h     |
            :                |            |     ....    |       |
            :                |            |             |       |
            :                |            +=============+       |
            :                |                                  |
            :                +------------ img_width -----------+

        Parameters
        ----------
        images : torch.Tensor
            A tensor of loaded images with shape (N, H, W).
        indices : list[pd.Index]
            Row indexes for particles from each corresponding image.
        extraction_size : tuple[int, int]
            Size of the extracted boxes in pixels (height, width).
        pos_reference : Literal["center", "top-left"], optional
            Reference point for the positions, by default "top-left".
        handle_bounds : Literal["pad", "error"], optional
            How to handle out-of-bounds regions, by default "pad".
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            Padding mode when ``handle_bounds="pad"``, by default "constant".
        padding_value : float, optional
            Constant padding value, by default 0.0.

        Returns
        -------
        torch.Tensor
            Stack of extracted images ``(N, extraction_h, extraction_w)``.
        """
        y_col, x_col = self.get_position_reference_columns()

        h, w = self.original_template_size
        box_h, box_w = self.extracted_box_size
        device = images.device
        image_stack = torch.zeros((self.num_particles, *extraction_size), device=device)

        if images.shape[0] != len(indices):
            raise ValueError(
                f"Number of images ({images.shape[0]}) does not match the number of "
                f"indices ({len(indices)})."
            )

        for i, indexes in enumerate(indices):
            img = images[i]
            pos_y = self._df.loc[indexes, y_col].to_numpy().copy()
            pos_x = self._df.loc[indexes, x_col].to_numpy().copy()

            if pos_reference == "center":
                pos_y = pos_y - h // 2
                pos_x = pos_x - w // 2

            pos_y = pos_y - (box_h - h) // 2
            pos_x = pos_x - (box_w - w) // 2

            pos_y = torch.tensor(pos_y, device=img.device)
            pos_x = torch.tensor(pos_x, device=img.device)

            cropped_images = get_cropped_image_regions(
                img,
                pos_y,
                pos_x,
                extraction_size,
                pos_reference="top-left",
                handle_bounds=handle_bounds,
                padding_mode=padding_mode,
                padding_value=padding_value,
            )
            image_stack[indexes] = cropped_images

        self.image_stack = image_stack

        return image_stack

    def construct_image_filters(
        self,
        preprocess_filters: PreprocessingFilters,
        output_shape: tuple[int, int],
        images_dft: torch.Tensor,
    ) -> torch.Tensor:
        """Get stack of Fourier filters from filter config and reference images.

        Note that here the filters are assumed to be applied globally (i.e. no local
        whitening, etc. is being done). Whitening filters are calculated with reference
        to each image (micrograph or particle).

        Parameters
        ----------
        preprocess_filters : PreprocessingFilters
            Configuration object of filters to apply.
        output_shape : tuple[int, int]
            What shape along the last two dimensions the filters should be.
        images_dft : torch.Tensor
            A tensor of images with shape (N, H, W) where N is the number of images
            (micrographs or particles) and (H, W) is the image size. in Fourier space.

        Returns
        -------
        torch.Tensor
            The stack of filters with shape (N, h, w) where N is the number of images
            and (h, w) is the output shape.
        """
        device = images_dft.device
        num_images = images_dft.shape[0]
        filter_stack = torch.zeros((num_images, *output_shape), device=device)

        for i in range(num_images):
            img_dft = images_dft[i]
            cumulative_filter = preprocess_filters.get_combined_filter(
                ref_img_rfft=img_dft,
                output_shape=output_shape,
            )

            filter_stack[i] = cumulative_filter

        return filter_stack

    def construct_projective_filters(
        self,
        preprocess_filters: PreprocessingFilters,
        output_shape: tuple[int, int],
        images_dft: torch.Tensor,
        indices: list[pd.Index],
    ) -> torch.Tensor:
        """Get stack of Fourier filters from filter config and reference micrographs.

        Parameters
        ----------
        preprocess_filters : PreprocessingFilters
            Configuration object of filters to apply.
        output_shape : tuple[int, int]
            What shape along the last two dimensions the filters should be.
        images_dft : torch.Tensor
            A tensor of micrograph images in Fourier space with shape (N, H, W).
        indices : list[pd.Index]
            Row indexes for particles from each corresponding micrograph.

        Returns
        -------
        torch.Tensor
            Filter stack of shape ``(M, h, w)`` where M is the number of particles.
        """
        device = images_dft.device
        filter_stack = torch.zeros((self.num_particles, *output_shape), device=device)
        if images_dft.shape[0] != len(indices):
            raise ValueError(
                f"Number of images ({images_dft.shape[0]}) does not match "
                f"the number of indices ({len(indices)})."
            )

        for i, indexes in enumerate(indices):
            img_dft = images_dft[i]
            cumulative_filter = preprocess_filters.get_combined_filter(
                ref_img_rfft=img_dft,
                output_shape=output_shape,
            )

            filter_stack[indexes] = cumulative_filter

        return filter_stack

    @staticmethod
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def _process_single_frame_with_shifts_checkpoint(
        movie_frame: torch.Tensor,
        shifts: torch.Tensor,  # (N, 2) -> (dy, dx)
        pos_y: torch.Tensor,
        pos_x: torch.Tensor,
        extracted_box_size: tuple[int, int],
        handle_bounds: Literal["pad", "error"],
        padding_mode: Literal["constant", "reflect", "replicate"],
        padding_value: float,
    ) -> torch.Tensor:
        """Process a single frame using precomputed particle shifts.

        Safe for gradient checkpointing; contains no deformation-field evaluation.

        Parameters
        ----------
        movie_frame : torch.Tensor
            Single movie frame (H, W).
        shifts : torch.Tensor
            Per-particle shifts with shape (N, 2) as (dy, dx).
        pos_y, pos_x : torch.Tensor
            Top-left extraction positions.
        extracted_box_size : tuple[int, int]
            ``(box_h, box_w)``.
        handle_bounds, padding_mode, padding_value
            Passed through to cropping.

        Returns
        -------
        torch.Tensor
            Shifted FFTs with shape ``(N, box_h, box_w//2 + 1)``.
        """
        box_h, box_w = extracted_box_size

        cropped_images = get_cropped_image_regions(
            movie_frame,
            pos_y,
            pos_x,
            extracted_box_size,
            pos_reference="top-left",
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

        cropped_images_dft = torch.fft.rfftn(  # pylint: disable=not-callable
            cropped_images, dim=(-2, -1)
        )

        shifted_fft = fourier_shift_dft_2d(
            dft=cropped_images_dft,
            image_shape=(box_h, box_w),
            shifts=shifts,
            rfft=True,
            fftshifted=False,
        )

        return shifted_fft

    def compute_frame_particle_shifts_from_deformation(
        self,
        movie_frame: torch.Tensor,
        deformation_field: CubicCatmullRomGrid3d,
        normalized_t_value: torch.Tensor,
        pixel_grid: torch.Tensor,
        pixel_spacing: float,
        pos_y_center: torch.Tensor,
        pos_x_center: torch.Tensor,
        gh: int,
        gw: int,
    ) -> torch.Tensor:
        """Compute per-particle shifts for a single frame from a deformation field.

        Parameters
        ----------
        movie_frame : torch.Tensor
            Single movie frame (H, W).
        deformation_field : CubicCatmullRomGrid3d
            The deformation field grid.
        normalized_t_value : torch.Tensor
            Normalized time value for the frame.
        pixel_grid : torch.Tensor
            The pixel grid tensor.
        pixel_spacing : float
            The pixel spacing.
        pos_y_center : torch.Tensor
            Center y positions.
        pos_x_center : torch.Tensor
            Center x positions.
        gh : int
            Height of the deformation field grid.
        gw : int
            Width of the deformation field grid.

        Returns
        -------
        torch.Tensor
            Shifts with shape ``(N, 2)`` as (dy, dx).
        """
        frame_deformation_field = deformation_field.evaluate_at_t(
            t=normalized_t_value.item(),
            grid_shape=(10 * gh, 10 * gw),
        )

        pixel_shifts = get_pixel_shifts(
            frame=movie_frame,
            pixel_spacing=pixel_spacing,
            frame_deformation_grid=frame_deformation_field,
            pixel_grid=pixel_grid,
        )

        y_shifts = -pixel_shifts[pos_y_center, pos_x_center, 0]
        x_shifts = -pixel_shifts[pos_y_center, pos_x_center, 1]

        return torch.stack((y_shifts, x_shifts), dim=-1)

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    def construct_image_stack_from_movie(
        self,
        movie: torch.Tensor,
        deformation_field: CubicCatmullRomGrid3d | None = None,
        particle_shifts: torch.Tensor | None = None,
        pos_reference: Literal["center", "top-left"] = "top-left",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
        pre_exposure: float = 0.0,
        fluence_per_frame: float = 0.0,
        use_gradient_checkpointing: bool = True,
        particle_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Construct a stack of images from a movie file.

        Parameters
        ----------
        movie : torch.Tensor
            The movie tensor.
        deformation_field : CubicCatmullRomGrid3d | None, optional
            The deformation field grid.
        particle_shifts : torch.Tensor | None, optional
            Per-particle shifts, shape ``(T, N, 2)``.  Exactly one of
            ``deformation_field`` and ``particle_shifts`` must be provided.
        pos_reference : Literal["center", "top-left"], optional
            Position reference for extraction, by default "top-left".
        handle_bounds : Literal["pad", "error"], optional
            How to handle out-of-bounds regions, by default "pad".
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            Padding mode, by default "constant".
        padding_value : float, optional
            Constant padding value, by default 0.0.
        pre_exposure : float, optional
            Pre-exposure in electrons per pixel, by default 0.0.
        fluence_per_frame : float, optional
            Dose per frame in electrons per pixel, by default 0.0.
        use_gradient_checkpointing : bool, optional
            Trade compute for memory during frame processing, by default True.
        particle_indices : list[int] | None, optional
            Subset of particles to process.  If None, all particles are used.

        Returns
        -------
        torch.Tensor
            Image stack of shape ``(N, box_h, box_w)``.
        """
        if (deformation_field is None) == (particle_shifts is None):
            raise ValueError(
                "One of `deformation_field` or `particle_shifts` must be provided."
            )
        pixel_sizes = self.get_pixel_size()
        y_col, x_col = self.get_position_reference_columns()
        h, w = self.original_template_size
        box_h, box_w = self.extracted_box_size
        t, img_h, img_w = movie.shape
        if deformation_field is not None:
            _, _, gh, gw = deformation_field.data.shape
        else:
            gh = gw = 0
        normalized_t = torch.linspace(0, 1, steps=t, device=movie.device)
        pixel_grid = coordinate_grid(
            image_shape=(img_h, img_w),
            device=movie.device,
        )
        if particle_indices is not None:
            paticle_indexes = [self._df.index[i] for i in particle_indices]
            num_particles_to_process = len(particle_indices)
        else:
            paticle_indexes = self._df.index.tolist()
            num_particles_to_process = self.num_particles

        pos_y = self._df.loc[paticle_indexes, y_col].to_numpy()
        pos_x = self._df.loc[paticle_indexes, x_col].to_numpy()
        if pos_reference == "center":
            pos_y = pos_y - h // 2
            pos_x = pos_x - w // 2

        pos_y_center = pos_y + h // 2
        pos_x_center = pos_x + w // 2
        pos_y -= (box_h - h) // 2
        pos_x -= (box_w - w) // 2
        pos_y = torch.tensor(pos_y)
        pos_x = torch.tensor(pos_x)
        pos_y_center = torch.tensor(pos_y_center)
        pos_x_center = torch.tensor(pos_x_center)

        aligned_particle_movies_rfft = torch.zeros(
            (num_particles_to_process, t, box_h, box_w // 2 + 1),
            dtype=torch.complex64,
            device=movie.device,
        )
        movie = movie - torch.mean(movie, dim=(-2, -1), keepdim=True)

        for frame_index, movie_frame in enumerate(movie):
            if particle_shifts is not None:
                frame_shifts = particle_shifts[frame_index]  # (N, 2)
            else:
                frame_shifts = self.compute_frame_particle_shifts_from_deformation(
                    movie_frame=movie_frame,
                    deformation_field=deformation_field,
                    normalized_t_value=normalized_t[frame_index],
                    pixel_grid=pixel_grid,
                    pixel_spacing=pixel_sizes[0].item(),
                    pos_y_center=pos_y_center,
                    pos_x_center=pos_x_center,
                    gh=gh,
                    gw=gw,
                )

            if use_gradient_checkpointing:
                shifted_fft = checkpoint(
                    self._process_single_frame_with_shifts_checkpoint,
                    movie_frame,
                    frame_shifts,
                    pos_y,
                    pos_x,
                    self.extracted_box_size,
                    handle_bounds,
                    padding_mode,
                    padding_value,
                    use_reentrant=False,
                )
            else:
                shifted_fft = self._process_single_frame_with_shifts_checkpoint(
                    movie_frame=movie_frame,
                    shifts=frame_shifts,
                    pos_y=pos_y,
                    pos_x=pos_x,
                    extracted_box_size=self.extracted_box_size,
                    handle_bounds=handle_bounds,
                    padding_mode=padding_mode,
                    padding_value=padding_value,
                )

            aligned_particle_movies_rfft[:, frame_index] = shifted_fft

            if frame_index % 10 == 0 and frame_index > 0:
                torch.cuda.empty_cache()

        aligned_particle_images = torch.zeros(
            (num_particles_to_process, box_h, box_w),
            device=movie.device,
        )
        for particle_index in range(num_particles_to_process):
            particle_dft = aligned_particle_movies_rfft[particle_index]

            df_idx = paticle_indexes[particle_index]
            df_loc = self._df.index.get_loc(df_idx)

            dw_sum = dose_weight_movie_to_micrograph(
                movie_fft=particle_dft,
                pixel_size=pixel_sizes[df_loc],
                pre_exposure=pre_exposure,
                fluence_per_frame=fluence_per_frame,
                voltage=self._df["voltage"].to_numpy()[df_loc],
            )
            aligned_particle_images[particle_index] = dw_sum

        if particle_indices is None:
            self.image_stack = aligned_particle_images
        return aligned_particle_images
