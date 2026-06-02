"""Pydantic model for running the refine template program."""

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from leopard_em.backend.core_differentiable_refine import core_differentiable_refine
from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.config import (
    ComputationalConfigRefine,
    DefocusSearchConfig,
    MovieConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.pydantic_models.formats import REFINED_DF_COLUMN_ORDER
from leopard_em.utils.backend_setup import setup_particle_backend_kwargs
from leopard_em.utils.data_io import (
    load_mrc_volume,
    load_template_tensor,
    read_particle_shifts_from_csv,
)


class RefineTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running the refine template program.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    defocus_refinement_config : DefocusSearchConfig
        Configuration for defocus refinement.
    pixel_size_refinement_config : PixelSizeSearchConfig
        Configuration for pixel size refinement.
    orientation_refinement_config : RefineOrientationConfig
        Configuration for orientation refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfigRefine
        What computational resources to allocate for the program.
    apply_global_filtering : bool
        If True, apply filtering to the full micrograph before particle extraction.
        If False, filter are calculated and applied to the cropped particle images.
        Default is True.
    template_volume : ExcludedTensor
        The template volume tensor (excluded from serialization).
    movie_config : MovieConfig
        Configuration for the movie.

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the refine template manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend refine_template core function.
    run_refine_template(self, correlation_batch_size: int = 32) -> None
        Run the refine template program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    pixel_size_refinement_config: PixelSizeSearchConfig
    orientation_refinement_config: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfigRefine
    movie_config: MovieConfig
    apply_global_filtering: bool = True

    # Excluded tensors
    template_volume: ExcludedTensor

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    # TODO: Make this a dict return for clarity
    def _prepare_search_tensors(
        self,
        prefer_refined_angles: bool = True,
        template_tensor: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the template and assemble the five search tensors.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to prefer refined Euler angles. Defaults to True.
        template_tensor : torch.Tensor | None
            Pre-loaded template.  If None, loaded from ``template_volume_path``.
        device : torch.device | None
            When provided, moves template and euler_angles to this device.

        Returns
        -------
        tuple of (template, euler_angles, euler_angle_offsets,
                  defocus_offsets, pixel_size_offsets)
        """
        if template_tensor is None:
            template = load_template_tensor(
                template_volume=self.template_volume,
                template_volume_path=self.template_volume_path,
            )
        else:
            template = template_tensor

        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)
        euler_angle_offsets = self.orientation_refinement_config.euler_angles_offsets
        defocus_offsets = self.defocus_refinement_config.defocus_values
        pixel_size_offsets = self.pixel_size_refinement_config.pixel_size_values

        if device is not None:
            template = template.to(device)
            euler_angles = euler_angles.to(device)

        return (
            template,
            euler_angles,
            euler_angle_offsets,
            defocus_offsets,
            pixel_size_offsets,
        )

    def make_backend_core_function_kwargs(
        self, prefer_refined_angles: bool = True
    ) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use the refined angles from the particle stack. Defaults to
            True.
        """
        (
            template,
            euler_angles,
            euler_angle_offsets,
            defocus_offsets,
            pixel_size_offsets,
        ) = self._prepare_search_tensors(prefer_refined_angles)

        # Load movie, deformation field, and particle shifts
        movie = self.movie_config.movie
        particle_shifts = None
        deformation_field = None

        if self.movie_config.enabled:
            # Particle shifts take precedence over deformation field
            if self.movie_config.particle_shifts_path:
                if movie is None:
                    raise ValueError(
                        "Movie must be loaded when using particle shifts. "
                        "Ensure movie_path is set and movie is enabled."
                    )
                t, _, _ = movie.shape
                num_particles = self.particle_stack.num_particles
                particle_shifts = read_particle_shifts_from_csv(
                    csv_path=self.movie_config.particle_shifts_path,
                    num_frames=t,
                    num_particles=num_particles,
                )
            else:
                # Use deformation field if particle shifts not provided
                deformation_field_tensor = self.movie_config.deformation_field
                if deformation_field_tensor is not None:
                    deformation_field = CubicCatmullRomGrid3d.from_grid_data(
                        deformation_field_tensor
                    )

        # Use the common utility function to set up the backend kwargs
        return setup_particle_backend_kwargs(
            particle_stack=self.particle_stack,
            template=template,
            preprocessing_filters=self.preprocessing_filters,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            apply_global_filtering=self.apply_global_filtering,
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
            pre_exposure=self.movie_config.pre_exposure,
            fluence_per_frame=self.movie_config.fluence_per_frame,
            device_list=self.computational_config.gpu_devices,
        )

    def make_differentiable_backend_kwargs(
        self,
        image_stack: torch.Tensor,
        mean_stack: torch.Tensor,
        std_stack: torch.Tensor,
        particle_indices: list[pd.Index],
        template_tensor: torch.Tensor | None = None,
        prefer_refined_angles: bool = True,
        images_are_particles: bool = False,
    ) -> dict[str, Any]:
        """Create the kwargs for the backend differentiable refine core function.

        Parameters
        ----------
        image_stack : torch.Tensor
            Pre-loaded image stack tensor.
        mean_stack : torch.Tensor
            Pre-loaded mean stack tensor.
        std_stack : torch.Tensor
            Pre-loaded std stack tensor.
        particle_indices : list[pd.Index]
            The particle indices to process.
        template_tensor : torch.Tensor | None
            Pre-loaded template tensor. If None, will be loaded from the template volume
            path. Defaults to None.
        prefer_refined_angles : bool
            Whether to use the refined angles from the particle stack. Defaults to
            True.
        images_are_particles : bool
            Whether the images are particles or not. Defaults to False.
        """
        device = image_stack.device
        (
            template,
            euler_angles,
            euler_angle_offsets,
            defocus_offsets,
            pixel_size_offsets,
        ) = self._prepare_search_tensors(
            prefer_refined_angles,
            template_tensor=template_tensor,
            device=device,
        )

        return setup_particle_backend_kwargs(
            particle_stack=self.particle_stack,
            template=template,
            preprocessing_filters=self.preprocessing_filters,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            apply_global_filtering=self.apply_global_filtering,
            device_list=[device],
            image_stack=image_stack,
            mean_stack=mean_stack,
            std_stack=std_stack,
            particle_indices=particle_indices,
            images_are_particles=images_are_particles,
        )

    def run_refine_template(
        self, output_dataframe_path: str, correlation_batch_size: int = 32
    ) -> None:
        """Run the refine template program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        correlation_batch_size : int
            Number of cross-correlations to process in one batch, defaults to 32.
        """
        backend_kwargs = self.make_backend_core_function_kwargs()

        result = self.get_refine_result(backend_kwargs, correlation_batch_size)

        self.refine_result_to_dataframe(
            output_dataframe_path=output_dataframe_path, result=result
        )

    def run_differentiable_refine(
        self,
        output_dataframe_path: str,
        image_stack: torch.Tensor,
        mean_stack: torch.Tensor,
        std_stack: torch.Tensor,
        particle_indices: list[pd.Index],
        template_tensor: torch.Tensor | None = None,
        correlation_batch_size: int = 32,
        images_are_particles: bool = False,
    ) -> None:
        """Run the differentiable refine template program and saves DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        image_stack : torch.Tensor
            Pre-loaded image stack tensor.
        mean_stack : torch.Tensor
            Pre-loaded mean stack tensor.
        std_stack : torch.Tensor
            Pre-loaded std stack tensor.
        particle_indices : list[pd.Index]
            The particle indices to process.
        template_tensor : torch.Tensor | None
            Pre-loaded template tensor. If None, will be loaded from the template volume
            path. Defaults to None.
        correlation_batch_size : int
            Number of cross-correlations to process in one batch, defaults to 32.
        images_are_particles : bool
            Whether the images are particles or not. Defaults to False.

        """
        backend_kwargs = self.make_differentiable_backend_kwargs(
            image_stack=image_stack,
            mean_stack=mean_stack,
            std_stack=std_stack,
            template_tensor=template_tensor,
            particle_indices=particle_indices,
            images_are_particles=images_are_particles,
        )

        result = self.get_refine_result(
            backend_kwargs, correlation_batch_size, use_differentiable=True
        )

        self.refine_result_to_dataframe(
            output_dataframe_path=output_dataframe_path, result=result
        )

    def get_refine_result(
        self,
        backend_kwargs: dict,
        correlation_batch_size: int = 32,
        use_differentiable: bool = False,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Get refine template result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        correlation_batch_size : int
            Number of orientations to process at once. Defaults to 32.
        use_differentiable : bool
            If True, use differentiable refine. If False, use regular refine.
            Defaults to False.

        Returns
        -------
        dict[str, np.ndarray | torch.Tensor]
            The result of the refine template program. Returns torch.Tensor
            for differentiable refine, np.ndarray for regular refine.
        """
        # pylint: disable=duplicate-code
        if use_differentiable:
            result = core_differentiable_refine(
                batch_size=correlation_batch_size,
                num_cuda_streams=self.computational_config.num_cpus,
                **backend_kwargs,
            )
            # Keep as torch.Tensor for differentiable refine
            return result
        result = core_refine_template(
            batch_size=correlation_batch_size,
            num_cuda_streams=self.computational_config.num_cpus,
            **backend_kwargs,
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}
        return result

    def refine_result_to_dataframe(
        self,
        output_dataframe_path: str,
        result: dict[str, np.ndarray | torch.Tensor],
        prefer_refined_angles: bool = True,
    ) -> None:
        """Convert refine template result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        result : dict[str, np.ndarray | torch.Tensor]
            The result of the refine template program. Can contain either
            np.ndarray (regular refine) or torch.Tensor (differentiable refine).
        prefer_refined_angles : bool
            Whether to prefer refined position columns when they exist.
            Defaults to True.
        """
        result_np: dict[str, np.ndarray] = {
            k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in result.items()
        }
        df_refined = self.particle_stack.build_refined_dataframe(
            result_np,
            column_order=REFINED_DF_COLUMN_ORDER,
            prefer_refined_positions=prefer_refined_angles,
        )
        df_refined.to_csv(output_dataframe_path)
