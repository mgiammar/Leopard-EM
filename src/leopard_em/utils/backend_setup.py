"""Backend setup and orchestration utility functions."""

from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

if TYPE_CHECKING:
    from leopard_em.pydantic_models.config.correlation_filters import (
        PreprocessingFilters,
    )
    from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack


def _assemble_backend_kwargs(
    particle_stack: "ParticleStack",
    template: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    device_list: list,
    corr_mean_stack: torch.Tensor,
    corr_variance_stack: torch.Tensor,
    particle_images_dft: torch.Tensor,
    template_dft: torch.Tensor,
    projective_filters: torch.Tensor | None,
) -> dict[str, Any]:
    """Assemble the final backend kwargs dict from pre-computed components.

    Variance → std conversion happens here — the single canonical site.

    Parameters
    ----------
    particle_stack : ParticleStack
        Particle stack used to extract defocus and CTF parameters.
    template : torch.Tensor
        3D template volume (used for shape and device).
    euler_angles : torch.Tensor
        Per-particle Euler angles.
    euler_angle_offsets : torch.Tensor
        Euler angle offsets to search over.
    defocus_offsets : torch.Tensor
        Defocus offsets to search over.
    pixel_size_offsets : torch.Tensor
        Pixel size offsets to search over.
    device_list : list
        Computational devices to use.
    corr_mean_stack : torch.Tensor
        Per-particle correlation mean.
    corr_variance_stack : torch.Tensor
        Per-particle correlation variance (converted to std internally).
    particle_images_dft : torch.Tensor
        Fourier-transformed particle images.
    template_dft : torch.Tensor
        Fourier-transformed template slices.
    projective_filters : torch.Tensor | None
        Per-particle projective filters.

    Returns
    -------
    dict[str, Any]
        Kwargs dictionary for ``core_refine_template`` or
        ``core_differentiable_refine``.
    """
    device = template.device

    defocus_u, defocus_v = particle_stack.get_absolute_defocus()
    defocus_u = defocus_u.to(device)
    defocus_v = defocus_v.to(device)
    defocus_angle = torch.tensor(particle_stack["astigmatism_angle"], device=device)

    ctf_kwargs = particle_stack.get_ctf_kwargs((template.shape[-2], template.shape[-1]))
    mag_matrix_tensor = ctf_kwargs["mag_matrix"]

    corr_std_stack = corr_variance_stack**0.5

    return {
        "particle_stack_dft": particle_images_dft,
        "template_dft": template_dft,
        "euler_angles": euler_angles,
        "euler_angle_offsets": euler_angle_offsets,
        "defocus_u": defocus_u,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "defocus_offsets": defocus_offsets,
        "pixel_size_offsets": pixel_size_offsets,
        "corr_mean": corr_mean_stack,
        "corr_std": corr_std_stack,
        "ctf_kwargs": ctf_kwargs,
        "projective_filters": projective_filters,
        "device": device_list,
        "mag_matrix": mag_matrix_tensor,
    }


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def setup_backend_kwargs_from_micrographs(
    particle_stack: "ParticleStack",
    template: torch.Tensor,
    preprocessing_filters: "PreprocessingFilters",
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    apply_global_filtering: bool,
    device_list: list,
    movie: torch.Tensor | None = None,
    deformation_field: CubicCatmullRomGrid3d | None = None,
    particle_shifts: torch.Tensor | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    image_stack: torch.Tensor | None = None,
    mean_stack: torch.Tensor | None = None,
    std_stack: torch.Tensor | None = None,
    particle_indices: list[pd.Index] | None = None,
) -> dict[str, Any]:
    """Build backend kwargs when particle images are extracted from micrographs.

    Parameters
    ----------
    particle_stack : ParticleStack
        Particle stack containing images to process.
    template : torch.Tensor
        3D template volume.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    euler_angles : torch.Tensor
        Per-particle Euler angles.
    euler_angle_offsets : torch.Tensor
        Euler angle offsets to search over.
    defocus_offsets : torch.Tensor
        Defocus offsets to search over.
    pixel_size_offsets : torch.Tensor
        Pixel size offsets to search over.
    apply_global_filtering : bool
        If True, apply filtering to the full micrograph before extraction.
    device_list : list
        Computational devices to use.
    movie : torch.Tensor | None
        Movie tensor for motion correction.
    deformation_field : CubicCatmullRomGrid3d | None
        Continuous deformation field for motion correction.
    particle_shifts : torch.Tensor | None
        Per-particle per-frame shifts ``(T, N, 2)``; takes precedence over
        ``deformation_field``.
    pre_exposure : float
        Pre-exposure fluence in e⁻/Å².
    fluence_per_frame : float
        Fluence per frame in e⁻/Å².
    image_stack : torch.Tensor | None
        Pre-loaded micrograph images; ``particle_indices`` required when given.
    mean_stack : torch.Tensor | None
        Pre-loaded correlation mean maps.
    std_stack : torch.Tensor | None
        Pre-loaded correlation variance maps.
    particle_indices : list[pd.Index] | None
        Row indices matching pre-loaded stacks to particles.

    Returns
    -------
    dict[str, Any]
        Kwargs dictionary for the backend refinement functions.
    """
    device = template.device
    h, w = particle_stack.original_template_size
    box_h, box_w = particle_stack.extracted_box_size
    extracted_box_size = (box_h - h + 1, box_w - w + 1)

    corr_mean, corr_variance = particle_stack.get_correlation_stacks(
        extracted_box_size,
        device,
        mean_stack=mean_stack,
        std_stack=std_stack,
        particle_indices=particle_indices,
    )

    imgs_dft, tmpl_dft, proj_filters = particle_stack.prepare_images_and_filters(
        template,
        preprocessing_filters,
        apply_global_filtering=apply_global_filtering,
        micrograph_images=image_stack,
        micrograph_indices=particle_indices,
        movie=movie,
        deformation_field=deformation_field,
        particle_shifts=particle_shifts,
        pre_exposure=pre_exposure,
        fluence_per_frame=fluence_per_frame,
    )

    return _assemble_backend_kwargs(
        particle_stack,
        template,
        euler_angles,
        euler_angle_offsets,
        defocus_offsets,
        pixel_size_offsets,
        device_list,
        corr_mean,
        corr_variance,
        imgs_dft,
        tmpl_dft,
        proj_filters,
    )


# pylint: disable=too-many-arguments
def setup_backend_kwargs_from_particles(
    particle_stack: "ParticleStack",
    template: torch.Tensor,
    preprocessing_filters: "PreprocessingFilters",
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    apply_global_filtering: bool,
    device_list: list,
    image_stack: torch.Tensor,
    mean_stack: torch.Tensor,
    std_stack: torch.Tensor,
    particle_indices: list[pd.Index],
) -> dict[str, Any]:
    """Build backend kwargs when the image stack is already extracted particles.

    Parameters
    ----------
    particle_stack : ParticleStack
        Particle stack containing particle data.
    template : torch.Tensor
        3D template volume.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    euler_angles : torch.Tensor
        Per-particle Euler angles.
    euler_angle_offsets : torch.Tensor
        Euler angle offsets to search over.
    defocus_offsets : torch.Tensor
        Defocus offsets to search over.
    pixel_size_offsets : torch.Tensor
        Pixel size offsets to search over.
    apply_global_filtering : bool
        Whether global filtering was already applied.
    device_list : list
        Computational devices to use.
    image_stack : torch.Tensor
        Pre-extracted particle image stack.
    mean_stack : torch.Tensor
        Per-particle correlation mean.
    std_stack : torch.Tensor
        Per-particle correlation variance.
    particle_indices : list[pd.Index]
        Row indices for the provided arrays.

    Returns
    -------
    dict[str, Any]
        Kwargs dictionary for the backend refinement functions.
    """
    imgs_dft, tmpl_dft, proj_filters = particle_stack.prepare_images_and_filters(
        template,
        preprocessing_filters,
        apply_global_filtering=apply_global_filtering,
        particle_images=image_stack,
    )

    return _assemble_backend_kwargs(
        particle_stack,
        template,
        euler_angles,
        euler_angle_offsets,
        defocus_offsets,
        pixel_size_offsets,
        device_list,
        mean_stack,
        std_stack,
        imgs_dft,
        tmpl_dft,
        proj_filters,
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def setup_particle_backend_kwargs(
    particle_stack: "ParticleStack",
    template: torch.Tensor,
    preprocessing_filters: "PreprocessingFilters",
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    apply_global_filtering: bool,
    device_list: list,
    movie: torch.Tensor | None = None,
    deformation_field: CubicCatmullRomGrid3d | None = None,
    particle_shifts: torch.Tensor | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    image_stack: torch.Tensor | None = None,
    mean_stack: torch.Tensor | None = None,
    std_stack: torch.Tensor | None = None,
    particle_indices: list[pd.Index] | None = None,
    images_are_particles: bool = False,
) -> dict[str, Any]:
    """Build backend kwargs. Delegates to the appropriate named function.

    Use ``setup_backend_kwargs_from_micrographs`` or
    ``setup_backend_kwargs_from_particles`` directly for new call sites.
    """
    if images_are_particles:
        if image_stack is None or mean_stack is None or std_stack is None:
            raise ValueError(
                "image_stack, mean_stack, and std_stack must be provided "
                "when images_are_particles=True."
            )
        if particle_indices is None:
            raise ValueError(
                "particle_indices must be provided when images_are_particles=True."
            )
        return setup_backend_kwargs_from_particles(
            particle_stack=particle_stack,
            template=template,
            preprocessing_filters=preprocessing_filters,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            apply_global_filtering=apply_global_filtering,
            device_list=device_list,
            image_stack=image_stack,
            mean_stack=mean_stack,
            std_stack=std_stack,
            particle_indices=particle_indices,
        )
    return setup_backend_kwargs_from_micrographs(
        particle_stack=particle_stack,
        template=template,
        preprocessing_filters=preprocessing_filters,
        euler_angles=euler_angles,
        euler_angle_offsets=euler_angle_offsets,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
        apply_global_filtering=apply_global_filtering,
        device_list=device_list,
        movie=movie,
        deformation_field=deformation_field,
        particle_shifts=particle_shifts,
        pre_exposure=pre_exposure,
        fluence_per_frame=fluence_per_frame,
        image_stack=image_stack,
        mean_stack=mean_stack,
        std_stack=std_stack,
        particle_indices=particle_indices,
    )
