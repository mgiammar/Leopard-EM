"""CTF (Contrast Transfer Function) utility functions."""

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch_ctf import calculate_ctf_2d
from torch_fourier_filter.envelopes import b_envelope

from leopard_em.utils.search_utils import get_cs_range

# Using the TYPE_CHECKING statement to avoid circular imports
if TYPE_CHECKING:
    from leopard_em.pydantic_models.data_structures.optics_group import OpticsGroup
    from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack


def calculate_ctf_filter_stack_full_args(
    template_shape: tuple[int, int],
    defocus_u: float,  # in Angstrom
    defocus_v: float,  # in Angstrom
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
    **kwargs: Any,
) -> torch.Tensor:
    """Calculate a CTF filter stack for a given set of parameters and search offsets.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    defocus_u : float
        Defocus along the major axis, in Angstroms.
    defocus_v : float
        Defocus along the minor axis, in Angstroms.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.
    **kwargs
        Additional keyword to pass to the calculate_ctf_2d function.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and parameters. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)
    """
    # Calculate the defocus values + offsets in terms of Angstrom
    defocus = defocus_offsets + ((defocus_u + defocus_v) / 2)
    astigmatism = abs(defocus_u - defocus_v) / 2

    # The different Cs values to search over as another dimension
    cs_values = get_cs_range(
        pixel_size=kwargs["pixel_size"],
        pixel_size_offsets=pixel_size_offsets,
        cs=kwargs["spherical_aberration"],
    )

    # Ensure defocus and astigmatism have a batch dimension so Cs and defocus can be
    # interleaved correctly
    if defocus.dim() == 0:
        defocus = defocus.unsqueeze(0)

    # Convert mag_matrix from list to 2x2 tensor if provided
    mag_matrix = kwargs.get("mag_matrix")
    if mag_matrix is not None:
        if isinstance(mag_matrix, list):
            mag_matrix = torch.tensor(
                [
                    [mag_matrix[0], mag_matrix[1]],
                    [mag_matrix[2], mag_matrix[3]],
                ],
                dtype=torch.float32,
            )
        elif not isinstance(mag_matrix, torch.Tensor):
            # If it's neither a list nor a tensor, try to convert it
            mag_matrix = torch.tensor(mag_matrix, dtype=torch.float32)

        # Ensure mag_matrix is on the same device and has the correct dtype
        mag_matrix = mag_matrix.to(device=defocus.device, dtype=torch.float32)

    # Loop over spherical aberrations one at a time and collect results
    ctf_list = []
    for cs_val in cs_values:
        tmp = calculate_ctf_2d(
            defocus=defocus * 1e-4,  # Convert to um from Angstrom
            astigmatism=astigmatism * 1e-4,  # Convert to um from Angstrom
            astigmatism_angle=kwargs["astigmatism_angle"],
            voltage=kwargs["voltage"],
            spherical_aberration=cs_val,
            amplitude_contrast=kwargs["amplitude_contrast_ratio"],
            phase_shift=kwargs["phase_shift"],
            pixel_size=kwargs["pixel_size"],
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
            even_zernike_coeffs=kwargs["even_zernikes"],
            odd_zernike_coeffs=kwargs["odd_zernikes"],
            transform_matrix=mag_matrix,
        )
        # calc B-envelope and apply
        b_envelope_tmp = b_envelope(
            B=kwargs["ctf_B_factor"],
            image_shape=template_shape,
            pixel_size=kwargs["pixel_size"],
            rfft=True,
            fftshift=False,
            device=tmp.device,
        )
        tmp *= b_envelope_tmp
        ctf_list.append(tmp)

    ctf = torch.stack(ctf_list, dim=0)

    return ctf


def calculate_ctf_filter_stack(
    template_shape: tuple[int, int],
    optics_group: "OpticsGroup",
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
) -> torch.Tensor:
    """Calculate searched CTF filter values for a given shape and optics group.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    optics_group : OpticsGroup
        OpticsGroup object containing the optics defining the CTF parameters.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and optics group. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)
    """
    return calculate_ctf_filter_stack_full_args(
        template_shape,
        optics_group.defocus_u,
        optics_group.defocus_v,
        defocus_offsets,
        pixel_size_offsets,
        astigmatism_angle=optics_group.astigmatism_angle,
        voltage=optics_group.voltage,
        spherical_aberration=optics_group.spherical_aberration,
        amplitude_contrast_ratio=optics_group.amplitude_contrast_ratio,
        ctf_B_factor=optics_group.ctf_B_factor,
        phase_shift=optics_group.phase_shift,
        pixel_size=optics_group.pixel_size,
        even_zernikes=optics_group.even_zernikes,
        odd_zernikes=optics_group.odd_zernikes,
        mag_matrix=optics_group.mag_matrix_tensor,
    )


def _parse_json_string_from_series_value(value: Any) -> dict | None:
    """Parse a value that may be a JSON string, dict, None, or NaN.

    Parameters
    ----------
    value : Any
        The value to parse. Can be a JSON string, dict, None, or NaN
        (from empty CSV fields).

    Returns
    -------
    dict | None
        Parsed dict if value was a JSON string, original dict if already a dict,
        or None if value was None or NaN.

    Raises
    ------
    ValueError
        If the value cannot be parsed as a dict or is not None/NaN.
    """
    # Handle NaN values from empty CSV fields (pandas converts empty fields to NaN)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and value:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(
            f"Expected JSON string to parse to dict, but got {type(parsed).__name__}: "
            f"{parsed}"
        )
    if isinstance(value, dict):
        return value
    raise ValueError(
        f"Expected dict, JSON string, None, or NaN, but got {type(value).__name__}: "
        f"{value}"
    )


def _setup_ctf_kwargs_from_particle_stack(
    particle_stack: "ParticleStack", template_shape: tuple[int, int]
) -> dict[str, Any]:
    """Build CTF kwargs dict from a particle stack.

    Delegates to ``particle_stack.get_ctf_kwargs``.
    """
    warnings.warn(
        "_setup_ctf_kwargs_from_particle_stack is deprecated. "
        "Please use particle_stack.get_ctf_kwargs(template_shape) directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return particle_stack.get_ctf_kwargs(template_shape)
