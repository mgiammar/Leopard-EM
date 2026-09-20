"""Tests for the ctf_utils module."""

import pandas as pd

from leopard_em.utils.ctf_utils import _setup_ctf_kwargs_from_particle_stack


def test_setup_ctf_kwargs_from_particle_stack_with_string_index():
    """CTF kwargs are read from the first row via positional access.

    ``ParticleStackHDF5``-backed stacks are indexed by string ``particle_id``
    labels rather than a default integer RangeIndex. The lookups here must use
    ``.iloc[0]`` (positional) rather than ``[0]`` (label-based), which would
    raise a ``KeyError`` on such a stack.
    """
    df = pd.DataFrame(
        {
            "voltage": [300.0, 300.0],
            "spherical_aberration": [2.7, 2.7],
            "amplitude_contrast_ratio": [0.07, 0.07],
            "phase_shift": [0.0, 0.0],
            "ctf_B_factor": [0.0, 0.0],
            "refined_pixel_size": [1.06, 1.06],
            "mag_matrix": [None, None],
            "even_zernikes": [None, None],
            "odd_zernikes": [None, None],
        },
        index=pd.Index(["p_00000", "p_00001"], name="particle_id"),
    )

    kwargs = _setup_ctf_kwargs_from_particle_stack(df, template_shape=(64, 64))

    assert kwargs["voltage"] == 300.0
    assert kwargs["spherical_aberration"] == 2.7
    assert kwargs["amplitude_contrast_ratio"] == 0.07
    assert kwargs["ctf_B_factor"] == 0.0
    assert kwargs["phase_shift"] == 0.0
    assert kwargs["pixel_size"] == 1.06
    assert kwargs["template_shape"] == (64, 64)
