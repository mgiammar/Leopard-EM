"""Utility functions and constants shared across the particle_stack module."""

import json
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch

from leopard_em.utils.data_io import load_mrc_image

TORCH_TO_NUMPY_PADDING_MODE: dict[str, str] = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}

_HDF5_PARTICLES_GROUP = "particles"
_HDF5_LOCAL_STATS_GROUP = "local_stats"
_HDF5_IMAGE_STACK_DATASET = "image_stack"
_HDF5_STRING_DTYPE = h5py.string_dtype()

# Maps DataFrame path names --> dataset paths inside a MatchTemplateResultHDF5 file
_COLUMN_TO_HDF5_DATASET: dict[str, str] = {
    "mip_path": "tensors/mip",
    "scaled_mip_path": "tensors/scaled_mip",
    "psi_path": "tensors/orientation_psi",
    "theta_path": "tensors/orientation_theta",
    "phi_path": "tensors/orientation_phi",
    "defocus_path": "tensors/relative_defocus",
    "correlation_average_path": "tensors/correlation_average",
    "correlation_variance_path": "tensors/correlation_variance",
}

_HDF5_EXTENSIONS = frozenset({".h5", ".hdf5"})


def _load_image_2d_from_hdf5(path: str, column_name: str) -> torch.Tensor:
    """Load a single 2-D result tensor from a MatchTemplateResultHDF5 file."""
    dataset_path = _COLUMN_TO_HDF5_DATASET.get(column_name)
    if dataset_path is None:
        raise ValueError(
            f"Column '{column_name}' has no HDF5 dataset mapping. "
            "Only result tensor path columns support HDF5 loading."
        )
    with h5py.File(path, "r") as f:
        if dataset_path not in f:
            raise ValueError(
                f"Dataset '{dataset_path}' not found in '{path}'. "
                "File may not be a MatchTemplateResultHDF5 file."
            )
        return torch.from_numpy(f[dataset_path][:].astype("float32"))


def _load_image_2d(path: str, column_name: str) -> torch.Tensor:
    """Load a single 2-D image tensor, dispatching on file extension."""
    if Path(path).suffix.lower() in _HDF5_EXTENSIONS:
        return _load_image_2d_from_hdf5(path, column_name)
    return load_mrc_image(path)


# TODO: Make this a shared utility function across the package somehow
def _leopard_em_version() -> str:
    try:
        return version("leopard_em")
    except PackageNotFoundError:
        return "uninstalled"


def _any_nan_or_inf(s: pd.Series | np.ndarray) -> bool:
    """Helper to check if any value in the Series or array is NaN or infinite."""
    if isinstance(s, np.ndarray):
        return bool(np.any(np.isnan(s)) or np.any(np.isinf(s)))
    return bool(s.isna().any() or s.isin([float("inf"), float("-inf")]).any())


def _generate_particle_ids(df: pd.DataFrame) -> list[str]:
    """Generate particle IDs of the form ``{mic_stem}_{local_idx:05d}``."""
    ids: pd.Series = pd.Series("", index=df.index, dtype=object)
    for mic_path, group in df.groupby("micrograph_path", sort=False):
        stem = Path(str(mic_path)).stem
        for local_idx, row_label in enumerate(group.index):
            ids.at[row_label] = f"{stem}_{local_idx:05d}"
    return [str(v) for v in ids]


# TODO: Better management of Zernikie coefficient columns/arrays in the HDF5 format...
#       This is a lot of boilerplate code, and probably a better schema would eliminate
#       these parsing needs.
def _value_to_str(v: Any) -> str:
    """Serialize a value to a string for HDF5 storage."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v)


def _str_to_value(s: str) -> Any:
    """Deserialize a string back to a Python value after HDF5 load."""
    if s == "":
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, dict)):
            return parsed
        # Plain JSON scalars (numbers) that were originally strings stay as strings
        return s
    except (json.JSONDecodeError, ValueError):
        return s


def _parse_json_series_value(value: Any) -> dict | None:
    """Parse a Series value that may be a JSON string, dict, None, or NaN.

    Returns the parsed dict, the original dict, or None for missing values.
    Raises ValueError for any other type.
    """
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


# NOTE: How are the internals of the hdf5 particle stack being handled? Are they just a
#       pass through for the DataFrame type backed class (don't want this). Need to
#       implement things at the base class level somehow.
def _write_df_to_hdf5_group(f: h5py.File, df: pd.DataFrame) -> None:
    """Write a DataFrame's columns to ``f[_HDF5_PARTICLES_GROUP]``.

    Numeric columns are stored as float64 datasets.  String / object columns
    (including path columns, Zernike coefficient arrays, etc.) are serialized
    to variable-length UTF-8 strings via ``_value_to_str``.

    The dataset names match the DataFrame column names.  ``particle_id``
    (which may be the DataFrame index) is always written as an explicit
    dataset and listed first in ``attrs["columns"]``.
    """
    grp = f.create_group(_HDF5_PARTICLES_GROUP)

    # Build the list of columns to write, ensuring particle_id comes first.
    if df.index.name == "particle_id":
        particle_ids = df.index.tolist()
        col_names = ["particle_id", *list(df.columns)]
    else:
        # particle_id may be an ordinary column
        particle_ids = df["particle_id"].tolist() if "particle_id" in df.columns else []
        col_names = list(df.columns)

    grp.attrs["columns"] = col_names

    # Write particle_id dataset
    if particle_ids:
        grp.create_dataset(
            "particle_id",
            data=np.array([str(v) for v in particle_ids], dtype=object),
            dtype=_HDF5_STRING_DTYPE,
        )

    for col in df.columns:
        if col == "particle_id":
            # Already written above (or will be skipped if index)
            if df.index.name != "particle_id":
                continue
        series = df[col]
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
            grp.create_dataset(col, data=series.to_numpy(dtype=np.float64))
        else:
            str_data = [_value_to_str(v) for v in series]
            grp.create_dataset(
                col,
                data=np.array(str_data, dtype=object),
                dtype=_HDF5_STRING_DTYPE,
            )


def _read_df_from_hdf5_group(f: h5py.File) -> pd.DataFrame:
    """Reconstruct a DataFrame from ``f[_HDF5_PARTICLES_GROUP]``.

    ``particle_id`` is restored as the pandas ``Index``.
    """
    grp = f[_HDF5_PARTICLES_GROUP]
    columns: list[str] = list(grp.attrs["columns"])

    data: dict[str, Any] = {}
    for col in columns:
        if col not in grp:
            continue
        raw = grp[col][:]
        if raw.dtype.kind in ("O", "S", "U"):
            decoded = [s.decode() if isinstance(s, bytes) else s for s in raw]
            data[col] = [_str_to_value(s) for s in decoded]
        else:
            data[col] = raw

    df = pd.DataFrame(data)

    if "particle_id" in df.columns:
        df = df.set_index("particle_id")
        df.index.name = "particle_id"

    return df


# ---------------------------------------------------------------------------
# Stand-alone image-extraction helpers (unchanged from original module)
# ---------------------------------------------------------------------------


def get_cropped_image_regions(
    image: torch.Tensor | np.ndarray,
    pos_y: torch.Tensor | np.ndarray,
    pos_x: torch.Tensor | np.ndarray,
    box_size: int | tuple[int, int],
    pos_reference: Literal["center", "top-left"] = "top-left",
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Extracts regions from an image into a stack of cropped images.

    The `pos_reference` argument determines how the (y, x) coordinates are interpreted
    when extracting boxes:

    - If ``pos_reference="center"``:
        The (y, x) coordinate refers to the **center** of the box.
        The box extends from (y - height // 2, x - width // 2) to
        (y + height // 2, x + width // 2).

        Example:
            :                +------------------+
            :                |                  |
            :              height      * (y, x) |
            :                |                  |
            :                +------ width -----+

    - If ``pos_reference="top-left"``:
        The (y, x) coordinate refers to the **top-left corner** of the box.
        The box extends from (y, x) to (y + height, x + width).

        Example:
            :         (y, x) *------ width -----+
            :                |                  |
            :                |                height
            :                |                  |
            :                +------------------+

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        The input image from which to extract the regions.
    pos_y : torch.Tensor | np.ndarray
        The y positions of the regions to extract. Type must mach `image`
    pos_x : torch.Tensor | np.ndarray
        The x positions of the regions to extract. Type must mach `image`
    box_size : int | tuple[int, int]
        The size of the box to extract. If an integer is passed, the box will be square.
    pos_reference : Literal["center", "top-left"], optional
        The reference point for the positions, by default "center". If "center", the
        boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...]. If
        "top-left", the boxes will be image[y : y + box_size, ...].
    handle_bounds : Literal["pad", "clip", "error"], optional
        How to handle the bounds of the image, by default "pad". If "pad", the image
        will be padded with the padding value based on the padding mode. If "error", an
        error will be raised if any region exceeds the image bounds. Note clipping is
        not supported since returned stack may have inhomogeneous sizes.
    padding_mode : Literal["constant", "reflect", "replicate"], optional
        The padding mode to use when padding the image, by default "constant".
        "constant" pads with the value `padding_value`, "reflect" pads with the
        reflection of the image at the edge, and "replicate" pads with the last pixel
        of the image. These match the modes available in `torch.nn.functional.pad`.
    padding_value : float, optional
        The value to use for padding when `padding_mode` is "constant", by default 0.0.

    Returns
    -------
    torch.Tensor | np.ndarray
        The stack of cropped images extracted from the input image. Type will match the
        input image type.

    Raises
    ------
    ValueError
        If `pos_reference` is not one of "center" or "top-left", or if `image` is not a
        torch.Tensor or np.ndarray.
    """
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    if pos_reference == "center":
        pos_y = pos_y - box_size[0] // 2
        pos_x = pos_x - box_size[1] // 2
    elif pos_reference == "top-left":
        pass
    else:
        raise ValueError(f"Unknown pos_reference: {pos_reference}")

    if isinstance(image, torch.Tensor):
        return _get_cropped_image_regions_torch(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    if isinstance(image, np.ndarray):
        padding_mode_np = TORCH_TO_NUMPY_PADDING_MODE[padding_mode]
        return _get_cropped_image_regions_numpy(
            image=image,
            pos_y=pos_y,
            pos_x=pos_x,
            box_size=box_size,
            handle_bounds=handle_bounds,
            padding_mode=padding_mode_np,
            padding_value=padding_value,
        )

    raise ValueError(f"Unknown image type: {type(image)}")


# pylint: disable=too-many-locals
def _get_cropped_image_regions_numpy(
    image: np.ndarray,
    pos_y: np.ndarray,
    pos_x: np.ndarray,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: str,
    padding_value: float,
) -> np.ndarray:
    """Helper function for extracting regions from a numpy array.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] - 1
        bs0 = box_size[0] - 1
        pad_kwargs = {}
        if padding_mode == "constant":
            pad_kwargs["constant_values"] = padding_value
        image = np.pad(
            image,
            pad_width=((bs0, bs0), (bs1, bs1)),
            mode=padding_mode,
            **pad_kwargs,
        )
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    regions = []
    for y, x in zip(pos_y, pos_x):
        if (
            y < 0
            or x < 0
            or y + box_size[0] > image.shape[0]
            or x + box_size[1] > image.shape[1]
        ):
            raise IndexError(
                f"Region bounds [{y}:{y + box_size[0]}, {x}:{x + box_size[1]}] exceed "
                f"image dimensions {image.shape}"
            )

        regions.append(image[y : y + box_size[0], x : x + box_size[1]])

    cropped_images = np.stack(regions)

    return cropped_images


# pylint: disable=too-many-locals
def _get_cropped_image_regions_torch(
    image: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: Literal["constant", "reflect", "replicate"],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Helper function for extracting regions from a torch tensor.

    NOTE: this function assumes that the position reference is the top-left corner.
    Reference value is handled by the user-exposed 'get_cropped_image_regions' function.
    """
    if handle_bounds == "pad":
        bs1 = box_size[1] - 1
        bs0 = box_size[0] - 1
        pad_kwargs = {}
        if padding_mode == "constant":
            pad_kwargs["value"] = padding_value
        # NOTE: Need to do unsqueeze/squeeze workaround to make torch happy with input
        # tensor shapes. Looks like API for padding may change in the future torch...
        image = torch.nn.functional.pad(
            image.unsqueeze(0),
            pad=(bs1, bs1, bs0, bs0),
            mode=padding_mode,
            **pad_kwargs,
        ).squeeze(0)
        pos_y = pos_y + bs0
        pos_x = pos_x + bs1

    regions = []
    for y, x in zip(pos_y, pos_x):
        y = int(y.item() if hasattr(y, "item") else y)
        x = int(x.item() if hasattr(x, "item") else x)
        original_y, original_x = y, x

        if (
            y < 0
            or x < 0
            or y + box_size[0] > image.shape[0]
            or x + box_size[1] > image.shape[1]
        ):
            if handle_bounds == "error":
                raise IndexError(
                    f"Region bounds [{original_y}:{original_y + box_size[0]}, "
                    f"{original_x}:{original_x + box_size[1]}] exceed "
                    f"image dimensions {image.shape}"
                )
            warnings.warn(
                f"Region bounds [{original_y}:{original_y + box_size[0]}, "
                f"{original_x}:{original_x + box_size[1]}] exceed "
                f"image dimensions {image.shape}. Clamping to edges.",
                UserWarning,
                stacklevel=2,
            )
            y = max(0, min(y, image.shape[0] - box_size[0]))
            x = max(0, min(x, image.shape[1] - box_size[1]))

        regions.append(image[y : y + box_size[0], x : x + box_size[1]])

    cropped_images = torch.stack(regions)

    return cropped_images


def _parse_mag_matrix_value_to_tensor(value: Any) -> torch.Tensor | None:
    """Parsing logic for an element in the "mag_matrix" column of a particle stack."""
    mag_matrix_list: list[float] | None = None

    # Case of string representing list
    if isinstance(value, str):
        mag_matrix_list = [float(x) for x in value.strip("[]").split(",")]
        assert len(mag_matrix_list) == 4

    # Case of list of 4 numeric values
    if isinstance(value, list):
        assert len(value) == 4
        if all(isinstance(x, (int, float)) and not np.isnan(x) for x in value):
            mag_matrix_list = [float(x) for x in value]
        else:
            raise ValueError(
                f"Expected list of 4 numeric values for mag_matrix, but got: {value}"
            )

    # No valid "mag_matrix" to parse
    if mag_matrix_list is None:
        return None

    mag_matrix_tensor = torch.tensor(
        [
            [mag_matrix_list[0], mag_matrix_list[1]],
            [mag_matrix_list[2], mag_matrix_list[3]],
        ],
        dtype=torch.float32,
    )
    return mag_matrix_tensor
