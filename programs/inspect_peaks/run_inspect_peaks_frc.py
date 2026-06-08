"""Example: print one particle's FRC curve from peak inspection."""

# NOTE: The ``if __name__ == "__main__"`` guard is required for multiprocessing.
import time

import pandas as pd
import torch

from leopard_em.pydantic_models.managers import PeakInspectionManager

#######################################
### Editable parameters for program ###
#######################################

# Edit the YAML the same way you would for refine template.
YAML_CONFIG_PATH = "/path/to/inspect_peaks_frc_configuration.yaml"

# Batched orientations per GPU call - lower if you run out of memory.
CORRELATION_BATCH_SIZE = 32

# Set inspect backend mode to "frc" to run frequency-ring correlation output.
OUTPUT_MODE = "frc"

# How we pick the top row in the particle CSV (try in order, higher = better).
SCORE_COLUMNS = (
    "refined_scaled_mip",
    "refined_mip",
    "scaled_mip",
    "mip",
)


def _stack_row_for_best_particle(df: pd.DataFrame) -> int:
    for col in SCORE_COLUMNS:
        if col in df.columns:
            return int(df[col].to_numpy().argmax())
    raise ValueError(f"No known score column found; expected one of {SCORE_COLUMNS!r}.")


def _print_frc_curve_for_top_particle(
    frc_tensor: torch.Tensor, frequency_bins: torch.Tensor, stack_row: int
) -> None:
    """Print FRC values for one particle at defocus/orientation index 0,0."""
    frc_curve = frc_tensor[stack_row, 0, 0, 0].detach().float().cpu()
    freq = frequency_bins.detach().float().cpu()
    print(
        f"\nFRC curve for stack row {stack_row} at defocus_index=0, orientation_index=0"
    )
    print("frequency,frc")
    for f, v in zip(freq.tolist(), frc_curve.tolist(), strict=True):
        print(f"{f:.6f},{v:.6f}")


def main() -> None:
    """Run peak inspection in FRC mode and print one particle's FRC curve."""
    manager = PeakInspectionManager.from_yaml(YAML_CONFIG_PATH)
    df = pd.read_csv(manager.particle_stack.df_path)
    n = len(df)
    if n == 0:
        raise ValueError("Particle dataframe is empty.")
    best_row = _stack_row_for_best_particle(df)

    print("Loaded configuration (same schema as refine template).")
    print("Running peak inspection in FRC mode (full stack; may take a while)...")

    start_time = time.time()
    inspection_result = manager.run_peak_inspection(
        correlation_batch_size=CORRELATION_BATCH_SIZE,
        prefer_refined_angles=True,
        output_mode=OUTPUT_MODE,
    )
    end_time = time.time()
    print(f"Finished ``run_peak_inspection`` in {end_time - start_time:.1f} s")

    if (
        not isinstance(inspection_result, tuple)
        or len(inspection_result) != 2
        or not isinstance(inspection_result[0], torch.Tensor)
        or not isinstance(inspection_result[1], torch.Tensor)
    ):
        raise ValueError("Expected (frc_tensor, frequency_bins) tuple in frc mode.")
    inspection, frequency_bins = inspection_result
    if inspection.ndim != 5:
        raise ValueError(
            "Expected FRC tensor (N, n_px, n_def, n_orient, n_freq); "
            f"got {tuple(inspection.shape)}"
        )
    if int(inspection.shape[0]) != n:
        raise ValueError(
            f"Inspect output length {inspection.shape[0]} != dataframe rows {n}."
        )
    print(f"FRC tensor shape: {tuple(inspection.shape)}")
    print(f"Frequency bins shape: {tuple(frequency_bins.shape)}")

    if "particle_index" in df.columns:
        pidx = int(df["particle_index"].iloc[best_row])
        print(f"\nBest-particle row in CSV: {best_row} (particle_index={pidx})")
    else:
        print(f"\nBest-particle row in CSV: {best_row}")

    _print_frc_curve_for_top_particle(inspection, frequency_bins, best_row)
    print("\nDone!")


if __name__ == "__main__":
    main()
