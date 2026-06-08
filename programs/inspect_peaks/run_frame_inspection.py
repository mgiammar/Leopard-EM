"""Example: per-frame local peak inspection (frame correlations)."""

from __future__ import annotations

from pathlib import Path

from leopard_em.pydantic_models.managers import FrameInspectionManager

#######################################
### Editable parameters for program ###
#######################################

# Edit the YAML similarly to refine/inspect configs.
YAML_CONFIG_PATH = "/path/to/frame_inspection_configuration.yaml"

# Batched orientations per GPU call - lower if you run out of memory.
CORRELATION_BATCH_SIZE = 32

# If provided in cross-correlation mode, CSV summaries are written:
# - <base>_frames_mip.csv
# - <base>_frames_pos_x.csv
# - <base>_frames_pos_y.csv
# - <base>.csv (summed refined_mip)
DATAFRAME_OUTPUT_PATH = "/path/to/results_frame_inspection.csv"

# Set True to apply cumulative template dose filtering per frame interval.
APPLY_TEMPLATE_DOSE_WEIGHTING = False


def main() -> None:
    """Run per-frame inspect workflow and write refine-like CSV summaries."""
    manager = FrameInspectionManager.from_yaml(YAML_CONFIG_PATH)
    manager.run_peak_inspection_per_frame(
        correlation_batch_size=CORRELATION_BATCH_SIZE,
        prefer_refined_angles=True,
        apply_projection_normalization=True,
        output_mode="cross_correlation",
        apply_template_dose_weighting=APPLY_TEMPLATE_DOSE_WEIGHTING,
        output_dataframe_path=str(Path(DATAFRAME_OUTPUT_PATH)),
    )


if __name__ == "__main__":
    main()
