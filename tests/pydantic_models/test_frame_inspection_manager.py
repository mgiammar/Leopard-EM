"""Unit tests for the per-frame peak inspection manager."""

import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.config import (
    ComputationalConfigRefine,
    DefocusSearchConfig,
    MovieConfig,
    PixelSizeSearchConfig,
    PreprocessingFilters,
    RefineOrientationConfig,
)
from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack
from leopard_em.pydantic_models.managers.frame_inspection_manager import (
    FrameInspectionManager,
)


def make_frame_inspection_manager() -> FrameInspectionManager:
    """Construct a minimal frame inspection manager for method-level tests."""
    particle_stack = ParticleStack(
        df_path="",
        extracted_box_size=(2, 2),
        original_template_size=(2, 2),
        skip_df_load=True,
    )
    return FrameInspectionManager.model_construct(
        template_volume_path="",
        particle_stack=particle_stack,
        defocus_refinement_config=DefocusSearchConfig(enabled=False),
        pixel_size_refinement_config=PixelSizeSearchConfig(
            enabled=False,
            pixel_size_step=1.0,
        ),
        orientation_refinement_config=RefineOrientationConfig(enabled=False),
        preprocessing_filters=PreprocessingFilters(),
        computational_config=ComputationalConfigRefine(gpu_ids="cpu"),
        movie_config=MovieConfig(enabled=False),
        apply_global_filtering=True,
        template_volume=None,
    )


# ---------------------------------------------------------------------------
# _reduce_cc_peak: pure static reduction of a cross-correlation tensor.
# ---------------------------------------------------------------------------


def test_reduce_cc_peak_single_frame():
    """A single-frame CC tensor reduces to per-particle peak and XY coordinates."""
    # (N=2, n_px=1, n_defocus=1, n_orient=1, H=2, W=2)
    cc = torch.zeros((2, 1, 1, 1, 2, 2), dtype=torch.float32)
    cc[0, 0, 0, 0, 1, 0] = 3.0  # particle 0 peak at (y=1, x=0)
    cc[1, 0, 0, 0, 0, 1] = 5.0  # particle 1 peak at (y=0, x=1)

    mip, pos_x, pos_y = FrameInspectionManager._reduce_cc_peak(cc, n_batch_dims=1)

    assert isinstance(mip, torch.Tensor)
    assert mip.device.type == "cpu"
    assert torch.equal(mip, torch.tensor([3.0, 5.0]))
    assert torch.equal(pos_x, torch.tensor([0, 1]))
    assert torch.equal(pos_y, torch.tensor([1, 0]))


def test_reduce_cc_peak_stacked_frames_preserves_batch_dims():
    """A stacked (T, N, ...) CC tensor reduces over both leading batch dims."""
    # (T=2, N=2, n_px=1, n_defocus=1, n_orient=1, H=2, W=2)
    cc = torch.zeros((2, 2, 1, 1, 1, 2, 2), dtype=torch.float32)
    cc[0, 0, 0, 0, 0, 1, 0] = 3.0
    cc[0, 1, 0, 0, 0, 0, 1] = 5.0
    cc[1, 0, 0, 0, 0, 0, 0] = 7.0
    cc[1, 1, 0, 0, 0, 1, 1] = 11.0

    mip, pos_x, pos_y = FrameInspectionManager._reduce_cc_peak(cc, n_batch_dims=2)

    assert isinstance(mip, torch.Tensor)
    assert mip.shape == (2, 2)
    assert mip.tolist() == [[3.0, 5.0], [7.0, 11.0]]
    assert pos_y.tolist() == [[1, 0], [0, 1]]
    assert pos_x.tolist() == [[0, 1], [0, 1]]


def test_reduce_cc_peak_rejects_wrong_dimensionality():
    """Reduction validates the expected ``n_batch_dims + 5`` tensor rank."""
    cc = torch.zeros((2, 1, 1, 2, 2), dtype=torch.float32)  # 5 dims, expects 6
    with pytest.raises(ValueError, match="Expected cross-correlation tensor"):
        FrameInspectionManager._reduce_cc_peak(cc, n_batch_dims=1)


# ---------------------------------------------------------------------------
# _stack_frame_results: pure static stacking of per-frame backend outputs.
# ---------------------------------------------------------------------------


def test_stack_frame_results_cross_correlation():
    """Cross-correlation results stack with frame as the leading axis, in order."""
    frame_results = [
        torch.full((2, 1, 1, 1, 2, 2), float(i), dtype=torch.float32) for i in range(3)
    ]
    stacked = FrameInspectionManager._stack_frame_results(
        frame_results, output_mode="cross_correlation"
    )
    assert isinstance(stacked, torch.Tensor)
    assert stacked.shape == (3, 2, 1, 1, 1, 2, 2)
    assert torch.all(stacked[0] == 0)
    assert torch.all(stacked[2] == 2)


def test_stack_frame_results_frc_returns_tensor_and_shared_bins():
    """FRC results stack the spectra and carry the frequency bins through once."""
    freq_bins = torch.linspace(0.0, 0.5, 4)
    frame_results = [
        (torch.full((2, 1, 1, 1, 4), float(i)), freq_bins) for i in range(3)
    ]
    stacked, bins = FrameInspectionManager._stack_frame_results(
        frame_results, output_mode="frc"
    )
    assert stacked.shape == (3, 2, 1, 1, 1, 4)
    assert torch.equal(bins, freq_bins)


def test_stack_frame_results_empty_raises():
    """An empty result list is an error in either output mode."""
    with pytest.raises(ValueError, match="No frame results"):
        FrameInspectionManager._stack_frame_results([], output_mode="cross_correlation")


# ---------------------------------------------------------------------------
# _frame_dose_template: per-frame template selection / dose weighting.
# ---------------------------------------------------------------------------


def test_frame_dose_template_disabled_returns_same_template():
    """With dose weighting off, the shared template is returned unchanged."""
    manager = make_frame_inspection_manager()
    template = torch.randn((4, 4, 4))
    result = manager._frame_dose_template(
        template, frame_idx=2, apply_template_dose_weighting=False
    )
    assert result is template


def test_frame_dose_template_applies_distinct_dose_per_frame():
    """Each frame's template is dose-weighted over its own exposure interval."""
    manager = make_frame_inspection_manager()
    manager.particle_stack._df = pd.DataFrame(
        {
            "particle_index": [0, 1],
            "pixel_size": [1.0, 1.0],
            "refined_pixel_size": [1.0, 1.0],
        }
    )
    manager.movie_config = MovieConfig(
        enabled=True,
        movie_path="movie.mrc",
        pre_exposure=0.0,
        fluence_per_frame=10.0,
    )

    torch.manual_seed(0)
    template = torch.randn((6, 6, 6))

    frame0 = manager._frame_dose_template(
        template, frame_idx=0, apply_template_dose_weighting=True
    )
    frame1 = manager._frame_dose_template(
        template, frame_idx=1, apply_template_dose_weighting=True
    )

    # Dose weighting actually attenuates the template ...
    assert not torch.allclose(frame0, template)
    # ... and later frames (higher cumulative exposure) differ from earlier ones.
    assert not torch.allclose(frame0, frame1)
    assert frame0.shape == template.shape


# ---------------------------------------------------------------------------
# run_peak_inspection_per_frame: real input-validation guard (no backend).
# ---------------------------------------------------------------------------


def test_run_peak_inspection_per_frame_requires_movie():
    """Per-frame inspection fails fast when the movie config is disabled."""
    manager = make_frame_inspection_manager()
    with pytest.raises(ValueError, match="requires movie_config.enabled"):
        manager.run_peak_inspection_per_frame(template_tensor=torch.zeros((2, 2, 2)))


# ---------------------------------------------------------------------------
# CSV output: real reduction + dataframe writing, no faked internals.
# ---------------------------------------------------------------------------


def test_process_frame_results_writes_refine_style_csvs(tmp_path):
    """Frame result processing should emit per-frame and summed CSV outputs."""
    manager = make_frame_inspection_manager()
    manager.particle_stack._df = pd.DataFrame({"particle_index": [0, 1]})
    manager.movie_config = MovieConfig(enabled=True, movie_path="movie.mrc")

    frame_results = torch.zeros((2, 2, 1, 1, 1, 2, 2), dtype=torch.float32)
    # Frame 0
    frame_results[0, 0, 0, 0, 0, 1, 0] = 3.0
    frame_results[0, 1, 0, 0, 0, 0, 1] = 5.0
    # Frame 1
    frame_results[1, 0, 0, 0, 0, 0, 0] = 7.0
    frame_results[1, 1, 0, 0, 0, 1, 1] = 11.0

    output_csv = tmp_path / "frame_results.csv"
    manager.process_frame_results(
        frame_results=frame_results,
        output_dataframe_path=str(output_csv),
    )

    frames_mip = pd.read_csv(tmp_path / "frame_results_frames_mip.csv")
    frames_pos_x = pd.read_csv(tmp_path / "frame_results_frames_pos_x.csv")
    frames_pos_y = pd.read_csv(tmp_path / "frame_results_frames_pos_y.csv")
    summed = pd.read_csv(output_csv)

    assert list(frames_mip["frame_0_mip"]) == [3.0, 5.0]
    assert list(frames_mip["frame_1_mip"]) == [7.0, 11.0]
    assert list(frames_mip["sum_frames_mip"]) == [10.0, 16.0]
    assert list(frames_pos_x["frame_0_pos_x"]) == [0, 1]
    assert list(frames_pos_y["frame_0_pos_y"]) == [1, 0]
    assert list(summed["refined_mip"]) == [10.0, 16.0]
    assert "refined_scaled_mip" not in summed.columns
    assert not (tmp_path / "frame_results_frames_zscore.csv").exists()


def test_write_reduced_cross_correlation_csvs_includes_cc_of_sum(tmp_path):
    """Frames and main CSV should expose cc_of_sum_mip alongside sum_frames_mip."""
    manager = make_frame_inspection_manager()
    manager.particle_stack._df = pd.DataFrame({"particle_index": [0, 1]})
    manager.movie_config = MovieConfig(enabled=True, movie_path="movie.mrc")

    refined_mips = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    pos_x = torch.zeros((2, 2), dtype=torch.float64)
    pos_y = torch.zeros((2, 2), dtype=torch.float64)
    cc_of_sum = torch.tensor([42.0, 43.0], dtype=torch.float64)
    output_csv = tmp_path / "results.csv"

    manager._write_reduced_cross_correlation_csvs(
        refined_mips=refined_mips,
        pos_x=pos_x,
        pos_y=pos_y,
        output_dataframe_path=str(output_csv),
        cc_of_sum_mip=cc_of_sum,
    )

    frames_mip = pd.read_csv(tmp_path / "results_frames_mip.csv")
    summed = pd.read_csv(output_csv)

    assert list(frames_mip["sum_frames_mip"]) == [3.0, 7.0]
    assert list(frames_mip["cc_of_sum_mip"]) == [42.0, 43.0]
    assert list(summed["refined_mip"]) == [3.0, 7.0]
    assert list(summed["cc_of_sum_mip"]) == [42.0, 43.0]
