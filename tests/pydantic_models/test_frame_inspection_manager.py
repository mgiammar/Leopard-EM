import numpy as np
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


def test_run_peak_inspection_per_frame_requires_movie():
    """Per-frame inspection should fail before doing template/backend work."""
    manager = make_frame_inspection_manager()

    with pytest.raises(ValueError, match="requires movie_config.enabled"):
        manager.run_peak_inspection_per_frame(template_tensor=torch.zeros((2, 2, 2)))


def test_run_peak_inspection_per_frame_stacks_cross_correlation(monkeypatch):
    """The public manager API should stack per-frame inspect outputs."""
    manager = make_frame_inspection_manager()

    def fake_load_and_setup(self):
        return torch.zeros((3, 4, 4)), None, None

    def fake_prepare_template(self, template_tensor=None):
        return torch.zeros((2, 2, 2))

    def fake_setup_independent(self, template, prefer_refined_angles=True):
        return {"frame_independent": torch.tensor(1)}

    def fake_construct_particle_movie_stack(self, **kwargs):
        return torch.zeros((3, 2, 4, 4))

    def fake_setup_frame_kwargs(self, frame_particle_stack, template):
        return {"frame_shape": torch.tensor(frame_particle_stack.shape)}

    call_count = {"value": 0}

    def fake_get_peak_inspection_result(
        self,
        backend_kwargs,
        correlation_batch_size=32,
        apply_projection_normalization=True,
        output_mode="cross_correlation",
    ):
        frame_idx = call_count["value"]
        call_count["value"] += 1
        return torch.full((2, 1, 1, 1, 2, 2), frame_idx, dtype=torch.float32)

    monkeypatch.setattr(
        FrameInspectionManager,
        "_load_and_setup_frame_inspection",
        fake_load_and_setup,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_prepare_frame_template",
        fake_prepare_template,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_setup_frame_independent_kwargs",
        fake_setup_independent,
    )
    monkeypatch.setattr(
        ParticleStack,
        "construct_particle_movie_stack",
        fake_construct_particle_movie_stack,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_setup_frame_kwargs",
        fake_setup_frame_kwargs,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "get_peak_inspection_result",
        fake_get_peak_inspection_result,
    )

    result = manager.run_peak_inspection_per_frame()

    assert result.shape == (3, 2, 1, 1, 1, 2, 2)
    assert torch.all(result[0] == 0)
    assert torch.all(result[2] == 2)


def test_run_peak_inspection_per_frame_applies_template_dose_weighting(monkeypatch):
    """Dose-weighted template path should call per-frame dose filter helper."""
    manager = make_frame_inspection_manager()

    def fake_load_and_setup(self):
        return torch.zeros((2, 4, 4)), None, None

    def fake_prepare_template(self, template_tensor=None):
        return torch.zeros((2, 2, 2))

    def fake_setup_independent(self, template, prefer_refined_angles=True):
        return {"frame_independent": torch.tensor(1)}

    def fake_construct_particle_movie_stack(self, **kwargs):
        return torch.zeros((2, 1, 4, 4))

    def fake_setup_frame_kwargs(self, frame_particle_stack, template):
        return {"template_sum": template.sum()}

    dose_calls = {"count": 0}

    def fake_apply_dose(self, template, frame_idx):
        dose_calls["count"] += 1
        return template + frame_idx

    def fake_get_peak_inspection_result(
        self,
        backend_kwargs,
        correlation_batch_size=32,
        apply_projection_normalization=True,
        output_mode="cross_correlation",
    ):
        return torch.zeros((1, 1, 1, 1, 2, 2))

    monkeypatch.setattr(
        FrameInspectionManager,
        "_load_and_setup_frame_inspection",
        fake_load_and_setup,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_prepare_frame_template",
        fake_prepare_template,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_setup_frame_independent_kwargs",
        fake_setup_independent,
    )
    monkeypatch.setattr(
        ParticleStack,
        "construct_particle_movie_stack",
        fake_construct_particle_movie_stack,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_setup_frame_kwargs",
        fake_setup_frame_kwargs,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "_apply_template_dose_filter_for_frame",
        fake_apply_dose,
    )
    monkeypatch.setattr(
        FrameInspectionManager,
        "get_peak_inspection_result",
        fake_get_peak_inspection_result,
    )

    manager.run_peak_inspection_per_frame(apply_template_dose_weighting=True)
    assert dose_calls["count"] == 2


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

    refined_mips = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pos_x = np.zeros((2, 2), dtype=np.float64)
    pos_y = np.zeros((2, 2), dtype=np.float64)
    cc_of_sum = np.array([42.0, 43.0], dtype=np.float64)
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
