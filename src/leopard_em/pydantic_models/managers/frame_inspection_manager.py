"""Pydantic model for per-frame local peak inspection."""

# pylint: disable=duplicate-code

import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import tqdm
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_motion_correction.correct_motion import correct_motion
from torch_motion_correction.deformation_field import (  # pyright: ignore[reportMissingImports]
    DeformationField,
)

from leopard_em.pydantic_models.managers.peak_inspection_manager import (
    PeakInspectionManager,
)
from leopard_em.utils.backend_setup import setup_frame_filters_particle_stack
from leopard_em.utils.ctf_utils import _setup_ctf_kwargs_from_particle_stack
from leopard_em.utils.data_io import load_template_tensor, read_particle_shifts_from_csv
from leopard_em.utils.image_processing import get_image_normalization_factor


class FrameInspectionManager(PeakInspectionManager):
    """Run peak inspection independently for each frame in a movie."""

    def _prepare_frame_template(
        self,
        template_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Load a template volume once for all frame-level inspections.

        Parameters
        ----------
        template_tensor : torch.Tensor | None, optional
            Optional in-memory template volume override. When ``None``, load from
            manager-configured template inputs.

        Returns
        -------
        torch.Tensor
            Template volume on the primary configured GPU device.
        """
        if template_tensor is None:
            template = load_template_tensor(
                template_volume=self.template_volume,
                template_volume_path=self.template_volume_path,
            )
        else:
            template = load_template_tensor(template_volume=template_tensor)
        return template.to(self.computational_config.gpu_devices[0])

    def _setup_frame_independent_kwargs(  # pylint: disable=too-many-locals
        self,
        template: torch.Tensor,
        prefer_refined_angles: bool = True,
    ) -> dict[str, Any]:
        """Build inspect-backend kwargs that remain constant across frames.

        Parameters
        ----------
        template : torch.Tensor
            Template volume used to derive static CTF/setup dimensions.
        prefer_refined_angles : bool, optional
            If ``True``, use refined Euler angles when available in particle metadata.

        Returns
        -------
        dict[str, Any]
            Backend kwargs shared by all frame calls (angles, defocus, CTF kwargs,
            per-particle correlation mean/std, device list).
        """
        device = template.device
        h, w = self.particle_stack.original_template_size
        box_h, box_w = self.particle_stack.extracted_box_size
        extracted_box_size = (box_h - h + 1, box_w - w + 1)

        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)
        euler_angle_offsets = self.orientation_refinement_config.euler_angles_offsets
        defocus_offsets = self.defocus_refinement_config.defocus_values
        pixel_size_offsets = self.pixel_size_refinement_config.pixel_size_values

        defocus_u, defocus_v = self.particle_stack.get_absolute_defocus()
        defocus_angle = torch.tensor(
            self.particle_stack["astigmatism_angle"],
            device=device,
        )

        corr_avg_images, corr_avg_indices = (
            self.particle_stack.load_images_grouped_by_column(
                column_name="correlation_average_path"
            )
        )
        corr_var_images, corr_var_indices = (
            self.particle_stack.load_images_grouped_by_column(
                column_name="correlation_variance_path"
            )
        )
        corr_mean_stack = self.particle_stack.construct_image_stack(
            images=corr_avg_images,
            indices=corr_avg_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=0.0,
        ).to(device)
        corr_std_stack = self.particle_stack.construct_image_stack(
            images=corr_var_images,
            indices=corr_var_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=1e10,
        ).to(device)
        corr_std_stack = corr_std_stack**0.5

        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            self.particle_stack,
            (template.shape[-2], template.shape[-1]),
        )

        return {
            "euler_angles": euler_angles,
            "euler_angle_offsets": euler_angle_offsets,
            "defocus_u": defocus_u.to(device),
            "defocus_v": defocus_v.to(device),
            "defocus_angle": defocus_angle,
            "defocus_offsets": defocus_offsets,
            "pixel_size_offsets": pixel_size_offsets,
            "corr_mean": corr_mean_stack,
            "corr_std": corr_std_stack,
            "ctf_kwargs": ctf_kwargs,
            "device": self.computational_config.gpu_devices,
            "mag_matrix": ctf_kwargs["mag_matrix"],
        }

    def _load_and_setup_frame_inspection(
        self,
    ) -> tuple[torch.Tensor, Any | None, torch.Tensor | None]:
        """Load movie input and resolve motion information for frame processing.

        Returns
        -------
        tuple[torch.Tensor, Any | None, torch.Tensor | None]
            ``(movie, deformation_field, particle_shifts)`` where movie is on the
            primary GPU, and either deformation field or particle shifts may be set.
        """
        if not self.movie_config.enabled:
            raise ValueError("Per-frame peak inspection requires movie_config.enabled.")
        if not self.movie_config.movie_path:
            raise ValueError(
                "Per-frame peak inspection requires movie_config.movie_path."
            )

        device = self.computational_config.gpu_devices[0]
        movie = self.movie_config.movie
        if movie is None:
            raise ValueError("Per-frame peak inspection requires a loaded movie.")
        movie = movie.to(device)

        particle_shifts = None
        deformation_field = None
        if self.movie_config.particle_shifts_path:
            particle_shifts = read_particle_shifts_from_csv(
                csv_path=self.movie_config.particle_shifts_path,
                num_frames=movie.shape[0],
                num_particles=self.particle_stack.num_particles,
            ).to(device)
        elif self.movie_config.deformation_field_path:
            deformation_field_tensor = self.movie_config.deformation_field
            if deformation_field_tensor is not None:
                deformation_field_data = deformation_field_tensor.to(device)
                deformation_field = DeformationField(
                    data=deformation_field_data,
                    grid_type="catmull_rom",
                )

        return movie, deformation_field, particle_shifts

    def _build_summed_particle_stack_from_movie(
        self,
        movie: torch.Tensor,
        deformation_field: Any | None,
        particle_shifts: torch.Tensor | None,
    ) -> torch.Tensor:
        """Accumulate per-frame particle crops into one summed particle stack.

        Parameters
        ----------
        movie : torch.Tensor
            Movie tensor with leading frame dimension.
        deformation_field : Any | None
            Optional deformation-field object used to derive per-frame shifts.
        particle_shifts : torch.Tensor | None
            Optional explicit per-frame/per-particle shifts; takes precedence over
            deformation-derived shifts when provided.

        Returns
        -------
        torch.Tensor
            Summed particle image stack with shape ``(N, H_box, W_box)``.
        """
        num_frames = movie.shape[0]
        summed_particle_images = None
        for frame_idx in range(num_frames):
            normalized_t_values = torch.tensor(
                [0.0 if num_frames == 1 else frame_idx / (num_frames - 1)],
                device=movie.device,
            )
            frame_movie = movie[frame_idx : frame_idx + 1]
            frame_particle_shifts = None
            if particle_shifts is not None:
                frame_particle_shifts = particle_shifts[frame_idx : frame_idx + 1]
            frame_particle_stack = self.particle_stack.construct_particle_movie_stack(
                movie=frame_movie,
                deformation_field=deformation_field,
                particle_shifts=frame_particle_shifts,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="reflect",
                padding_value=0.0,
                use_gradient_checkpointing=False,
                normalized_t_values=normalized_t_values,
            )[0]
            if summed_particle_images is None:
                summed_particle_images = frame_particle_stack
            else:
                summed_particle_images = summed_particle_images + frame_particle_stack

        if summed_particle_images is None:
            raise ValueError("Movie has no frames to sum for particle stack.")
        return summed_particle_images

    def _apply_template_dose_filter_for_summed_movie(
        self,
        template: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply cumulative dose weighting over the full movie interval.

        Parameters
        ----------
        template : torch.Tensor
            Non-dose-weighted template volume.
        num_frames : int
            Number of movie frames contributing to the summed particle stack.

        Returns
        -------
        torch.Tensor
            Dose-weighted template in real space.
        """
        pixel_size = float(self.particle_stack.get_pixel_size()[0].item())
        start_exposure = self.movie_config.pre_exposure
        end_exposure = start_exposure + num_frames * self.movie_config.fluence_per_frame
        template_rfft = torch.fft.rfftn(  # pylint: disable=not-callable
            template, dim=(-3, -2, -1)
        )
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=template.shape,
            pixel_size=pixel_size,
            start_exposure=start_exposure,
            end_exposure=end_exposure,
            crit_exposure_bfactor=-1,
            rfft=True,
            fftshift=False,
            device=template.device,
        )
        dose_weighted_template_rfft = template_rfft * dose_filter
        return torch.fft.irfftn(  # pylint: disable=not-callable
            dose_weighted_template_rfft,
            s=template.shape,
            dim=(-3, -2, -1),
        )

    def _peak_inspection_cc_on_summed_particle_stack(  # pylint: disable=too-many-locals
        self,
        movie: torch.Tensor,
        deformation_field: Any | None,
        particle_shifts: torch.Tensor | None,
        template: torch.Tensor,
        frame_independent_kwargs: dict[str, Any],
        correlation_batch_size: int,
        apply_projection_normalization: bool,
        apply_template_dose_weighting: bool,
        output_mode: Literal["cross_correlation", "frc"],
    ) -> np.ndarray:
        """Run one inspect pass on particle crops summed across frames.

        Uses the same fixed whitening / normalization as each per-frame pass.
        """
        if output_mode != "cross_correlation":
            raise ValueError(
                "Summed-stack inspection is implemented for cross-correlation mode."
            )
        num_frames = movie.shape[0]
        summed_stack = self._build_summed_particle_stack_from_movie(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
        )
        summed_template = (
            self._apply_template_dose_filter_for_summed_movie(template, num_frames)
            if apply_template_dose_weighting
            else template
        )
        summed_frame_kwargs = self._setup_frame_kwargs(
            frame_particle_stack=summed_stack,
            template=summed_template,
        )
        backend_kwargs = {**frame_independent_kwargs, **summed_frame_kwargs}
        summed_result = self.get_peak_inspection_result(
            backend_kwargs=backend_kwargs,
            correlation_batch_size=correlation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
            output_mode=output_mode,
        )
        if not isinstance(summed_result, torch.Tensor):
            raise TypeError("Expected tensor from peak inspection on summed stack.")
        cc_mip, _, _ = self._reduce_single_frame_cross_correlation_metrics(
            summed_result
        )
        return cc_mip

    def _setup_frame_kwargs(
        self,
        frame_particle_stack: torch.Tensor,
        template: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build backend kwargs that vary per frame (or summed stack).

        Parameters
        ----------
        frame_particle_stack : torch.Tensor
            Particle image stack for one frame (or summed across frames).
        template : torch.Tensor
            Frame-specific template (optionally dose-filtered for this frame).

        Returns
        -------
        dict[str, torch.Tensor]
            Frame-local backend kwargs containing particle/image/template DFT tensors
            and projective filters.
        """
        fixed_image_filters = getattr(self, "_frame_fixed_image_filters", None)
        fixed_projective_filters = getattr(
            self, "_frame_fixed_projective_filters", None
        )
        fixed_normalization_factor = getattr(
            self, "_frame_fixed_normalization_factor", None
        )
        full_image_shape = getattr(self, "_frame_filter_full_shape", None)
        extracted_box_shape = getattr(self, "_frame_filter_extracted_shape", None)
        particle_images_dft, template_dft, projective_filters = (
            setup_frame_filters_particle_stack(
                particle_stack=self.particle_stack,
                preprocessing_filters=self.preprocessing_filters,
                template=template,
                particle_images=frame_particle_stack.to(template.device),
                apply_global_filtering=self.apply_global_filtering,
                fixed_image_filters=fixed_image_filters,
                fixed_projective_filters=fixed_projective_filters,
                fixed_normalization_factor=fixed_normalization_factor,
                full_image_shape=full_image_shape,
                extracted_box_shape=extracted_box_shape,
            )
        )
        return {
            "particle_stack_dft": particle_images_dft,
            "template_dft": template_dft,
            "projective_filters": projective_filters,
        }

    def _setup_fixed_frame_normalization_filters(  # pylint: disable=too-many-locals
        self,
        movie: torch.Tensor,
        deformation_field: Any | None,
        particle_shifts: torch.Tensor | None,
        template: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[int, int],
        tuple[int, int],
    ]:
        """Compute fixed image/projective filters and normalization for all frames.

        Parameters
        ----------
        movie : torch.Tensor
            Input movie tensor with shape ``(T, H, W)``.
        deformation_field : Any | None
            Optional deformation-field object for motion correction.
        particle_shifts : torch.Tensor | None
            Optional explicit per-frame particle shifts.
        template : torch.Tensor
            Template volume used to derive template Fourier output shape.

        Returns
        -------
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]
        ]
            ``(fixed_image_filters, fixed_projective_filters, normalization_factor,
            full_image_shape, extracted_box_shape)`` used by per-frame preprocessing.
        """
        num_frames = movie.shape[0]
        box_h, box_w = self.particle_stack.extracted_box_size
        template_output_shape = (template.shape[-2], template.shape[-1] // 2 + 1)
        has_dataframe = hasattr(self.particle_stack, "_df")
        if self.apply_global_filtering and has_dataframe:
            # Reference micrograph for global whitening: sum frames along time.
            # - No deformation field (aligned movie or shifts-only path): raw sum.
            # - Deformation grid: motion-correct each frame like torch_motion_correction
            #   (correct_motion), then sum corrected frames.
            if deformation_field is not None and particle_shifts is None:
                pixel_spacing = float(
                    self.particle_stack.get_pixel_size().mean().item()
                )
                corrected_movie = correct_motion(
                    image=movie,
                    deformation_field=deformation_field,
                    pixel_spacing=pixel_spacing,
                    device=movie.device,
                )
                summed_movie_image = corrected_movie.sum(dim=0, keepdim=True)
            else:
                summed_movie_image = movie.sum(dim=0, keepdim=True)
            summed_movie_dft = torch.fft.rfftn(  # pylint: disable=not-callable
                summed_movie_image, dim=(-2, -1)
            )
            summed_movie_dft[..., 0, 0] = 0.0 + 0.0j
            df_index = self.particle_stack.get_dataframe_copy().index
            fixed_projective_filters = self.particle_stack.construct_projective_filters(
                self.preprocessing_filters,
                output_shape=template_output_shape,
                images_dft=summed_movie_dft.detach(),
                indices=[df_index],
            ).to(template.device)
            summed_particle_images = self.particle_stack.construct_image_stack(
                images=summed_movie_image,
                indices=[df_index],
                extraction_size=self.particle_stack.extracted_box_size,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="reflect",
                padding_value=0.0,
            )
            summed_particle_images_dft = torch.fft.rfftn(  # pylint: disable=not-callable
                summed_particle_images, dim=(-2, -1)
            )
            summed_particle_images_dft[..., 0, 0] = 0.0 + 0.0j
            particle_rfft_shape = summed_particle_images_dft.shape[-2:]
            fixed_image_filters = self.particle_stack.construct_projective_filters(
                self.preprocessing_filters,
                output_shape=particle_rfft_shape,
                images_dft=summed_movie_dft.detach(),
                indices=[df_index],
            ).to(template.device)
            bandpass_filter = (
                self.preprocessing_filters.bandpass_filter.calculate_bandpass_filter(
                    summed_particle_images_dft.shape[-2:]
                ).to(template.device)
            )
            normalization_factor = get_image_normalization_factor(
                image_rfft=summed_particle_images_dft,
                cumulative_fourier_filters=fixed_image_filters,
                bandpass_filter=bandpass_filter,
                full_image_shape=(box_h, box_w),
                extracted_box_shape=(box_h, box_w),
            )
            return (
                fixed_image_filters,
                fixed_projective_filters,
                normalization_factor,
                (box_h, box_w),
                (box_h, box_w),
            )

        summed_particle_images = None
        for frame_idx in range(num_frames):
            normalized_t_values = torch.tensor(
                [0.0 if num_frames == 1 else frame_idx / (num_frames - 1)],
                device=movie.device,
            )
            frame_movie = movie[frame_idx : frame_idx + 1]
            frame_particle_shifts = None
            if particle_shifts is not None:
                frame_particle_shifts = particle_shifts[frame_idx : frame_idx + 1]
            frame_particle_stack = self.particle_stack.construct_particle_movie_stack(
                movie=frame_movie,
                deformation_field=deformation_field,
                particle_shifts=frame_particle_shifts,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="reflect",
                padding_value=0.0,
                use_gradient_checkpointing=False,
                normalized_t_values=normalized_t_values,
            )[0]
            if summed_particle_images is None:
                summed_particle_images = frame_particle_stack
            else:
                summed_particle_images = summed_particle_images + frame_particle_stack

        if summed_particle_images is None:
            raise ValueError("Movie has no frames for fixed normalization setup.")
        summed_particle_images_dft = torch.fft.rfftn(  # pylint: disable=not-callable
            summed_particle_images, dim=(-2, -1)
        )
        summed_particle_images_dft[..., 0, 0] = 0.0 + 0.0j
        particle_rfft_shape = summed_particle_images_dft.shape[-2:]
        fixed_image_filters = self.particle_stack.construct_image_filters(
            self.preprocessing_filters,
            output_shape=particle_rfft_shape,
            images_dft=summed_particle_images_dft.detach(),
        ).to(template.device)
        fixed_projective_filters = self.particle_stack.construct_image_filters(
            self.preprocessing_filters,
            output_shape=template_output_shape,
            images_dft=summed_particle_images_dft.detach(),
        ).to(template.device)
        bandpass_filter = (
            self.preprocessing_filters.bandpass_filter.calculate_bandpass_filter(
                summed_particle_images_dft.shape[-2:]
            ).to(template.device)
        )
        normalization_factor = get_image_normalization_factor(
            image_rfft=summed_particle_images_dft,
            cumulative_fourier_filters=fixed_image_filters,
            bandpass_filter=bandpass_filter,
            full_image_shape=(box_h, box_w),
            extracted_box_shape=(box_h, box_w),
        )
        return (
            fixed_image_filters,
            fixed_projective_filters,
            normalization_factor,
            (box_h, box_w),
            (box_h, box_w),
        )

    def _apply_template_dose_filter_for_frame(
        self,
        template: torch.Tensor,
        frame_idx: int,
    ) -> torch.Tensor:
        """Apply dose weighting corresponding to one frame's exposure interval.

        Parameters
        ----------
        template : torch.Tensor
            Non-dose-weighted template volume.
        frame_idx : int
            Zero-based movie frame index.

        Returns
        -------
        torch.Tensor
            Dose-weighted template in real space for this frame interval.
        """
        pixel_size = float(self.particle_stack.get_pixel_size()[0].item())
        start_exposure = self.movie_config.pre_exposure + (
            frame_idx * self.movie_config.fluence_per_frame
        )
        end_exposure = start_exposure + self.movie_config.fluence_per_frame
        template_rfft = torch.fft.rfftn(  # pylint: disable=not-callable
            template, dim=(-3, -2, -1)
        )
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=template.shape,
            pixel_size=pixel_size,
            start_exposure=start_exposure,
            end_exposure=end_exposure,
            crit_exposure_bfactor=-1,
            rfft=True,
            fftshift=False,
            device=template.device,
        )
        dose_weighted_template_rfft = template_rfft * dose_filter
        return torch.fft.irfftn(  # pylint: disable=not-callable
            dose_weighted_template_rfft,
            s=template.shape,
            dim=(-3, -2, -1),
        )

    @staticmethod
    def _reduce_single_frame_cross_correlation_metrics(
        frame_cc: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce one-frame inspect tensor to max score and XY peak coordinates.

        Parameters
        ----------
        frame_cc : torch.Tensor
            Tensor with shape ``(N, n_px, n_defocus, n_orient, H, W)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(mip, pos_x, pos_y)`` arrays, each with shape ``(N,)``.
        """
        if frame_cc.ndim != 6:
            raise ValueError(
                "Expected one-frame cross-correlation tensor with shape "
                "(N, n_px, n_defocus, n_orient, H, W)."
            )
        n, _, _, _, h, w = frame_cc.shape
        flattened = frame_cc.reshape(n, -1)
        max_values, max_indices = torch.max(flattened, dim=-1)
        spatial_size = h * w
        local_spatial_index = max_indices % spatial_size
        pos_y = torch.div(local_spatial_index, w, rounding_mode="floor")
        pos_x = local_spatial_index % w
        return (
            max_values.cpu().numpy(),
            pos_x.cpu().numpy(),
            pos_y.cpu().numpy(),
        )

    def _write_reduced_cross_correlation_csvs(
        self,
        refined_mips: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        output_dataframe_path: str,
        cc_of_sum_mip: np.ndarray | None = None,
    ) -> None:
        """Write frame and summed CSV outputs from reduced cross-correlation data.

        Columns ``sum_frames_mip`` and optional ``cc_of_sum_mip`` compare Σ(frame
        peak CC) to peak CC on the particle stack summed across frames with the same
        preprocessing (fixed whitening filters / normalization).
        """
        num_frames = refined_mips.shape[1]
        df_refined = self.particle_stack.get_dataframe_copy()
        base_columns = {
            "particle_index": df_refined["particle_index"],
            "movie_path": self.movie_config.movie_path,
        }
        frames_df_mip = pd.DataFrame(base_columns)
        frames_df_pos_x = pd.DataFrame(base_columns)
        frames_df_pos_y = pd.DataFrame(base_columns)

        for frame_idx in range(num_frames):
            frames_df_mip[f"frame_{frame_idx}_mip"] = refined_mips[:, frame_idx]
            frames_df_pos_x[f"frame_{frame_idx}_pos_x"] = pos_x[:, frame_idx]
            frames_df_pos_y[f"frame_{frame_idx}_pos_y"] = pos_y[:, frame_idx]

        frames_df_mip["sum_frames_mip"] = np.sum(refined_mips, axis=1)
        if cc_of_sum_mip is not None:
            frames_df_mip["cc_of_sum_mip"] = cc_of_sum_mip

        base_path = os.path.splitext(output_dataframe_path)[0]
        frames_df_mip.to_csv(f"{base_path}_frames_mip.csv", index=False)
        frames_df_pos_x.to_csv(f"{base_path}_frames_pos_x.csv", index=False)
        frames_df_pos_y.to_csv(f"{base_path}_frames_pos_y.csv", index=False)

        df_refined["refined_mip"] = np.sum(refined_mips, axis=1)
        if cc_of_sum_mip is not None:
            df_refined["cc_of_sum_mip"] = cc_of_sum_mip
        df_refined.to_csv(output_dataframe_path, index=False)

    @staticmethod
    def _reduce_frame_results_to_refine_like_metrics(
        per_frame_cc: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce inspect tensors to refine-like max scores and XY positions.

        Parameters
        ----------
        per_frame_cc : torch.Tensor
            Tensor with shape ``(T, N, n_px, n_defocus, n_orient, H, W)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(refined_mips, pos_x, pos_y)`` with shape ``(N, T)`` each.
        """
        if per_frame_cc.ndim != 7:
            raise ValueError(
                "Expected cross-correlation stack with shape "
                "(T, N, n_px, n_defocus, n_orient, H, W)."
            )

        t, n, _, _, _, h, w = per_frame_cc.shape
        flattened = per_frame_cc.reshape(t, n, -1)
        max_values, max_indices = torch.max(flattened, dim=-1)

        spatial_size = h * w
        local_spatial_index = max_indices % spatial_size
        pos_y = torch.div(local_spatial_index, w, rounding_mode="floor")
        pos_x = local_spatial_index % w

        refined_mips = max_values.permute(1, 0).cpu().numpy()
        pos_x_np = pos_x.permute(1, 0).cpu().numpy()
        pos_y_np = pos_y.permute(1, 0).cpu().numpy()
        return refined_mips, pos_x_np, pos_y_np

    def process_frame_results(
        self,
        frame_results: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        output_dataframe_path: str,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
    ) -> None:
        """Reduce frame outputs and write CSV summaries.

        Parameters
        ----------
        frame_results : torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Stacked frame results from :meth:`run_peak_inspection_per_frame`.
            Cross-correlation mode expects a tensor.
        output_dataframe_path : str
            Path to the main reduced output CSV.
        output_mode : Literal["cross_correlation", "frc"], optional
            Output mode used to validate expected result format.
        """
        if output_mode != "cross_correlation":
            raise ValueError(
                "CSV processing is currently implemented for cross-correlation mode."
            )
        if not isinstance(frame_results, torch.Tensor):
            raise TypeError("Expected tensor frame results in cross-correlation mode.")

        refined_mips, pos_x, pos_y = self._reduce_frame_results_to_refine_like_metrics(
            frame_results
        )
        self._write_reduced_cross_correlation_csvs(
            refined_mips=refined_mips,
            pos_x=pos_x,
            pos_y=pos_y,
            output_dataframe_path=output_dataframe_path,
        )

    @staticmethod
    def _stack_frame_results(
        frame_results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        output_mode: Literal["cross_correlation", "frc"],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Stack per-frame backend outputs with frame as leading axis.

        Parameters
        ----------
        frame_results : list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
            Per-frame outputs returned by the inspect backend.
        output_mode : Literal["cross_correlation", "frc"]
            Determines expected type and final stacked return format.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Cross-correlation mode returns ``(T, N, ...)`` tensor; FRC mode returns
            ``(stacked_frc, frequency_bins)``.
        """
        if not frame_results:
            raise ValueError("No frame results were generated.")
        if output_mode == "cross_correlation":
            cc_results = []
            for result in frame_results:
                if not isinstance(result, torch.Tensor):
                    raise TypeError(
                        "Expected tensor results for cross-correlation mode."
                    )
                cc_results.append(result)
            return torch.stack(cc_results)

        frc_results = []
        frequency_bins = None
        for result in frame_results:
            if isinstance(result, torch.Tensor):
                raise TypeError("Expected tuple results for FRC mode.")
            frc_tensor, current_frequency_bins = result
            frc_results.append(frc_tensor)
            if frequency_bins is None:
                frequency_bins = current_frequency_bins
        if frequency_bins is None:
            raise ValueError("No FRC frequency bins were generated.")
        return torch.stack(frc_results), frequency_bins

    def run_peak_inspection_per_frame(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        correlation_batch_size: int = 32,
        prefer_refined_angles: bool = True,
        apply_projection_normalization: bool = True,
        template_tensor: torch.Tensor | None = None,
        output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
        apply_template_dose_weighting: bool = False,
        output_dataframe_path: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run peak inspection independently for every movie frame.

        Parameters
        ----------
        correlation_batch_size : int, optional
            Number of orientation offsets processed per backend batch.
        prefer_refined_angles : bool, optional
            If True, use refined Euler angles from the particle stack when available.
        apply_projection_normalization : bool, optional
            Whether to normalize each projection before scoring.
        template_tensor : torch.Tensor | None, optional
            Optional template volume override.
        output_mode : Literal["cross_correlation", "frc"], optional
            Score mode (CC maps or FRC spectra).
        apply_template_dose_weighting : bool, optional
            If True, apply cumulative dose filtering to the provided non-dose-
            weighted template separately for each frame interval.
        output_dataframe_path : str | None, optional
            If provided (CC mode), write per-frame and summed CSV summaries including
            ``sum_frames_mip`` (sum of per-frame peak CC) and ``cc_of_sum_mip`` (peak
            CC on the particle stack summed across frames; same whitening as per-frame).
        """
        movie, deformation_field, particle_shifts = (
            self._load_and_setup_frame_inspection()
        )
        template = self._prepare_frame_template(template_tensor=template_tensor)
        (
            fixed_image_filters,
            fixed_projective_filters,
            fixed_normalization_factor,
            filter_full_shape,
            filter_extracted_shape,
        ) = self._setup_fixed_frame_normalization_filters(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
            template=template,
        )
        object.__setattr__(
            self,
            "_frame_fixed_image_filters",
            fixed_image_filters,
        )
        object.__setattr__(
            self,
            "_frame_fixed_projective_filters",
            fixed_projective_filters,
        )
        object.__setattr__(
            self,
            "_frame_fixed_normalization_factor",
            fixed_normalization_factor,
        )
        object.__setattr__(self, "_frame_filter_full_shape", filter_full_shape)
        object.__setattr__(
            self,
            "_frame_filter_extracted_shape",
            filter_extracted_shape,
        )
        frame_independent_kwargs = self._setup_frame_independent_kwargs(
            template=template,
            prefer_refined_angles=prefer_refined_angles,
        )
        frame_results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        reduced_mips: list[np.ndarray] = []
        reduced_pos_x: list[np.ndarray] = []
        reduced_pos_y: list[np.ndarray] = []
        cc_of_sum_mip: np.ndarray | None = None
        num_frames = movie.shape[0]
        frame_iter = tqdm.tqdm(
            range(num_frames),
            total=num_frames,
            desc="Inspecting frames",
            unit="frame",
            dynamic_ncols=True,
        )
        try:
            for frame_idx in frame_iter:
                normalized_t_values = torch.tensor(
                    [0.0 if num_frames == 1 else frame_idx / (num_frames - 1)],
                    device=movie.device,
                )
                frame_movie = movie[frame_idx : frame_idx + 1]
                frame_particle_shifts = None
                if particle_shifts is not None:
                    frame_particle_shifts = particle_shifts[frame_idx : frame_idx + 1]
                frame_particle_stack = (
                    self.particle_stack.construct_particle_movie_stack(
                        movie=frame_movie,
                        deformation_field=deformation_field,
                        particle_shifts=frame_particle_shifts,
                        pos_reference="top-left",
                        handle_bounds="pad",
                        padding_mode="reflect",
                        padding_value=0.0,
                        use_gradient_checkpointing=False,
                        normalized_t_values=normalized_t_values,
                    )[0]
                )
                frame_template = (
                    self._apply_template_dose_filter_for_frame(template, frame_idx)
                    if apply_template_dose_weighting
                    else template
                )
                frame_kwargs = self._setup_frame_kwargs(
                    frame_particle_stack=frame_particle_stack,
                    template=frame_template,
                )
                backend_kwargs = {**frame_independent_kwargs, **frame_kwargs}
                frame_result = self.get_peak_inspection_result(
                    backend_kwargs=backend_kwargs,
                    correlation_batch_size=correlation_batch_size,
                    apply_projection_normalization=apply_projection_normalization,
                    output_mode=output_mode,
                )
                if (
                    output_mode == "cross_correlation"
                    and output_dataframe_path is not None
                ):
                    if not isinstance(frame_result, torch.Tensor):
                        raise TypeError(
                            "Expected tensor frame results in cross-correlation mode."
                        )
                    frame_mip, frame_x, frame_y = (
                        self._reduce_single_frame_cross_correlation_metrics(
                            frame_result
                        )
                    )
                    reduced_mips.append(frame_mip)
                    reduced_pos_x.append(frame_x)
                    reduced_pos_y.append(frame_y)
                else:
                    frame_results.append(frame_result)

            if output_mode == "cross_correlation" and output_dataframe_path is not None:
                cc_of_sum_mip = self._peak_inspection_cc_on_summed_particle_stack(
                    movie=movie,
                    deformation_field=deformation_field,
                    particle_shifts=particle_shifts,
                    template=template,
                    frame_independent_kwargs=frame_independent_kwargs,
                    correlation_batch_size=correlation_batch_size,
                    apply_projection_normalization=apply_projection_normalization,
                    apply_template_dose_weighting=apply_template_dose_weighting,
                    output_mode=output_mode,
                )
        finally:
            if hasattr(self, "_frame_fixed_image_filters"):
                object.__delattr__(self, "_frame_fixed_image_filters")
            if hasattr(self, "_frame_fixed_projective_filters"):
                object.__delattr__(self, "_frame_fixed_projective_filters")
            if hasattr(self, "_frame_filter_full_shape"):
                object.__delattr__(self, "_frame_filter_full_shape")
            if hasattr(self, "_frame_filter_extracted_shape"):
                object.__delattr__(self, "_frame_filter_extracted_shape")
            if hasattr(self, "_frame_fixed_normalization_factor"):
                object.__delattr__(self, "_frame_fixed_normalization_factor")

        if output_mode == "cross_correlation" and output_dataframe_path is not None:
            refined_mips = np.stack(reduced_mips, axis=1)
            pos_x = np.stack(reduced_pos_x, axis=1)
            pos_y = np.stack(reduced_pos_y, axis=1)
            self._write_reduced_cross_correlation_csvs(
                refined_mips=refined_mips,
                pos_x=pos_x,
                pos_y=pos_y,
                output_dataframe_path=output_dataframe_path,
                cc_of_sum_mip=cc_of_sum_mip,
            )
            return torch.empty(0)

        stacked_results = self._stack_frame_results(frame_results, output_mode)
        if output_dataframe_path is not None:
            self.process_frame_results(
                frame_results=stacked_results,
                output_dataframe_path=output_dataframe_path,
                output_mode=output_mode,
            )
        return stacked_results
