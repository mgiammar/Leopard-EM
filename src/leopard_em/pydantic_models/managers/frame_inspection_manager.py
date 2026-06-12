"""Pydantic model for per-frame local peak inspection."""

# pylint: disable=duplicate-code

import os
from collections.abc import Iterator
from dataclasses import dataclass
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
from leopard_em.utils.backend_setup import (
    setup_frame_filters_particle_stack,
    setup_static_particle_kwargs,
)
from leopard_em.utils.data_io import load_template_tensor, read_particle_shifts_from_csv
from leopard_em.utils.image_processing import get_image_normalization_factor


@dataclass(frozen=True)
class FixedFrameFilters:
    """Fixed whitening/normalization artifacts shared by every movie frame.

    Attributes
    ----------
    image_filters : torch.Tensor
        Fixed cumulative Fourier filters applied to each frame's particle images.
    projective_filters : torch.Tensor
        Fixed projective (template-shaped) Fourier filters.
    normalization_factor : torch.Tensor
        Fixed per-particle variance normalization factor.
    full_image_shape : tuple[int, int]
        Shape used as the "full image" reference during normalization.
    extracted_box_shape : tuple[int, int]
        Shape of the extracted particle boxes.
    """

    image_filters: torch.Tensor
    projective_filters: torch.Tensor
    normalization_factor: torch.Tensor
    full_image_shape: tuple[int, int]
    extracted_box_shape: tuple[int, int]


@dataclass(frozen=True)
class FrameInspectionContext:
    """Immutable inputs threaded through one per-frame inspection run.

    Bundles the movie/motion inputs, the shared template and fixed filters, the frame-
    independent backend kwargs, and the scoring settings so the per-frame helpers
    compose by passing one object instead of re-deriving or stashing transient state on
    the manager.

    Attributes
    ----------
    movie : torch.Tensor
        Movie tensor with leading frame dimension, on the primary GPU device.
    deformation_field : Any | None
        Optional deformation-field object used to derive per-frame shifts.
    particle_shifts : torch.Tensor | None
        Optional explicit per-frame/per-particle shifts (takes precedence over
        deformation-derived shifts).
    template : torch.Tensor
        Template volume shared by every frame (optionally dose-weighted per frame).
    fixed_filters : FixedFrameFilters
        Fixed whitening/normalization artifacts shared across frames.
    frame_independent_kwargs : dict[str, Any]
        Backend kwargs that do not vary frame to frame.
    correlation_batch_size : int
        Number of orientation offsets processed per backend batch.
    apply_projection_normalization : bool
        Whether to normalize each projection before scoring.
    apply_template_dose_weighting : bool
        Whether to dose-weight the template separately for each frame interval.
    output_mode : Literal["cross_correlation", "frc"]
        Score mode used by the inspect backend.
    """

    movie: torch.Tensor
    deformation_field: Any | None
    particle_shifts: torch.Tensor | None
    template: torch.Tensor
    fixed_filters: FixedFrameFilters
    frame_independent_kwargs: dict[str, Any]
    correlation_batch_size: int
    apply_projection_normalization: bool
    apply_template_dose_weighting: bool
    output_mode: Literal["cross_correlation", "frc"]


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

    def _setup_frame_independent_kwargs(
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
            Backend kwargs shared by all frame calls.
        """
        return setup_static_particle_kwargs(
            particle_stack=self.particle_stack,
            template=template,
            euler_angles=self.particle_stack.get_euler_angles(prefer_refined_angles),
            euler_angle_offsets=self.orientation_refinement_config.euler_angles_offsets,
            defocus_offsets=self.defocus_refinement_config.defocus_values,
            pixel_size_offsets=self.pixel_size_refinement_config.pixel_size_values,
            device_list=self.computational_config.gpu_devices,
        )

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

    def _iter_frame_particle_stacks(
        self,
        movie: torch.Tensor,
        deformation_field: Any | None,
        particle_shifts: torch.Tensor | None,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield ``(frame_idx, particle_image_stack)`` for each movie frame.

        Parameters
        ----------
        movie : torch.Tensor
            Movie tensor with leading frame dimension.
        deformation_field : Any | None
            Optional deformation-field object used to derive per-frame shifts.
        particle_shifts : torch.Tensor | None
            Optional explicit per-frame/per-particle shifts; takes precedence over
            deformation-derived shifts when provided.

        Yields
        ------
        tuple[int, torch.Tensor]
            The frame index and its particle image stack ``(N, H_box, W_box)``.
        """
        num_frames = movie.shape[0]
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
            yield frame_idx, frame_particle_stack

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
        summed_particle_images = None
        for _, frame_particle_stack in self._iter_frame_particle_stacks(
            movie, deformation_field, particle_shifts
        ):
            if summed_particle_images is None:
                summed_particle_images = frame_particle_stack
            else:
                summed_particle_images = summed_particle_images + frame_particle_stack

        if summed_particle_images is None:
            raise ValueError("Movie has no frames to sum for particle stack.")
        return summed_particle_images

    def _apply_template_dose_filter(
        self,
        template: torch.Tensor,
        start_exposure: float,
        end_exposure: float,
    ) -> torch.Tensor:
        """Apply cumulative dose weighting over an exposure interval.

        Parameters
        ----------
        template : torch.Tensor
            Non-dose-weighted template volume.
        start_exposure : float
            Start of the exposure interval (electrons / Angstrom^2).
        end_exposure : float
            End of the exposure interval (electrons / Angstrom^2).

        Returns
        -------
        torch.Tensor
            Dose-weighted template in real space.
        """
        pixel_size = float(self.particle_stack.get_pixel_size()[0].item())
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

    def _peak_inspection_cc_on_summed_particle_stack(
        self,
        ctx: FrameInspectionContext,
    ) -> torch.Tensor:
        """Run one inspect pass on particle crops summed across frames.

        Parameters
        ----------
        ctx : FrameInspectionContext
            Shared inputs and settings for this inspection run.

        Returns
        -------
        torch.Tensor
            Peak cross-correlation per particle on the summed stack, shape ``(N,)``,
            as a CPU tensor.
        """
        if ctx.output_mode != "cross_correlation":
            raise ValueError(
                "Summed-stack inspection is implemented for cross-correlation mode."
            )
        num_frames = ctx.movie.shape[0]
        summed_stack = self._build_summed_particle_stack_from_movie(
            movie=ctx.movie,
            deformation_field=ctx.deformation_field,
            particle_shifts=ctx.particle_shifts,
        )
        start_exposure = self.movie_config.pre_exposure
        end_exposure = start_exposure + num_frames * self.movie_config.fluence_per_frame
        summed_template = (
            self._apply_template_dose_filter(ctx.template, start_exposure, end_exposure)
            if ctx.apply_template_dose_weighting
            else ctx.template
        )
        summed_result = self._inspect_particle_stack(
            ctx, frame_particle_stack=summed_stack, template=summed_template
        )
        if not isinstance(summed_result, torch.Tensor):
            raise TypeError("Expected tensor from peak inspection on summed stack.")
        cc_mip, _, _ = self._reduce_cc_peak(summed_result, n_batch_dims=1)
        return cc_mip

    def _setup_frame_kwargs(
        self,
        frame_particle_stack: torch.Tensor,
        template: torch.Tensor,
        fixed_filters: FixedFrameFilters,
    ) -> dict[str, torch.Tensor]:
        """Build backend kwargs that vary per frame (or summed stack).

        Parameters
        ----------
        frame_particle_stack : torch.Tensor
            Particle image stack for one frame (or summed across frames).
        template : torch.Tensor
            Frame-specific template (optionally dose-filtered for this frame).
        fixed_filters : FixedFrameFilters
            Fixed whitening/normalization artifacts reused across every frame.

        Returns
        -------
        dict[str, torch.Tensor]
            Frame-local backend kwargs containing particle/image/template DFT tensors
            and projective filters.
        """
        particle_images_dft, template_dft, projective_filters = (
            setup_frame_filters_particle_stack(
                particle_stack=self.particle_stack,
                preprocessing_filters=self.preprocessing_filters,
                template=template,
                particle_images=frame_particle_stack.to(template.device),
                apply_global_filtering=self.apply_global_filtering,
                fixed_image_filters=fixed_filters.image_filters,
                fixed_projective_filters=fixed_filters.projective_filters,
                fixed_normalization_factor=fixed_filters.normalization_factor,
                full_image_shape=fixed_filters.full_image_shape,
                extracted_box_shape=fixed_filters.extracted_box_shape,
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
    ) -> FixedFrameFilters:
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
        FixedFrameFilters
            Fixed image/projective filters, normalization factor, and the full-image /
            extracted-box shapes used by per-frame preprocessing.
        """
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
            summed_particle_images_dft = torch.fft.rfftn(
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
            return FixedFrameFilters(
                image_filters=fixed_image_filters,
                projective_filters=fixed_projective_filters,
                normalization_factor=normalization_factor,
                full_image_shape=(box_h, box_w),
                extracted_box_shape=(box_h, box_w),
            )

        summed_particle_images = self._build_summed_particle_stack_from_movie(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
        )
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
        return FixedFrameFilters(
            image_filters=fixed_image_filters,
            projective_filters=fixed_projective_filters,
            normalization_factor=normalization_factor,
            full_image_shape=(box_h, box_w),
            extracted_box_shape=(box_h, box_w),
        )

    @staticmethod
    def _reduce_cc_peak(
        cc: torch.Tensor, n_batch_dims: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce a cross-correlation tensor to peak score and XY peak coordinates.

        Parameters
        ----------
        cc : torch.Tensor
            Cross-correlation tensor ``(*batch, n_px, n_defocus, n_orient, H, W)``
            with ``n_batch_dims`` leading batch dims (1 for a single frame ``(N, ...)``;
            2 for a stacked ``(T, N, ...)`` tensor).
        n_batch_dims : int
            Number of leading batch dims to preserve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(mip, pos_x, pos_y)`` CPU tensors, each with shape equal to the leading
            batch dims of ``cc``.
        """
        expected_ndim = n_batch_dims + 5
        if cc.ndim != expected_ndim:
            raise ValueError(
                f"Expected cross-correlation tensor with {expected_ndim} dims "
                f"({n_batch_dims} batch + n_px, n_defocus, n_orient, H, W); "
                f"got {cc.ndim}."
            )
        h, w = cc.shape[-2], cc.shape[-1]
        batch_shape = cc.shape[:n_batch_dims]
        flattened = cc.reshape(*batch_shape, -1)
        max_values, max_indices = torch.max(flattened, dim=-1)
        spatial_size = h * w
        local_spatial_index = max_indices % spatial_size
        pos_y = torch.div(local_spatial_index, w, rounding_mode="floor")
        pos_x = local_spatial_index % w
        return (
            max_values.cpu(),
            pos_x.cpu(),
            pos_y.cpu(),
        )

    def _write_reduced_cross_correlation_csvs(
        self,
        refined_mips: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        output_dataframe_path: str,
        cc_of_sum_mip: torch.Tensor | None = None,
    ) -> None:
        """Write frame and summed CSV outputs from reduced cross-correlation data.

        Inputs are ``(N, T)`` CPU tensors; they are converted to NumPy here only
        because that is the boundary where data is handed to pandas for CSV output.

        Columns ``sum_frames_mip`` and optional ``cc_of_sum_mip`` compare Σ(frame
        peak CC) to peak CC on the particle stack summed across frames with the same
        preprocessing (fixed whitening filters / normalization).
        """
        refined_mips_np = refined_mips.cpu().numpy()
        pos_x_np = pos_x.cpu().numpy()
        pos_y_np = pos_y.cpu().numpy()
        cc_of_sum_mip_np = (
            cc_of_sum_mip.cpu().numpy() if cc_of_sum_mip is not None else None
        )
        num_frames = refined_mips_np.shape[1]
        df_refined = self.particle_stack.get_dataframe_copy()
        base_columns = {
            "particle_index": df_refined["particle_index"],
            "movie_path": self.movie_config.movie_path,
        }
        frames_df_mip = pd.DataFrame(base_columns)
        frames_df_pos_x = pd.DataFrame(base_columns)
        frames_df_pos_y = pd.DataFrame(base_columns)

        for frame_idx in range(num_frames):
            frames_df_mip[f"frame_{frame_idx}_mip"] = refined_mips_np[:, frame_idx]
            frames_df_pos_x[f"frame_{frame_idx}_pos_x"] = pos_x_np[:, frame_idx]
            frames_df_pos_y[f"frame_{frame_idx}_pos_y"] = pos_y_np[:, frame_idx]

        frames_df_mip["sum_frames_mip"] = np.sum(refined_mips_np, axis=1)
        if cc_of_sum_mip_np is not None:
            frames_df_mip["cc_of_sum_mip"] = cc_of_sum_mip_np

        base_path = os.path.splitext(output_dataframe_path)[0]
        frames_df_mip.to_csv(f"{base_path}_frames_mip.csv", index=False)
        frames_df_pos_x.to_csv(f"{base_path}_frames_pos_x.csv", index=False)
        frames_df_pos_y.to_csv(f"{base_path}_frames_pos_y.csv", index=False)

        df_refined["refined_mip"] = np.sum(refined_mips_np, axis=1)
        if cc_of_sum_mip_np is not None:
            df_refined["cc_of_sum_mip"] = cc_of_sum_mip_np
        df_refined.to_csv(output_dataframe_path, index=False)

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

        # frame_results is (T, N, n_px, n_defocus, n_orient, H, W); reduce over the
        # two leading (T, N) batch dims, then transpose to (N, T) for CSV columns.
        mip_tn, pos_x_tn, pos_y_tn = self._reduce_cc_peak(frame_results, n_batch_dims=2)
        self._write_reduced_cross_correlation_csvs(
            refined_mips=mip_tn.transpose(0, 1),
            pos_x=pos_x_tn.transpose(0, 1),
            pos_y=pos_y_tn.transpose(0, 1),
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

    def _frame_dose_template(
        self,
        template: torch.Tensor,
        frame_idx: int,
        apply_template_dose_weighting: bool,
    ) -> torch.Tensor:
        """Return the template to use for a single frame (cumulative dose filter).

        Parameters
        ----------
        template : torch.Tensor
            Shared non-dose-weighted template volume.
        frame_idx : int
            Index of the frame being processed.
        apply_template_dose_weighting : bool
            Whether to dose-weight the template for this frame's exposure interval.

        Returns
        -------
        torch.Tensor
            Frame-specific template.
        """
        if not apply_template_dose_weighting:
            return template
        start_exposure = self.movie_config.pre_exposure + (
            frame_idx * self.movie_config.fluence_per_frame
        )
        end_exposure = start_exposure + self.movie_config.fluence_per_frame
        return self._apply_template_dose_filter(template, start_exposure, end_exposure)

    def _inspect_particle_stack(
        self,
        ctx: FrameInspectionContext,
        frame_particle_stack: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the inspect backend on one particle stack (single frame or summed).

        Parameters
        ----------
        ctx : FrameInspectionContext
            Shared inputs and settings for this inspection run.
        frame_particle_stack : torch.Tensor
            Particle image stack to score.
        template : torch.Tensor
            Template to score against (optionally dose-weighted).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Inspect backend output (CC tensor or FRC tuple).
        """
        frame_kwargs = self._setup_frame_kwargs(
            frame_particle_stack=frame_particle_stack,
            template=template,
            fixed_filters=ctx.fixed_filters,
        )
        backend_kwargs = {**ctx.frame_independent_kwargs, **frame_kwargs}
        return self.get_peak_inspection_result(
            backend_kwargs=backend_kwargs,
            correlation_batch_size=ctx.correlation_batch_size,
            apply_projection_normalization=ctx.apply_projection_normalization,
            output_mode=ctx.output_mode,
        )

    def _iter_frame_inspection_results(
        self,
        ctx: FrameInspectionContext,
    ) -> Iterator[tuple[int, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]]:
        """Yield ``(frame_idx, inspect_result)`` for each movie frame.

        Parameters
        ----------
        ctx : FrameInspectionContext
            Shared inputs and settings for this inspection run.

        Yields
        ------
        tuple[int, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
            The frame index and its inspect backend output.
        """
        num_frames = ctx.movie.shape[0]
        frame_iter = tqdm.tqdm(
            self._iter_frame_particle_stacks(
                ctx.movie, ctx.deformation_field, ctx.particle_shifts
            ),
            total=num_frames,
            desc="Inspecting frames",
            unit="frame",
            dynamic_ncols=True,
        )
        for frame_idx, frame_particle_stack in frame_iter:
            frame_template = self._frame_dose_template(
                ctx.template, frame_idx, ctx.apply_template_dose_weighting
            )
            yield (
                frame_idx,
                self._inspect_particle_stack(
                    ctx,
                    frame_particle_stack=frame_particle_stack,
                    template=frame_template,
                ),
            )

    def _run_per_frame_cross_correlation_csv(
        self,
        ctx: FrameInspectionContext,
        output_dataframe_path: str,
    ) -> None:
        """Score every frame, reducing to peaks, and write CSV summaries.

        Reduces each frame's cross-correlation to per-particle peaks as it is
        produced (so full CC tensors are not all held in memory at once), then adds
        the ``cc_of_sum_mip`` reference computed on the summed particle stack.

        Parameters
        ----------
        ctx : FrameInspectionContext
            Shared inputs and settings for this inspection run.
        output_dataframe_path : str
            Destination path for the reduced CSV outputs.
        """
        reduced_mips: list[torch.Tensor] = []
        reduced_pos_x: list[torch.Tensor] = []
        reduced_pos_y: list[torch.Tensor] = []
        for _, frame_result in self._iter_frame_inspection_results(ctx):
            if not isinstance(frame_result, torch.Tensor):
                raise TypeError(
                    "Expected tensor frame results in cross-correlation mode."
                )
            frame_mip, frame_x, frame_y = self._reduce_cc_peak(
                frame_result, n_batch_dims=1
            )
            reduced_mips.append(frame_mip)
            reduced_pos_x.append(frame_x)
            reduced_pos_y.append(frame_y)

        cc_of_sum_mip = self._peak_inspection_cc_on_summed_particle_stack(ctx)
        self._write_reduced_cross_correlation_csvs(
            refined_mips=torch.stack(reduced_mips, dim=1),
            pos_x=torch.stack(reduced_pos_x, dim=1),
            pos_y=torch.stack(reduced_pos_y, dim=1),
            output_dataframe_path=output_dataframe_path,
            cc_of_sum_mip=cc_of_sum_mip,
        )

    def _run_per_frame_collect(
        self,
        ctx: FrameInspectionContext,
        output_dataframe_path: str | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score every frame and return the stacked results.

        Parameters
        ----------
        ctx : FrameInspectionContext
            Shared inputs and settings for this inspection run.
        output_dataframe_path : str | None
            If provided, also write reduced CSV summaries from the stacked results.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Stacked per-frame results (``(T, N, ...)`` CC tensor or FRC tuple).
        """
        frame_results = [
            result for _, result in self._iter_frame_inspection_results(ctx)
        ]
        stacked_results = self._stack_frame_results(frame_results, ctx.output_mode)
        if output_dataframe_path is not None:
            self.process_frame_results(
                frame_results=stacked_results,
                output_dataframe_path=output_dataframe_path,
                output_mode=ctx.output_mode,
            )
        return stacked_results

    def run_peak_inspection_per_frame(
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

        Composes the per-frame pipeline: load the movie/motion inputs, prepare the
        shared template and fixed whitening filters, build the frame-independent
        backend kwargs, then dispatch to the streaming-CSV or result-collecting path.

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

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Stacked per-frame results, or an empty tensor when results were streamed
            directly to CSV (CC mode with ``output_dataframe_path``).
        """
        movie, deformation_field, particle_shifts = (
            self._load_and_setup_frame_inspection()
        )
        template = self._prepare_frame_template(template_tensor=template_tensor)
        fixed_filters = self._setup_fixed_frame_normalization_filters(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
            template=template,
        )
        frame_independent_kwargs = self._setup_frame_independent_kwargs(
            template=template,
            prefer_refined_angles=prefer_refined_angles,
        )
        ctx = FrameInspectionContext(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
            template=template,
            fixed_filters=fixed_filters,
            frame_independent_kwargs=frame_independent_kwargs,
            correlation_batch_size=correlation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
            apply_template_dose_weighting=apply_template_dose_weighting,
            output_mode=output_mode,
        )

        if output_mode == "cross_correlation" and output_dataframe_path is not None:
            self._run_per_frame_cross_correlation_csv(ctx, output_dataframe_path)
            return torch.empty(0)

        return self._run_per_frame_collect(ctx, output_dataframe_path)
