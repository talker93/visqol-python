"""
Dynamic programming patch matching and fine alignment.

Corresponds to C++ file: comparison_patches_selector.cc (384 lines)
"""

import logging
import math
import numpy as np
from typing import List, Optional

from visqol.audio_utils import AudioSignal
from visqol.analysis_window import AnalysisWindow
from visqol.nsim import PatchSimilarityResult, measure_patch_similarity
from visqol.gammatone import (
    GammatoneSpectrogramBuilder, Spectrogram,
    prepare_spectrograms_for_comparison,
)
from visqol.alignment import align_and_truncate

logger = logging.getLogger(__name__)


def build_degraded_patch(spectrogram_data: np.ndarray,
                         window_beginning: int, window_end: int,
                         window_height: int, window_width: int) -> np.ndarray:
    """
    Build a degraded patch from spectrogram data with zero-padding for out-of-bounds.

    Matches C++ ComparisonPatchesSelector::BuildDegradedPatch.

    Args:
        spectrogram_data: (num_bands, num_frames) degraded spectrogram.
        window_beginning: Start frame index (can be negative).
        window_end: End frame index (inclusive).
        window_height: Number of frequency bands.
        window_width: Number of frames per patch.

    Returns:
        (window_height, window_width) patch matrix.
    """
    num_cols = spectrogram_data.shape[1]
    deg_patch = np.zeros((window_height, window_width))

    first_real_frame = max(0, window_beginning)
    last_real_frame = min(window_end, num_cols - 1)

    for row_idx in range(spectrogram_data.shape[0]):
        if first_real_frame <= last_real_frame:
            row_data = spectrogram_data[row_idx,
                                        first_real_frame:last_real_frame + 1].copy()
        else:
            row_data = np.array([])

        # Prepend zeros for negative start indices
        if window_beginning < 0:
            row_data = np.concatenate([np.zeros(-window_beginning), row_data])

        # Append zeros for indices beyond spectrogram
        if window_end > num_cols - 1:
            row_data = np.concatenate(
                [row_data, np.zeros(window_end - (num_cols - 1))]
            )

        # Ensure correct width
        if len(row_data) >= window_width:
            deg_patch[row_idx, :] = row_data[:window_width]
        else:
            deg_patch[row_idx, :len(row_data)] = row_data

    return deg_patch


def _calc_max_num_patches(ref_patch_indices: List[int],
                          num_frames_in_deg: int,
                          num_frames_per_patch: int) -> int:
    """
    Calculate max number of patches that fit within degraded spectrogram.
    Matches C++ ComparisonPatchesSelector::CalcMaxNumPatches.
    """
    num_patches = len(ref_patch_indices)
    if num_patches > 0:
        while (num_patches > 0 and
               ref_patch_indices[num_patches - 1] -
               math.floor(num_frames_per_patch / 2) > num_frames_in_deg):
            num_patches -= 1
    return num_patches


def find_most_optimal_deg_patches(
    ref_patches: List[np.ndarray],
    ref_patch_indices: List[int],
    spectrogram_data: np.ndarray,
    frame_duration: float,
    search_window_radius: int = 60,
) -> List[PatchSimilarityResult]:
    """
    Find the most optimal degraded patches using dynamic programming.

    Matches C++ ComparisonPatchesSelector::FindMostOptimalDegPatches.

    Args:
        ref_patches: List of reference patch matrices.
        ref_patch_indices: Start frame indices for reference patches.
        spectrogram_data: Full degraded spectrogram.
        frame_duration: Duration of one frame in seconds.
        search_window_radius: Search window radius in patch units.

    Returns:
        List of PatchSimilarityResult for each matched pair.
    """
    num_frames_per_patch = ref_patches[0].shape[1]
    num_bands = ref_patches[0].shape[0]
    num_frames_in_deg = spectrogram_data.shape[1]
    patch_duration = frame_duration * num_frames_per_patch
    search_window = search_window_radius * num_frames_per_patch

    num_patches = _calc_max_num_patches(
        ref_patch_indices, num_frames_in_deg, num_frames_per_patch
    )

    if num_patches == 0:
        raise ValueError(
            "Degraded file was too short, different, or misaligned to score "
            "any of the reference patches."
        )

    if num_patches < len(ref_patch_indices):
        logger.warning(
            "Dropping %d (of %d) reference patches due to degraded file "
            "being misaligned or too short.",
            len(ref_patch_indices) - num_patches, len(ref_patch_indices)
        )

    # Pre-build all degraded patches
    deg_patches = []
    for offset in range(num_frames_in_deg):
        patch = build_degraded_patch(
            spectrogram_data, offset,
            offset + num_frames_per_patch - 1,
            num_bands, num_frames_per_patch
        )
        deg_patches.append(patch)

    # DP tables
    cumulative_dp = np.full((len(ref_patch_indices), num_frames_in_deg),
                            0.0, dtype=np.float64)
    backtrace = np.full((len(ref_patch_indices), num_frames_in_deg),
                        -1, dtype=np.int64)

    # Forward pass
    for patch_index in range(num_patches):
        ref_frame_index = ref_patch_indices[patch_index]

        low = max(0, ref_frame_index - search_window)
        high = min(num_frames_in_deg - 1, ref_frame_index + search_window)

        for slide_offset in range(low, high + 1):
            if slide_offset >= num_frames_in_deg:
                break

            deg_patch = deg_patches[slide_offset]
            sim_result = measure_patch_similarity(
                ref_patches[patch_index], deg_patch
            )
            sim_val = sim_result.similarity

            past_slide_offset = -1
            highest_sim = -np.inf

            if patch_index > 0:
                lower_limit = max(0,
                                  ref_patch_indices[patch_index - 1] - search_window)

                # Search backwards for best previous cumulative score
                back_offset = slide_offset - 1
                while back_offset >= lower_limit:
                    if cumulative_dp[patch_index - 1, back_offset] > highest_sim:
                        highest_sim = cumulative_dp[patch_index - 1, back_offset]
                        past_slide_offset = back_offset
                    back_offset -= 1

                sim_val += highest_sim

                # Packet loss handling: check if skipping current is better
                if cumulative_dp[patch_index - 1, slide_offset] > sim_val:
                    sim_val = cumulative_dp[patch_index - 1, slide_offset]
                    past_slide_offset = slide_offset

            cumulative_dp[patch_index, slide_offset] = sim_val
            backtrace[patch_index, slide_offset] = past_slide_offset

    # Find best ending offset
    last_index = num_patches - 1
    lower_limit = max(0, ref_patch_indices[last_index] - search_window)
    upper_limit = min(num_frames_in_deg - 1,
                      ref_patch_indices[last_index] + search_window)

    max_score = -np.inf
    last_offset = lower_limit
    for slide_offset in range(lower_limit, upper_limit + 1):
        if slide_offset >= num_frames_in_deg:
            break
        if cumulative_dp[last_index, slide_offset] > max_score:
            max_score = cumulative_dp[last_index, slide_offset]
            last_offset = slide_offset

    # Backtrace to find best path
    best_deg_patches = [PatchSimilarityResult() for _ in range(num_patches)]

    for patch_index in range(num_patches - 1, -1, -1):
        ref_patch = ref_patches[patch_index]
        deg_patch = build_degraded_patch(
            spectrogram_data, last_offset,
            last_offset + num_frames_per_patch - 1,
            num_bands, num_frames_per_patch
        )

        sim_result = measure_patch_similarity(ref_patch, deg_patch)

        # Check if this was a packet loss (no match found)
        if last_offset == backtrace[patch_index, last_offset]:
            sim_result.deg_patch_start_time = 0.0
            sim_result.deg_patch_end_time = 0.0
            sim_result.similarity = 0.0
            sim_result.freq_band_means = np.zeros(num_bands)
        else:
            sim_result.deg_patch_start_time = last_offset * frame_duration
            sim_result.deg_patch_end_time = (
                sim_result.deg_patch_start_time + patch_duration
            )

        sim_result.ref_patch_start_time = (
            ref_patch_indices[patch_index] * frame_duration
        )
        sim_result.ref_patch_end_time = (
            sim_result.ref_patch_start_time + patch_duration
        )

        best_deg_patches[patch_index] = sim_result
        last_offset = backtrace[patch_index, last_offset]

    return best_deg_patches


def slice_signal(signal: AudioSignal, start_time: float,
                 end_time: float) -> AudioSignal:
    """
    Slice an audio signal by time range.
    Matches C++ ComparisonPatchesSelector::Slice.
    """
    start_index = max(0, int(start_time * signal.sample_rate))
    end_index = min(len(signal.data) - 1,
                    int(end_time * signal.sample_rate))

    sliced = signal.data[start_index:end_index].copy()

    # Add silence at end if needed
    end_time_diff = end_time * signal.sample_rate - len(signal.data)
    if end_time_diff > 0:
        sliced = np.concatenate([np.zeros(int(end_time_diff)), sliced])

    # Add silence at beginning if needed
    if start_time < 0:
        pre_silence = np.zeros(int(-start_time * signal.sample_rate))
        sliced = np.concatenate([pre_silence, sliced])

    return AudioSignal(sliced, signal.sample_rate)


def finely_align_and_recreate_patches(
    sim_results: List[PatchSimilarityResult],
    ref_signal: AudioSignal,
    deg_signal: AudioSignal,
    spect_builder: GammatoneSpectrogramBuilder,
    window: AnalysisWindow,
) -> List[PatchSimilarityResult]:
    """
    Fine-align each matched patch pair in the time domain.

    Matches C++ ComparisonPatchesSelector::FinelyAlignAndRecreatePatches.

    For each matched pair:
    1. Extract audio sub-signals
    2. Re-align at fine granularity
    3. Rebuild spectrograms
    4. Recompute NSIM
    5. Keep the better result (original or re-aligned)
    """
    realigned_results = list(sim_results)  # copy

    for i, sim_result in enumerate(sim_results):
        # Skip packet-loss patches
        if (sim_result.deg_patch_start_time == sim_result.deg_patch_end_time
                and sim_result.deg_patch_start_time == 0.0):
            continue

        # 1. Extract audio segments
        ref_audio = slice_signal(ref_signal,
                                 sim_result.ref_patch_start_time,
                                 sim_result.ref_patch_end_time)
        deg_audio = slice_signal(deg_signal,
                                 sim_result.deg_patch_start_time,
                                 sim_result.deg_patch_end_time)

        # 2. Fine alignment
        try:
            ref_aligned, deg_aligned, lag = align_and_truncate(
                ref_audio, deg_audio
            )
        except Exception:
            continue

        # Check we have enough samples
        if (len(ref_aligned.data) <= window.size or
                len(deg_aligned.data) <= window.size):
            continue

        # 3. Rebuild spectrograms
        try:
            ref_spec = spect_builder.build(ref_aligned, window)
            deg_spec = spect_builder.build(deg_aligned, window)
        except (ValueError, Exception):
            continue

        ref_db, deg_db = prepare_spectrograms_for_comparison(ref_spec, deg_spec)

        # 4. Recompute NSIM
        new_sim = measure_patch_similarity(ref_db, deg_db)

        # 5. Keep better result
        if new_sim.similarity < sim_result.similarity:
            realigned_results[i] = sim_result
        else:
            new_ref_duration = ref_aligned.duration
            new_deg_duration = deg_aligned.duration

            if lag > 0:
                new_sim.ref_patch_start_time = (
                    sim_result.ref_patch_start_time + lag
                )
                new_sim.deg_patch_start_time = sim_result.deg_patch_start_time
            else:
                new_sim.ref_patch_start_time = sim_result.ref_patch_start_time
                new_sim.deg_patch_start_time = (
                    sim_result.deg_patch_start_time - lag
                )

            new_sim.ref_patch_end_time = (
                new_sim.ref_patch_start_time + new_ref_duration
            )
            new_sim.deg_patch_end_time = (
                new_sim.deg_patch_start_time + new_deg_duration
            )
            realigned_results[i] = new_sim

    return realigned_results
