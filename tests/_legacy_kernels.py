"""Frozen 3.7.0 kernels used as independent regression/benchmark oracles.

Source: talker93/visqol-python, commit 1c3953ec4b6ed2e7fa2a2ce56eab3bad648d98d4.
Apache-2.0, as in the repository LICENSE. Do not optimize these reference loops.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from visqol.numba_accel import _measure_patch_similarity_numba


@njit(cache=True)
def legacy_dp_forward_pass(
    ref_patches: NDArray[np.float64],  # (num_patches, num_bands, num_frames)
    deg_patches: NDArray[np.float64],  # (num_deg_frames, num_bands, num_frames)
    ref_patch_indices: NDArray[np.int64],  # (num_patches,)
    num_patches: int,
    num_frames_in_deg: int,
    search_window: int,
    gw: NDArray[np.float64],
    c1: float,
    c3: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    DP forward pass — fills cumulative similarity table and backtrace table.

    Returns ``(cumulative_dp, backtrace)`` both of shape
    ``(num_ref_patches, num_frames_in_deg)``.
    """
    num_ref_patches = ref_patch_indices.shape[0]

    cumulative_dp = np.zeros((num_ref_patches, num_frames_in_deg), dtype=np.float64)
    backtrace = np.full((num_ref_patches, num_frames_in_deg), -1, dtype=np.int64)

    for patch_index in range(num_patches):
        ref_frame_index = ref_patch_indices[patch_index]

        low = max(0, ref_frame_index - search_window)
        high = min(num_frames_in_deg - 1, ref_frame_index + search_window)

        for slide_offset in range(low, high + 1):
            if slide_offset >= num_frames_in_deg:
                break

            # Compute NSIM for this (ref_patch, deg_patch) pair
            sim_val_tuple = _measure_patch_similarity_numba(
                ref_patches[patch_index],
                deg_patches[slide_offset],
                gw,
                c1,
                c3,
            )
            sim_val = sim_val_tuple[0]

            past_slide_offset = -1
            highest_sim = -1e308  # ~ -inf

            if patch_index > 0:
                lower_limit = max(
                    0,
                    ref_patch_indices[patch_index - 1] - search_window,
                )

                back_offset = slide_offset - 1
                while back_offset >= lower_limit:
                    if cumulative_dp[patch_index - 1, back_offset] > highest_sim:
                        highest_sim = cumulative_dp[patch_index - 1, back_offset]
                        past_slide_offset = back_offset
                    back_offset -= 1

                sim_val += highest_sim

                if cumulative_dp[patch_index - 1, slide_offset] > sim_val:
                    sim_val = cumulative_dp[patch_index - 1, slide_offset]
                    past_slide_offset = slide_offset

            cumulative_dp[patch_index, slide_offset] = sim_val
            backtrace[patch_index, slide_offset] = past_slide_offset

    return cumulative_dp, backtrace


@njit(cache=True, parallel=True)
def legacy_gammatone_spectrogram(
    sig: NDArray[np.float64],
    hann_window: NDArray[np.float64],
    b_stages: NDArray[np.float64],
    a_denom: NDArray[np.float64],
    window_size: int,
    hop_size: int,
    num_bands: int,
    num_cols: int,
) -> NDArray[np.float64]:
    """
    Build a full Gammatone spectrogram using Numba-accelerated IIR filtering.

    Processes all frames **in parallel** (``prange`` over frames).  Each
    frame's IIR state is independent (reset to zero), so parallelism is
    bit-safe.  ``fastmath`` enables FMA and reassociation at the LLVM
    level (ULP-level deviation only, < 1e-14 relative error).

    Parameters
    ----------
    sig : (n_total_samples,) float64
        Raw audio signal.
    hann_window : (window_size,) float64
        Pre-computed Hann window.
    b_stages : (4, num_bands, 3) float64
        Numerator coefficients for each IIR stage.
    a_denom : (num_bands, 3) float64
        Denominator coefficients.
    window_size : int
        Analysis window size in samples.
    hop_size : int
        Hop size in samples.
    num_bands : int
        Number of frequency bands.
    num_cols : int
        Number of output frames.

    Returns
    -------
    out_matrix : (num_bands, num_cols) float64
        RMS-energy spectrogram.
    """
    out_matrix = np.zeros((num_bands, num_cols), dtype=np.float64)

    # prange parallelises over frames — each frame's IIR filter state is
    # completely independent (zi = 0), so this is embarrassingly parallel.
    for i in prange(num_cols):
        start = i * hop_size
        # Apply Hann window to frame
        frame = np.empty(window_size, dtype=np.float64)
        for j in range(window_size):
            frame[j] = sig[start + j] * hann_window[j]

        # Apply 4-stage IIR filter for all bands (inlined to avoid
        # cross-thread function-call overhead in the parallel region)
        n_samples = window_size
        num_b = b_stages.shape[1]  # == num_bands

        # Allocate per-thread filtered buffer
        filtered = np.empty((num_b, n_samples), dtype=np.float64)

        for chan in range(num_b):
            a1 = a_denom[chan, 1]
            a2 = a_denom[chan, 2]

            # Stage 1
            b0 = b_stages[0, chan, 0]
            b1 = b_stages[0, chan, 1]
            b2 = b_stages[0, chan, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(n_samples):
                x = frame[k]
                y = b0 * x + z0
                z0 = b1 * x - a1 * y + z1
                z1 = b2 * x - a2 * y
                filtered[chan, k] = y

            # Stages 2, 3, 4
            for stage in range(1, 4):
                b0 = b_stages[stage, chan, 0]
                b1 = b_stages[stage, chan, 1]
                b2 = b_stages[stage, chan, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(n_samples):
                    x = filtered[chan, k]
                    y = b0 * x + z0
                    z0 = b1 * x - a1 * y + z1
                    z1 = b2 * x - a2 * y
                    filtered[chan, k] = y

        # RMS per band: sqrt(mean(filtered²))
        for b in range(num_b):
            s = 0.0
            for j in range(n_samples):
                v = filtered[b, j]
                s += v * v
            out_matrix[b, i] = np.sqrt(s / n_samples)

    return out_matrix
