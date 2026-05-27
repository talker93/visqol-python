"""
Neurogram Similarity Index Measure (NSIM).

A variant of SSIM adapted for neurogram (spectrogram) comparison.

Corresponds to C++ files:
  - neurogram_similiarity_index_measure.cc
  - convolution_2d.cc
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from visqol.numba_accel import (
    _C1,
    _C3,
    _GW,
    _measure_patch_similarity_numba,
    has_numba,
)

# 3×3 Gaussian window weights (hardcoded from C++)
GAUSSIAN_WINDOW: NDArray[np.float64] = np.array(
    [
        [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
        [0.0838251475442633, 0.619485845753726, 0.0838251475442633],
        [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
    ]
)

# Constants for NSIM calculation
INTENSITY_RANGE: float = 1.0
K1: float = 0.01
K2: float = 0.03
C1: float = (K1 * INTENSITY_RANGE) ** 2  # = 0.0001
C3: float = (K2 * INTENSITY_RANGE) ** 2 / 2.0  # = 0.00045


@dataclass
class PatchSimilarityResult:
    """Result of comparing a reference patch with a degraded patch."""

    similarity: float = 0.0
    freq_band_means: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    freq_band_stddevs: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    freq_band_deg_energy: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Timing info
    ref_patch_start_time: float = 0.0
    ref_patch_end_time: float = 0.0
    deg_patch_start_time: float = 0.0
    deg_patch_end_time: float = 0.0

    def __str__(self) -> str:
        return (
            f"PatchSimilarityResult(sim={self.similarity:.4f}, "
            f"ref=[{self.ref_patch_start_time:.3f}-{self.ref_patch_end_time:.3f}], "
            f"deg=[{self.deg_patch_start_time:.3f}-{self.deg_patch_end_time:.3f}])"
        )

    def __repr__(self) -> str:
        return (
            f"PatchSimilarityResult(similarity={self.similarity!r}, "
            f"freq_band_means=<{len(self.freq_band_means)} bands>)"
        )


def _valid_2d_conv_with_boundary(
    kernel: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    2-D convolution with boundary replication padding, then 'valid' convolution.

    Matches C++ ``Convolution2D::Valid2DConvWithBoundary`` which:

    1. Pads matrix by 1 on each side with edge replication
    2. Performs valid convolution with REVERSED kernel

    The Gaussian kernel is symmetric so reversal has no effect.
    Output has the same shape as the input matrix.
    """
    from scipy.signal import correlate2d

    # Pad matrix by 1 on each side with edge replication
    padded = np.pad(matrix, pad_width=1, mode="edge")
    result: NDArray[np.float64] = correlate2d(padded, kernel, mode="valid")
    return result


def measure_patch_similarity(
    ref_patch: NDArray[np.float64], deg_patch: NDArray[np.float64]
) -> PatchSimilarityResult:
    """
    Compute NSIM similarity between a reference and degraded patch.

    Matches C++ ``NeurogramSimiliarityIndexMeasure::MeasurePatchSimilarity``.

    When Numba is available, dispatches to the fused JIT kernel
    (:func:`visqol.numba_accel._measure_patch_similarity_numba`) — same code
    path the DP patch matcher uses, so fine realignment shares the same
    speedup. Falls back to the NumPy implementation when Numba is absent.

    Args:
        ref_patch: ``(num_bands, num_frames)`` reference spectrogram patch.
        deg_patch: ``(num_bands, num_frames)`` degraded spectrogram patch.

    Returns:
        :class:`PatchSimilarityResult` with similarity score and per-band
        statistics.
    """
    if has_numba():
        sim_mean, fb_means, fb_stddevs, fb_deg_energy = _measure_patch_similarity_numba(
            np.ascontiguousarray(ref_patch, dtype=np.float64),
            np.ascontiguousarray(deg_patch, dtype=np.float64),
            _GW,
            _C1,
            _C3,
        )
        return PatchSimilarityResult(
            similarity=float(sim_mean),
            freq_band_means=fb_means,
            freq_band_stddevs=fb_stddevs,
            freq_band_deg_energy=fb_deg_energy,
        )

    w = GAUSSIAN_WINDOW

    # Local means
    mu_r = _valid_2d_conv_with_boundary(w, ref_patch)
    mu_d = _valid_2d_conv_with_boundary(w, deg_patch)

    # Squared means
    ref_mu_sq = mu_r * mu_r
    deg_mu_sq = mu_d * mu_d
    mu_r_mu_d = mu_r * mu_d

    # Variances
    sigma_r_sq = _valid_2d_conv_with_boundary(w, ref_patch * ref_patch) - ref_mu_sq
    sigma_d_sq = _valid_2d_conv_with_boundary(w, deg_patch * deg_patch) - deg_mu_sq
    sigma_r_d = _valid_2d_conv_with_boundary(w, ref_patch * deg_patch) - mu_r_mu_d

    # Intensity (luminance) component
    intensity = (2.0 * mu_r_mu_d + C1) / (ref_mu_sq + deg_mu_sq + C1)

    # Structure component
    structure_numer = sigma_r_d + C3
    var_product = sigma_r_sq * sigma_d_sq
    # Handle negative variance (can occur with silent patches)
    structure_denom = np.where(var_product < 0, C3, np.sqrt(var_product) + C3)
    structure = structure_numer / structure_denom

    # Combined similarity map
    sim_map = intensity * structure

    # Per-frequency-band statistics
    # Note: C++ uses Armadillo's `arma::stddev(..., 0)` which is the *unbiased*
    # estimator (divides by N-1). NumPy's default `ddof=0` divides by N and was
    # the original Python behaviour; switching to `ddof=1` aligns with C++.
    freq_band_means: NDArray[np.float64] = np.mean(sim_map, axis=1)
    freq_band_stddevs: NDArray[np.float64] = np.std(sim_map, axis=1, ddof=1)
    freq_band_deg_energy: NDArray[np.float64] = np.mean(deg_patch, axis=1)

    # Overall similarity (mean of frequency band means)
    mean_freq_band_means: float = float(np.mean(freq_band_means))

    return PatchSimilarityResult(
        similarity=mean_freq_band_means,
        freq_band_means=freq_band_means,
        freq_band_stddevs=freq_band_stddevs,
        freq_band_deg_energy=freq_band_deg_energy,
    )
