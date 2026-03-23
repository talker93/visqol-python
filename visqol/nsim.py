"""
Neurogram Similarity Index Measure (NSIM).

A variant of SSIM adapted for neurogram (spectrogram) comparison.

Corresponds to C++ files:
  - neurogram_similiarity_index_measure.cc
  - convolution_2d.cc
"""

import numpy as np
from dataclasses import dataclass, field


# 3x3 Gaussian window weights (hardcoded from C++)
GAUSSIAN_WINDOW = np.array([
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
    [0.0838251475442633, 0.619485845753726,  0.0838251475442633],
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052]
])

# Constants for NSIM calculation
INTENSITY_RANGE = 1.0
K1 = 0.01
K2 = 0.03
C1 = (K1 * INTENSITY_RANGE) ** 2  # = 0.0001
C3 = (K2 * INTENSITY_RANGE) ** 2 / 2.0  # = 0.00045


@dataclass
class PatchSimilarityResult:
    """Result of comparing a reference patch with a degraded patch."""
    similarity: float = 0.0
    freq_band_means: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_band_stddevs: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_band_deg_energy: np.ndarray = field(default_factory=lambda: np.array([]))

    # Timing info
    ref_patch_start_time: float = 0.0
    ref_patch_end_time: float = 0.0
    deg_patch_start_time: float = 0.0
    deg_patch_end_time: float = 0.0


def _valid_2d_conv_with_boundary(kernel: np.ndarray,
                                  matrix: np.ndarray) -> np.ndarray:
    """
    2D convolution with boundary replication padding, then 'valid' convolution.

    Matches C++ Convolution2D::Valid2DConvWithBoundary which:
    1. Pads matrix by 1 on each side with edge replication
    2. Performs valid convolution with REVERSED kernel

    The C++ code reverses the kernel in column-major order during convolution.
    Since our Gaussian kernel is symmetric (both row-symmetric and column-symmetric),
    the reversal has no effect. We can use scipy's fast convolution directly.

    The output has the same shape as the input matrix (edge-padded then valid conv).
    """
    from scipy.ndimage import convolve

    # Pad matrix by 1 on each side with edge replication
    padded = np.pad(matrix, pad_width=1, mode='edge')

    # The C++ reverses the kernel in column-major layout.
    # For the symmetric Gaussian kernel, this is equivalent to no reversal.
    # Use scipy.ndimage.convolve which does correlation (no kernel flip)
    # on the padded matrix, then take the valid region.

    # scipy.ndimage.convolve handles the full convolution;
    # we use mode='constant' with cval=0 since we already padded.
    # But more efficiently: just use 'valid' equivalent by slicing.
    from scipy.signal import correlate2d
    result = correlate2d(padded, kernel, mode='valid')
    return result


def measure_patch_similarity(ref_patch: np.ndarray,
                              deg_patch: np.ndarray) -> PatchSimilarityResult:
    """
    Compute NSIM similarity between a reference and degraded patch.

    Matches C++ NeurogramSimiliarityIndexMeasure::MeasurePatchSimilarity.

    Args:
        ref_patch: (num_bands, num_frames) reference spectrogram patch.
        deg_patch: (num_bands, num_frames) degraded spectrogram patch.

    Returns:
        PatchSimilarityResult with similarity score and per-band statistics.
    """
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
    freq_band_means = np.mean(sim_map, axis=1)
    freq_band_stddevs = np.std(sim_map, axis=1, ddof=0)
    freq_band_deg_energy = np.mean(deg_patch, axis=1)

    # Overall similarity (mean of frequency band means)
    mean_freq_band_means = np.mean(freq_band_means)

    return PatchSimilarityResult(
        similarity=float(mean_freq_band_means),
        freq_band_means=freq_band_means,
        freq_band_stddevs=freq_band_stddevs,
        freq_band_deg_energy=freq_band_deg_energy,
    )
