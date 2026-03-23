"""
Signal processing utilities: envelope, cross-correlation, normalization.

Corresponds to C++ files: envelope.cc, xcorr.cc, misc_math.cc
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft


def upper_envelope(sig: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the upper envelope using Hilbert transform.

    Matches C++ ``Envelope::CalcUpperEnv`` which:

    1. Centers signal by subtracting mean
    2. Computes Hilbert transform
    3. Takes absolute value (amplitude envelope)
    4. Adds mean back
    """
    mean_val: float = float(np.mean(sig))
    centered = sig - mean_val
    analytic = scipy_signal.hilbert(centered)
    env: NDArray[np.float64] = np.abs(analytic) + mean_val
    return env


def find_best_lag(ref: NDArray[np.float64], deg: NDArray[np.float64]) -> int:
    """
    Find the lag that maximises cross-correlation between two signals.

    Returns the lag (in samples) — positive means *deg* is delayed
    relative to *ref*.

    Matches C++ ``XCorr::FindLowestLagIndex`` which uses FFT-based
    cross-correlation.
    """
    max_lag: int = max(len(ref), len(deg)) - 1

    # Pad to same length
    n = max(len(ref), len(deg))
    ref_padded = np.zeros(n)
    deg_padded = np.zeros(n)
    ref_padded[:len(ref)] = ref
    deg_padded[:len(deg)] = deg

    # FFT-based cross-correlation
    # fft_points = next power of 2 >= 2*n - 1
    fft_points = 1
    while fft_points < 2 * n - 1:
        fft_points *= 2

    fft_ref = fft(ref_padded, n=fft_points)
    fft_deg = fft(deg_padded, n=fft_points)
    pointwise = fft_ref * np.conj(fft_deg)
    xcorr_full: NDArray[np.float64] = np.real(ifft(pointwise))

    # Build correlation vector: [negative lags, positive lags]
    neg_corrs = xcorr_full[-max_lag:].tolist()
    pos_corrs = xcorr_full[:max_lag + 1].tolist()
    corrs = neg_corrs + pos_corrs

    best_idx = int(np.argmax(corrs))
    return best_idx - max_lag


def normalize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize a matrix / vector to ``[0, 1]`` range.

    Matches C++ ``MiscMath::Normalize``.
    """
    min_val = np.min(mat)
    max_val = np.max(mat)
    if max_val == min_val:
        return np.zeros_like(mat)
    return (mat - min_val) / (max_val - min_val)


def exponential_from_fit(x: float, a: float, b: float, x0: float) -> float:
    """
    Evaluate exponential function: ``a + exp(b * (x - x0))``.

    Matches C++ ``MiscMath::ExponentialFromFit``.
    """
    return float(a + np.exp(b * (x - x0)))
