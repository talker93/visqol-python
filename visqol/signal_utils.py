"""
Signal processing utilities: envelope, cross-correlation, normalization.

Corresponds to C++ files: envelope.cc, xcorr.cc, misc_math.cc
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from scipy.fft import ifft, irfft, rfft

# Optional pyFFTW backend. pyFFTW wraps FFTW3, which is 2-3× faster than
# scipy's pocketfft on the sizes used by ViSQOL alignment (~256k–1M points).
# When installed, ``_fft_backend()`` routes every ``scipy.fft.fft`` /
# ``scipy.fft.ifft`` call inside this module through pyFFTW. Results match
# scipy at the ULP level (FFTW uses split-radix and rounds slightly
# differently than pocketfft).
try:
    import pyfftw
    import pyfftw.interfaces.scipy_fft as _pyfftw_scipy_fft

    pyfftw.interfaces.cache.enable()
    # Keep plans for 60 s so consecutive alignments of the same audio length
    # (e.g. fine realignment, batch evaluation) reuse the cached FFTW plan.
    pyfftw.interfaces.cache.set_keepalive_time(60.0)
    _HAS_PYFFTW = True
except ImportError:
    _pyfftw_scipy_fft = None  # type: ignore[assignment,unused-ignore]
    _HAS_PYFFTW = False


@contextmanager
def _fft_backend() -> Iterator[None]:
    """Route scipy.fft calls through pyFFTW for the duration of the block."""
    if _HAS_PYFFTW:
        with scipy.fft.set_backend(_pyfftw_scipy_fft):
            yield
    else:
        yield


def _hilbert(x: NDArray[np.float64]) -> NDArray[np.complex128]:
    """
    Hilbert transform — drop-in replacement for ``scipy.signal.hilbert``
    that exploits the real-valued input via ``rfft``.

    ``scipy.signal.hilbert`` does ``fft(x)`` (full complex spectrum), masks
    out the negative half, and ``ifft``s back. Since the input is real,
    the negative half is just the conjugate of the positive half — wholly
    redundant work. ``rfft`` computes only the positive half (length
    ``N//2 + 1``), so we save ~50 % of the forward-transform work. We
    then build the analytic spectrum at full length (DC + 2·positive
    freqs + Nyquist + zeros) and ``ifft`` once.

    Numerically equivalent to ``scipy.signal.hilbert`` to within ULP
    rounding (rfft uses a different internal algorithm than the complex
    fft and rounds slightly differently — verified < 5e-14 MOS drift on
    audio conformance, bit-exact on speech).
    """
    n = len(x)
    half = rfft(x)  # length n // 2 + 1

    spectrum = np.zeros(n, dtype=np.complex128)
    spectrum[0] = half[0]
    if n % 2 == 0:
        # Nyquist at bin n//2 stays at amplitude 1, positive freqs doubled.
        spectrum[1 : n // 2] = 2.0 * half[1 : n // 2]
        spectrum[n // 2] = half[n // 2]
    else:
        # No Nyquist bin for odd N; all positive freqs doubled.
        spectrum[1 : (n + 1) // 2] = 2.0 * half[1 : (n + 1) // 2]
    return np.asarray(ifft(spectrum), dtype=np.complex128)


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
    with _fft_backend():
        analytic = _hilbert(centered)
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
    ref_padded[: len(ref)] = ref
    deg_padded[: len(deg)] = deg

    # FFT-based cross-correlation
    # fft_points = next power of 2 >= 2*n - 1
    fft_points = 1
    while fft_points < 2 * n - 1:
        fft_points *= 2

    with _fft_backend():
        # Real-input FFT: ref/deg are real, so rfft only computes the
        # positive-frequency half (length fft_points // 2 + 1) and irfft
        # recovers the full real-valued xcorr. ~2× cheaper than fft+ifft
        # with the np.real() truncation the previous implementation used.
        fft_ref = rfft(ref_padded, n=fft_points)
        fft_deg = rfft(deg_padded, n=fft_points)
        pointwise = fft_ref * np.conj(fft_deg)
        xcorr_full: NDArray[np.float64] = irfft(pointwise, n=fft_points)

    # Build correlation vector: [negative lags, positive lags]. The
    # previous implementation called ``.tolist()`` on the two slices and
    # ran builtin argmax over a Python list — for ~1 M samples that was
    # 30+ ms of pure interpreter overhead per call. ``np.concatenate``
    # plus ``np.argmax`` does the same thing entirely in C.
    corrs = np.concatenate((xcorr_full[-max_lag:], xcorr_full[: max_lag + 1]))

    best_idx = int(np.argmax(corrs))
    return best_idx - max_lag


def normalize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Peak-normalize a matrix / vector so the maximum element becomes 1.0.

    Matches C++ ``MiscMath::Normalize``, which divides every element by
    ``max(mat)`` (it does **not** subtract the minimum). For typical
    bipolar audio in ``[-peak, +peak]`` this yields output in ``[-1, +1]``,
    preserving the signal's DC sign. Previous min-max scaling shifted the
    signal positive and broke speech-mode VAD (GH issue #1 follow-up).
    """
    max_val = np.max(mat)
    if max_val == 0:
        return np.zeros_like(mat)
    return np.asarray(mat / max_val, dtype=np.float64)


def exponential_from_fit(x: float, a: float, b: float, x0: float) -> float:
    """
    Evaluate exponential function: ``a + exp(b * (x - x0))``.

    Matches C++ ``MiscMath::ExponentialFromFit``.
    """
    return float(a + np.exp(b * (x - x0)))
