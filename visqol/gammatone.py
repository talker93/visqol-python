"""
Gammatone filterbank, ERB coefficient computation, and spectrogram builder.

Corresponds to C++ files:
  - equivalent_rectangular_bandwidth.cc
  - gammatone_filterbank.cc
  - gammatone_spectrogram_builder.cc
  - signal_filter.cc
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import AudioSignal
from visqol.numba_accel import (
    _gammatone_spectrogram_numba,
    has_numba,
)

logger = logging.getLogger(__name__)

# Glasberg and Moore Parameters
EAR_Q: float = 9.26449
MIN_BW: float = 24.7
ORDER: float = 1.0

# Speech mode max frequency
SPEECH_MODE_MAX_FREQ: float = 8000.0


class ErbFiltersResult:
    """Result of ERB filter coefficient computation."""

    __slots__ = ("center_freqs", "filter_coeffs")

    def __init__(
        self,
        center_freqs: NDArray[np.float64],
        filter_coeffs: NDArray[np.float64],
    ) -> None:
        """
        Args:
            center_freqs: Array of center frequencies ``(num_channels,)``.
            filter_coeffs: Coefficient matrix ``(10, num_channels)``.
                Rows: A0, A11, A12, A13, A14, A2, B0, B1, B2, gain
        """
        self.center_freqs = center_freqs
        self.filter_coeffs = filter_coeffs


def calc_center_freqs(
    low_freq: float, high_freq: float, num_channels: int
) -> NDArray[np.float64]:
    """
    Compute uniformly-spaced center frequencies on ERB scale.

    Equivalent to Slaney's ``ERBSpace`` function.

    Args:
        low_freq: Lowest center frequency.
        high_freq: Highest center frequency.
        num_channels: Number of frequency channels.

    Returns:
        Array of center frequencies ``(num_channels,)``.
    """
    a = -(EAR_Q * MIN_BW)
    b = -np.log(high_freq + EAR_Q * MIN_BW)
    c = np.log(low_freq + EAR_Q * MIN_BW)
    d = high_freq + EAR_Q * MIN_BW
    e = (b + c) / num_channels

    cfs: NDArray[np.float64] = a + np.exp(np.arange(1, num_channels + 1) * e) * d
    return cfs


def make_erb_filters(
    sample_rate: int,
    num_channels: int,
    low_freq: float,
    high_freq: float,
) -> ErbFiltersResult:
    """
    Compute ERB gammatone filter coefficients.

    Python port of ``EquivalentRectangularBandwidth::MakeFilters``.

    Args:
        sample_rate: Sample rate in Hz.
        num_channels: Number of filter channels.
        low_freq: Lowest center frequency.
        high_freq: Highest center frequency.

    Returns:
        :class:`ErbFiltersResult` containing center freqs and filter coefficients.
    """
    if high_freq > sample_rate / 2.0:
        logger.warning(
            "high_freq (%.1f) >= sample_rate/2 (%.1f), falling back to sample_rate/2",
            high_freq,
            sample_rate / 2.0,
        )
        high_freq = sample_rate / 2.0

    cf = calc_center_freqs(low_freq, high_freq, num_channels)
    T: float = 1.0 / sample_rate

    # ERB bandwidth
    erb = ((cf / EAR_Q) ** ORDER + MIN_BW**ORDER) ** (1.0 / ORDER)
    B = 1.019 * 2.0 * np.pi * erb

    # Filter coefficients
    expBT = np.exp(B * T)
    B1_coeff = -2.0 * np.cos(2.0 * cf * np.pi * T) / expBT
    B2_coeff = np.exp(-2.0 * B * T)

    b1 = np.sin(2.0 * cf * np.pi * T) * T
    bPos = b1 * 2.0 * np.sqrt(3.0 + 2.0**1.5)
    bNeg = b1 * 2.0 * np.sqrt(3.0 - 2.0**1.5)
    a = np.cos(2.0 * cf * np.pi * T) * 2.0 * T

    A11 = -(a / expBT + bPos / expBT) / 2.0
    A12 = -(a / expBT - bPos / expBT) / 2.0
    A13 = -(a / expBT + bNeg / expBT) / 2.0
    A14 = -(a / expBT - bNeg / expBT) / 2.0

    # Gain calculation (complex arithmetic)
    p1 = 2.0 ** (3.0 / 2.0)
    s1 = np.sqrt(3.0 - p1)
    s2 = np.sqrt(3.0 + p1)

    # Complex exponentials
    xExp = np.exp(4.0j * cf * np.pi * T)
    x01 = -2.0 * xExp * T
    x02 = 2.0 * np.exp((-B + 2.0j * cf * np.pi) * T) * T

    xCos = np.cos(2.0 * cf * np.pi * T)
    xSin = np.sin(2.0 * cf * np.pi * T)

    x1 = x01 + x02 * (xCos - s1 * xSin)
    x2 = x01 + x02 * (xCos + s1 * xSin)
    x3 = x01 + x02 * (xCos - s2 * xSin)
    x4 = x01 + x02 * (xCos + s2 * xSin)

    x5 = -2.0 / np.exp(2.0 * B * T) - 2.0 * xExp + 2.0 * (1.0 + xExp) / np.exp(B * T)

    gain = np.abs(x1 * x2 * x3 * x4 / (x5**4))

    # Assemble coefficient matrix (10 rows × num_channels columns)
    A0 = np.full(num_channels, T)
    A2 = np.zeros(num_channels)
    B0 = np.ones(num_channels)

    filter_coeffs: NDArray[np.float64] = np.array(
        [
            A0,  # 0: A0
            A11,  # 1: A11
            A12,  # 2: A12
            A13,  # 3: A13
            A14,  # 4: A14
            A2,  # 5: A2
            B0,  # 6: B0
            B1_coeff,  # 7: B1
            B2_coeff,  # 8: B2
            gain,  # 9: gain
        ]
    )

    return ErbFiltersResult(center_freqs=cf, filter_coeffs=filter_coeffs)


class GammatoneFilterBank:
    """
    Gammatone filterbank that applies 4-stage cascaded IIR filtering.

    Each stage uses different A coefficients but the same B (denominator)
    coefficients.
    """

    def __init__(self, num_bands: int, min_freq: float) -> None:
        self.num_bands: int = num_bands
        self.min_freq: float = min_freq
        self._conditions: list[list[NDArray[np.float64]]] | None = None

    def reset_conditions(self) -> None:
        """Reset all filter conditions to zero."""
        self._conditions = [[np.zeros(2) for _ in range(self.num_bands)] for _ in range(4)]

    def apply_filter(
        self,
        signal: NDArray[np.float64],
        filter_coeffs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply 4-stage cascaded gammatone filter to signal for all bands.

        Args:
            signal: Input signal frame (1-D array).
            filter_coeffs: ``(10, num_bands)`` coefficient matrix.

        Returns:
            ``(num_bands, len(signal))`` filtered output matrix.
        """
        assert self._conditions is not None, "Call reset_conditions() first"

        nb = self.num_bands

        # Extract coefficient vectors (all bands at once)
        A0 = filter_coeffs[0]
        A11 = filter_coeffs[1]
        A12 = filter_coeffs[2]
        A13 = filter_coeffs[3]
        A14 = filter_coeffs[4]
        A2 = filter_coeffs[5]
        B0 = filter_coeffs[6]
        B1 = filter_coeffs[7]
        B2 = filter_coeffs[8]
        gain = filter_coeffs[9]

        # Pre-build numerator arrays for all 4 stages × all channels
        # Stage 1: normalize by gain
        b1 = np.column_stack([A0 / gain, A11 / gain, A2 / gain])  # (nb, 3)
        # Stage 2-4
        b2 = np.column_stack([A0, A12, A2])  # (nb, 3)
        b3 = np.column_stack([A0, A13, A2])  # (nb, 3)
        b4 = np.column_stack([A0, A14, A2])  # (nb, 3)

        # Denominator is the same for all 4 stages (per channel)
        denom = np.column_stack([B0, B1, B2])  # (nb, 3)

        # Process all channels with vectorised per-channel lfilter
        output = np.empty((nb, len(signal)), dtype=np.float64)
        stages_b = [b1, b2, b3, b4]

        for chan in range(nb):
            y = signal
            for stage_idx, sb in enumerate(stages_b):
                y, zf = lfilter(
                    sb[chan],
                    denom[chan],
                    y,
                    zi=self._conditions[stage_idx][chan],
                )
                self._conditions[stage_idx][chan] = zf
            output[chan] = y

        return output


class Spectrogram:
    """Spectrogram data container with dB conversion and noise floor processing."""

    __slots__ = ("center_freq_bands", "data")

    def __init__(
        self,
        data: NDArray[np.float64],
        center_freq_bands: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Args:
            data: ``(num_bands, num_frames)`` spectrogram matrix.
            center_freq_bands: Center frequencies for each band (low to high).
        """
        self.data: NDArray[np.float64] = np.asarray(data, dtype=np.float64)
        self.center_freq_bands: NDArray[np.float64] = (
            center_freq_bands if center_freq_bands is not None else np.array([])
        )

    @property
    def num_bands(self) -> int:
        return self.data.shape[0]

    @property
    def num_frames(self) -> int:
        return self.data.shape[1]

    def __repr__(self) -> str:
        return f"Spectrogram(bands={self.num_bands}, frames={self.num_frames})"


def convert_to_db(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert spectrogram values to decibels: ``10 * log10(|x|)``.

    Zero values are replaced with machine epsilon.
    Matches C++ ``Spectrogram::ConvertSampleToDb``.
    """
    abs_matrix = np.abs(matrix)
    abs_matrix = np.where(abs_matrix == 0, np.finfo(np.float64).eps, abs_matrix)
    return np.asarray(10.0 * np.log10(abs_matrix), dtype=np.float64)


def prepare_spectrograms_for_comparison(
    ref_spec: Spectrogram, deg_spec: Spectrogram
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Prepare reference and degraded spectrograms for comparison.

    1. Convert to dB
    2. Apply absolute noise floor (−45 dB)
    3. Apply per-frame relative noise floor (peak − 45 dB)
    4. Normalize to 0 dB global floor

    Matches C++ ``MiscAudio::PrepareSpectrogramsForComparison``.

    Returns:
        Tuple of ``(ref_db, deg_db)`` as numpy arrays.
    """
    NOISE_FLOOR_ABSOLUTE_DB: float = -45.0
    NOISE_FLOOR_RELATIVE_TO_PEAK_DB: float = 45.0

    # 1. Convert to dB
    ref_db = convert_to_db(ref_spec.data)
    deg_db = convert_to_db(deg_spec.data)

    # 2. Absolute noise floor
    ref_db = np.maximum(ref_db, NOISE_FLOOR_ABSOLUTE_DB)
    deg_db = np.maximum(deg_db, NOISE_FLOOR_ABSOLUTE_DB)

    # 3. Per-frame relative noise floor (vectorised)
    min_cols = min(ref_db.shape[1], deg_db.shape[1])
    ref_view = ref_db[:, :min_cols]
    deg_view = deg_db[:, :min_cols]
    ref_max = np.max(ref_view, axis=0)  # (min_cols,)
    deg_max = np.max(deg_view, axis=0)  # (min_cols,)
    any_max = np.maximum(ref_max, deg_max)
    floor_db = any_max - NOISE_FLOOR_RELATIVE_TO_PEAK_DB  # (min_cols,)
    ref_db[:, :min_cols] = np.maximum(ref_view, floor_db[np.newaxis, :])
    deg_db[:, :min_cols] = np.maximum(deg_view, floor_db[np.newaxis, :])

    # 4. Global normalization: subtract global minimum
    lowest = min(float(np.min(ref_db)), float(np.min(deg_db)))
    ref_db -= lowest
    deg_db -= lowest

    return ref_db, deg_db


class GammatoneSpectrogramBuilder:
    """Builds a gammatone-filtered spectrogram from an audio signal."""

    def __init__(
        self,
        num_bands: int,
        min_freq: float,
        speech_mode: bool = False,
    ) -> None:
        """
        Args:
            num_bands: Number of frequency bands.
            min_freq: Minimum center frequency.
            speech_mode: If *True*, cap max frequency at 8000 Hz.
        """
        self.filter_bank: GammatoneFilterBank = GammatoneFilterBank(num_bands, min_freq)
        self.speech_mode: bool = speech_mode

    def build(self, signal: AudioSignal, window: AnalysisWindow) -> Spectrogram:
        """
        Build a gammatone spectrogram from an audio signal.

        When ``numba`` is installed the entire frame loop is JIT-compiled,
        giving a **2-4x** speedup.  Results are identical to the
        ``scipy.lfilter`` path.

        Args:
            signal: Input audio signal.
            window: Analysis window parameters.

        Returns:
            :class:`Spectrogram` object.

        Raises:
            ValueError: If signal is too short.
        """
        sig = signal.data
        sample_rate = signal.sample_rate
        num_bands = self.filter_bank.num_bands

        max_freq = SPEECH_MODE_MAX_FREQ if self.speech_mode else sample_rate / 2.0

        # Compute ERB filter coefficients
        erb_result = make_erb_filters(
            sample_rate,
            num_bands,
            self.filter_bank.min_freq,
            max_freq,
        )
        # Flip updown (reverse row order) to match C++
        filter_coeffs = erb_result.filter_coeffs[:, ::-1]

        # Setup windowing
        hop_size = int(window.size * window.overlap)

        if len(sig) <= window.size:
            raise ValueError(
                f"Too few samples ({len(sig)}) to build spectrogram "
                f"({window.size} required minimum)."
            )

        num_cols = 1 + int(np.floor((len(sig) - window.size) / hop_size))

        if has_numba():
            out_matrix = self._build_numba(
                sig,
                window,
                filter_coeffs,
                hop_size,
                num_cols,
                num_bands,
            )
        else:
            out_matrix = self._build_scipy(
                sig,
                window,
                filter_coeffs,
                hop_size,
                num_cols,
                num_bands,
            )

        # Order center frequencies from lowest to highest (reverse the ERB order)
        ordered_cfs: NDArray[np.float64] = erb_result.center_freqs[::-1].copy()

        return Spectrogram(out_matrix, center_freq_bands=ordered_cfs)

    # ------------------------------------------------------------------
    # Numba-accelerated path
    # ------------------------------------------------------------------

    @staticmethod
    def _build_numba(
        sig: NDArray[np.float64],
        window: AnalysisWindow,
        filter_coeffs: NDArray[np.float64],
        hop_size: int,
        num_cols: int,
        num_bands: int,
    ) -> NDArray[np.float64]:
        """Build spectrogram using Numba JIT-compiled IIR filtering."""
        # Extract and pack coefficients into the (4, num_bands, 3) layout
        # expected by _gammatone_spectrogram_numba.
        A0 = filter_coeffs[0]
        A11 = filter_coeffs[1]
        A12 = filter_coeffs[2]
        A13 = filter_coeffs[3]
        A14 = filter_coeffs[4]
        A2 = filter_coeffs[5]
        B0 = filter_coeffs[6]
        B1 = filter_coeffs[7]
        B2 = filter_coeffs[8]
        gain = filter_coeffs[9]

        # Stage 1: normalize by gain
        b1 = np.column_stack([A0 / gain, A11 / gain, A2 / gain])  # (nb, 3)
        b2 = np.column_stack([A0, A12, A2])
        b3 = np.column_stack([A0, A13, A2])
        b4 = np.column_stack([A0, A14, A2])

        b_stages = np.ascontiguousarray(
            np.stack([b1, b2, b3, b4], axis=0),  # (4, nb, 3)
            dtype=np.float64,
        )

        a_denom = np.ascontiguousarray(
            np.column_stack([B0, B1, B2]),  # (nb, 3)
            dtype=np.float64,
        )

        hann = np.ascontiguousarray(window.hann_window, dtype=np.float64)
        sig_c = np.ascontiguousarray(sig, dtype=np.float64)

        result: NDArray[np.float64] = _gammatone_spectrogram_numba(
            sig_c,
            hann,
            b_stages,
            a_denom,
            window.size,
            hop_size,
            num_bands,
            num_cols,
        )
        return result

    # ------------------------------------------------------------------
    # scipy.lfilter fallback path (original implementation)
    # ------------------------------------------------------------------

    def _build_scipy(
        self,
        sig: NDArray[np.float64],
        window: AnalysisWindow,
        filter_coeffs: NDArray[np.float64],
        hop_size: int,
        num_cols: int,
        num_bands: int,
    ) -> NDArray[np.float64]:
        """Build spectrogram using scipy.lfilter (pure-Python fallback)."""
        out_matrix = np.zeros((num_bands, num_cols))

        for i in range(num_cols):
            start = i * hop_size
            frame = sig[start : start + window.size].copy()

            # Apply Hann window
            windowed_frame = window.apply_hann_window(frame)

            # Reset filter conditions for each frame
            self.filter_bank.reset_conditions()

            # Apply gammatone filter bank
            filtered = self.filter_bank.apply_filter(windowed_frame, filter_coeffs)

            # RMS per band: sqrt(mean(filtered²))
            out_matrix[:, i] = np.sqrt(np.mean(filtered**2, axis=1))

        return out_matrix
