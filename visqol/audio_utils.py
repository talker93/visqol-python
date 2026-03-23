"""
Audio utilities: WAV loading, SPL calculation, mono conversion.

Corresponds to C++ files: wav_reader.cc, misc_audio.cc (partial)
"""

from __future__ import annotations

import logging
import os

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Sound pressure level reference point (20 µPa)
SPL_REFERENCE_POINT: float = 2e-5


class AudioSignal:
    """Container for audio signal data."""

    __slots__ = ("data", "sample_rate")

    def __init__(self, data: NDArray[np.floating], sample_rate: int) -> None:
        """
        Args:
            data: 1-D numpy array of audio samples (mono), float64.
            sample_rate: Sample rate in Hz.

        Raises:
            ValueError: If *sample_rate* is not positive or *data* is empty.
        """
        self.data: NDArray[np.float64] = np.asarray(data, dtype=np.float64).ravel()
        self.sample_rate: int = int(sample_rate)

        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"AudioSignal(samples={self.num_samples}, "
            f"sample_rate={self.sample_rate}, "
            f"duration={self.duration:.3f}s)"
        )

    def __str__(self) -> str:
        return f"AudioSignal({self.duration:.3f}s @ {self.sample_rate} Hz)"


def load_audio(path: str) -> tuple[NDArray[np.float64], int]:
    """
    Load a WAV file and return ``(data, sample_rate)``.

    Data is normalized to float64 range ``[-1, 1]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If *soundfile* cannot read the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    import soundfile as sf

    try:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio file '{path}': {exc}") from exc
    return data, int(sr)


def to_mono(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert multi-channel audio to mono by averaging channels."""
    if data.ndim == 2 and data.shape[1] > 1:
        return np.asarray(np.mean(data, axis=1), dtype=np.float64)
    elif data.ndim == 2:
        return data[:, 0]
    return data


def load_as_mono(path: str) -> AudioSignal:
    """Load a WAV file as mono :class:`AudioSignal`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the file cannot be read.
    """
    data, sr = load_audio(path)
    mono_data = to_mono(data)
    return AudioSignal(mono_data, sr)


def calc_sound_pressure_level(signal: AudioSignal) -> float:
    """
    Calculate sound pressure level (dB SPL).

    ``SPL = 20 * log10(rms / reference_point)``
    """
    data = signal.data
    rms: float = float(np.sqrt(np.mean(data**2)))
    if rms == 0:
        return float(-np.inf)
    return float(20.0 * np.log10(rms / SPL_REFERENCE_POINT))


def scale_to_match_sound_pressure_level(
    reference: AudioSignal, degraded: AudioSignal
) -> AudioSignal:
    """
    Scale the degraded signal to match the SPL of the reference signal.

    Returns:
        A new :class:`AudioSignal` with scaled data.
    """
    ref_spl = calc_sound_pressure_level(reference)
    deg_spl = calc_sound_pressure_level(degraded)
    scale_factor: float = 10.0 ** ((ref_spl - deg_spl) / 20.0)
    scaled_data = degraded.data * scale_factor
    return AudioSignal(scaled_data, degraded.sample_rate)
