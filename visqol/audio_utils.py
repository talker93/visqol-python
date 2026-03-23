"""
Audio utilities: WAV loading, SPL calculation, mono conversion.

Corresponds to C++ files: wav_reader.cc, misc_audio.cc (partial)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Sound pressure level reference point (20 µPa)
SPL_REFERENCE_POINT = 2e-5


class AudioSignal:
    """Container for audio signal data."""

    def __init__(self, data: np.ndarray, sample_rate: int):
        """
        Args:
            data: 1D numpy array of audio samples (mono), float64.
            sample_rate: Sample rate in Hz.
        """
        self.data = np.asarray(data, dtype=np.float64).ravel()
        self.sample_rate = int(sample_rate)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        return len(self.data)

    def __len__(self):
        return len(self.data)


def load_audio(path: str):
    """
    Load a WAV file and return (data, sample_rate).
    Data is normalized to float64 range [-1, 1].
    """
    import soundfile as sf
    data, sr = sf.read(path, dtype='float64', always_2d=True)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if data.ndim == 2 and data.shape[1] > 1:
        return np.mean(data, axis=1)
    elif data.ndim == 2:
        return data[:, 0]
    return data


def load_as_mono(path: str) -> AudioSignal:
    """Load a WAV file as mono AudioSignal."""
    data, sr = load_audio(path)
    mono_data = to_mono(data)
    return AudioSignal(mono_data, sr)


def calc_sound_pressure_level(signal: AudioSignal) -> float:
    """
    Calculate sound pressure level (dB SPL).
    SPL = 20 * log10(rms / reference_point)
    """
    data = signal.data
    rms = np.sqrt(np.mean(data ** 2))
    if rms == 0:
        return -np.inf
    return 20.0 * np.log10(rms / SPL_REFERENCE_POINT)


def scale_to_match_sound_pressure_level(
    reference: AudioSignal, degraded: AudioSignal
) -> AudioSignal:
    """
    Scale the degraded signal to match the SPL of the reference signal.
    Returns a new AudioSignal with scaled data.
    """
    ref_spl = calc_sound_pressure_level(reference)
    deg_spl = calc_sound_pressure_level(degraded)
    scale_factor = 10.0 ** ((ref_spl - deg_spl) / 20.0)
    scaled_data = degraded.data * scale_factor
    return AudioSignal(scaled_data, degraded.sample_rate)
