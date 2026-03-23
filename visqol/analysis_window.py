"""
Analysis window for spectrogram construction.

Corresponds to C++ files: analysis_window.cc/h
"""

import numpy as np


class AnalysisWindow:
    """
    Analysis window used for spectrogram frame windowing.

    Attributes:
        size: Window size in samples.
        overlap: Overlap ratio (e.g. 0.25 means 25% of window size as hop).
        window_duration: Duration of window in seconds.
    """

    def __init__(self, sample_rate: int, overlap: float = 0.25,
                 window_duration: float = 0.08):
        """
        Args:
            sample_rate: Sample rate of the audio signal.
            overlap: Overlap as a fraction of window size (used as hop = size * overlap).
            window_duration: Duration of the analysis window in seconds.
        """
        self.window_duration = window_duration
        self.overlap = overlap
        self.size = int(round(sample_rate * window_duration))
        self._hann_window = None

    @property
    def hop_size(self) -> int:
        """Hop size = window_size * overlap."""
        return int(self.size * self.overlap)

    @property
    def hann_window(self) -> np.ndarray:
        """Precomputed Hann window."""
        if self._hann_window is None:
            # Match C++ exactly: 0.5 - 0.5 * cos(2*pi*i/(size-1))
            n = self.size
            self._hann_window = 0.5 - 0.5 * np.cos(
                2.0 * np.pi * np.arange(n) / (n - 1)
            )
        return self._hann_window

    def apply_hann_window(self, frame: np.ndarray) -> np.ndarray:
        """Apply Hann window to a frame."""
        assert len(frame) == self.size
        return frame * self.hann_window
