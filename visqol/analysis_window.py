"""
Analysis window for spectrogram construction.

Corresponds to C++ files: analysis_window.cc/h
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AnalysisWindow:
    """
    Analysis window used for spectrogram frame windowing.

    Attributes:
        size: Window size in samples.
        overlap: Overlap ratio (e.g. 0.25 means 25 % of window size as hop).
        window_duration: Duration of window in seconds.
    """

    def __init__(
        self,
        sample_rate: int,
        overlap: float = 0.25,
        window_duration: float = 0.08,
    ) -> None:
        """
        Args:
            sample_rate: Sample rate of the audio signal.
            overlap: Overlap as a fraction of window size
                (used as ``hop = size * overlap``).
            window_duration: Duration of the analysis window in seconds.

        Raises:
            ValueError: If *sample_rate* ≤ 0 or *overlap* not in (0, 1).
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not (0.0 < overlap < 1.0):
            raise ValueError(f"overlap must be in (0, 1), got {overlap}")

        self.window_duration: float = window_duration
        self.overlap: float = overlap
        self.size: int = round(sample_rate * window_duration)
        self._hann_window: NDArray[np.float64] | None = None

    @property
    def hop_size(self) -> int:
        """Hop size = window_size × overlap."""
        return int(self.size * self.overlap)

    @property
    def hann_window(self) -> NDArray[np.float64]:
        """Pre-computed Hann window."""
        if self._hann_window is None:
            # Match C++ exactly: 0.5 − 0.5 * cos(2π i / (size − 1))
            n = self.size
            self._hann_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
        return self._hann_window

    def apply_hann_window(self, frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Hann window to a frame.

        Raises:
            ValueError: If *frame* length does not match window size.
        """
        if len(frame) != self.size:
            raise ValueError(
                f"Frame length ({len(frame)}) does not match window size ({self.size})."
            )
        return frame * self.hann_window
