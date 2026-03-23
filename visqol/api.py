"""
ViSQOL public API.

Provides a simple interface for comparing audio quality.

Corresponds to C++ file: visqol_api.cc
"""

import os
import numpy as np
from typing import Optional

from visqol.audio_utils import AudioSignal
from visqol.visqol_manager import VisqolManager
from visqol.visqol_core import SimilarityResult


# Default SVR model path (bundled inside the package)
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
_DEFAULT_SVR_MODEL = os.path.join(_DEFAULT_MODEL_DIR, "libsvm_nu_svr_model.txt")


class VisqolApi:
    """
    Public API for ViSQOL audio quality assessment.

    Usage:
        api = VisqolApi()
        api.create(mode="audio")
        result = api.measure("reference.wav", "degraded.wav")
        print(f"MOS-LQO: {result.moslqo}")
    """

    def __init__(self):
        self._manager = VisqolManager()
        self._is_created = False

    def create(self,
               mode: str = "audio",
               model_path: Optional[str] = None,
               search_window: int = 60,
               use_unscaled_speech: bool = False,
               disable_global_alignment: bool = False,
               disable_realignment: bool = False):
        """
        Initialize ViSQOL with the specified configuration.

        Args:
            mode: "audio" for music/general audio (48kHz, SVR model) or
                  "speech" for speech signals (16kHz, exponential fit).
            model_path: Path to SVR model file (Audio mode only).
                       If None, uses the bundled default model.
            search_window: Search window radius (default 60).
            use_unscaled_speech: If True, don't scale speech MOS to 5.0.
            disable_global_alignment: Skip global alignment step.
            disable_realignment: Skip fine realignment step.
        """
        use_speech_mode = mode.lower() == "speech"

        if not use_speech_mode and model_path is None:
            model_path = _DEFAULT_SVR_MODEL

        self._manager.init(
            model_path=model_path or "",
            use_speech_mode=use_speech_mode,
            use_unscaled_speech=use_unscaled_speech,
            search_window=search_window,
            disable_global_alignment=disable_global_alignment,
            disable_realignment=disable_realignment,
        )
        self._is_created = True

    def measure(self, ref_path: str, deg_path: str) -> SimilarityResult:
        """
        Compare two audio files and return quality assessment.

        Args:
            ref_path: Path to reference audio file (WAV).
            deg_path: Path to degraded audio file (WAV).

        Returns:
            SimilarityResult containing MOS-LQO score and detailed results.
        """
        if not self._is_created:
            raise RuntimeError(
                "VisqolApi must be created (call .create()) before measuring."
            )
        return self._manager.run(ref_path, deg_path)

    def measure_from_arrays(self, ref_array: np.ndarray,
                            deg_array: np.ndarray,
                            sample_rate: int) -> SimilarityResult:
        """
        Compare two audio signals from numpy arrays.

        Args:
            ref_array: Reference audio signal (1D numpy array).
            deg_array: Degraded audio signal (1D numpy array).
            sample_rate: Sample rate of both signals.

        Returns:
            SimilarityResult containing MOS-LQO score and detailed results.
        """
        if not self._is_created:
            raise RuntimeError(
                "VisqolApi must be created (call .create()) before measuring."
            )
        ref_signal = AudioSignal(ref_array, sample_rate)
        deg_signal = AudioSignal(deg_array, sample_rate)
        return self._manager.run_from_signals(ref_signal, deg_signal)
