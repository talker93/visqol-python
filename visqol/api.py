"""
ViSQOL public API.

Provides a simple interface for comparing audio quality.

Corresponds to C++ file: visqol_api.cc
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from visqol.audio_utils import AudioSignal
from visqol.visqol_manager import VisqolManager
from visqol.visqol_core import SimilarityResult

# Valid mode names
_VALID_MODES = frozenset({"audio", "speech"})

# Default SVR model path (bundled inside the package)
_DEFAULT_MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "model")
_DEFAULT_SVR_MODEL: str = os.path.join(_DEFAULT_MODEL_DIR, "libsvm_nu_svr_model.txt")


class VisqolApi:
    """
    Public API for ViSQOL audio quality assessment.

    Usage::

        api = VisqolApi()
        api.create(mode="audio")
        result = api.measure("reference.wav", "degraded.wav")
        print(f"MOS-LQO: {result.moslqo}")
    """

    def __init__(self) -> None:
        self._manager: VisqolManager = VisqolManager()
        self._is_created: bool = False

    def create(
        self,
        mode: str = "audio",
        model_path: Optional[str] = None,
        search_window: int = 60,
        use_unscaled_speech: bool = False,
        disable_global_alignment: bool = False,
        disable_realignment: bool = False,
    ) -> None:
        """
        Initialize ViSQOL with the specified configuration.

        Args:
            mode: ``"audio"`` for music/general audio (48 kHz, SVR model) or
                ``"speech"`` for speech signals (16 kHz, exponential fit).
            model_path: Path to SVR model file (Audio mode only).
                If *None*, uses the bundled default model.
            search_window: Search window radius (default 60).
            use_unscaled_speech: If *True*, don't scale speech MOS to 5.0.
            disable_global_alignment: Skip global alignment step.
            disable_realignment: Skip fine realignment step.

        Raises:
            ValueError: If *mode* is not ``"audio"`` or ``"speech"``.
            ValueError: If *search_window* is not positive.
            FileNotFoundError: If *model_path* is given but does not exist.
        """
        mode_lower = mode.lower()
        if mode_lower not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Must be one of {sorted(_VALID_MODES)}."
            )

        if search_window <= 0:
            raise ValueError(
                f"search_window must be a positive integer, got {search_window}."
            )

        use_speech_mode = mode_lower == "speech"

        if not use_speech_mode and model_path is None:
            model_path = _DEFAULT_SVR_MODEL

        if model_path is not None and not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SVR model file not found: {model_path}"
            )

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
            :class:`SimilarityResult` containing MOS-LQO score and detailed results.

        Raises:
            RuntimeError: If :meth:`create` has not been called.
            FileNotFoundError: If either audio file does not exist.
        """
        self._ensure_created()

        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Reference audio file not found: {ref_path}")
        if not os.path.isfile(deg_path):
            raise FileNotFoundError(f"Degraded audio file not found: {deg_path}")

        return self._manager.run(ref_path, deg_path)

    def measure_from_arrays(
        self,
        ref_array: NDArray[np.floating],
        deg_array: NDArray[np.floating],
        sample_rate: int,
    ) -> SimilarityResult:
        """
        Compare two audio signals from numpy arrays.

        Args:
            ref_array: Reference audio signal (1-D numpy array).
            deg_array: Degraded audio signal (1-D numpy array).
            sample_rate: Sample rate of both signals in Hz.

        Returns:
            :class:`SimilarityResult` containing MOS-LQO score and detailed results.

        Raises:
            RuntimeError: If :meth:`create` has not been called.
            ValueError: If arrays are empty or *sample_rate* is not positive.
            TypeError: If arrays are not numpy arrays.
        """
        self._ensure_created()

        if not isinstance(ref_array, np.ndarray):
            raise TypeError(
                f"ref_array must be a numpy array, got {type(ref_array).__name__}"
            )
        if not isinstance(deg_array, np.ndarray):
            raise TypeError(
                f"deg_array must be a numpy array, got {type(deg_array).__name__}"
            )
        if ref_array.size == 0:
            raise ValueError("ref_array must not be empty.")
        if deg_array.size == 0:
            raise ValueError("deg_array must not be empty.")
        if sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive integer, got {sample_rate}."
            )

        ref_signal = AudioSignal(ref_array, sample_rate)
        deg_signal = AudioSignal(deg_array, sample_rate)
        return self._manager.run_from_signals(ref_signal, deg_signal)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_created(self) -> None:
        """Raise if :meth:`create` has not been called."""
        if not self._is_created:
            raise RuntimeError(
                "VisqolApi must be created (call .create()) before measuring."
            )
