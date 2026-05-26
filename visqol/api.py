"""
ViSQOL public API.

Provides a simple interface for comparing audio quality.

Corresponds to C++ file: visqol_api.cc
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from visqol.audio_utils import AudioSignal
from visqol.visqol_core import SimilarityResult
from visqol.visqol_manager import VisqolManager

logger = logging.getLogger(__name__)

# Valid mode names
_VALID_MODES = frozenset({"audio", "speech"})

# Default SVR model path (bundled inside the package)
_DEFAULT_MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "model")
_DEFAULT_SVR_MODEL: str = os.path.join(_DEFAULT_MODEL_DIR, "libsvm_nu_svr_model.txt")

# Type alias for progress callback: (completed_count, total_count) -> None
ProgressCallback = Callable[[int, int], None]


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
        self._create_kwargs: dict[str, object] = {}

    def create(
        self,
        mode: str = "audio",
        model_path: str | None = None,
        search_window: int = 60,
        use_unscaled_speech: bool = False,
        use_lattice_model: bool | None = None,
        lattice_model_path: str | None = None,
        disable_global_alignment: bool = False,
        disable_realignment: bool = False,
    ) -> None:
        """
        Initialize ViSQOL with the specified configuration.

        Args:
            mode: ``"audio"`` for music/general audio (48 kHz, SVR model) or
                ``"speech"`` for speech signals (16 kHz).
            model_path: Path to SVR model file (Audio mode only).
                If *None*, uses the bundled default model.
            search_window: Search window radius (default 60).
            use_unscaled_speech: If *True*, don't scale polynomial speech MOS
                to 5.0. Ignored when the lattice mapper is active.
            use_lattice_model: Speech mode only. If *True*, use the deep-lattice
                TFLite mapper (matches C++ default behaviour, requires
                ``pip install visqol-python[lattice]``). If *False*, use the
                polynomial fallback. If *None* (default), auto-enable when the
                runtime is available and warn-and-fallback otherwise.
            lattice_model_path: Custom path to the lattice ``.tflite`` model.
                If *None*, uses the bundled default model.
            disable_global_alignment: Skip global alignment step.
            disable_realignment: Skip fine realignment step.

        Raises:
            ValueError: If *mode* is not ``"audio"`` or ``"speech"``.
            ValueError: If *search_window* is not positive.
            FileNotFoundError: If *model_path* or *lattice_model_path* is
                given but does not exist.
            ImportError: If ``use_lattice_model=True`` was requested but
                ``ai-edge-litert`` is not installed.
        """
        mode_lower = mode.lower()
        if mode_lower not in _VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of {sorted(_VALID_MODES)}.")

        if search_window <= 0:
            raise ValueError(f"search_window must be a positive integer, got {search_window}.")

        use_speech_mode = mode_lower == "speech"

        if not use_speech_mode and model_path is None:
            model_path = _DEFAULT_SVR_MODEL

        if model_path is not None and not os.path.isfile(model_path):
            raise FileNotFoundError(f"SVR model file not found: {model_path}")

        if lattice_model_path is not None and not os.path.isfile(lattice_model_path):
            raise FileNotFoundError(f"Lattice model file not found: {lattice_model_path}")

        # Store kwargs for batch mode (subprocess recreation)
        self._create_kwargs = {
            "model_path": model_path or "",
            "use_speech_mode": use_speech_mode,
            "use_unscaled_speech": use_unscaled_speech,
            "use_lattice_model": use_lattice_model,
            "lattice_model_path": lattice_model_path or "",
            "search_window": search_window,
            "disable_global_alignment": disable_global_alignment,
            "disable_realignment": disable_realignment,
        }

        self._manager.init(**self._create_kwargs)  # type: ignore[arg-type]
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
            raise TypeError(f"ref_array must be a numpy array, got {type(ref_array).__name__}")
        if not isinstance(deg_array, np.ndarray):
            raise TypeError(f"deg_array must be a numpy array, got {type(deg_array).__name__}")
        if ref_array.size == 0:
            raise ValueError("ref_array must not be empty.")
        if deg_array.size == 0:
            raise ValueError("deg_array must not be empty.")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer, got {sample_rate}.")

        ref_signal = AudioSignal(ref_array, sample_rate)
        deg_signal = AudioSignal(deg_array, sample_rate)
        return self._manager.run_from_signals(ref_signal, deg_signal)

    def measure_batch(
        self,
        file_pairs: Sequence[tuple[str, str]],
        *,
        max_workers: int | None = None,
        parallel: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> list[SimilarityResult | Exception]:
        """
        Evaluate multiple file pairs, optionally in parallel.

        When *parallel* is ``True`` (the default), evaluation is distributed
        across multiple **processes** via :class:`ProcessPoolExecutor`.  Each
        worker recreates its own ``VisqolManager`` from the same configuration
        so there is **zero** shared state and no GIL contention.

        Args:
            file_pairs: Sequence of ``(ref_path, deg_path)`` tuples.
            max_workers: Maximum number of worker processes.  *None* means
                ``min(len(file_pairs), os.cpu_count())``.  Ignored when
                *parallel* is ``False``.
            parallel: If ``False``, fall back to sequential evaluation
                (useful for debugging or when multiprocessing is unavailable).
            progress_callback: Optional ``(completed, total) -> None`` callback
                invoked after each pair is processed.

        Returns:
            List of :class:`SimilarityResult` (on success) or :class:`Exception`
            (on failure) for each pair, **in the same order** as *file_pairs*.

        Raises:
            RuntimeError: If :meth:`create` has not been called.
        """
        self._ensure_created()
        total = len(file_pairs)

        if not parallel or total <= 1:
            return self._measure_batch_sequential(file_pairs, progress_callback)

        return self._measure_batch_parallel(
            file_pairs,
            max_workers,
            progress_callback,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _measure_batch_sequential(
        self,
        file_pairs: Sequence[tuple[str, str]],
        progress_callback: ProgressCallback | None = None,
    ) -> list[SimilarityResult | Exception]:
        """Evaluate file pairs one by one (original behaviour)."""
        total = len(file_pairs)
        results: list[SimilarityResult | Exception] = []

        for idx, (ref_path, deg_path) in enumerate(file_pairs):
            try:
                result = self.measure(ref_path, deg_path)
                results.append(result)
            except Exception as exc:
                results.append(exc)
                logger.warning(
                    "Pair %d/%d failed: %s",
                    idx + 1,
                    total,
                    exc,
                )
            if progress_callback is not None:
                progress_callback(idx + 1, total)

        return results

    def _measure_batch_parallel(
        self,
        file_pairs: Sequence[tuple[str, str]],
        max_workers: int | None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[SimilarityResult | Exception]:
        """Evaluate file pairs in parallel using ProcessPoolExecutor."""
        total = len(file_pairs)

        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            max_workers = min(total, cpu_count)

        # We pickle the create_kwargs so each worker can build its own manager.
        create_kwargs = self._create_kwargs

        results: list[SimilarityResult | Exception] = [
            RuntimeError("not yet evaluated")
        ] * total

        completed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            # Submit all futures and track their original index.
            future_to_idx = {}
            for idx, (ref_path, deg_path) in enumerate(file_pairs):
                future = pool.submit(
                    _worker_measure,
                    create_kwargs,
                    ref_path,
                    deg_path,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = exc
                    logger.warning(
                        "Pair %d/%d failed: %s",
                        idx + 1,
                        total,
                        exc,
                    )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_created(self) -> None:
        """Raise if :meth:`create` has not been called."""
        if not self._is_created:
            raise RuntimeError("VisqolApi must be created (call .create()) before measuring.")


# =====================================================================
# Module-level worker function for multiprocessing (must be picklable)
# =====================================================================


def _worker_measure(
    create_kwargs: dict[str, object],
    ref_path: str,
    deg_path: str,
) -> SimilarityResult:
    """
    Evaluate a single file pair in a worker process.

    Each worker lazily creates its own ``VisqolManager`` on first call
    (stored in a global to reuse across tasks in the same process).
    """
    global _worker_manager

    try:
        mgr = _worker_manager
    except NameError:
        mgr = None

    if mgr is None:
        mgr = VisqolManager()
        mgr.init(**create_kwargs)  # type: ignore[arg-type]
        _worker_manager = mgr

    ref_signal = __import__("visqol.audio_utils", fromlist=["load_as_mono"]).load_as_mono(
        ref_path
    )
    deg_signal = __import__("visqol.audio_utils", fromlist=["load_as_mono"]).load_as_mono(
        deg_path
    )
    return mgr.run_from_signals(ref_signal, deg_signal)


_worker_manager: VisqolManager | None = None
