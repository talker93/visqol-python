"""
Quality mappers: SVR (Audio mode), TFLite Lattice (Speech default) and Exponential (Speech fallback).

Corresponds to C++ files:
  - svr_similarity_to_quality_mapper.cc
  - speech_similarity_to_quality_mapper.cc
  - tflite_quality_mapper.cc
  - support_vector_regression_model.cc
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def has_lattice_runtime() -> bool:
    """Return *True* if the ai-edge-litert TFLite runtime is importable."""
    try:
        import ai_edge_litert.interpreter  # noqa: F401
    except ImportError:
        return False
    return True


class SimilarityToQualityMapper(ABC):
    """Abstract base class for similarity-to-quality mapping."""

    @abstractmethod
    def init(self) -> None:
        """Initialize the mapper (e.g. load model)."""

    @abstractmethod
    def predict_quality(
        self,
        fvnsim: NDArray[np.float64],
        fvnsim10: NDArray[np.float64] | None = None,
        fstdnsim: NDArray[np.float64] | None = None,
        fvdegenergy: NDArray[np.float64] | None = None,
    ) -> float:
        """Predict MOS quality from NSIM feature vectors."""


class SvrSimilarityToQualityMapper(SimilarityToQualityMapper):
    """
    SVR-based quality mapper for Audio mode.

    Uses libsvm to load and predict from a pre-trained SVR model.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path: str = model_path
        self.model: object = None  # svm_model from libsvm

    def init(self) -> None:
        """Load the libsvm model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ImportError: If libsvm is not installed.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SVR model file not found: {self.model_path}")

        try:
            from svmutil import svm_load_model

            self.model = svm_load_model(self.model_path)
        except ImportError:
            try:
                from libsvm.svmutil import svm_load_model

                self.model = svm_load_model(self.model_path)
            except ImportError:
                raise ImportError(
                    "libsvm is required for Audio mode SVR quality mapping. "
                    "Install with: pip install libsvm-official"
                ) from None

    def predict_quality(
        self,
        fvnsim: NDArray[np.float64],
        fvnsim10: NDArray[np.float64] | None = None,
        fstdnsim: NDArray[np.float64] | None = None,
        fvdegenergy: NDArray[np.float64] | None = None,
    ) -> float:
        """Predict MOS using SVR model.  Only *fvnsim* is used as features."""
        try:
            from svmutil import svm_predict
        except ImportError:
            from libsvm.svmutil import svm_predict

        # Convert to libsvm format: {1: val1, 2: val2, ...}
        x: dict[int, float] = {i + 1: float(v) for i, v in enumerate(fvnsim)}
        predicted_labels, _, _ = svm_predict([0], [x], self.model, "-q")
        return float(np.clip(predicted_labels[0], 1.0, 5.0))


class SpeechSimilarityToQualityMapper(SimilarityToQualityMapper):
    """
    Exponential-fit quality mapper for Speech mode.

    Uses hardcoded parameters fitted on the TCD-VOIP dataset.
    """

    # Fitted parameters (from C++ speech_similarity_to_quality_mapper.cc)
    FIT_A: float = -262.847869
    FIT_B: float = 0.0154302525
    FIT_X0: float = -361.063949
    FIT_SCALE: float = 1.245063

    def __init__(self, scale_to_max_mos: bool = True) -> None:
        self.scale: float = self.FIT_SCALE if scale_to_max_mos else 1.0

    def init(self) -> None:
        """No initialization needed for exponential mapper."""

    def predict_quality(
        self,
        fvnsim: NDArray[np.float64],
        fvnsim10: NDArray[np.float64] | None = None,
        fstdnsim: NDArray[np.float64] | None = None,
        fvdegenergy: NDArray[np.float64] | None = None,
    ) -> float:
        """Predict MOS using exponential fit: ``a + exp(b * (x - x0))``."""
        nsim_mean: float = float(np.mean(fvnsim))
        mos: float = self.FIT_A + float(np.exp(self.FIT_B * (nsim_mean - self.FIT_X0)))
        return float(np.clip(mos * self.scale, 1.0, 5.0))


class TFLiteSpeechQualityMapper(SimilarityToQualityMapper):
    """
    Deep-lattice TFLite quality mapper for Speech mode.

    Loads the same ``.tflite`` lattice model used by C++ ViSQOL
    (``--use_lattice_model=true``, the C++ default). Inference runs in the
    bundled TFLite C++ runtime via the ``ai-edge-litert`` Python binding, so
    results are numerically equivalent to the C++ implementation.

    Install the runtime with::

        pip install visqol-python[lattice]
    """

    NUM_FREQUENCY_BANDS: int = 21
    TAU: float = 0.5  # quantile feature; matches C++ default

    def __init__(self, model_path: str) -> None:
        self.model_path: str = model_path
        self._interpreter: Any = None
        self._runner: Any = None

    def init(self) -> None:
        """Load the TFLite lattice model.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ImportError: If ai-edge-litert is not installed.
        """
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"TFLite lattice model file not found: {self.model_path}")

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise ImportError(
                "ai-edge-litert is required for the lattice speech quality mapper. "
                "Install with: pip install visqol-python[lattice]"
            ) from exc

        logger.debug("Loading TFLite lattice model: %s", self.model_path)
        # Single-threaded interpreter to match C++ SetNumThreads(1) default.
        self._interpreter = Interpreter(model_path=self.model_path, num_threads=1)
        self._interpreter.allocate_tensors()
        self._runner = self._interpreter.get_signature_runner("predict")

    def predict_quality(
        self,
        fvnsim: NDArray[np.float64],
        fvnsim10: NDArray[np.float64] | None = None,
        fstdnsim: NDArray[np.float64] | None = None,
        fvdegenergy: NDArray[np.float64] | None = None,
    ) -> float:
        """Predict MOS by running the TFLite lattice network.

        Feeds ``4 × 21 = 84`` per-band scalar features plus the ``tau`` quantile
        scalar, exactly mirroring ``TFLiteQualityMapper::PredictQuality`` in C++.
        """
        if self._runner is None:
            raise RuntimeError(
                "TFLiteSpeechQualityMapper.init() must be called before predict_quality()."
            )
        if fvnsim10 is None or fstdnsim is None or fvdegenergy is None:
            raise ValueError(
                "Lattice mapper requires fvnsim, fvnsim10, fstdnsim and fvdegenergy."
            )

        nb = self.NUM_FREQUENCY_BANDS
        if (
            len(fvnsim) < nb
            or len(fvnsim10) < nb
            or len(fstdnsim) < nb
            or len(fvdegenergy) < nb
        ):
            raise ValueError(
                f"Lattice mapper expects {nb} bands per feature; "
                f"got fvnsim={len(fvnsim)} fvnsim10={len(fvnsim10)} "
                f"fstdnsim={len(fstdnsim)} fvdegenergy={len(fvdegenergy)}"
            )

        # Per-band scalar inputs (float32 to match TFLite tensor dtype).
        kwargs: dict[str, NDArray[np.float32]] = {}
        for band in range(nb):
            kwargs[f"fvnsim{band}"] = np.array([fvnsim[band]], dtype=np.float32)
            kwargs[f"fvnsim10_{band}"] = np.array([fvnsim10[band]], dtype=np.float32)
            kwargs[f"fstdnsim{band}"] = np.array([fstdnsim[band]], dtype=np.float32)
            kwargs[f"fvdegenergy{band}"] = np.array([fvdegenergy[band]], dtype=np.float32)
        kwargs["tau"] = np.array([self.TAU], dtype=np.float32)

        out = self._runner(**kwargs)
        return float(out["predictions"].flatten()[0])
