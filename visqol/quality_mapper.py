"""
Quality mappers: SVR (Audio mode) and Exponential (Speech mode).

Corresponds to C++ files:
  - svr_similarity_to_quality_mapper.cc
  - speech_similarity_to_quality_mapper.cc
  - support_vector_regression_model.cc
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
