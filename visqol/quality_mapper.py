"""
Quality mappers: SVR (Audio mode) and Exponential (Speech mode).

Corresponds to C++ files:
  - svr_similarity_to_quality_mapper.cc
  - speech_similarity_to_quality_mapper.cc
  - support_vector_regression_model.cc
"""

import os
import logging
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SimilarityToQualityMapper(ABC):
    """Abstract base class for similarity-to-quality mapping."""

    @abstractmethod
    def init(self) -> None:
        """Initialize the mapper (e.g. load model)."""
        pass

    @abstractmethod
    def predict_quality(self, fvnsim: np.ndarray,
                        fvnsim10: np.ndarray = None,
                        fstdnsim: np.ndarray = None,
                        fvdegenergy: np.ndarray = None) -> float:
        """Predict MOS quality from NSIM feature vectors."""
        pass


class SvrSimilarityToQualityMapper(SimilarityToQualityMapper):
    """
    SVR-based quality mapper for Audio mode.
    Uses libsvm to load and predict from pre-trained SVR model.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def init(self):
        """Load the libsvm model file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"SVR model file not found: {self.model_path}"
            )

        try:
            from svmutil import svm_load_model
            self.model = svm_load_model(self.model_path)
        except ImportError:
            # Try alternative import paths
            try:
                from libsvm.svmutil import svm_load_model
                self.model = svm_load_model(self.model_path)
            except ImportError:
                raise ImportError(
                    "libsvm is required for Audio mode SVR quality mapping. "
                    "Install with: pip install libsvm-official"
                )

    def predict_quality(self, fvnsim: np.ndarray,
                        fvnsim10: np.ndarray = None,
                        fstdnsim: np.ndarray = None,
                        fvdegenergy: np.ndarray = None) -> float:
        """
        Predict MOS using SVR model.
        Only fvnsim is used as input features.
        """
        try:
            from svmutil import svm_predict
        except ImportError:
            from libsvm.svmutil import svm_predict

        # Convert to libsvm format: {1: val1, 2: val2, ...}
        x = {i + 1: float(v) for i, v in enumerate(fvnsim)}
        # svm_predict returns (predicted_labels, (MSE, SCC, ...), decision_values)
        predicted_labels, _, _ = svm_predict([0], [x], self.model, '-q')
        return float(np.clip(predicted_labels[0], 1.0, 5.0))


class SpeechSimilarityToQualityMapper(SimilarityToQualityMapper):
    """
    Exponential-fit quality mapper for Speech mode.
    Uses hardcoded parameters fitted on TCD-VOIP dataset.
    """

    # Fitted parameters (from C++ speech_similarity_to_quality_mapper.cc)
    FIT_A = -262.847869
    FIT_B = 0.0154302525
    FIT_X0 = -361.063949
    FIT_SCALE = 1.245063

    def __init__(self, scale_to_max_mos: bool = True):
        self.scale = self.FIT_SCALE if scale_to_max_mos else 1.0

    def init(self):
        """No initialization needed for exponential mapper."""
        pass

    def predict_quality(self, fvnsim: np.ndarray,
                        fvnsim10: np.ndarray = None,
                        fstdnsim: np.ndarray = None,
                        fvdegenergy: np.ndarray = None) -> float:
        """
        Predict MOS using exponential fit: a + exp(b * (x - x0)).
        """
        nsim_mean = float(np.mean(fvnsim))
        mos = self.FIT_A + np.exp(self.FIT_B * (nsim_mean - self.FIT_X0))
        return float(np.clip(mos * self.scale, 1.0, 5.0))
