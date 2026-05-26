"""
ViSQOL Manager: orchestrates the complete ViSQOL workflow.

Corresponds to C++ file: visqol_manager.cc
"""

from __future__ import annotations

import logging
import os

from visqol.alignment import globally_align
from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import AudioSignal, load_as_mono
from visqol.gammatone import GammatoneSpectrogramBuilder
from visqol.patch_creator import ImagePatchCreator, VadPatchCreator
from visqol.quality_mapper import (
    SimilarityToQualityMapper,
    SpeechSimilarityToQualityMapper,
    SvrSimilarityToQualityMapper,
    TFLiteSpeechQualityMapper,
    has_lattice_runtime,
)
from visqol.visqol_core import SimilarityResult, VisqolCore

logger = logging.getLogger(__name__)

# Default parameters (matching C++ VisqolManager constants)
PATCH_SIZE_AUDIO: int = 30
PATCH_SIZE_SPEECH: int = 20
NUM_BANDS_AUDIO: int = 32
NUM_BANDS_SPEECH: int = 21
MINIMUM_FREQ: float = 50.0  # Hz (wideband)
OVERLAP: float = 0.25  # 25 % overlap
DURATION_MISMATCH_TOLERANCE: float = 1.0  # seconds

K_16K_SAMPLE_RATE: int = 16000
K_48K_SAMPLE_RATE: int = 48000

# Default bundled lattice TFLite model for speech mode (C++ default behaviour).
_DEFAULT_LATTICE_MODEL: str = os.path.join(
    os.path.dirname(__file__),
    "model",
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
)


class VisqolManager:
    """Main manager class that configures and runs ViSQOL."""

    def __init__(self) -> None:
        self.use_speech_mode: bool = False
        self.use_lattice_model: bool = True
        self.search_window: int = 60
        self.disable_global_alignment: bool = False
        self.disable_realignment: bool = False
        self.is_initialized: bool = False

        self.spect_builder: GammatoneSpectrogramBuilder | None = None
        self.patch_creator: ImagePatchCreator | VadPatchCreator | None = None
        self.quality_mapper: SimilarityToQualityMapper | None = None
        self.visqol_core: VisqolCore = VisqolCore()

    def init(
        self,
        model_path: str = "",
        use_speech_mode: bool = False,
        use_unscaled_speech: bool = False,
        use_lattice_model: bool | None = None,
        lattice_model_path: str = "",
        search_window: int = 60,
        disable_global_alignment: bool = False,
        disable_realignment: bool = False,
    ) -> None:
        """
        Initialize the ViSQOL manager.

        Args:
            model_path: Path to SVR model file (Audio mode).
            use_speech_mode: Use speech mode (16 kHz, 21 bands).
            use_unscaled_speech: Don't scale speech MOS to max 5.0 (polynomial only).
            use_lattice_model: Speech mode only. If *True*, use the deep-lattice
                TFLite mapper (matches C++ default). If *False*, use the
                polynomial fallback. If *None* (default), auto-enable when
                ``ai-edge-litert`` is installed, otherwise fall back to
                polynomial with a one-time warning.
            lattice_model_path: Custom path to the lattice ``.tflite`` model.
                Empty string means use the bundled default.
            search_window: Search window radius in patch units.
            disable_global_alignment: Skip global alignment.
            disable_realignment: Skip fine realignment.
        """
        self.use_speech_mode = use_speech_mode
        self.search_window = search_window
        self.disable_global_alignment = disable_global_alignment
        self.disable_realignment = disable_realignment

        # Initialize patch creator
        if use_speech_mode:
            self.patch_creator = VadPatchCreator(PATCH_SIZE_SPEECH)
        else:
            self.patch_creator = ImagePatchCreator(PATCH_SIZE_AUDIO)

        # Initialize spectrogram builder
        if use_speech_mode:
            self.spect_builder = GammatoneSpectrogramBuilder(
                NUM_BANDS_SPEECH,
                MINIMUM_FREQ,
                speech_mode=True,
            )
        else:
            self.spect_builder = GammatoneSpectrogramBuilder(
                NUM_BANDS_AUDIO,
                MINIMUM_FREQ,
                speech_mode=False,
            )

        # Initialize quality mapper
        if use_speech_mode:
            self.use_lattice_model = self._resolve_use_lattice(use_lattice_model)
            if self.use_lattice_model:
                tflite_path = lattice_model_path or _DEFAULT_LATTICE_MODEL
                self.quality_mapper = TFLiteSpeechQualityMapper(tflite_path)
            else:
                self.quality_mapper = SpeechSimilarityToQualityMapper(
                    scale_to_max_mos=not use_unscaled_speech,
                )
        else:
            self.use_lattice_model = False
            self.quality_mapper = SvrSimilarityToQualityMapper(model_path)

        self.quality_mapper.init()
        self.is_initialized = True

    @staticmethod
    def _resolve_use_lattice(requested: bool | None) -> bool:
        """Decide whether to use the lattice mapper based on user request + runtime availability."""
        if requested is True:
            # User explicitly asked for lattice; surface clear ImportError if missing.
            return True
        if requested is False:
            return False
        # Auto-detect: prefer lattice if the runtime is installed.
        if has_lattice_runtime():
            return True
        logger.warning(
            "Speech mode lattice runtime (ai-edge-litert) not installed; "
            "falling back to polynomial mapping, which may over-predict MOS "
            "by 1-2 points vs the C++ default. "
            "Install with: pip install visqol-python[lattice]"
        )
        return False

    def run(self, ref_path: str, deg_path: str) -> SimilarityResult:
        """
        Run ViSQOL on two audio files.

        Args:
            ref_path: Path to reference audio file.
            deg_path: Path to degraded audio file.

        Returns:
            :class:`SimilarityResult` with MOS-LQO score.

        Raises:
            RuntimeError: If manager has not been initialized.
        """
        self._ensure_initialized()

        # Load audio
        ref_signal: AudioSignal = load_as_mono(ref_path)
        deg_signal: AudioSignal = load_as_mono(deg_path)

        return self.run_from_signals(ref_signal, deg_signal)

    def run_from_signals(
        self,
        ref_signal: AudioSignal,
        deg_signal: AudioSignal,
    ) -> SimilarityResult:
        """
        Run ViSQOL on two audio signals.

        Args:
            ref_signal: Reference audio signal.
            deg_signal: Degraded audio signal.

        Returns:
            :class:`SimilarityResult` with MOS-LQO score.

        Raises:
            RuntimeError: If manager has not been initialized.
            ValueError: If sample rates do not match.
        """
        self._ensure_initialized()

        # Type narrowing — guaranteed non-None after _ensure_initialized
        assert self.spect_builder is not None
        assert self.patch_creator is not None
        assert self.quality_mapper is not None

        # Validate
        self._validate_input(ref_signal, deg_signal)

        # Global alignment
        if not self.disable_global_alignment:
            aligned_deg, _lag = globally_align(ref_signal, deg_signal)
            deg_signal = aligned_deg

        # Create analysis window
        window = AnalysisWindow(ref_signal.sample_rate, OVERLAP)

        # Run core algorithm
        return self.visqol_core.calculate_similarity(
            ref_signal=ref_signal,
            deg_signal=deg_signal,
            spect_builder=self.spect_builder,
            window=window,
            patch_creator=self.patch_creator,
            search_window=self.search_window,
            quality_mapper=self.quality_mapper,
            disable_realignment=self.disable_realignment,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Raise if :meth:`init` has not been called."""
        if not self.is_initialized:
            raise RuntimeError("VisqolManager must be initialized before use.")

    def _validate_input(self, ref_signal: AudioSignal, deg_signal: AudioSignal) -> None:
        """Validate input audio signals.

        Raises:
            ValueError: If sample rates do not match.
        """
        # Check sample rate match
        if ref_signal.sample_rate != deg_signal.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: reference={ref_signal.sample_rate}, "
                f"degraded={deg_signal.sample_rate}. "
                "Both signals must have the same sample rate."
            )

        # Check duration mismatch
        ref_dur = ref_signal.duration
        deg_dur = deg_signal.duration
        if abs(ref_dur - deg_dur) > DURATION_MISMATCH_TOLERANCE:
            logger.warning(
                "Duration mismatch: reference=%.2fs, degraded=%.2fs",
                ref_dur,
                deg_dur,
            )

        if self.use_speech_mode:
            if ref_signal.sample_rate > K_16K_SAMPLE_RATE:
                logger.warning(
                    "Input sample rate (%d) is above 16 kHz. "
                    "Consider resampling for speech mode.",
                    ref_signal.sample_rate,
                )
        else:
            if ref_signal.sample_rate != K_48K_SAMPLE_RATE:
                logger.warning(
                    "Input sample rate (%d) is not 48 kHz. "
                    "This may affect audio mode scoring.",
                    ref_signal.sample_rate,
                )
