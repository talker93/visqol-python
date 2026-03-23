"""
ViSQOL Manager: orchestrates the complete ViSQOL workflow.

Corresponds to C++ file: visqol_manager.cc
"""

import os
import logging
from typing import Optional

from visqol.audio_utils import AudioSignal, load_as_mono
from visqol.analysis_window import AnalysisWindow
from visqol.alignment import globally_align
from visqol.gammatone import GammatoneSpectrogramBuilder
from visqol.patch_creator import ImagePatchCreator, VadPatchCreator
from visqol.quality_mapper import (
    SvrSimilarityToQualityMapper,
    SpeechSimilarityToQualityMapper,
)
from visqol.visqol_core import VisqolCore, SimilarityResult

logger = logging.getLogger(__name__)

# Default parameters (matching C++ VisqolManager constants)
PATCH_SIZE_AUDIO = 30
PATCH_SIZE_SPEECH = 20
NUM_BANDS_AUDIO = 32
NUM_BANDS_SPEECH = 21
MINIMUM_FREQ = 50.0  # Hz (wideband)
OVERLAP = 0.25  # 25% overlap
DURATION_MISMATCH_TOLERANCE = 1.0  # seconds

K_16K_SAMPLE_RATE = 16000
K_48K_SAMPLE_RATE = 48000


class VisqolManager:
    """
    Main manager class that configures and runs ViSQOL.
    """

    def __init__(self):
        self.use_speech_mode = False
        self.search_window = 60
        self.disable_global_alignment = False
        self.disable_realignment = False
        self.is_initialized = False

        self.spect_builder = None
        self.patch_creator = None
        self.quality_mapper = None
        self.visqol_core = VisqolCore()

    def init(self,
             model_path: str = "",
             use_speech_mode: bool = False,
             use_unscaled_speech: bool = False,
             search_window: int = 60,
             disable_global_alignment: bool = False,
             disable_realignment: bool = False):
        """
        Initialize the ViSQOL manager.

        Args:
            model_path: Path to SVR model file (Audio mode).
            use_speech_mode: Use speech mode (16kHz, 21 bands).
            use_unscaled_speech: Don't scale speech MOS to max 5.0.
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
                NUM_BANDS_SPEECH, MINIMUM_FREQ, speech_mode=True
            )
        else:
            self.spect_builder = GammatoneSpectrogramBuilder(
                NUM_BANDS_AUDIO, MINIMUM_FREQ, speech_mode=False
            )

        # Initialize quality mapper
        if use_speech_mode:
            self.quality_mapper = SpeechSimilarityToQualityMapper(
                scale_to_max_mos=not use_unscaled_speech
            )
        else:
            self.quality_mapper = SvrSimilarityToQualityMapper(model_path)

        self.quality_mapper.init()
        self.is_initialized = True

    def run(self, ref_path: str, deg_path: str) -> SimilarityResult:
        """
        Run ViSQOL on two audio files.

        Args:
            ref_path: Path to reference audio file.
            deg_path: Path to degraded audio file.

        Returns:
            SimilarityResult with MOS-LQO score.
        """
        if not self.is_initialized:
            raise RuntimeError("VisqolManager must be initialized before use.")

        # Load audio
        ref_signal = load_as_mono(ref_path)
        deg_signal = load_as_mono(deg_path)

        return self.run_from_signals(ref_signal, deg_signal)

    def run_from_signals(self, ref_signal: AudioSignal,
                         deg_signal: AudioSignal) -> SimilarityResult:
        """
        Run ViSQOL on two audio signals.

        Args:
            ref_signal: Reference audio signal.
            deg_signal: Degraded audio signal.

        Returns:
            SimilarityResult with MOS-LQO score.
        """
        if not self.is_initialized:
            raise RuntimeError("VisqolManager must be initialized before use.")

        # Validate
        self._validate_input(ref_signal, deg_signal)

        # Global alignment
        if not self.disable_global_alignment:
            aligned_deg, lag = globally_align(ref_signal, deg_signal)
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

    def _validate_input(self, ref_signal: AudioSignal,
                        deg_signal: AudioSignal):
        """Validate input audio signals."""
        # Check duration mismatch
        ref_dur = ref_signal.duration
        deg_dur = deg_signal.duration
        if abs(ref_dur - deg_dur) > DURATION_MISMATCH_TOLERANCE:
            logger.warning(
                "Duration mismatch: reference=%.2fs, degraded=%.2fs",
                ref_dur, deg_dur
            )

        # Check sample rate match
        if ref_signal.sample_rate != deg_signal.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: reference={ref_signal.sample_rate}, "
                f"degraded={deg_signal.sample_rate}"
            )

        if self.use_speech_mode:
            if ref_signal.sample_rate > K_16K_SAMPLE_RATE:
                logger.warning(
                    "Input sample rate (%d) is above 16kHz. "
                    "Consider resampling for speech mode.",
                    ref_signal.sample_rate
                )
        else:
            if ref_signal.sample_rate != K_48K_SAMPLE_RATE:
                logger.warning(
                    "Input sample rate (%d) is not 48kHz. "
                    "This may affect audio mode scoring.",
                    ref_signal.sample_rate
                )
