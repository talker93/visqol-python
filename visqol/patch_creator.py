"""
Patch creation for reference spectrograms.

- ``ImagePatchCreator``: evenly-spaced patches (Audio mode)
- ``VadPatchCreator``: VAD-filtered patches (Speech mode)

Corresponds to C++ files:
  - image_patch_creator.cc
  - vad_patch_creator.cc
  - rms_vad.cc/h
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import AudioSignal
from visqol.signal_utils import normalize

# ============ RMS VAD ============


class RmsVad:
    """
    Simple RMS-based Voice Activity Detection.

    Matches C++ ``RmsVad`` class.
    """

    VOICE_ACTIVITY_PRESENT: float = 1.0
    VOICE_ACTIVITY_ABSENT: float = 0.0
    SILENT_CHUNK_COUNT: int = 3
    RMS_THRESHOLD: float = 5000.0

    def __init__(self) -> None:
        self.each_chunk_result: list[float] = []
        # Initialize first (kSilentChunkCount - 1) results as voice-active
        # to avoid false negatives
        self.vad_results: list[float] = [self.VOICE_ACTIVITY_PRESENT] * (
            self.SILENT_CHUNK_COUNT - 1
        )

    def process_chunk(self, chunk: NDArray[np.int16]) -> float:
        """Process a chunk of int16 samples."""
        rms: float = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < self.RMS_THRESHOLD:
            self.each_chunk_result.append(self.VOICE_ACTIVITY_ABSENT)
        else:
            self.each_chunk_result.append(self.VOICE_ACTIVITY_PRESENT)
        return rms

    def get_vad_results(self) -> list[float]:
        """Get VAD results for all processed chunks."""
        for i in range(self.SILENT_CHUNK_COUNT - 1, len(self.each_chunk_result)):
            if not self.each_chunk_result[i] and self._check_previous_chunks_for_silence(i):
                self.vad_results.append(self.VOICE_ACTIVITY_ABSENT)
            else:
                self.vad_results.append(self.VOICE_ACTIVITY_PRESENT)
        return self.vad_results

    def _check_previous_chunks_for_silence(self, idx: int) -> bool:
        """Check if previous chunks are also silent."""
        for j in range(1, self.SILENT_CHUNK_COUNT):
            if self.each_chunk_result[idx - j] == self.VOICE_ACTIVITY_PRESENT:
                return False
        return True


# ============ Image Patch Creator (Audio mode) ============


class ImagePatchCreator:
    """
    Creates evenly-spaced patches from a spectrogram.

    Used for Audio mode.
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size: int = patch_size

    def create_ref_patch_indices(
        self,
        spectrogram: NDArray[np.float64],
        ref_signal: AudioSignal | None = None,
        window: AnalysisWindow | None = None,
    ) -> list[int]:
        """
        Create reference patch indices at evenly-spaced intervals.

        Matches C++ ``ImagePatchCreator::CreateRefPatchIndices``.

        Args:
            spectrogram: ``(num_bands, num_frames)`` spectrogram matrix.
            ref_signal: (unused, kept for interface compat.)
            window: (unused, kept for interface compat.)

        Returns:
            List of start column indices for patches.

        Raises:
            ValueError: If spectrum is too short for a single patch.
        """
        spectrum_length: int = spectrogram.shape[1]
        init_patch_index: int = self.patch_size // 2

        if spectrum_length < self.patch_size + init_patch_index:
            raise ValueError(
                f"Reference spectrum size ({spectrum_length}) smaller than "
                f"minimum patch size ({self.patch_size + init_patch_index})."
            )

        # C++: max_index logic
        if init_patch_index < (spectrum_length - self.patch_size):
            max_index = spectrum_length - self.patch_size
        else:
            max_index = init_patch_index + 1

        indices: list[int] = []
        i = init_patch_index
        while i < max_index:
            indices.append(i - 1)  # C++ uses 0-based, pushes i-1
            i += self.patch_size

        return indices

    def create_patches_from_indices(
        self,
        spectrogram: NDArray[np.float64],
        patch_indices: list[int],
    ) -> list[NDArray[np.float64]]:
        """
        Extract patches from spectrogram at given start indices.

        Args:
            spectrogram: ``(num_bands, num_frames)`` matrix.
            patch_indices: List of start column indices.

        Returns:
            List of ``(num_bands, patch_size)`` patch matrices.
        """
        patches: list[NDArray[np.float64]] = []
        for start_col in patch_indices:
            end_col = start_col + self.patch_size
            patch = spectrogram[:, start_col:end_col]
            patches.append(patch)
        return patches


# ============ VAD Patch Creator (Speech mode) ============


class VadPatchCreator:
    """
    Creates VAD-filtered patches from a spectrogram.

    Used for Speech mode.
    """

    FRAMES_WITH_VA_THRESHOLD: float = 1.0

    def __init__(self, patch_size: int) -> None:
        self.patch_size: int = patch_size

    def _get_voice_activity(
        self,
        signal: AudioSignal,
        start_sample: int,
        total_samples: int,
        frame_len: int,
    ) -> list[float]:
        """
        Get voice activity detection results for a signal segment.

        Matches C++ ``VadPatchCreator::GetVoiceActivity``.
        """
        rms_vad = RmsVad()
        data = signal.data[start_sample : start_sample + total_samples]

        # Convert to int16 range and process in chunks
        frame: list[int] = []
        for val in data:
            int_val = val * (1 << 15)
            int_val = max(-1.0 * (1 << 15), min(1.0 * ((1 << 15) - 1), int_val))
            frame.append(int(int_val))
            if len(frame) == frame_len:
                rms_vad.process_chunk(np.array(frame, dtype=np.int16))
                frame = []

        return rms_vad.get_vad_results()

    def create_ref_patch_indices(
        self,
        spectrogram: NDArray[np.float64],
        ref_signal: AudioSignal,
        window: AnalysisWindow,
    ) -> list[int]:
        """
        Create VAD-filtered reference patch indices.

        Matches C++ ``VadPatchCreator::CreateRefPatchIndices``.
        """
        # Normalize signal
        norm_data = normalize(ref_signal.data)
        norm_signal = AudioSignal(norm_data, ref_signal.sample_rate)

        frame_size: int = int(window.size * window.overlap)
        patch_sample_len: int = self.patch_size * frame_size
        spectrum_length: int = spectrogram.shape[1]
        first_patch_idx: int = self.patch_size // 2 - 1
        patch_count: int = (spectrum_length - first_patch_idx) // self.patch_size
        total_sample_count: int = patch_count * patch_sample_len

        # Get VAD results
        vad_res = self._get_voice_activity(
            norm_signal,
            first_patch_idx,
            total_sample_count,
            frame_size,
        )

        # Filter patches based on VAD
        ref_patch_indices: list[int] = []
        patch_idx = first_patch_idx
        for i in range(patch_count):
            start = i * self.patch_size
            end = start + self.patch_size
            patch_vad = vad_res[start:end] if end <= len(vad_res) else vad_res[start:]

            frames_with_va: float = sum(patch_vad)
            if frames_with_va >= self.FRAMES_WITH_VA_THRESHOLD:
                ref_patch_indices.append(patch_idx)

            patch_idx += self.patch_size

        return ref_patch_indices

    def create_patches_from_indices(
        self,
        spectrogram: NDArray[np.float64],
        patch_indices: list[int],
    ) -> list[NDArray[np.float64]]:
        """Extract patches from spectrogram at given indices."""
        patches: list[NDArray[np.float64]] = []
        for start_col in patch_indices:
            end_col = start_col + self.patch_size
            patch = spectrogram[:, start_col:end_col]
            patches.append(patch)
        return patches
