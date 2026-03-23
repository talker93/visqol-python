"""
Core ViSQOL algorithm: assembles all components into a similarity computation pipeline.

Corresponds to C++ file: visqol.cc
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import AudioSignal, scale_to_match_sound_pressure_level
from visqol.gammatone import (
    GammatoneSpectrogramBuilder,
    Spectrogram,
    prepare_spectrograms_for_comparison,
)
from visqol.nsim import PatchSimilarityResult
from visqol.patch_selector import (
    find_most_optimal_deg_patches,
    finely_align_and_recreate_patches,
)
from visqol.quality_mapper import SimilarityToQualityMapper

if TYPE_CHECKING:
    from visqol.patch_creator import ImagePatchCreator, VadPatchCreator


@dataclass
class SimilarityResult:
    """Complete result of a ViSQOL similarity comparison."""

    moslqo: float = 0.0
    vnsim: float = 0.0
    fvnsim: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    fvnsim10: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    fstdnsim: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    fvdegenergy: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    center_freq_bands: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    patch_sims: list[PatchSimilarityResult] = field(default_factory=list)

    def __str__(self) -> str:
        return f"SimilarityResult(moslqo={self.moslqo:.4f}, vnsim={self.vnsim:.4f})"

    def __repr__(self) -> str:
        return (
            f"SimilarityResult(moslqo={self.moslqo!r}, vnsim={self.vnsim!r}, "
            f"fvnsim=<{len(self.fvnsim)} bands>, "
            f"patch_sims=<{len(self.patch_sims)} patches>)"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def calc_per_patch_mean_freq_band_means(
    sim_results: list[PatchSimilarityResult],
) -> NDArray[np.float64]:
    """
    Mean of per-frequency-band means across all patches (``fvnsim``).

    Matches C++ ``Visqol::CalcPerPatchMeanFreqBandMeans``.
    """
    all_means = np.array([r.freq_band_means for r in sim_results])
    return np.asarray(np.mean(all_means, axis=0), dtype=np.float64)


def calc_per_patch_freq_band_quantile(
    sim_results: list[PatchSimilarityResult],
    quantile: float = 0.10,
) -> NDArray[np.float64]:
    """
    Quantile of per-frequency-band means across patches (``fvnsim10``).

    Matches C++ ``Visqol::CalcPerPatchFreqBandQuantile``.
    """
    num_freq_bands: int = len(sim_results[0].freq_band_means)
    result = np.zeros(num_freq_bands)

    for band in range(num_freq_bands):
        band_nsims = sorted([r.freq_band_means[band] for r in sim_results])
        num_in_quantile = max(1, int(len(band_nsims) * quantile))
        result[band] = np.mean(band_nsims[:num_in_quantile])

    return result


def calc_per_patch_mean_freq_band_deg_energy(
    sim_results: list[PatchSimilarityResult],
) -> NDArray[np.float64]:
    """
    Mean of per-frequency-band degraded energy across patches (``fvdegenergy``).

    Matches C++ ``Visqol::CalcPerPatchMeanFreqBandDegradedEnergy``.
    """
    all_energy = np.array([r.freq_band_deg_energy for r in sim_results])
    return np.asarray(np.mean(all_energy, axis=0), dtype=np.float64)


def calc_per_patch_mean_freq_band_stddevs(
    sim_results: list[PatchSimilarityResult],
    frame_duration: float,
) -> NDArray[np.float64]:
    """
    Pooled standard deviation across patches (``fstdnsim``).

    Matches C++ ``Visqol::CalcPerPatchMeanFreqBandStdDevs``.

    Uses the pooled variance formula:
    https://en.wikipedia.org/wiki/Pooled_variance
    """
    num_freq_bands: int = len(sim_results[0].freq_band_means)

    # First compute fvnsim (global mean per band)
    fvnsim = calc_per_patch_mean_freq_band_means(sim_results)

    total_frame_count: int = 0
    contribution = np.zeros(num_freq_bands)

    for patch in sim_results:
        secs_in_patch: float = patch.ref_patch_end_time - patch.ref_patch_start_time
        frame_count = math.ceil(secs_in_patch / frame_duration)
        total_frame_count += frame_count

        for band in range(num_freq_bands):
            stddev: float = patch.freq_band_stddevs[band]
            mean: float = patch.freq_band_means[band]
            contribution[band] += (frame_count - 1) * stddev * stddev
            contribution[band] += frame_count * mean * mean

    if total_frame_count <= 1:
        return np.zeros(num_freq_bands)

    variance = (contribution - fvnsim * fvnsim * total_frame_count) / (total_frame_count - 1)

    # sqrt, filtering negative values due to floating-point precision
    with np.errstate(invalid="ignore"):
        result: NDArray[np.float64] = np.where(variance < 0, 0.0, np.sqrt(variance))
    return result


def alter_for_similarity_extremes(vnsim: float, moslqo: float) -> float:
    """
    Handle extreme similarity cases.

    Matches C++ ``Visqol::AlterForSimilarityExtremes``.
    """
    if vnsim < 0.15:
        return 1.0
    return moslqo


def calc_frame_duration(frame_size: int, sample_rate: int) -> float:
    """Calculate frame duration in seconds."""
    return frame_size / float(sample_rate)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class VisqolCore:
    """
    Core ViSQOL algorithm.

    Orchestrates the complete similarity calculation pipeline.
    """

    def calculate_similarity(
        self,
        ref_signal: AudioSignal,
        deg_signal: AudioSignal,
        spect_builder: GammatoneSpectrogramBuilder,
        window: AnalysisWindow,
        patch_creator: ImagePatchCreator | VadPatchCreator,
        search_window: int,
        quality_mapper: SimilarityToQualityMapper,
        disable_realignment: bool = False,
    ) -> SimilarityResult:
        """
        Calculate full similarity between reference and degraded signals.

        Args:
            ref_signal: Reference audio signal.
            deg_signal: Degraded audio signal.
            spect_builder: Gammatone spectrogram builder.
            window: Analysis window.
            patch_creator: Patch creator (Image or VAD).
            search_window: Search window radius in patch units.
            quality_mapper: Similarity-to-quality mapper.
            disable_realignment: If *True*, skip fine realignment.

        Returns:
            :class:`SimilarityResult` with MOS-LQO score and all intermediate
            data.
        """
        # Stage 1: Preprocessing — SPL matching
        deg_signal = scale_to_match_sound_pressure_level(ref_signal, deg_signal)

        # Build spectrograms
        ref_spectrogram: Spectrogram = spect_builder.build(ref_signal, window)
        deg_spectrogram: Spectrogram = spect_builder.build(deg_signal, window)

        # Prepare spectrograms for comparison (dB conversion + noise floor)
        ref_db, deg_db = prepare_spectrograms_for_comparison(ref_spectrogram, deg_spectrogram)

        # Stage 2: Feature selection and similarity measure
        ref_patch_indices: list[int] = patch_creator.create_ref_patch_indices(
            ref_db, ref_signal, window
        )

        frame_duration: float = calc_frame_duration(
            int(window.size * window.overlap), ref_signal.sample_rate
        )

        ref_patches: list[NDArray[np.float64]] = patch_creator.create_patches_from_indices(
            ref_db, ref_patch_indices
        )

        # DP patch matching
        sim_match_info: list[PatchSimilarityResult] = find_most_optimal_deg_patches(
            ref_patches,
            ref_patch_indices,
            deg_db,
            frame_duration,
            search_window,
        )

        # Fine realignment
        if not disable_realignment:
            sim_match_info = finely_align_and_recreate_patches(
                sim_match_info,
                ref_signal,
                deg_signal,
                spect_builder,
                window,
            )

        # Aggregate statistics
        fvnsim = calc_per_patch_mean_freq_band_means(sim_match_info)
        fvnsim10 = calc_per_patch_freq_band_quantile(sim_match_info, 0.10)
        fstdnsim = calc_per_patch_mean_freq_band_stddevs(
            sim_match_info,
            frame_duration,
        )
        fvdegenergy = calc_per_patch_mean_freq_band_deg_energy(sim_match_info)

        # Predict MOS
        moslqo: float = quality_mapper.predict_quality(
            fvnsim,
            fvnsim10,
            fstdnsim,
            fvdegenergy,
        )

        # Calculate vnsim (mean of fvnsim)
        vnsim: float = float(np.mean(fvnsim))

        # Handle extreme cases
        moslqo = alter_for_similarity_extremes(vnsim, moslqo)

        return SimilarityResult(
            moslqo=moslqo,
            vnsim=vnsim,
            fvnsim=fvnsim,
            fvnsim10=fvnsim10,
            fstdnsim=fstdnsim,
            fvdegenergy=fvdegenergy,
            center_freq_bands=ref_spectrogram.center_freq_bands,
            patch_sims=sim_match_info,
        )
