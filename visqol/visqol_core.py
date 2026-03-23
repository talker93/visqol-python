"""
Core ViSQOL algorithm: assembles all components into a similarity computation pipeline.

Corresponds to C++ file: visqol.cc
"""

import math
import numpy as np
from typing import List
from dataclasses import dataclass, field

from visqol.audio_utils import AudioSignal, scale_to_match_sound_pressure_level
from visqol.analysis_window import AnalysisWindow
from visqol.gammatone import (
    GammatoneSpectrogramBuilder, Spectrogram,
    prepare_spectrograms_for_comparison,
)
from visqol.nsim import PatchSimilarityResult, measure_patch_similarity
from visqol.patch_selector import (
    find_most_optimal_deg_patches,
    finely_align_and_recreate_patches,
)
from visqol.quality_mapper import SimilarityToQualityMapper


@dataclass
class SimilarityResult:
    """Complete result of a ViSQOL similarity comparison."""
    moslqo: float = 0.0
    vnsim: float = 0.0
    fvnsim: np.ndarray = field(default_factory=lambda: np.array([]))
    fvnsim10: np.ndarray = field(default_factory=lambda: np.array([]))
    fstdnsim: np.ndarray = field(default_factory=lambda: np.array([]))
    fvdegenergy: np.ndarray = field(default_factory=lambda: np.array([]))
    center_freq_bands: np.ndarray = field(default_factory=lambda: np.array([]))
    patch_sims: List[PatchSimilarityResult] = field(default_factory=list)


def calc_per_patch_mean_freq_band_means(
    sim_results: List[PatchSimilarityResult]
) -> np.ndarray:
    """
    Calculate mean of per-frequency-band means across all patches.
    This is fvnsim.
    Matches C++ Visqol::CalcPerPatchMeanFreqBandMeans.
    """
    all_means = np.array([r.freq_band_means for r in sim_results])
    return np.mean(all_means, axis=0)


def calc_per_patch_freq_band_quantile(
    sim_results: List[PatchSimilarityResult],
    quantile: float = 0.10,
) -> np.ndarray:
    """
    Calculate quantile of per-frequency-band means across patches.
    This is fvnsim10.
    Matches C++ Visqol::CalcPerPatchFreqBandQuantile.
    """
    num_freq_bands = len(sim_results[0].freq_band_means)
    result = np.zeros(num_freq_bands)

    for band in range(num_freq_bands):
        band_nsims = sorted([r.freq_band_means[band] for r in sim_results])
        num_in_quantile = max(1, int(len(band_nsims) * quantile))
        result[band] = np.mean(band_nsims[:num_in_quantile])

    return result


def calc_per_patch_mean_freq_band_deg_energy(
    sim_results: List[PatchSimilarityResult]
) -> np.ndarray:
    """
    Calculate mean of per-frequency-band degraded energy across patches.
    This is fvdegenergy.
    Matches C++ Visqol::CalcPerPatchMeanFreqBandDegradedEnergy.
    """
    all_energy = np.array([r.freq_band_deg_energy for r in sim_results])
    return np.mean(all_energy, axis=0)


def calc_per_patch_mean_freq_band_stddevs(
    sim_results: List[PatchSimilarityResult],
    frame_duration: float,
) -> np.ndarray:
    """
    Calculate pooled standard deviation across patches.
    This is fstdnsim.
    Matches C++ Visqol::CalcPerPatchMeanFreqBandStdDevs.

    Uses the pooled variance formula:
    https://en.wikipedia.org/wiki/Pooled_variance
    """
    num_freq_bands = len(sim_results[0].freq_band_means)

    # First compute fvnsim (global mean per band)
    fvnsim = calc_per_patch_mean_freq_band_means(sim_results)

    total_frame_count = 0
    contribution = np.zeros(num_freq_bands)

    for patch in sim_results:
        secs_in_patch = patch.ref_patch_end_time - patch.ref_patch_start_time
        frame_count = int(math.ceil(secs_in_patch / frame_duration))
        total_frame_count += frame_count

        for band in range(num_freq_bands):
            stddev = patch.freq_band_stddevs[band]
            mean = patch.freq_band_means[band]
            contribution[band] += (frame_count - 1) * stddev * stddev
            contribution[band] += frame_count * mean * mean

    if total_frame_count <= 1:
        return np.zeros(num_freq_bands)

    variance = (contribution - fvnsim * fvnsim * total_frame_count) / (
        total_frame_count - 1
    )

    # sqrt, filtering negative values due to precision
    result = np.where(variance < 0, 0.0, np.sqrt(variance))
    return result


def alter_for_similarity_extremes(vnsim: float, moslqo: float) -> float:
    """
    Handle extreme similarity cases.
    Matches C++ Visqol::AlterForSimilarityExtremes.
    """
    if vnsim < 0.15:
        return 1.0
    return moslqo


def calc_frame_duration(frame_size: int, sample_rate: int) -> float:
    """Calculate frame duration in seconds."""
    return frame_size / float(sample_rate)


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
        patch_creator,
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
            patch_creator: Patch creator (ImagePatchCreator or VadPatchCreator).
            search_window: Search window radius in patch units.
            quality_mapper: Similarity-to-quality mapper.
            disable_realignment: If True, skip fine realignment.

        Returns:
            SimilarityResult with MOS-LQO score and all intermediate data.
        """
        # Stage 1: Preprocessing - SPL matching
        deg_signal = scale_to_match_sound_pressure_level(ref_signal, deg_signal)

        # Build spectrograms
        ref_spectrogram = spect_builder.build(ref_signal, window)
        deg_spectrogram = spect_builder.build(deg_signal, window)

        # Prepare spectrograms for comparison (dB conversion + noise floor)
        ref_db, deg_db = prepare_spectrograms_for_comparison(
            ref_spectrogram, deg_spectrogram
        )

        # Stage 2: Feature selection and similarity measure
        ref_patch_indices = patch_creator.create_ref_patch_indices(
            ref_db, ref_signal, window
        )

        frame_duration = calc_frame_duration(
            int(window.size * window.overlap), ref_signal.sample_rate
        )

        ref_patches = patch_creator.create_patches_from_indices(
            ref_db, ref_patch_indices
        )

        # DP patch matching
        sim_match_info = find_most_optimal_deg_patches(
            ref_patches, ref_patch_indices, deg_db,
            frame_duration, search_window
        )

        # Fine realignment
        if not disable_realignment:
            sim_match_info = finely_align_and_recreate_patches(
                sim_match_info, ref_signal, deg_signal,
                spect_builder, window
            )

        # Aggregate statistics
        fvnsim = calc_per_patch_mean_freq_band_means(sim_match_info)
        fvnsim10 = calc_per_patch_freq_band_quantile(sim_match_info, 0.10)
        fstdnsim = calc_per_patch_mean_freq_band_stddevs(
            sim_match_info, frame_duration
        )
        fvdegenergy = calc_per_patch_mean_freq_band_deg_energy(sim_match_info)

        # Predict MOS
        moslqo = quality_mapper.predict_quality(
            fvnsim, fvnsim10, fstdnsim, fvdegenergy
        )

        # Calculate vnsim (mean of fvnsim)
        vnsim = float(np.mean(fvnsim))

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
