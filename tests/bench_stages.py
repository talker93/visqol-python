#!/usr/bin/env python3
"""
Performance breakdown by pipeline stage.

Instruments each stage of the ViSQOL pipeline and prints a structured table
comparable to the C++ version estimates.
"""

import os
import sys
import time

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISQOL_ROOT = os.path.dirname(ROOT)
TESTDATA = os.path.join(VISQOL_ROOT, "testdata", "conformance_testdata_subset")

sys.path.insert(0, ROOT)

REF_FILE = os.path.join(TESTDATA, "guitar48_stereo.wav")
DEG_FILE = os.path.join(TESTDATA, "guitar48_stereo_64kbps_aac.wav")

# ── imports (after path setup) ─────────────────────────────────────────
from visqol.alignment import globally_align
from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import load_as_mono, scale_to_match_sound_pressure_level
from visqol.gammatone import (
    GammatoneSpectrogramBuilder,
    prepare_spectrograms_for_comparison,
)
from visqol.numba_accel import has_numba, warmup
from visqol.patch_creator import ImagePatchCreator
from visqol.patch_selector import (
    find_most_optimal_deg_patches,
    finely_align_and_recreate_patches,
)
from visqol.quality_mapper import SvrSimilarityToQualityMapper
from visqol.visqol_core import (
    alter_for_similarity_extremes,
    calc_frame_duration,
    calc_per_patch_freq_band_quantile,
    calc_per_patch_mean_freq_band_deg_energy,
    calc_per_patch_mean_freq_band_means,
    calc_per_patch_mean_freq_band_stddevs,
)

# ── constants ──────────────────────────────────────────────────────────
NUM_BANDS = 32
MINIMUM_FREQ = 50.0
OVERLAP = 0.25
PATCH_SIZE = 30
SEARCH_WINDOW = 60
RUNS = 3  # repeat full pipeline and take average


def find_default_model():
    """Locate the default SVR model file."""
    candidates = [
        os.path.join(ROOT, "visqol", "model", "libsvm_nu_svr_model.txt"),
        os.path.join(VISQOL_ROOT, "model", "libsvm_nu_svr_model.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Cannot find SVR model file")


def run_instrumented(
    ref_signal, deg_signal, spect_builder, window, patch_creator, quality_mapper
):
    """Run the full pipeline with per-stage timing."""
    timings = {}

    # ── Stage 0: Global alignment + FFT ────────────────────────────────
    t0 = time.perf_counter()
    aligned_deg, _lag = globally_align(ref_signal, deg_signal)
    deg_sig = aligned_deg
    t1 = time.perf_counter()
    timings["global_align"] = t1 - t0

    # ── Stage 1: SPL matching ──────────────────────────────────────────
    t0 = time.perf_counter()
    deg_sig = scale_to_match_sound_pressure_level(ref_signal, deg_sig)
    t1 = time.perf_counter()
    timings["spl_match"] = t1 - t0

    # ── Stage 2: Gammatone filterbank (both ref + deg) ─────────────────
    t0 = time.perf_counter()
    ref_spect = spect_builder.build(ref_signal, window)
    deg_spect = spect_builder.build(deg_sig, window)
    t1 = time.perf_counter()
    timings["gammatone"] = t1 - t0

    # ── Stage 3: Spectrogram post-processing (dB + noise floor) ────────
    t0 = time.perf_counter()
    ref_db, deg_db = prepare_spectrograms_for_comparison(ref_spect, deg_spect)
    t1 = time.perf_counter()
    timings["spect_postproc"] = t1 - t0

    # ── Stage 4: Patch creation ────────────────────────────────────────
    t0 = time.perf_counter()
    ref_patch_indices = patch_creator.create_ref_patch_indices(
        ref_db,
        ref_signal,
        window,
    )
    frame_duration = calc_frame_duration(
        int(window.size * window.overlap),
        ref_signal.sample_rate,
    )
    ref_patches = patch_creator.create_patches_from_indices(
        ref_db,
        ref_patch_indices,
    )
    t1 = time.perf_counter()
    timings["patch_create"] = t1 - t0

    # ── Stage 5: DP Patch matching ─────────────────────────────────────
    t0 = time.perf_counter()
    sim_match_info = find_most_optimal_deg_patches(
        ref_patches,
        ref_patch_indices,
        deg_db,
        frame_duration,
        SEARCH_WINDOW,
    )
    t1 = time.perf_counter()
    timings["dp_patch_match"] = t1 - t0

    # ── Stage 6: Fine realignment + NSIM ───────────────────────────────
    t0 = time.perf_counter()
    sim_match_info = finely_align_and_recreate_patches(
        sim_match_info,
        ref_signal,
        deg_sig,
        spect_builder,
        window,
    )
    t1 = time.perf_counter()
    timings["fine_align_nsim"] = t1 - t0

    # ── Stage 7: Aggregate statistics ──────────────────────────────────
    t0 = time.perf_counter()
    fvnsim = calc_per_patch_mean_freq_band_means(sim_match_info)
    fvnsim10 = calc_per_patch_freq_band_quantile(sim_match_info, 0.10)
    fstdnsim = calc_per_patch_mean_freq_band_stddevs(sim_match_info, frame_duration)
    fvdegenergy = calc_per_patch_mean_freq_band_deg_energy(sim_match_info)
    t1 = time.perf_counter()
    timings["aggregate"] = t1 - t0

    # ── Stage 8: SVR prediction ────────────────────────────────────────
    t0 = time.perf_counter()
    moslqo = quality_mapper.predict_quality(fvnsim, fvnsim10, fstdnsim, fvdegenergy)
    vnsim = float(np.mean(fvnsim))
    moslqo = alter_for_similarity_extremes(vnsim, moslqo)
    t1 = time.perf_counter()
    timings["svr_predict"] = t1 - t0

    return timings, moslqo


def main():
    print("=" * 72)
    print("   ViSQOL Python — Pipeline Stage Performance Breakdown")
    print("=" * 72)
    print()

    # ── setup ──────────────────────────────────────────────────────────
    numba_on = has_numba()
    print(f"Numba available: {numba_on}")
    if numba_on:
        print("Warming up Numba JIT …")
        warmup()
        print()

    ref_signal = load_as_mono(REF_FILE)
    deg_signal = load_as_mono(DEG_FILE)
    audio_dur = ref_signal.duration
    print(f"Audio: guitar48_stereo  duration={audio_dur:.2f}s  SR={ref_signal.sample_rate}")
    print()

    model_path = find_default_model()
    spect_builder = GammatoneSpectrogramBuilder(NUM_BANDS, MINIMUM_FREQ, speech_mode=False)
    window = AnalysisWindow(ref_signal.sample_rate, OVERLAP)
    patch_creator = ImagePatchCreator(PATCH_SIZE)
    quality_mapper = SvrSimilarityToQualityMapper(model_path)
    quality_mapper.init()

    # ── warm-up run ────────────────────────────────────────────────────
    print("Warm-up run …")
    _, mos = run_instrumented(
        ref_signal, deg_signal, spect_builder, window, patch_creator, quality_mapper
    )
    print(f"  MOS-LQO = {mos:.4f}")
    print()

    # ── timed runs ─────────────────────────────────────────────────────
    all_timings = []
    for run_idx in range(RUNS):
        timings, mos = run_instrumented(
            ref_signal, deg_signal, spect_builder, window, patch_creator, quality_mapper
        )
        all_timings.append(timings)
        total = sum(timings.values())
        print(
            f"  Run {run_idx + 1}: total={total:.3f}s  RTF={total / audio_dur:.4f}  MOS={mos:.4f}"
        )

    # ── average ────────────────────────────────────────────────────────
    avg = {}
    for key in all_timings[0]:
        avg[key] = np.mean([t[key] for t in all_timings])

    total_avg = sum(avg.values())

    # ── C++ estimates (for 12s audio) ──────────────────────────────────
    cpp_estimates = {
        "global_align": 0.03,
        "spl_match": 0.005,
        "gammatone": 0.80,
        "spect_postproc": 0.15,
        "patch_create": 0.005,
        "dp_patch_match": 0.10,
        "fine_align_nsim": 0.05,
        "aggregate": 0.005,
        "svr_predict": 0.01,
    }
    cpp_total = sum(cpp_estimates.values())

    # ── display ────────────────────────────────────────────────────────
    LABELS = {
        "global_align": "全局对齐 / FFT",
        "spl_match": "SPL 匹配",
        "gammatone": "Gammatone 滤波",
        "spect_postproc": "频谱图后处理",
        "patch_create": "Patch 创建",
        "dp_patch_match": "DP Patch 匹配",
        "fine_align_nsim": "精细对齐 + NSIM",
        "aggregate": "统计聚合",
        "svr_predict": "SVR 预测",
    }

    print()
    print("=" * 72)
    print(f"  阶段性能分解  (avg of {RUNS} runs, audio={audio_dur:.1f}s)")
    print("=" * 72)
    print()
    header = f"{'阶段':<20s} {'Python耗时':>10s} {'C++耗时(估)':>11s} {'差距倍数':>8s} {'Python占比':>10s}"
    print(header)
    print("-" * len(header) + "-" * 10)

    for key in LABELS:
        py_t = avg[key]
        cpp_t = cpp_estimates[key]
        ratio = py_t / cpp_t if cpp_t > 0 else float("inf")
        pct = py_t / total_avg * 100
        label = LABELS[key]
        print(f"{label:<20s} {py_t:>9.3f}s {cpp_t:>10.3f}s {ratio:>7.1f}x {pct:>9.1f}%")

    print("-" * len(header) + "-" * 10)
    ratio_total = total_avg / cpp_total
    print(
        f"{'总计':<20s} {total_avg:>9.3f}s {cpp_total:>10.3f}s {ratio_total:>7.1f}x {'100.0%':>10s}"
    )

    print()
    print(f"  Python RTF = {total_avg / audio_dur:.4f}")
    print(f"  C++ RTF (估) = {cpp_total / audio_dur:.4f}")
    print(f"  MOS-LQO = {mos:.4f}")
    print()


if __name__ == "__main__":
    main()
