#!/usr/bin/env python3
"""
Gammatone filterbank micro-benchmark — break down where time is spent.

Profiles:
  1. ERB coefficient computation (make_erb_filters)
  2. Coefficient packing (np.column_stack, np.stack)
  3. Numba IIR kernel (_gammatone_spectrogram_numba)
  4. Total _build_numba overhead

Also profiles the IIR kernel's internal breakdown:
  - Hann windowing per frame
  - 4-stage IIR filtering per frame
  - RMS computation per frame
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISQOL_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import load_as_mono
from visqol.gammatone import GammatoneSpectrogramBuilder, make_erb_filters
from visqol.numba_accel import (
    _gammatone_spectrogram_numba,
    _iir_4stage_filter_frame,
    has_numba,
    warmup,
)

TESTDATA = os.path.join(VISQOL_ROOT, "testdata", "conformance_testdata_subset")
REF_FILE = os.path.join(TESTDATA, "guitar48_stereo.wav")

NUM_BANDS = 32
MINIMUM_FREQ = 50.0
OVERLAP = 0.25
RUNS = 5


def main():
    print("=" * 72)
    print("   Gammatone Filterbank — Internal Breakdown")
    print("=" * 72)
    print()

    assert has_numba(), "This benchmark requires numba"
    print("Warming up Numba JIT …")
    warmup()
    print()

    signal = load_as_mono(REF_FILE)
    window = AnalysisWindow(signal.sample_rate, OVERLAP)
    sr = signal.sample_rate
    sig = signal.data
    max_freq = sr / 2.0
    hop_size = int(window.size * window.overlap)
    num_cols = 1 + int(np.floor((len(sig) - window.size) / hop_size))

    print(f"Audio: {len(sig)} samples, SR={sr}, duration={len(sig) / sr:.2f}s")
    print(f"Window: size={window.size}, hop={hop_size}, frames={num_cols}")
    print(f"Bands: {NUM_BANDS}")
    print()

    # ── Benchmark 1: ERB coefficient computation ──────────────────────
    times_erb = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        erb_result = make_erb_filters(sr, NUM_BANDS, MINIMUM_FREQ, max_freq)
        filter_coeffs = erb_result.filter_coeffs[:, ::-1]
        t1 = time.perf_counter()
        times_erb.append(t1 - t0)
    avg_erb = np.mean(times_erb)

    # ── Benchmark 2: Coefficient packing ──────────────────────────────
    erb_result = make_erb_filters(sr, NUM_BANDS, MINIMUM_FREQ, max_freq)
    filter_coeffs = erb_result.filter_coeffs[:, ::-1]

    times_pack = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        A0 = filter_coeffs[0]
        A11 = filter_coeffs[1]
        A12 = filter_coeffs[2]
        A13 = filter_coeffs[3]
        A14 = filter_coeffs[4]
        A2 = filter_coeffs[5]
        B0 = filter_coeffs[6]
        B1 = filter_coeffs[7]
        B2 = filter_coeffs[8]
        gain = filter_coeffs[9]

        b1 = np.column_stack([A0 / gain, A11 / gain, A2 / gain])
        b2 = np.column_stack([A0, A12, A2])
        b3 = np.column_stack([A0, A13, A2])
        b4 = np.column_stack([A0, A14, A2])

        b_stages = np.ascontiguousarray(
            np.stack([b1, b2, b3, b4], axis=0),
            dtype=np.float64,
        )
        a_denom = np.ascontiguousarray(
            np.column_stack([B0, B1, B2]),
            dtype=np.float64,
        )
        hann = np.ascontiguousarray(window.hann_window, dtype=np.float64)
        sig_c = np.ascontiguousarray(sig, dtype=np.float64)
        t1 = time.perf_counter()
        times_pack.append(t1 - t0)
    avg_pack = np.mean(times_pack)

    # ── Benchmark 3: Pure Numba IIR kernel ────────────────────────────
    # (pre-packed coefficients, only the JIT function call)
    A0 = filter_coeffs[0]
    A11 = filter_coeffs[1]
    A12 = filter_coeffs[2]
    A13 = filter_coeffs[3]
    A14 = filter_coeffs[4]
    A2 = filter_coeffs[5]
    B0 = filter_coeffs[6]
    B1_c = filter_coeffs[7]
    B2_c = filter_coeffs[8]
    gain = filter_coeffs[9]

    b1 = np.column_stack([A0 / gain, A11 / gain, A2 / gain])
    b2 = np.column_stack([A0, A12, A2])
    b3 = np.column_stack([A0, A13, A2])
    b4 = np.column_stack([A0, A14, A2])
    b_stages = np.ascontiguousarray(np.stack([b1, b2, b3, b4], axis=0), dtype=np.float64)
    a_denom = np.ascontiguousarray(np.column_stack([B0, B1_c, B2_c]), dtype=np.float64)
    hann = np.ascontiguousarray(window.hann_window, dtype=np.float64)
    sig_c = np.ascontiguousarray(sig, dtype=np.float64)

    # Warmup call
    _gammatone_spectrogram_numba(
        sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
    )

    times_kernel = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _gammatone_spectrogram_numba(
            sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
        )
        t1 = time.perf_counter()
        times_kernel.append(t1 - t0)
    avg_kernel = np.mean(times_kernel)

    # ── Benchmark 4: Full _build_numba (end-to-end) ───────────────────
    builder = GammatoneSpectrogramBuilder(NUM_BANDS, MINIMUM_FREQ, speech_mode=False)
    # Warmup
    builder.build(signal, window)

    times_full = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        builder.build(signal, window)
        t1 = time.perf_counter()
        times_full.append(t1 - t0)
    avg_full = np.mean(times_full)

    # ── Benchmark 5: Single-frame IIR to understand per-frame cost ────
    frame = sig_c[: window.size].copy() * hann
    # Warmup
    _iir_4stage_filter_frame(frame, b_stages, a_denom)

    times_single_frame = []
    n_iters = 100
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _iir_4stage_filter_frame(frame, b_stages, a_denom)
        t1 = time.perf_counter()
        times_single_frame.append((t1 - t0) / n_iters)
    avg_single_frame = np.mean(times_single_frame)

    # ── Display ───────────────────────────────────────────────────────
    overhead = avg_full - avg_erb - avg_kernel
    if overhead < 0:
        overhead = avg_pack  # fallback

    print("=" * 72)
    print(f"  Gammatone Breakdown  (avg of {RUNS} runs)")
    print("=" * 72)
    print()
    print(f"  {'步骤':<35s} {'耗时':>10s} {'占比':>8s}")
    print(f"  {'-' * 55}")
    print(
        f"  {'ERB 系数计算 (make_erb_filters)':<35s} {avg_erb * 1000:>8.1f}ms {avg_erb / avg_full * 100:>7.1f}%"
    )
    print(
        f"  {'系数打包 (column_stack/stack)':<35s} {avg_pack * 1000:>8.1f}ms {avg_pack / avg_full * 100:>7.1f}%"
    )
    print(
        f"  {'Numba IIR 核心 (_gammatone_spec)':<35s} {avg_kernel * 1000:>8.1f}ms {avg_kernel / avg_full * 100:>7.1f}%"
    )
    print(
        f"  {'Python 胶水开销 (其他)':<35s} {overhead * 1000:>8.1f}ms {overhead / avg_full * 100:>7.1f}%"
    )
    print(f"  {'-' * 55}")
    print(f"  {'总计 (build 完整调用)':<35s} {avg_full * 1000:>8.1f}ms {'100.0%':>8s}")
    print()
    print(f"  单帧 IIR 滤波耗时: {avg_single_frame * 1000:.3f}ms")
    print(
        f"  帧数 × 单帧: {num_cols} × {avg_single_frame * 1000:.3f}ms = {num_cols * avg_single_frame * 1000:.1f}ms"
    )
    print(f"  实际核心耗时: {avg_kernel * 1000:.1f}ms")
    print(f"  帧循环开销: {(avg_kernel - num_cols * avg_single_frame) * 1000:.1f}ms")
    print()

    # ── Theoretical analysis ──────────────────────────────────────────
    total_samples_processed = num_cols * window.size * NUM_BANDS * 4  # 4 stages
    msamples_per_sec = total_samples_processed / avg_kernel / 1e6
    print(f"  总 IIR 采样点: {total_samples_processed:,d}")
    print(f"  吞吐量: {msamples_per_sec:.1f} M samples/s")
    print()

    # ── Parallelism analysis ──────────────────────────────────────────
    print("=" * 72)
    print("  可优化空间分析")
    print("=" * 72)
    print()
    print(f"  1. ERB 系数计算:     {avg_erb * 1000:.1f}ms — 每次 build 都重算")
    print("     → 可缓存? 如果 SR/bands/freq 不变, 系数完全相同")
    print()
    print(
        f"  2. Numba IIR 核心:   {avg_kernel * 1000:.1f}ms — 占比 {avg_kernel / avg_full * 100:.0f}%"
    )
    print("     → 当前: 串行帧循环 + 串行 band 循环")
    print("     → 可用 prange 并行化 band 维度?")
    print("     → 可用 fastmath=True?")
    print()
    print(f"  3. 系数打包:         {avg_pack * 1000:.1f}ms — 每次 build 都重新打包")
    print("     → 可缓存? 同上")
    print()


if __name__ == "__main__":
    main()
