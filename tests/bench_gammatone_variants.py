#!/usr/bin/env python3
"""
Benchmark different Gammatone IIR kernel variants to find optimal strategy.
Runs each variant in isolation to avoid parallel-related crashes.
"""

import multiprocessing
import os
import sys
import time

# Must set threading layer BEFORE importing numba
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
os.environ.setdefault("NUMBA_NUM_THREADS", str(multiprocessing.cpu_count()))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISQOL_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from numba import config as numba_config
from numba import njit, prange

from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import load_as_mono
from visqol.gammatone import make_erb_filters
from visqol.numba_accel import _gammatone_spectrogram_numba

TESTDATA = os.path.join(VISQOL_ROOT, "testdata", "conformance_testdata_subset")
REF_FILE = os.path.join(TESTDATA, "guitar48_stereo.wav")

NUM_BANDS = 32
MINIMUM_FREQ = 50.0
OVERLAP = 0.25
RUNS = 5


# ── Variant D: fastmath only ─────────────────────────────────────────
@njit(cache=False, fastmath=True)
def _gammatone_spec_fastmath(
    sig, hann_window, b_stages, a_denom, window_size, hop_size, num_bands, num_cols
):
    out_matrix = np.zeros((num_bands, num_cols), dtype=np.float64)
    for i in range(num_cols):
        start = i * hop_size
        frame = np.empty(window_size, dtype=np.float64)
        for j in range(window_size):
            frame[j] = sig[start + j] * hann_window[j]
        for chan in range(num_bands):
            a1 = a_denom[chan, 1]
            a2 = a_denom[chan, 2]
            buf = np.empty(window_size, dtype=np.float64)
            b0 = b_stages[0, chan, 0]
            b1 = b_stages[0, chan, 1]
            b2 = b_stages[0, chan, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(window_size):
                x = frame[k]
                y = b0 * x + z0
                z0 = b1 * x - a1 * y + z1
                z1 = b2 * x - a2 * y
                buf[k] = y
            for stage in range(1, 4):
                b0 = b_stages[stage, chan, 0]
                b1 = b_stages[stage, chan, 1]
                b2 = b_stages[stage, chan, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(window_size):
                    x = buf[k]
                    y = b0 * x + z0
                    z0 = b1 * x - a1 * y + z1
                    z1 = b2 * x - a2 * y
                    buf[k] = y
            s = 0.0
            for k in range(window_size):
                v = buf[k]
                s += v * v
            out_matrix[chan, i] = np.sqrt(s / window_size)
    return out_matrix


# ── Variant C: prange over frames ────────────────────────────────────
@njit(cache=False, parallel=True)
def _gammatone_spec_parallel_frames(
    sig, hann_window, b_stages, a_denom, window_size, hop_size, num_bands, num_cols
):
    out_matrix = np.zeros((num_bands, num_cols), dtype=np.float64)
    for i in prange(num_cols):
        start = i * hop_size
        frame = np.empty(window_size, dtype=np.float64)
        for j in range(window_size):
            frame[j] = sig[start + j] * hann_window[j]
        for chan in range(num_bands):
            a1 = a_denom[chan, 1]
            a2 = a_denom[chan, 2]
            buf = np.empty(window_size, dtype=np.float64)
            b0 = b_stages[0, chan, 0]
            b1 = b_stages[0, chan, 1]
            b2 = b_stages[0, chan, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(window_size):
                x = frame[k]
                y = b0 * x + z0
                z0 = b1 * x - a1 * y + z1
                z1 = b2 * x - a2 * y
                buf[k] = y
            for stage in range(1, 4):
                b0 = b_stages[stage, chan, 0]
                b1 = b_stages[stage, chan, 1]
                b2 = b_stages[stage, chan, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(window_size):
                    x = buf[k]
                    y = b0 * x + z0
                    z0 = b1 * x - a1 * y + z1
                    z1 = b2 * x - a2 * y
                    buf[k] = y
            s = 0.0
            for k in range(window_size):
                v = buf[k]
                s += v * v
            out_matrix[chan, i] = np.sqrt(s / window_size)
    return out_matrix


# ── Variant E: prange(frames) + fastmath ─────────────────────────────
@njit(cache=False, parallel=True, fastmath=True)
def _gammatone_spec_parallel_frames_fastmath(
    sig, hann_window, b_stages, a_denom, window_size, hop_size, num_bands, num_cols
):
    out_matrix = np.zeros((num_bands, num_cols), dtype=np.float64)
    for i in prange(num_cols):
        start = i * hop_size
        frame = np.empty(window_size, dtype=np.float64)
        for j in range(window_size):
            frame[j] = sig[start + j] * hann_window[j]
        for chan in range(num_bands):
            a1 = a_denom[chan, 1]
            a2 = a_denom[chan, 2]
            buf = np.empty(window_size, dtype=np.float64)
            b0 = b_stages[0, chan, 0]
            b1 = b_stages[0, chan, 1]
            b2 = b_stages[0, chan, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(window_size):
                x = frame[k]
                y = b0 * x + z0
                z0 = b1 * x - a1 * y + z1
                z1 = b2 * x - a2 * y
                buf[k] = y
            for stage in range(1, 4):
                b0 = b_stages[stage, chan, 0]
                b1 = b_stages[stage, chan, 1]
                b2 = b_stages[stage, chan, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(window_size):
                    x = buf[k]
                    y = b0 * x + z0
                    z0 = b1 * x - a1 * y + z1
                    z1 = b2 * x - a2 * y
                    buf[k] = y
            s = 0.0
            for k in range(window_size):
                v = buf[k]
                s += v * v
            out_matrix[chan, i] = np.sqrt(s / window_size)
    return out_matrix


def prepare_data():
    signal = load_as_mono(REF_FILE)
    window = AnalysisWindow(signal.sample_rate, OVERLAP)
    sr = signal.sample_rate
    sig = signal.data
    max_freq = sr / 2.0
    hop_size = int(window.size * window.overlap)
    num_cols = 1 + int(np.floor((len(sig) - window.size) / hop_size))

    erb_result = make_erb_filters(sr, NUM_BANDS, MINIMUM_FREQ, max_freq)
    filter_coeffs = erb_result.filter_coeffs[:, ::-1]

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

    return (sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols)


def bench_one(name, func, args, baseline_result=None, baseline_time=None):
    """Benchmark a single variant."""
    try:
        # Warmup / JIT compile
        result = func(*args)

        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            func(*args)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = np.mean(times)
        std = np.std(times)

        if baseline_result is not None:
            max_err = float(np.max(np.abs(result - baseline_result)))
        else:
            max_err = 0.0

        speedup = baseline_time / avg if baseline_time is not None else 1.0

        print(
            f"  {name:<42s} {avg * 1000:>7.1f}ms  ±{std * 1000:>5.1f}ms  {speedup:>5.2f}x  err={max_err:.2e}"
        )
        return avg, result
    except Exception as e:
        print(f"  {name:<42s}  FAILED: {e}")
        return None, None


def main():
    print("=" * 72)
    print("   Gammatone IIR Kernel — Variant Comparison")
    print("=" * 72)
    print()
    print(f"  CPU cores: {multiprocessing.cpu_count()}")
    print(f"  Numba threading: {numba_config.THREADING_LAYER}")
    print(f"  Numba threads: {os.environ.get('NUMBA_NUM_THREADS', 'default')}")
    print()

    args = prepare_data()
    sig_c, _hann, _b_stages, _a_denom, window_size, _hop_size, num_bands, num_cols = args
    print(
        f"  Audio: {len(sig_c) / 48000:.2f}s, frames={num_cols}, bands={num_bands}, window={window_size}"
    )
    print()

    # A: baseline (current implementation)
    print("  A. 当前版本 (serial, cache=True)")
    t_a, res_a = bench_one("serial", _gammatone_spectrogram_numba, args)
    print()

    # D: fastmath only
    print("  D. fastmath=True (serial)")
    t_d, _res_d = bench_one("fastmath", _gammatone_spec_fastmath, args, res_a, t_a)
    print()

    # C: parallel frames
    print("  C. prange(frames)")
    t_c, res_c = bench_one(
        "parallel frames", _gammatone_spec_parallel_frames, args, res_a, t_a
    )
    print()

    # E: parallel frames + fastmath
    print("  E. prange(frames) + fastmath")
    t_e, res_e = bench_one(
        "parallel frames + fastmath",
        _gammatone_spec_parallel_frames_fastmath,
        args,
        res_a,
        t_a,
    )
    print()

    print("=" * 72)
    print("  分析总结")
    print("=" * 72)
    print()
    if t_a:
        print(f"  基线 (A): {t_a * 1000:.1f}ms")
    if t_d:
        print(f"  fastmath (D): {t_d * 1000:.1f}ms  → {t_a / t_d:.2f}x")
    if t_c:
        print(f"  parallel frames (C): {t_c * 1000:.1f}ms  → {t_a / t_c:.2f}x")
    if t_e:
        print(f"  parallel+fastmath (E): {t_e * 1000:.1f}ms  → {t_a / t_e:.2f}x")
    print()

    # Check if parallel results are accurate enough
    if res_c is not None and res_a is not None:
        err_c = float(np.max(np.abs(res_c - res_a)))
        print(f"  parallel frames max error: {err_c:.2e}")
        print(f"  → {'✅ 精度无损' if err_c < 1e-10 else '⚠️ 有精度差异'}")
    if res_e is not None and res_a is not None:
        err_e = float(np.max(np.abs(res_e - res_a)))
        print(f"  parallel+fastmath max error: {err_e:.2e}")
        print(f"  → {'✅ 精度无损' if err_e < 1e-10 else '⚠️ 有精度差异 (fastmath 预期)'}")
    print()


if __name__ == "__main__":
    main()
