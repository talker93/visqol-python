#!/usr/bin/env python3
"""Test Numba parallel with different threading backends."""

import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISQOL_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from numba import config as numba_config
from numba import njit, prange

print(f"Numba threading layer: {numba_config.THREADING_LAYER}")


@njit(parallel=True)
def test_parallel(n):
    out = np.zeros(n)
    for i in prange(n):
        s = 0.0
        for j in range(1000):
            s += np.sin(float(i + j))
        out[i] = s
    return out


print("Compiling parallel function...")
r = test_parallel(100)
print(f"Result sum: {r.sum():.4f}")
print("✅ Parallel works!")

# Now test the actual Gammatone parallel variant
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

signal = load_as_mono(REF_FILE)
window = AnalysisWindow(signal.sample_rate, OVERLAP)
sr = signal.sample_rate
sig = signal.data
max_freq = sr / 2.0
hop_size = int(window.size * window.overlap)
num_cols = 1 + int(np.floor((len(sig) - window.size) / hop_size))

erb_result = make_erb_filters(sr, NUM_BANDS, MINIMUM_FREQ, max_freq)
fc = erb_result.filter_coeffs[:, ::-1]
A0 = fc[0]
A11 = fc[1]
A12 = fc[2]
A13 = fc[3]
A14 = fc[4]
A2 = fc[5]
B0 = fc[6]
B1_c = fc[7]
B2_c = fc[8]
gain = fc[9]

b1 = np.column_stack([A0 / gain, A11 / gain, A2 / gain])
b2 = np.column_stack([A0, A12, A2])
b3 = np.column_stack([A0, A13, A2])
b4 = np.column_stack([A0, A14, A2])
b_stages = np.ascontiguousarray(np.stack([b1, b2, b3, b4], axis=0), dtype=np.float64)
a_denom = np.ascontiguousarray(np.column_stack([B0, B1_c, B2_c]), dtype=np.float64)
hann = np.ascontiguousarray(window.hann_window, dtype=np.float64)
sig_c = np.ascontiguousarray(sig, dtype=np.float64)

# Baseline
print("\nBaseline (serial)...")
_ = _gammatone_spectrogram_numba(
    sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
)
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    res_a = _gammatone_spectrogram_numba(
        sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
    )
    times.append(time.perf_counter() - t0)
t_a = np.mean(times)
print(f"  Serial: {t_a * 1000:.1f}ms")


# Parallel frames
@njit(cache=False, parallel=True)
def _gspec_pframes(sig, hw, bs, ad, ws, hs, nb, nc):
    out = np.zeros((nb, nc), dtype=np.float64)
    for i in prange(nc):
        start = i * hs
        f = np.empty(ws, dtype=np.float64)
        for j in range(ws):
            f[j] = sig[start + j] * hw[j]
        for c in range(nb):
            a1 = ad[c, 1]
            a2 = ad[c, 2]
            buf = np.empty(ws, dtype=np.float64)
            b0 = bs[0, c, 0]
            b1v = bs[0, c, 1]
            b2v = bs[0, c, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(ws):
                x = f[k]
                y = b0 * x + z0
                z0 = b1v * x - a1 * y + z1
                z1 = b2v * x - a2 * y
                buf[k] = y
            for st in range(1, 4):
                b0 = bs[st, c, 0]
                b1v = bs[st, c, 1]
                b2v = bs[st, c, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(ws):
                    x = buf[k]
                    y = b0 * x + z0
                    z0 = b1v * x - a1 * y + z1
                    z1 = b2v * x - a2 * y
                    buf[k] = y
            s = 0.0
            for k in range(ws):
                v = buf[k]
                s += v * v
            out[c, i] = np.sqrt(s / ws)
    return out


print("\nParallel frames...")
try:
    _ = _gspec_pframes(
        sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
    )
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        res_c = _gspec_pframes(
            sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
        )
        times.append(time.perf_counter() - t0)
    t_c = np.mean(times)
    err = float(np.max(np.abs(res_c - res_a)))
    print(f"  Parallel frames: {t_c * 1000:.1f}ms  → {t_a / t_c:.2f}x  err={err:.2e}")
except Exception as e:
    print(f"  FAILED: {e}")


# Parallel frames + fastmath
@njit(cache=False, parallel=True, fastmath=True)
def _gspec_pframes_fm(sig, hw, bs, ad, ws, hs, nb, nc):
    out = np.zeros((nb, nc), dtype=np.float64)
    for i in prange(nc):
        start = i * hs
        f = np.empty(ws, dtype=np.float64)
        for j in range(ws):
            f[j] = sig[start + j] * hw[j]
        for c in range(nb):
            a1 = ad[c, 1]
            a2 = ad[c, 2]
            buf = np.empty(ws, dtype=np.float64)
            b0 = bs[0, c, 0]
            b1v = bs[0, c, 1]
            b2v = bs[0, c, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(ws):
                x = f[k]
                y = b0 * x + z0
                z0 = b1v * x - a1 * y + z1
                z1 = b2v * x - a2 * y
                buf[k] = y
            for st in range(1, 4):
                b0 = bs[st, c, 0]
                b1v = bs[st, c, 1]
                b2v = bs[st, c, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(ws):
                    x = buf[k]
                    y = b0 * x + z0
                    z0 = b1v * x - a1 * y + z1
                    z1 = b2v * x - a2 * y
                    buf[k] = y
            s = 0.0
            for k in range(ws):
                v = buf[k]
                s += v * v
            out[c, i] = np.sqrt(s / ws)
    return out


print("\nParallel frames + fastmath...")
try:
    _ = _gspec_pframes_fm(
        sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
    )
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        res_e = _gspec_pframes_fm(
            sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
        )
        times.append(time.perf_counter() - t0)
    t_e = np.mean(times)
    err = float(np.max(np.abs(res_e - res_a)))
    print(f"  Parallel+fastmath: {t_e * 1000:.1f}ms  → {t_a / t_e:.2f}x  err={err:.2e}")
except Exception as e:
    print(f"  FAILED: {e}")


# Fastmath only
@njit(cache=False, fastmath=True)
def _gspec_fm(sig, hw, bs, ad, ws, hs, nb, nc):
    out = np.zeros((nb, nc), dtype=np.float64)
    for i in range(nc):
        start = i * hs
        f = np.empty(ws, dtype=np.float64)
        for j in range(ws):
            f[j] = sig[start + j] * hw[j]
        for c in range(nb):
            a1 = ad[c, 1]
            a2 = ad[c, 2]
            buf = np.empty(ws, dtype=np.float64)
            b0 = bs[0, c, 0]
            b1v = bs[0, c, 1]
            b2v = bs[0, c, 2]
            z0 = 0.0
            z1 = 0.0
            for k in range(ws):
                x = f[k]
                y = b0 * x + z0
                z0 = b1v * x - a1 * y + z1
                z1 = b2v * x - a2 * y
                buf[k] = y
            for st in range(1, 4):
                b0 = bs[st, c, 0]
                b1v = bs[st, c, 1]
                b2v = bs[st, c, 2]
                z0 = 0.0
                z1 = 0.0
                for k in range(ws):
                    x = buf[k]
                    y = b0 * x + z0
                    z0 = b1v * x - a1 * y + z1
                    z1 = b2v * x - a2 * y
                    buf[k] = y
            s = 0.0
            for k in range(ws):
                v = buf[k]
                s += v * v
            out[c, i] = np.sqrt(s / ws)
    return out


print("\nFastmath only...")
_ = _gspec_fm(sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols)
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    res_d = _gspec_fm(
        sig_c, hann, b_stages, a_denom, window.size, hop_size, NUM_BANDS, num_cols
    )
    times.append(time.perf_counter() - t0)
t_d = np.mean(times)
err = float(np.max(np.abs(res_d - res_a)))
print(f"  Fastmath: {t_d * 1000:.1f}ms  → {t_a / t_d:.2f}x  err={err:.2e}")

print("\n" + "=" * 60)
print("Done!")
