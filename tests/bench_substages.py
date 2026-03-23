#!/usr/bin/env python3
"""Sub-stage micro-benchmark for the two main bottlenecks."""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISQOL_ROOT = os.path.dirname(ROOT)
TESTDATA = os.path.join(VISQOL_ROOT, "testdata", "conformance_testdata_subset")
sys.path.insert(0, ROOT)

REF = os.path.join(TESTDATA, "guitar48_stereo.wav")
DEG = os.path.join(TESTDATA, "guitar48_stereo_64kbps_aac.wav")

from visqol.alignment import align_and_truncate, globally_align
from visqol.analysis_window import AnalysisWindow
from visqol.audio_utils import load_as_mono, scale_to_match_sound_pressure_level
from visqol.gammatone import (
    GammatoneFilterBank,
    GammatoneSpectrogramBuilder,
    make_erb_filters,
    prepare_spectrograms_for_comparison,
)
from visqol.nsim import (
    GAUSSIAN_WINDOW,
    _valid_2d_conv_with_boundary,
    measure_patch_similarity,
)
from visqol.numba_accel import has_numba, warmup
from visqol.patch_creator import ImagePatchCreator
from visqol.patch_selector import (
    find_most_optimal_deg_patches,
    slice_signal,
)
from visqol.visqol_core import calc_frame_duration

if has_numba():
    warmup()

ref_signal = load_as_mono(REF)
deg_signal = load_as_mono(DEG)
window = AnalysisWindow(48000, 0.25)
spect_builder = GammatoneSpectrogramBuilder(32, 50.0, speech_mode=False)
patch_creator = ImagePatchCreator(30)

# ================================================================
# Part 1  Gammatone 滤波 子阶段分解
# ================================================================
print("=" * 70)
print("  Gammatone 滤波 子阶段分解")
print("=" * 70)

sig = ref_signal.data
hop_size = int(window.size * window.overlap)
num_cols = 1 + int(np.floor((len(sig) - window.size) / hop_size))

erb_result = make_erb_filters(48000, 32, 50.0, 24000.0)
filter_coeffs = erb_result.filter_coeffs[:, ::-1]

fb = GammatoneFilterBank(32, 50.0)

# Total gammatone time
t0 = time.perf_counter()
out_matrix = np.zeros((32, num_cols))
for i in range(num_cols):
    start = i * hop_size
    frame = sig[start : start + window.size].copy()
    windowed_frame = window.apply_hann_window(frame)
    fb.reset_conditions()
    filtered = fb.apply_filter(windowed_frame, filter_coeffs)
    out_matrix[:, i] = np.sqrt(np.mean(filtered**2, axis=1))
t1 = time.perf_counter()
total_gammatone = t1 - t0
print(f"  帧循环总耗时:        {total_gammatone * 1000:.1f}ms  ({num_cols} 帧)")
print(f"    每帧平均:          {total_gammatone / num_cols * 1000:.3f}ms")

# Per-frame breakdown
N = min(50, num_cols)
t_win = t_reset = t_filt = t_rms = 0.0
for i in range(N):
    start = i * hop_size
    frame = sig[start : start + window.size].copy()
    ta = time.perf_counter()
    wf = window.apply_hann_window(frame)
    tb = time.perf_counter()
    fb.reset_conditions()
    tc = time.perf_counter()
    filtered = fb.apply_filter(wf, filter_coeffs)
    td = time.perf_counter()
    _ = np.sqrt(np.mean(filtered**2, axis=1))
    te = time.perf_counter()
    t_win += tb - ta
    t_reset += tc - tb
    t_filt += td - tc
    t_rms += te - td

per_total = t_win + t_reset + t_filt + t_rms
print(f"  单帧分解 (avg {N} 帧):")
print(f"    Hann窗:           {t_win / N * 1000:.3f}ms  ({t_win / per_total * 100:.0f}%)")
print(f"    reset_conditions: {t_reset / N * 1000:.3f}ms  ({t_reset / per_total * 100:.0f}%)")
print(f"    apply_filter:     {t_filt / N * 1000:.3f}ms  ({t_filt / per_total * 100:.0f}%)")
print(f"    RMS:              {t_rms / N * 1000:.3f}ms  ({t_rms / per_total * 100:.0f}%)")
print("  每帧 lfilter 调用:    32 bands × 4 stages = 128 次")
print(f"  总 lfilter 调用:     {num_cols} × 128 = {num_cols * 128} 次")

# ================================================================
# Part 2  精细对齐 + NSIM 子阶段分解
# ================================================================
print()
print("=" * 70)
print("  精细对齐 + NSIM 子阶段分解")
print("=" * 70)

aligned_deg, _ = globally_align(ref_signal, deg_signal)
deg_sig = scale_to_match_sound_pressure_level(ref_signal, aligned_deg)
ref_spect = spect_builder.build(ref_signal, window)
deg_spect = spect_builder.build(deg_sig, window)
ref_db, deg_db = prepare_spectrograms_for_comparison(ref_spect, deg_spect)
ref_pi = patch_creator.create_ref_patch_indices(ref_db, ref_signal, window)
fd = calc_frame_duration(int(window.size * window.overlap), 48000)
ref_patches = patch_creator.create_patches_from_indices(ref_db, ref_pi)
sim_info = find_most_optimal_deg_patches(ref_patches, ref_pi, deg_db, fd, 60)

num_patches = len(sim_info)
print(f"  Patch 总数: {num_patches}")

t_slice = t_align = t_spect = t_nsim = 0.0
count = 0
for sim_result in sim_info:
    if sim_result.deg_patch_start_time == sim_result.deg_patch_end_time == 0.0:
        continue
    ta = time.perf_counter()
    ref_audio = slice_signal(
        ref_signal, sim_result.ref_patch_start_time, sim_result.ref_patch_end_time
    )
    deg_audio = slice_signal(
        deg_sig, sim_result.deg_patch_start_time, sim_result.deg_patch_end_time
    )
    tb = time.perf_counter()
    try:
        ra, da, lag = align_and_truncate(ref_audio, deg_audio)
    except Exception:
        continue
    tc = time.perf_counter()
    if len(ra.data) <= window.size or len(da.data) <= window.size:
        continue
    try:
        rs = spect_builder.build(ra, window)
        ds = spect_builder.build(da, window)
    except Exception:
        continue
    td = time.perf_counter()
    rd, dd = prepare_spectrograms_for_comparison(rs, ds)
    ns = measure_patch_similarity(rd, dd)
    te = time.perf_counter()
    t_slice += tb - ta
    t_align += tc - tb
    t_spect += td - tc
    t_nsim += te - td
    count += 1

total_fine = t_slice + t_align + t_spect + t_nsim
print(f"  处理的 Patch 数: {count}")
print(f"  总耗时:             {total_fine * 1000:.1f}ms")
print("  子阶段分解:")
print(f"    音频切片:           {t_slice * 1000:.1f}ms  ({t_slice / total_fine * 100:.1f}%)")
print(
    f"    精细对齐(Hilbert+FFT): {t_align * 1000:.1f}ms  ({t_align / total_fine * 100:.1f}%)"
)
print(
    f"    重建频谱图(Gammatone): {t_spect * 1000:.1f}ms  ({t_spect / total_fine * 100:.1f}%)"
)
print(f"    NSIM 计算:         {t_nsim * 1000:.1f}ms  ({t_nsim / total_fine * 100:.1f}%)")
print()
print("  每 Patch 平均:")
print(f"    精细对齐:           {t_align / count * 1000:.2f}ms")
print(f"    重建频谱图:         {t_spect / count * 1000:.2f}ms")
print(f"    NSIM:              {t_nsim / count * 1000:.2f}ms")

# ================================================================
# Part 3  精细对齐内部 — Hilbert vs FFT xcorr 分拆
# ================================================================
print()
print("=" * 70)
print("  精细对齐内部分拆 (单 Patch 级)")
print("=" * 70)

from visqol.signal_utils import find_best_lag, upper_envelope

# pick one valid patch
sr = sim_info[0]
ref_au = slice_signal(ref_signal, sr.ref_patch_start_time, sr.ref_patch_end_time)
deg_au = slice_signal(deg_sig, sr.deg_patch_start_time, sr.deg_patch_end_time)
print(f"  Patch 信号长度: ref={len(ref_au.data)}, deg={len(deg_au.data)} samples")

t_hilbert = 0
t_xcorr = 0
REPS = 200
for _ in range(REPS):
    ta = time.perf_counter()
    re = upper_envelope(ref_au.data)
    de = upper_envelope(deg_au.data)
    tb = time.perf_counter()
    bl = find_best_lag(re, de)
    tc = time.perf_counter()
    t_hilbert += tb - ta
    t_xcorr += tc - tb

print(f"  avg {REPS} reps:")
print(f"    Hilbert (×2):     {t_hilbert / REPS * 1000:.3f}ms")
print(f"    FFT xcorr:        {t_xcorr / REPS * 1000:.3f}ms")

# ================================================================
# Part 4  NSIM 内部 — conv2d 开销
# ================================================================
print()
print("=" * 70)
print("  NSIM 内部分拆")
print("=" * 70)

# use a realistic patch
rs2 = spect_builder.build(ra, window)
ds2 = spect_builder.build(da, window)
rd2, dd2 = prepare_spectrograms_for_comparison(rs2, ds2)
print(f"  Patch shape: {rd2.shape}")

w = GAUSSIAN_WINDOW
REPS = 500
t_conv = t_arith = 0.0
for _ in range(REPS):
    ta = time.perf_counter()
    mu_r = _valid_2d_conv_with_boundary(w, rd2)
    mu_d = _valid_2d_conv_with_boundary(w, dd2)
    s_rr = _valid_2d_conv_with_boundary(w, rd2 * rd2)
    s_dd = _valid_2d_conv_with_boundary(w, dd2 * dd2)
    s_rd = _valid_2d_conv_with_boundary(w, rd2 * dd2)
    tb = time.perf_counter()
    sig_r_sq = s_rr - mu_r * mu_r
    sig_d_sq = s_dd - mu_d * mu_d
    sig_rd_ = s_rd - mu_r * mu_d
    intensity = (2 * mu_r * mu_d + 0.0001) / (mu_r * mu_r + mu_d * mu_d + 0.0001)
    sn = sig_rd_ + 0.00045
    sd = np.where(
        sig_r_sq * sig_d_sq < 0, 0.00045, np.sqrt(np.abs(sig_r_sq * sig_d_sq)) + 0.00045
    )
    sim = intensity * (sn / sd)
    fb_means = np.mean(sim, axis=1)
    tc = time.perf_counter()
    t_conv += tb - ta
    t_arith += tc - tb

print(f"  avg {REPS} reps:")
print(f"    5× conv2d:        {t_conv / REPS * 1000:.3f}ms")
print(f"    算术运算:           {t_arith / REPS * 1000:.3f}ms")
print()

# ================================================================
# Summary
# ================================================================
print("=" * 70)
print("  总结")
print("=" * 70)
print()
print("  瓶颈 1 — Gammatone 滤波 (~2.9s, 47.7%)")
print(f"    → 主因: {num_cols}×128 = {num_cols * 128} 次 scipy.lfilter 调用")
print(f"    → apply_filter (IIR级联) 占帧内 ~{t_filt / per_total * 100:.0f}% 时间")
print()
print("  瓶颈 2 — 精细对齐 + NSIM (~2.6s, 43.0%)")
print(f"    → 对 {count} 个 Patch 逐一做: Hilbert+对齐 + 重建频谱图 + NSIM")
print(f"    → 重建频谱图占 {t_spect / total_fine * 100:.1f}% (本质上又做了一轮 Gammatone)")
print(f"    → 精细对齐占 {t_align / total_fine * 100:.1f}% (Hilbert变换 + FFT互相关)")
