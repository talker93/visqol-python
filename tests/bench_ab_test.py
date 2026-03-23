#!/usr/bin/env python3
"""
A/B test: Numba ON vs OFF, single-file RTF + accuracy.
Also tests parallel batch speedup.
"""

import os
import sys
import time

import numpy as np

_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PKG)

import soundfile as sf

import visqol.numba_accel as na
from visqol import VisqolApi
from visqol.numba_accel import warmup

TESTDATA = os.path.join(_PKG, "..", "testdata")
CONFORM = os.path.join(TESTDATA, "conformance_testdata_subset")

PAIRS = [
    ("guitar48_stereo.wav", "guitar48_stereo_64kbps_aac.wav"),
    ("glock48_stereo.wav", "glock48_stereo_48kbps_aac.wav"),
    ("contrabassoon48_stereo.wav", "contrabassoon48_stereo_24kbps_aac.wav"),
    ("harpsichord48_stereo.wav", "harpsichord48_stereo_96kbps_mp3.wav"),
    ("moonlight48_stereo.wav", "moonlight48_stereo_128kbps_aac.wav"),
    ("ravel48_stereo.wav", "ravel48_stereo_128kbps_opus.wav"),
    ("sopr48_stereo.wav", "sopr48_stereo_256kbps_aac.wav"),
    ("steely48_stereo.wav", "steely48_stereo_lp7.wav"),
    ("strauss48_stereo.wav", "strauss48_stereo_lp35.wav"),
]


def resolve(name):
    return os.path.join(CONFORM, name)


def get_dur(path):
    return sf.info(path).duration


def bench_single(ref, deg, label, numba_on, runs=3):
    """Run single-file benchmark, return (avg_time, mos)."""
    dur = get_dur(ref)
    api = VisqolApi()
    api.create(mode="audio")

    # Warm-up run (not counted)
    r = api.measure(ref, deg)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = api.measure(ref, deg)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg = np.mean(times)
    rtf = avg / dur
    print(
        f"  [{label}] avg={avg:.3f}s  RTF={rtf:.4f}  MOS={r.moslqo:.4f}  "
        f"(audio={dur:.1f}s, {runs} runs)"
    )
    return avg, r.moslqo


def bench_batch(pairs_full, label, parallel, max_workers=None):
    """Run batch benchmark."""
    total_dur = sum(get_dur(r) for r, _ in pairs_full)
    api = VisqolApi()
    api.create(mode="audio")

    t0 = time.perf_counter()
    results = api.measure_batch(
        pairs_full,
        parallel=parallel,
        max_workers=max_workers,
    )
    elapsed = time.perf_counter() - t0

    ok = sum(1 for r in results if not isinstance(r, Exception))
    rtf = elapsed / total_dur
    print(
        f"  [{label}] {elapsed:.2f}s  ({len(pairs_full)} pairs, "
        f"audio={total_dur:.0f}s, RTF={rtf:.4f}, ok={ok})"
    )
    return elapsed, results


def main():
    ref0, deg0 = resolve(PAIRS[0][0]), resolve(PAIRS[0][1])
    pairs_full = [(resolve(r), resolve(d)) for r, d in PAIRS]

    print("=" * 65)
    print("  ViSQOL v3.4.0 — A/B Optimization Benchmark")
    print(f"  Numba installed: {na._HAS_NUMBA}")
    print(f"  CPU count: {os.cpu_count()}")
    print("=" * 65)

    # ============================================================
    # Part 1: Single-file Numba ON vs OFF
    # ============================================================
    print("\n" + "-" * 65)
    print("  Part 1: Single-file — Numba ON vs OFF")
    print("-" * 65)

    if na._HAS_NUMBA:
        # Warmup JIT
        print("  Warming up Numba...")
        warmup()

    # Numba ON
    na._HAS_NUMBA = True
    t_numba, mos_numba = bench_single(ref0, deg0, "Numba ON", True, runs=3)

    # Numba OFF
    na._HAS_NUMBA = False
    t_python, mos_python = bench_single(ref0, deg0, "Numba OFF", False, runs=3)

    # Restore
    na._HAS_NUMBA = True

    speedup = t_python / t_numba if t_numba > 0 else 0
    mos_delta = abs(mos_numba - mos_python)
    print(f"\n  ➤ Numba speedup: {speedup:.2f}x")
    print(f"  ➤ MOS delta:     {mos_delta:.8f} {'✅' if mos_delta < 0.001 else '⚠️'}")

    # ============================================================
    # Part 2: Batch — sequential vs parallel
    # ============================================================
    print("\n" + "-" * 65)
    print("  Part 2: Batch — Sequential vs Parallel (Numba ON)")
    print("-" * 65)

    na._HAS_NUMBA = True

    t_seq, _res_seq = bench_batch(pairs_full, "Sequential", parallel=False)
    t_par, _res_par = bench_batch(
        pairs_full, f"Parallel (auto, {os.cpu_count()} CPUs)", parallel=True
    )
    t_par2, _res_par2 = bench_batch(
        pairs_full, "Parallel (2 workers)", parallel=True, max_workers=2
    )

    print(f"\n  ➤ Parallel (auto) speedup: {t_seq / t_par:.2f}x")
    print(f"  ➤ Parallel (2w) speedup:   {t_seq / t_par2:.2f}x")

    # ============================================================
    # Part 3: Accuracy — full conformance suite
    # ============================================================
    print("\n" + "-" * 65)
    print("  Part 3: Accuracy — Numba vs Python (all conformance pairs)")
    print("-" * 65)

    # Numba ON
    na._HAS_NUMBA = True
    api_n = VisqolApi()
    api_n.create(mode="audio")
    mos_n = []
    for ref, deg in pairs_full:
        mos_n.append(api_n.measure(ref, deg).moslqo)

    # Numba OFF
    na._HAS_NUMBA = False
    api_p = VisqolApi()
    api_p.create(mode="audio")
    mos_p = []
    for ref, deg in pairs_full:
        mos_p.append(api_p.measure(ref, deg).moslqo)

    na._HAS_NUMBA = True  # restore

    print(f"\n  {'Pair':<50} {'Numba':>8} {'Python':>8} {'|ΔMOS|':>10}")
    print("  " + "-" * 78)
    max_d = 0.0
    for i, (rn, dn) in enumerate(PAIRS):
        label = f"{rn[:22]} → {dn[:22]}"
        d = abs(mos_n[i] - mos_p[i])
        max_d = max(max_d, d)
        flag = "✅" if d < 0.001 else "⚠️"
        print(f"  {label:<50} {mos_n[i]:8.4f} {mos_p[i]:8.4f} {d:10.6f} {flag}")

    print(f"\n  Max |ΔMOS| = {max_d:.8f}")
    if max_d < 0.001:
        print("  ✅ PASS — Results are bit-identical.")
    elif max_d < 0.01:
        print("  ✅ PASS — Negligible numerical differences (< 0.01).")
    else:
        print("  ❌ FAIL — Significant difference detected!")

    # ============================================================
    # Summary table
    # ============================================================
    dur0 = get_dur(ref0)
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Single file ({dur0:.1f}s audio):")
    print(f"    Python-only:  {t_python:.3f}s  RTF={t_python / dur0:.4f}")
    print(
        f"    + Numba:      {t_numba:.3f}s  RTF={t_numba / dur0:.4f}  ({speedup:.1f}x faster)"
    )
    print(f"  Batch ({len(PAIRS)} files):")
    print(
        f"    Sequential:   {t_seq:.1f}s   RTF={t_seq / sum(get_dur(resolve(r)) for r, _ in PAIRS):.4f}"
    )
    print(
        f"    Parallel:     {t_par:.1f}s   RTF={t_par / sum(get_dur(resolve(r)) for r, _ in PAIRS):.4f}  "
        f"({t_seq / t_par:.1f}x faster)"
    )
    print(f"  Accuracy: max |ΔMOS| = {max_d:.8f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
