#!/usr/bin/env python3
"""
Benchmark & accuracy test for ViSQOL v3.4.0 optimizations.

Compares:
  1. Pure-Python fallback  vs  Numba-accelerated DP
  2. Sequential batch      vs  Parallel batch (multiprocessing)
  3. Accuracy: verifies that Numba path produces identical MOS-LQO scores
     as the pure-Python path.

Usage:
    python -m tests.bench_optimization        # from visqol_python/
    python tests/bench_optimization.py        # direct run
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure visqol package is importable when running this script directly
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import numpy as np

# ---------------------------------------------------------------------------
# Test data discovery
# ---------------------------------------------------------------------------
_TESTDATA = Path(__file__).resolve().parent.parent.parent / "testdata"

# Audio-mode pairs (48 kHz stereo → mono)
_AUDIO_PAIRS: list[tuple[str, str]] = []

_CONFORMANCE = _TESTDATA / "conformance_testdata_subset"
if _CONFORMANCE.exists():
    _ref_deg_mapping = [
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
    for ref_name, deg_name in _ref_deg_mapping:
        ref = _CONFORMANCE / ref_name
        deg = _CONFORMANCE / deg_name
        if ref.exists() and deg.exists():
            _AUDIO_PAIRS.append((str(ref), str(deg)))

# Short file pair for quick single-file RTF tests
_SHORT_PAIR: tuple[str, str] | None = None
_SHORT_DIR = _TESTDATA / "short_duration" / "5_second"
if _SHORT_DIR.exists():
    ref_short = _CONFORMANCE / "guitar48_stereo.wav"
    deg_short = _SHORT_DIR / "guitar48_stereo_5_sec.wav"
    if ref_short.exists() and deg_short.exists():
        _SHORT_PAIR = (str(ref_short), str(deg_short))


def _get_audio_duration(path: str) -> float:
    """Return audio file duration in seconds."""
    import soundfile as sf

    info = sf.info(path)
    return info.duration


# =====================================================================
# Benchmark helpers
# =====================================================================


def _time_single(api, ref: str, deg: str, label: str, runs: int = 1) -> float:
    """Time a single measure() call and print RTF."""
    dur = _get_audio_duration(ref)
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = api.measure(ref, deg)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = np.mean(times)
    rtf = avg / dur
    print(
        f"  [{label}] {avg:.3f}s  (audio {dur:.1f}s, RTF={rtf:.4f}, "
        f"MOS={result.moslqo:.4f}, runs={runs})"
    )
    return avg


def _time_batch(
    api,
    pairs: list[tuple[str, str]],
    label: str,
    parallel: bool = False,
    max_workers: int | None = None,
) -> float:
    """Time measure_batch() and print aggregate RTF."""
    total_dur = sum(_get_audio_duration(r) for r, _ in pairs)
    t0 = time.perf_counter()
    results = api.measure_batch(
        pairs,
        parallel=parallel,
        max_workers=max_workers,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0

    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    rtf = elapsed / total_dur
    print(
        f"  [{label}] {elapsed:.2f}s total  ({len(pairs)} pairs, "
        f"audio {total_dur:.0f}s, RTF={rtf:.4f}, "
        f"success={len(successes)}, fail={len(failures)})"
    )
    return elapsed


# =====================================================================
# Accuracy test
# =====================================================================


def _test_accuracy(pairs: list[tuple[str, str]]) -> None:
    """
    Run the same pairs with Numba ON and OFF and compare MOS-LQO scores.
    """
    import visqol.numba_accel as _na
    from visqol import VisqolApi
    from visqol.numba_accel import has_numba as _hn

    if not _hn():
        print(
            "\n⚠️  Numba not installed — skipping accuracy comparison "
            "(only one code path available).\n"
        )
        return

    print("\n" + "=" * 65)
    print("  ACCURACY TEST: Numba vs pure-Python (same pairs)")
    print("=" * 65)

    # ----- Run with Numba -----
    api_numba = VisqolApi()
    api_numba.create(mode="audio")
    mos_numba = []
    for ref, deg in pairs:
        r = api_numba.measure(ref, deg)
        mos_numba.append(r.moslqo)

    # ----- Temporarily disable Numba -----
    orig = _na._HAS_NUMBA
    _na._HAS_NUMBA = False
    # Also need to re-import has_numba in patch_selector — it reads the flag
    # via the function, so just patching the module variable is enough.

    api_python = VisqolApi()
    api_python.create(mode="audio")
    mos_python = []
    for ref, deg in pairs:
        r = api_python.measure(ref, deg)
        mos_python.append(r.moslqo)

    # Restore
    _na._HAS_NUMBA = orig

    # ----- Compare -----
    print(f"\n  {'Pair':<55} {'Numba':>8} {'Python':>8} {'Δ MOS':>10}")
    print("  " + "-" * 83)

    max_delta = 0.0
    for i, (ref, deg) in enumerate(pairs):
        ref_name = Path(ref).stem[:25]
        deg_name = Path(deg).stem[:25]
        label = f"{ref_name} → {deg_name}"
        delta = abs(mos_numba[i] - mos_python[i])
        max_delta = max(max_delta, delta)
        flag = " ⚠️" if delta > 0.001 else " ✅"
        print(f"  {label:<55} {mos_numba[i]:8.4f} {mos_python[i]:8.4f} {delta:10.6f}{flag}")

    print(f"\n  Max |ΔMOS| = {max_delta:.8f}")
    if max_delta < 0.01:
        print("  ✅ PASS — Numba and pure-Python paths produce equivalent results.")
    elif max_delta < 0.05:
        print("  ⚠️  MARGINAL — Small numerical differences detected (< 0.05 MOS).")
    else:
        print("  ❌ FAIL — Significant accuracy difference detected!")


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    from visqol import VisqolApi
    from visqol.numba_accel import has_numba, warmup

    print("=" * 65)
    print("  ViSQOL v3.4.0 Performance & Accuracy Benchmark")
    print(f"  Numba available: {has_numba()}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  Test pairs: {len(_AUDIO_PAIRS)} conformance pairs")
    print("=" * 65)

    # Warm up Numba (compile JIT kernels)
    if has_numba():
        print("\n⏳ Warming up Numba JIT kernels...")
        t0 = time.perf_counter()
        warmup()
        print(f"  Warmup done in {time.perf_counter() - t0:.2f}s")

    # ---- Single-file RTF ----
    print("\n" + "-" * 65)
    print("  SINGLE-FILE RTF TEST")
    print("-" * 65)

    api = VisqolApi()
    api.create(mode="audio")

    if _AUDIO_PAIRS:
        ref, deg = _AUDIO_PAIRS[0]
        _time_single(api, ref, deg, "Conformance pair #1", runs=1)

    if _SHORT_PAIR:
        _time_single(api, _SHORT_PAIR[0], _SHORT_PAIR[1], "5s short pair", runs=2)

    # ---- Batch sequential vs parallel ----
    if len(_AUDIO_PAIRS) >= 2:
        print("\n" + "-" * 65)
        print("  BATCH TEST (sequential vs parallel)")
        print("-" * 65)

        # Sequential
        t_seq = _time_batch(api, _AUDIO_PAIRS, "Sequential", parallel=False)

        # Parallel (auto workers)
        t_par = _time_batch(api, _AUDIO_PAIRS, "Parallel (auto)", parallel=True)

        # Parallel (2 workers)
        t_par2 = _time_batch(
            api, _AUDIO_PAIRS, "Parallel (2 workers)", parallel=True, max_workers=2
        )

        speedup_auto = t_seq / t_par if t_par > 0 else 0
        speedup_2 = t_seq / t_par2 if t_par2 > 0 else 0
        print(f"\n  Speedup (auto workers): {speedup_auto:.2f}x")
        print(f"  Speedup (2 workers):   {speedup_2:.2f}x")

    # ---- Accuracy test ----
    test_pairs = _AUDIO_PAIRS[:5] if len(_AUDIO_PAIRS) >= 5 else _AUDIO_PAIRS
    if test_pairs:
        _test_accuracy(test_pairs)

    print("\n" + "=" * 65)
    print("  BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
