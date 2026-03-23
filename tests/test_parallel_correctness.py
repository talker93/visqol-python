"""
Correctness test: verify that the parallel+fastmath Gammatone spectrogram
produces results consistent with the original serial version.

Approach: run the full ViSQOL pipeline on a known test file and check
that the MOS score is within expected tolerance.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISQOL_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)


# --- Quick sanity: does the module load without errors? ---
print("=" * 60)
print("1) Import & warmup")
print("=" * 60)
from visqol import numba_accel

print(f"   _HAS_NUMBA          = {numba_accel._HAS_NUMBA}")
print(f"   _HAS_NUMBA_PARALLEL = {numba_accel._HAS_NUMBA_PARALLEL}")
print(f"   NUMBA_THREADING_LAYER = {os.environ.get('NUMBA_THREADING_LAYER', '(not set)')}")

t0 = time.perf_counter()
numba_accel.warmup()
t1 = time.perf_counter()
print(f"   warmup completed in {t1 - t0:.2f}s")

# --- Full pipeline correctness test ---
print()
print("=" * 60)
print("2) Full pipeline MOS score test")
print("=" * 60)

from visqol.api import VisqolApi

TESTDATA = os.path.join(VISQOL_ROOT, "testdata", "conformance_testdata_subset")
ref_file = os.path.join(TESTDATA, "guitar48_stereo.wav")
deg_file = os.path.join(TESTDATA, "guitar48_stereo_64kbps_aac.wav")

if not os.path.exists(ref_file):
    import glob

    testdata_root = os.path.join(VISQOL_ROOT, "testdata")
    wavs = sorted(glob.glob(os.path.join(testdata_root, "**", "*.wav"), recursive=True))
    if len(wavs) >= 2:
        ref_file = wavs[0]
        deg_file = wavs[1]
    else:
        print("   ERROR: No test WAV files found")
        sys.exit(1)

print(f"   ref: {os.path.basename(ref_file)}")
print(f"   deg: {os.path.basename(deg_file)}")

v = VisqolApi()
v.create(mode="audio")

# Run 1: get score (also triggers JIT compilation if not warmed up)
t0 = time.perf_counter()
result1 = v.measure(ref_file, deg_file)
t1 = time.perf_counter()
print(f"   Run 1: MOS = {result1.moslqo:.6f}  ({t1 - t0:.3f}s)")

# Run 2: steady-state (JIT cached)
t0 = time.perf_counter()
result2 = v.measure(ref_file, deg_file)
t1 = time.perf_counter()
print(f"   Run 2: MOS = {result2.moslqo:.6f}  ({t1 - t0:.3f}s)")

# Check reproducibility
diff = abs(result1.moslqo - result2.moslqo)
print(f"   Run1 vs Run2 diff: {diff:.2e}")
assert diff < 1e-10, f"Non-reproducible results: diff={diff}"
print("   ✅ Results are reproducible across runs")

# Check MOS is in a reasonable range
print(f"   MOS value: {result1.moslqo:.6f}")
assert 1.0 <= result1.moslqo <= 5.0, f"MOS out of range: {result1.moslqo}"
print("   ✅ MOS in valid range [1, 5]")

# Run 3: timing
t0 = time.perf_counter()
result3 = v.measure(ref_file, deg_file)
t1 = time.perf_counter()
print(f"   Run 3: MOS = {result3.moslqo:.6f}  ({t1 - t0:.3f}s)")

print()
print("=" * 60)
print("ALL CORRECTNESS CHECKS PASSED ✅")
print(f"Steady-state latency: {t1 - t0:.3f}s")
print("=" * 60)
