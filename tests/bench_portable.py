"""Compare 3.7.0 and current kernels on the same machine, backend and input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path(".testdata"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fft", choices=["scipy", "fftw"], default="fftw")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")

    os.environ.setdefault("NUMBA_NUM_THREADS", str(args.threads))
    # Match the package default even though this harness imports Numba first.
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ.setdefault(variable, "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numba

    from tests._legacy_kernels import legacy_dp_forward_pass, legacy_gammatone_spectrogram
    from tests._result_comparison import assert_results_equal
    from visqol import VisqolApi, gammatone, numba_accel, patch_selector, signal_utils
    from visqol.audio_utils import load_as_mono

    numba.set_num_threads(args.threads)
    if args.fft == "fftw" and not signal_utils._HAS_PYFFTW:
        parser.error("FFTW is unavailable; install .[fftw] or select --fft scipy")
    signal_utils._HAS_PYFFTW = args.fft == "fftw"
    directory = args.testdata / "conformance_testdata_subset"
    ref = load_as_mono(str(directory / "guitar48_stereo.wav"))
    deg = load_as_mono(str(directory / "guitar48_stereo_64kbps_aac.wav"))
    api = VisqolApi()
    api.create(mode="audio")
    implementations = {
        "optimized": (patch_selector._dp_forward_pass, gammatone._gammatone_spectrogram_numba),
        "legacy_3_7_0": (legacy_dp_forward_pass, legacy_gammatone_spectrogram),
    }

    def measure(implementation):
        patch_selector._dp_forward_pass, gammatone._gammatone_spectrogram_numba = (
            implementations[implementation]
        )
        start = time.perf_counter()
        result = api.measure_from_arrays(ref.data, deg.data, ref.sample_rate)
        return time.perf_counter() - start, result

    timings = {name: [] for name in implementations}
    # Warm both implementations at the real shape; exclude JIT and FFT planning.
    _, expected = measure("legacy_3_7_0")
    _, actual = measure("optimized")
    assert_results_equal(actual, expected)
    for repeat in range(args.repeats):
        order = ["legacy_3_7_0", "optimized"]
        if repeat % 2:
            order.reverse()
        for name in order:
            elapsed, result = measure(name)
            assert_results_equal(result, expected)
            timings[name].append(elapsed)
    medians = {name: statistics.median(times) for name, times in timings.items()}
    versions = {}
    for name in ["numpy", "scipy", "numba", "llvmlite", "pyfftw", "ai-edge-litert"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "baseline_commit": "1c3953ec4b6ed2e7fa2a2ce56eab3bad648d98d4",
        "kernel_sha256": hashlib.sha256(Path(numba_accel.__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
        "threads": args.threads,
        "threading_layer": numba.threading_layer(),
        "fft_backend": args.fft,
        "repeats": args.repeats,
        "case": "official guitar48_stereo_64kbps_aac",
        "duration_sec": ref.duration,
        "protocol": "audio, FP64, search_window=60, both alignments enabled; warmed; excludes audio I/O",
        "seconds": timings,
        "median_sec": medians,
        "speedup": medians["legacy_3_7_0"] / medians["optimized"],
        "moslqo": actual.moslqo,
        "all_outputs_exact": True,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
