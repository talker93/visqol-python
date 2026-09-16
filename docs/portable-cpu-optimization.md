# Portable CPU optimization

This work is on `perf/portable-cpu-kernels`. It is unreleased; keep Windows
validation as a merge gate before integrating it into `main`. Creating or
pushing this branch does not publish a PyPI release.

## Scope

The implementation changes two optional Numba kernels:

- DP patch matching caches the reference-only mean and variance once per patch,
  uses prefix maxima for predecessor lookup, and evaluates independent candidate
  offsets in parallel. The original rightmost tie rule is preserved.
- Gammatone filtering fuses four IIR stages and RMS accumulation into a single
  sample loop. It preserves filter state resets and each stage's arithmetic order
  while avoiding a bands-by-samples intermediate buffer.

The public API, score models, search window, global alignment and fine alignment
are unchanged. No `fastmath`, FP32 conversion, reduced search or skipped alignment
is introduced. SciPy FFT and pyFFTW remain supported. There are no Linux-specific
kernel calls. Optional dependency wheels and runtime behavior still need testing
for each OS, CPU architecture and Python version.

This branch does not change process-pool scheduling, add a reference-file cache,
or automatically tune thread counts. Set `NUMBA_NUM_THREADS` before importing
ViSQOL. Account for workers × threads when running multiple evaluations.

## Numerical validation

The comparison baseline is release 3.7.0, commit
`1c3953ec4b6ed2e7fa2a2ce56eab3bad648d98d4`. Its two original kernels are frozen in
`tests/_legacy_kernels.py`; the unchanged NSIM helper is shared. Both versions
run within the same process with identical inputs and dependencies.

The full suite has 335 tests with all optional dependencies installed:

- 256 DP cases check the complete cumulative tables, predecessor tables and
  selected paths, including silence, ties, near-constant data and search bounds.
- 32 spectrogram cases compare exact outputs at 16/48 kHz with one/four threads.
- Six synthetic end-to-end cases compare every result field in audio, polynomial
  speech and lattice speech modes with one/four threads.
- Twelve official conformance cases retain the existing C++ MOS tolerance and
  also compare every output exactly against the legacy Python kernels.
- The existing smoke cases and a process-pool regression cover the public API,
  result ordering and progress callbacks after the parent has initialized Numba.

Exact comparison means numerical equality with no tolerance on MOS, NSIM,
frequency statistics, energy and patch locations for the old/new implementations
**on the same platform and FFT backend**. It does not promise byte-identical
results across CPUs, dependency versions, FFT implementations or C++ ViSQOL.
Passing this finite test set is regression evidence, not a proof for every input.
The existing C++ conformance tolerance remains 0.05 MOS; it is separate from the
zero-tolerance old/new regression check.

The downloader fetches only 21 public WAV files (about 49 MB) from Google's
ViSQOL v3.3.3 commit `c3aa2e498e0f7f14202643594335a0b9ee40bdd9`. It verifies file
length and Git blob SHA against `tests/conformance_files.json`. Audio and local
test artifacts are ignored by Git.

## Validation record

The measurements below describe this branch, not the published PyPI package.
Windows testing is pending. Intel macOS is also outside the local validation set.

Local validation completed on 2026-09-16:

| Platform | Python | FFTW suite | SciPy suite | Old/new numerical differences |
| --- | --- | --- | --- | --- |
| macOS 15.7.3, Apple M4 Pro (arm64) | 3.13.2 | 335 passed, 0 skipped | 335 passed, 0 skipped | None in tested outputs |
| Linux x86-64, AMD EPYC 9K84 | 3.10.20 | 335 passed, 0 skipped | 335 passed, 0 skipped | None in tested outputs |

Ruff, mypy, wheel/sdist build and Twine metadata checks also passed locally.
The Linux suites additionally passed with TBB, but the reproducible baseline
below uses the package's default `workqueue` threading layer on both platforms.

Warmed compute benchmark, official `guitar48_stereo_64kbps_aac`, 12.453833 s:

| Platform | 3.7.0 median | Optimized median | Speedup |
| --- | ---: | ---: | ---: |
| Apple M4 Pro | 1.080057 s | 0.420863 s | 2.57× |
| AMD EPYC 9K84 | 1.509666 s | 0.539855 s | 2.80× |

Both runs use four Numba threads, `workqueue`, pyFFTW 0.15.0, Numba 0.65.1,
FP64, search window 60, and both alignment stages. BLAS/OpenMP thread limits
are one. The Linux process is restricted to eight physical cores. The table
uses five-run medians, excludes audio I/O/JIT warmup, and compares the old/new
kernels on the same machine. It is not a throughput guarantee for other audio,
worker counts or cold starts.

Raw timings, dependency versions, source checksum and exact-comparison status:
[macOS JSON](benchmarks/2026-09-16-macos-arm64.json),
[Linux JSON](benchmarks/2026-09-16-linux-x86_64.json).

## macOS / Linux reproduction

Run commands from the repository root. Use a separate environment so the
validation dependencies do not replace those of an existing evaluation job.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" "numba==0.65.1" "pyfftw==0.15.0" "ai-edge-litert==2.2.0"
export NUMBA_NUM_THREADS=4
export NUMBA_THREADING_LAYER=workqueue
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python tests/fetch_conformance_data.py
python -m pytest tests -q --testdata .testdata --fft-backend fftw --junitxml=test-results/fftw.xml
python -m pytest tests -q --testdata .testdata --fft-backend scipy --junitxml=test-results/scipy.xml
python tests/bench_portable.py --threads 4 --repeats 5 --fft fftw --json test-results/benchmark.json

python -m ruff check visqol tests
python -m ruff format --check visqol tests
python -m mypy visqol --ignore-missing-imports
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Keep the explicit `tests` argument so pytest loads the local `--testdata` and
`--fft-backend` options. A skipped dependency or conformance test means that
configuration has not received the full 335-test validation. The first run may
spend additional time compiling Numba kernels.

GitHub Actions repeats accelerated tests with both FFT backends on Ubuntu x86-64
and macOS Apple Silicon, and stores JUnit reports and a benchmark JSON artifact.
The existing Linux Python/NumPy matrix also exercises the unaccelerated fallback.

## Windows handoff (PowerShell, x86-64)

Use 64-bit Python 3.12 and an installed Git client. Start in a directory suitable
for a fresh checkout:

```powershell
git clone --branch perf/portable-cpu-kernels https://github.com/talker93/visqol-python.git
Set-Location visqol-python
git rev-parse HEAD
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[dev]" "numba==0.65.1" "pyfftw==0.15.0" "ai-edge-litert==2.2.0"

$env:NUMBA_NUM_THREADS = "4"
$env:NUMBA_THREADING_LAYER = "workqueue"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& $py tests/fetch_conformance_data.py
& $py -m pytest tests -q --testdata .testdata --fft-backend fftw --junitxml=test-results/windows-fftw.xml
& $py -m pytest tests -q --testdata .testdata --fft-backend scipy --junitxml=test-results/windows-scipy.xml
& $py tests/bench_portable.py --threads 4 --repeats 5 --fft fftw --json test-results/windows-benchmark.json
& $py -m ruff check visqol tests
& $py -m ruff format --check visqol tests
& $py -m mypy visqol --ignore-missing-imports
& $py -m pip freeze > test-results/windows-requirements.txt
```

Run each command only after the previous one succeeds. If a wheel cannot be
installed, record the Python version, architecture and error; that is an
unvalidated configuration. Do not remove the lattice or Numba dependency and
interpret resulting skips as a complete pass.

Acceptance before merging:

1. Both FFT-backend runs pass all 335 tests with zero failures and zero skips.
2. The benchmark reports `all_outputs_exact: true`. Compare old/new medians on
   the same Windows machine; there is no required cross-platform speed ratio.
3. Review any regression on representative user workloads, including batch
   multiprocessing. In standalone Windows scripts, put process-pool calls under
   `if __name__ == "__main__":` so spawned workers can import the script safely.
4. Record the tested commit and environment, attach the test/benchmark artifacts
   to the review, then merge after the platform checks pass. Publish separately
   through the existing version/tag release workflow if a release is desired.

The benchmark excludes file I/O, first-use JIT compilation and initial FFT
planning. It warms both implementations, alternates measurement order, performs
five repetitions by default, and reports median times. Short, cold-start CLI
jobs and workloads with different lengths or thread budgets can behave
differently from the reported warmed benchmark.
