# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [3.6.0] - 2026-05-27

### Added
- **Optional pyFFTW backend** (`pip install visqol-python[fftw]`):
  - Routes `scipy.fft.fft / ifft / rfft / irfft` through FFTW3 via
    `pyfftw.interfaces.scipy_fft` for the alignment and cross-correlation
    FFTs. Detected at module load time in `signal_utils` and applied
    transparently via a thin `_fft_backend()` context manager.
  - Plan cache enabled with 60 s keep-alive so consecutive measurements
    on equal-length signals reuse the FFTW plan.

### Improved
- **Fused NSIM kernel**: `_measure_patch_similarity_numba` merged its 5
  separate 2-D convolutions (μ_r, μ_d, ref², deg², ref·deg) and the
  intensity/structure recombination into one `(r, c)` double loop. Each
  patch element is read from L1 once per visit instead of five times,
  and the four intermediate `(rows × cols)` matrices are no longer
  materialised between convs. Bit-exact with the previous split-conv
  path (ULP-level FP rounding only).
- **`nsim.measure_patch_similarity`** now dispatches to the fused JIT
  kernel when Numba is available — the same code path the DP patch
  matcher uses, so `finely_align_and_recreate_patches` shares the
  speedup. The pure-NumPy implementation is preserved as a fallback.
- **`signal_utils._hilbert`** is a drop-in `scipy.signal.hilbert`
  replacement built on `rfft` for the real-valued input (~2× less
  forward-transform work than the original full complex `fft`).
- **`signal_utils.find_best_lag`** now uses `rfft + irfft` for the
  cross-correlation, again exploiting the real input. Net effect:
  alignment FFTs see roughly 2× less work overall.
- Removed dead helper `_conv2d_boundary_valid` from `numba_accel.py`
  (subsumed by the fused kernel).

### Fixed
- **`find_best_lag` interpreter hot loop**: the previous implementation
  ran `xcorr_full[-max_lag:].tolist() + xcorr_full[:max_lag+1].tolist()`
  and then `builtin argmax(list)` over a ~1.2 M-element Python list,
  costing ~33 ms per call in pure interpreter overhead. Now uses
  `np.concatenate + np.argmax` entirely in C.

### Performance

Apple M-series, Python 3.13, audio mode, the `guitar48_stereo` 12.5 s
conformance case, average of 3 runs with Numba + pyFFTW both installed:

| Stage              | v3.5.0   | v3.6.0   | Speedup |
|--------------------|----------|----------|---------|
| DP Patch matching  | 0.397 s  | 0.131 s  | **3.0×** |
| Global align / FFT | 0.173 s  | 0.091 s  | 1.9×    |
| Fine align + NSIM  | 0.093 s  | 0.043 s  | 2.2×    |
| Gammatone          | 0.173 s  | 0.179 s  | ~       |
| **Total**          | **0.839 s** | **0.447 s** | **1.9×** |
| **RTF**            | 0.067    | **0.036**   | (C++ est. 0.093) |

### Numerical parity

All v3.5.0 conformance baselines preserved within ULP precision:

| Test                          | Max MOS diff vs v3.5.0 |
|-------------------------------|------------------------|
| Audio (10 conformance cases)  | < 5 × 10⁻¹⁴            |
| Speech polynomial CA01        | 0.0 (bit-exact)        |
| Speech lattice CA01           | 0.0 (bit-exact)        |

## [3.5.0] - 2026-05-26

### Added
- **Deep-lattice TFLite speech quality mapper** (`pip install visqol-python[lattice]`):
  - `TFLiteSpeechQualityMapper` loads the same `.tflite` lattice network used by
    C++ ViSQOL's default `--use_lattice_model=true` and runs inference through
    the upstream Google TFLite C++ runtime via `ai-edge-litert`
  - New `use_lattice_model` parameter on `VisqolApi.create()` (default `None`
    auto-enables lattice when the runtime is installed)
  - New `lattice_model_path` parameter to override the bundled model
  - New CLI flags `--no_lattice_model` and `--lattice_model PATH`
- New `[lattice]` and `[all]` extras in `pyproject.toml`
- Bundled `lattice_*.tflite` (2.1 MB) into the wheel as package data

### Fixed
- **GH issue #1**: Speech-mode MOS scores were systematically 1–2 points higher
  than C++ ViSQOL's default. Root cause: the Python port only implemented the
  legacy polynomial mapper (`SpeechSimilarityToQualityMapper`, equivalent to
  C++ `--use_lattice_model=false`), while the C++ default routes through the
  TFLite lattice network. Installing `visqol-python[lattice]` now matches C++
  default scoring (CA01 conformance: diff 0.027 vs 1–2 MOS before).
- **`signal_utils.normalize()` parity bug**: the previous implementation did
  min–max scaling to ``[0, 1]`` (shifting the signal positive and adding a DC
  offset), while C++ ``MiscMath::Normalize`` only divides by the peak. This
  inflated the RMS values fed to the speech-mode VAD, causing Python to keep
  every patch as voice-active and adding spurious patches the C++ binary
  would have discarded. Fixing this brought polynomial speech parity from
  diff 0.007 → 0.001 and was a prerequisite for lattice parity. Only the
  speech-mode VAD path used this function; audio mode is unaffected.
- **`nsim` stddev estimator mismatch**: both ``nsim.measure_patch_similarity``
  (``np.std(..., ddof=0)``) and the Numba ``_measure_patch_similarity_numba``
  kernel (``sqrt(ss / cols)``) used the population estimator (divide by N).
  C++ uses Armadillo's ``stddev(..., 0)`` which is the *unbiased* sample
  estimator (divide by N-1, despite the misleading ``0`` flag). The
  per-band ``freq_band_stddevs`` was therefore systematically smaller by
  ``sqrt((N-1)/N) ≈ 0.974``, which fed into the pooled ``fstdnsim`` and
  perturbed every lattice prediction.
- **`numba_accel.fastmath=True` on the Gammatone spectrogram kernel**: the
  compounded LLVM-level FP reassociation across the 4-stage cascaded IIR ×
  thousands of samples × hundreds of frames pushed lattice MOS off by
  another ~0.02 vs strict IEEE-754. ``fastmath`` has been removed from the
  spectrogram kernel; ``parallel=True`` is kept (each frame's IIR state is
  independent so the reduction is safe).

### Speech-mode parity numbers (CA01 conformance)

| Mode | Before all fixes | After all fixes | C++ baseline |
|------|------------------|-----------------|--------------|
| Polynomial | diff 0.0067 | **diff 0.0011** | 3.3745 |
| Lattice | diff 0.0856 (≈1–2 MOS on Nils's TCD-VOIP samples) | **diff 0.0023** | 3.3130 |

### Changed
- Speech mode `create(mode="speech")` now auto-uses lattice when available; when
  `ai-edge-litert` is missing, it logs a one-time warning and falls back to
  polynomial (existing scores reproduce exactly).
- `tests/test_conformance.py` split the single speech case into
  `test_speech_polynomial_conformance` (existing C++ polynomial baseline 3.3745)
  and `test_speech_lattice_conformance` (regression baseline captured from this
  implementation).
- README: documented the polynomial-vs-lattice distinction, new install matrix,
  and parity caveats.

## [3.4.0] - 2026-03-23

### Added
- **Numba JIT acceleration** (`pip install visqol-python[accel]`):
  - DP patch matching inner loops compiled to machine code via `@njit`
  - Gammatone IIR filterbank compiled with `parallel=True` + `fastmath=True` — frames processed in parallel across all CPU cores
  - NSIM similarity kernel JIT-compiled
  - Automatic `NUMBA_THREADING_LAYER=workqueue` setup for macOS compatibility
  - Zero-loss parallel accuracy (each frame's IIR state is independent)
- **Batch evaluation API**: `VisqolApi.measure_batch()` with optional `parallel=True` and `max_workers` for multi-process execution
- Exported `PatchSimilarityResult` and `ProgressCallback` from top-level package

### Performance
- **12x Gammatone speedup** via parallel + fastmath (1.53s → 0.13s per signal pair)
- **8.7x DP patch matching speedup** via Numba JIT (3.5s → 0.40s)
- **Overall 9x speedup**: RTF 0.58 → 0.064 (surpasses C++ estimate of 0.093)
- Fine alignment skip optimization: 29x speedup when lag == 0

### Improved
- `__repr__` / `__str__` for `SimilarityResult`, `AudioSignal`, `PatchSimilarityResult`, `Spectrogram`
- Logging replaces print statements in CLI verbose output
- Development tooling: ruff lint/format + mypy strict type checking in CI

### Fixed
- **CI failures**: resolved all ruff lint (308 errors), ruff format (24 files), and mypy (24 errors) issues
- Added `per-file-ignores` for benchmark test scripts (E402, E702)
- Added mypy override for `numba_accel.py` (untyped `@njit` decorators)
- Fixed `no-any-return` errors across `audio_utils.py`, `gammatone.py`, `visqol_core.py`, `api.py`
- Added `TYPE_CHECKING` imports for `ImagePatchCreator` / `VadPatchCreator` in `visqol_core.py`

## [3.3.6] - 2026-03-23

### Added
- **Batch evaluation API**: `VisqolApi.measure_batch()` with `progress_callback` support
- **Numba optional acceleration**: `visqol/numba_accel.py` with JIT-compiled DP forward pass and NSIM kernel
- `[accel]` optional dependency group: `pip install visqol-python[accel]`

### Improved
- `GammatoneFilterBank.apply_filter()` pre-builds coefficient arrays (avoids per-channel allocation)
- `prepare_spectrograms_for_comparison()` vectorized per-frame noise floor
- Ruff lint/format configuration added to `pyproject.toml`
- CI enhanced with lint and type-check jobs
- Development dependencies: `[project.optional-dependencies] dev`

## [3.3.5] - 2026-03-23

### Added
- **Type hints** on all public and internal APIs (`from __future__ import annotations`)
- **`py.typed`** marker (PEP 561) — mypy / pyright can now type-check dependents
- **CONTRIBUTING.md** with development setup, code style, and PR guidelines
- Exported `SimilarityResult` and `AudioSignal` from top-level `visqol` package
- `mypy` configuration in `pyproject.toml`

### Improved
- **Error handling**: friendly `ValueError` / `FileNotFoundError` / `TypeError` throughout:
  - `VisqolApi.create()` now validates mode, search_window, and model_path
  - `VisqolApi.measure()` checks file existence before processing
  - `VisqolApi.measure_from_arrays()` validates array types, emptiness, and sample rate
  - `AudioSignal` validates sample rate on construction
  - `AnalysisWindow` validates sample_rate and overlap range
  - CLI now catches exceptions and prints user-friendly error messages
- `AnalysisWindow.apply_hann_window()` uses `ValueError` instead of bare `assert`

## [3.3.4] - 2026-03-23

### Improved
- Tests rewritten in **pytest** format with `parametrize` and fixtures
- Added **CI workflow** (GitHub Actions): auto-test on Python 3.9–3.13 for every push/PR
- Added **smoke tests** (`test_quick.py`) that run without external testdata
- Version number now managed in a single place (`visqol/__init__.py`)
- Removed redundant `setup.py` — `pyproject.toml` is the single source of truth
- Added this CHANGELOG
- README: added PyPI / CI / License badges

### Fixed
- `requires-python` updated from `>=3.8` to `>=3.9` (numpy/scipy dropped 3.8 support)

## [3.3.3] - 2026-03-23

### Added
- Initial PyPI release as `visqol-python`
- Pure Python port of [Google's ViSQOL v3.3.3](https://github.com/google/visqol)
- **Audio mode** (48 kHz, SVR quality mapping) — 10/10 conformance tests pass
- **Speech mode** (16 kHz, exponential polynomial mapping) — 1/1 conformance test passes
- Python API: `VisqolApi.measure()` and `VisqolApi.measure_from_arrays()`
- CLI: `python -m visqol` / `visqol` command
- Bundled SVR model (`libsvm_nu_svr_model.txt`)
- GitHub Actions workflow for auto-publish to PyPI via Trusted Publisher

[3.6.0]: https://github.com/talker93/visqol-python/compare/v3.5.0...v3.6.0
[3.5.0]: https://github.com/talker93/visqol-python/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/talker93/visqol-python/compare/v3.3.6...v3.4.0
[3.3.6]: https://github.com/talker93/visqol-python/compare/v3.3.5...v3.3.6
[3.3.5]: https://github.com/talker93/visqol-python/compare/v3.3.4...v3.3.5
[3.3.4]: https://github.com/talker93/visqol-python/compare/v3.3.3...v3.3.4
[3.3.3]: https://github.com/talker93/visqol-python/releases/tag/v3.3.3
