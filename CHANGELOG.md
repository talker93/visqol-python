# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

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

[3.5.0]: https://github.com/talker93/visqol-python/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/talker93/visqol-python/compare/v3.3.6...v3.4.0
[3.3.6]: https://github.com/talker93/visqol-python/compare/v3.3.5...v3.3.6
[3.3.5]: https://github.com/talker93/visqol-python/compare/v3.3.4...v3.3.5
[3.3.4]: https://github.com/talker93/visqol-python/compare/v3.3.3...v3.3.4
[3.3.3]: https://github.com/talker93/visqol-python/releases/tag/v3.3.3
