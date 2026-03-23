# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

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

[3.3.6]: https://github.com/talker93/visqol-python/compare/v3.3.5...v3.3.6
[3.3.5]: https://github.com/talker93/visqol-python/compare/v3.3.4...v3.3.5
[3.3.4]: https://github.com/talker93/visqol-python/compare/v3.3.3...v3.3.4
[3.3.3]: https://github.com/talker93/visqol-python/releases/tag/v3.3.3
