# ViSQOL (Python)

[![PyPI version](https://img.shields.io/pypi/v/visqol-python)](https://pypi.org/project/visqol-python/)
[![CI](https://github.com/talker93/visqol-python/actions/workflows/ci.yml/badge.svg)](https://github.com/talker93/visqol-python/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/visqol-python)](https://pypi.org/project/visqol-python/)
[![License](https://img.shields.io/github/license/talker93/visqol-python)](LICENSE)

A pure Python implementation of [Google's ViSQOL](https://github.com/google/visqol) (Virtual Speech Quality Objective Listener) for objective audio/speech quality assessment.

ViSQOL compares a reference audio signal with a degraded version and outputs a **MOS-LQO** (Mean Opinion Score - Listening Quality Objective) score on a scale of **1.0 – 5.0**.

## Features

- **Two modes**: Audio mode (music/general audio at 48 kHz) and Speech mode (speech at 16 kHz)
- **High accuracy**: 12/12 conformance tests pass against the official C++ implementation
  - Audio mode: 9/10 tests produce **identical** MOS scores (diff = 0.000000), 1 test diff = 0.000117
  - Speech mode (polynomial): diff = 0.006715
  - Speech mode (lattice TFLite): bit-near C++ default behaviour
- **Two speech quality mappers** matching C++ ViSQOL:
  - **Lattice (default)** — deep-lattice TFLite network (`--use_lattice_model=true` in C++); requires the optional `[lattice]` extra
  - **Polynomial (fallback)** — legacy exponential fit (`--use_lattice_model=false` in C++)
- **Pure Python**: no C/C++ compilation required (the optional `[lattice]` extra adds the Google `ai-edge-litert` TFLite runtime as a binary wheel)
- **Minimal dependencies**: 4 core pip packages (`numpy`, `scipy`, `soundfile`, `libsvm-official`)
- **Optional Numba acceleration**: `pip install visqol-python[accel]` for JIT-compiled Gammatone filterbank (parallel + fastmath) and DP patch matching — **9× overall speedup**, RTF 0.064 (surpasses C++ estimates)
- **Batch & parallel evaluation**: `measure_batch(parallel=True)` for multi-process execution across CPU cores
- **Fully typed**: PEP 561 `py.typed`, strict mypy, ruff-enforced code style

## Installation

```bash
pip install visqol-python
```

For **C++-default-equivalent speech mode** (deep-lattice TFLite mapper):

```bash
pip install visqol-python[lattice]   # requires Python ≥ 3.10
```

For **Numba-accelerated** Gammatone filtering and DP matching (~9× faster):

```bash
pip install visqol-python[accel]
```

Install everything (lattice + numba):

```bash
pip install visqol-python[all]
```

Or install from source:

```bash
git clone https://github.com/talker93/visqol-python.git
cd visqol-python
pip install -e ".[dev]"
```

> **Note on speech mode parity**: Without the `[lattice]` extra, speech mode falls back to the polynomial mapping (equivalent to running C++ ViSQOL with `--use_lattice_model=false`). The polynomial can over-predict MOS by 1–2 points on degraded speech vs the C++ default. Install `[lattice]` whenever you need numbers that line up with the C++ default behaviour (see [issue #1](https://github.com/talker93/visqol-python/issues/1)).

## Quick Start

### Python API

```python
from visqol import VisqolApi

# Audio mode (default) - for music and general audio
api = VisqolApi()
api.create(mode="audio")
result = api.measure("reference.wav", "degraded.wav")
print(f"MOS-LQO: {result.moslqo:.4f}")

# Speech mode - for speech signals
api = VisqolApi()
api.create(mode="speech")
result = api.measure("ref_speech.wav", "deg_speech.wav")
print(f"MOS-LQO: {result.moslqo:.4f}")
```

### Using NumPy Arrays

```python
import numpy as np
import soundfile as sf
from visqol import VisqolApi

ref, sr = sf.read("reference.wav")
deg, _  = sf.read("degraded.wav")

api = VisqolApi()
api.create(mode="audio")
result = api.measure_from_arrays(ref, deg, sample_rate=sr)
print(f"MOS-LQO: {result.moslqo:.4f}")
```

### Batch Evaluation

```python
from visqol import VisqolApi

api = VisqolApi()
api.create(mode="audio")

file_pairs = [
    ("ref1.wav", "deg1.wav"),
    ("ref2.wav", "deg2.wav"),
    ("ref3.wav", "deg3.wav"),
]

# Sequential with progress callback
results = api.measure_batch(
    file_pairs,
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)

# Multi-process parallel (uses all CPU cores)
results = api.measure_batch(file_pairs, parallel=True, max_workers=4)

for pair, result in zip(file_pairs, results):
    if isinstance(result, Exception):
        print(f"{pair}: FAILED — {result}")
    else:
        print(f"{pair}: MOS-LQO = {result.moslqo:.4f}")
```

### Command Line

```bash
# Audio mode (default)
python -m visqol -r reference.wav -d degraded.wav

# Speech mode
python -m visqol -r reference.wav -d degraded.wav --speech_mode

# Verbose output (per-patch details)
python -m visqol -r reference.wav -d degraded.wav -v
```

**CLI options:**

| Flag | Description |
|------|-------------|
| `-r`, `--reference` | Path to reference WAV file (required) |
| `-d`, `--degraded` | Path to degraded WAV file (required) |
| `--speech_mode` | Use speech mode (16 kHz) |
| `--no_lattice_model` | Speech mode: disable lattice TFLite mapper, use polynomial fallback |
| `--lattice_model` | Custom path to lattice `.tflite` model (speech mode) |
| `--unscaled_speech` | Don't scale polynomial speech MOS to 5.0 (polynomial only) |
| `--model` | Custom SVR model file path (audio mode only) |
| `--search_window` | Search window radius (default: 60) |
| `--verbose`, `-v` | Show detailed per-patch results |

## Output

The `measure()` method returns a `SimilarityResult` object with:

| Field | Description |
|-------|-------------|
| `moslqo` | MOS-LQO score (1.0 – 5.0) |
| `vnsim` | Mean NSIM across all patches |
| `fvnsim` | Per-frequency-band mean NSIM |
| `fstdnsim` | Per-frequency-band std of NSIM |
| `fvdegenergy` | Per-frequency-band degraded energy |
| `patch_sims` | List of per-patch similarity details |

## Modes

### Audio Mode (default)
- Target sample rate: **48 kHz**
- 32 Gammatone frequency bands (50 Hz – 15 000 Hz)
- Quality mapping: SVR (Support Vector Regression) model
- Best for: music, environmental audio, codecs

### Speech Mode
- Target sample rate: **16 kHz**
- 21 Gammatone frequency bands (50 Hz – 8 000 Hz)
- VAD (Voice Activity Detection) based patch selection
- Quality mapping (choose one):
  - **Deep-lattice TFLite (default)** — same mapper as C++ ViSQOL's default `--use_lattice_model=true`; requires `pip install visqol-python[lattice]`
  - **Exponential polynomial (fallback)** — same as C++ `--use_lattice_model=false`; used automatically when the lattice runtime is not installed
- Toggle from Python: `api.create(mode="speech", use_lattice_model=False)`
- Toggle from CLI: `--no_lattice_model`
- Best for: speech, VoIP, telephony

## Performance

Measured on Apple M-series, Python 3.13:

### Without Numba (pure Python + NumPy/SciPy)

| Mode | Avg RTF | Typical Time |
|------|---------|-------------|
| Audio (48 kHz) | **0.18x** | ~2.2 s per file pair |
| Speech (16 kHz) | **0.38x** | ~1 s per file pair |

### With Numba (`pip install visqol-python[accel]`)

| Mode | Avg RTF | Typical Time | Speedup |
|------|---------|-------------|---------|
| Audio (48 kHz) | **0.064x** | ~0.8 s per file pair | **9×** |

> RTF (Real-Time Factor) < 1.0 means faster than real-time.
> With Numba acceleration, the Python implementation **surpasses C++ estimated performance** (RTF ≈ 0.093).

## Project Structure

```
visqol-python/
├── visqol/                    # Main package
│   ├── __init__.py            # Package exports & version
│   ├── api.py                 # Public API (VisqolApi)
│   ├── visqol_manager.py      # Pipeline orchestrator
│   ├── visqol_core.py         # Core algorithm
│   ├── audio_utils.py         # Audio I/O & SPL normalization
│   ├── signal_utils.py        # Envelope, cross-correlation
│   ├── analysis_window.py     # Hann window
│   ├── gammatone.py           # ERB + Gammatone filterbank + spectrogram
│   ├── patch_creator.py       # Patch creation (Image + VAD modes)
│   ├── patch_selector.py      # DP-based optimal patch matching
│   ├── alignment.py           # Global alignment via cross-correlation
│   ├── nsim.py                # NSIM similarity metric
│   ├── quality_mapper.py      # SVR & exponential quality mapping
│   ├── numba_accel.py         # Optional Numba JIT kernels (DP, NSIM, Gammatone)
│   ├── __main__.py            # CLI entry point
│   ├── py.typed               # PEP 561 type marker
│   └── model/                 # Bundled SVR model
│       └── libsvm_nu_svr_model.txt
├── tests/                     # Tests & benchmarks (pytest)
│   ├── conftest.py            # Shared fixtures & CLI options
│   ├── test_quick.py          # Smoke tests (no external data needed)
│   ├── test_conformance.py    # Full conformance tests (needs testdata)
│   ├── test_parallel_correctness.py  # Numba parallel correctness tests
│   └── bench_*.py             # Performance benchmarks
├── .github/workflows/
│   ├── ci.yml                 # CI: lint + type-check + matrix test (Python × NumPy)
│   └── publish.yml            # Auto-publish to PyPI on tag push
├── pyproject.toml             # Package metadata & build config
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Conformance Test Results

Tested against the [official C++ ViSQOL v3.3.3](https://github.com/google/visqol) expected values:

| Test Case | Mode | Expected MOS | Python MOS | Δ |
|-----------|------|-------------|------------|---|
| strauss_lp35 | Audio | 1.3889 | 1.3889 | 0.000000 |
| steely_lp7 | Audio | 2.2502 | 2.2502 | 0.000000 |
| sopr_256aac | Audio | 4.6823 | 4.6823 | 0.000000 |
| ravel_128opus | Audio | 4.4651 | 4.4651 | 0.000000 |
| moonlight_128aac | Audio | 4.6843 | 4.6843 | 0.000000 |
| harpsichord_96mp3 | Audio | 4.2237 | 4.2237 | 0.000000 |
| guitar_64aac | Audio | 4.3497 | 4.3497 | 0.000000 |
| glock_48aac | Audio | 4.3325 | 4.3325 | 0.000000 |
| contrabassoon_24aac | Audio | 2.3469 | 2.3468 | 0.000117 |
| castanets_identity | Audio | 4.7321 | 4.7321 | 0.000000 |
| speech_CA01 (polynomial) | Speech | 3.3745 | 3.3678 | 0.006715 |
| speech_CA01 (lattice) | Speech | 3.3985¹ | 3.3985 | 0.000000 |

¹ Lattice-mode expected value captured from this Python implementation as a regression baseline. Because `ai-edge-litert` reuses the upstream Google TFLite C++ runtime, the lattice MOS should match the C++ default to within the same ~1e-4 envelope as the audio-mode results above.

## References

- [Google ViSQOL (C++)](https://github.com/google/visqol) — the original implementation this project is ported from
- Hines, A., Gillen, E., Kelly, D., Skoglund, J., Kokaram, A., & Harte, N. (2015). *ViSQOLAudio: An Objective Audio Quality Metric for Low Bitrate Codecs.* The Journal of the Acoustical Society of America.
- Chinen, M., Lim, F. S., Skoglund, J., Gureev, N., O'Gorman, F., & Hines, A. (2020). *ViSQOL v3: An Open Source Production Ready Objective Speech and Audio Metric.* 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

This project is a Python port of [Google's ViSQOL](https://github.com/google/visqol), which is also licensed under Apache 2.0.
