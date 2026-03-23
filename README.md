# ViSQOL (Python)

A pure Python implementation of [Google's ViSQOL](https://github.com/google/visqol) (Virtual Speech Quality Objective Listener) v3.3.3 for objective audio/speech quality assessment.

ViSQOL compares a reference audio signal with a degraded version and outputs a **MOS-LQO** (Mean Opinion Score - Listening Quality Objective) score on a scale of **1.0 – 5.0**.

## Features

- **Two modes**: Audio mode (music/general audio at 48 kHz) and Speech mode (speech at 16 kHz)
- **High accuracy**: 11/11 conformance tests pass against the official C++ implementation
  - Audio mode: 9/10 tests produce **identical** MOS scores (diff = 0.000000), 1 test diff = 0.000117
  - Speech mode: diff = 0.006715
- **Pure Python**: no C/C++ compilation required
- **Minimal dependencies**: only 4 pip packages (`numpy`, `scipy`, `soundfile`, `libsvm-official`)
- **Faster than real-time**: Audio RTF ≈ 0.71x, Speech RTF ≈ 0.38x

## Installation

```bash
pip install visqol-python
```

Or install from source:

```bash
git clone https://github.com/talker93/visqol-python.git
cd visqol-python
pip install -e .
```

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
| `--speech_mode` | Use speech mode (16 kHz, polynomial mapping) |
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
- 32 Gammatone frequency bands (50 Hz – 8 000 Hz)
- Quality mapping: exponential polynomial fit
- VAD (Voice Activity Detection) based patch selection
- Best for: speech, VoIP, telephony

## Performance

Measured on Apple M-series, Python 3.13:

| Mode | Avg RTF | Typical Time |
|------|---------|-------------|
| Audio (48 kHz) | **0.71x** | 7 – 12 s per file pair |
| Speech (16 kHz) | **0.38x** | ~1 s per file pair |

> RTF (Real-Time Factor) < 1.0 means faster than real-time.

## Project Structure

```
visqol-python/
├── visqol/                    # Main package
│   ├── __init__.py            # Package exports
│   ├── api.py                 # Public API
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
│   └── __main__.py            # CLI entry point
├── model/                     # Bundled SVR model
│   └── libsvm_nu_svr_model.txt
├── tests/                     # Conformance tests
│   ├── test_conformance.py
│   └── test_quick.py
├── setup.py
├── requirements.txt
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
| speech_CA01 | Speech | 3.3745 | 3.3678 | 0.006715 |

## References

- [Google ViSQOL (C++)](https://github.com/google/visqol) — the original implementation this project is ported from
- Hines, A., Gillen, E., Kelly, D., Skoglund, J., Kokaram, A., & Harte, N. (2015). *ViSQOLAudio: An Objective Audio Quality Metric for Low Bitrate Codecs.* The Journal of the Acoustical Society of America.
- Chinen, M., Lim, F. S., Skoglund, J., Gureev, N., O'Gorman, F., & Hines, A. (2020). *ViSQOL v3: An Open Source Production Ready Objective Speech and Audio Metric.* 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

This project is a Python port of [Google's ViSQOL](https://github.com/google/visqol), which is also licensed under Apache 2.0.
