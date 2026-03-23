"""
ViSQOL Python conformance tests.

Requires the official ViSQOL testdata directory. Provide it via:
    pytest tests/test_conformance.py --testdata /path/to/visqol/testdata

The testdata directory should contain:
    conformance_testdata_subset/  (audio test WAV files)
    clean_speech/                 (speech test WAV files)

You can obtain these from the official ViSQOL repository:
    https://github.com/google/visqol
"""

import os

import pytest

from visqol import VisqolApi

# ── Fixtures ──


@pytest.fixture(scope="session")
def testdata_dir(request):
    """Resolve testdata directory from --testdata or auto-detect."""
    td = request.config.getoption("--testdata")
    if td and os.path.isdir(td):
        return td
    # Fallback: look relative to this file (when inside the original visqol repo)
    candidate = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
    if os.path.isdir(candidate):
        return candidate
    pytest.skip("testdata directory not found — use --testdata /path/to/visqol/testdata")


@pytest.fixture(scope="session")
def conf_dir(testdata_dir):
    return os.path.join(testdata_dir, "conformance_testdata_subset")


@pytest.fixture(scope="session")
def speech_dir(testdata_dir):
    return os.path.join(testdata_dir, "clean_speech")


@pytest.fixture(scope="session")
def audio_api():
    api = VisqolApi()
    api.create(mode="audio")
    return api


@pytest.fixture(scope="session")
def speech_api():
    api = VisqolApi()
    api.create(mode="speech")
    return api


# ── Test data ──

TOLERANCE = 0.05

AUDIO_CASES = [
    ("strauss48_stereo.wav", "strauss48_stereo_lp35.wav", 1.3888791489130758, "strauss_lp35"),
    ("steely48_stereo.wav", "steely48_stereo_lp7.wav", 2.2501683734385183, "steely_lp7"),
    ("sopr48_stereo.wav", "sopr48_stereo_256kbps_aac.wav", 4.68228969737946, "sopr_256aac"),
    (
        "ravel48_stereo.wav",
        "ravel48_stereo_128kbps_opus.wav",
        4.465141897255348,
        "ravel_128opus",
    ),
    (
        "moonlight48_stereo.wav",
        "moonlight48_stereo_128kbps_aac.wav",
        4.684292801646114,
        "moonlight_128aac",
    ),
    (
        "harpsichord48_stereo.wav",
        "harpsichord48_stereo_96kbps_mp3.wav",
        4.22374532766003,
        "harpsichord_96mp3",
    ),
    (
        "guitar48_stereo.wav",
        "guitar48_stereo_64kbps_aac.wav",
        4.349722308064298,
        "guitar_64aac",
    ),
    ("glock48_stereo.wav", "glock48_stereo_48kbps_aac.wav", 4.332452943882108, "glock_48aac"),
    (
        "contrabassoon48_stereo.wav",
        "contrabassoon48_stereo_24kbps_aac.wav",
        2.346868205375293,
        "contrabassoon_24aac",
    ),
    (
        "castanets48_stereo.wav",
        "castanets48_stereo.wav",
        4.732101253042348,
        "castanets_identity",
    ),
]

SPEECH_CASES = [
    ("CA01_01.wav", "transcoded_CA01_01.wav", 3.374505555111911, "CA01_transcoded"),
]


# ── Audio mode tests ──


@pytest.mark.parametrize(
    "ref_name, deg_name, expected_mos, test_id",
    AUDIO_CASES,
    ids=[c[3] for c in AUDIO_CASES],
)
def test_audio_conformance(audio_api, conf_dir, ref_name, deg_name, expected_mos, test_id):
    ref_path = os.path.join(conf_dir, ref_name)
    deg_path = os.path.join(conf_dir, deg_name)
    result = audio_api.measure(ref_path, deg_path)
    diff = abs(result.moslqo - expected_mos)
    assert diff < TOLERANCE, (
        f"[{test_id}] MOS={result.moslqo:.6f}, expected={expected_mos:.6f}, diff={diff:.6f}"
    )


# ── Speech mode tests ──


@pytest.mark.parametrize(
    "ref_name, deg_name, expected_mos, test_id",
    SPEECH_CASES,
    ids=[c[3] for c in SPEECH_CASES],
)
def test_speech_conformance(speech_api, speech_dir, ref_name, deg_name, expected_mos, test_id):
    ref_path = os.path.join(speech_dir, ref_name)
    deg_path = os.path.join(speech_dir, deg_name)
    result = speech_api.measure(ref_path, deg_path)
    diff = abs(result.moslqo - expected_mos)
    assert diff < TOLERANCE, (
        f"[{test_id}] MOS={result.moslqo:.6f}, expected={expected_mos:.6f}, diff={diff:.6f}"
    )
