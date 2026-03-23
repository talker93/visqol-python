#!/usr/bin/env python3
"""
ViSQOL Python conformance tests.

Usage:
    python tests/test_conformance.py --testdata /path/to/visqol/testdata

The testdata directory should contain:
    conformance_testdata_subset/  (audio test WAV files)
    clean_speech/                 (speech test WAV files)

You can obtain these from the official ViSQOL repository:
    https://github.com/google/visqol
"""

import argparse
import time
import sys
import os

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visqol.api import VisqolApi


def _get_testdata_dir():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--testdata', default=None)
    args, _ = parser.parse_known_args()
    if args.testdata:
        return args.testdata
    # Fallback: look relative to this file (when inside the original visqol repo)
    candidate = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata')
    if os.path.isdir(candidate):
        return candidate
    print("ERROR: testdata directory not found.")
    print("Usage: python tests/test_conformance.py --testdata /path/to/visqol/testdata")
    sys.exit(1)


TD = _get_testdata_dir()
CONF = os.path.join(TD, 'conformance_testdata_subset')
SPEECH = os.path.join(TD, 'clean_speech')

TOLERANCE = 0.05

AUDIO_TESTS = [
    ('strauss48_stereo.wav', 'strauss48_stereo_lp35.wav',
     1.3888791489130758, 'strauss_lp35'),
    ('steely48_stereo.wav', 'steely48_stereo_lp7.wav',
     2.2501683734385183, 'steely_lp7'),
    ('sopr48_stereo.wav', 'sopr48_stereo_256kbps_aac.wav',
     4.68228969737946, 'sopr_256aac'),
    ('ravel48_stereo.wav', 'ravel48_stereo_128kbps_opus.wav',
     4.465141897255348, 'ravel_128opus'),
    ('moonlight48_stereo.wav', 'moonlight48_stereo_128kbps_aac.wav',
     4.684292801646114, 'moonlight_128aac'),
    ('harpsichord48_stereo.wav', 'harpsichord48_stereo_96kbps_mp3.wav',
     4.22374532766003, 'harpsichord_96mp3'),
    ('guitar48_stereo.wav', 'guitar48_stereo_64kbps_aac.wav',
     4.349722308064298, 'guitar_64aac'),
    ('glock48_stereo.wav', 'glock48_stereo_48kbps_aac.wav',
     4.332452943882108, 'glock_48aac'),
    ('contrabassoon48_stereo.wav', 'contrabassoon48_stereo_24kbps_aac.wav',
     2.346868205375293, 'contrabassoon_24aac'),
    ('castanets48_stereo.wav', 'castanets48_stereo.wav',
     4.732101253042348, 'castanets_identity'),
]

SPEECH_TESTS = [
    ('CA01_01.wav', 'transcoded_CA01_01.wav',
     3.374505555111911, 'CA01_transcoded_exp'),
]


def run_audio_tests():
    print("=" * 70)
    print("AUDIO MODE CONFORMANCE TESTS")
    print("=" * 70)

    api = VisqolApi()
    api.create(mode="audio")

    pass_count = 0
    fail_count = 0
    total_time = 0

    for ref, deg, expected, name in AUDIO_TESTS:
        ref_path = os.path.join(CONF, ref)
        deg_path = os.path.join(CONF, deg)

        t0 = time.time()
        result = api.measure(ref_path, deg_path)
        elapsed = time.time() - t0
        total_time += elapsed

        diff = abs(result.moslqo - expected)
        passed = diff < TOLERANCE
        status = "PASS" if passed else "FAIL"

        if passed:
            pass_count += 1
        else:
            fail_count += 1

        print(f"  {status} {name:30s} "
              f"MOS={result.moslqo:.4f} "
              f"exp={expected:.4f} "
              f"diff={diff:.6f} "
              f"({elapsed:.1f}s)")

    print(f"\nAudio: {pass_count}/{len(AUDIO_TESTS)} passed "
          f"(total: {total_time:.1f}s)")
    return fail_count


def run_speech_tests():
    print("\n" + "=" * 70)
    print("SPEECH MODE CONFORMANCE TESTS (exponential mapping)")
    print("=" * 70)

    api = VisqolApi()
    api.create(mode="speech")

    pass_count = 0
    fail_count = 0
    total_time = 0

    for ref, deg, expected, name in SPEECH_TESTS:
        ref_path = os.path.join(SPEECH, ref)
        deg_path = os.path.join(SPEECH, deg)

        t0 = time.time()
        result = api.measure(ref_path, deg_path)
        elapsed = time.time() - t0
        total_time += elapsed

        diff = abs(result.moslqo - expected)
        passed = diff < TOLERANCE
        status = "PASS" if passed else "FAIL"

        if passed:
            pass_count += 1
        else:
            fail_count += 1

        print(f"  {status} {name:30s} "
              f"MOS={result.moslqo:.4f} "
              f"exp={expected:.4f} "
              f"diff={diff:.6f} "
              f"({elapsed:.1f}s)")

    print(f"\nSpeech: {pass_count}/{len(SPEECH_TESTS)} passed "
          f"(total: {total_time:.1f}s)")
    return fail_count


if __name__ == "__main__":
    audio_fails = run_audio_tests()
    speech_fails = run_speech_tests()

    total_tests = len(AUDIO_TESTS) + len(SPEECH_TESTS)
    total_fails = audio_fails + speech_fails

    print("\n" + "=" * 70)
    if total_fails == 0:
        print(f"ALL {total_tests} CONFORMANCE TESTS PASSED!")
    else:
        print(f"FAILED: {total_fails}/{total_tests} tests")
    print("=" * 70)

    sys.exit(total_fails)
