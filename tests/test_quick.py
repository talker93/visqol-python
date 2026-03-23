#!/usr/bin/env python3
"""
Quick ViSQOL conformance tests (subset of 3 audio + 1 speech).

Usage:
    python tests/test_quick.py --testdata /path/to/visqol/testdata
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
    candidate = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata')
    if os.path.isdir(candidate):
        return candidate
    print("ERROR: testdata directory not found.")
    print("Usage: python tests/test_quick.py --testdata /path/to/visqol/testdata")
    sys.exit(1)


TD = _get_testdata_dir()
CONF = os.path.join(TD, 'conformance_testdata_subset')
SPEECH = os.path.join(TD, 'clean_speech')

# Test 3 audio + 1 speech
api_audio = VisqolApi()
api_audio.create(mode="audio")

tests = [
    (CONF, 'guitar48_stereo.wav', 'guitar48_stereo_64kbps_aac.wav',
     4.349722, 'guitar_64aac'),
    (CONF, 'strauss48_stereo.wav', 'strauss48_stereo_lp35.wav',
     1.388879, 'strauss_lp35'),
    (CONF, 'castanets48_stereo.wav', 'castanets48_stereo.wav',
     4.732101, 'castanets_id'),
]

all_pass = True
for td, ref, deg, expected, name in tests:
    t0 = time.time()
    r = api_audio.measure(os.path.join(td, ref), os.path.join(td, deg))
    dt = time.time() - t0
    d = abs(r.moslqo - expected)
    ok = d < 0.05
    if not ok:
        all_pass = False
    print(f"{'PASS' if ok else 'FAIL'} {name:20s} "
          f"MOS={r.moslqo:.4f} exp={expected:.4f} "
          f"diff={d:.6f} ({dt:.1f}s)")
    sys.stdout.flush()

# Speech test
api_speech = VisqolApi()
api_speech.create(mode="speech")
t0 = time.time()
r = api_speech.measure(
    os.path.join(SPEECH, 'CA01_01.wav'),
    os.path.join(SPEECH, 'transcoded_CA01_01.wav'))
dt = time.time() - t0
d = abs(r.moslqo - 3.374506)
ok = d < 0.05
if not ok:
    all_pass = False
print(f"{'PASS' if ok else 'FAIL'} {'speech_CA01':20s} "
      f"MOS={r.moslqo:.4f} exp=3.3745 "
      f"diff={d:.6f} ({dt:.1f}s)")

print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
sys.exit(0 if all_pass else 1)
