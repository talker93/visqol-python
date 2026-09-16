"""Exercise the public process-pool API with real work and no external data."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from tests._result_comparison import assert_results_equal
from visqol import VisqolApi


def test_process_pool_matches_sequential_after_warmup(tmp_path):
    # Warm the parent first, also exercising a pool after Numba initialization.
    sr = 16000
    rng = np.random.default_rng(817)
    t = np.arange(2 * sr) / sr
    ref = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(len(t))
    pairs = []
    for index, noise in enumerate([0.01, 0.1]):
        ref_path = tmp_path / f"ref_{index}.wav"
        deg_path = tmp_path / f"deg_{index}.wav"
        sf.write(ref_path, ref, sr, subtype="PCM_16")
        sf.write(deg_path, ref + noise * rng.standard_normal(len(t)), sr, subtype="PCM_16")
        pairs.append((str(ref_path), str(deg_path)))
    api = VisqolApi()
    api.create(mode="speech", use_lattice_model=False)
    expected = api.measure_batch(pairs, parallel=False)
    progress = []
    actual = api.measure_batch(
        pairs,
        parallel=True,
        max_workers=2,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(1, 2), (2, 2)]
    for left, right in zip(actual, expected, strict=True):
        assert not isinstance(left, Exception), str(left)
        assert not isinstance(right, Exception), str(right)
        assert_results_equal(left, right)
