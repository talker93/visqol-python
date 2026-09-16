"""Exact regression against the frozen 3.7.0 kernels on each test platform."""

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from tests._legacy_kernels import (  # noqa: E402
    legacy_dp_forward_pass,
    legacy_gammatone_spectrogram,
)
from tests._result_comparison import assert_results_equal  # noqa: E402
from visqol import VisqolApi, gammatone, numba_accel, patch_selector  # noqa: E402
from visqol.analysis_window import AnalysisWindow  # noqa: E402
from visqol.audio_utils import AudioSignal  # noqa: E402
from visqol.quality_mapper import has_lattice_runtime  # noqa: E402


@pytest.fixture(params=[1, 4], ids=["one_thread", "four_threads"])
def kernel_threads(request):
    previous = numba.get_num_threads()
    if request.param > numba.config.NUMBA_NUM_THREADS:
        pytest.skip("Numba thread budget is smaller than this test configuration")
    numba.set_num_threads(request.param)
    yield request.param
    numba.set_num_threads(previous)


@pytest.mark.parametrize("bands,width", [(32, 30), (21, 20), (4, 7), (1, 1)])
@pytest.mark.parametrize("window", [1, 7, 40, 1800])
@pytest.mark.parametrize("kind", ["random", "silent", "tied", "near_constant"])
@pytest.mark.parametrize("count", [1, 4])
def test_dp_tables_and_path_match_legacy(bands, width, window, kind, count, kernel_threads):
    rng = np.random.default_rng(20260916)
    indices = np.array([0, 11, 30, 41], dtype=np.int64)
    ref = rng.uniform(0, 70, (4, bands, width))
    deg = rng.uniform(0, 70, (61, bands, width))
    if kind == "silent":
        ref[:] = deg[:] = 0.0
    elif kind == "tied":
        ref[:] = deg[:] = 3.0
    elif kind == "near_constant":
        ref = 3.0 + ref * 1e-12
        deg = 3.0 + deg * 1e-12
    args = (
        ref,
        deg,
        indices,
        count,
        len(deg),
        window,
        numba_accel._GW,
        numba_accel._C1,
        numba_accel._C3,
    )
    expected = legacy_dp_forward_pass(*args)
    actual = numba_accel._dp_forward_pass(*args)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(
        numba_accel._dp_backtrace(*actual, indices, count, len(deg), window),
        numba_accel._dp_backtrace(*expected, indices, count, len(deg), window),
    )


@pytest.mark.parametrize("sr,bands", [(16000, 21), (48000, 32)])
@pytest.mark.parametrize("duration", [0.601, 2.917])
@pytest.mark.parametrize("kind", ["random", "sine", "silence", "tiny"])
def test_fused_spectrogram_matches_legacy(
    sr, bands, duration, kind, kernel_threads, monkeypatch
):
    rng = np.random.default_rng(20260916)
    length = int(duration * sr)
    signal = rng.uniform(-0.8, 0.8, length)
    if kind == "sine":
        signal = np.sin(np.arange(length) * 2 * np.pi * 997 / sr) * 0.5
    elif kind == "silence":
        signal[:] = 0.0
    elif kind == "tiny":
        signal *= 1e-12
    window = AnalysisWindow(sr)
    builder = gammatone.GammatoneSpectrogramBuilder(bands, 50.0, speech_mode=sr == 16000)
    actual = builder.build(AudioSignal(signal, sr), window).data
    with monkeypatch.context() as context:
        context.setattr(
            gammatone, "_gammatone_spectrogram_numba", legacy_gammatone_spectrogram
        )
        expected = builder.build(AudioSignal(signal, sr), window).data
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "mode,lattice", [("audio", False), ("speech", False), ("speech", True)]
)
def test_complete_result_matches_legacy(mode, lattice, kernel_threads, monkeypatch):
    if lattice and not has_lattice_runtime():
        pytest.skip("Optional speech lattice runtime is unavailable")
    sr = 48000 if mode == "audio" else 16000
    rng = np.random.default_rng(7819)
    t = np.arange(int(3.17 * sr)) / sr
    ref = 0.3 * np.sin(2 * np.pi * 997 * t) + 0.08 * rng.standard_normal(len(t))
    deg = np.concatenate((np.zeros(173), ref[:-173])) + 0.015 * rng.standard_normal(len(t))
    api = VisqolApi()
    api.create(mode=mode, use_lattice_model=lattice)
    actual = api.measure_from_arrays(ref, deg, sr)
    with monkeypatch.context() as context:
        context.setattr(patch_selector, "_dp_forward_pass", legacy_dp_forward_pass)
        context.setattr(
            gammatone, "_gammatone_spectrogram_numba", legacy_gammatone_spectrogram
        )
        expected = api.measure_from_arrays(ref, deg, sr)
    assert_results_equal(actual, expected)
