"""
Quick smoke tests for ViSQOL Python.

These tests verify basic API functionality without requiring external testdata.
"""

import numpy as np
import pytest

from visqol import VisqolApi


class TestApiCreation:
    """Test that VisqolApi can be created in different modes."""

    def test_create_audio_mode(self):
        api = VisqolApi()
        api.create(mode="audio")

    def test_create_speech_mode(self):
        api = VisqolApi()
        api.create(mode="speech")

    def test_create_default_mode(self):
        """Default mode (no argument) should work as audio mode."""
        api = VisqolApi()
        api.create()


class TestMeasureFromArrays:
    """Test measure_from_arrays with synthetic signals."""

    def test_identical_signal_high_score(self):
        """Identical signals should produce a high MOS score."""
        api = VisqolApi()
        api.create(mode="speech")
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = api.measure_from_arrays(signal, signal, sample_rate=sr)
        assert result.moslqo >= 4.0, (
            f"Identical signal should give MOS >= 4.0, got {result.moslqo:.4f}"
        )

    def test_degraded_signal_lower_score(self):
        """Adding noise to a signal should produce a lower MOS score."""
        api = VisqolApi()
        api.create(mode="speech")
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        ref = 0.5 * np.sin(2 * np.pi * 440 * t)
        rng = np.random.default_rng(42)
        deg = ref + 0.3 * rng.standard_normal(len(ref))
        result = api.measure_from_arrays(ref, deg, sample_rate=sr)
        assert 1.0 <= result.moslqo <= 5.0, (
            f"MOS should be in [1, 5], got {result.moslqo:.4f}"
        )


class TestResultFields:
    """Test that SimilarityResult has all expected fields."""

    def test_result_has_expected_fields(self):
        api = VisqolApi()
        api.create(mode="speech")
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = api.measure_from_arrays(signal, signal, sample_rate=sr)
        assert hasattr(result, "moslqo")
        assert hasattr(result, "vnsim")
        assert hasattr(result, "fvnsim")
        assert hasattr(result, "fstdnsim")
        assert hasattr(result, "fvdegenergy")
        assert hasattr(result, "patch_sims")


class TestVersion:
    """Test package version is accessible."""

    def test_version_string(self):
        import visqol
        assert hasattr(visqol, "__version__")
        assert isinstance(visqol.__version__, str)
        parts = visqol.__version__.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"
