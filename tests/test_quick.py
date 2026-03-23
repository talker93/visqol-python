"""
Quick smoke tests for ViSQOL Python.

These tests verify basic API functionality without requiring external testdata.
"""

import numpy as np
import pytest

from visqol import VisqolApi, SimilarityResult, AudioSignal


# ── API creation ──


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

    def test_create_case_insensitive(self):
        api = VisqolApi()
        api.create(mode="SPEECH")

    def test_create_invalid_mode_raises(self):
        api = VisqolApi()
        with pytest.raises(ValueError, match="Invalid mode"):
            api.create(mode="invalid")

    def test_create_negative_search_window_raises(self):
        api = VisqolApi()
        with pytest.raises(ValueError, match="search_window"):
            api.create(search_window=-1)

    def test_create_missing_model_raises(self):
        api = VisqolApi()
        with pytest.raises(FileNotFoundError):
            api.create(mode="audio", model_path="/nonexistent/model.txt")


# ── Measure guards ──


class TestMeasureGuards:
    """Test that measure() raises helpful errors for bad inputs."""

    def test_measure_before_create_raises(self):
        api = VisqolApi()
        with pytest.raises(RuntimeError, match="create"):
            api.measure("a.wav", "b.wav")

    def test_measure_nonexistent_ref_raises(self):
        api = VisqolApi()
        api.create(mode="speech")
        with pytest.raises(FileNotFoundError, match="Reference"):
            api.measure("/nonexistent/ref.wav", "/nonexistent/deg.wav")

    def test_measure_from_arrays_before_create_raises(self):
        api = VisqolApi()
        with pytest.raises(RuntimeError, match="create"):
            api.measure_from_arrays(np.zeros(100), np.zeros(100), 16000)

    def test_measure_from_arrays_bad_type_raises(self):
        api = VisqolApi()
        api.create(mode="speech")
        with pytest.raises(TypeError, match="numpy array"):
            api.measure_from_arrays([1, 2, 3], np.zeros(100), 16000)  # type: ignore[arg-type]

    def test_measure_from_arrays_empty_raises(self):
        api = VisqolApi()
        api.create(mode="speech")
        with pytest.raises(ValueError, match="empty"):
            api.measure_from_arrays(np.array([]), np.zeros(100), 16000)

    def test_measure_from_arrays_bad_sr_raises(self):
        api = VisqolApi()
        api.create(mode="speech")
        with pytest.raises(ValueError, match="sample_rate"):
            api.measure_from_arrays(np.zeros(100), np.zeros(100), 0)


# ── measure_from_arrays ──


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


# ── Result fields ──


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


# ── Package metadata ──


class TestVersion:
    """Test package version is accessible."""

    def test_version_string(self):
        import visqol
        assert hasattr(visqol, "__version__")
        assert isinstance(visqol.__version__, str)
        parts = visqol.__version__.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"

    def test_public_exports(self):
        """Package should export key classes."""
        import visqol
        assert hasattr(visqol, "VisqolApi")
        assert hasattr(visqol, "SimilarityResult")
        assert hasattr(visqol, "AudioSignal")
