"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest

# Select the package default before test modules import Numba directly.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")


def pytest_addoption(parser):
    """Register data and backend options for conformance tests."""
    parser.addoption(
        "--testdata",
        action="store",
        default=None,
        help="Path to the ViSQOL testdata directory (for conformance tests)",
    )
    parser.addoption(
        "--fft-backend",
        choices=["auto", "scipy", "fftw"],
        default="auto",
        help="Select an FFT backend for the entire test session",
    )


@pytest.fixture(scope="session", autouse=True)
def fft_backend(request):
    from visqol import signal_utils

    choice = request.config.getoption("--fft-backend")
    original = signal_utils._HAS_PYFFTW
    if choice == "fftw" and not original:
        pytest.fail("--fft-backend=fftw requires the optional pyFFTW dependency")
    if choice != "auto":
        signal_utils._HAS_PYFFTW = choice == "fftw"
    yield
    signal_utils._HAS_PYFFTW = original
