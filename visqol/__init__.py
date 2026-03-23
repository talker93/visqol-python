"""
ViSQOL - Virtual Speech Quality Objective Listener (Pure Python Implementation)

A pure Python port of Google's ViSQOL v3.3.3 for objective audio quality assessment.
Compares a reference audio signal with a degraded version and outputs a MOS-LQO score (1-5).

Usage:
    from visqol import VisqolApi

    api = VisqolApi()
    api.create(mode="audio")
    result = api.measure("reference.wav", "degraded.wav")
    print(f"MOS-LQO: {result.moslqo}")
"""

__version__ = "3.3.3"

from visqol.api import VisqolApi

__all__ = ["VisqolApi"]
