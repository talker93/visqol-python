"""
Global signal alignment using upper envelope cross-correlation.

Corresponds to C++ file: alignment.cc
"""

from __future__ import annotations

import numpy as np

from visqol.audio_utils import AudioSignal
from visqol.signal_utils import find_best_lag, upper_envelope


def globally_align(reference: AudioSignal, degraded: AudioSignal) -> tuple[AudioSignal, float]:
    """
    Globally align degraded signal to reference signal.

    Uses upper envelope cross-correlation to find the best time-domain lag,
    then shifts the degraded signal accordingly.

    Matches C++ ``Alignment::GloballyAlign``.

    Args:
        reference: Reference audio signal.
        degraded: Degraded audio signal.

    Returns:
        Tuple of ``(aligned_degraded, lag_seconds)``.
    """
    ref_env = upper_envelope(reference.data)
    deg_env = upper_envelope(degraded.data)

    best_lag: int = find_best_lag(ref_env, deg_env)

    # Limit lag to half the reference length
    if best_lag == 0 or abs(best_lag) > len(reference.data) / 2.0:
        return degraded, 0.0

    if best_lag < 0:
        # Degraded comes before reference: truncate front of degraded
        new_data = degraded.data[abs(best_lag) :]
    else:
        # Reference comes before degraded: prepend zeros to degraded
        new_data = np.concatenate([np.zeros(best_lag), degraded.data])

    aligned_signal = AudioSignal(new_data, degraded.sample_rate)
    lag_seconds: float = best_lag / float(degraded.sample_rate)
    return aligned_signal, lag_seconds


def align_and_truncate(
    reference: AudioSignal, degraded: AudioSignal
) -> tuple[AudioSignal, AudioSignal, float]:
    """
    Align and truncate signals to the same length.

    Matches C++ ``Alignment::AlignAndTruncate``.

    Returns:
        Tuple of ``(aligned_ref, aligned_deg, lag_seconds)``.
    """
    aligned_deg, lag_seconds = globally_align(reference, degraded)

    ref_data = reference.data
    deg_data = aligned_deg.data

    if len(ref_data) > len(deg_data):
        ref_data = ref_data[: len(deg_data)]
    elif len(ref_data) < len(deg_data):
        # For positive lag, the beginning of ref aligns with zeros
        lag_samples = int(lag_seconds * reference.sample_rate)
        if lag_samples > 0:
            ref_data = ref_data[lag_samples:]
            deg_data = deg_data[lag_samples : lag_samples + len(ref_data)]
        else:
            deg_data = deg_data[: len(ref_data)]

    # Ensure same length
    min_len = min(len(ref_data), len(deg_data))
    ref_data = ref_data[:min_len]
    deg_data = deg_data[:min_len]

    return (
        AudioSignal(ref_data, reference.sample_rate),
        AudioSignal(deg_data, degraded.sample_rate),
        lag_seconds,
    )
