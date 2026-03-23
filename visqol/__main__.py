"""
ViSQOL command-line interface.

Usage::

    python -m visqol --reference ref.wav --degraded deg.wav [--speech_mode]
"""

from __future__ import annotations

import argparse
import sys
import logging

from visqol.api import VisqolApi


def main() -> None:
    """Entry point for the ``visqol`` CLI."""
    parser = argparse.ArgumentParser(
        description="ViSQOL - Virtual Speech Quality Objective Listener (Python)",
    )
    parser.add_argument(
        "--reference", "-r", required=True,
        help="Path to reference audio file (WAV)",
    )
    parser.add_argument(
        "--degraded", "-d", required=True,
        help="Path to degraded audio file (WAV)",
    )
    parser.add_argument(
        "--speech_mode", action="store_true",
        help="Use speech mode (16 kHz, exponential mapping)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to SVR model file (Audio mode only)",
    )
    parser.add_argument(
        "--search_window", type=int, default=60,
        help="Search window radius (default: 60)",
    )
    parser.add_argument(
        "--unscaled_speech", action="store_true",
        help="Don't scale speech MOS to max 5.0",
    )
    parser.add_argument(
        "--no_alignment", action="store_true",
        help="Disable global alignment",
    )
    parser.add_argument(
        "--no_realignment", action="store_true",
        help="Disable fine realignment",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Run ViSQOL
    mode: str = "speech" if args.speech_mode else "audio"

    try:
        api = VisqolApi()
        api.create(
            mode=mode,
            model_path=args.model,
            search_window=args.search_window,
            use_unscaled_speech=args.unscaled_speech,
            disable_global_alignment=args.no_alignment,
            disable_realignment=args.no_realignment,
        )

        result = api.measure(args.reference, args.degraded)

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Output results
    print(f"MOS-LQO:      {result.moslqo:.6f}")
    print(f"VNSIM:        {result.vnsim:.6f}")
    if args.verbose:
        print(f"FVNSIM:       {result.fvnsim}")
        print(f"FVNSIM10:     {result.fvnsim10}")
        print(f"FSTDNSIM:     {result.fstdnsim}")
        print(f"FVDEGENERGY:  {result.fvdegenergy}")
        print(f"Patches:      {len(result.patch_sims)}")
        for i, p in enumerate(result.patch_sims):
            print(
                f"  Patch {i}: sim={p.similarity:.4f} "
                f"ref=[{p.ref_patch_start_time:.3f}-{p.ref_patch_end_time:.3f}] "
                f"deg=[{p.deg_patch_start_time:.3f}-{p.deg_patch_end_time:.3f}]"
            )


if __name__ == "__main__":
    main()
