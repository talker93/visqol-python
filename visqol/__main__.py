"""
ViSQOL command-line interface.

Usage::

    python -m visqol --reference ref.wav --degraded deg.wav [--speech_mode]
"""

from __future__ import annotations

import argparse
import logging
import sys

from visqol.api import VisqolApi

logger = logging.getLogger("visqol")


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
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

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

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        sys.exit(2)

    # Output results
    print(f"MOS-LQO:      {result.moslqo:.6f}")
    print(f"VNSIM:        {result.vnsim:.6f}")

    if args.verbose:
        logger.info("FVNSIM:       %s", result.fvnsim)
        logger.info("FVNSIM10:     %s", result.fvnsim10)
        logger.info("FSTDNSIM:     %s", result.fstdnsim)
        logger.info("FVDEGENERGY:  %s", result.fvdegenergy)
        logger.info("Patches:      %d", len(result.patch_sims))
        for i, p in enumerate(result.patch_sims):
            logger.info(
                "  Patch %d: sim=%.4f ref=[%.3f-%.3f] deg=[%.3f-%.3f]",
                i, p.similarity,
                p.ref_patch_start_time, p.ref_patch_end_time,
                p.deg_patch_start_time, p.deg_patch_end_time,
            )


if __name__ == "__main__":
    main()
