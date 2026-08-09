#!/usr/bin/env python3
"""Run one isolated DwarfStar frontier and emit a normalized measurement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from dwarfstar_measurement_lib import MeasurementError, run_dwarfstar_measurement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path, help="new or empty result directory")
    parser.add_argument("--timeout-seconds", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_dwarfstar_measurement(
            executable=args.executable,
            model=args.model,
            prompt=args.prompt,
            context=args.context,
            gen_tokens=args.gen_tokens,
            output_dir=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    except (MeasurementError, OSError) as exc:
        print(f"measure_dwarfstar.py: {exc}", file=sys.stderr)
        return 2
    if result["status"] != "passed":
        print(f"DwarfStar measurement failed; evidence: {args.output}", file=sys.stderr)
        return 1
    print(args.output / "measurement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
