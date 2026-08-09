#!/usr/bin/env python3
"""Validate and aggregate a Rust Star paired benchmark JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from paired_benchmark_lib import PairedBenchmarkError, load_paired_run, summarize_paired_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="rust-star-paired-raw-v1 JSON")
    parser.add_argument("--json", type=Path, dest="json_output", help="write summary JSON here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_paired_run(load_paired_run(args.input))
    except PairedBenchmarkError as exc:
        print(f"summarize_paired_benchmark.py: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_output.with_name(args.json_output.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.json_output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
