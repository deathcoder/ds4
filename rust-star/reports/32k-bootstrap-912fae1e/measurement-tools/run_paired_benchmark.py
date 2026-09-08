#!/usr/bin/env python3
"""Run, resume, retry, or inspect a paired Rust Star benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from paired_runner_lib import (
    PairedRunnerError,
    initialize_or_load_state,
    load_paired_plan,
    retry_pair,
    run_remaining,
    state_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="start or resume the predeclared schedule")
    run.add_argument("plan", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--max-new-pairs", type=int)

    retry = subparsers.add_parser("retry", help="invalidate and rerun one complete pair")
    retry.add_argument("plan", type=Path)
    retry.add_argument("--output", required=True, type=Path)
    retry.add_argument("--context", required=True, type=int)
    retry.add_argument("--repetition", required=True, type=int)
    retry.add_argument("--reason", required=True)

    status = subparsers.add_parser("status", help="show checkpoint progress")
    status.add_argument("plan", type=Path)
    status.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_paired_plan(args.plan)
        output, state = initialize_or_load_state(plan, args.output)
        if args.command == "run":
            ok = run_remaining(
                plan,
                output,
                state,
                max_new_pairs=args.max_new_pairs,
            )
        elif args.command == "retry":
            ok = retry_pair(
                plan,
                output,
                state,
                context=args.context,
                repetition=args.repetition,
                reason=args.reason,
            )
        else:
            ok = True
        print(json.dumps(state_summary(plan, state), indent=2, sort_keys=True))
        if not ok:
            print(
                "paired benchmark is blocked; inspect state.json and use an explicit retry reason "
                "only for an invalidated timed pair",
                file=sys.stderr,
            )
            return 3
        return 0
    except PairedRunnerError as exc:
        print(f"run_paired_benchmark.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
