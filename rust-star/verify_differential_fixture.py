#!/usr/bin/env python3
"""Verify a Rust Star differential fixture directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from artifact_lib import ArtifactError, validate_differential_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a Rust Star differential fixture.")
    parser.add_argument("fixture", type=Path, help="fixture directory containing manifest.json")
    parser.add_argument("--json", dest="json_path", type=Path, help="write verification report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_differential_fixture(args.fixture)
    except ArtifactError as exc:
        print(f"verify_differential_fixture.py: {exc}", file=sys.stderr)
        return 1
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"VALID {report['fixture_id']}: scope={report['scope']}, "
        f"operations={report['operations']}, tensors={report['tensors']}, "
        f"bytes={report['verified_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
