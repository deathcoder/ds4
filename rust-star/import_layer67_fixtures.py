#!/usr/bin/env python3
"""Import independently repeated layer-6/7 position-0 through -3 fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import import_layer45_fixtures as common


CAPTURED_AT_UTC = "2026-08-15T10:11:21Z"
LAYERS = (6, 7)
POSITIONS = (1, 2, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    common.CAPTURED_AT_UTC = CAPTURED_AT_UTC
    for layer in LAYERS:
        for position in POSITIONS:
            print(
                common.import_complete(
                    args.capture_root,
                    args.fixtures_root,
                    layer,
                    position,
                )
            )
        print(common.import_prime(args.capture_root, args.fixtures_root, layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
