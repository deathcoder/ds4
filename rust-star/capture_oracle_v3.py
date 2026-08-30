#!/usr/bin/env python3
"""Capture the strongly ordered long-context DwarfStar oracle-v3."""

from __future__ import annotations

import capture_oracle_v1 as capture


capture.ORACLE_ID = "oracle-v3"
capture.SOURCE_REPOSITORY = "https://github.com/deathcoder/ds4.git"
capture.SOURCE_COMMIT = "1f8c45f120819afaa10dcd338f88a9dc2ce7b9eb"
capture.SOURCE_TREE = "ad248db417c5b2fa58afb63db238d21a99920be8"
capture.DEFAULT_CONFORMANCE_REPETITIONS = 4


if __name__ == "__main__":
    raise SystemExit(capture.main())
