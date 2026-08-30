#!/usr/bin/env python3
"""Capture the strongly ordered long-context DwarfStar oracle-v3."""

from __future__ import annotations

import capture_oracle_v1 as capture


capture.ORACLE_ID = "oracle-v3"
capture.SOURCE_REPOSITORY = "https://github.com/deathcoder/ds4.git"
capture.SOURCE_COMMIT = "d35fb12d01d500b9cefcef24092c295687ceaf7e"
capture.SOURCE_TREE = "617415ee9f8ea7dc176d63dada1d5a7582063824"
capture.DEFAULT_CONFORMANCE_REPETITIONS = 4


if __name__ == "__main__":
    raise SystemExit(capture.main())
