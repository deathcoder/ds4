#!/usr/bin/env python3
"""Capture the deterministic long-context DwarfStar oracle-v2."""

from __future__ import annotations

import capture_oracle_v1 as capture


capture.ORACLE_ID = "oracle-v2"
capture.SOURCE_REPOSITORY = "https://github.com/deathcoder/ds4.git"
capture.SOURCE_COMMIT = "b81c099b1f7888358fcdc820e7e70566c04aafae"
capture.SOURCE_TREE = "4b913890d4dc8a12872cf5462b41ce21f2007400"
capture.DEFAULT_CONFORMANCE_REPETITIONS = 2


if __name__ == "__main__":
    raise SystemExit(capture.main())
