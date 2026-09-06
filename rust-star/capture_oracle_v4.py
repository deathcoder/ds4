#!/usr/bin/env python3
"""Capture the streaming top-k synchronization repair as explicit oracle-v4."""
from __future__ import annotations

import capture_oracle_v1 as capture
from artifact_lib import ORACLE_V4_SOURCE_COMMIT, ORACLE_V4_SOURCE_TREE

capture.ORACLE_ID = "oracle-v4"
capture.SOURCE_REPOSITORY = "https://github.com/deathcoder/ds4.git"
capture.SOURCE_COMMIT = ORACLE_V4_SOURCE_COMMIT
capture.SOURCE_TREE = ORACLE_V4_SOURCE_TREE
capture.DEFAULT_CONFORMANCE_REPETITIONS = 4

if __name__ == "__main__":
    raise SystemExit(capture.main())
