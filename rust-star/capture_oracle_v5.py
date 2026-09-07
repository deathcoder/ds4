#!/usr/bin/env python3
"""Capture pinned upstream refresh oracle-v5; acceptance requires full gates."""
from artifact_lib import ORACLE_V5_SOURCE_COMMIT, ORACLE_V5_SOURCE_TREE
import capture_oracle_v1 as capture

capture.ORACLE_ID = "oracle-v5"
capture.SOURCE_REPOSITORY = "https://github.com/deathcoder/ds4.git"
capture.SOURCE_COMMIT = ORACLE_V5_SOURCE_COMMIT
capture.SOURCE_TREE = ORACLE_V5_SOURCE_TREE
capture.DEFAULT_CONFORMANCE_REPETITIONS = 4

if __name__ == "__main__":
    raise SystemExit(capture.main())
