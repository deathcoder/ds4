#!/usr/bin/env python3
"""Verify a Rust Star oracle result directory or compressed bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from artifact_lib import ArtifactError, SHA256_RE, open_bundle, validate_oracle_bundle


def sibling_checksum(path: Path) -> str | None:
    checksum_path = Path(str(path) + ".sha256")
    if not checksum_path.is_file():
        return None
    fields = checksum_path.read_text(encoding="utf-8").split()
    if not fields or not SHA256_RE.fullmatch(fields[0].lower()):
        raise ArtifactError(f"invalid checksum file: {checksum_path}")
    return fields[0].lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Rust Star oracle bundle structure and hashes.")
    parser.add_argument("bundle", type=Path, help="result directory or .tar.gz archive")
    parser.add_argument("--sha256", help="expected archive SHA-256; defaults to sibling .sha256 file")
    parser.add_argument("--allow-partial", action="store_true", help="allow interrupted/failed manifests")
    parser.add_argument("--json", dest="json_path", type=Path, help="write verification report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        expected = args.sha256.lower() if args.sha256 else sibling_checksum(args.bundle)
        if expected is not None and not SHA256_RE.fullmatch(expected):
            raise ArtifactError("expected SHA-256 must be 64 lowercase hexadecimal characters")
        with open_bundle(args.bundle) as (root, archive_sha256):
            if expected is not None:
                if archive_sha256 is None:
                    raise ArtifactError("--sha256 applies only to an archive, not a directory")
                if archive_sha256 != expected:
                    raise ArtifactError(
                        f"archive SHA-256 mismatch: expected={expected}, actual={archive_sha256}"
                    )
            report = validate_oracle_bundle(root, allow_partial=args.allow_partial)
            report["archive_sha256"] = archive_sha256
            report["archive_sha256_verified"] = expected is not None
    except ArtifactError as exc:
        print(f"verify_oracle_bundle.py: {exc}", file=sys.stderr)
        return 1

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"VALID {report['oracle_id']}: status={report['status']}, "
        f"contexts={report['contexts']}, artifacts={report['verified_artifacts']}, "
        f"bytes={report['verified_bytes']}"
    )
    if report["archive_sha256"]:
        suffix = " (matched expected checksum)" if report["archive_sha256_verified"] else ""
        print(f"Archive SHA-256: {report['archive_sha256']}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
