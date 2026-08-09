#!/usr/bin/env python3
"""Compare two full-vocabulary logit artifacts under the Rust Star contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from artifact_lib import ArtifactError, compare_logit_artifacts, load_logit_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare oracle and candidate FP32 logit JSON artifacts.")
    parser.add_argument("reference", type=Path, help="DwarfStar oracle logit JSON")
    parser.add_argument("candidate", type=Path, help="candidate engine logit JSON")
    parser.add_argument("--json", dest="json_path", type=Path, help="write the complete comparison report")
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="exit successfully for valid non-C0 artifacts; metrics are still reported",
    )
    parser.add_argument("--mismatch-limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mismatch_limit < 0:
        print("compare_logits.py: --mismatch-limit cannot be negative", file=sys.stderr)
        return 2
    try:
        reference = load_logit_artifact(args.reference)
        candidate = load_logit_artifact(args.candidate)
        report = compare_logit_artifacts(reference, candidate, mismatch_limit=args.mismatch_limit)
    except ArtifactError as exc:
        print(f"compare_logits.py: {exc}", file=sys.stderr)
        return 2

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if report["c0_exact"]:
        print(f"C0 PASS: {report['vocab']} FP32 logits and required metadata are bit-identical")
        return 0
    if "shape_mismatch" in report:
        print(f"C0 FAIL: shape mismatch {report['shape_mismatch']}")
    else:
        print(
            "C0 FAIL: "
            f"{report['bit_mismatches']}/{report['vocab']} logits differ; "
            f"max_abs={report['max_absolute_error']:.9g}; "
            f"max_ulp={report['max_ulp_distance']}; "
            f"argmax_match={report['argmax_match']}"
        )
        print(
            "Diagnostics: "
            f"cosine={report['cosine_similarity']:.12g}; "
            f"KL(P||Q)={report['kl_reference_candidate']:.12g}; "
            f"top10_jaccard={report.get('top_k', {}).get('10', {}).get('jaccard', 'n/a')}"
        )
    if report.get("metadata_mismatches"):
        print(f"Metadata mismatches: {len(report['metadata_mismatches'])}")
    return 0 if args.allow_drift else 1


if __name__ == "__main__":
    raise SystemExit(main())
