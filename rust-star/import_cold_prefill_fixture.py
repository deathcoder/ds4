#!/usr/bin/env python3
"""Import two repeated one-token cold-prefill logit captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T12:59:25Z"
EXPECTED_LOGITS = 129_280
EXPECTED_TOKEN = 201
EXPECTED_LOGITS_SHA256 = (
    "a4973c1e1f53bf1659a9a15e66c3186d03432810e7606387c55f8ee083ffba35"
)
PROMPT_SHA256 = "99871ea61492b0fb650c49fe1dd2be7d81c8074542514671d2c585f4a3f247f1"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_capture(run: Path) -> bytes:
    logits = (run / "prefill-logits.f32le.bin").read_bytes()
    if len(logits) != EXPECTED_LOGITS * 4:
        raise SystemExit(f"invalid cold-prefill logits size in {run}")
    values = struct.unpack(f"<{EXPECTED_LOGITS}f", logits)
    selected = max(range(EXPECTED_LOGITS), key=values.__getitem__)
    if selected != EXPECTED_TOKEN:
        raise SystemExit(
            f"cold-prefill capture in {run} selected {selected}, expected {EXPECTED_TOKEN}"
        )
    return logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    first = read_capture(args.capture_root / "a")
    second = read_capture(args.capture_root / "b")
    if first != second:
        raise SystemExit("independent cold-prefill captures differ")
    if sha256(first) != EXPECTED_LOGITS_SHA256:
        raise SystemExit("cold-prefill logits identity changed")

    template = json.loads(
        (args.fixtures_root / "output-head-pos4-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "cold-prefill-pos0-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    shutil.copyfile(
        args.capture_root / "a" / "prefill-logits.f32le.bin",
        output / "logits.f32le.bin",
    )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-cold-prefill-pos0",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": "raw 26-space prefix of speed-bench/promessi_sposi.txt",
            "prompt_sha256": PROMPT_SHA256,
            "prompt_token_ids": [36662],
            "prefill_tokens": 1,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_DUMP_PREFILL_LOGITS": "prefill-logits.f32le.bin"
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "prefill",
            "layer": 43,
            "position": 0,
        },
        "operations": [
            {
                "name": "cold-prefill-full-logits",
                "kernel": "kernel_mul_mv_q8_0_f32",
                "weights": ["output.weight"],
            }
        ],
        "selection": {
            "method": "lowest-token-id-argmax",
            "token_id": EXPECTED_TOKEN,
        },
        "tensors": [
            {
                "name": "logits",
                "hook": "prefill_logits",
                "role": "output",
                "dtype": "f32",
                "shape": [EXPECTED_LOGITS],
                "encoding": "little-endian-ieee754-binary32",
                "path": "logits.f32le.bin",
                "bytes": len(first),
                "sha256": EXPECTED_LOGITS_SHA256,
            }
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
