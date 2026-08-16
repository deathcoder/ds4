#!/usr/bin/env python3
"""Import the full native-batch layer-2 normalized-KV oracle at 2K."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-16T06:51:54Z"
PREFILL_ROWS = 2_048
KV_WIDTH = 512
EXPECTED_BYTES = PREFILL_ROWS * KV_WIDTH * 4
EXPECTED_SHA256 = "089138d8fc82c1eb55754451707f59475f2afb2a356dcc505314ddf29814e7b6"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    first = args.first.read_bytes()
    second = args.second.read_bytes()
    if len(first) != EXPECTED_BYTES or len(second) != EXPECTED_BYTES:
        raise SystemExit("layer-2 KVnorm capture has the wrong size")
    if first != second:
        raise SystemExit("fresh-process layer-2 KVnorm captures differ")
    if sha256(first) != EXPECTED_SHA256:
        raise SystemExit("layer-2 KVnorm capture identity changed")

    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer1-complete-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer2-kvnorm-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    payload_name = "layer2-kv-norm.f32le.bin"
    (output / payload_name).write_bytes(first)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer2-kvnorm-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "KVnorm",
            },
            "device_path": (
                "native layers-0/1 batched prefill feeding layer-2 HC ingress, "
                "learned attention norm, Q8_0 KV projection, and fused KV RMSNorm"
            ),
            "full_capture_sha256": {"KVnorm": EXPECTED_SHA256},
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer1-complete-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [0, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "layer2-attention-ingress-kvnorm",
                "kernel": (
                    "HC batch projection/collapse plus kernel_mul_mm_q8_0_f32 "
                    "and kernel_dsv4_qkv_rms_norm_f32_4"
                ),
            }
        ],
        "tensors": [
            {
                "name": "layer2_kv_norm",
                "hook": "KVnorm",
                "role": "output",
                "dtype": "f32",
                "shape": [PREFILL_ROWS, KV_WIDTH],
                "encoding": "little-endian-ieee754-binary32",
                "path": payload_name,
                "bytes": len(first),
                "sha256": sha256(first),
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
