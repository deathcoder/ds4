#!/usr/bin/env python3
"""Import the final layer-0 attention-output tile of a 2K Metal prefill."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T18:32:02Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
LOW_WIDTH = 8_192
OUTPUT_WIDTH = 4_096
HC_WIDTH = 16_384
EXPECTED_FULL_SHA256 = {
    "attn_low": "edd7304f5f41313b19f432b4077c6bf08c97605c0bbaeb002e6afc749b768fa9",
    "attn_out": "99d1251d729592a383a208258bf96579c5025fb35ed3215be26d0616f2094871",
    "hc_attn_post": "19c0a248fce8b530bbc39f4c7e7ba0ff277b97466d8e2bc86ce199639b6739c1",
}
WIDTHS = {
    "attn_low": LOW_WIDTH,
    "attn_out": OUTPUT_WIDTH,
    "hc_attn_post": HC_WIDTH,
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first_payload) != expected_size or len(second_payload) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first_payload


def final_tile(payload: bytes, width: int) -> bytes:
    return payload[-TILE_ROWS * width * 4 :]


def tensor(name: str, hook: str, role: str, path: str, payload: bytes, width: int) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": "f32",
        "shape": [TILE_ROWS, width],
        "encoding": "little-endian-ieee754-binary32",
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("attn-low", "attn-out", "hc-attn-post"):
        parser.add_argument(f"--{name}-first", type=Path, required=True)
        parser.add_argument(f"--{name}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payloads = {
        "attn_low": final_tile(
            checked_repeated("attn_low", args.attn_low_first, args.attn_low_second),
            LOW_WIDTH,
        ),
        "attn_out": final_tile(
            checked_repeated("attn_out", args.attn_out_first, args.attn_out_second),
            OUTPUT_WIDTH,
        ),
        "hc_attn_post": final_tile(
            checked_repeated(
                "hc_attn_post", args.hc_attn_post_first, args.hc_attn_post_second
            ),
            HC_WIDTH,
        ),
    }
    template = json.loads(
        (args.fixtures_root / "prefill-attention-read-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-attention-output-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        "attn_low": "attn-low-final-tile.f32le.bin",
        "attn_out": "attn-out-final-tile.f32le.bin",
        "hc_attn_post": "hc-attn-post-final-tile.f32le.bin",
    }
    for name, payload in payloads.items():
        (output / files[name]).write_bytes(payload)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-attention-output-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_low,attn_out,hc_attn_post",
            },
            "device_path": (
                "legacy grouped Q8_0 batch matmul, legacy dense Q8_0 batch matmul, "
                "and four-stream HC expand; Metal 4 TensorOps unavailable and the "
                "Metal F16 attention-output shortcut returns unsupported"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-attention-read-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 0,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "grouped-attention-output-low-prefill",
                "kernel": "kernel_mul_mm_id_q8_0_f32",
                "weights": ["blk.0.attn_output_a.weight"],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "groups": 8,
                    "group_dim": 4_096,
                    "rank": 1_024,
                    "map_kernel": "kernel_mul_mm_id_map0_ne20_8",
                    "threads_per_threadgroup": [128, 1, 1],
                    "threadgroups": [16, 16, 1],
                    "threadgroup_memory_bytes": 8_192,
                },
            },
            {
                "name": "attention-output-prefill",
                "kernel": "kernel_mul_mm_q8_0_f32",
                "weights": ["blk.0.attn_output_b.weight"],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "input_elements": LOW_WIDTH,
                    "output_elements": OUTPUT_WIDTH,
                    "threads_per_threadgroup": [128, 1, 1],
                    "threadgroups": [1, 64, 1],
                    "threadgroup_memory_bytes": 6_144,
                },
            },
            {
                "name": "attention-hc-post-prefill",
                "kernel": "kernel_dsv4_hc_expand4",
                "weights": [],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "width": OUTPUT_WIDTH,
                    "hc_streams": 4,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [512, 1, 1],
                },
            },
        ],
        "tensors": [
            tensor(
                "attn_low_final_tile", "attn_low", "intermediate",
                files["attn_low"], payloads["attn_low"], LOW_WIDTH,
            ),
            tensor(
                "attn_out_final_tile", "attn_out", "intermediate",
                files["attn_out"], payloads["attn_out"], OUTPUT_WIDTH,
            ),
            tensor(
                "hc_attn_post_final_tile", "hc_attn_post", "output",
                files["hc_attn_post"], payloads["hc_attn_post"], HC_WIDTH,
            ),
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
