#!/usr/bin/env python3
"""Import the final layer-0 HC-ingress tile of a 2K native Metal prefill."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T16:37:17Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
N_EMBD = 4_096
EXPECTED_FULL_SHA256 = {
    "hc_attn_pre": "37b12a76bdedb54375492a6785dd5813b924e631280cd0a1324b7a5d8e0b7290",
    "attn_norm": "5a4c83daeb603f714146b93438a38672515e45c55f0776527e76c86db99476b4",
    "token_ids": "5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    expected_size = PREFILL_ROWS * N_EMBD * 4
    if len(first_payload) != expected_size or len(second_payload) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first_payload


def tensor(name: str, hook: str, role: str, path: str, payload: bytes, width: int) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": "i32" if hook == "tokenizer" else "f32",
        "shape": [TILE_ROWS] if width == 1 else [TILE_ROWS, width],
        "encoding": (
            "little-endian-signed-integer32"
            if hook == "tokenizer"
            else "little-endian-ieee754-binary32"
        ),
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hc-attn-pre-first", type=Path, required=True)
    parser.add_argument("--hc-attn-pre-second", type=Path, required=True)
    parser.add_argument("--attn-norm-first", type=Path, required=True)
    parser.add_argument("--attn-norm-second", type=Path, required=True)
    parser.add_argument("--token-ids", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    collapsed = checked_repeated(
        "hc_attn_pre", args.hc_attn_pre_first, args.hc_attn_pre_second
    )
    attn_norm = checked_repeated(
        "attn_norm", args.attn_norm_first, args.attn_norm_second
    )
    token_ids = args.token_ids.read_bytes()
    if len(token_ids) != PREFILL_ROWS * 4:
        raise SystemExit("token-ID artifact has the wrong size")
    if sha256(token_ids) != EXPECTED_FULL_SHA256["token_ids"]:
        raise SystemExit("token-ID artifact identity changed")

    token_tile = token_ids[-TILE_ROWS * 4 :]
    collapsed_tile = collapsed[-TILE_ROWS * N_EMBD * 4 :]
    attn_norm_tile = attn_norm[-TILE_ROWS * N_EMBD * 4 :]
    template = json.loads(
        (args.fixtures_root / "prefill-qkv-boundary-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-hc-ingress-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        "tokens": "token-ids-final-tile.i32le.bin",
        "collapsed": "hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "attn-norm-final-tile.f32le.bin",
    }
    (output / files["tokens"]).write_bytes(token_tile)
    (output / files["collapsed"]).write_bytes(collapsed_tile)
    (output / files["attn_norm"]).write_bytes(attn_norm_tile)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-hc-ingress-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prompt_token_ids_sha256": EXPECTED_FULL_SHA256["token_ids"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "hc_attn_pre,attn_norm",
            },
            "device_path": (
                "F16 embedding gather, HC repeat, plain row RMSNorm, legacy F16 "
                "batch matmul, and fused HC split/weighted-sum/learned RMSNorm"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
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
                "name": "token-embedding-prefill",
                "kernel": "kernel_get_rows_f16",
                "weights": ["token_embd.weight"],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "input_elements": 1,
                    "output_elements": N_EMBD,
                },
            },
            {
                "name": "hc-repeat-prefill",
                "kernel": "kernel_repeat_f32",
                "weights": [],
                "dispatch": {"rows": TILE_ROWS, "hc_streams": 4, "width": N_EMBD},
            },
            {
                "name": "hc-plain-rmsnorm-prefill",
                "kernel": "kernel_rms_norm_f32_4",
                "weights": [],
                "dispatch": {"rows": TILE_ROWS, "width": 4 * N_EMBD},
            },
            {
                "name": "hc-mixer-prefill",
                "kernel": "kernel_mul_mm_f16_f32",
                "weights": ["blk.0.hc_attn_fn.weight"],
                "dispatch": {
                    "input_elements": 4 * N_EMBD,
                    "output_elements": 24,
                    "rows": TILE_ROWS,
                    "threads_per_threadgroup": [128, 1, 1],
                    "threadgroups": [1, 1, 1],
                    "threadgroup_memory_bytes": 8192,
                },
            },
            {
                "name": "hc-split-collapse-attn-norm-prefill",
                "kernel": "kernel_dsv4_hc_split_weighted_sum_norm4",
                "weights": [
                    "blk.0.hc_attn_scale.weight",
                    "blk.0.hc_attn_base.weight",
                    "blk.0.attn_norm.weight",
                ],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "width": N_EMBD,
                    "threads_per_threadgroup": [1024, 1, 1],
                    "threadgroups": [TILE_ROWS, 1, 1],
                    "threadgroup_memory_bytes": 16512,
                },
            },
        ],
        "tensors": [
            tensor("token_ids_final_tile", "tokenizer", "input", files["tokens"], token_tile, 1),
            tensor(
                "hc_attn_pre_final_tile",
                "hc_attn_pre",
                "intermediate",
                files["collapsed"],
                collapsed_tile,
                N_EMBD,
            ),
            tensor(
                "attn_norm_final_tile",
                "attn_norm",
                "output",
                files["attn_norm"],
                attn_norm_tile,
                N_EMBD,
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
