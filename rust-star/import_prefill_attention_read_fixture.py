#!/usr/bin/env python3
"""Import the repeatable layer-0 zero-prefix batch attention-read boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T18:03:09Z"
PREFILL_ROWS = 2_048
PREFIX_ROWS = 2_016
TILE_ROWS = 32
KV_WIDTH = 512
HEADS = 64
HEAD_WIDTH = 512
ATTENTION_WIDTH = HEADS * HEAD_WIDTH
WINDOW = 128
EXPECTED_FULL_SHA256 = {
    "KVcur": "bef8d14d805a482960cbf7315ad0efccf211516a27bd767d0884b81f3ad33893",
    "kqv_out": "0678586b3fa811f40d053b4cef4f40c9172bb497d0ac1862dc43de7f5b04a1d9",
    "kqv_back": "79ec161cba188d6e9d1e0b3a84e3071e6c0cba53b5b7a56406d9ac6cf8ddabcc",
}
WIDTHS = {
    "KVcur": KV_WIDTH,
    "kqv_out": ATTENTION_WIDTH,
    "kqv_back": ATTENTION_WIDTH,
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


def tensor(
    name: str,
    hook: str,
    role: str,
    path: str,
    payload: bytes,
    shape: list[int],
) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": "f32",
        "shape": shape,
        "encoding": "little-endian-ieee754-binary32",
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("kv-current", "kqv-output", "kqv-back"):
        parser.add_argument(f"--{name}-first", type=Path, required=True)
        parser.add_argument(f"--{name}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    kv_full = checked_repeated(
        "KVcur", args.kv_current_first, args.kv_current_second
    )
    output_full = checked_repeated(
        "kqv_out", args.kqv_output_first, args.kqv_output_second
    )
    back_full = checked_repeated(
        "kqv_back", args.kqv_back_first, args.kqv_back_second
    )
    kv_prefix = kv_full[: PREFIX_ROWS * KV_WIDTH * 4]
    output_tile = final_tile(output_full, ATTENTION_WIDTH)
    back_tile = final_tile(back_full, ATTENTION_WIDTH)

    template = json.loads(
        (args.fixtures_root / "prefill-kv-state-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-attention-read-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        "KVcur": "kv-current-prefix.f32le.bin",
        "kqv_out": "kqv-output-final-tile.f32le.bin",
        "kqv_back": "kqv-back-final-tile.f32le.bin",
    }
    (output / files["KVcur"]).write_bytes(kv_prefix)
    (output / files["kqv_out"]).write_bytes(output_tile)
    (output / files["kqv_back"]).write_bytes(back_tile)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-attention-read-2048",
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
                "DS4_METAL_GRAPH_DUMP_NAME": "kqv_out,kqv_back",
            },
            "device_path": (
                "legacy Metal rectangular non-vector raw-head FlashAttention plus "
                "inverse RoPE; Metal 4 TensorOps disabled on M1 Ultra"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "kv_prefix_source": "the repeated full KVcur capture from prefill-kv-state-2048-v1",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 0,
            "position": PREFILL_ROWS - 1,
            "query_position_range": [PREFIX_ROWS, PREFILL_ROWS - 1],
            "kv_position_range": [0, PREFILL_ROWS - 1],
            "retained_kv_prefix_range": [0, PREFIX_ROWS - 1],
            "live_kv_tile_range": [PREFIX_ROWS, PREFILL_ROWS - 1],
            "attention_window": WINDOW,
        },
        "operations": [
            {
                "name": "assemble-contiguous-kv",
                "kernel": "MTLBlitCommandEncoder.copyFromBuffer",
                "dispatch": {
                    "captured_prefix_rows": PREFIX_ROWS,
                    "live_tile_rows": TILE_ROWS,
                    "head_dim": KV_WIDTH,
                },
            },
            {
                "name": "raw-kv-f32-to-f16",
                "kernel": "kernel_cpy_contig_f32_f16_4",
                "dispatch": {
                    "elements": PREFILL_ROWS * KV_WIDTH,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [PREFILL_ROWS * KV_WIDTH // (4 * 256), 1, 1],
                },
            },
            {
                "name": "raw-flash-attention-block-map",
                "kernel": "kernel_flash_attn_ext_blk",
                "dispatch": {
                    "query_rows": TILE_ROWS,
                    "kv_rows": PREFILL_ROWS,
                    "queries_per_threadgroup": 8,
                    "kv_columns_per_simdgroup": 64,
                    "threads_per_threadgroup": [32, 1, 1],
                    "threadgroups": [32, 4, 1],
                },
            },
            {
                "name": "raw-flash-attention-nonvector",
                "kernel": "kernel_flash_attn_ext_f16_dk512_dv512",
                "weights": ["blk.0.attn_sinks.weight"],
                "dispatch": {
                    "query_rows": TILE_ROWS,
                    "query_row_start": PREFIX_ROWS,
                    "kv_rows": PREFILL_ROWS,
                    "heads": HEADS,
                    "head_dim": HEAD_WIDTH,
                    "attention_window": WINDOW,
                    "simdgroups": 8,
                    "threads_per_threadgroup": [32, 8, 1],
                    "threadgroups": [4, HEADS, 1],
                    "threadgroup_memory_bytes": 28_672,
                },
            },
            {
                "name": "attention-inverse-rope",
                "kernel": "kernel_dsv4_rope_tail_f32",
                "dispatch": {
                    "rows": TILE_ROWS,
                    "heads": HEADS,
                    "head_dim": HEAD_WIDTH,
                    "rotary_dim": 64,
                    "position_start": PREFIX_ROWS,
                    "inverse": True,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [HEADS, TILE_ROWS, 1],
                },
            },
        ],
        "tensors": [
            tensor(
                "kv_current_prefix",
                "KVcur",
                "input",
                files["KVcur"],
                kv_prefix,
                [PREFIX_ROWS, KV_WIDTH],
            ),
            tensor(
                "kqv_output_final_tile",
                "kqv_out",
                "intermediate",
                files["kqv_out"],
                output_tile,
                [TILE_ROWS, HEADS, HEAD_WIDTH],
            ),
            tensor(
                "kqv_back_final_tile",
                "kqv_back",
                "output",
                files["kqv_back"],
                back_tile,
                [TILE_ROWS, HEADS, HEAD_WIDTH],
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
