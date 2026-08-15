#!/usr/bin/env python3
"""Import repeated layer-0 batch KV finalization captures for the final tile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T19:11:35Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
KV_WIDTH = 512
ROTARY_WIDTH = 64
RAW_CACHE_ROWS = 128
RAW_CACHE_TARGET_START = 96
EXPECTED_FULL_SHA256 = {
    "KVrope": "9a9642491e6ae5018a5dc5012eac67884458d4439f58a8bedb071c82dc3aaeb7",
    "KVcur": "bef8d14d805a482960cbf7315ad0efccf211516a27bd767d0884b81f3ad33893",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    expected_size = PREFILL_ROWS * KV_WIDTH * 4
    if len(first_payload) != expected_size or len(second_payload) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first_payload


def final_tile(payload: bytes) -> bytes:
    return payload[-TILE_ROWS * KV_WIDTH * 4 :]


def f16_round_trip(payload: bytes) -> bytes:
    values = struct.iter_unpack("<f", payload)
    return b"".join(struct.pack("<f", struct.unpack("<e", struct.pack("<e", value))[0]) for value, in values)


def tensor(name: str, hook: str, role: str, path: str, payload: bytes) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": "f32",
        "shape": [TILE_ROWS, KV_WIDTH],
        "encoding": "little-endian-ieee754-binary32",
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("kv-rope", "kv-current"):
        parser.add_argument(f"--{name}-first", type=Path, required=True)
        parser.add_argument(f"--{name}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    rope_full = checked_repeated("KVrope", args.kv_rope_first, args.kv_rope_second)
    current_full = checked_repeated(
        "KVcur", args.kv_current_first, args.kv_current_second
    )
    rope_tile = final_tile(rope_full)
    current_tile = final_tile(current_full)
    cache_tile = f16_round_trip(current_tile)

    template = json.loads(
        (args.fixtures_root / "prefill-qkv-boundary-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-kv-state-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        "KVrope": "kv-rope-final-tile.f32le.bin",
        "KVcur": "kv-current-final-tile.f32le.bin",
        "cache": "raw-cache-final-tile.f32le.bin",
    }
    (output / files["KVrope"]).write_bytes(rope_tile)
    (output / files["KVcur"]).write_bytes(current_tile)
    (output / files["cache"]).write_bytes(cache_tile)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-kv-state-2048",
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
                "DS4_METAL_GRAPH_DUMP_NAME": "KVrope,KVcur",
            },
            "device_path": (
                "legacy Metal batch KV RoPE and E4M3FN simulation; Metal 4 "
                "TensorOps disabled on M1 Ultra"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 0,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
            "raw_cache_capacity_rows": RAW_CACHE_ROWS,
            "raw_cache_target_rows": [
                RAW_CACHE_TARGET_START,
                RAW_CACHE_TARGET_START + TILE_ROWS - 1,
            ],
            "raw_cache_guard_rows": [0, RAW_CACHE_TARGET_START - 1],
        },
        "operations": [
            {
                "name": "kv-rope-prefill",
                "kernel": "kernel_dsv4_rope_tail_f32",
                "dispatch": {
                    "rows": TILE_ROWS,
                    "heads": 1,
                    "head_dim": KV_WIDTH,
                    "rotary_dim": ROTARY_WIDTH,
                    "position_start": PREFILL_ROWS - TILE_ROWS,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [1, TILE_ROWS, 1],
                },
            },
            {
                "name": "kv-fp8-simulation-prefill",
                "kernel": "kernel_dsv4_fp8_kv_quantize_f32",
                "dispatch": {
                    "rows": TILE_ROWS,
                    "head_dim": KV_WIDTH,
                    "rotary_dim": ROTARY_WIDTH,
                    "threads_per_threadgroup": [64, 1, 1],
                    "threadgroups": [TILE_ROWS, 1, 1],
                },
            },
            {
                "name": "raw-cache-f32-to-f16",
                "kernel": "kernel_cpy_contig_f32_f16_4",
                "dispatch": {
                    "elements": TILE_ROWS * KV_WIDTH,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [TILE_ROWS * KV_WIDTH // (4 * 256), 1, 1],
                },
            },
            {
                "name": "raw-cache-f16-to-f32",
                "kernel": "kernel_cpy_contig_f16_f32_4",
                "dispatch": {
                    "elements": TILE_ROWS * KV_WIDTH,
                    "target_row_start": RAW_CACHE_TARGET_START,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [TILE_ROWS * KV_WIDTH // (4 * 256), 1, 1],
                },
            },
        ],
        "derivations": [
            {
                "output": "raw_cache_final_tile",
                "input": "kv_current_final_tile",
                "operation": "IEEE-754 binary32 to binary16 round-to-nearest-even, then binary16 to binary32",
                "reason": (
                    "DwarfStar exposes KVcur before raw-cache storage; its batch store "
                    "performs this exact F16 round trip before scattering rows into the ring"
                ),
            }
        ],
        "tensors": [
            tensor("kv_rope_final_tile", "KVrope", "intermediate", files["KVrope"], rope_tile),
            tensor("kv_current_final_tile", "KVcur", "output", files["KVcur"], current_tile),
            tensor("raw_cache_final_tile", "derived", "output", files["cache"], cache_tile),
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
