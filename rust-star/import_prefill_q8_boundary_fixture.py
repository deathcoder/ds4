#!/usr/bin/env python3
"""Import the final 128 rows at DwarfStar's first 2K prefill divergence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T16:40:02Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 128
INPUT_ELEMENTS = 4_096
OUTPUT_ELEMENTS = 1_024
EXPECTED_BATCH_INPUT_SHA256 = (
    "5a4c83daeb603f714146b93438a38672515e45c55f0776527e76c86db99476b4"
)
EXPECTED_BATCH_OUTPUT_SHA256 = (
    "ce24ef63d0604b73769c7fd1141d70a25e85ae88b6c2e6770d649fdf40b0b62e"
)
EXPECTED_DECODE_OUTPUT_SHA256 = (
    "fad25edc722941f86db6549a380cf15c6b434137d73d66fcbc45bb7d7f8e94f1"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(first: Path, second: Path, size: int, digest: str) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    if len(first_payload) != size or len(second_payload) != size:
        raise SystemExit(f"capture has the wrong size: {first} or {second}")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process captures differ: {first} and {second}")
    if sha256(first_payload) != digest:
        raise SystemExit(f"capture identity changed: {first}")
    return first_payload


def tensor(name: str, hook: str, role: str, shape: list[int], path: str, payload: bytes) -> dict:
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
    parser.add_argument("first_batch_input", type=Path)
    parser.add_argument("second_batch_input", type=Path)
    parser.add_argument("first_batch_output", type=Path)
    parser.add_argument("second_batch_output", type=Path)
    parser.add_argument("decode_input", type=Path)
    parser.add_argument("decode_output", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    full_input = checked_repeated(
        args.first_batch_input,
        args.second_batch_input,
        PREFILL_ROWS * INPUT_ELEMENTS * 4,
        EXPECTED_BATCH_INPUT_SHA256,
    )
    full_output = checked_repeated(
        args.first_batch_output,
        args.second_batch_output,
        PREFILL_ROWS * OUTPUT_ELEMENTS * 4,
        EXPECTED_BATCH_OUTPUT_SHA256,
    )
    decode_input = args.decode_input.read_bytes()
    decode_output = args.decode_output.read_bytes()
    if len(decode_input) != INPUT_ELEMENTS * 4:
        raise SystemExit("decode input has the wrong size")
    if len(decode_output) != OUTPUT_ELEMENTS * 4:
        raise SystemExit("decode output has the wrong size")
    if sha256(decode_output) != EXPECTED_DECODE_OUTPUT_SHA256:
        raise SystemExit("decode output identity changed")

    input_tile = full_input[-TILE_ROWS * INPUT_ELEMENTS * 4 :]
    batch_output_tile = full_output[-TILE_ROWS * OUTPUT_ELEMENTS * 4 :]
    if input_tile[-INPUT_ELEMENTS * 4 :] != decode_input:
        raise SystemExit("batched and sequential schedules do not share the final input row")
    final_batch_output = batch_output_tile[-OUTPUT_ELEMENTS * 4 :]
    batch_values = struct.unpack(f"<{OUTPUT_ELEMENTS}f", final_batch_output)
    decode_values = struct.unpack(f"<{OUTPUT_ELEMENTS}f", decode_output)
    mismatches = sum(
        struct.pack("<f", batch) != struct.pack("<f", decode)
        for batch, decode in zip(batch_values, decode_values)
    )
    max_abs_error = max(abs(batch - decode) for batch, decode in zip(batch_values, decode_values))
    if mismatches != OUTPUT_ELEMENTS or max_abs_error != 3.62396240234375e-05:
        raise SystemExit("captured batch/decode arithmetic boundary changed")

    template = json.loads(
        (args.fixtures_root / "q8-attn-q-a-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-q8-boundary-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    paths = {
        "attn-norm-final-tile.f32le.bin": input_tile,
        "q-lora-batch-final-tile.f32le.bin": batch_output_tile,
        "q-lora-decode-final-row.f32le.bin": decode_output,
    }
    for path, payload in paths.items():
        (output / path).write_bytes(payload)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-q8-boundary-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": "first 2048 raw tokens of speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": "ds4-bench --metal --ctx-start 2048 --ctx-max 2048 --gen-tokens 0 --warm-weights",
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_norm,q_lora",
            },
            "device_path": "legacy Metal matmul; Metal 4 TensorOps disabled on M1 Ultra",
            "sequential_control": {
                "construction": "one-token cold prefill followed by 2047 ds4_session_eval calls",
                "final_input_equals_batch_final_input": True,
                "final_output_equals_batch_final_output": False,
            },
        },
        "scope": {
            "kind": "kernel",
            "phase": "prefill",
            "layer": 0,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "attn-q-a-prefill-projection-final-tile",
                "kernel": "kernel_mul_mm_q8_0_f32",
                "weights": ["blk.0.attn_q_a.weight"],
                "dispatch": {
                    "input_elements": INPUT_ELEMENTS,
                    "output_elements": OUTPUT_ELEMENTS,
                    "rows": TILE_ROWS,
                    "threads_per_threadgroup": [128, 1, 1],
                    "threadgroups": [4, 16, 1],
                    "threadgroup_memory_bytes": 6144,
                },
            },
            {
                "name": "attn-q-a-sequential-final-row-control",
                "kernel": "kernel_mul_mv_q8_0_f32",
                "weights": ["blk.0.attn_q_a.weight"],
            },
        ],
        "arithmetic_boundary": {
            "final_row_mismatches": mismatches,
            "final_row_elements": OUTPUT_ELEMENTS,
            "max_abs_error": max_abs_error,
        },
        "tensors": [
            tensor(
                "attn_norm_final_tile",
                "attn_norm",
                "input",
                [TILE_ROWS, INPUT_ELEMENTS],
                "attn-norm-final-tile.f32le.bin",
                input_tile,
            ),
            tensor(
                "q_lora_batch_final_tile",
                "q_lora",
                "output",
                [TILE_ROWS, OUTPUT_ELEMENTS],
                "q-lora-batch-final-tile.f32le.bin",
                batch_output_tile,
            ),
            tensor(
                "q_lora_decode_final_row",
                "q_lora",
                "output",
                [OUTPUT_ELEMENTS],
                "q-lora-decode-final-row.f32le.bin",
                decode_output,
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
