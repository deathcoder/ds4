#!/usr/bin/env python3
"""Import the exact retained layer-2 seed and row-1,025 sparse boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


POSITION = 4099
LAYER = 2
RAW_ROWS = 128
PRIOR_RAW_ROWS = 127
PRIOR_COMPRESSED_ROWS = 1024
CAPTURED_AT_UTC = "2026-08-23T12:08:00Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "c9f32f4c36cc5cc3562939de7adc96cf6a466a86997b8bf8ecedc36d81353c94"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated(first: Path, second: Path, name: str, expected_bytes: int) -> bytes:
    left = (first / name).read_bytes()
    right = (second / name).read_bytes()
    if len(left) != expected_bytes:
        raise SystemExit(f"{name} has {len(left)} bytes, expected {expected_bytes}")
    if left != right:
        raise SystemExit(f"fresh-process captures differ: {name}")
    return left


def tensor(
    output: Path,
    *,
    name: str,
    hook: str,
    role: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> dict:
    suffix = "i32le.bin" if dtype == "i32" else "f32le.bin"
    destination = output / f"{name.replace('_', '-')}.{suffix}"
    destination.write_bytes(payload)
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "encoding": (
            "little-endian-signed-integer32"
            if dtype == "i32"
            else "little-endian-ieee754-binary32"
        ),
        "path": destination.name,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_capture", type=Path)
    parser.add_argument("second_capture", type=Path)
    parser.add_argument("seed_first", type=Path)
    parser.add_argument("seed_second", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    template = json.loads(
        (args.fixtures_root / "layer2-pos3-complete-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "retained-sparse-layer2-pos4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    seed_specs = [
        ("retained_input_hc", "retained_input_hc", "f32", [4, 4096]),
        ("attention_state_kv_pre", "attn_state_kv_pre", "f32", [8, 1024]),
        # Unoccupied recurrent score slots are -infinity. Store their exact
        # bit patterns as i32 so the finite-FP32 differential envelope remains
        # strict while the runtime can reinterpret the payload during seeding.
        ("attention_state_score_pre_bits", "attn_state_score_pre", "i32", [8, 1024]),
        ("indexer_state_kv_pre", "indexer_state_kv_pre", "f32", [8, 256]),
        ("indexer_state_score_pre_bits", "indexer_state_score_pre", "i32", [8, 256]),
    ]
    for name, hook, dtype, shape in seed_specs:
        elements = 1
        for dimension in shape:
            elements *= dimension
        payload = repeated(
            args.seed_first,
            args.seed_second,
            f"capture_{hook}-2_pos4099.bin",
            elements * 4,
        )
        tensors.append(
            tensor(
                output,
                name=name,
                hook=hook,
                role="input",
                dtype=dtype,
                shape=shape,
                payload=payload,
            )
        )

    raw_full = repeated(
        args.first_capture,
        args.second_capture,
        "capture_raw_cache-2_pos4099.bin",
        4101 * 512 * 4,
    )
    raw_start = POSITION + 1 - RAW_ROWS
    raw_prior = raw_full[raw_start * 512 * 4 : POSITION * 512 * 4]
    tensors.append(
        tensor(
            output,
            name="raw_cache_prior",
            hook="raw_cache",
            role="input",
            dtype="f32",
            shape=[PRIOR_RAW_ROWS, 512],
            payload=raw_prior,
        )
    )

    attention_cache = repeated(
        args.first_capture,
        args.second_capture,
        "capture_attention_comp_cache-2_pos4099.bin",
        1025 * 512 * 4,
    )
    indexer_cache = repeated(
        args.first_capture,
        args.second_capture,
        "capture_indexer_comp_cache-2_pos4099.bin",
        1025 * 128 * 4,
    )
    tensors.extend(
        [
            tensor(
                output,
                name="attention_compressed_prior",
                hook="attention_comp_cache",
                role="input",
                dtype="f32",
                shape=[PRIOR_COMPRESSED_ROWS, 512],
                payload=attention_cache[: PRIOR_COMPRESSED_ROWS * 512 * 4],
            ),
            tensor(
                output,
                name="indexer_compressed_prior",
                hook="indexer_comp_cache",
                role="input",
                dtype="f32",
                shape=[PRIOR_COMPRESSED_ROWS, 128],
                payload=indexer_cache[: PRIOR_COMPRESSED_ROWS * 128 * 4],
            ),
            tensor(
                output,
                name="compressed_indexer_row1024",
                hook="indexer_comp_cache",
                role="output",
                dtype="f32",
                shape=[128],
                payload=indexer_cache[PRIOR_COMPRESSED_ROWS * 128 * 4 :],
            ),
        ]
    )

    for source in template["tensors"]:
        if source["name"] == "cache_row0":
            continue
        hook = source["hook"]
        suffix = ".i32" if source["dtype"] == "i32" else ".bin"
        expected_bytes = 4
        for dimension in source["shape"]:
            expected_bytes *= dimension
        payload = repeated(
            args.first_capture,
            args.second_capture,
            f"capture_{hook}-2_pos4099{suffix}",
            expected_bytes,
        )
        tensors.append(
            tensor(
                output,
                name=source["name"],
                hook=hook,
                role=source["role"],
                dtype=source["dtype"],
                shape=source["shape"],
                payload=payload,
            )
        )

    sparse_specs = [
        ("indexer_q", "indexer_q", "f32", [64, 128]),
        ("indexer_weights", "indexer_weights", "f32", [64]),
        ("indexer_scores", "indexer_scores", "f32", [1025]),
        ("indexer_topk", "indexer_topk", "i32", [512]),
        ("kqv_out", "kqv_out", "f32", [64, 512]),
    ]
    for name, hook, dtype, shape in sparse_specs:
        elements = 1
        for dimension in shape:
            elements *= dimension
        suffix = ".i32" if dtype == "i32" else ".bin"
        payload = repeated(
            args.first_capture,
            args.second_capture,
            f"capture_{hook}-2_pos4099{suffix}",
            elements * 4,
        )
        tensors.append(
            tensor(
                output,
                name=name,
                hook=hook,
                role="intermediate" if name.startswith("indexer_") else "output",
                dtype=dtype,
                shape=shape,
                payload=payload,
            )
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-retained-layer2-pos4099-sparse",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": {
            **copy.deepcopy(template["oracle"]),
            "capture_executable_sha256": CAPTURE_EXECUTABLE_SHA256,
        },
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefill_tokens": 4096,
            "decode_steps": 4,
            "captured_position": POSITION,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "temporary_seed_hooks_removed_after_capture": True,
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "decode",
            "layer": LAYER,
            "position": POSITION,
            "raw_rows": RAW_ROWS,
            "prior_raw_rows": PRIOR_RAW_ROWS,
            "compressed_rows": 1025,
            "prior_compressed_rows": PRIOR_COMPRESSED_ROWS,
            "top_k": 512,
        },
        "operations": [
            {"name": "retained-state-seed", "kernel": "host-to-shared-Metal-state"},
            *copy.deepcopy(template["operations"]),
            {"name": "default-sparse-indexer", "kernel": "kernel_dsv4_indexer_score_one_direct"},
            {"name": "two-block-top512", "kernel": "kernel_argsort_f32_i32_desc+kernel_argsort_merge_f32_i32_desc"},
            {"name": "indexed-mixed-attention", "kernel": "kernel_dsv4_indexed_mixed_attention_heads8_split+reduce"},
        ],
        "claims": {
            "retained_layer_execution": True,
            "default_threshold_boundary": True,
            "complete_decoder": False,
            "output_logits": False,
            "throughput": False,
        },
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
