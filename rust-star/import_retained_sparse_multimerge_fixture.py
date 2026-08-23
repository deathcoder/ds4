#!/usr/bin/env python3
"""Import the exact retained layer-2 row-2,049 repeated-merge boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


POSITION = 8195
TOKEN = 381
LAYER = 2
RAW_ROWS = 128
PRIOR_RAW_ROWS = 127
COMPRESSED_ROWS = 2049
PRIOR_COMPRESSED_ROWS = 2048
RAW_CAPTURE_ROWS = 4352
CAPTURED_AT_UTC = "2026-08-23T11:10:22Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "28bd016423a25e1a5426a349ac2bf3728b6352808545f04a3af8b8a4f48097f1"
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


def capture_name(hook: str, dtype: str) -> str:
    suffix = ".i32" if dtype == "i32" else ".bin"
    return f"capture_{hook}-{LAYER}_pos{POSITION}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_capture", type=Path)
    parser.add_argument("second_capture", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    template = json.loads(
        (
            args.fixtures_root
            / "retained-sparse-layer2-pos4099-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "retained-sparse-layer2-pos8195-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    raw_capture = repeated(
        args.first_capture,
        args.second_capture,
        capture_name("raw_cache", "f32"),
        RAW_CAPTURE_ROWS * 512 * 4,
    )
    raw_prior = bytearray()
    for logical_position in range(POSITION - PRIOR_RAW_ROWS, POSITION):
        slot = logical_position % RAW_CAPTURE_ROWS
        start = slot * 512 * 4
        raw_prior.extend(raw_capture[start : start + 512 * 4])

    attention_cache = repeated(
        args.first_capture,
        args.second_capture,
        capture_name("attention_comp_cache", "f32"),
        COMPRESSED_ROWS * 512 * 4,
    )
    indexer_cache = repeated(
        args.first_capture,
        args.second_capture,
        capture_name("indexer_comp_cache", "f32"),
        COMPRESSED_ROWS * 128 * 4,
    )

    tensors = []
    for source in template["tensors"]:
        name = source["name"]
        hook = source["hook"]
        role = source["role"]
        dtype = source["dtype"]
        shape = copy.deepcopy(source["shape"])
        if name == "raw_cache_prior":
            payload = bytes(raw_prior)
        elif name == "attention_compressed_prior":
            shape = [PRIOR_COMPRESSED_ROWS, 512]
            payload = attention_cache[: PRIOR_COMPRESSED_ROWS * 512 * 4]
        elif name == "indexer_compressed_prior":
            shape = [PRIOR_COMPRESSED_ROWS, 128]
            payload = indexer_cache[: PRIOR_COMPRESSED_ROWS * 128 * 4]
        elif name == "compressed_indexer_row1024":
            name = "compressed_indexer_row2048"
            start = PRIOR_COMPRESSED_ROWS * 128 * 4
            payload = indexer_cache[start:]
        else:
            if name == "indexer_scores":
                shape = [COMPRESSED_ROWS]
            capture_dtype = "f32" if name.endswith("_score_pre_bits") else dtype
            expected_bytes = 4
            for dimension in shape:
                expected_bytes *= dimension
            payload = repeated(
                args.first_capture,
                args.second_capture,
                capture_name(hook, capture_dtype),
                expected_bytes,
            )
        tensors.append(
            tensor(
                output,
                name=name,
                hook=hook,
                role=role,
                dtype=dtype,
                shape=shape,
                payload=payload,
            )
        )

    operations = copy.deepcopy(template["operations"])
    for operation in operations:
        if operation["name"] == "two-block-top512":
            operation["name"] = "three-block-two-pass-top512"

    manifest = {
        **copy.deepcopy(template),
        "fixture_id": "dwarfstar-oracle-v1-retained-layer2-pos8195-sparse-multimerge",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": {
            **copy.deepcopy(template["oracle"]),
            "capture_executable_sha256": CAPTURE_EXECUTABLE_SHA256,
        },
        "capture": {
            **copy.deepcopy(template["capture"]),
            "prefill_tokens": 8192,
            "captured_position": POSITION,
            "captured_token": TOKEN,
            "token_fresh_process_captures": 2,
            "token_fresh_process_match": True,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "excluded_nondeterministic_scratch_hooks": [
                "ffn_moe_gate_clamped",
                "ffn_moe_up_clamped",
            ],
            "temporary_seed_hooks_removed_after_capture": True,
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "decode",
            "layer": LAYER,
            "position": POSITION,
            "raw_rows": RAW_ROWS,
            "raw_capture_capacity_rows": RAW_CAPTURE_ROWS,
            "prior_raw_rows": PRIOR_RAW_ROWS,
            "compressed_rows": COMPRESSED_ROWS,
            "prior_compressed_rows": PRIOR_COMPRESSED_ROWS,
            "sort_blocks": 3,
            "merge_passes": 2,
            "top_k": 512,
        },
        "operations": operations,
        "claims": {
            **copy.deepcopy(template["claims"]),
            "repeated_merge_boundary": True,
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
