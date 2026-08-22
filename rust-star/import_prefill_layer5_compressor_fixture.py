#!/usr/bin/env python3
"""Import repeated full-2K layer-5 ratio-128 compressor captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-22T17:21:14Z"
CAPTURE_NAMES = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "e534398a1e4884db4b451318b09512220714663ad31dea51a2d7bc6197198386",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [128, 512],
        262_144,
        "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [128, 512],
        262_144,
        "6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778",
    ),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, hook: str) -> Path:
    return root / f"{root.name}_{hook}-5_pos0.bin"


def checked_repeated(hook: str, first: Path, second: Path) -> bytes:
    filename, _shape, expected_bytes, expected_sha = CAPTURE_NAMES[hook]
    first_payload = capture_path(first, hook).read_bytes()
    second_payload = capture_path(second, hook).read_bytes()
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"layer-5 {hook} capture has the wrong size for {filename}")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process layer-5 {hook} captures differ")
    if sha256(first_payload) != expected_sha:
        raise SystemExit(f"layer-5 {hook} capture identity changed")
    return first_payload


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

    captures = {
        hook: checked_repeated(hook, args.first, args.second)
        for hook in CAPTURE_NAMES
    }
    template = json.loads(
        (args.fixtures_root / "prefill-layer5-qkv-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer5-compressor-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for hook, payload in captures.items():
        filename, shape, _expected_bytes, _expected_sha = CAPTURE_NAMES[hook]
        (output / filename).write_bytes(payload)
        score_bits = hook == "attn_state_score"
        tensors.append(
            {
                "name": hook.lower() + ("_bits" if score_bits else ""),
                "hook": hook,
                "role": "output",
                "dtype": "i32" if score_bits else "f32",
                "shape": shape,
                "encoding": (
                    "little-endian-signed-integer32"
                    if score_bits
                    else "little-endian-ieee754-binary32"
                ),
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer5-compressor-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 2048,
            "fresh_process_captures_per_hook": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "5",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(CAPTURE_NAMES),
            },
            "device_path": (
                "M1 legacy batched F16 paired projection, ratio-128 pooling, "
                "learned norm, compressed RoPE, E4M3FN finalization, and "
                "recurrent-state ownership"
            ),
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer5-qkv-2048",
            "state_note": (
                "The aligned 2K batch leaves the terminal KV/score scratch in "
                "canonical +0.0/-infinity reset state."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 5,
            "position": 2047,
            "captured_position_range": [0, 2047],
            "compressed_rows": 16,
        },
        "operations": [
            {
                "name": "attention-compressor-paired-batch-projections",
                "kernel": "kernel_mul_mm_f16_f32",
            },
            {
                "name": "attention-compressor-ratio128-pool",
                "kernel": "legacy concat+softmax+mul+sum",
            },
            {
                "name": "attention-compressor-finalize",
                "kernel": "weighted RMSNorm+RoPE+E4M3FN",
            },
        ],
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
