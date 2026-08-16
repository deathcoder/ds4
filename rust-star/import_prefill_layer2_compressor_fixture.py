#!/usr/bin/env python3
"""Import repeated full-2K layer-2 paired-compressor captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-16T07:59:00Z"
CAPTURE_NAMES = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "d82172dd21cc354819ce02d8665ce30682ba9efbf7eeb2b4132c9b94a6805c8a"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "39b7ee5d316b026823997cbf01ca366d7d2f8b8127bca4c44047320cdb7667a2"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "37d4cb10a5084816a93a23ffec5282d17d52979eda26540c8bb59d53847bb6cc"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "def38f2c76f9a546c3af1829d41286edc6149506780449e65672bc0ad8653cbd"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "1ac4424366d1f7a717ec7d5f9fc6640a4844d8c2fe16d2601048469d4a9e3ebd"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "189f7eda2306c28e6182e3f906e3103d113fee759e81877dbe58f53f48cda8eb"),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, hook: str) -> Path:
    return root / f"oracle_{hook}-2_pos0.bin"


def checked_repeated(hook: str, first: Path, second: Path) -> bytes:
    filename, _shape, expected_bytes, expected_sha = CAPTURE_NAMES[hook]
    first_payload = capture_path(first, hook).read_bytes()
    second_payload = capture_path(second, hook).read_bytes()
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"layer-2 {hook} capture has the wrong size for {filename}")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process layer-2 {hook} captures differ")
    if sha256(first_payload) != expected_sha:
        raise SystemExit(f"layer-2 {hook} capture identity changed")
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
        (
            args.fixtures_root
            / "prefill-layer2-kv-state-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer2-compressors-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    for hook, payload in captures.items():
        filename = CAPTURE_NAMES[hook][0]
        (output / filename).write_bytes(payload)

    tensors = []
    for hook, payload in captures.items():
        filename, shape, _expected_bytes, _expected_sha = CAPTURE_NAMES[hook]
        score_bits = hook.endswith("state_score")
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer2-compressors-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(CAPTURE_NAMES),
            },
            "device_path": (
                "M1 legacy batched F16 paired attention/indexer projections, "
                "ratio-4 replay pooling, learned norm, compressed RoPE, "
                "E4M3FN/indexer QAT, and recurrent-state refresh"
            ),
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer2-kv-state-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": 2047,
            "captured_position_range": [0, 2047],
        },
        "operations": [
            {"name": "attention-compressor-paired-batch-projections", "kernel": "kernel_mul_mm_f16_f32"},
            {"name": "attention-compressor-ratio4-replay-pool", "kernel": "legacy concat+softmax+mul+sum"},
            {"name": "attention-compressor-finalize", "kernel": "weighted RMSNorm+RoPE+E4M3FN"},
            {"name": "attention-compressor-state-refresh", "kernel": "ratio4 tail projection/state store"},
            {"name": "indexer-compressor-paired-batch-projections", "kernel": "kernel_mul_mm_f16_f32"},
            {"name": "indexer-compressor-ratio4-replay-pool", "kernel": "legacy concat+softmax+mul+sum"},
            {"name": "indexer-compressor-finalize", "kernel": "weighted RMSNorm+RoPE+indexer QAT"},
            {"name": "indexer-compressor-state-refresh", "kernel": "ratio4 tail projection/state store"},
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
