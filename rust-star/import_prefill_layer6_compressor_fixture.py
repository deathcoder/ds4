#!/usr/bin/env python3
"""Import repeated full-2K layer-6 paired-compressor captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-22T17:34:06Z"
CAPTURE_NAMES = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "a5768768273f01fc65c4bff2bcc493d7964d9763feb0edec76eb7fb8de0f3a5e"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "cab084f5c0810062d05e4a64cb4264ed6126eefd90e3a890fc02921975505228"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "35b80554f5b7ce86b97aa8d6b075636edba5665b058c9682b86a63bf506686e7"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "8a325c78472cd0d68b181f7d490a439329c341d5f3d9157afd61930dc6e9f1e9"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "9f54a4bc8d263cd62ab16360a471f9e96a7dd6f13a6100bdfd1acaf8651daca0"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "c473edfcd8b0145c3df05c6866f8d83b4c35a32d7035c6606c9248abc33dd9cc"),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, hook: str) -> Path:
    return root / f"{root.name}_{hook}-6_pos0.bin"


def checked_repeated(hook: str, first: Path, second: Path) -> bytes:
    filename, _shape, expected_bytes, expected_sha = CAPTURE_NAMES[hook]
    first_payload = capture_path(first, hook).read_bytes()
    second_payload = capture_path(second, hook).read_bytes()
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"layer-6 {hook} capture has the wrong size for {filename}")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process layer-6 {hook} captures differ")
    if sha256(first_payload) != expected_sha:
        raise SystemExit(f"layer-6 {hook} capture identity changed")
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
        (args.fixtures_root / "prefill-layer6-qkv-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer6-compressors-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    for hook, payload in captures.items():
        (output / CAPTURE_NAMES[hook][0]).write_bytes(payload)

    tensors = []
    for hook, payload in captures.items():
        filename, shape, _expected_bytes, _expected_sha = CAPTURE_NAMES[hook]
        score_bits = hook.endswith("state_score")
        tensors.append({
            "name": hook.lower() + ("_bits" if score_bits else ""),
            "hook": hook,
            "role": "output",
            "dtype": "i32" if score_bits else "f32",
            "shape": shape,
            "encoding": ("little-endian-signed-integer32" if score_bits
                         else "little-endian-ieee754-binary32"),
            "path": filename,
            "bytes": len(payload),
            "sha256": sha256(payload),
        })

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer6-compressors-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "6",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(CAPTURE_NAMES),
            },
            "device_path": (
                "M1 legacy batched F16 paired attention/indexer projections, "
                "ratio-4 replay pooling, learned norm, compressed RoPE, "
                "E4M3FN/indexer QAT, and recurrent-state refresh"
            ),
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer6-qkv-2048",
        },
        "scope": {
            "kind": "layer-segment", "phase": "prefill", "layer": 6,
            "position": 2047, "captured_position_range": [0, 2047],
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
