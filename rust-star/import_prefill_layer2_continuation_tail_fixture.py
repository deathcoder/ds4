#!/usr/bin/env python3
"""Import a repeated 32-row layer-2 tail from the second 8K prefill chunk."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_START = 4_096
CHUNK_ROWS = 4_096
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-31T08:22:14Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3"
)
TENSORS = {
    "kqv_back": ("kqv-back-first-tile.f32le.bin", 32_768, "input", "f32"),
    "attn_low": ("attention-low-first-tile.f32le.bin", 8_192, "intermediate", "f32"),
    "attn_out": ("attention-output-first-tile.f32le.bin", 4_096, "intermediate", "f32"),
    "hc_attn_post": ("attention-hc-post-first-tile.f32le.bin", 16_384, "intermediate", "f32"),
    "hc_ffn_pre": ("ffn-current-first-tile.f32le.bin", 4_096, "intermediate", "f32"),
    "ffn_norm": ("ffn-norm-first-tile.f32le.bin", 4_096, "intermediate", "f32"),
    "ffn_moe_logits": ("router-logits-first-tile.f32le.bin", 256, "intermediate", "f32"),
    "ffn_moe_probs": ("router-probs-first-tile.f32le.bin", 256, "intermediate", "f32"),
    "ffn_moe_topk": ("router-selected-first-tile.i32le.bin", 6, "intermediate", "i32"),
    "ffn_moe_weights_scaled": ("router-weights-first-tile.f32le.bin", 6, "intermediate", "f32"),
    "ffn_moe_weighted_swiglu": ("routed-mid-first-tile.f32le.bin", 12_288, "intermediate", "f32"),
    "ffn_moe_out": ("routed-output-first-tile.f32le.bin", 4_096, "intermediate", "f32"),
    "ffn_shexp": ("shared-output-first-tile.f32le.bin", 4_096, "intermediate", "f32"),
    "hc_ffn_post": ("ffn-hc-post-first-tile.f32le.bin", 16_384, "output", "f32"),
}
FULL_CAPTURE_SHA256 = {
    "kqv_back": "372140699ec97a8734cdf14572d88caf62063a3098ba1da43875190365816eba",
    "attn_low": "1c25067bc70440c748b16bb88d8ab194e68a2c39568c3899a9f2048c9a363035",
    "attn_out": "bba8a9e4d5d83f89896b4c4686c4a7d08f6c228a12b88e2f1160cc41fea2a327",
    "hc_attn_post": "5afba314c39d8ab0cc1ac8a50f49b519c96711f0ec7558db0b958839238b65ef",
    "hc_ffn_pre": "a506d06e253ee2f321c86a23bcca4d2f8818dca2aafbcfac1625a99e6ebb0a6a",
    "ffn_norm": "3a1cedca7e24ba6d12c6c5430fb14b90bf2f9a48546686410a72c6940f61881b",
    "ffn_moe_logits": "a98ee193bfa0c23447b2cef80674d5eccc4f8146433d74180e1ee324835ede44",
    "ffn_moe_probs": "230b22291296e20864171313e02ab2d937d44224e665289d67a96f5675b6b3a6",
    "ffn_moe_topk": "e701f078ceafb71ed4eb381bce1cec5ad4f4688951e22f5e945c0326f390a732",
    "ffn_moe_weights_scaled": "ca169a1e63d760999749a387f47f10d6b263cf0e9be655edcdbe2c6642d8e784",
    "ffn_moe_weighted_swiglu": "b839b0e1bb2caccd13014ec2ca956840d8a3e658f97cabe593aa68e80cc5bb38",
    "ffn_moe_out": "a473ff5f7810bca629064de4ecc113844f0f88f52b9caeee434791abf1f51bb3",
    "ffn_shexp": "eb0bd068f09ef28fdbc7ec7029624f463abb1d89bba183653b080a290f7a894f",
    "hc_ffn_post": "bea6c65976d9f54a425ad36f4ed166571ddb2425d505688d4e2a40ec592e7567",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, name: str) -> Path:
    if name == "kqv_back":
        return root / "transition_kqv_back-2_pos4096.bin"
    return root / f"oracle_{name}-2_pos4096.{'i32' if name == 'ffn_moe_topk' else 'bin'}"


def checked_repeated(name: str, first: Path, second: Path, width: int) -> bytes:
    first_payload = capture_path(first, name).read_bytes()
    second_payload = capture_path(second, name).read_bytes()
    expected_bytes = CHUNK_ROWS * width * 4
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != FULL_CAPTURE_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first_payload[: TILE_ROWS * width * 4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-first", type=Path, required=True)
    parser.add_argument("--transition-second", type=Path, required=True)
    parser.add_argument("--low-first", type=Path, required=True)
    parser.add_argument("--low-second", type=Path, required=True)
    parser.add_argument("--tail-first", type=Path, required=True)
    parser.add_argument("--tail-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    output = args.fixtures_root / "prefill-layer2-continuation-tail-4096-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    roots = {
        "kqv_back": (args.transition_first, args.transition_second),
        "attn_low": (args.low_first, args.low_second),
    }
    output.mkdir()
    tensors = []
    for name, (filename, width, role, dtype) in TENSORS.items():
        first, second = roots.get(name, (args.tail_first, args.tail_second))
        payload = checked_repeated(name, first, second, width)
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": name,
                "hook": name,
                "role": role,
                "dtype": dtype,
                "shape": [TILE_ROWS, width],
                "encoding": (
                    "little-endian-signed-integer32"
                    if dtype == "i32"
                    else "little-endian-ieee754-binary32"
                ),
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer2-continuation-tail-4096",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": {
            "id": "oracle-v3",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "d35fb12d01d500b9cefcef24092c295687ceaf7e",
            "tree": "617415ee9f8ea7dc176d63dada1d5a7582063824",
            "capture_executable_sha256": CAPTURE_EXECUTABLE_SHA256,
        },
        "model": {
            "family": "DeepSeek-V4-Flash-0731",
            "sha256": "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
        },
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefill_tokens": 8_192,
            "prefill_chunk": CHUNK_ROWS,
            "chunk_start": CHUNK_START,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "csv_sha256": [
                "045906a5a6c7c16c1921fd3d577382bb51c325898863abe5673264c09dd08ad1",
                "c067575212cafc8a588a81632ced6c23912cd20513b5426274ff6b9a4f557686",
                "a57596781d68386f6e2a1b48d3d75995ed8b2686470c345e7458566cb87a7b0c",
                "8ef3853bbd1d6237663f1b91f438b2723a120ed5e57b1b143361ef83c709a3ea",
            ],
            "full_batch_sha256": FULL_CAPTURE_SHA256,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": list(TENSORS),
            },
            "storage_note": (
                "Only the exact first 32-row continuation tile is retained; "
                "full 4096-row identities remain pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": CHUNK_START + TILE_ROWS - 1,
            "captured_position_range": [CHUNK_START, CHUNK_START + TILE_ROWS - 1],
            "tile_rows": TILE_ROWS,
        },
        "operations": [
            {
                "name": "oracle-kqv-back-input",
                "kernel": "captured boundary after indexed mixed attention plus inverse RoPE",
            },
            {"name": "attention-output-projection", "kernel": "grouped Q8_0 low plus Q8_0 output"},
            {"name": "attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
            {"name": "token-hash-router", "kernel": "batch softplus/sqrt/hash/gather/normalize"},
            {"name": "routed-experts", "kernel": "IQ2_XXS pair SwiGLU plus Q2_K down"},
            {"name": "shared-expert", "kernel": "Q8_0 gate/up/down plus flat SwiGLU"},
            {"name": "ffn-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
        ],
        "claims": {
            "native_batch_schedule": True,
            "oracle_seeded_kqv_back_input": True,
            "complete_downstream_tail_tile": True,
            "complete_layer_tile": False,
            "complete_layer": False,
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
