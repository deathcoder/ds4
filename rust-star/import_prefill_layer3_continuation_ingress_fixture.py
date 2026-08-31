#!/usr/bin/env python3
"""Import repeated layer-3 ingress/QKV captures for the second 8K chunk."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_START = 4_096
CHUNK_ROWS = 4_096
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-31T13:02:10Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3"
)
TENSORS = {
    "hc_attn_pre": ("hc-attn-pre-first-tile.f32le.bin", 4_096),
    "attn_norm": ("attn-norm-first-tile.f32le.bin", 4_096),
    "q_lora": ("q-lora-first-tile.f32le.bin", 1_024),
    "q_lora_norm": ("q-lora-norm-first-tile.f32le.bin", 1_024),
    "KVraw": ("kv-raw-first-tile.f32le.bin", 512),
    "KVnorm": ("kv-norm-first-tile.f32le.bin", 512),
    "Qcur": ("q-current-first-tile.f32le.bin", 32_768),
    "KVrope": ("kv-rope-first-tile.f32le.bin", 512),
    "KVcur": ("kv-current-first-tile.f32le.bin", 512),
}
FULL_CAPTURE_SHA256 = {
    "hc_attn_pre": "b84a97eb150f64251f7d8896101afcfd2c717e87d258904de49acafae2aeacb3",
    "attn_norm": "ee201298cbf2acbfc72d56865905bfa94d8a4e4c185b9f76514ddd3e61f21af0",
    "q_lora": "d5034d3a536b4df6a5aa464ed810fb2dd656fa21ef853e5b9e6cb3ff33fbc99d",
    "q_lora_norm": "54a2aac900690016eb24e3ca3c64015f6533aa2cd305c0bc17ad950dddd88607",
    "KVraw": "ba8e31b3c68a68b59ff97407b82d21f2ad57897dfcb45388050772a0bc98f854",
    "KVnorm": "00c3c5ac47cacd7a7284bdb71c2a5c92019efe2946c1c2f73c84ef5a9f1e837e",
    "Qcur": "9eb09a970bf97d8ea4d078d3b4e2c1db6ea5cf40888fc1888334161e632c9f91",
    "KVrope": "e447bbace00ad4b70ecf24c3c91dbbe38be09ea4e21c64711d030cca861f565e",
    "KVcur": "edaba3fe7eeb97e4e8903a425d7e31c233fa8c8d4355eea82dd100d1a162abd3",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, hook: str) -> Path:
    return root / f"oracle_{hook}-3_pos4096.bin"


def checked_repeated(hook: str, first: Path, second: Path, width: int) -> bytes:
    first_payload = capture_path(first, hook).read_bytes()
    second_payload = capture_path(second, hook).read_bytes()
    expected_bytes = CHUNK_ROWS * width * 4
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"{hook} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {hook} captures differ")
    if sha256(first_payload) != FULL_CAPTURE_SHA256[hook]:
        raise SystemExit(f"{hook} capture identity changed")
    return first_payload[: TILE_ROWS * width * 4]


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
    output = args.fixtures_root / "prefill-layer3-continuation-ingress-4096-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for hook, (filename, width) in TENSORS.items():
        payload = checked_repeated(hook, args.first, args.second, width)
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": hook.lower(),
                "hook": hook,
                "role": "output" if hook == "KVcur" else "intermediate",
                "dtype": "f32",
                "shape": [TILE_ROWS, width],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer3-continuation-ingress-4096",
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
                "1febcd9915cf42c507ed9740fc8b3315ce588dd3fef57fdd1fd39f358cf22d3b",
                "ec68e5eab583b57f44a0d1f37af244dcc916ba1cac77bc40a9ed9f082d98f291",
            ],
            "full_batch_sha256": FULL_CAPTURE_SHA256,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "3",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": list(TENSORS),
            },
            "input_fixture": "dwarfstar-oracle-v3-prefill-layer2-continuation-tail-4096",
            "storage_note": (
                "Only the exact first 32-row continuation tile is retained; "
                "full 4096-row identities remain pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 3,
            "position": CHUNK_START + TILE_ROWS - 1,
            "captured_position_range": [CHUNK_START, CHUNK_START + TILE_ROWS - 1],
            "tile_rows": TILE_ROWS,
        },
        "operations": [
            {"name": "attention-hc-ingress", "kernel": "RMSNorm + aligned F16 HC projection + HC ingress"},
            {"name": "q-a-and-kv-projections", "kernel": "aligned Q8_0 prefill matmul"},
            {"name": "q-kv-learned-norm", "kernel": "kernel_dsv4_qkv_rms_norm_f32"},
            {"name": "q-b-projection", "kernel": "aligned Q8_0 prefill matmul"},
            {"name": "q-and-kv-rope", "kernel": "head norm/RoPE plus KV RoPE"},
            {"name": "kv-finalization", "kernel": "kernel_dsv4_compressor_fp8_f32"},
        ],
        "claims": {
            "live_layer2_hc_input": True,
            "native_batch_schedule": True,
            "complete_layer3_ingress_qkv_tile": True,
            "complete_layer3_tile": False,
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
