#!/usr/bin/env python3
"""Import two repeated 8K prefill layer-0 transition captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_ROWS = 4096
ROW = 3
TENSORS = {
    "attn_norm": ("attn-norm.f32le.bin", 4096),
    "q_lora_norm": ("q-lora-norm.f32le.bin", 1024),
    "Qcur": ("q-current.f32le.bin", 32768),
    "kqv_back": ("kqv-back.f32le.bin", 32768),
    "hc_attn_post": ("after-attention-hc.f32le.bin", 16384),
    "ffn_norm": ("ffn-norm.f32le.bin", 4096),
    "hc_ffn_post": ("after-ffn-hc.f32le.bin", 16384),
}
FULL_SHA256 = {
    "attn_norm": "37481491ee4dc9c41ff4e65e8a9070268a5623c18762595e57f15d5737f0bf2e",
    "q_lora_norm": "e62b3e409981d69e28909544fd124b3ae5c98e2adefba0a6563228750e03c8cb",
    "Qcur": "dbd8eb69586dd4d4ea5dcd86e49f9bae19e94ae6cda149ec64d818adb3c333bc",
    "kqv_back": "20a2b96fe8d7d1eb99e613f39f92b99180229d9440df22fe22ffcc74ff201af8",
    "hc_attn_post": "df4879602d189994fd8341f5caf0d68d792fd5116805594e8031ec9b623234f6",
    "ffn_norm": "2bd5d04d44869b404bc4f078c66bbc62aea55ae91649a05de11abc4012bfdc43",
    "hc_ffn_post": "c05cfe7063dce4a0afd05b8542dcb061611dc965cb71bf27119f25cb0bf85df1",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    output = args.fixtures_root / "prefill-layer0-transition-pos4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    tensors = []
    full_hashes = {}
    payloads = {}
    for name, (filename, width) in TENSORS.items():
        source = f"transition_{name}-0_pos4096.bin"
        first = (args.first_capture / source).read_bytes()
        second = (args.second_capture / source).read_bytes()
        if first != second:
            raise SystemExit(f"fresh-process captures differ: {name}")
        full_hashes[name] = digest(first)
        if full_hashes[name] != FULL_SHA256[name]:
            raise SystemExit(f"unexpected full-capture SHA-256 for {name}")
        expected_bytes = CHUNK_ROWS * width * 4
        if len(first) != expected_bytes:
            raise SystemExit(f"unexpected full-capture size for {name}")
        row_bytes = width * 4
        payload = first[ROW * row_bytes : (ROW + 1) * row_bytes]
        payloads[filename] = payload
        tensors.append(
            {
                "name": name,
                "role": "intermediate",
                "dtype": "f32",
                "shape": [width],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": digest(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer0-pos4099-transition",
        "captured_at_utc": "2026-08-30T16:35:00Z",
        "oracle": {
            "id": "oracle-v3",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "d35fb12d01d500b9cefcef24092c295687ceaf7e",
            "tree": "617415ee9f8ea7dc176d63dada1d5a7582063824",
            "capture_executable_sha256": "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3",
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
            "prefill_tokens": 8192,
            "prefill_chunk": 4096,
            "chunk_start": 4096,
            "captured_position": 4099,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "csv_sha256": [
                "76841a1ab8797a7ae1cf83d88a91b70b9a7222c128351ee0db4ff3ef41bde425",
                "edd82151ec40b947c37869c2fa680dab238cdd8e914c6bbbb4e51efeb39e89ec",
            ],
            "full_batch_sha256": full_hashes,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_norm,q_lora_norm,Qcur,kqv_back,hc_attn_post,ffn_norm,hc_ffn_post",
            },
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 0,
            "position": 4099,
            "chunk_row": ROW,
        },
        "claims": {
            "complete_layer_boundary": True,
            "complete_transformer": False,
            "output_logits": False,
            "throughput": False,
        },
        "tensors": tensors,
    }
    output.mkdir()
    for filename, payload in payloads.items():
        (output / filename).write_bytes(payload)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
