#!/usr/bin/env python3
"""Import repeated 8K prefill layer-0 FFN transition captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_ROWS = 4096
ROW = 3
TENSORS = {
    "ffn_moe_logits": ("router-logits.f32le.bin", 256, "f32", "bin"),
    "ffn_moe_probs": ("router-probs.f32le.bin", 256, "f32", "bin"),
    "ffn_moe_topk": ("router-selected.i32le.bin", 6, "i32", "i32"),
    "ffn_moe_weights_scaled": ("router-weights.f32le.bin", 6, "f32", "bin"),
    "ffn_moe_out": ("routed-out.f32le.bin", 4096, "f32", "bin"),
    "ffn_shexp": ("shared-out.f32le.bin", 4096, "f32", "bin"),
    "ffn_out": ("ffn-out.f32le.bin", 4096, "f32", "bin"),
}
FULL_SHA256 = {
    "ffn_moe_logits": "665e90acb13b07b5e2428a051c2ed75a8f5900a54a8dddddac241b73dc47eb8f",
    "ffn_moe_probs": "bc9cf632b6df978bc1ad64ff2434d1c68a8d36a4d113a6672e78a751f7599832",
    "ffn_moe_topk": "0d77fee929f2c10200c90e081cd3a52b97a8548ac53e8dc7b040bd140e988be3",
    "ffn_moe_weights_scaled": "d5e314a28abac9efc42cb7beadce159455fa7489f1592a7e69f0ce7a9acd7608",
    "ffn_moe_out": "48862a6c5d0eb4a4699fa6014c19ccf39674e3a66f3c0b104b671d104bbf3f13",
    "ffn_shexp": "84a51f501b7b67972d5c0ad229097427a465e3ed743da277cc3917a0e1e753e3",
    "ffn_out": "5291918151e17348a4414c4698e239edc259194aad775f646ed95d3672dd6a96",
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
    output = args.fixtures_root / "prefill-layer0-ffn-transition-pos4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    tensors = []
    full_hashes = {}
    payloads = {}
    for name, (filename, width, dtype, extension) in TENSORS.items():
        source = f"transition_{name}-0_pos4096.{extension}"
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
                "dtype": dtype,
                "shape": [width],
                "encoding": (
                    "little-endian-ieee754-binary32"
                    if dtype == "f32"
                    else "little-endian-signed-integer32"
                ),
                "path": filename,
                "bytes": len(payload),
                "sha256": digest(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer0-pos4099-ffn-transition",
        "captured_at_utc": "2026-08-30T17:25:00Z",
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
                "bd4d972baf245f5d84e7bb67d8f28286c528f59c9a7399f1b1e22be5424d910a",
                "3224650b9ca53cb72a2a70826c1afab15abb05bf6dfe4e74f0a543b1616145d4",
            ],
            "full_batch_sha256": full_hashes,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": "ffn_moe_logits,ffn_moe_probs,ffn_moe_topk,ffn_moe_weights_scaled,ffn_moe_out,ffn_shexp,ffn_out",
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
