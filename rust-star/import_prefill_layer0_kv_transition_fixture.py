#!/usr/bin/env python3
"""Import repeated 8K prefill layer-0 KV transition captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_ROWS = 4096
ROW = 3
TENSORS = {
    "KVraw": ("kv-raw.f32le.bin", "5f13a3432316675e9a4c6bee3703aad14fecfe1ba26dbc7f9397f63f47d4a740"),
    "KVnorm": ("kv-norm.f32le.bin", "03d0be8db661a7962d5cc47044036ce0bd371619fcd7f2a0fcdc28f351fd40ba"),
    "KVrope": ("kv-rope.f32le.bin", "492bb8d2daaeb75a880f3c24bbb1a9c1bed1f9c7dd6939f29f674b5c74710e81"),
    "KVcur": ("kv-current.f32le.bin", "8fa045c915d533f675299d104072c8400cbe45480878c0359b28648e6701c948"),
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
    output = args.fixtures_root / "prefill-layer0-kv-transition-pos4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    tensors = []
    payloads = {}
    full_hashes = {}
    for name, (filename, expected_hash) in TENSORS.items():
        source = f"transition_{name}-0_pos4096.bin"
        first = (args.first_capture / source).read_bytes()
        second = (args.second_capture / source).read_bytes()
        if first != second:
            raise SystemExit(f"fresh-process captures differ: {name}")
        full_hashes[name] = digest(first)
        if full_hashes[name] != expected_hash:
            raise SystemExit(f"unexpected full-capture SHA-256 for {name}")
        expected_bytes = CHUNK_ROWS * 512 * 4
        if len(first) != expected_bytes:
            raise SystemExit(f"unexpected full-capture size for {name}")
        row_bytes = 512 * 4
        payload = first[ROW * row_bytes : (ROW + 1) * row_bytes]
        payloads[filename] = payload
        tensors.append(
            {
                "name": name,
                "role": "intermediate",
                "dtype": "f32",
                "shape": [512],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": digest(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer0-pos4099-kv-transition",
        "captured_at_utc": "2026-08-30T18:15:00Z",
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
                "7cea18718f1c167b7d99821323c279aa9dcc7c99ce67d3c7cba704d07c528636",
                "ec4e910d7872c8889d45daa9dfe3a91593bb77dd2966711b584509bae157b4cb",
            ],
            "full_batch_sha256": full_hashes,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": "KVraw,KVnorm,KVrope,KVcur",
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
            "complete_layer_boundary": False,
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
