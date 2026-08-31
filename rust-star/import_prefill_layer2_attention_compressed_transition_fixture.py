#!/usr/bin/env python3
"""Import two repeated oracle-v3 layer-2 continuation compressor captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FULL_SHA256 = "dea1353a700f85039af6fcd761b606bb2a9e4b0015f6f46897d36a5729a913b6"
SOURCE = "transition_KVcompress-2_pos4096.bin"
ROWS = 1024
WIDTH = 512


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
    first = (args.first_capture / SOURCE).read_bytes()
    second = (args.second_capture / SOURCE).read_bytes()
    if len(first) != ROWS * WIDTH * 4:
        raise SystemExit(f"capture has {len(first)} bytes, expected {ROWS * WIDTH * 4}")
    if first != second:
        raise SystemExit("fresh-process captures differ")
    if digest(first) != FULL_SHA256:
        raise SystemExit("unexpected full-capture SHA-256")

    payload = first[: WIDTH * 4]
    output = (
        args.fixtures_root
        / "prefill-layer2-attention-compressed-transition-pos4099-v1"
    )
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    filename = "attention-compressed-row1024.f32le.bin"
    (output / filename).write_bytes(payload)
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer2-attention-compressed-pos4099",
        "captured_at_utc": "2026-08-31T05:50:00Z",
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
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "full_batch_sha256": FULL_SHA256,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": "KVcompress",
            },
        },
        "scope": {
            "kind": "compressor-row",
            "phase": "prefill",
            "layer": 2,
            "compressed_row": 1024,
            "input_positions": [4096, 4099],
        },
        "claims": {
            "attention_compressed_row": True,
            "complete_layer": False,
            "output_logits": False,
            "throughput": False,
        },
        "tensors": [
            {
                "name": "KVcompress",
                "role": "intermediate",
                "dtype": "f32",
                "shape": [WIDTH],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": digest(payload),
            }
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
