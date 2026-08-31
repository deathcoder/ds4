#!/usr/bin/env python3
"""Import repeated dump-only oracle captures of the layer-0 FFN HC split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROWS = 4096
WIDTH = 24
ROW = 3
FULL_SHA256 = "11b02d4b513e512bf25ea1279c03191a92fcdfafb7d2a6d42c452cb794d1dbbc"


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
    output = args.fixtures_root / "prefill-layer0-hc-split-transition-pos4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    source = "transition_hc_ffn_split-0_pos4096.bin"
    first = (args.first_capture / source).read_bytes()
    second = (args.second_capture / source).read_bytes()
    if first != second or digest(first) != FULL_SHA256:
        raise SystemExit("fresh-process HC split captures are not the accepted pair")
    if len(first) != ROWS * WIDTH * 4:
        raise SystemExit("unexpected full-capture size")
    row_bytes = WIDTH * 4
    payload = first[ROW * row_bytes : (ROW + 1) * row_bytes]
    filename = "ffn-hc-split.f32le.bin"
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer0-pos4099-ffn-hc-split",
        "captured_at_utc": "2026-08-31T00:15:00Z",
        "oracle": {
            "id": "oracle-v3-dump-only-patch",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "d35fb12d01d500b9cefcef24092c295687ceaf7e",
            "tree": "617415ee9f8ea7dc176d63dada1d5a7582063824",
            "base_capture_executable_sha256": "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3",
            "dump_patch_executable_sha256": "e85a366af188a4a61d1c0b6d6f12447371ff3a3b9f8e6327e75d8aeb2dc17866",
            "dump_patch_scope": "Adds one debug-dump call after the existing FFN HC split; arithmetic is unchanged.",
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
            "full_batch_sha256": FULL_SHA256,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": "hc_ffn_split",
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
        "tensors": [
            {
                "name": "hc_ffn_split",
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
    output.mkdir()
    (output / filename).write_bytes(payload)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
