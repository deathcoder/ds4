#!/usr/bin/env python3
"""Import repeated layer-3/5 ratio-128 compressor replay fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T09:44:11Z"
LAYERS = (3, 5)
POSITIONS = 128
ACTIVATION_ELEMENTS = 4096
OUTPUT_ELEMENTS = 512

ORACLE = {
    "id": "oracle-v1",
    "repository": "https://github.com/antirez/ds4.git",
    "commit": "b0309611041655f4e45671cfd9c9886aff161406",
    "tree": "20c11af22f90a0bdf25da860da5ef06de4064060",
    "capture_executable_sha256": (
        "b0b7c31d9832cc3f5cccf810cd2af6f4bc4f6e71de945493674315314f666fc8"
    ),
}
MODEL = {
    "family": "DeepSeek-V4-Flash-0731",
    "sha256": "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated_payload(capture_root: Path, layer: int, name: str) -> bytes:
    first = capture_root / "a" / f"layer{layer}" / name
    second = capture_root / "b" / f"layer{layer}" / name
    payload = first.read_bytes()
    if payload != second.read_bytes():
        raise SystemExit(f"independent captures differ for layer {layer}: {name}")
    return payload


def import_layer(capture_root: Path, fixtures_root: Path, layer: int) -> Path:
    output = fixtures_root / f"layer{layer}-pos127-compressor-replay-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    activations = bytearray()
    for position in range(POSITIONS):
        payload = repeated_payload(
            capture_root,
            layer,
            f"oracle_attn_norm-{layer}_pos{position}.bin",
        )
        expected_bytes = ACTIVATION_ELEMENTS * 4
        if len(payload) != expected_bytes:
            raise SystemExit(
                f"layer {layer} position {position} attn_norm has "
                f"{len(payload)} bytes, expected {expected_bytes}"
            )
        activations.extend(payload)

    compressed = repeated_payload(
        capture_root,
        layer,
        f"oracle_KVcompress-{layer}_pos127.bin",
    )
    if len(compressed) != OUTPUT_ELEMENTS * 4:
        raise SystemExit(
            f"layer {layer} compressed row has {len(compressed)} bytes, "
            f"expected {OUTPUT_ELEMENTS * 4}"
        )

    activation_path = output / "attn-norm-sequence.f32le.bin"
    compressed_path = output / "compressed-kv-row0.f32le.bin"
    activation_path.write_bytes(activations)
    compressed_path.write_bytes(compressed)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer{layer}-pos127-compressor-replay",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": ORACLE,
        "model": MODEL,
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": (
                "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f"
            ),
            "prefill_tokens": 1,
            "decode_steps": 127,
            "final_decode_position": 127,
            "intermediate_tokens": "oracle-generated-not-retained",
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": str(layer),
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_norm,KVcompress",
                "DS4_METAL_GRAPH_DUMP_POS": "unset-all-positions",
            },
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "decode",
            "layer": layer,
            "position": 127,
        },
        "operations": [
            {
                "name": "attention-compressor-paired-projection-replay",
                "kernel": "kernel_mul_mv_f16_f32_pair_4",
                "weights": [
                    f"blk.{layer}.attn_compressor_kv.weight",
                    f"blk.{layer}.attn_compressor_gate.weight",
                ],
            },
            {
                "name": "attention-compressor-recurrent-state-store",
                "kernel": "kernel_dsv4_compressor_store_one",
                "weights": [f"blk.{layer}.attn_compressor_ape.weight"],
            },
            {
                "name": "attention-compressor-ratio128-pool",
                "kernel": "kernel_dsv4_compressor_pack_legacy+kernel_soft_max_4+kernel_mul_mm_f32_4+kernel_sum_rows_f32_4",
                "weights": [],
            },
            {
                "name": "attention-compressor-finalize",
                "kernel": "kernel_rms_norm_f32_4+kernel_dsv4_rope_tail_f32+kernel_dsv4_fp8_kv_quantize",
                "weights": [f"blk.{layer}.attn_compressor_norm.weight"],
            },
        ],
        "tensors": [
            {
                "name": "attn_norm_sequence",
                "hook": "attn_norm",
                "role": "input",
                "dtype": "f32",
                "shape": [POSITIONS, ACTIVATION_ELEMENTS],
                "encoding": "little-endian-ieee754-binary32",
                "path": activation_path.name,
                "bytes": len(activations),
                "sha256": sha256(activations),
            },
            {
                "name": "compressed_kv_row0",
                "hook": "KVcompress",
                "role": "output",
                "dtype": "f32",
                "shape": [OUTPUT_ELEMENTS],
                "encoding": "little-endian-ieee754-binary32",
                "path": compressed_path.name,
                "bytes": len(compressed),
                "sha256": sha256(compressed),
            },
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    for layer in LAYERS:
        print(import_layer(args.capture_root, args.fixtures_root, layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
