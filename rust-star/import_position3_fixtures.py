#!/usr/bin/env python3
"""Import independently repeated position-3 compressor-transition fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


POSITION = 3
TOKEN = 1915
CAPTURED_AT_UTC = "2026-08-15T10:46:35Z"


def captured_path(run: Path, layer: int, hook: str, position: int = POSITION) -> Path:
    suffix = ".i32" if hook == "ffn_moe_topk" else ".bin"
    return run / f"oracle_{hook}-{layer}_pos{position}{suffix}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def operations_for_layer(template: dict, layer: int) -> list[dict]:
    operations = copy.deepcopy(template["operations"])
    if layer == 0:
        operations = [
            {
                "name": "token-embedding",
                "kernel": "kernel_get_rows_f16",
                "weights": ["token_embd.weight"],
            },
            {
                "name": "embedding-hc-repeat",
                "kernel": "kernel_repeat_f32",
                "weights": [],
            },
            *operations,
        ]
    for operation in operations:
        operation["weights"] = [
            weight.replace("blk.1.", f"blk.{layer}.")
            for weight in operation["weights"]
        ]
    if layer >= 2:
        operations[10:10] = [
            {
                "name": "attention-compressor-paired-projection",
                "kernel": "kernel_mul_mv_f16_f32_pair_4",
                "weights": [
                    f"blk.{layer}.attn_compressor_kv.weight",
                    f"blk.{layer}.attn_compressor_gate.weight",
                ],
            },
            {
                "name": "attention-compressor-state-store",
                "kernel": "kernel_dsv4_compressor_store_one",
                "weights": [f"blk.{layer}.attn_compressor_ape.weight"],
            },
        ]
    if layer == 2:
        operations[12:12] = [
            {
                "name": "attention-compressor-pool",
                "kernel": "kernel_dsv4_softmax_pool",
                "weights": [f"blk.{layer}.attn_compressor_norm.weight"],
            },
            {
                "name": "attention-compressor-rope-shift-fp8",
                "kernel": "kernel_dsv4_rope_tail_f32+kernel_dsv4_ratio4_shift+kernel_dsv4_fp8_kv_quantize",
                "weights": [],
            },
            {
                "name": "indexer-compressor-paired-projection",
                "kernel": "kernel_mul_mv_f16_f32_pair_4",
                "weights": [
                    f"blk.{layer}.indexer_compressor_kv.weight",
                    f"blk.{layer}.indexer_compressor_gate.weight",
                ],
            },
            {
                "name": "indexer-compressor-update-and-qat",
                "kernel": "kernel_dsv4_compressor_store_one+kernel_dsv4_softmax_pool+kernel_dsv4_indexer_qat",
                "weights": [
                    f"blk.{layer}.indexer_compressor_ape.weight",
                    f"blk.{layer}.indexer_compressor_norm.weight",
                ],
            },
        ]
    return operations


def import_layer(capture_root: Path, fixtures_root: Path, layer: int) -> Path:
    template_layer = 1 if layer == 0 else layer
    template_path = fixtures_root / f"layer{template_layer}-complete-v1" / "manifest.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output = fixtures_root / f"layer{layer}-pos3-complete-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for tensor in template["tensors"]:
        if tensor["name"] == "cache_row0":
            continue
        hook = tensor["hook"]
        first = captured_path(capture_root / "a" / f"layer{layer}", layer, hook)
        second = captured_path(capture_root / "b" / f"layer{layer}", layer, hook)
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
        if first_bytes != second_bytes:
            raise SystemExit(f"independent captures differ for layer {layer} hook {hook}")
        expected_bytes = 4
        for dimension in tensor["shape"]:
            expected_bytes *= dimension
        if len(first_bytes) != expected_bytes:
            raise SystemExit(
                f"layer {layer} hook {hook} has {len(first_bytes)} bytes, expected {expected_bytes}"
            )
        destination = output / tensor["path"]
        shutil.copyfile(first, destination)
        imported = copy.deepcopy(tensor)
        imported["bytes"] = len(first_bytes)
        imported["sha256"] = sha256(first_bytes)
        tensors.append(imported)

    if layer == 2:
        first = captured_path(capture_root / "a" / "layer2", layer, "KVcompress")
        second = captured_path(capture_root / "b" / "layer2", layer, "KVcompress")
        payload = first.read_bytes()
        if payload != second.read_bytes():
            raise SystemExit("independent layer-2 KVcompress captures differ")
        destination = output / "compressed-kv-row0.f32le.bin"
        shutil.copyfile(first, destination)
        tensors.append(
            {
                "name": "compressed_kv_row0",
                "hook": "KVcompress",
                "role": "output",
                "dtype": "f32",
                "shape": [512],
                "encoding": "little-endian-ieee754-binary32",
                "path": destination.name,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer{layer}-pos3-complete",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": template["oracle"],
        "model": template["model"],
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 1,
            "decode_step": POSITION,
            "token_id": TOKEN,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": str(layer),
                "DS4_METAL_GRAPH_DUMP_POS": str(POSITION),
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(
                    tensor["hook"] for tensor in tensors
                ),
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "layer": layer,
            "position": POSITION,
        },
        "operations": operations_for_layer(template, layer),
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def import_prime(capture_root: Path, fixtures_root: Path, layer: int) -> Path:
    template = json.loads(
        (fixtures_root / f"layer{layer}-complete-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = fixtures_root / f"layer{layer}-pos0-compressor-prime-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    first = captured_path(capture_root / "pos0" / "a" / f"layer{layer}", layer, "attn_norm", 0)
    second = captured_path(capture_root / "pos0" / "b" / f"layer{layer}", layer, "attn_norm", 0)
    payload = first.read_bytes()
    if payload != second.read_bytes():
        raise SystemExit(f"independent layer-{layer} position-0 priming captures differ")
    destination = output / "attn-norm.f32le.bin"
    shutil.copyfile(first, destination)
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer{layer}-pos0-compressor-prime",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": template["oracle"],
        "model": template["model"],
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 1,
            "decode_step": 0,
            "token_id": None,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": str(layer),
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_norm",
            },
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": layer,
            "position": 0,
        },
        "operations": [
            {
                "name": "capture-compressor-prime-input",
                "kernel": "kernel_dsv4_hc_split_weighted_sum_norm4",
                "weights": [f"blk.{layer}.attn_norm.weight"],
            }
        ],
        "tensors": [
            {
                "name": "attn_norm",
                "hook": "attn_norm",
                "role": "input",
                "dtype": "f32",
                "shape": [4096],
                "encoding": "little-endian-ieee754-binary32",
                "path": destination.name,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
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
    for layer in range(4):
        print(import_layer(args.capture_root, args.fixtures_root, layer))
    for layer in (2, 3):
        print(import_prime(args.capture_root, args.fixtures_root, layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
