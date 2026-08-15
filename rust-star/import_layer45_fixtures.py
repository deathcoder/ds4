#!/usr/bin/env python3
"""Import independently repeated layer-4/5 position-0 through -3 fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T09:21:11Z"
TOKENS = {1: 201, 2: 361, 3: 1915}


def captured_path(run: Path, layer: int, hook: str, position: int) -> Path:
    suffix = ".i32" if hook == "ffn_moe_topk" else ".bin"
    return (
        run
        / f"layer{layer}"
        / f"pos{position}"
        / f"oracle_{hook}-{layer}_pos{position}{suffix}"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repeated_payload(
    capture_root: Path, layer: int, hook: str, position: int
) -> tuple[Path, bytes]:
    first = captured_path(capture_root / "a", layer, hook, position)
    second = captured_path(capture_root / "b", layer, hook, position)
    payload = first.read_bytes()
    if payload != second.read_bytes():
        raise SystemExit(
            f"independent captures differ for layer {layer} position {position} hook {hook}"
        )
    return first, payload


def rewrite_operations(template: dict, layer: int, position: int) -> list[dict]:
    operations = copy.deepcopy(template["operations"])
    for operation in operations:
        operation["weights"] = [
            weight.replace("blk.3.", f"blk.{layer}.")
            for weight in operation["weights"]
        ]
    if not any(
        operation["name"] == "attention-compressor-paired-projection"
        for operation in operations
    ):
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
    if position == 3 and layer % 2 == 0:
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
        ]
    if layer % 2 == 0:
        insertion = 14 if position == 3 else 12
        operations[insertion:insertion] = [
            {
                "name": "indexer-compressor-paired-projection",
                "kernel": "kernel_mul_mv_f16_f32_pair_4",
                "weights": [
                    f"blk.{layer}.indexer_compressor_kv.weight",
                    f"blk.{layer}.indexer_compressor_gate.weight",
                ],
            },
            {
                "name": "indexer-compressor-update",
                "kernel": "kernel_dsv4_compressor_store_one"
                + (
                    "+kernel_dsv4_softmax_pool+kernel_dsv4_indexer_qat"
                    if position == 3
                    else ""
                ),
                "weights": [
                    f"blk.{layer}.indexer_compressor_ape.weight",
                    f"blk.{layer}.indexer_compressor_norm.weight",
                ],
            },
        ]
    return operations


def template_path(fixtures_root: Path, position: int) -> Path:
    suffix = "complete-v1" if position == 1 else f"pos{position}-complete-v1"
    return fixtures_root / f"layer3-{suffix}" / "manifest.json"


def import_complete(
    capture_root: Path, fixtures_root: Path, layer: int, position: int
) -> Path:
    template = json.loads(
        template_path(fixtures_root, position).read_text(encoding="utf-8")
    )
    output = fixtures_root / f"layer{layer}-pos{position}-complete-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for tensor in template["tensors"]:
        hook = tensor["hook"]
        source_position = position
        if tensor["name"] == "cache_row0":
            hook = "KVcur"
            source_position = 0
        source, payload = repeated_payload(
            capture_root, layer, hook, source_position
        )
        expected_bytes = 4
        for dimension in tensor["shape"]:
            expected_bytes *= dimension
        if len(payload) != expected_bytes:
            raise SystemExit(
                f"layer {layer} position {position} hook {hook} has "
                f"{len(payload)} bytes, expected {expected_bytes}"
            )
        destination = output / tensor["path"]
        shutil.copyfile(source, destination)
        imported = copy.deepcopy(tensor)
        imported["bytes"] = len(payload)
        imported["sha256"] = sha256(payload)
        tensors.append(imported)

    if position == 3 and layer % 2 == 0:
        source, payload = repeated_payload(capture_root, layer, "KVcompress", position)
        destination = output / "compressed-kv-row0.f32le.bin"
        shutil.copyfile(source, destination)
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

    environment_hooks = [tensor["hook"] for tensor in tensors]
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer{layer}-pos{position}-complete",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": template["oracle"],
        "model": template["model"],
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 1,
            "decode_step": position,
            "token_id": TOKENS[position],
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": str(layer),
                "DS4_METAL_GRAPH_DUMP_POS": str(position),
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(environment_hooks),
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "layer": layer,
            "position": position,
        },
        "operations": rewrite_operations(template, layer, position),
        "tensors": tensors,
    }
    if position == 1:
        manifest["capture"]["position0_cache_row_capture"] = True
        manifest["capture"]["position0_environment"] = {
            "DS4_METAL_GRAPH_DUMP_LAYER": str(layer),
            "DS4_METAL_GRAPH_DUMP_POS": "0",
            "DS4_METAL_GRAPH_DUMP_NAME": "KVcur",
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def import_prime(capture_root: Path, fixtures_root: Path, layer: int) -> Path:
    template = json.loads(
        (fixtures_root / "layer3-pos0-compressor-prime-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = fixtures_root / f"layer{layer}-pos0-compressor-prime-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    source, payload = repeated_payload(capture_root, layer, "attn_norm", 0)
    destination = output / "attn-norm.f32le.bin"
    shutil.copyfile(source, destination)
    manifest = copy.deepcopy(template)
    manifest["fixture_id"] = f"dwarfstar-oracle-v1-layer{layer}-pos0-compressor-prime"
    manifest["captured_at_utc"] = CAPTURED_AT_UTC
    manifest["capture"]["environment"]["DS4_METAL_GRAPH_DUMP_LAYER"] = str(layer)
    manifest["scope"]["layer"] = layer
    manifest["operations"][0]["weights"] = [f"blk.{layer}.attn_norm.weight"]
    manifest["tensors"][0]["bytes"] = len(payload)
    manifest["tensors"][0]["sha256"] = sha256(payload)
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
    for layer in (4, 5):
        for position in (1, 2, 3):
            print(import_complete(args.capture_root, args.fixtures_root, layer, position))
        print(import_prime(args.capture_root, args.fixtures_root, layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
