#!/usr/bin/env python3
"""Import independently repeated position-2 DwarfStar graph dumps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


POSITION = 2
TOKEN = 361
CAPTURED_AT_UTC = "2026-08-15T07:39:44Z"


def captured_path(run: Path, layer: int, hook: str) -> Path:
    suffix = ".i32" if hook == "ffn_moe_topk" else ".bin"
    return run / f"oracle_{hook}-{layer}_pos{POSITION}{suffix}"


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
    return operations


def import_layer(capture_root: Path, fixtures_root: Path, layer: int) -> Path:
    template_layer = 1 if layer == 0 else layer
    template_path = fixtures_root / f"layer{template_layer}-complete-v1" / "manifest.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output = fixtures_root / f"layer{layer}-pos2-complete-v1"
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

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer{layer}-pos2-complete",
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
        output = import_layer(args.capture_root, args.fixtures_root, layer)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
