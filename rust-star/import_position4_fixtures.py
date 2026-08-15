#!/usr/bin/env python3
"""Import repeated full-decoder position-4 fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T12:00:34Z"
INPUT_TOKEN = 262
EXPECTED_ARGMAX = 1554
OUTPUT_HOOKS = (
    ("output_hc_pre", "result_hc_pre", "output-hc-pre.f32le.bin", [4]),
    ("output_hc_weights", "result_hc_weights", "output-hc-weights.f32le.bin", [4]),
    ("output_hc", "result_hc", "output-hc.f32le.bin", [4096]),
    ("output_norm", "result_norm", "output-norm.f32le.bin", [4096]),
    ("logits", "result_output", "logits.f32le.bin", [129280]),
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated_payload(first: Path, second: Path, label: str) -> bytes:
    payload = first.read_bytes()
    if payload != second.read_bytes():
        raise SystemExit(f"independent captures differ for {label}")
    return payload


def expected_bytes(tensor: dict) -> int:
    elements = 1
    for dimension in tensor["shape"]:
        elements *= dimension
    return elements * 4


def import_layer(capture_root: Path, fixtures_root: Path, layer: int) -> Path:
    template = json.loads(
        (fixtures_root / f"layer{layer}-pos3-complete-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    operations_template = json.loads(
        (fixtures_root / f"layer{layer}-pos2-complete-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = fixtures_root / f"layer{layer}-pos4-complete-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for tensor in template["tensors"]:
        if tensor["name"] in {"cache_row0", "compressed_kv_row0"}:
            continue
        hook = tensor["hook"]
        suffix = ".i32" if tensor["dtype"] == "i32" else ".bin"
        filename = f"oracle_{hook}-{layer}_pos4{suffix}"
        payload = repeated_payload(
            capture_root / "a" / "layers" / filename,
            capture_root / "b" / "layers" / filename,
            f"layer {layer} hook {hook}",
        )
        if len(payload) != expected_bytes(tensor):
            raise SystemExit(
                f"layer {layer} hook {hook} has {len(payload)} bytes, "
                f"expected {expected_bytes(tensor)}"
            )
        destination = output / tensor["path"]
        shutil.copyfile(capture_root / "a" / "layers" / filename, destination)
        imported = copy.deepcopy(tensor)
        imported["bytes"] = len(payload)
        imported["sha256"] = sha256(payload)
        tensors.append(imported)

    operations = copy.deepcopy(operations_template["operations"])
    if layer >= 2:
        for operation in operations:
            if operation["name"] == "raw-cache-stage-f32-to-f16":
                operation["name"] = "raw-and-persistent-compressed-cache-stage-f32-to-f16"
                break

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer{layer}-pos4-complete",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 1,
            "decode_step": 4,
            "token_id": INPUT_TOKEN,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "all",
                "DS4_METAL_GRAPH_DUMP_POS": "4",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(
                    tensor["hook"] for tensor in tensors
                ),
            },
            "batch_capture_layers": [0, 42],
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "layer": layer,
            "position": 4,
        },
        "operations": operations,
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def argmax_lowest_id(payload: bytes) -> int:
    values = struct.unpack(f"<{len(payload) // 4}f", payload)
    return max(range(len(values)), key=values.__getitem__)


def import_output(capture_root: Path, fixtures_root: Path) -> Path:
    template = json.loads(
        (fixtures_root / "output-head-pos3-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = fixtures_root / "output-head-pos4-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    logits = b""
    for name, hook, filename, shape in OUTPUT_HOOKS:
        captured = f"oracle_{hook}-43_pos0.bin"
        payload = repeated_payload(
            capture_root / "a" / "output" / captured,
            capture_root / "b" / "output" / captured,
            f"output-head hook {hook}",
        )
        size = 4
        for dimension in shape:
            size *= dimension
        if len(payload) != size:
            raise SystemExit(f"output-head hook {hook} has {len(payload)} bytes, expected {size}")
        shutil.copyfile(capture_root / "a" / "output" / captured, output / filename)
        tensors.append(
            {
                "name": name,
                "hook": hook,
                "role": "output" if name == "logits" else "intermediate",
                "dtype": "f32",
                "shape": shape,
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
        if name == "logits":
            logits = payload

    selected = argmax_lowest_id(logits)
    if selected != EXPECTED_ARGMAX:
        raise SystemExit(f"position 4 argmax is {selected}, expected {EXPECTED_ARGMAX}")
    manifest = copy.deepcopy(template)
    manifest["fixture_id"] = "dwarfstar-oracle-v1-output-head-pos4"
    manifest["captured_at_utc"] = CAPTURED_AT_UTC
    manifest["capture"]["decode_step"] = 4
    manifest["capture"]["input_token_id"] = INPUT_TOKEN
    manifest["capture"]["generation_tokens"] = 4
    manifest["capture"]["terminal_output_dump_writes"] = 5
    manifest["scope"]["position"] = 4
    manifest["selection"]["token_id"] = selected
    manifest["tensors"] = tensors
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
    for layer in range(43):
        print(import_layer(args.capture_root, args.fixtures_root, layer))
    print(import_output(args.capture_root, args.fixtures_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
