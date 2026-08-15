#!/usr/bin/env python3
"""Import independently repeated position-1 through -3 output-head fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T11:01:44Z"
INPUT_TOKENS = {1: 201, 2: 361, 3: 1915}
EXPECTED_ARGMAX = {1: 361, 2: 1915, 3: 262}
HOOKS = (
    ("output_hc_pre", "result_hc_pre", "output-hc-pre.f32le.bin", [4]),
    ("output_hc_weights", "result_hc_weights", "output-hc-weights.f32le.bin", [4]),
    ("output_hc", "result_hc", "output-hc.f32le.bin", [4096]),
    ("output_norm", "result_norm", "output-norm.f32le.bin", [4096]),
    ("logits", "result_output", "logits.f32le.bin", [129280]),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def captured_path(run: Path, position: int, hook: str) -> Path:
    return run / f"pos{position}" / f"oracle_{hook}-43_pos0.bin"


def repeated_payload(capture_root: Path, position: int, hook: str) -> tuple[Path, bytes]:
    first = captured_path(capture_root / "a", position, hook)
    second = captured_path(capture_root / "b", position, hook)
    payload = first.read_bytes()
    if payload != second.read_bytes():
        raise SystemExit(
            f"independent output-head captures differ at position {position} hook {hook}"
        )
    return first, payload


def argmax_lowest_id(payload: bytes) -> int:
    values = struct.unpack(f"<{len(payload) // 4}f", payload)
    return max(range(len(values)), key=values.__getitem__)


def import_fixture(capture_root: Path, fixtures_root: Path, position: int) -> Path:
    template = json.loads(
        (
            fixtures_root
            / f"layer42-pos{position}-complete-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = fixtures_root / f"output-head-pos{position}-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    logits = b""
    for name, hook, filename, shape in HOOKS:
        source, payload = repeated_payload(capture_root, position, hook)
        expected_bytes = 4
        for dimension in shape:
            expected_bytes *= dimension
        if len(payload) != expected_bytes:
            raise SystemExit(
                f"position {position} hook {hook} has {len(payload)} bytes, "
                f"expected {expected_bytes}"
            )
        shutil.copyfile(source, output / filename)
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
    if selected != EXPECTED_ARGMAX[position]:
        raise SystemExit(
            f"position {position} argmax is {selected}, expected {EXPECTED_ARGMAX[position]}"
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-output-head-pos{position}",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 1,
            "decode_step": position,
            "input_token_id": INPUT_TOKENS[position],
            "generation_tokens": position,
            "terminal_output_dump_writes": position + 1,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "43",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(hook for _, hook, _, _ in HOOKS),
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "layer": 43,
            "position": position,
        },
        "operations": [
            {
                "name": "output-hc-flat-rmsnorm",
                "kernel": "kernel_rms_norm_f32_4",
                "weights": [],
            },
            {
                "name": "output-hc-projection",
                "kernel": "kernel_mul_mv_f16_f32_4",
                "weights": ["output_hc_fn.weight"],
            },
            {
                "name": "output-hc-weights",
                "kernel": "kernel_dsv4_output_hc_weights4",
                "weights": ["output_hc_scale.weight", "output_hc_base.weight"],
            },
            {
                "name": "output-hc-collapse-and-norm",
                "kernel": "kernel_dsv4_hc_weighted_sum_norm4",
                "weights": ["output_norm.weight"],
            },
            {
                "name": "vocabulary-projection",
                "kernel": "kernel_mul_mv_q8_0_f32",
                "weights": ["output.weight"],
            },
            {
                "name": "next-token-selection",
                "kernel": "cpu-lowest-id-argmax",
                "weights": [],
            },
        ],
        "selection": {
            "method": "lowest-token-id-argmax",
            "token_id": selected,
        },
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
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
    for position in (1, 2, 3):
        print(import_fixture(args.capture_root, args.fixtures_root, position))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
