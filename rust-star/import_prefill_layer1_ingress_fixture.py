#!/usr/bin/env python3
"""Import the final layer-1 attention-ingress tile of a 2K Metal prefill."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T19:41:45Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
WIDTHS = {
    "hc_attn_pre": 4_096,
    "attn_norm": 4_096,
    "q_lora": 1_024,
}
EXPECTED_FULL_SHA256 = {
    "hc_attn_pre": "a5ad89aaa9a5c3537c22a26730b918bdde4e07177e72c643059306bbb28439bc",
    "attn_norm": "37b0f1a783c0968445dd77214b2f62b62a0f5dcac01778ca5e0627948c392571",
    "q_lora": "ff0a9a83d0f3077c83ad1ba223310a496d3d8b51709da986d97887f4b0836901",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first_payload) != expected_size or len(second_payload) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first_payload


def final_tile(payload: bytes, width: int) -> bytes:
    return payload[-TILE_ROWS * width * 4 :]


def tensor(name: str, role: str, path: str, payload: bytes) -> dict:
    return {
        "name": f"layer1_{name}_final_tile",
        "hook": name,
        "role": role,
        "dtype": "f32",
        "shape": [TILE_ROWS, WIDTHS[name]],
        "encoding": "little-endian-ieee754-binary32",
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in WIDTHS:
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}-first", type=Path, required=True)
        parser.add_argument(f"--{option}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payloads = {}
    for name, width in WIDTHS.items():
        payloads[name] = final_tile(
            checked_repeated(
                name,
                getattr(args, f"{name}_first"),
                getattr(args, f"{name}_second"),
            ),
            width,
        )

    template = json.loads(
        (
            args.fixtures_root
            / "prefill-ffn-output-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer1-ingress-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        name: f"layer1-{name.replace('_', '-')}-final-tile.f32le.bin"
        for name in WIDTHS
    }
    for name, payload in payloads.items():
        (output / files[name]).write_bytes(payload)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer1-ingress-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "1",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "device_path": (
                "live layer-0 HC handoff; plain four-stream RMSNorm; legacy "
                "F16 HC batch mixer; fused HC split/collapse/learned norm; "
                "legacy Q8_0 Q-A batch projection"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-ffn-output-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 1,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "layer1-hc-ingress-and-norm-prefill",
                "kernel": (
                    "kernel_rms_norm_f32_4 plus kernel_mul_mm_f16_f32 plus "
                    "kernel_dsv4_hc_split_weighted_sum_norm4"
                ),
                "weights": [
                    "blk.1.hc_attn_fn.weight",
                    "blk.1.hc_attn_scale.weight",
                    "blk.1.hc_attn_base.weight",
                    "blk.1.attn_norm.weight",
                ],
                "dispatch": {"rows": TILE_ROWS, "width": 4_096, "hc_streams": 4},
            },
            {
                "name": "layer1-q-a-prefill",
                "kernel": "kernel_mul_mm_q8_0_f32",
                "weights": ["blk.1.attn_q_a.weight"],
                "dispatch": {"rows": TILE_ROWS, "input": 4_096, "output": 1_024},
            },
        ],
        "tensors": [
            tensor(
                name,
                "output" if name == "q_lora" else "intermediate",
                files[name],
                payloads[name],
            )
            for name in WIDTHS
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
