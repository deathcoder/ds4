#!/usr/bin/env python3
"""Import the final layer-0 FFN tile of a 2K Metal prefill."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T19:00:18Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
WIDTHS = {
    "hc_ffn_pre": 4_096,
    "ffn_norm": 4_096,
    "ffn_moe_logits": 256,
    "ffn_moe_probs": 256,
    "ffn_moe_topk": 6,
    "ffn_moe_weights_scaled": 6,
    "ffn_moe_weighted_swiglu": 6 * 2_048,
    "ffn_moe_out": 4_096,
    "ffn_shexp": 4_096,
    "hc_ffn_post": 16_384,
}
DTYPES = {name: "f32" for name in WIDTHS}
DTYPES["ffn_moe_topk"] = "i32"
EXPECTED_FULL_SHA256 = {
    "hc_ffn_pre": "3bc23ac98132e33d0f774a001fb589625b485489ddf6fb9524bb49d64c4c00bc",
    "ffn_norm": "5178b044209404cb3d418a94d92d3c3f5800ebf90f280edbcac5409f7106160e",
    "ffn_moe_logits": "d2166f0ae63305bde07e63183f76087c58b21e1b0bf49ad5cf0a3156693a0a6d",
    "ffn_moe_probs": "94b008e42d9d47716772f7e38414adeaade55941d7f63892e8360d4d5600dd60",
    "ffn_moe_topk": "2af6c8b65918ddbfa1b06afba3be9b8820450cfa6027865fbd8973a8dc526bf1",
    "ffn_moe_weights_scaled": "ebc4bede76fd61495a585285b14f4bd28afb7c2e5d234f338123cd34599f1640",
    "ffn_moe_weighted_swiglu": "e9735faca9e95a6d50e296d439656dfd6300a183d1213cbf6c9f41b1659a1be8",
    "ffn_moe_out": "aa85518aeb35e700fc7eee2b7fcb0d7ba26275d9455b8154aac484ba916fb8fa",
    "ffn_shexp": "b5b2b465cd8cee6bd7c169bc3670519077211b087600679060a58701bc5fe859",
    "hc_ffn_post": "5540a37fd14e8d3f9eb5d2b2ac0e515366410ea7ae66530c2eb7ccf6fdf12930",
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
    dtype = DTYPES[name]
    return {
        "name": f"{name}_final_tile",
        "hook": name,
        "role": role,
        "dtype": dtype,
        "shape": [TILE_ROWS, WIDTHS[name]],
        "encoding": (
            "little-endian-signed-integer32"
            if dtype == "i32"
            else "little-endian-ieee754-binary32"
        ),
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
        option = name.replace("_", "-")
        payloads[name] = final_tile(
            checked_repeated(
                name,
                getattr(args, f"{option.replace('-', '_')}_first"),
                getattr(args, f"{option.replace('-', '_')}_second"),
            ),
            width,
        )

    template = json.loads(
        (
            args.fixtures_root
            / "prefill-attention-output-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-ffn-output-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        name: f"{name.replace('_', '-')}-final-tile.{DTYPES[name]}le.bin"
        for name in WIDTHS
    }
    for name, payload in payloads.items():
        (output / files[name]).write_bytes(payload)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-ffn-output-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "device_path": (
                "fused four-stream HC split/collapse/learned norm; legacy F16 "
                "router batch matmul; token-hash routing; legacy expert-major "
                "IQ2_XXS/Q2_K batch matmul with an F16 weighted-SwiGLU "
                "intermediate; legacy Q8_0 shared expert; additive HC expand"
            ),
            "graph_dump_fusion_note": (
                "DwarfStar disables the routed pair-SwiGLU fusion while graph "
                "dumping; the production source contract requires its F16 "
                "intermediate and routed output to be bit-identical to the "
                "retained unfused diagnostic path"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-attention-output-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 0,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "ffn-hc-ingress-and-norm-prefill",
                "kernel": "kernel_dsv4_hc_split_weighted_sum_norm4",
                "weights": [
                    "blk.0.hc_ffn_fn.weight",
                    "blk.0.hc_ffn_scale.weight",
                    "blk.0.hc_ffn_base.weight",
                    "blk.0.ffn_norm.weight",
                ],
                "dispatch": {"rows": TILE_ROWS, "width": 4_096, "hc_streams": 4},
            },
            {
                "name": "ffn-router-prefill",
                "kernel": "kernel_mul_mm_f16_f32 plus legacy batch router kernels",
                "weights": ["blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_tid2eid.weight"],
                "dispatch": {"rows": TILE_ROWS, "experts": 256, "experts_used": 6},
            },
            {
                "name": "routed-experts-prefill",
                "kernel": "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16",
                "weights": [
                    "blk.0.ffn_gate_exps.weight",
                    "blk.0.ffn_up_exps.weight",
                    "blk.0.ffn_down_exps.weight",
                ],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "experts": 256,
                    "experts_used": 6,
                    "map_kernel": "kernel_mul_mm_id_map0_ne20_6",
                    "down_kernel": "kernel_mul_mm_id_q2_K_f16",
                    "sum_kernel": "kernel_dsv4_moe_sum6_f32",
                },
            },
            {
                "name": "shared-expert-prefill",
                "kernel": "kernel_mul_mm_q8_0_f32 plus kernel_swiglu_flat_f32",
                "weights": [
                    "blk.0.ffn_gate_shexp.weight",
                    "blk.0.ffn_up_shexp.weight",
                    "blk.0.ffn_down_shexp.weight",
                ],
                "dispatch": {"rows": TILE_ROWS, "inner_width": 2_048},
            },
            {
                "name": "ffn-hc-post-prefill",
                "kernel": "kernel_dsv4_hc_expand4",
                "weights": [],
                "dispatch": {"rows": TILE_ROWS, "width": 4_096, "has_add": True},
            },
        ],
        "tensors": [
            tensor(name, "output" if name == "hc_ffn_post" else "intermediate", files[name], payloads[name])
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
