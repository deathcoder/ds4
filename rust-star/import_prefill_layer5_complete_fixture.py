#!/usr/bin/env python3
"""Import the final layer-5 FFN tile and pin full 2K oracle identities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T16:27:08Z"
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
EXPECTED_FULL_SHA256 = {
    "hc_ffn_pre": "a65589d1df043427c2fcc293163a326f3de22a394f1549584c7ce98058bca375",
    "ffn_norm": "397899af6f1f96f31227b1591ee89005ab6ae69111394cfe2ce9f0120e1651a9",
    "ffn_moe_logits": "3ad067f5619f79683328293642b1cc4806b46dc4e3271d8617335cf1d70edf57",
    "ffn_moe_probs": "da17cba1c0234f27a91207c4d1cf79136523b00302a0a27d6ed6dc9052ab50db",
    "ffn_moe_topk": "c242155d9919b0ebebd1df8ec489c2ecd1e342d7855df337e37cab11fb9e1f5f",
    "ffn_moe_weights_scaled": "5f4391666ce871629091e8f1855e2a8807bf7f53224a507b2f2b6a5c56a210c1",
    "ffn_moe_weighted_swiglu": "46ceb7379920f2591ee1a44333153d5ef5138aa2017ec8bd4f1fa5bac732832e",
    "ffn_moe_out": "f35a76c312d1e6df02bc695b9e09dbf201b888b8861594188f01314eec02b763",
    "ffn_shexp": "bf69681c377c88397f39a239fe27566df66110aa997f0e5f84225c56db780737",
    "hc_ffn_post": "438f1ea2118caec39106a222ff239145b1e64cd89f79e49068917ddb5c74b7ce",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, label: str, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return directory / f"{label}_{name}-5_pos0.{suffix}"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, "first", name).read_bytes()
    second = capture_path(second_dir, "second", name).read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-5 {name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-5 {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"layer-5 {name} capture identity changed")
    return first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    full = {
        name: checked_repeated(name, args.first_dir, args.second_dir)
        for name in WIDTHS
    }
    template = json.loads(
        (args.fixtures_root / "prefill-layer5-attention-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-layer5-complete-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, width in WIDTHS.items():
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        payload = full[name][-TILE_ROWS * width * 4 :]
        filename = f"layer5-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer5_{name}_final_tile",
                "hook": name,
                "role": "output" if name == "hc_ffn_post" else "intermediate",
                "dtype": dtype,
                "shape": [TILE_ROWS, width],
                "encoding": (
                    "little-endian-signed-integer32"
                    if dtype == "i32"
                    else "little-endian-ieee754-binary32"
                ),
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer5-complete-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "5",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer5-attention-2048",
            "storage_note": (
                "Exact final-tile payloads are retained; full 2K tensor identities "
                "are pinned by SHA-256 to avoid a duplicate 128 MiB final-HC blob."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 5,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "ffn-hc-ingress",
                "kernel": "kernel_dsv4_hc_split_weighted_sum_norm4",
            },
            {
                "name": "biased-topk-router",
                "kernel": "sqrt(softplus(logits)) plus bias top-6",
            },
            {
                "name": "routed-experts",
                "kernel": "fused IQ2_XXS pair-SwiGLU plus Q2_K down",
            },
            {
                "name": "shared-expert",
                "kernel": "Q8_0 gate/up/down plus flat SwiGLU",
            },
            {"name": "ffn-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
        ],
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
