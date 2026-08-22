#!/usr/bin/env python3
"""Import the final layer-6 FFN tile and pin full 2K oracle identities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T18:54:01Z"
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
    "hc_ffn_pre": "cea7b754dee8e9dca2c642006b7d389851473fe392d0a0d33fe74347db563a54",
    "ffn_norm": "94196f7c685cdbbd7834da441d45e985ca0eb7cb8823ba70baf12e1049cb0546",
    "ffn_moe_logits": "c7a387c86c1c13a99b495c0e5384f4d8cbf9ed628f2cee0f990c2605d998da28",
    "ffn_moe_probs": "446d57f88a4fd4307f0effc2c3e698e3076c59f538710fd146c13c1c9d70379c",
    "ffn_moe_topk": "77aa003e005203e04253f0fbcf769b5478873fcb01c021b70fdcf33cca1d7dd4",
    "ffn_moe_weights_scaled": "ce0c93f615093f42df3d1c9769689fcfb5a2ebb2bafbd4ed4883e76499224f18",
    "ffn_moe_weighted_swiglu": "18754f302c93f8f41fad7dfb9968ceb1d98ee70086242f4acf24477e52976363",
    "ffn_moe_out": "438fb18dfaa989b602df314298be289782b6877f5150bc691c153769ea5aca27",
    "ffn_shexp": "2202c5671d0c5daa211471f97930b1ea5c62f02a1246678835a97e64a1126606",
    "hc_ffn_post": "ac0c7719df5c6ae1a646bd5670a69e83dbe23908f0ab5bb2be353c8aad3d6047",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, label: str, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return directory / f"{label}_{name}-6_pos0.{suffix}"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, "first", name).read_bytes()
    second = capture_path(second_dir, "second", name).read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-6 {name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-6 {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"layer-6 {name} capture identity changed")
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
        (args.fixtures_root / "prefill-layer6-attention-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-layer6-complete-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, width in WIDTHS.items():
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        payload = full[name][-TILE_ROWS * width * 4 :]
        filename = f"layer6-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer6_{name}_final_tile",
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer6-complete-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "6",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer6-attention-2048",
            "storage_note": (
                "Exact final-tile payloads are retained; full 2K tensor identities "
                "are pinned by SHA-256 to avoid a duplicate 128 MiB final-HC blob."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 6,
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
