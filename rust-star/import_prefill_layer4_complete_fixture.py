#!/usr/bin/env python3
"""Import the final layer-4 FFN tile and pin full 2K oracle identities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T15:16:51Z"
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
    "hc_ffn_pre": "f373db94f0ee17a449aad60bba5dce5b7323e71e35dec652077266bb6e0992ad",
    "ffn_norm": "cf8fe357bab0da682ec0953d6ef08be14e6496ed78e0f964a4e272bf410a7505",
    "ffn_moe_logits": "c85ad9533be6ba54783ddd732392c85708b9363456c1310a63917d39e34f39e2",
    "ffn_moe_probs": "1d044c0972257f0b142e28306c23ddb517ff9f75766a85417b61424548b88d9f",
    "ffn_moe_topk": "409de6c217732c663fa727a3fc41e64e33c543832a1be8badf0af926d8956913",
    "ffn_moe_weights_scaled": "02c3532764b966213a639be76d7620a0133f4a7deb5a31e91efc1d8f415b9679",
    "ffn_moe_weighted_swiglu": "097b5b96bc974914269f7021d116aa4a8712ccc9dd666104ef03c689e446f1e7",
    "ffn_moe_out": "84809f73332955cb67362e885c78b22feb4d671183d7fdfcb98fbc1022edf707",
    "ffn_shexp": "8164d3b0d45922c37d0d5b84cd87b1c28976dd7d3300673f55d24aea25651083",
    "hc_ffn_post": "2a308ac3b3ad0aa603f088249e59d83a6c63a08c8af373283c696b52dc59d40e",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, label: str, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return directory / f"{label}_{name}-4_pos0.{suffix}"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, "first", name).read_bytes()
    second = capture_path(second_dir, "second", name).read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-4 {name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-4 {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"layer-4 {name} capture identity changed")
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
        (args.fixtures_root / "prefill-layer4-attention-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer4-complete-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, width in WIDTHS.items():
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        payload = full[name][-TILE_ROWS * width * 4 :]
        filename = f"layer4-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer4_{name}_final_tile",
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer4-complete-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "4",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer4-attention-2048",
            "storage_note": (
                "Exact final-tile payloads are retained; full 2K tensor identities "
                "are pinned by SHA-256 to avoid a duplicate 128 MiB final-HC blob."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 4,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {"name": "ffn-hc-ingress", "kernel": "kernel_dsv4_hc_split_weighted_sum_norm4"},
            {"name": "biased-topk-router", "kernel": "sqrt(softplus(logits)) plus bias top-6"},
            {"name": "routed-experts", "kernel": "fused IQ2_XXS pair-SwiGLU plus Q2_K down"},
            {"name": "shared-expert", "kernel": "Q8_0 gate/up/down plus flat SwiGLU"},
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
