#!/usr/bin/env python3
"""Import the final layer-2 tile and pin full 2K oracle identities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
WIDTHS = {
    "hc_attn_post": 16_384,
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
    "hc_attn_post": "4f7c61ad617347f186cb959457d5c6f1c95692451bf186c751f863a6417baad2",
    "hc_ffn_pre": "aef50808ade7e95aa16ee2fac006d2c2ae5db58045811e15b40efdf20536ca60",
    "ffn_norm": "d15b7bbd190a86e59949c8037070608c993b647f9c89929afbebf6fa50ef532e",
    "ffn_moe_logits": "0f0061283e8061efc101effb2cdf52af61a8edb9c9053a2e7ae4d72a2b93802b",
    "ffn_moe_probs": "00dc7108e22c24a97e922aa5cb4f9a75cc79c6045b35d776430950ffc1226778",
    "ffn_moe_topk": "d3ca48f1ed1e127333ab32929607d48e7c5a151d0d9436cc9cdc5545e64e7f26",
    "ffn_moe_weights_scaled": "493fbd165ce6684ca318dbb1aa11e056e9d38d7b966bc81ece78c947923a3a1e",
    "ffn_moe_weighted_swiglu": "d51cd0e38a5b84197622dc6485db0bb9a5e8c54bf33c24d95adebdc0bc0e6e9b",
    "ffn_moe_out": "1ddb0dfea9d19991535365dbe972ace93c1e0d9f1a094910062ea201a36b04d2",
    "ffn_shexp": "a5f91ee9f904e0bf42d3fd8b6eb32dcd4bf0d38a136139771e2899e93297e090",
    "hc_ffn_post": "67dbac97346ee6bea6bb967eafa6c7841fb3477798f9cc154c9dfed24ff5564b",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, label: str, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return directory / f"{label}_{name}-2_pos0.{suffix}"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, "first", name).read_bytes()
    second = capture_path(second_dir, "second", name).read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
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
        (args.fixtures_root / "prefill-layer2-attention-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer2-complete-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, width in WIDTHS.items():
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        payload = full[name][-TILE_ROWS * width * 4 :]
        filename = f"layer2-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        (output / filename).write_bytes(payload)
        tensors.append({
            "name": f"layer2_{name}_final_tile",
            "hook": name,
            "role": "output" if name == "hc_ffn_post" else "intermediate",
            "dtype": dtype,
            "shape": [TILE_ROWS, width],
            "encoding": (
                "little-endian-signed-integer32"
                if dtype == "i32" else "little-endian-ieee754-binary32"
            ),
            "path": filename,
            "bytes": len(payload),
            "sha256": sha256(payload),
        })

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer2-complete-2048",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer2-attention-2048",
            "storage_note": (
                "Exact final-tile payloads are retained; full 2K tensor identities "
                "are pinned by SHA-256 to avoid a 128 MiB final-HC fixture blob."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {"name": "attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
            {"name": "ffn-ingress-router", "kernel": "batch HC ingress plus token-hash router"},
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
