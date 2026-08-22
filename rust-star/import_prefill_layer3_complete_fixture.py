#!/usr/bin/env python3
"""Import the final layer-3 FFN tile and pin full 2K oracle identities."""

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
    "hc_ffn_pre": "2b27cb1ea3add5ca308c3bcf9e3394590a2006a96dcfd5a33d3db1c9d3068466",
    "ffn_norm": "10252f5833e8da20687ac844e021dd26a55a45fa52153d2dab7adaadf2a95357",
    "ffn_moe_logits": "881c80cf28e0dee6f80c4fbe415214cec2d62b50b0299831c19c3e78eae26982",
    "ffn_moe_probs": "cdd179a9ca528a62c32ab0c33dcd61e674a8deb88211f832ecdbe522f942f1e1",
    "ffn_moe_topk": "b81db8474bb98befe5804ecb811ae6163e6ac34c8fc83d613925394eb2b680ee",
    "ffn_moe_weights_scaled": "da5526d0ccfec266ffb893e8e15cef3cc204819c9e7d66f89550ca786fe45731",
    "ffn_moe_weighted_swiglu": "231b0c6fe9f4f6edfab2aac6c89464a2f3b8f60a1d20f0ed647d30235ee73b8e",
    "ffn_moe_out": "0d16b8a089e4851265500c84dcdcdce23e33133d685c034224c1d807ac690b17",
    "ffn_shexp": "39d50ff560f63b75a8e73a1eda97c6359b3e55590468dd696f8b99917d357c2c",
    "hc_ffn_post": "0b369e305e9627648677cf3032aa221a7d921fcbeb1514037d900f90cb71f7fd",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, label: str, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return directory / f"{label}_{name}-3_pos0.{suffix}"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, "first", name).read_bytes()
    second = capture_path(second_dir, "second", name).read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-3 {name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-3 {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"layer-3 {name} capture identity changed")
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
        (
            args.fixtures_root
            / "prefill-layer3-attention-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer3-complete-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, width in WIDTHS.items():
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        payload = full[name][-TILE_ROWS * width * 4 :]
        filename = f"layer3-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer3_{name}_final_tile",
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer3-complete-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "3",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer3-attention-2048",
            "storage_note": (
                "Exact final-tile payloads are retained; full 2K tensor identities "
                "are pinned by SHA-256 to avoid a duplicate 128 MiB final-HC blob."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 3,
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
