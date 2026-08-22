#!/usr/bin/env python3
"""Import repeated full-2K layer-5 attention and HC-post captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
WIDTHS = {"attn_out": 4_096, "hc_attn_post": 16_384}
EXPECTED_FULL_SHA256 = {
    "attn_out": "cd8fa45e025155709682f9479441998fb89566bb80f70bab61c1275fc1e4e3cd",
    "hc_attn_post": "529c1139057c0733b2c660c718a05605e09c6f66c09ace454c5443dece0d181a",
}
DIAGNOSTIC_WIDTHS = {"kqv_out": 32_768, "kqv_back": 32_768, "attn_low": 8_192}
EXPECTED_DIAGNOSTIC_FULL_SHA256 = {
    "kqv_out": "daf21a8ac02efc190e778fcb747a69f7e92d7a15a9635e3a2574526bd00728f8",
    "kqv_back": "e473d63c789f3cb6a43c189a9906589b48e44ca546e938916aeae486c8663aa0",
    "attn_low": "1edfcb86724792007946bdddfb81ffd038ecf2e389510aec62e0600392aa8eb8",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, name: str) -> Path:
    return directory / f"{directory.name}_{name}-5_pos0.bin"


def checked(name: str, first: Path, second: Path, width: int, identity: str) -> bytes:
    left = capture_path(first, name).read_bytes()
    right = capture_path(second, name).read_bytes()
    if len(left) != PREFILL_ROWS * width * 4 or len(right) != len(left):
        raise SystemExit(f"layer-5 {name} capture has the wrong size")
    if left != right:
        raise SystemExit(f"fresh-process layer-5 {name} captures differ")
    if sha256(left) != identity:
        raise SystemExit(f"layer-5 {name} capture identity changed")
    return left


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
        name: checked(name, args.first_dir, args.second_dir, width, EXPECTED_FULL_SHA256[name])
        for name, width in WIDTHS.items()
    }
    diagnostic = {
        name: checked(
            name,
            args.first_dir,
            args.second_dir,
            width,
            EXPECTED_DIAGNOSTIC_FULL_SHA256[name],
        )
        for name, width in DIAGNOSTIC_WIDTHS.items()
    }
    template = json.loads(
        (args.fixtures_root / "prefill-layer5-compressor-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-layer5-attention-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    attn_filename = "layer5-attention-output.f32le.bin"
    (output / attn_filename).write_bytes(full["attn_out"])
    hc_payload = full["hc_attn_post"][-TILE_ROWS * WIDTHS["hc_attn_post"] * 4 :]
    hc_filename = "layer5-hc-attn-post-final-tile.f32le.bin"
    (output / hc_filename).write_bytes(hc_payload)
    tensors = []
    for name, width in DIAGNOSTIC_WIDTHS.items():
        payload = diagnostic[name][: width * 4]
        filename = f"layer5-{name.replace('_', '-')}-row0.f32le.bin"
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer5_{name}_row0",
                "hook": name,
                "role": "intermediate",
                "dtype": "f32",
                "shape": [1, width],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    tensors.extend(
        [
            {
                "name": "layer5_attention_output",
                "hook": "attn_out",
                "role": "output",
                "dtype": "f32",
                "shape": [PREFILL_ROWS, WIDTHS["attn_out"]],
                "encoding": "little-endian-ieee754-binary32",
                "path": attn_filename,
                "bytes": len(full["attn_out"]),
                "sha256": sha256(full["attn_out"]),
            },
            {
                "name": "layer5_hc_attn_post_final_tile",
                "hook": "hc_attn_post",
                "role": "output",
                "dtype": "f32",
                "shape": [TILE_ROWS, WIDTHS["hc_attn_post"]],
                "encoding": "little-endian-ieee754-binary32",
                "path": hc_filename,
                "bytes": len(hc_payload),
                "sha256": sha256(hc_payload),
            },
        ]
    )
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer5-attention-2048",
        "captured_at_utc": "2026-08-22T15:58:37Z",
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
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_out,hc_attn_post,kqv_out,kqv_back,attn_low",
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "diagnostic_full_capture_sha256": EXPECTED_DIAGNOSTIC_FULL_SHA256,
            "device_path": "full Q projection, dense mixed 2048 raw plus 16 ratio-128 compressed FlashAttention, inverse compressed RoPE, grouped/dense Q8_0 output projections, and additive HC post",
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer5-compressor-2048",
            "storage_note": "The full 32 MiB attention output is retained. Only the final 32-row HC tile is retained; the full 128 MiB HC identity is SHA-256 pinned.",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 5,
            "position": 2047,
            "captured_position_range": [0, 2047],
        },
        "operations": [
            {
                "name": "layer5-dense-mixed-attention",
                "kernel": "kernel_flash_attn_ext_f16_dk512_dv512",
            },
            {"name": "layer5-attention-inverse-rope", "kernel": "kernel_rope_norm_f32"},
            {
                "name": "layer5-attention-output-projections",
                "kernel": "kernel_mul_mm_id_q8_0_f32 + kernel_mul_mm_q8_0_f32",
            },
            {"name": "layer5-attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
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
