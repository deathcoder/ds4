#!/usr/bin/env python3
"""Import repeated full-2K layer-6 attention and HC-post captures."""

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
    "attn_out": "09dfe1c15053cc8dd0eaa3ea3ea565684fc3e599ab5dfe69dfb721f63b7f6a0a",
    "hc_attn_post": "c9a2dc9721e5d5cecbcc207a8c834eb8f5115f99496c0f99d96c8db91e490b44",
}
DIAGNOSTIC_WIDTHS = {"kqv_out": 32_768, "kqv_back": 32_768, "attn_low": 8_192}
EXPECTED_DIAGNOSTIC_FULL_SHA256 = {
    "kqv_out": "cf85fa944f7164ca64bdc10f48e188572bf24d0967d2c2eb2ba6537e2ced2c85",
    "kqv_back": "63938bb309f28ccfb7284ed89ab7f3b701f80275546c89ed844b730b287195a9",
    "attn_low": "8c5aec1ddca5ab367e7d520cd1b4fb5d53e5f1bee41e3d0f8201b659d9202ac8",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, name: str) -> Path:
    return directory / f"{directory.name}_{name}-6_pos0.bin"


def checked(name: str, first: Path, second: Path, width: int, identity: str) -> bytes:
    left = capture_path(first, name).read_bytes()
    right = capture_path(second, name).read_bytes()
    if len(left) != PREFILL_ROWS * width * 4 or len(right) != len(left):
        raise SystemExit(f"layer-6 {name} capture has the wrong size")
    if left != right:
        raise SystemExit(f"fresh-process layer-6 {name} captures differ")
    if sha256(left) != identity:
        raise SystemExit(f"layer-6 {name} capture identity changed")
    return left


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--fixtures-root", type=Path,
        default=Path(__file__).resolve().parent / "fixtures")
    args = parser.parse_args()

    full = {name: checked(name, args.first_dir, args.second_dir, width,
                          EXPECTED_FULL_SHA256[name])
            for name, width in WIDTHS.items()}
    diagnostic = {name: checked(name, args.first_dir, args.second_dir, width,
                                EXPECTED_DIAGNOSTIC_FULL_SHA256[name])
                  for name, width in DIAGNOSTIC_WIDTHS.items()}
    template = json.loads((args.fixtures_root / "prefill-layer6-compressors-2048-v1" /
                           "manifest.json").read_text(encoding="utf-8"))
    output = args.fixtures_root / "prefill-layer6-attention-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    attn_filename = "layer6-attention-output.f32le.bin"
    (output / attn_filename).write_bytes(full["attn_out"])
    hc_payload = full["hc_attn_post"][-TILE_ROWS * WIDTHS["hc_attn_post"] * 4:]
    hc_filename = "layer6-hc-attn-post-final-tile.f32le.bin"
    (output / hc_filename).write_bytes(hc_payload)
    tensors = []
    for name, width in DIAGNOSTIC_WIDTHS.items():
        payload = diagnostic[name][:width * 4]
        filename = f"layer6-{name.replace('_', '-')}-row0.f32le.bin"
        (output / filename).write_bytes(payload)
        tensors.append({"name": f"layer6_{name}_row0", "hook": name,
            "role": "intermediate", "dtype": "f32", "shape": [1, width],
            "encoding": "little-endian-ieee754-binary32", "path": filename,
            "bytes": len(payload), "sha256": sha256(payload)})

    tensors.extend([
        {"name": "layer6_attention_output", "hook": "attn_out", "role": "output",
         "dtype": "f32", "shape": [PREFILL_ROWS, WIDTHS["attn_out"]],
         "encoding": "little-endian-ieee754-binary32", "path": attn_filename,
         "bytes": len(full["attn_out"]), "sha256": sha256(full["attn_out"])},
        {"name": "layer6_hc_attn_post_final_tile", "hook": "hc_attn_post",
         "role": "output", "dtype": "f32",
         "shape": [TILE_ROWS, WIDTHS["hc_attn_post"]],
         "encoding": "little-endian-ieee754-binary32", "path": hc_filename,
         "bytes": len(hc_payload), "sha256": sha256(hc_payload)},
    ])
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer6-attention-2048",
        "captured_at_utc": "2026-08-22T18:21:38Z",
        "oracle": copy.deepcopy(template["oracle"]), "model": copy.deepcopy(template["model"]),
        "capture": {"backend": "metal", "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS, "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True, "command": template["capture"]["command"],
            "environment": {"DS4_METAL_GRAPH_DUMP_LAYER": "6",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_out,hc_attn_post,kqv_out,kqv_back,attn_low"},
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "diagnostic_full_capture_sha256": EXPECTED_DIAGNOSTIC_FULL_SHA256,
            "device_path": "full Q projection, dense mixed 2048 raw plus 512 ratio-4 compressed FlashAttention, inverse compressed RoPE, grouped/dense Q8_0 output projections, and additive HC post",
            "indexer_policy": "all 512 compressed rows remain dense at the sparse-indexer threshold",
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer6-compressors-2048",
            "storage_note": "The full 32 MiB attention output is retained. Only the final 32-row HC tile is retained; the full 128 MiB HC identity is SHA-256 pinned."},
        "scope": {"kind": "layer-segment", "phase": "prefill", "layer": 6,
            "position": 2047, "captured_position_range": [0, 2047]},
        "operations": [
            {"name": "layer6-dense-mixed-attention", "kernel": "kernel_flash_attn_ext_f16_dk512_dv512"},
            {"name": "layer6-attention-inverse-rope", "kernel": "kernel_rope_norm_f32"},
            {"name": "layer6-attention-output-projections", "kernel": "kernel_mul_mm_id_q8_0_f32 + kernel_mul_mm_q8_0_f32"},
            {"name": "layer6-attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"}],
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
