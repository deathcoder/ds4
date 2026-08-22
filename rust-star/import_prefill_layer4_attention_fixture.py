#!/usr/bin/env python3
"""Import repeated full-2K layer-4 attention and HC-post captures."""

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
    "attn_out": "a45f8d789e6c9db117fe46102695042be83bb67be235822c7034feb321c00909",
    "hc_attn_post": "61a26f21534f8d5eb076386fa6fba6ed96fdddc591df90e27078ae924c9feb05",
}
DIAGNOSTIC_WIDTHS = {"kqv_out": 32_768, "kqv_back": 32_768, "attn_low": 8_192}
EXPECTED_DIAGNOSTIC_FULL_SHA256 = {
    "kqv_out": "691215cf6ff0fb79a4074329baeb9dfdfd365795a71a847874bc27b75991c6ba",
    "kqv_back": "0a8e986d40523ae78078a7e31f23358838e6b3cab42dc4a9411a5baaef870417",
    "attn_low": "ce05b7ee002f57330eec842766f9120fa0c547c8591e3a1711a3319061cf5dd6",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, name: str) -> Path:
    return directory / f"{directory.name}_{name}-4_pos0.bin"


def checked(name: str, first: Path, second: Path, width: int, identity: str) -> bytes:
    left = capture_path(first, name).read_bytes()
    right = capture_path(second, name).read_bytes()
    if len(left) != PREFILL_ROWS * width * 4 or len(right) != len(left):
        raise SystemExit(f"layer-4 {name} capture has the wrong size")
    if left != right:
        raise SystemExit(f"fresh-process layer-4 {name} captures differ")
    if sha256(left) != identity:
        raise SystemExit(f"layer-4 {name} capture identity changed")
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
    template = json.loads((args.fixtures_root / "prefill-layer4-compressors-2048-v1" /
                           "manifest.json").read_text(encoding="utf-8"))
    output = args.fixtures_root / "prefill-layer4-attention-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    attn_filename = "layer4-attention-output.f32le.bin"
    (output / attn_filename).write_bytes(full["attn_out"])
    hc_payload = full["hc_attn_post"][-TILE_ROWS * WIDTHS["hc_attn_post"] * 4:]
    hc_filename = "layer4-hc-attn-post-final-tile.f32le.bin"
    (output / hc_filename).write_bytes(hc_payload)
    tensors = []
    for name, width in DIAGNOSTIC_WIDTHS.items():
        payload = diagnostic[name][:width * 4]
        filename = f"layer4-{name.replace('_', '-')}-row0.f32le.bin"
        (output / filename).write_bytes(payload)
        tensors.append({"name": f"layer4_{name}_row0", "hook": name,
            "role": "intermediate", "dtype": "f32", "shape": [1, width],
            "encoding": "little-endian-ieee754-binary32", "path": filename,
            "bytes": len(payload), "sha256": sha256(payload)})

    tensors.extend([
        {"name": "layer4_attention_output", "hook": "attn_out", "role": "output",
         "dtype": "f32", "shape": [PREFILL_ROWS, WIDTHS["attn_out"]],
         "encoding": "little-endian-ieee754-binary32", "path": attn_filename,
         "bytes": len(full["attn_out"]), "sha256": sha256(full["attn_out"])},
        {"name": "layer4_hc_attn_post_final_tile", "hook": "hc_attn_post",
         "role": "output", "dtype": "f32",
         "shape": [TILE_ROWS, WIDTHS["hc_attn_post"]],
         "encoding": "little-endian-ieee754-binary32", "path": hc_filename,
         "bytes": len(hc_payload), "sha256": sha256(hc_payload)},
    ])
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer4-attention-2048",
        "captured_at_utc": "2026-08-22T13:37:13Z",
        "oracle": copy.deepcopy(template["oracle"]), "model": copy.deepcopy(template["model"]),
        "capture": {"backend": "metal", "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS, "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True, "command": template["capture"]["command"],
            "environment": {"DS4_METAL_GRAPH_DUMP_LAYER": "4",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_out,hc_attn_post,kqv_out,kqv_back,attn_low"},
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "diagnostic_full_capture_sha256": EXPECTED_DIAGNOSTIC_FULL_SHA256,
            "device_path": "full Q projection, dense mixed 2048 raw plus 512 ratio-4 compressed FlashAttention, inverse compressed RoPE, grouped/dense Q8_0 output projections, and additive HC post",
            "indexer_policy": "all 512 compressed rows remain dense at the sparse-indexer threshold",
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer4-compressors-2048",
            "storage_note": "The full 32 MiB attention output is retained. Only the final 32-row HC tile is retained; the full 128 MiB HC identity is SHA-256 pinned."},
        "scope": {"kind": "layer-segment", "phase": "prefill", "layer": 4,
            "position": 2047, "captured_position_range": [0, 2047]},
        "operations": [
            {"name": "layer4-dense-mixed-attention", "kernel": "kernel_flash_attn_ext_f16_dk512_dv512"},
            {"name": "layer4-attention-inverse-rope", "kernel": "kernel_rope_norm_f32"},
            {"name": "layer4-attention-output-projections", "kernel": "kernel_mul_mm_id_q8_0_f32 + kernel_mul_mm_q8_0_f32"},
            {"name": "layer4-attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"}],
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
