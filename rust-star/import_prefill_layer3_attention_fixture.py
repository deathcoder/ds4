#!/usr/bin/env python3
"""Import repeated full-2K layer-3 attention and HC-post captures."""

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
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
}
EXPECTED_FULL_SHA256 = {
    "attn_out": "5bfadcbc1d2ee7b42753b506420045409ba277ae2ffc7dbd0e87187e51b74e13",
    "hc_attn_post": "47c26665144097e0912284961d95f9b3ae72c8ce40c271e028dfdf71f4ee453b",
}
DIAGNOSTIC_WIDTHS = {
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
EXPECTED_DIAGNOSTIC_FULL_SHA256 = {
    "kqv_out": "cb3736c069ce269d8fcfcc3eec2318e21962166c943e30e642688862eca819a8",
    "kqv_back": "ff5919d442d37b7e2cd7ab6710a9bc814d20b1558f08b2c19db455b9c559367b",
    "attn_low": "c95872dab4cc689e2303ec85ebd90aafdcaa375824146042d13fd0290991d669",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, name: str) -> Path:
    return directory / f"oracle_{name}-3_pos0.bin"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, name).read_bytes()
    second = capture_path(second_dir, name).read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-3 {name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-3 {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"layer-3 {name} capture identity changed")
    return first


def checked_diagnostic(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first = capture_path(first_dir, name).read_bytes()
    second = capture_path(second_dir, name).read_bytes()
    expected_size = PREFILL_ROWS * DIAGNOSTIC_WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-3 {name} diagnostic capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-3 {name} diagnostic captures differ")
    if sha256(first) != EXPECTED_DIAGNOSTIC_FULL_SHA256[name]:
        raise SystemExit(f"layer-3 {name} diagnostic identity changed")
    return first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-first-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-second-dir", type=Path, required=True)
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
    diagnostic = {
        name: checked_diagnostic(
            name, args.diagnostic_first_dir, args.diagnostic_second_dir
        )
        for name in DIAGNOSTIC_WIDTHS
    }
    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer3-compressor-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer3-attention-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    attn_filename = "layer3-attention-output.f32le.bin"
    (output / attn_filename).write_bytes(full["attn_out"])
    hc_payload = full["hc_attn_post"][-TILE_ROWS * WIDTHS["hc_attn_post"] * 4 :]
    hc_filename = "layer3-hc-attn-post-final-tile.f32le.bin"
    (output / hc_filename).write_bytes(hc_payload)
    diagnostic_tensors = []
    for name, width in DIAGNOSTIC_WIDTHS.items():
        payload = diagnostic[name][: width * 4]
        filename = f"layer3-{name.replace('_', '-')}-row0.f32le.bin"
        (output / filename).write_bytes(payload)
        diagnostic_tensors.append(
            {
                "name": f"layer3_{name}_row0",
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

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer3-attention-2048",
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
                "DS4_METAL_GRAPH_DUMP_NAME": (
                    "attn_out,hc_attn_post,kqv_out,kqv_back,attn_low"
                ),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "diagnostic_full_capture_sha256": EXPECTED_DIAGNOSTIC_FULL_SHA256,
            "device_path": (
                "full 2048-row Q projection, dense mixed 2048 raw plus 16 "
                "ratio-128 compressed FlashAttention, inverse compressed RoPE, "
                "legacy grouped/dense Q8_0 output projections, and additive HC post"
            ),
            "indexer_policy": (
                "all 16 compressed rows remain dense because the compressed cache "
                "is below the 512-row sparse-indexer threshold"
            ),
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer3-compressor-2048",
            "storage_note": (
                "The full 32 MiB attention output is retained. Only the exact "
                "final 32-row HC tile is retained; the full 128 MiB HC identity "
                "is pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 3,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [0, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "layer3-dense-mixed-attention",
                "kernel": "kernel_flash_attn_ext_f16_dk512_dv512",
            },
            {
                "name": "layer3-attention-inverse-rope",
                "kernel": "kernel_rope_norm_f32",
            },
            {
                "name": "layer3-attention-output-projections",
                "kernel": "kernel_mul_mm_id_q8_0_f32 + kernel_mul_mm_q8_0_f32",
            },
            {
                "name": "layer3-attention-hc-post",
                "kernel": "kernel_dsv4_hc_expand4",
            },
        ],
        "tensors": diagnostic_tensors + [
            {
                "name": "layer3_attention_output",
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
                "name": "layer3_hc_attn_post_final_tile",
                "hook": "hc_attn_post",
                "role": "output",
                "dtype": "f32",
                "shape": [TILE_ROWS, WIDTHS["hc_attn_post"]],
                "encoding": "little-endian-ieee754-binary32",
                "path": hc_filename,
                "bytes": len(hc_payload),
                "sha256": sha256(hc_payload),
            },
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
