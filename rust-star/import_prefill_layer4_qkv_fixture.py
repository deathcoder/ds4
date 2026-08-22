#!/usr/bin/env python3
"""Import the full-2K native layer-4 HC/QKV boundary."""

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
    "hc_attn_pre": 4_096,
    "attn_norm": 4_096,
    "q_lora": 1_024,
    "q_lora_norm": 1_024,
    "KVraw": 512,
    "KVnorm": 512,
    "Qraw": 32_768,
    "Qcur": 32_768,
    "KVrope": 512,
    "KVcur": 512,
}
EXPECTED_FULL_SHA256 = {
    "hc_attn_pre": "bab5ff4d27a369f36f82e46d7a9cb059738f7bf6a35877c03785b0016258cf93",
    "attn_norm": "6d5215eb619a6a4f919e34fa9486f32e7081a0409c79108ddc3d31c5d57428e8",
    "q_lora": "f19e6447b0911412fb13f80f85de57a8cf8b79ba472f3382be63c154f5387e25",
    "q_lora_norm": "ee9e06b952520e1e601215f4a67b67154fe4bc998636da635f2edc0961516825",
    "KVraw": "49a541ef32cd800c57732ba2a2097856d51f8a8d1181ed88eb0a12e97278acea",
    "KVnorm": "c5cf4c12a010e2c31959fc42d4cf07a694f2659915536b5848d1fb8b41adda39",
    "Qraw": "625d6189a74783414c5ea82c5f8760069b8f71f464e51951328a0e2372ae5fd6",
    "Qcur": "4c82eadf4c69749f6f04f14d3c08a1338677beee50cfadb2fe4d1fc6cf52f821",
    "KVrope": "20d5592195a2e974319b945c444f2f0cceeae5f631ce7a488f5cf9f721aabc41",
    "KVcur": "6c6908f2971df9035a1002d4d48fa3490fd80df9d01119cbca51e2675dd88eae",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    suffix = f"{name}-4_pos0.bin"
    first = (first_dir / f"{first_dir.name}_{suffix}").read_bytes()
    second = (second_dir / f"{second_dir.name}_{suffix}").read_bytes()
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
    parser.add_argument("--qraw-first", type=Path, required=True)
    parser.add_argument("--qraw-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payloads = {
        name: (
            checked_repeated(name, args.first_dir, args.second_dir)
            if name != "Qraw"
            else checked_repeated(name, args.qraw_first, args.qraw_second)
        )
        for name in WIDTHS
    }
    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer3-complete-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer4-qkv-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    filenames = {
        "hc_attn_pre": "layer4-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer4-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer4-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer4-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer4-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer4-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer4-q-raw-final-tile.f32le.bin",
        "Qcur": "layer4-q-current-final-tile.f32le.bin",
        "KVrope": "layer4-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer4-kv-current-final-tile.f32le.bin",
    }
    tensors = []
    for name, width in WIDTHS.items():
        payload = payloads[name][-TILE_ROWS * width * 4 :]
        filename = filenames[name]
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer4_{name.lower()}_final_tile",
                "hook": name,
                "role": "output" if name == "KVcur" else "intermediate",
                "dtype": "f32",
                "shape": [TILE_ROWS, width],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer4-qkv-2048",
        "captured_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 4,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "primary_environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "4",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": (
                    "hc_attn_pre,attn_norm,q_lora,q_lora_norm,"
                    "KVraw,KVnorm,Qcur,KVrope,KVcur"
                ),
            },
            "qraw_environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "4",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "Qraw",
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer3-complete-2048",
            "storage_note": (
                "Exact final tiles are retained; complete 2K identities are "
                "pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 4,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
            "raw_kv_rows": PREFILL_ROWS,
        },
        "operations": [
            {
                "name": "attention-hc-ingress-and-norm",
                "kernel": "kernel_rms_norm_f32_4 plus kernel_mul_mm_f16_f32 plus kernel_dsv4_hc_split_weighted_sum_norm4",
            },
            {"name": "q-lora-and-kv-projections", "kernel": "kernel_mul_mm_q8_0_f32"},
            {"name": "fused-q-kv-learned-norm", "kernel": "kernel_dsv4_qkv_rms_norm_f32"},
            {"name": "q-b-projection", "kernel": "kernel_mul_mm_q8_0_f32"},
            {"name": "q-head-norm-and-rope", "kernel": "kernel_dsv4_head_rms_norm_rope_tail_f32"},
            {"name": "kv-rope", "kernel": "kernel_dsv4_rope_tail_f32"},
            {"name": "kv-finalization", "kernel": "kernel_dsv4_compressor_fp8_f32"},
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
