#!/usr/bin/env python3
"""Import the full-2K native layer-6 HC/QKV boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T17:03:24Z"
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
    "hc_attn_pre": "86f1bb961d20e90d2f14fb22cd38f5a6a7895b014a2992cae70578e1d54d2a65",
    "attn_norm": "1f8a083bda7e966f6d771453c3108bd247856ad693343878c56d09e65bc85f23",
    "q_lora": "23284f0503e8bd818437ce5b87aa0b908482f0cbf1d89083d37eb5101daf2d61",
    "q_lora_norm": "d3f2652576105a19343eb64e3c4508bc22966d9bf20a94684d43ebc409e09739",
    "KVraw": "29821920d2e5b21a35130a9405f2eccfd6f106ceca84e03ba42187d109ff9de7",
    "KVnorm": "294b217777d025062d6b083c37a1c38c03258fc09fc18fb361240820e3314662",
    "Qraw": "577c987efc263a9764216c81799607afe47e7f4bf483a60ceb3c73d809d40ef1",
    "Qcur": "1897d9089814ff3409a64305ae7b423e1209872cc6d570f220e22e2de123b1dd",
    "KVrope": "c6d72c217f69eeda750846c242d6daa723eb9a6494c4a324c81b42c02281fc53",
    "KVcur": "4e03f8eabe018b0556e36724a1195af70a7eb757cc1a4162f8b5881f8b9001e7",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    suffix = f"{name}-6_pos0.bin"
    first = (first_dir / f"{first_dir.name}_{suffix}").read_bytes()
    second = (second_dir / f"{second_dir.name}_{suffix}").read_bytes()
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
        (args.fixtures_root / "prefill-layer5-complete-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer6-qkv-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    filenames = {
        "hc_attn_pre": "layer6-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer6-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer6-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer6-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer6-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer6-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer6-q-raw-final-tile.f32le.bin",
        "Qcur": "layer6-q-current-final-tile.f32le.bin",
        "KVrope": "layer6-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer6-kv-current-final-tile.f32le.bin",
    }
    tensors = []
    for name, width in WIDTHS.items():
        payload = payloads[name][-TILE_ROWS * width * 4 :]
        filename = filenames[name]
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer6_{name.lower()}_final_tile",
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer6-qkv-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "6",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": (
                    "hc_attn_pre,attn_norm,q_lora,q_lora_norm,"
                    "KVraw,KVnorm,Qcur,KVrope,KVcur"
                ),
            },
            "qraw_environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "6",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "Qraw",
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer5-complete-2048",
            "storage_note": (
                "Exact final tiles are retained; complete 2K identities are "
                "pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 6,
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
