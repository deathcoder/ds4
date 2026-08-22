#!/usr/bin/env python3
"""Import the full-2K native layer-5 HC/QKV boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T16:53:00Z"
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
    "hc_attn_pre": "1eb77abe3ed2dd5f7ddc04759423a5f3acf7f69b6a28fcc0261b628356635671",
    "attn_norm": "f851cb17d19e8cbb2243e08522fba55ce598753bf0134e18adbb287c0ea457a7",
    "q_lora": "85ebf3f69e0ecc6b8a8ad098e8edf2100600c6415103c4c59a1d92e7c2d519eb",
    "q_lora_norm": "d19c918abbd543d2498128feec46670950fe0463ca468913fe3f4c7cf9bd6328",
    "KVraw": "fb03594ec1728956dde7cc17397ba94b347b56e3a64d7f3fa383dab6c4aa179e",
    "KVnorm": "d300267636f703b6c0d7bf89df1782f3b410d433c3a6935e9104b0c65f7ea90f",
    "Qraw": "ee643bc3a765aa779bf822843a71bceef1774347ad7f69597440e4ecbb9c795e",
    "Qcur": "f8f0c69cfabaa35d9611988c5d59d2fe0efe51309be90f2a7c03dcb5a973f6b7",
    "KVrope": "e4a48e94b98a1d7e5090a13af1688e0898c27656014b113cf3f7a867c7713885",
    "KVcur": "40db6d88a337b74f06448c1e8760c8b21bdeee8f97e0f54c530f6733f0ee2323",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    suffix = f"{name}-5_pos0.bin"
    first = (first_dir / f"{first_dir.name}_{suffix}").read_bytes()
    second = (second_dir / f"{second_dir.name}_{suffix}").read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"layer-5 {name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-5 {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"layer-5 {name} capture identity changed")
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
        (args.fixtures_root / "prefill-layer4-complete-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer5-qkv-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    filenames = {
        "hc_attn_pre": "layer5-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer5-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer5-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer5-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer5-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer5-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer5-q-raw-final-tile.f32le.bin",
        "Qcur": "layer5-q-current-final-tile.f32le.bin",
        "KVrope": "layer5-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer5-kv-current-final-tile.f32le.bin",
    }
    tensors = []
    for name, width in WIDTHS.items():
        payload = payloads[name][-TILE_ROWS * width * 4 :]
        filename = filenames[name]
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer5_{name.lower()}_final_tile",
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer5-qkv-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "5",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": (
                    "hc_attn_pre,attn_norm,q_lora,q_lora_norm,"
                    "KVraw,KVnorm,Qcur,KVrope,KVcur"
                ),
            },
            "qraw_environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "5",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "Qraw",
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer4-complete-2048",
            "storage_note": (
                "Exact final tiles are retained; complete 2K identities are "
                "pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 5,
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
