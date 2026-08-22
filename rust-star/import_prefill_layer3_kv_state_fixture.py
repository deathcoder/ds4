#!/usr/bin/env python3
"""Import the full-2K native layer-3 Q/KV and raw-state boundary."""

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
    "q_lora_norm": 1_024,
    "KVraw": 512,
    "KVnorm": 512,
    "Qraw": 32_768,
    "Qcur": 32_768,
    "KVrope": 512,
    "KVcur": 512,
}
EXPECTED_FULL_SHA256 = {
    "q_lora_norm": "b5a3e87829a6929bfe3634e1ac0803b14f87e9c5bf8f42409d22a721fb3ef746",
    "KVraw": "6e165f8ec3c704163ae33f222f2f26a080b519f58e91a05e741b095228d500e3",
    "KVnorm": "ff8641fa85260a7aa44f9d2cf90ed582a870fe756579bf6a0cacc44b3874123d",
    "Qraw": "6cf146751bfba8893f8ba898a085bd9cd5c201a504fcace48a9d748d8eef28a8",
    "Qcur": "86d9d66ef5b1747a424b3a3b049ccf3a1b67bf3325883d9ef808d22473a065f8",
    "KVrope": "121bb797c9df1d0fe4106a6cf33798f1972acdf136527ca2ab6d825ac8bcb46f",
    "KVcur": "29f24fbdfed6419a98d6d1001ea9770fde169a7234ddcd01f9632650c355ed3f",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first_payload) != expected_size or len(second_payload) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-first-dir", type=Path, required=True)
    parser.add_argument("--primary-second-dir", type=Path, required=True)
    parser.add_argument("--qraw-first", type=Path, required=True)
    parser.add_argument("--qraw-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payloads = {}
    for name in WIDTHS:
        if name == "Qraw":
            first, second = args.qraw_first, args.qraw_second
        else:
            first = args.primary_first_dir / f"first_{name}-3_pos0.bin"
            second = args.primary_second_dir / f"second_{name}-3_pos0.bin"
        payloads[name] = checked_repeated(name, first, second)

    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer3-ingress-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer3-kv-state-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    filenames = {
        "q_lora_norm": "layer3-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer3-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer3-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer3-q-raw-final-tile.f32le.bin",
        "Qcur": "layer3-q-current-final-tile.f32le.bin",
        "KVrope": "layer3-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer3-kv-current-final-tile.f32le.bin",
    }
    tensors = []
    for name, width in WIDTHS.items():
        payload = payloads[name][-TILE_ROWS * width * 4 :]
        filename = filenames[name]
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": f"layer3_{name.lower()}_final_tile",
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
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer3-kv-state-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "3",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": (
                    "q_lora_norm,KVraw,KVnorm,Qcur,KVrope,KVcur"
                ),
            },
            "qraw_environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "3",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "Qraw",
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer3-ingress-2048",
            "storage_note": (
                "Exact final tiles are retained; complete 2K identities are "
                "pinned by SHA-256."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 3,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [
                PREFILL_ROWS - TILE_ROWS,
                PREFILL_ROWS - 1,
            ],
            "raw_kv_rows": PREFILL_ROWS,
        },
        "operations": [
            {
                "name": "kv-projection",
                "kernel": "kernel_mul_mm_q8_0_f32",
            },
            {
                "name": "fused-q-kv-learned-norm",
                "kernel": "kernel_dsv4_qkv_rms_norm_f32",
            },
            {
                "name": "q-b-projection",
                "kernel": "kernel_mul_mm_q8_0_f32",
            },
            {
                "name": "q-head-norm-and-rope",
                "kernel": "kernel_dsv4_head_rms_norm_rope_tail_f32",
            },
            {
                "name": "kv-rope",
                "kernel": "kernel_dsv4_rope_tail_f32",
            },
            {
                "name": "kv-finalization",
                "kernel": "kernel_dsv4_compressor_fp8_f32",
            },
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
