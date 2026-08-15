#!/usr/bin/env python3
"""Import the final legacy Metal tile of layer-0 native batched Q/KV setup."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T15:29:36Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
WIDTHS = {
    "attn_norm": 4_096,
    "q_lora": 1_024,
    "q_lora_norm": 1_024,
    "KVraw": 512,
    "KVnorm": 512,
    "Qraw": 32_768,
    "Qcur": 32_768,
}
EXPECTED_FULL_SHA256 = {
    "attn_norm": "5a4c83daeb603f714146b93438a38672515e45c55f0776527e76c86db99476b4",
    "q_lora": "ce24ef63d0604b73769c7fd1141d70a25e85ae88b6c2e6770d649fdf40b0b62e",
    "q_lora_norm": "58b1d0862677620d157e4af057d7a8dc34eb8d0dd244598f1f54e830b8ac60f0",
    "KVraw": "faeb690f3e227c98510073958f829594e2c53e1c21d706fd52988cae014a7d18",
    "KVnorm": "4472b2d89a29c8181f87b7d224570838f5937afdadacc2514df82e775c0821e9",
    "Qraw": "93f5f53993eacd8f6fee82ef5ceabc36f9bab8fbad321b09394b6d260f1819d2",
    "Qcur": "e48ac9f2fdecaf532d25f93fb5fe90edbb6f85054cffe7556697d5fd01883ea3",
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


def final_tile(payload: bytes, width: int) -> bytes:
    return payload[-TILE_ROWS * width * 4 :]


def tensor(name: str, hook: str, role: str, path: str, payload: bytes) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": "f32",
        "shape": [TILE_ROWS, WIDTHS[hook]],
        "encoding": "little-endian-ieee754-binary32",
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def q8_dispatch(name: str, weight: str, input_width: int, output_width: int) -> dict:
    return {
        "name": name,
        "kernel": "kernel_mul_mm_q8_0_f32",
        "weights": [weight],
        "dispatch": {
            "input_elements": input_width,
            "output_elements": output_width,
            "rows": TILE_ROWS,
            "threads_per_threadgroup": [128, 1, 1],
            "threadgroups": [1, output_width // 64, 1],
            "threadgroup_memory_bytes": 6_144,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in WIDTHS:
        parser.add_argument(f"--{name.lower().replace('_', '-')}-first", type=Path, required=True)
        parser.add_argument(f"--{name.lower().replace('_', '-')}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payloads = {}
    for name in WIDTHS:
        key = name.lower()
        payloads[name] = checked_repeated(
            name,
            getattr(args, f"{key}_first"),
            getattr(args, f"{key}_second"),
        )

    tiles = {name: final_tile(payloads[name], WIDTHS[name]) for name in WIDTHS}
    template = json.loads(
        (args.fixtures_root / "prefill-q8-boundary-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-qkv-boundary-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    filenames = {
        "attn_norm": "attn-norm-final-tile.f32le.bin",
        "q_lora": "q-lora-final-tile.f32le.bin",
        "q_lora_norm": "q-lora-norm-final-tile.f32le.bin",
        "KVraw": "kv-raw-final-tile.f32le.bin",
        "KVnorm": "kv-norm-final-tile.f32le.bin",
        "Qraw": "q-raw-final-tile.f32le.bin",
        "Qcur": "q-current-final-tile.f32le.bin",
    }
    for name, filename in filenames.items():
        (output / filename).write_bytes(tiles[name])

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-qkv-boundary-2048",
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "q_lora,q_lora_norm,KVraw,KVnorm,Qcur",
            },
            "q_raw_control_environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "0",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "Qraw",
            },
            "device_path": (
                "legacy Metal Q8_0 batch matmul plus fused Q/KV RMS norm and "
                "head RMSNorm/RoPE; Metal 4 TensorOps disabled on M1 Ultra"
            ),
            "q_raw_control": (
                "Qraw capture requests the standalone Q-B path; the pinned M1 build's "
                "fused Q-B/F16 entry point is unavailable and falls back to the same path"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 0,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            q8_dispatch("attn-q-a-prefill", "blk.0.attn_q_a.weight", 4096, 1024),
            q8_dispatch("attn-kv-a-prefill", "blk.0.attn_kv.weight", 4096, 512),
            {
                "name": "fused-q-lora-kv-rmsnorm-prefill",
                "kernel": "kernel_dsv4_qkv_rms_norm_f32_4",
                "weights": [
                    "blk.0.attn_q_a_norm.weight",
                    "blk.0.attn_kv_a_norm.weight",
                ],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [TILE_ROWS, 2, 1],
                    "threadgroup_memory_bytes": 128,
                },
            },
            q8_dispatch("attn-q-b-prefill", "blk.0.attn_q_b.weight", 1024, 32768),
            {
                "name": "head-rmsnorm-rope-prefill",
                "kernel": "kernel_dsv4_head_rms_norm_rope_tail_f32",
                "weights": [],
                "dispatch": {
                    "rows": TILE_ROWS,
                    "heads": 64,
                    "head_dim": 512,
                    "rotary_dim": 64,
                    "position_start": PREFILL_ROWS - TILE_ROWS,
                    "threads_per_threadgroup": [256, 1, 1],
                    "threadgroups": [64, TILE_ROWS, 1],
                    "threadgroup_memory_bytes": 128,
                },
            },
        ],
        "tensors": [
            tensor("attn_norm_final_tile", "attn_norm", "input", filenames["attn_norm"], tiles["attn_norm"]),
            tensor("q_lora_final_tile", "q_lora", "intermediate", filenames["q_lora"], tiles["q_lora"]),
            tensor("kv_raw_final_tile", "KVraw", "intermediate", filenames["KVraw"], tiles["KVraw"]),
            tensor("q_lora_norm_final_tile", "q_lora_norm", "intermediate", filenames["q_lora_norm"], tiles["q_lora_norm"]),
            tensor("kv_norm_final_tile", "KVnorm", "intermediate", filenames["KVnorm"], tiles["KVnorm"]),
            tensor("q_raw_final_tile", "Qraw", "intermediate", filenames["Qraw"], tiles["Qraw"]),
            tensor("q_current_final_tile", "Qcur", "output", filenames["Qcur"], tiles["Qcur"]),
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
