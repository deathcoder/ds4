#!/usr/bin/env python3
"""Import two repeated diagnostic sparse-indexed-attention captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAYER = 2
TOP_K = 512
RAW_ROWS = 128
ATTENTION_WIDTH = 512
INDEXER_WIDTH = 128

ORACLE = {
    "id": "oracle-v1",
    "repository": "https://github.com/antirez/ds4.git",
    "commit": "b0309611041655f4e45671cfd9c9886aff161406",
    "tree": "20c11af22f90a0bdf25da860da5ef06de4064060",
    "capture_executable_sha256": (
        "70bc7a047a87234e6b2f46f5ba679e48b7eb8001bbf8894602e2daa87f1a66d7"
    ),
}
MODEL = {
    "family": "DeepSeek-V4-Flash-0731",
    "sha256": "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
}

def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated_payload(first: Path, second: Path, captures: dict[str, tuple[str, int]], key: str) -> bytes:
    name, expected_bytes = captures[key]
    a = (first / name).read_bytes()
    b = (second / name).read_bytes()
    if len(a) != expected_bytes:
        raise SystemExit(f"{name} has {len(a)} bytes, expected {expected_bytes}")
    if a != b:
        raise SystemExit(f"fresh-process captures differ: {name}")
    return a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_capture", type=Path)
    parser.add_argument("second_capture", type=Path)
    parser.add_argument(
        "--default-boundary",
        action="store_true",
        help="import the production 1,025-row switch at position 4099",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    position = 4099 if args.default_boundary else 2051
    compressed_rows = 1025 if args.default_boundary else 513
    prefill_tokens = 4096 if args.default_boundary else 2048
    raw_capacity_rows = prefill_tokens + 5
    captured_at = "2026-08-23T10:29:00Z" if args.default_boundary else "2026-08-23T07:44:34Z"
    oracle = dict(ORACLE)
    if args.default_boundary:
        oracle["capture_executable_sha256"] = (
            "4c7e69ab2a20a7c04598f1ec68f26802868210b26d3db9d4ee15113e520ae177"
        )
    captures = {
        "q_lora_norm": (f"capture_q_lora_norm-2_pos{position}.bin", 1024 * 4),
        "attn_norm": (f"capture_attn_norm-2_pos{position}.bin", 4096 * 4),
        "q_current": (f"capture_Qcur-2_pos{position}.bin", 64 * 512 * 4),
        "indexer_q": (f"capture_indexer_q-2_pos{position}.bin", 64 * 128 * 4),
        "indexer_weights": (f"capture_indexer_weights-2_pos{position}.bin", 64 * 4),
        "indexer_scores": (f"capture_indexer_scores-2_pos{position}.bin", compressed_rows * 4),
        "indexer_topk": (f"capture_indexer_topk-2_pos{position}.i32", 512 * 4),
        "raw_cache_full": (f"capture_raw_cache-2_pos{position}.bin", raw_capacity_rows * 512 * 4),
        "attention_comp_cache": (
            f"capture_attention_comp_cache-2_pos{position}.bin",
            compressed_rows * 512 * 4,
        ),
        "indexer_comp_cache": (
            f"capture_indexer_comp_cache-2_pos{position}.bin",
            compressed_rows * 128 * 4,
        ),
        "kqv_out": (f"capture_kqv_out-2_pos{position}.bin", 64 * 512 * 4),
        "kqv_back": (f"capture_kqv_back-2_pos{position}.bin", 64 * 512 * 4),
    }

    output = args.fixtures_root / (
        "sparse-indexed-attention-pos4099-v1"
        if args.default_boundary
        else "sparse-indexed-attention-pos2051-v1"
    )
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    payloads = {
        key: repeated_payload(args.first_capture, args.second_capture, captures, key)
        for key in captures
    }
    raw_row0 = position + 1 - RAW_ROWS
    raw_stride = ATTENTION_WIDTH * 4
    raw_cache = payloads.pop("raw_cache_full")[
        raw_row0 * raw_stride : (raw_row0 + RAW_ROWS) * raw_stride
    ]
    payloads["raw_cache"] = raw_cache

    tensor_specs = {
        "q_lora_norm": ("input", "f32", [1024]),
        "attn_norm": ("input", "f32", [4096]),
        "q_current": ("input", "f32", [64, 512]),
        "raw_cache": ("input", "f32", [128, 512]),
        "attention_comp_cache": ("input", "f32", [compressed_rows, 512]),
        "indexer_comp_cache": ("input", "f32", [compressed_rows, 128]),
        "indexer_q": ("intermediate", "f32", [64, 128]),
        "indexer_weights": ("intermediate", "f32", [64]),
        "indexer_scores": ("intermediate", "f32", [compressed_rows]),
        "indexer_topk": ("intermediate", "i32", [512]),
        "kqv_out": ("output", "f32", [64, 512]),
        "kqv_back": ("output", "f32", [64, 512]),
    }
    tensors = []
    for name, (role, dtype, shape) in tensor_specs.items():
        suffix = "i32le.bin" if dtype == "i32" else "f32le.bin"
        path = output / f"{name.replace('_', '-')}.{suffix}"
        path.write_bytes(payloads[name])
        tensors.append(
            {
                "name": name,
                "role": role,
                "dtype": dtype,
                "shape": shape,
                "encoding": (
                    "little-endian-signed-integer32"
                    if dtype == "i32"
                    else "little-endian-ieee754-binary32"
                ),
                "path": path.name,
                "bytes": len(payloads[name]),
                "sha256": sha256(payloads[name]),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": f"dwarfstar-oracle-v1-layer2-pos{position}-sparse-indexed-attention",
        "captured_at_utc": captured_at,
        "oracle": oracle,
        "model": MODEL,
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefill_tokens": prefill_tokens,
            "decode_steps": 4,
            "captured_position": position,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                **({} if args.default_boundary else {"DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD": "512"}),
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": str(position),
            },
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "decode",
            "layer": LAYER,
            "position": position,
            "compressed_rows": compressed_rows,
            "raw_rows": RAW_ROWS,
            "raw_source_physical_rows": [raw_row0, position],
            "raw_fixture_start": 0,
            "top_k": TOP_K,
        },
        "threshold_semantics": {
            "diagnostic_override": None if args.default_boundary else 512,
            "pinned_oracle_default": 1024,
            "first_default_sparse_compressed_rows": 1025,
            "model_top_k": 512,
            "validates_default_switch_position": args.default_boundary,
        },
        "operations": [
            {"name": "indexer-q-f16-projection", "kernel": "kernel_mul_mv_f16_f32_4"},
            {"name": "indexer-q-rope-tail", "kernel": "kernel_dsv4_rope_tail_f32"},
            {"name": "indexer-q-hadamard-fp4-qat", "kernel": "kernel_dsv4_indexer_hadamard_fp4_f32"},
            {"name": "indexer-weight-f16-projection", "kernel": "kernel_mul_mv_f16_f32_4"},
            {"name": "indexer-direct-score", "kernel": "kernel_dsv4_indexer_score_one_direct"},
            {
                "name": (
                    "exact-descending-top512-block-sort"
                    if args.default_boundary
                    else "exact-descending-top512"
                ),
                "kernel": "kernel_argsort_f32_i32_desc",
            },
            *(
                [{"name": "exact-descending-top512-merge", "kernel": "kernel_argsort_merge_f32_i32_desc"}]
                if args.default_boundary
                else []
            ),
            {"name": "indexed-mixed-attention-split12", "kernel": "kernel_dsv4_indexed_mixed_attention_heads8_split"},
            {"name": "indexed-mixed-attention-reduce", "kernel": "kernel_dsv4_indexed_mixed_attention_heads8_split_reduce"},
            {"name": "attention-inverse-rope-tail", "kernel": "kernel_dsv4_rope_tail_f32"},
        ],
        "claims": {
            "complete_decode": False,
            "output_logits": False,
            "default_threshold_boundary": args.default_boundary,
            "throughput": False,
        },
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
