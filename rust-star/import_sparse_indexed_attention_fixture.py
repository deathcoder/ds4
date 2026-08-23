#!/usr/bin/env python3
"""Import two repeated diagnostic sparse-indexed-attention captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAYER = 2
POSITION = 2051
COMPRESSED_ROWS = 513
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

CAPTURES = {
    "q_lora_norm": ("capture_q_lora_norm-2_pos2051.bin", 1024 * 4),
    "attn_norm": ("capture_attn_norm-2_pos2051.bin", 4096 * 4),
    "q_current": ("capture_Qcur-2_pos2051.bin", 64 * 512 * 4),
    "indexer_q": ("capture_indexer_q-2_pos2051.bin", 64 * 128 * 4),
    "indexer_weights": ("capture_indexer_weights-2_pos2051.bin", 64 * 4),
    "indexer_scores": ("capture_indexer_scores-2_pos2051.bin", 513 * 4),
    "indexer_topk": ("capture_indexer_topk-2_pos2051.i32", 512 * 4),
    "raw_cache_full": ("capture_raw_cache-2_pos2051.bin", 2053 * 512 * 4),
    "attention_comp_cache": ("capture_attention_comp_cache-2_pos2051.bin", 513 * 512 * 4),
    "indexer_comp_cache": ("capture_indexer_comp_cache-2_pos2051.bin", 513 * 128 * 4),
    "kqv_out": ("capture_kqv_out-2_pos2051.bin", 64 * 512 * 4),
    "kqv_back": ("capture_kqv_back-2_pos2051.bin", 64 * 512 * 4),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated_payload(first: Path, second: Path, key: str) -> bytes:
    name, expected_bytes = CAPTURES[key]
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
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    output = args.fixtures_root / "sparse-indexed-attention-pos2051-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    payloads = {
        key: repeated_payload(args.first_capture, args.second_capture, key)
        for key in CAPTURES
    }
    raw_row0 = POSITION + 1 - RAW_ROWS
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
        "attention_comp_cache": ("input", "f32", [513, 512]),
        "indexer_comp_cache": ("input", "f32", [513, 128]),
        "indexer_q": ("intermediate", "f32", [64, 128]),
        "indexer_weights": ("intermediate", "f32", [64]),
        "indexer_scores": ("intermediate", "f32", [513]),
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
        "fixture_id": "dwarfstar-oracle-v1-layer2-pos2051-sparse-indexed-attention",
        "captured_at_utc": "2026-08-23T07:44:34Z",
        "oracle": ORACLE,
        "model": MODEL,
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefill_tokens": 2048,
            "decode_steps": 4,
            "captured_position": POSITION,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD": "512",
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "2051",
            },
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "decode",
            "layer": LAYER,
            "position": POSITION,
            "compressed_rows": COMPRESSED_ROWS,
            "raw_rows": RAW_ROWS,
            "raw_source_physical_rows": [raw_row0, POSITION],
            "raw_fixture_start": 0,
            "top_k": TOP_K,
        },
        "threshold_semantics": {
            "diagnostic_override": 512,
            "pinned_oracle_default": 1024,
            "first_default_sparse_compressed_rows": 1025,
            "model_top_k": 512,
            "validates_default_switch_position": False,
        },
        "operations": [
            {"name": "indexer-q-f16-projection", "kernel": "kernel_mul_mv_f16_f32_4"},
            {"name": "indexer-q-rope-tail", "kernel": "kernel_dsv4_rope_tail_f32"},
            {"name": "indexer-q-hadamard-fp4-qat", "kernel": "kernel_dsv4_indexer_hadamard_fp4_f32"},
            {"name": "indexer-weight-f16-projection", "kernel": "kernel_mul_mv_f16_f32_4"},
            {"name": "indexer-direct-score", "kernel": "kernel_dsv4_indexer_score_one_direct"},
            {"name": "exact-descending-top512", "kernel": "kernel_argsort_f32_i32_desc"},
            {"name": "indexed-mixed-attention-split12", "kernel": "kernel_dsv4_indexed_mixed_attention_heads8_split"},
            {"name": "indexed-mixed-attention-reduce", "kernel": "kernel_dsv4_indexed_mixed_attention_heads8_split_reduce"},
            {"name": "attention-inverse-rope-tail", "kernel": "kernel_dsv4_rope_tail_f32"},
        ],
        "claims": {
            "complete_decode": False,
            "output_logits": False,
            "default_threshold_boundary": False,
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
