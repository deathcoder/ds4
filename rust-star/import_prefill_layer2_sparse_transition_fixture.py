#!/usr/bin/env python3
"""Import two repeated 8K prefill layer-2 sparse-transition captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_START = 4096
CHUNK_ROWS = 4096
POSITION = 4099
ROW = POSITION - CHUNK_START
TENSORS = {
    "layer1_attn_cur": ("layer1-attention-current.f32le.bin", 4096),
    "attn_cur": ("attention-current.f32le.bin", 4096),
    "attn_norm": ("attn-norm.f32le.bin", 4096),
    "q_lora_norm": ("q-lora-norm.f32le.bin", 1024),
    "Qcur": ("q-current.f32le.bin", 32768),
    "kqv_out": ("kqv-out.f32le.bin", 32768),
    "kqv_back": ("kqv-back.f32le.bin", 32768),
}
FULL_CAPTURE_SHA256 = {
    "layer1_attn_cur": "c382d7bd6a5ccc865b650622c06872e8223abed7dab91b91a3c5c2659535a111",
    "attn_cur": "652e34575a9453e75e9ce4e82277ae7c8be1ec40e3b8f67ab52e9f1fea86677a",
    "attn_norm": "07e4772bb88b8b21a917927da2f72e32749b3aeac6659703013cfcc6885519a9",
    "q_lora_norm": "55a3cd95d0152c194ba6d654c3810dd15b56b83291022eebe33963ac1d3acad9",
    "Qcur": "85cf0e0ea88d6493e14d999b068d350022d2f955d2e13588271b5c8d75c1116b",
    "kqv_out": "c28683b966717c85a3a88ecc92c97f6cc5cd906cbfba855e0545c1ea1d8c462c",
    "kqv_back": "372140699ec97a8734cdf14572d88caf62063a3098ba1da43875190365816eba",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, name: str) -> Path:
    layer = 1 if name == "layer1_attn_cur" else 2
    capture_name = "hc_attn_pre" if name in {"layer1_attn_cur", "attn_cur"} else name
    return root / f"transition_{capture_name}-{layer}_pos4096.bin"


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
    output = args.fixtures_root / "prefill-layer2-sparse-transition-pos4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    payloads: dict[str, bytes] = {}
    full_hashes: dict[str, str] = {}
    for name, (_, width) in TENSORS.items():
        first = capture_path(args.first_capture, name).read_bytes()
        second = capture_path(args.second_capture, name).read_bytes()
        # The upstream hc_attn_pre hook snapshots batch_attn_cur, the collapsed
        # n_embd-wide attention input, rather than the 4*n_embd HC state.
        expected_bytes = CHUNK_ROWS * width * 4
        if len(first) != expected_bytes:
            raise SystemExit(
                f"{name} has {len(first)} bytes, expected {expected_bytes}"
            )
        if first != second:
            raise SystemExit(f"fresh-process captures differ: {name}")
        full_hashes[name] = sha256(first)
        if full_hashes[name] != FULL_CAPTURE_SHA256[name]:
            raise SystemExit(f"unexpected full-capture SHA-256 for {name}")
        row_bytes = width * 4
        payloads[name] = first[ROW * row_bytes : (ROW + 1) * row_bytes]

    output.mkdir()
    tensors = []
    for name, (filename, width) in TENSORS.items():
        payload = payloads[name]
        path = output / filename
        path.write_bytes(payload)
        tensors.append(
            {
                "name": name,
                "role": (
                    "input"
                    if name in {"attn_cur", "layer1_attn_cur"}
                    else "output" if name in {"kqv_out", "kqv_back"} else "intermediate"
                ),
                "dtype": "f32",
                "shape": [width],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer2-pos4099-sparse-transition",
        "captured_at_utc": "2026-08-30T16:05:21Z",
        "oracle": {
            "id": "oracle-v3",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "d35fb12d01d500b9cefcef24092c295687ceaf7e",
            "tree": "617415ee9f8ea7dc176d63dada1d5a7582063824",
            "capture_executable_sha256": "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3",
        },
        "model": {
            "family": "DeepSeek-V4-Flash-0731",
            "sha256": "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
        },
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefill_tokens": 8192,
            "prefill_chunk": 4096,
            "chunk_start": CHUNK_START,
            "captured_position": POSITION,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "csv_sha256": [
                "24c3963f341f5dff91718bb2f75de769cf7cc9f0704f079147009025d37f28a3",
                "23174389d5449e4187c4b49e4aaaeb8ea1e3b111a38b7a20aaaefbe1c2a76356",
                "f26c17d3950b9b91a21ca8d1ca74cbe8d5e3e32c7504a27f804b3ad02c43487e",
                "23e268272504ba5a8df446113ec31c1c0dbe844e3f5235fe30f600493bec7473",
                "0d682425e441e798f313925bcce6517f36f4d01fcc1fa465010fbee7cfbf09ac",
                "68fdb2c85fb2bf74989ef013cee6ae349c0a99437e6e7aa6af37cef79e1902aa",
            ],
            "full_batch_sha256": full_hashes,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": [
                    "attn_norm,q_lora_norm,Qcur,kqv_out,kqv_back",
                    "hc_attn_pre",
                    "layer1:hc_attn_pre",
                ],
            },
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": POSITION,
            "chunk_row": ROW,
            "compressed_rows": 1025,
            "raw_rows": 128,
            "top_k": 512,
        },
        "claims": {
            "native_batched_q_projection": True,
            "production_sparse_attention": True,
            "complete_layer": False,
            "output_logits": False,
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
