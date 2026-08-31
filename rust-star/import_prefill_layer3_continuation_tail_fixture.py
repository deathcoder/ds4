#!/usr/bin/env python3
"""Import repeated layer-3 compressor, attention, and FFN continuation captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_START = 4_096
CHUNK_ROWS = 4_096
TILE_ROWS = 32
STATE_ROWS = 128
CAPTURED_AT_UTC = "2026-08-31T15:57:00Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3"
)
STATE_CAPTURE_EXECUTABLE_SHA256 = (
    "e7f9705861ffc9a3726e1a8dbf3208cb9afd396ddb42ac6e430fdcfc18216dbb"
)
STATE_CAPTURE_DIAGNOSTIC_PATCH_SHA256 = (
    "c7541dda2520082af193dc5c2374d81fa8bd1cd9ac0e257b143ba961eae77de9"
)
CACHE_CAPTURE_EXECUTABLE_SHA256 = (
    "d52cafa6a0292054fecec5384586dd4acc50667694711630bec9f09e74b1b685"
)
CACHE_CAPTURE_DIAGNOSTIC_PATCH_SHA256 = (
    "e1dcf7e1815e6e7c0b72f7888fc66d6eb14636d71794e63f966926edc7f3229a"
)
TENSORS = {
    "mixed_raw_cache": ("history-raw-kv.f32le.bin", 512, "f32", "history"),
    "mixed_attn_comp_cache": (
        "history-compressed-kv.f32le.bin",
        512,
        "f32",
        "compressed",
    ),
    "attn_state_kv": ("attention-state-kv.f32le.bin", 512, "f32", "state"),
    "attn_state_score": (
        "attention-state-score-bits.i32le.bin",
        512,
        "i32",
        "state",
    ),
    "kqv_out": ("kqv-out-first-tile.f32le.bin", 32_768, "f32", "tile"),
    "kqv_back": ("kqv-back-first-tile.f32le.bin", 32_768, "f32", "tile"),
    "attn_low": ("attention-low-first-tile.f32le.bin", 8_192, "f32", "tile"),
    "attn_out": ("attention-output-first-tile.f32le.bin", 4_096, "f32", "tile"),
    "hc_attn_post": (
        "attention-hc-post-first-tile.f32le.bin",
        16_384,
        "f32",
        "tile",
    ),
    "hc_ffn_pre": ("ffn-current-first-tile.f32le.bin", 4_096, "f32", "tile"),
    "ffn_norm": ("ffn-norm-first-tile.f32le.bin", 4_096, "f32", "tile"),
    "ffn_moe_logits": ("router-logits-first-tile.f32le.bin", 256, "f32", "tile"),
    "ffn_moe_probs": ("router-probs-first-tile.f32le.bin", 256, "f32", "tile"),
    "ffn_moe_topk": ("router-selected-first-tile.i32le.bin", 6, "i32", "tile"),
    "ffn_moe_weights_scaled": (
        "router-weights-first-tile.f32le.bin",
        6,
        "f32",
        "tile",
    ),
    "ffn_moe_weighted_swiglu": (
        "routed-mid-first-tile.f32le.bin",
        12_288,
        "f32",
        "tile",
    ),
    "ffn_moe_out": ("routed-output-first-tile.f32le.bin", 4_096, "f32", "tile"),
    "ffn_shexp": ("shared-output-first-tile.f32le.bin", 4_096, "f32", "tile"),
    "hc_ffn_post": ("ffn-hc-post-first-tile.f32le.bin", 16_384, "f32", "tile"),
}
FULL_CAPTURE_SHA256 = {
    "mixed_raw_cache": "a36c19e0075991a1ca94be106d97234e00222fe7a7214eff0a620430c5b8d8e1",
    "mixed_attn_comp_cache": "e9b5beb684155bc7db780e0bb69bb991e562e41f9ed0aa27a6192df36f24756f",
    "attn_state_kv": "783a0837dc8aeaa758e4ae360e49b37f492730c71882ae0d522043faba66af0c",
    "attn_state_score": "ad9f410768deff98da7e03c1c33f6e8a033dcbf0a03c7dca7eadc2977899a1ac",
    "kqv_out": "fa12ab09e739b1046832b9764054257c3250c886a3a4eee220eadbbd77440b4c",
    "kqv_back": "6a24bdadcd430d672f887429ecb5d08d9c1da57041cbb301600a88bf835731a9",
    "attn_low": "99682fc59521c110ec252fcb05cd7827925215b397dd1aacc2f323b8bdd9e255",
    "attn_out": "ed58a1e8717b8a33f2ddfdaa50d8b24f0893908bad109a2c68e88394e75cf2f6",
    "hc_attn_post": "3eef9bc395ad66eb001877d8d9b96ccc781e64d4d4a721c74c88c23aabf3cb1b",
    "hc_ffn_pre": "ee3f776cd0bd3a925131e868920d28d81ca05e00640bac7cab66b64d39251f04",
    "ffn_norm": "2eace7a76c541b58a14ef279a0888c261391afc0271077752526cceaae949c35",
    "ffn_moe_logits": "ffcb80d47b4236c34756039d78bc158c0a1661e07a3c047b65881dabec6286d3",
    "ffn_moe_probs": "fcbea11ca6a2ecd3d3e8a8b115bf4478c4ce92bcc3182d035313f12529032599",
    "ffn_moe_topk": "3bdc05f1b245a4a9ec8fd95db63ebd79dd821c2943ebb2e4a6ac834ee746db31",
    "ffn_moe_weights_scaled": "649f989cb1c06d885684dc47e917bc64b695cb746cfbc2dbc07d6d28b454beaf",
    "ffn_moe_weighted_swiglu": "69de08ea095a395cb4094b80c7061d2926592a5f36923c8eadd13dbc66382e32",
    "ffn_moe_out": "36d8075ce85507668b9be864146dbf7387669b44a7036523bd118e37c5f1763a",
    "ffn_shexp": "4fcc70a8e06babad8080002734ef679a038b0d8ade3fb4c96b1d6291838f068a",
    "hc_ffn_post": "431145398039eeb84dbe0ec9eb009eac980ac09ccd427d8398e86ef5b1c73159",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return root / f"oracle_{name}-3_pos4096.{suffix}"


def checked_repeated(
    name: str, first: Path, second: Path, width: int, extent: str
) -> tuple[bytes, int]:
    first_payload = capture_path(first, name).read_bytes()
    second_payload = capture_path(second, name).read_bytes()
    source_rows = {
        "state": STATE_ROWS,
        "history": 4_129,
        "compressed": 32,
        "tile": TILE_ROWS,
    }[extent]
    expected_bytes = source_rows * width * 4
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != FULL_CAPTURE_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    retained_rows = {
        "state": STATE_ROWS,
        "history": 128,
        "compressed": 32,
        "tile": TILE_ROWS,
    }[extent]
    retained_start = 3_968 if extent == "history" else 0
    start = retained_start * width * 4
    return first_payload[start : start + retained_rows * width * 4], retained_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--cache-first", type=Path, required=True)
    parser.add_argument("--cache-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    output = args.fixtures_root / "prefill-layer3-continuation-tail-4096-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, (filename, width, dtype, extent) in TENSORS.items():
        first = args.cache_first if extent in {"history", "compressed"} else args.first
        second = (
            args.cache_second if extent in {"history", "compressed"} else args.second
        )
        payload, retained_rows = checked_repeated(
            name, first, second, width, extent
        )
        (output / filename).write_bytes(payload)
        tensors.append(
            {
                "name": name,
                "hook": name,
                "role": (
                    "input"
                    if extent in {"history", "compressed"}
                    else "output" if name == "hc_ffn_post" else "intermediate"
                ),
                "dtype": dtype,
                "shape": [retained_rows, width],
                "encoding": (
                    "little-endian-signed-integer32"
                    if dtype == "i32"
                    else "little-endian-ieee754-binary32"
                ),
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer3-continuation-tail-4096",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": {
            "id": "oracle-v3",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "d35fb12d01d500b9cefcef24092c295687ceaf7e",
            "tree": "617415ee9f8ea7dc176d63dada1d5a7582063824",
            "capture_executable_sha256": CAPTURE_EXECUTABLE_SHA256,
            "state_capture_executable_sha256": STATE_CAPTURE_EXECUTABLE_SHA256,
            "state_capture_diagnostic_patch_sha256": (
                STATE_CAPTURE_DIAGNOSTIC_PATCH_SHA256
            ),
            "cache_capture_executable_sha256": CACHE_CAPTURE_EXECUTABLE_SHA256,
            "cache_capture_diagnostic_patch_sha256": (
                CACHE_CAPTURE_DIAGNOSTIC_PATCH_SHA256
            ),
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
            "prefill_tokens": CHUNK_START + TILE_ROWS,
            "prefill_chunk": CHUNK_ROWS,
            "continuation_batch": TILE_ROWS,
            "chunk_start": CHUNK_START,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "csv_sha256": [
                "7c802828300bd71e452ad582d0c4b80db3e73c6eb29e8984bc6bfb0a0e976283",
                "9f18e5983909b5130de9df6039dfc41d7b73ef5bd7bf19798235677cfa29ebf0",
            ],
            "cache_csv_sha256": [
                "841ba7d32803c9f052ae8ed0d297115cd371a3424b28b67a9ed57c275dbecf1e",
                "efb0bd52a1964fceeca61135383705b7e58b265a220709822fd5a4fbe344af24",
            ],
            "full_batch_sha256": FULL_CAPTURE_SHA256,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "3",
                "DS4_METAL_GRAPH_DUMP_POS": "4096",
                "DS4_METAL_GRAPH_DUMP_NAME": list(TENSORS),
            },
            "input_fixture": "dwarfstar-oracle-v3-prefill-layer3-continuation-ingress-4096",
            "storage_note": (
                "The complete 128-row recurrent state, exact 32-row continuation "
                "batch, final 128 raw history rows, and 32 compressed history "
                "rows are retained from repeated 4128-token runs. Diagnostic-only "
                "hook extensions expose existing post-batch state and caches "
                "without changing model execution; their diffs and executables "
                "are pinned above."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 3,
            "position": CHUNK_START + TILE_ROWS - 1,
            "captured_position_range": [CHUNK_START, CHUNK_START + TILE_ROWS - 1],
            "tile_rows": TILE_ROWS,
        },
        "operations": [
            {
                "name": "ratio128-state-update",
                "kernel": "paired F16 projections plus recurrent store",
            },
            {
                "name": "dense-mixed-attention",
                "kernel": "FlashAttention over raw and ratio-128 compressed KV",
            },
            {"name": "inverse-rope", "kernel": "kernel_dsv4_rope_tail_f32"},
            {
                "name": "attention-output-projection",
                "kernel": "grouped Q8_0 low plus Q8_0 output",
            },
            {"name": "attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
            {"name": "biased-topk-router", "kernel": "batch softplus/sqrt/bias/top-k/normalize"},
            {"name": "routed-experts", "kernel": "IQ2_XXS pair SwiGLU plus Q2_K down"},
            {"name": "shared-expert", "kernel": "Q8_0 gate/up/down plus flat SwiGLU"},
            {"name": "ffn-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
        ],
        "claims": {
            "live_layer3_qkv_input": True,
            "ratio128_state_advanced_without_emission": True,
            "oracle_seeded_attention_history": True,
            "unseeded_attention_history": False,
            "complete_layer3_tile_from_pinned_attention_history": True,
            "complete_layer3_tile_without_seeded_attention_history": False,
            "complete_layer3": False,
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
