#!/usr/bin/env python3
"""Import repeated production-geometry layer-2/3 continuation tile-2 captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHUNK_START = 4_096
CHUNK_ROWS = 4_096
TILE_START = 4_128
TILE_ROWS = 32
TILE_OFFSET = TILE_START - CHUNK_START
CAPTURED_AT_UTC = "2026-08-31T18:25:13Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "d52cafa6a0292054fecec5384586dd4acc50667694711630bec9f09e74b1b685"
)
CAPTURE_PATCH_SHA256 = (
    "e1dcf7e1815e6e7c0b72f7888fc66d6eb14636d71794e63f966926edc7f3229a"
)

LAYER2 = {
    "indexer_scores": (2_048, "i32", "layer2-indexer-scores-tile2-bits.i32le.bin"),
    "indexer_topk": (512, "i32", "layer2-indexer-topk-tile2.i32le.bin"),
    "kqv_back": (32_768, "f32", "layer2-kqv-back-tile2.f32le.bin"),
    "attn_low": (8_192, "f32", "layer2-attention-low-tile2.f32le.bin"),
    "attn_out": (4_096, "f32", "layer2-attention-output-tile2.f32le.bin"),
    "hc_attn_post": (16_384, "f32", "layer2-attention-hc-post-tile2.f32le.bin"),
    "hc_ffn_pre": (4_096, "f32", "layer2-ffn-current-tile2.f32le.bin"),
    "ffn_norm": (4_096, "f32", "layer2-ffn-norm-tile2.f32le.bin"),
    "ffn_moe_logits": (256, "f32", "layer2-router-logits-tile2.f32le.bin"),
    "ffn_moe_probs": (256, "f32", "layer2-router-probs-tile2.f32le.bin"),
    "ffn_moe_topk": (6, "i32", "layer2-router-selected-tile2.i32le.bin"),
    "ffn_moe_weights_scaled": (6, "f32", "layer2-router-weights-tile2.f32le.bin"),
    "ffn_moe_weighted_swiglu": (12_288, "f32", "layer2-routed-mid-tile2.f32le.bin"),
    "ffn_moe_out": (4_096, "f32", "layer2-routed-output-tile2.f32le.bin"),
    "ffn_shexp": (4_096, "f32", "layer2-shared-output-tile2.f32le.bin"),
    "hc_ffn_post": (16_384, "f32", "layer2-ffn-hc-post-tile2.f32le.bin"),
}

LAYER2_FULL_SHA256 = {
    "indexer_scores": "f8e44be5ac093a6257fc9dafb08f04e9555560dcbda17e083b819adf111f1808",
    "indexer_topk": "8449e42c7e4dd74be4f23becb3edbc7151886e85e5d5a0492ebdb249acfcc60f",
    "kqv_back": "372140699ec97a8734cdf14572d88caf62063a3098ba1da43875190365816eba",
    "attn_low": "1c25067bc70440c748b16bb88d8ab194e68a2c39568c3899a9f2048c9a363035",
    "attn_out": "bba8a9e4d5d83f89896b4c4686c4a7d08f6c228a12b88e2f1160cc41fea2a327",
    "hc_attn_post": "5afba314c39d8ab0cc1ac8a50f49b519c96711f0ec7558db0b958839238b65ef",
    "hc_ffn_pre": "a506d06e253ee2f321c86a23bcca4d2f8818dca2aafbcfac1625a99e6ebb0a6a",
    "ffn_norm": "3a1cedca7e24ba6d12c6c5430fb14b90bf2f9a48546686410a72c6940f61881b",
    "ffn_moe_logits": "a98ee193bfa0c23447b2cef80674d5eccc4f8146433d74180e1ee324835ede44",
    "ffn_moe_probs": "230b22291296e20864171313e02ab2d937d44224e665289d67a96f5675b6b3a6",
    "ffn_moe_topk": "e701f078ceafb71ed4eb381bce1cec5ad4f4688951e22f5e945c0326f390a732",
    "ffn_moe_weights_scaled": "ca169a1e63d760999749a387f47f10d6b263cf0e9be655edcdbe2c6642d8e784",
    "ffn_moe_weighted_swiglu": "b839b0e1bb2caccd13014ec2ca956840d8a3e658f97cabe593aa68e80cc5bb38",
    "ffn_moe_out": "a473ff5f7810bca629064de4ecc113844f0f88f52b9caeee434791abf1f51bb3",
    "ffn_shexp": "eb0bd068f09ef28fdbc7ec7029624f463abb1d89bba183653b080a290f7a894f",
    "hc_ffn_post": "bea6c65976d9f54a425ad36f4ed166571ddb2425d505688d4e2a40ec592e7567",
}

LAYER3 = {
    "hc_attn_pre": (4_096, "layer3-hc-attn-pre-tile2.f32le.bin"),
    "attn_norm": (4_096, "layer3-attn-norm-tile2.f32le.bin"),
    "q_lora": (1_024, "layer3-q-lora-tile2.f32le.bin"),
    "q_lora_norm": (1_024, "layer3-q-lora-norm-tile2.f32le.bin"),
    "KVraw": (512, "layer3-kv-raw-tile2.f32le.bin"),
    "KVnorm": (512, "layer3-kv-norm-tile2.f32le.bin"),
    "Qcur": (32_768, "layer3-q-current-tile2.f32le.bin"),
    "KVrope": (512, "layer3-kv-rope-tile2.f32le.bin"),
    "KVcur": (512, "layer3-kv-current-tile2.f32le.bin"),
    "kqv_out": (32_768, "layer3-kqv-out-tile2.f32le.bin"),
    "kqv_back": (32_768, "layer3-kqv-back-tile2.f32le.bin"),
    "attn_low": (8_192, "layer3-attention-low-tile2.f32le.bin"),
    "attn_out": (4_096, "layer3-attention-output-tile2.f32le.bin"),
    "hc_attn_post": (16_384, "layer3-attention-hc-post-tile2.f32le.bin"),
    "hc_ffn_pre": (4_096, "layer3-ffn-current-tile2.f32le.bin"),
    "ffn_norm": (4_096, "layer3-ffn-norm-tile2.f32le.bin"),
    "ffn_moe_logits": (256, "layer3-router-logits-tile2.f32le.bin"),
    "ffn_moe_probs": (256, "layer3-router-probs-tile2.f32le.bin"),
    "ffn_moe_topk": (6, "layer3-router-selected-tile2.i32le.bin"),
    "ffn_moe_weights_scaled": (6, "layer3-router-weights-tile2.f32le.bin"),
    "ffn_moe_weighted_swiglu": (12_288, "layer3-routed-mid-tile2.f32le.bin"),
    "ffn_moe_out": (4_096, "layer3-routed-output-tile2.f32le.bin"),
    "ffn_shexp": (4_096, "layer3-shared-output-tile2.f32le.bin"),
    "hc_ffn_post": (16_384, "layer3-ffn-hc-post-tile2.f32le.bin"),
}

LAYER3_FULL_SHA256 = {
    "hc_attn_pre": "b84a97eb150f64251f7d8896101afcfd2c717e87d258904de49acafae2aeacb3",
    "attn_norm": "ee201298cbf2acbfc72d56865905bfa94d8a4e4c185b9f76514ddd3e61f21af0",
    "q_lora": "d5034d3a536b4df6a5aa464ed810fb2dd656fa21ef853e5b9e6cb3ff33fbc99d",
    "q_lora_norm": "54a2aac900690016eb24e3ca3c64015f6533aa2cd305c0bc17ad950dddd88607",
    "KVraw": "ba8e31b3c68a68b59ff97407b82d21f2ad57897dfcb45388050772a0bc98f854",
    "KVnorm": "00c3c5ac47cacd7a7284bdb71c2a5c92019efe2946c1c2f73c84ef5a9f1e837e",
    "Qcur": "9eb09a970bf97d8ea4d078d3b4e2c1db6ea5cf40888fc1888334161e632c9f91",
    "KVrope": "e447bbace00ad4b70ecf24c3c91dbbe38be09ea4e21c64711d030cca861f565e",
    "KVcur": "edaba3fe7eeb97e4e8903a425d7e31c233fa8c8d4355eea82dd100d1a162abd3",
    "kqv_out": "32afdaa9f83a00c9d3f2325b7ffd72d4efe689c1b349a65eebba1f4200116251",
    "kqv_back": "b3c3581cb3e2b0f80d1f85dcc36025484208eb1603936ff08b7bd1295fb402ba",
    "attn_low": "a28d7b047ffde613ada27537c82b33fcb3404b9fd5d5da757707ffc474cc408d",
    "attn_out": "3d5eb12440f3b50ec11e56e151b29caaa784d379665aef6e7739c1376cc73c5a",
    "hc_attn_post": "87a9d6b8214e29ff59dedbfb3e26c8625d7d7d3a7c81a8e826e31773f6740de4",
    "hc_ffn_pre": "d5601b9b9acab731ed47c1101031904be02e27c847c1143e726ec05e169e38e8",
    "ffn_norm": "8102d5b5cca1c48f44958fbfb0328949fefc95f97c9bdf2b585da35bd60a0731",
    "ffn_moe_logits": "01960ac6a89587151a9c32334484e3ff1eca2f5088a7b9b0b55fc5952966b6ef",
    "ffn_moe_probs": "45ee62ca0d49dee00a2a0afde32057b7640f17d4a78a3a811b338cb05e272871",
    "ffn_moe_topk": "e043d545aa049176846e324de8372791fb787afe9506836f073306685773be95",
    "ffn_moe_weights_scaled": "9bf595cf4bb8c3e03480b04d9195ce7d62403a596a66cc58bef5a47c1de7fe9d",
    "ffn_moe_weighted_swiglu": "74de94e52f39db07fe10162ebe2e893f7f59c2170e6e02ba3ad64089ac9b8426",
    "ffn_moe_out": "896030ae1baa4efe342a9b075b4d68cfd45966cc5bf8e1fbcfef35b5c15b03e6",
    "ffn_shexp": "95006dc68a7d9b253e67c4b15fc7f3fe5cf8b4271aa845d2db1d265291b4da8d",
    "hc_ffn_post": "09b31ad2a23ef6805ab62c9cbd353caf280546770e432760f14252e400c42cca",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path, layer: int, name: str) -> Path:
    suffix = "i32" if name in {"indexer_topk", "ffn_moe_topk"} else "bin"
    return root / f"oracle_{name}-{layer}_pos4096.{suffix}"


def repeated_full(
    first: Path, second: Path, layer: int, name: str, expected_sha: str
) -> bytes:
    first_payload = capture_path(first, layer, name).read_bytes()
    second_payload = capture_path(second, layer, name).read_bytes()
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process layer-{layer} {name} captures differ")
    if sha256(first_payload) != expected_sha:
        raise SystemExit(f"layer-{layer} {name} capture identity changed")
    return first_payload


def write_tensor(
    output: Path,
    tensors: list[dict[str, object]],
    name: str,
    hook: str,
    payload: bytes,
    rows: int,
    width: int,
    dtype: str,
    role: str,
) -> None:
    path = output / name
    path.write_bytes(payload)
    descriptor: dict[str, object] = {
        "name": name.rsplit(".", 3)[0],
        "hook": hook,
        "role": role,
        "dtype": dtype,
        "shape": [rows, width],
        "encoding": (
            "little-endian-signed-integer32"
            if dtype == "i32"
            else "little-endian-ieee754-binary32"
        ),
        "path": name,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }
    if hook == "indexer_scores":
        descriptor["value_semantics"] = "ieee754-binary32-bit-pattern"
    tensors.append(descriptor)


def tile_slice(payload: bytes, width: int) -> bytes:
    row_bytes = width * 4
    expected = CHUNK_ROWS * row_bytes
    if len(payload) != expected:
        raise SystemExit(f"full capture has {len(payload)} bytes, expected {expected}")
    start = TILE_OFFSET * row_bytes
    return payload[start : start + TILE_ROWS * row_bytes]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer2-first", type=Path, required=True)
    parser.add_argument("--layer2-second", type=Path, required=True)
    parser.add_argument("--layer3-first", type=Path, required=True)
    parser.add_argument("--layer3-second", type=Path, required=True)
    parser.add_argument("--state-first", type=Path, required=True)
    parser.add_argument("--state-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    output = args.fixtures_root / "prefill-layer23-continuation-second-tile-4128-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    tensors: list[dict[str, object]] = []

    for hook, (width, dtype, filename) in LAYER2.items():
        payload = repeated_full(
            args.layer2_first,
            args.layer2_second,
            2,
            hook,
            LAYER2_FULL_SHA256[hook],
        )
        write_tensor(
            output,
            tensors,
            filename,
            hook,
            tile_slice(payload, width),
            TILE_ROWS,
            width,
            dtype,
            "output" if hook == "hc_ffn_post" else "intermediate",
        )

    layer3_payloads: dict[str, bytes] = {}
    for hook, (width, filename) in LAYER3.items():
        payload = repeated_full(
            args.layer3_first,
            args.layer3_second,
            3,
            hook,
            LAYER3_FULL_SHA256[hook],
        )
        layer3_payloads[hook] = payload
        dtype = "i32" if hook == "ffn_moe_topk" else "f32"
        write_tensor(
            output,
            tensors,
            filename,
            hook,
            tile_slice(payload, width),
            TILE_ROWS,
            width,
            dtype,
            "output" if hook == "hc_ffn_post" else "intermediate",
        )

    fixtures = Path(__file__).resolve().parent / "fixtures"
    first_history = (
        fixtures
        / "prefill-layer3-continuation-tail-4096-v1"
        / "history-raw-kv.f32le.bin"
    ).read_bytes()
    first_kv = (
        fixtures
        / "prefill-layer3-continuation-ingress-4096-v1"
        / "kv-current-first-tile.f32le.bin"
    ).read_bytes()
    captured_first_kv = layer3_payloads["KVcur"][: TILE_ROWS * 512 * 4]
    if captured_first_kv != first_kv:
        raise SystemExit("full-batch layer-3 first tile changed from the pinned fixture")
    row_bytes = 512 * 4
    history = first_history[32 * row_bytes :] + first_kv
    write_tensor(
        output,
        tensors,
        "layer3-history-raw-before-tile2.f32le.bin",
        "mixed_raw_cache",
        history,
        128,
        512,
        "f32",
        "input",
    )
    compressed = (
        fixtures
        / "prefill-layer3-continuation-tail-4096-v1"
        / "history-compressed-kv.f32le.bin"
    ).read_bytes()
    write_tensor(
        output,
        tensors,
        "layer3-history-compressed-before-tile2.f32le.bin",
        "mixed_attn_comp_cache",
        compressed,
        32,
        512,
        "f32",
        "input",
    )

    for hook, filename, expected_sha in [
        (
            "attn_state_kv",
            "layer3-attention-state-kv-after-tile2.f32le.bin",
            "0b6c55380a3bd6655d208b5e7239decb95b900ed328579b0b67d79909754c735",
        ),
        (
            "attn_state_score",
            "layer3-attention-state-score-after-tile2.i32le.bin",
            "9ae471b7f779cd35445619199d1dc6a1ac9c15962f0c8f24a6d22603347bf713",
        ),
    ]:
        payload = repeated_full(
            args.state_first, args.state_second, 3, hook, expected_sha
        )
        if len(payload) != 128 * 512 * 4:
            raise SystemExit(f"layer-3 {hook} state has the wrong size")
        write_tensor(
            output,
            tensors,
            filename,
            hook,
            payload,
            128,
            512,
            "i32" if hook == "attn_state_score" else "f32",
            "output",
        )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-layer23-continuation-second-tile-4128",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": {
            "id": "oracle-v3",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "d35fb12d01d500b9cefcef24092c295687ceaf7e",
            "tree": "617415ee9f8ea7dc176d63dada1d5a7582063824",
            "capture_executable_sha256": CAPTURE_EXECUTABLE_SHA256,
            "diagnostic_patch_sha256": CAPTURE_PATCH_SHA256,
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
            "production_prefill_tokens": 8_192,
            "state_prefill_tokens": 4_160,
            "prefill_chunk": CHUNK_ROWS,
            "chunk_start": CHUNK_START,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "layer2_full_batch_sha256": LAYER2_FULL_SHA256,
            "layer3_full_batch_sha256": LAYER3_FULL_SHA256,
            "storage_note": (
                "Only positions 4128--4159, their exact prior layer-3 history, "
                "and the ratio-128 state after 64 continuation rows are retained."
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layers": [2, 3],
            "position": TILE_START + TILE_ROWS - 1,
            "captured_position_range": [TILE_START, TILE_START + TILE_ROWS - 1],
            "tile_rows": TILE_ROWS,
        },
        "operations": [
            {"name": "layer2-indexed-attention-and-ffn", "kernel": "production M1 batch schedule"},
            {"name": "layer3-ingress-qkv", "kernel": "production M1 batch schedule"},
            {"name": "layer3-ratio128-attention-and-ffn", "kernel": "production M1 batch schedule"},
        ],
        "claims": {
            "repeated_production_geometry": True,
            "second_consecutive_layer2_tile": True,
            "second_consecutive_layer3_tile": True,
            "native_retained_history_target": True,
            "complete_layer3": False,
            "complete_8k_transformer": False,
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
