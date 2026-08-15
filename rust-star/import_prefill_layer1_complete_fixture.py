#!/usr/bin/env python3
"""Import the complete final layer-1 tile of a 2K Metal prefill."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T20:16:32Z"
PREFILL_ROWS = 2_048
PREFIX_ROWS = 2_016
TILE_ROWS = 32
WIDTHS = {
    "q_lora_norm": 1_024,
    "KVnorm": 512,
    "Qcur": 32_768,
    "KVrope": 512,
    "KVcur": 512,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "hc_ffn_pre": 4_096,
    "ffn_norm": 4_096,
    "ffn_moe_logits": 256,
    "ffn_moe_probs": 256,
    "ffn_moe_topk": 6,
    "ffn_moe_weights_scaled": 6,
    "ffn_moe_weighted_swiglu": 6 * 2_048,
    "ffn_moe_out": 4_096,
    "ffn_shexp": 4_096,
    "hc_ffn_post": 16_384,
}
EXPECTED_FULL_SHA256 = {
    "q_lora_norm": "3d82b5aea3a88b747a257c333e127b7f4e43968bd9dc69500e4523cedb51e495",
    "KVnorm": "cf8703af11601aed00c4c69646ac11e9fe199383e90e9da336028991cc594b43",
    "Qcur": "569f5d6b77aa261423e47d5a11ba4566758318e547fb0eae23d34bf798bc3e6c",
    "KVrope": "3e4070bc5f458004740126d9f54a54ef615a9ca5003129525cf3c15d2c9a7458",
    "KVcur": "a008066c234ef8b9bf162b38fb55e1593302a7d517138fc6c36df13595449cbb",
    "kqv_out": "54a6c9feb7082378488676842c7161866c90b92309c392ede685541b9cf08de2",
    "kqv_back": "867ad2c08d2ed747a15a37d4af2d2d821dce208d7f1c54d039e77aa187e85762",
    "attn_low": "5a85f9cd903b2a767bba1f3fd04cb1563dc93c7a2c4ec7735f6003cb3861d5dc",
    "attn_out": "928a3847a3627695030b4fcd28580888984d692318a47faac4fcd4cf61bc5daf",
    "hc_attn_post": "94ba908546cb0e677fbe9f3194f85cb6505ecfca14a3cf4ce84b8cd0efd32fbe",
    "hc_ffn_pre": "5f74a9f6b6b9f032cfd116f6692218c1ad0880948c32731bfb85b4936a42f2c5",
    "ffn_norm": "e18e5231f01bf536be965eda13256683b731661e02814ac7dcb7254c9fb39c92",
    "ffn_moe_logits": "8e91bc19729001fad7f5667aa63929b3e565b550b26f15cf948925220a08dccb",
    "ffn_moe_probs": "94b1364931f2a466f65f2b2694d4362fea7ce6dbffd5aac52d5e7750addc0c29",
    "ffn_moe_topk": "e727333d04bf36fdaa50fbf931cae10088195cfe69e57518363535f9466240a5",
    "ffn_moe_weights_scaled": "bece92bdcc51c7cecc8ae114169bb5b075f560ac1bbcffd4c4193883612288ad",
    "ffn_moe_weighted_swiglu": "0554f3f6ccf349495658f08a8875cc9b65427cda4419922a78a2fc51b1ea827f",
    "ffn_moe_out": "2a1a548f743755cec79b07bd05e8c072cd04e3b9c4653dfd9be09dbfc9299306",
    "ffn_shexp": "0610d98358347392787288b1b060d8f156cc9ff78651b12df4dfbe9d0d2713e9",
    "hc_ffn_post": "2ed3f6203eed280bef18380e77511a6fc2122f25d1cb96dc1cf4921b95505e99",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(directory: Path, label: str, name: str) -> Path:
    suffix = "i32" if name == "ffn_moe_topk" else "bin"
    return directory / f"{label}__{name}-1_pos0.{suffix}"


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    first_payload = capture_path(first_dir, "first", name).read_bytes()
    second_payload = capture_path(second_dir, "second", name).read_bytes()
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


def tensor(name: str, role: str, path: str, payload: bytes, shape: list[int]) -> dict:
    dtype = "i32" if name == "ffn_moe_topk" else "f32"
    return {
        "name": name,
        "hook": name.removeprefix("layer1_").removesuffix("_prefix"),
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "encoding": (
            "little-endian-signed-integer32"
            if dtype == "i32"
            else "little-endian-ieee754-binary32"
        ),
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    full = {
        name: checked_repeated(name, args.first_dir, args.second_dir)
        for name in WIDTHS
    }
    payloads = {name: final_tile(payload, WIDTHS[name]) for name, payload in full.items()}
    payloads["layer1_kv_current_prefix"] = full["KVcur"][: PREFIX_ROWS * 512 * 4]

    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer1-ingress-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer1-complete-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    files = {
        name: f"layer1-{name.replace('_', '-').lower()}-final-tile."
        f"{'i32' if name == 'ffn_moe_topk' else 'f32'}le.bin"
        for name in WIDTHS
    }
    files["layer1_kv_current_prefix"] = "layer1-kv-current-prefix.f32le.bin"
    for name, payload in payloads.items():
        (output / files[name]).write_bytes(payload)

    tensor_entries = [
        tensor(
            f"layer1_{name}_final_tile",
            "output" if name == "hc_ffn_post" else "intermediate",
            files[name],
            payloads[name],
            [TILE_ROWS, WIDTHS[name]],
        )
        for name in WIDTHS
    ]
    tensor_entries.insert(
        5,
        tensor(
            "layer1_kv_current_prefix",
            "input",
            files["layer1_kv_current_prefix"],
            payloads["layer1_kv_current_prefix"],
            [PREFIX_ROWS, 512],
        ),
    )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer1-complete-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "1",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "device_path": (
                "legacy uncompressed Q/KV batch setup; guarded raw KV; rectangular "
                "FlashAttention; grouped Q8_0 output; token-hash MoE with fused "
                "IQ2_XXS weighted-SwiGLU/Q2_K down; Q8_0 shared expert; additive HC"
            ),
            "graph_dump_fusion_note": (
                "DwarfStar disables routed pair-SwiGLU fusion while graph dumping; "
                "the production source contract requires the retained F16 intermediate "
                "and routed output to be bit-identical"
            ),
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer1-ingress-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 1,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
            "kv_position_range": [0, PREFILL_ROWS - 1],
        },
        "operations": [
            {"name": "qkv-setup", "kernel": "legacy Q8_0 batch plus fused QKV RMSNorm/RoPE"},
            {"name": "kv-finalize", "kernel": "kernel_dsv4_rope_tail_f32 plus FP8/cache staging"},
            {"name": "attention", "kernel": "kernel_flash_attn_ext_f16_dk512_dv512"},
            {"name": "attention-output", "kernel": "grouped and dense kernel_mul_mm_q8_0_f32"},
            {"name": "attention-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
            {"name": "ffn-ingress-router", "kernel": "kernel_mul_mm_f16_f32 plus decomposed M1 router"},
            {"name": "routed-experts", "kernel": "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16 plus Q2_K down"},
            {"name": "shared-expert", "kernel": "kernel_mul_mm_q8_0_f32 plus kernel_swiglu_flat_f32"},
            {"name": "ffn-hc-post", "kernel": "kernel_dsv4_hc_expand4"},
        ],
        "tensors": tensor_entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
