#!/usr/bin/env python3
"""Import repeated full-2K layer-16 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-25T06:58:43Z"
QKV_WIDTHS = {
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
QKV_SHA256 = {
    "hc_attn_pre": "4d18937052a014370ef19a5ad4ec2a6d5f7d4fd77d13a282feba3ef97506c713",
    "attn_norm": "d77a84e5b1702c06020174703dcaa5dbca5b5ffcef3496f3f9acf309f30f9129",
    "q_lora": "e7dc248f69f9e65188fe8b2a5091b10ca367ec6a9e0814e6b687772396dd61a9",
    "q_lora_norm": "3e7072771e389abcbc13373b60d5d5374bc141d6e158ce849c37d31083fd196b",
    "KVraw": "462d3b54d194a66febb822a599249be2ab1b61b452fc4f46654242a633ca9a9b",
    "KVnorm": "ea8a4b3dc4b18c118815dcc48e648d554ecdbdf0cbda1d6f600ab018ebe5af53",
    "Qraw": "2bbcdb01509854272b260e63fafd47d85f578ffb9dec8522e2fc48efad9f654e",
    "Qcur": "318307d3b955f67788ec755156e46f333be02527fe4aa953443c51f60d79082c",
    "KVrope": "e77219b3739ef59aa84b207ecfe0077f762b464f1f03611315b2544fbfeaa305",
    "KVcur": "3ad4d356c46e71103dec98f7945fe9ee3cc7e94ee4fae829b51ccf7f7e5f9822",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "900d877ee2f03980df888bea162ec01f3ca2aac66f60184c6c811a45cb3c19a5"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "59f0f2d409a6da3f2152caf8bee3058f2fc902ba4d82ac15368abb614c956781"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "762b07d51ee1f207ee842dc41845b0276b6f81acbe08b432d7a34c5a2cb607cf"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "1260659cf68bd2cc122956662d54e93423295e103a9a60cdb77677bb107c1be3"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "2cfc35bae22aed97acb2ecb93444fb33b312c1b38a434e18ecace5a64963af61"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "0e3ba938963472ab5d70b4001150dd44e8cb311f1aefed81a154af96af38b82b"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "5ba6877c24fcf5787368737271e63216b50eb6bc6ca3454b8ab91255dbb2d12a",
    "hc_attn_post": "03c8f77997b70959d215acc4e0e6ec8470c3e6f5a065408b12de16071e32e7c7",
    "kqv_out": "a785817f6b180b26724d6012cc076aea6c70bc1a70b63960311de8167c5c5ee5",
    "kqv_back": "28546fba99858c63101b4c9eece33ed617bb02afcd3be80e3ef8835db08ff1a2",
    "attn_low": "789a7b9760c02fb9b2011693e177f26fdfd69ac17bcb26d1fe5793916c829382",
}
FFN_WIDTHS = {
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
FFN_SHA256 = {
    "hc_ffn_pre": "79730d856ee12e00d4afb2b7e5f1fda67d444d4299f24155e1f4197d4749d6cc",
    "ffn_norm": "7f817b639db949f179e235c040aaccdb467d3216181b97eafb3afa43578ed1c4",
    "ffn_moe_logits": "13eb3f89a21bc3b9b86ab5eec01a5e1f3837691432b322442f5ced4095f5537c",
    "ffn_moe_probs": "2d08656b31c68154df5685518594b2e9d9f0295bc0eb5140f9ee4dff391bbf7d",
    "ffn_moe_topk": "d827292540848fe42822222dc3ec493fe5fbdd0cdc0946d2fa6ac26bb45c0df2",
    "ffn_moe_weights_scaled": "f1b36b79c10f5657890e74cfe4aed778423554ae6aae7849e60b7f75d739bd4b",
    "ffn_moe_weighted_swiglu": "feadd3913305e2846911a086d2cc8708377e6f2956a8fddabee416c97181e760",
    "ffn_moe_out": "e7511c6bbb47e9f66cd27849360087cc644fe886491c7f19ec128b587ba085c9",
    "ffn_shexp": "c73e415d4f57b299ff29917a51f2fe4adea06bccc166a07e161e89f5740b1039",
    "hc_ffn_post": "6222f1726b68e8403f76b44708c4351ae2b4351334ebe88e7c28ca4212dd3935",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-16_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-16 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-16 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-16 {hook} capture identity changed")
    return first


def tensor(
    name: str,
    hook: str,
    dtype: str,
    shape: list[int],
    filename: str,
    payload: bytes,
    role: str = "intermediate",
) -> dict[str, object]:
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "encoding": (
            "little-endian-signed-integer32"
            if dtype == "i32"
            else "little-endian-ieee754-binary32"
        ),
        "path": filename,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def write_fixture(
    fixtures_root: Path,
    directory_name: str,
    fixture_id: str,
    template: dict[str, object],
    capture_metadata: dict[str, object],
    operations: list[dict[str, str]],
    tensors: list[tuple[str, bytes, dict[str, object]]],
) -> None:
    output = fixtures_root / directory_name
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    records = []
    for filename, payload, record in tensors:
        (output / filename).write_bytes(payload)
        records.append(record)
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": fixture_id,
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": capture_metadata,
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 16,
            "position": ROWS - 1,
            "captured_position_range": [0, ROWS - 1],
        },
        "operations": operations,
        "tensors": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv-root", type=Path, required=True)
    parser.add_argument("--tail-root", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    template = json.loads(
        (args.fixtures_root / "prefill-layer15-complete-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    common_capture = {
        "backend": "metal",
        "machine": template["capture"]["machine"],
        "prompt": template["capture"]["prompt"],
        "prompt_sha256": template["capture"]["prompt_sha256"],
        "prefill_tokens": ROWS,
        "fresh_process_bitwise_match": True,
        "command": template["capture"]["command"],
    }

    qkv_payloads = {
        name: repeated(
            args.qkv_root,
            "qraw" if name == "Qraw" else "primary",
            name,
            ROWS * width * 4,
            QKV_SHA256[name],
        )
        for name, width in QKV_WIDTHS.items()
    }
    qkv_filenames = {
        "hc_attn_pre": "layer16-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer16-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer16-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer16-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer16-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer16-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer16-q-raw-final-tile.f32le.bin",
        "Qcur": "layer16-q-current-final-tile.f32le.bin",
        "KVrope": "layer16-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer16-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer16_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer16-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer16-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer15-complete-2048"},
        [
            {"name": "attention-hc-ingress-and-norm", "kernel": "HC ingress plus learned norm"},
            {"name": "q-kv-state", "kernel": "Q8 projections plus RoPE and E4M3FN"},
        ],
        qkv_tensors,
    )

    compressor_payloads = {
        hook: repeated(
            args.tail_root,
            "compressor",
            hook,
            spec[2],
            spec[3],
        )
        for hook, spec in COMPRESSOR.items()
    }
    compressor_tensors = []
    for hook, (filename, shape, _size, _identity) in COMPRESSOR.items():
        payload = compressor_payloads[hook]
        dtype = "i32" if hook.endswith("state_score") else "f32"
        compressor_tensors.append((filename, payload, tensor(
            f"layer16_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer16-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer16-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer16-qkv-2048"},
        [
            {"name": "ratio4-attention-and-indexer-projections", "kernel": "kernel_mul_mm_f16_f32 x4"},
            {"name": "ratio4-pool-and-finalize", "kernel": "replay pool plus norm/RoPE/E4M3FN/indexer QAT"},
        ],
        compressor_tensors,
    )

    attention_payloads = {
        name: repeated(
            args.tail_root, "attention", name, ROWS * width * 4,
            ATTENTION_SHA256[name]
        )
        for name, width in ATTENTION_WIDTHS.items()
    }
    attention_tensors = []
    for name in ("kqv_out", "kqv_back", "attn_low"):
        width = ATTENTION_WIDTHS[name]
        payload = attention_payloads[name][: width * 4]
        filename = f"layer16-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer16_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer16-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer16_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer16-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer16_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer16-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer16-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer16-compressor-2048"},
        [
            {"name": "dense-mixed-attention", "kernel": "kernel_flash_attn_ext_f16_dk512_dv512"},
            {"name": "attention-output-and-hc-post", "kernel": "Q8 output projections plus HC expand4"},
        ],
        attention_tensors,
    )

    ffn_payloads = {
        name: repeated(
            args.tail_root, "ffn", name, ROWS * width * 4, FFN_SHA256[name]
        )
        for name, width in FFN_WIDTHS.items()
    }
    ffn_tensors = []
    for name, width in FFN_WIDTHS.items():
        payload = ffn_payloads[name][-TILE_ROWS * width * 4 :]
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        filename = f"layer16-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer16_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer16-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer16-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer16-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
