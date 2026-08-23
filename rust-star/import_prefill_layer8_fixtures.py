#!/usr/bin/env python3
"""Import repeated full-2K layer-8 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T21:08:00Z"
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
    "hc_attn_pre": "0969e4c07110bce34df755e545ad92d81eb6933957e5c58b58f65d1413ce0f14",
    "attn_norm": "8722a95b6752dbf03665486f05ecbbb819ed49a6485b67cd6a0c2c1772c7d023",
    "q_lora": "50c640e8b159d47c2e8235baf23ecb21f500da4771066605a8c36ea18bed9b66",
    "q_lora_norm": "1020f36a270a9636d3d78cb67fc2aea55229d6c0d325ffed00dbe27b742e2c02",
    "KVraw": "0d78bc424dd19b033490d227516a08eedb1fbf7104438350ab5b7c27c683a6b4",
    "KVnorm": "c362bde154053c423859ab38024e2ad748e938bae1059b2d54e7bcddf8ca767d",
    "Qraw": "c25227196057e10b1add2a6175a438b47f4e939cf4ce0b2762937caaa788fa9c",
    "Qcur": "552771e437375f84ec420200e5bedd33a72d786508a1c77d68c78f0b75659b95",
    "KVrope": "9fd973b0485033a38ea13c8f3f1a5647e8819c1251b40856536f490aa1d45728",
    "KVcur": "576b2eb468d4ed25dc3e44e055479a5a6e5c5ac5428020326219e1b5649e5d56",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "0da512cc10ca98601b291cd6d341a26fa3538582e6ffebfa7ca19f119b78e45f"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "222444e5c22cb68e0653e55733fe75e0ccfc0cebecd36fcbf6828425b4fc278b"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "645b93152d5c6f69c29471808f9f14e9d9952df81193549e7ee4bfd69d7e041b"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "bbfdcc0ca22f660f9fa2eb791428565e7f0d08b21d41e67a1f58bef223167b8c"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "e31cd95830b775bbd41e513f2c9a3da8125c901fff68b7300db6342e36ca064a"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "92ffe1b810aa118551c098f166c6d4926c78e1aeeecdb248d75b0996feb8bbd5"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "f2ddf9dba72e4f656b441f7acca26e315df26663b4d56548534cdc45e4aefccb",
    "hc_attn_post": "e1013c47c3bc69a16376ae5f3d31e18772217cd7a6c10ce01717f53bf62cd860",
    "kqv_out": "82f758b70b7648f26f9978ea2dc3d9517626c4595f1a9326a4cbbafbe0c00d86",
    "kqv_back": "9b5c679d861e432f64ba5f2dc3f6e9e742b068df3e0b67d116138a12791b3c64",
    "attn_low": "7f352f7b6ad2207521bbe7b167612a1be7d4c5f90ade61a0b90645312271d498",
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
    "hc_ffn_pre": "df90fac2081ede6f29b3d4fcfc203a5b7640f5250ba8e49ea67cf0499ad9c2a4",
    "ffn_norm": "3ed27dcb9d42eca0dfaad98ae1fead0dc19501c6641232f28d94884faee4a48b",
    "ffn_moe_logits": "6dbe65e6d6b3c4b582e88a8f99bc079b8fcde9f65bc785dc6dc23edf1c416217",
    "ffn_moe_probs": "b1fc6c90a5998e6bb31d321016f95332cd526e841f8b8288c5ab79871f3c38f1",
    "ffn_moe_topk": "8e641af7970e0be726e6968112bc7fecca4f669f4934a8aeddac716ac015c5b7",
    "ffn_moe_weights_scaled": "e5a93f4b1382aa0b25e72cdc531e39c29d14aa3a78240f8945923a756a66626d",
    "ffn_moe_weighted_swiglu": "307f08e52eda61a29b7f35506287b89b61bdcf38754fc1771e5912ac05c17241",
    "ffn_moe_out": "146755fac00beb93e6dcdb50afb80d0b86aff07c2d95c3b7e13e58a21b5edaa9",
    "ffn_shexp": "5bd7f1a1aa7616fdc275608ad1ba119fb82a83172c100e394315c6df0008993b",
    "hc_ffn_post": "b4807aa8b9493a57c40de455ca38a25eca9e225967ec73b3b79c464852949137",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-8_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-8 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-8 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-8 {hook} capture identity changed")
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
            "layer": 8,
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
        (args.fixtures_root / "prefill-layer7-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer8-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer8-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer8-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer8-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer8-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer8-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer8-q-raw-final-tile.f32le.bin",
        "Qcur": "layer8-q-current-final-tile.f32le.bin",
        "KVrope": "layer8-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer8-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer8_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer8-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer8-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer7-complete-2048"},
        [
            {"name": "attention-hc-ingress-and-norm", "kernel": "HC ingress plus learned norm"},
            {"name": "q-kv-state", "kernel": "Q8 projections plus RoPE and E4M3FN"},
        ],
        qkv_tensors,
    )

    compressor_payloads = {
        hook: repeated(
            args.tail_root,
            "indexer" if hook.startswith("indexer_") else "compressor",
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
            f"layer8_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer8-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer8-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer8-qkv-2048"},
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
        filename = f"layer8-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer8_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer8-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer8_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer8-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer8_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer8-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer8-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer8-compressor-2048"},
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
        filename = f"layer8-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer8_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer8-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer8-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer8-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
