#!/usr/bin/env python3
"""Import repeated full-2K layer-7 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-22T20:10:41Z"
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
    "hc_attn_pre": "e41f039ec7ff8ce39c5034ebd9a147b0e5e821708faffc85ce5c10778269b866",
    "attn_norm": "1b13b72d86a1f4eb0fcfd6489cce937c640cb8e47411ce8b43bbe573c5f4c46c",
    "q_lora": "ec07bd324c99ff9ce4dd0899cdb5688858127a8344d6dcd2e058d84d743f1327",
    "q_lora_norm": "2c34db8deef6722150a484abe5994785c3e6ad509f176b39ec60e4be3d01911c",
    "KVraw": "53977f422cee204783963a1d366815ec79d093366b28aee84e303568bb2d87dd",
    "KVnorm": "ff42f8a2ca668e15a23859d712f6817e44eb81c81e243f1156c3bec44b9e23a4",
    "Qraw": "0f44f7619ded6c0df7d0cab3f2dba09501d8d7349dbc1a6061eb39047c17be4b",
    "Qcur": "1c42981475d71b778b0da2085e0b2a30eb101a37b02f0fb8a3617baca2a734fe",
    "KVrope": "c854a8ef53f601756438e925e4c0ec21b6b3a82589f31ec590be471d3c9988f0",
    "KVcur": "0d159d1e47d6ea63081ce1b77dbec94f5b04815d4cb61e801e70ab29e4c5172c",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [16, 512], 32_768,
        "1e09097888d13e3ce604a7abbf60ed4ba278d0697b5dc2d4aa69cffebdc704dd"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [128, 512], 262_144,
        "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90"),
    "attn_state_score": ("attention-state-score.i32le.bin", [128, 512], 262_144,
        "6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "3679d744a6c6c1c67300670efa4ba855913b13030f573af46ea528e03eafc10d",
    "hc_attn_post": "7dbf788967cb20ab9c3d2220d3d11b0f07769b26b7896da651c7258573d82aee",
    "kqv_out": "37de177bb00b22ed081a3e3d76c413860bb690949531e41f81e7cb0e8521f694",
    "kqv_back": "8bca205140508c75a84d69cd28af622ed05ab345e173496aa7ab5b2d8be6cf6b",
    "attn_low": "db46d993c04973f790047544631629414b40cf5e9ba7f80924ee482830aafc1c",
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
    "hc_ffn_pre": "af6e4a2acf5213bfe6447b1b00d00c94b7d321cd10e01f975b5dddd027fff632",
    "ffn_norm": "19c883a124914b090e8d6a3f16a7450dc554e69de790c188bb1c6020a2755ba9",
    "ffn_moe_logits": "834473b7f8137b4cc73a1cce3088fdc801a3735d07fa4b77a7b120cf1dbb1ca3",
    "ffn_moe_probs": "e4d872e1ca2aeaa7415f523149b3bd2474cf23eddc711773b868c8c106351504",
    "ffn_moe_topk": "d896637ac717eaa9e37bf6ffbf6129eccf2c16b0998100f9ffc09b9a21d4dcee",
    "ffn_moe_weights_scaled": "4f3d35964a683c0aca5bce01bad3bd1891b216196deec64555f609936f110a45",
    "ffn_moe_weighted_swiglu": "3c5737c5e1dbd0122bce545bac361b44fe12ea8d72691df16ca50ad9c5328dba",
    "ffn_moe_out": "026c6b20b7f0cf8eeba687fb5918f179e4915e03049bb8f10b282dbb8cf87933",
    "ffn_shexp": "538f6b8bac996fcff94f508f3590d619f4ba57cc5a090ad0dfe2afa368390585",
    "hc_ffn_post": "f87ae175ff285ada79ac50a88ad7b12409609ec2ae1a1d744dff6bc385457f80",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-7_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-7 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-7 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-7 {hook} capture identity changed")
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
            "layer": 7,
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
        (args.fixtures_root / "prefill-layer6-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer7-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer7-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer7-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer7-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer7-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer7-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer7-q-raw-final-tile.f32le.bin",
        "Qcur": "layer7-q-current-final-tile.f32le.bin",
        "KVrope": "layer7-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer7-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer7_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer7-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer7-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer6-complete-2048"},
        [
            {"name": "attention-hc-ingress-and-norm", "kernel": "HC ingress plus learned norm"},
            {"name": "q-kv-state", "kernel": "Q8 projections plus RoPE and E4M3FN"},
        ],
        qkv_tensors,
    )

    compressor_payloads = {
        hook: repeated(args.tail_root, "compressor", hook, spec[2], spec[3])
        for hook, spec in COMPRESSOR.items()
    }
    compressor_tensors = []
    for hook, (filename, shape, _size, _identity) in COMPRESSOR.items():
        payload = compressor_payloads[hook]
        dtype = "i32" if hook == "attn_state_score" else "f32"
        compressor_tensors.append((filename, payload, tensor(
            f"layer7_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer7-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer7-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer7-qkv-2048"},
        [
            {"name": "ratio128-paired-projections", "kernel": "kernel_mul_mm_f16_f32"},
            {"name": "ratio128-pool-and-finalize", "kernel": "softmax pool plus norm/RoPE/E4M3FN"},
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
        filename = f"layer7-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer7_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer7-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer7_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer7-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer7_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer7-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer7-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer7-compressor-2048"},
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
        filename = f"layer7-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer7_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer7-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer7-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer7-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
