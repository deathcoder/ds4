#!/usr/bin/env python3
"""Import repeated full-2K layer-15 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


LAYER = 15
ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-24T19:40:00Z"

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
    "hc_attn_pre": "80e4845bfec76026f366aeb7d9cafa1feb2189f1566b3f85a676806dafcc3183",
    "attn_norm": "7c1c6e128e45de22fe9b21ddb83cdf6406c7f8a9a47a8e678fd5a29bb9d8fbe1",
    "q_lora": "71aacab1728bc149a3d667f8955a3b31936a54f4ec602fd36be6aa657ae2d459",
    "q_lora_norm": "a4af08ebbbf9ec1ec5036c55cc8f89ee382bef603a13ca8df8439879b463d73d",
    "KVraw": "0f5c251980cd4c6779634caae1aa3d4dfafb91fa94ed7f2aca7c1b1edbd49f67",
    "KVnorm": "0c8ad08023b59841a4e47640b1f4df512afa4ef854e4569a4646e6f2f6b63bad",
    "Qraw": "9eb3851c9689c2b9e68f9b4d3f34e03a966a9f735a2cd1ebefb71c9c93652ff7",
    "Qcur": "4c6beab077811b3362d3f112098e972d56dbf69ec85b245608308188259845a7",
    "KVrope": "bc8d9a33c21431b86036eb7ec4d43ae3101057a6468d034f11c38255b0363e1c",
    "KVcur": "0d8a2dbe1d455f68f1497e85418cd7d613ed6de6fac6e5213f4622cc44c9be05",
}

COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "0ba4a4febf698dc4e638eea4f24220996aa5c124808d60eb53438fc1c25e943c",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [128, 512],
        262_144,
        "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [128, 512],
        262_144,
        "6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778",
    ),
}

ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "2c38453908b8a351dd05901baf0a5e49bba7902dfc222f4490b04c18951753b7",
    "hc_attn_post": "a7b4d53974b1fdee777f2c2200e73beebecb2b5b269bfde42a5966f3accf4f22",
    "kqv_out": "71124f6664f9385b995a672a738507625c123c89bb25d1b93b0045eb55d92c20",
    "kqv_back": "d33993a457062febbe87a71e0059e3247dd39be724eacb0f506d15e87905129f",
    "attn_low": "a8d7e10a54752e7e7300adb428369e615513aacb5ae8dfffdfab094ed4cc33cd",
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
    "hc_ffn_pre": "fc3b6f37f01fac90ed010ab410eb167c20deaf06dabb4a21dc143e758d16e317",
    "ffn_norm": "340f9b9c5d240e76357ee3e96587a02035f5a7a7bcb6a8978ddf3f2ccacdcdcd",
    "ffn_moe_logits": "3e3bd3b63ecb0a5215b78c5ccad69ffe53bac3133e6e09eb7e2fa07230aae8d3",
    "ffn_moe_probs": "0d96812f812addac92574248b367a323ed722a008123c6bad6e431b2f4cba273",
    "ffn_moe_topk": "66b23cff322b1fc42984c0ef2f037546f27520761d2bd8e70513074bc3f4bdd1",
    "ffn_moe_weights_scaled": "1c8a2a9d130c98c5a39990bdb271677898a809b95c6c271f9e56064307ee83b5",
    "ffn_moe_weighted_swiglu": "b82ff5755aac08ac33e184b411efe0cab051f03634bc6d44c960ed548b1eaceb",
    "ffn_moe_out": "68c9758ba9e84a652206a1eaad130ffcd0e05131fc89dde312da80c3a1fe0849",
    "ffn_shexp": "ab77c3ee2372e65fc2c987a219e19047f1b762210f19d4b8c71b51cd34b896a0",
    "hc_ffn_post": "3eb3fddd24f243bf5568eafbccc5cbcb3f618bfa9d3594ef16bca2cc7be20306",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-{LAYER}_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-{LAYER} {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-{LAYER} {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-{LAYER} {hook} capture identity changed")
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
    suffix: str,
    template: dict[str, object],
    capture_metadata: dict[str, object],
    operations: list[dict[str, str]],
    tensors: list[tuple[str, bytes, dict[str, object]]],
) -> None:
    directory_name = f"prefill-layer{LAYER}-{suffix}-2048-v1"
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
        "fixture_id": f"dwarfstar-oracle-v1-prefill-layer{LAYER}-{suffix}-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": capture_metadata,
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": LAYER,
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
        (
            args.fixtures_root
            / "prefill-layer14-complete-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
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
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = f"layer{LAYER}-{name.lower().replace('_', '-')}-final-tile.f32le.bin"
        qkv_tensors.append(
            (
                filename,
                payload,
                tensor(
                    f"layer{LAYER}_{name.lower()}_final_tile",
                    name,
                    "f32",
                    [TILE_ROWS, width],
                    filename,
                    payload,
                    "output" if name == "KVcur" else "intermediate",
                ),
            )
        )
    write_fixture(
        args.fixtures_root,
        "qkv",
        template,
        {
            **common_capture,
            "fresh_process_captures": 4,
            "full_capture_sha256": QKV_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer14-complete-2048",
        },
        [
            {
                "name": "attention-hc-ingress-and-norm",
                "kernel": "HC ingress plus learned norm",
            },
            {
                "name": "q-kv-state",
                "kernel": "Q8 projections plus RoPE and E4M3FN",
            },
        ],
        qkv_tensors,
    )

    compressor_tensors = []
    for hook, (filename, shape, size, identity) in COMPRESSOR.items():
        payload = repeated(args.tail_root, "compressor", hook, size, identity)
        dtype = "i32" if hook == "attn_state_score" else "f32"
        compressor_tensors.append(
            (
                filename,
                payload,
                tensor(
                    f"layer{LAYER}_{hook.lower()}",
                    hook,
                    dtype,
                    shape,
                    filename,
                    payload,
                    "output",
                ),
            )
        )
    write_fixture(
        args.fixtures_root,
        "compressor",
        template,
        {
            **common_capture,
            "fresh_process_captures": 2,
            "full_capture_sha256": {
                hook: spec[3] for hook, spec in COMPRESSOR.items()
            },
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer15-qkv-2048",
        },
        [
            {
                "name": "ratio128-paired-projections",
                "kernel": "kernel_mul_mm_f16_f32",
            },
            {
                "name": "ratio128-pool-and-finalize",
                "kernel": "softmax pool plus norm/RoPE/E4M3FN",
            },
        ],
        compressor_tensors,
    )

    attention_payloads = {
        name: repeated(
            args.tail_root,
            "attention",
            name,
            ROWS * width * 4,
            ATTENTION_SHA256[name],
        )
        for name, width in ATTENTION_WIDTHS.items()
    }
    attention_tensors = []
    for name in ("kqv_out", "kqv_back", "attn_low"):
        width = ATTENTION_WIDTHS[name]
        payload = attention_payloads[name][: width * 4]
        filename = f"layer{LAYER}-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append(
            (
                filename,
                payload,
                tensor(
                    f"layer{LAYER}_{name}_row0",
                    name,
                    "f32",
                    [1, width],
                    filename,
                    payload,
                ),
            )
        )
    attention_payload = attention_payloads["attn_out"]
    attention_filename = f"layer{LAYER}-attention-output.f32le.bin"
    attention_tensors.append(
        (
            attention_filename,
            attention_payload,
            tensor(
                f"layer{LAYER}_attention_output",
                "attn_out",
                "f32",
                [ROWS, 4096],
                attention_filename,
                attention_payload,
                "output",
            ),
        )
    )
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16_384 * 4 :]
    attention_hc_filename = f"layer{LAYER}-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append(
        (
            attention_hc_filename,
            attention_hc,
            tensor(
                f"layer{LAYER}_hc_attn_post_final_tile",
                "hc_attn_post",
                "f32",
                [TILE_ROWS, 16_384],
                attention_hc_filename,
                attention_hc,
                "output",
            ),
        )
    )
    write_fixture(
        args.fixtures_root,
        "attention",
        template,
        {
            **common_capture,
            "fresh_process_captures": 2,
            "full_capture_sha256": ATTENTION_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer15-compressor-2048",
        },
        [
            {
                "name": "dense-mixed-attention",
                "kernel": "kernel_flash_attn_ext_f16_dk512_dv512",
            },
            {
                "name": "attention-output-and-hc-post",
                "kernel": "Q8 output projections plus HC expand4",
            },
        ],
        attention_tensors,
    )

    ffn_tensors = []
    for name, width in FFN_WIDTHS.items():
        payload = repeated(
            args.tail_root,
            "ffn",
            name,
            ROWS * width * 4,
            FFN_SHA256[name],
        )[-TILE_ROWS * width * 4 :]
        dtype = "i32" if name == "ffn_moe_topk" else "f32"
        filename = (
            f"layer{LAYER}-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        )
        ffn_tensors.append(
            (
                filename,
                payload,
                tensor(
                    f"layer{LAYER}_{name}_final_tile",
                    name,
                    dtype,
                    [TILE_ROWS, width],
                    filename,
                    payload,
                    "output" if name == "hc_ffn_post" else "intermediate",
                ),
            )
        )
    write_fixture(
        args.fixtures_root,
        "complete",
        template,
        {
            **common_capture,
            "fresh_process_captures": 2,
            "full_capture_sha256": FFN_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer15-attention-2048",
        },
        [
            {
                "name": "ffn-hc-ingress-and-router",
                "kernel": "biased top-6 batch",
            },
            {
                "name": "routed-shared-experts-and-hc-post",
                "kernel": "IQ2/Q2 routed plus Q8 shared",
            },
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
