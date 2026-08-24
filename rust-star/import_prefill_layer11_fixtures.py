#!/usr/bin/env python3
"""Import repeated full-2K layer-11 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


LAYER = 11
ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-24T09:00:23Z"

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
    "hc_attn_pre": "6a34f5c080c6f629a29daf8800270bc2db6c6c526391940b5cff24392afa93ca",
    "attn_norm": "c10bc3c9456b91583a994a1a734ef7f10789079f8f1e86266eb5248901306d64",
    "q_lora": "7be7711a6c82be5fbbe43335b41e332c6704ccea9306d5815f0d0c96ac31a659",
    "q_lora_norm": "379b1e6594f41c92590bfbe7335de6959bcd46223bc16840964c601df1a9f67b",
    "KVraw": "4c321994c0d1b20af019d6da19212afab9a8b233a56a9dd62922f714f5fa773a",
    "KVnorm": "b557a1f0119d5df61077ddb62cf01590360d0f0d2c412b8ac2321fe4020a6b73",
    "Qraw": "488dd05440a31a911fc623e4d847f076f03466ab47c8a1ae0584116fff87af2f",
    "Qcur": "bfc82916eefe609ea7384c26b20a3c8dd5b11682f55f6c8e95fb0522df753000",
    "KVrope": "829277bae87693a7a8427acd783aeb93876e07436361c2441d6f3b07d1d8fa6a",
    "KVcur": "f43b4a0de448ffa4bf51130541967e7f5657d876ac304139730c8ce3cba68319",
}

COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "7a378df62c3e1c38179c904a98bb804b4ad52640a7b9fde48beb6c5dbad96fd3",
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
    "attn_out": "c4ec85bcad0156b9fd27d8fa1e4dccb63fb2c92ea4b1002a513cafbfbc81831a",
    "hc_attn_post": "7b84d220ae59f33b50463416071f9327706fadeb1071ac2d32e9fc17c21864b3",
    "kqv_out": "91916a4ac4624f32ac77f5ded6fb999609ffbfe07f67492c5451a8511e4fb900",
    "kqv_back": "07a84a7652f3615be878c1780a65845af18b476c5e160a08831799188628f607",
    "attn_low": "860e7295893810e62365b0564b37eb3377370a60309e0e173bbcb5a5f2b53d31",
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
    "hc_ffn_pre": "dd7446607886e13f99867253125a1f3358862a66f1e0406a935c8065db5bc706",
    "ffn_norm": "97323e3cf838ecb0b243870db444807684262f57fefaa290733081c95babc20b",
    "ffn_moe_logits": "1dee3c871e15c1cd7e2938a8fcde751c3e45d7fd1b1023533ccc3d6d3024acdf",
    "ffn_moe_probs": "26e4da9f2c160fcfde619e7345785a4604e1a5e942c114a3151c93b16088fa10",
    "ffn_moe_topk": "f9c1de4c78e47268a30d89abbe26dfac5f35a0abf72ae02e6a980f937f03425b",
    "ffn_moe_weights_scaled": "db5796420642c377c258f2f12a3394860f0c79e51b3fa8e0c74ada29038bbff2",
    "ffn_moe_weighted_swiglu": "5fa73ef8bbc2401329bda49de057324a0e5329c9cb1b62b78390ab84c4329a87",
    "ffn_moe_out": "4f35198a528bddbc8c77c0377495aaa45c2fdce33837fc29ae880593cf5a108a",
    "ffn_shexp": "df0bd1ff87c9a52d8ef0728dd10be4a54fd9af4561ea131b67e0cf8b46b453fb",
    "hc_ffn_post": "99facf656fdeba573f8de2a587830a1f8764be0109620212cd46852adf0e2f84",
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
            / "prefill-layer10-complete-2048-v1"
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer10-complete-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer11-qkv-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer11-compressor-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer11-attention-2048",
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
