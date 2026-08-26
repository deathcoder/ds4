#!/usr/bin/env python3
"""Import repeated full-2K layer-25 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


LAYER = 25
ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-25T19:29:55Z"
CAPTURE_EXECUTABLE_SHA256 = (
    "55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51"
)

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
    "hc_attn_pre": "392f79e88127ad97d32a02157ffb414bca3c2ac0b96e3e91b33f1731617f2eee",
    "attn_norm": "8c9a4d869b6899257882f9723722f3670732ae65445ba526c68264ff8810fff5",
    "q_lora": "1f79c093da42faa246fa235573dc60fa01e60cf3647e5678e153aefc87538928",
    "q_lora_norm": "cb3689725cdcd6b43d1ad3c7c3cfb17ee3b4ae6f9627393be024bdb50b3b996e",
    "KVraw": "b9ed9c02c2d084b69e6f45bbe71f9d85e3e2f1057fa1b75f8f3071e3209fa60c",
    "KVnorm": "d859fff5854150d5c6acf3716aedcb0facfd65b5a4ce69ef5732689ecc8aa3b5",
    "Qraw": "0f8a162ddbbefee9c669122b8e63c2ed74913aa0bb77b25fa4876ee881f3fc27",
    "Qcur": "89aa82b6649ceab239521ac9337bc924198ee965a481ffbef89031415a62d84b",
    "KVrope": "2e4b4afd937b372d6407bfd991160887d2db52d7c379e10de2337845622acd7c",
    "KVcur": "e49bb69c4015d4e2ea57cb4159b85b82b41136f8df7338d2550e56afc08d810f",
}

COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "9da2832888932227f3daf84b211a9b14c9edb5f96b4506a03ca5e83431f5242c",
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
    "attn_out": "311286dee550083a0501f408c499b3d645d6bfc0ce1533f3b4ccc91dc0605eb6",
    "hc_attn_post": "f218e113166bb5e105d767a411548433c40179816f35fdd88713d6ab7cf6306f",
    "kqv_out": "e8391e6da368e0ae79ef8adeb95b7282ba20c6ca63703202f545824743a6efee",
    "kqv_back": "94b3fd0db1e6b6e7c38fc84cc072337095816f263cb89603e63a1d9967b870e6",
    "attn_low": "018999a4645409a325ba63e9cd3dc4a0c32d712f47af90753041ff64b8ac17de",
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
    "hc_ffn_pre": "c38fe0019813df0bc9dde2eb2c81a61b0ddd8050edb7c4d53f795176f9a48c74",
    "ffn_norm": "c3bfab9055bfd611600f2ff4261a03568b8f161c508b3443a79b3826d3a00ac4",
    "ffn_moe_logits": "ad5273078e5626171467c107803f7258a5c521e80684fb2be06eeff1a4bf054e",
    "ffn_moe_probs": "8010abf2e160d343ab81b04e8250ddd4f828e7def1c950b5bd389d2ed272d03d",
    "ffn_moe_topk": "0e1011193c0bb3c73438f53c55dccb4551fd71ddb1d8dc5b6c2807606f499eaa",
    "ffn_moe_weights_scaled": "945567147dd928b2e8dc9d91d6d92f09660a4f342167dcb121486f4c8a56571f",
    "ffn_moe_weighted_swiglu": "45b16bb686160060248ab450fb8c803ca8efb83f00684d1b11743feb2f4d4ea9",
    "ffn_moe_out": "787fe52465c23a7139f2b821a72d6673b01c80f841d03d90a5d2683e020b6323",
    "ffn_shexp": "fe9f595bbdaf41bc50cc53acc60b6aa46eac21e0026e14b2b8893cdab6113445",
    "hc_ffn_post": "b9d2ef4bd9b85eb1e0f728683827ca44b7e73fc2123ed9acc17aacb3010450fa",
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
            / "prefill-layer24-complete-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    template["oracle"]["capture_executable_sha256"] = CAPTURE_EXECUTABLE_SHA256
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer24-complete-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer25-qkv-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer25-compressor-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer25-attention-2048",
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
