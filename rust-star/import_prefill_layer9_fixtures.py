#!/usr/bin/env python3
"""Import repeated full-2K layer-9 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


LAYER = 9
ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-24T05:07:04Z"

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
    "hc_attn_pre": "eccac6c3e71ad251702a22d409ab729d19cee9b48a3880d8a76eafa2ae26a317",
    "attn_norm": "bfb731da80cea839a660141496759c261ffee3af8e1982ea0f167a6ed4099e7c",
    "q_lora": "00e33dfabe162fd28b748143b43b3a5342a81842c2a9104a9ae0484d0b8d14ee",
    "q_lora_norm": "3facddb22397f0c6419ff455222d794ef2c778bc395910d87ee48d6549de1949",
    "KVraw": "5801b61c3e790b52df9f43a22a168f38184898d3c1be61d7516520e111f446a0",
    "KVnorm": "f8c25f239b765b21544e77d83001a777263ceb5e643d87b7cc8f1c57ed6f90a0",
    "Qraw": "1c34c04d5a8839e2eedab7830c17cefb313f86a6d0fadf8ea22b8e619dfca1d5",
    "Qcur": "c68ef18f97569f0ab9b8a462d137372a2ed7ce74d18d5b2040f617a09eef42e3",
    "KVrope": "26453a1279dd764905cf374ac97bc626bbb8e42758ee65fa831c47749734505d",
    "KVcur": "ff6832117f1e2de8c63d25c5ad59009e15b44e113678b3521206f818f50b3860",
}

COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "3b50b6cbaf0caa9974056a7a1e092d3a123845c2e9716abd743571eecc5c3129",
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
    "attn_out": "045ee049d4bc0a012deca1cf74ed6ba8442835146d4e506b4f7ad0021d9e6de1",
    "hc_attn_post": "1d9cb8851a8d9684d81763732a110b334596f5d9b49acfea4c4b815b39afc148",
    "kqv_out": "29022a23a8d70cb5fa1e7098b348cef677f56b358644e0f3cdae38aebb8297be",
    "kqv_back": "fc4c7dea5d90bea637fb495ebd96a8c272f2cf60a27c70525248f8bb25bb2215",
    "attn_low": "693aadfa689e759de7993415d71d3c8c395ca3eb69842e0cb822b4bcfeb515ee",
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
    "hc_ffn_pre": "d93dcbcea97a7ada6743bbe8065c5c0722580c1196aad03d8499747732744942",
    "ffn_norm": "261f65faac0d76895b16ef75a689cd7663aba64a7c0cd97ab964c7efd52ab1e3",
    "ffn_moe_logits": "257a26edfe96829ea164eadc63d0d7d126080cd286c5675ad7c76dd4fb7409d2",
    "ffn_moe_probs": "4a75d3d5359ec4ce5f6a8f9f80ae0b9c02d40167a5306aa0aa825471e671069d",
    "ffn_moe_topk": "b8753713ffe7970458e80b4b4cb11c11e6c3fb6e7ebb84bd820860f10a6ed4b2",
    "ffn_moe_weights_scaled": "e05af6f6d93efff0e9ab38a95ab4086cf516919f75b093a66d5c569c715e1801",
    "ffn_moe_weighted_swiglu": "3c2e0ba9f35112fbf4b24973178e90611f4ba0ae4778302aee5985237cb0503f",
    "ffn_moe_out": "15f118a46c98fb2a41f17af5f7496d04d55f292002b33c51c0a1690156e1b606",
    "ffn_shexp": "8fdf99a27d0de25be2a1f3f95a722199dd89b61173544a77e5b156c892d3b154",
    "hc_ffn_post": "d7004f39cf7c011b24c128ceb91172136eb515655261db2784514d350a707af6",
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
            / "prefill-layer8-complete-2048-v1"
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer8-complete-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer9-qkv-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer9-compressor-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer9-attention-2048",
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
