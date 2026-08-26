#!/usr/bin/env python3
"""Import repeated full-2K layer-27 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


LAYER = 27
ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-26T06:57:06Z"
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
    "hc_attn_pre": "1a92a1739d1a42e3b958ac81c495863961563ea2b1c37f202f44ca9121708a1a",
    "attn_norm": "c1939f7939249f495c4610d1bea186674e420910416cd237dbdda9cb10b9aa35",
    "q_lora": "700086476acae092f7e8352b4efc566b3c8b4555d315e1624b84532d25215c61",
    "q_lora_norm": "250836a2f66c686e2412b2398989de7f0ddd5a50758f08261a91b9dcaa092406",
    "KVraw": "ec042d396db86f2cab2f5621ba9c4f186da3a52e7210dcc0a1fa9e88cfed13aa",
    "KVnorm": "35ccc4da7fde6b9689dc51eaf9d70f9521fbe5013b54971830216b7b60b27d89",
    "Qraw": "644cfad7e949ee3da484e056ffa586685cc903f592ba29c1eb10de2c300f6918",
    "Qcur": "6085888d8fbdf3450a3137ab6c3a42ac754b02f2ccd5acfcbd6a7bfe8eeefad7",
    "KVrope": "b5aef2efd347d50e18b8bc4cb1d607ab65acb8ba045040c7449f6d0b9e43af43",
    "KVcur": "554a2a7a62e59ffe48b29c1756f3c44c82a29f54f359378580a402498d20a212",
}

COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "321652f1555b7f8c910872644d92f2286e736815a13a0f70f97f39921444e7b4",
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
    "attn_out": "10e6bb4dd390ebbeeb962df285df4750ba1a1fe92eb86c728819448c5ef6d2a3",
    "hc_attn_post": "404e4c51216e2af27db0d975fcbe42476d495cc84f76814baeec76de497bbdd2",
    "kqv_out": "8478931a95c1feb5314f976d6d8c53e32d02af16c8b13978f8efca4c7fcb1239",
    "kqv_back": "98fb9b47c3d7022dad0d1f72c4644a221402a43fb871b2e88f8f1c4fae8e6482",
    "attn_low": "867fbb67d7d495573914386c7fbb00e6d34f7cf6d34e7f9f8b1e1b281514fb98",
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
    "hc_ffn_pre": "9f91842eb421f741a80dc61ac7839bfe37b83052a2f0657f27312c4eff415c9a",
    "ffn_norm": "42232b3f9c8b118458f24ff4e1b4924cc66db43f317425f1d4ea379251381510",
    "ffn_moe_logits": "16c794036e7c4d6670a4304da2d76e08bc469411bec1f4457e5ecc8cf1ed248c",
    "ffn_moe_probs": "386d4571d05d76471e9abf933a2d68fdee502e3ddce2b596b17cdbb644310b3b",
    "ffn_moe_topk": "ac582d705548cf124e230780c9c378dae43c53671332a008a027d455fde883ce",
    "ffn_moe_weights_scaled": "98f401725ef7dd629ba7e1811d0889db2b47981284672396f72b4d3653ba6bf8",
    "ffn_moe_weighted_swiglu": "7919df28125df46cbd98c11607711f8cba6393706f62a26447bda9507109861b",
    "ffn_moe_out": "848919ec8c3e588613d11f4148c361e76f37a3cfcfbb92eefbc53cc99907501d",
    "ffn_shexp": "b11234de50001c6aae10f7b471e19f629298caa6c612aa12e2ac2f1910189c00",
    "hc_ffn_post": "55bbc00e374c0acfc11fb5b431061de4baf465bc11f25b3b7205ff31dbea4e8c",
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
            / "prefill-layer26-complete-2048-v1"
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer26-complete-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer27-qkv-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer27-compressor-2048",
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
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer27-attention-2048",
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
