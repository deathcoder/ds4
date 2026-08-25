#!/usr/bin/env python3
"""Import repeated full-2K layer-20 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-25T12:18:01Z"
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
    "hc_attn_pre": "672c6b6e5a0d4f6e3aa550f246180f5e58b23880072f40c430a47a7bb6cc6bd0",
    "attn_norm": "840414e6e929fecb5df0e0749e63c7485b4bc6feba2ded2730cef11979c38bac",
    "q_lora": "cf5f4fbf7e7887704f7ac3c38690a9bd2d822763722cc39b29e99f810b149914",
    "q_lora_norm": "c70dda3c9a68f2c31af04d8e95cba4ae5a610b1c68728495d19bc1d5b2a53eec",
    "KVraw": "4584598222d8714bd93ac1c010e90c0bb7dc31b5f8574b74b78451e6fb0e55f3",
    "KVnorm": "95bdbc6fb9ee80f7fd4df1568f07801c58d63dcf92b6f630e3e3399f2075f640",
    "Qraw": "ac72e352d7fb4b55878c33665c2cfb2e29b963438ebd119da320a31a36764a1c",
    "Qcur": "4b2516f23fe5d28dfcc354ac57b70cfd591d68187afd51f2f0a3d6aed899cd1b",
    "KVrope": "d3f71f87011fcfd9674b636fa8134b793b0c5456819fed611523c0c282dda593",
    "KVcur": "dd9ed4f2a458219068bb2d261d37b9a45b1f6573f88437990b8eed32377f2d25",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "a300ef416ba80154c6114c2266705d4854b94fd97c11c2530d3c804d55bfed76"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "557897d75aa6a06d4f740816e8c31b510ced6d656d516de089dbaa8bfa1998d4"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "b8da44ab6b03c1f352ab4374b2e4bd1ffc686c1404cc489b80a6912180ba3704"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "21958cc24cb795808e7da6ccf02265870871ec57a1eccb008c2d5510c585b697"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "a56f8d210a920961c20cdafc12929fac4e5d562fbfea00de7312773176977b99"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "07ecf5f666606303164491ffa2c638882fb6e9fde48244a0f3814d4a6973cfa5"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "32315546aa8f4b88e67118cae5157b6bd8a686b2d7da67ebaaeeda7f22da451f",
    "hc_attn_post": "a28decea95f02fbb124713fc72aa0cb6d165e77cd27c3dd0901ec83c87d39a2e",
    "kqv_out": "3db8b35bb5fc16dbb3eeb5870b63230bca58a5f7ea1efec6f0825b7ba83bf478",
    "kqv_back": "ab71e8064a827f0ee14772ad2fdcc610a118396522cfd301feee05ebc38ec3aa",
    "attn_low": "ed22c6efe3a76bca10d45834e3da82eebf66ca9309f3ff7958595205337b8723",
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
    "hc_ffn_pre": "e7ff7d6de0916f423d6278d2503af76727cfc9d56e7c6d2b9eb277e7a527865a",
    "ffn_norm": "ce7a1100e580bf02199d0628c8a0caf6f659c003abeae8b0225e454c9a1f5326",
    "ffn_moe_logits": "01c93ddbae5c5eb6fe92c4a4182bd3eee9b5f025515849f36d8b9c50afcc6187",
    "ffn_moe_probs": "80ab293dfdc2996cd74f6b4ba07a863158f992548ab9544caec2063bcce42cf1",
    "ffn_moe_topk": "f9ef0eb0dda1207b99d270f465c97a6c7c3ac9cec62e1fe7649f7c429968f623",
    "ffn_moe_weights_scaled": "326a6135ed5a3c70eb374b7938df8cb53bf0c7fe5e05bec933aa0a8e438edf20",
    "ffn_moe_weighted_swiglu": "4837991bb06edfeaabcffa72879b30510c917854fd51ced2cd831d996069d5ef",
    "ffn_moe_out": "aa942358a86205e7a22aa04a9167d3bd136e3df5dc12eb554c3ce639ed43fe7f",
    "ffn_shexp": "6d4bf2dd6bb380a12d2cb10caaddd3c432ea8e41cbfc8374bf1b6241005fc844",
    "hc_ffn_post": "af93fcfccd785f24cb26496c827904a9d366df2b823845691fa93ba940e37c64",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-20_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-20 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-20 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-20 {hook} capture identity changed")
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
            "layer": 20,
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
        (args.fixtures_root / "prefill-layer19-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer20-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer20-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer20-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer20-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer20-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer20-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer20-q-raw-final-tile.f32le.bin",
        "Qcur": "layer20-q-current-final-tile.f32le.bin",
        "KVrope": "layer20-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer20-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer20_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer20-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer20-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer19-complete-2048"},
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
            f"layer20_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer20-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer20-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer20-qkv-2048"},
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
        filename = f"layer20-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer20_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer20-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer20_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer20-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer20_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer20-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer20-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer20-compressor-2048"},
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
        filename = f"layer20-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer20_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer20-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer20-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer20-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
