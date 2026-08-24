#!/usr/bin/env python3
"""Import repeated full-2K layer-12 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-24T10:50:37Z"
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
    "hc_attn_pre": "d32de0ad21bdda8469fcdd6d0b1cf8f826a8af2feae84415a921a09f13381a0b",
    "attn_norm": "22bd53f3031aee300f676211f4793c07833ae9448a8fc8c568fbe957559503df",
    "q_lora": "e5109a29594e8860e19c50df68a42089825b45712f598f2f29f0e338a5dc1f3e",
    "q_lora_norm": "fb3407414604b15fa04a100fa4b967c4d1243c12eb2037016d4a394f7b5be636",
    "KVraw": "c7b062a54072018c41fe5e62b4846e994c31a1a51929ffcad7db58f10a33ee2c",
    "KVnorm": "6de19d4910cf4a22600c4df4ed681aaa85e65bd6f3152dbfddcf495c41961184",
    "Qraw": "25ec9571e37412dd28173786a653e89f7cce79b40117a594b1f81e5917f7627a",
    "Qcur": "ca26f32c2f85d9342b22ce969a8aebd662146f92a282e34bb32e67ef9b1baa53",
    "KVrope": "73c89e498074137063b4285f6b82918c1c3ddd4e5df10caea6e869a6569fd512",
    "KVcur": "b7ebb4cdd64fb1918d5811ef56011664a3fa241ec54283ff153963bab1d0087c",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "1153215b81981530959dd068d82049053f1b5521c73d8712779ae80bbf1b3981"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "a94ccb557e1d514406a50c5b62b5ca760955862378bfa31867f8efb243961333"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "3b9ec621d28400fcb76f4c6ac448afabc7afe4526b06fa0b3d29a127e30e3be3"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "29316c6c5a54a6f68ac83e21393313cb0d64958cd9f7ddc00de903c9cf87c247"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "0df5a614f85feea39724d5e110739c4e578fdcce684bab86da6787ebdf1d0205"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "0b7fb521fecc079ffc26fe0b3bb1643d44216f1e7eb9f2391e103326a6cd48df"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "2eb1133303f37610a0e2c62c4d3608da3e19ee799e74eeb75a70cc21d0e0073e",
    "hc_attn_post": "45fa22eef3db34b639fd76910a5e6a6ac8717e7dc7d552b766f5a3a43c6372e2",
    "kqv_out": "eaaa225fc476bbfa4640ef7a201dbaac483c02800a020fa37d31048e84bb4c4f",
    "kqv_back": "3deb9b7927c2e8e8e4dfd0e4ad2e543be9462a5f5bdac5b237714d1480ed8bb2",
    "attn_low": "0fb3e9f3923288604bdd9e28b2fffdf3ebc8b5153ee3d07e06901c587264ed06",
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
    "hc_ffn_pre": "cec2e4d668dae8a3ac50b78ff58333dfdbc7a102e3e98ecf1d31a78700cf7f16",
    "ffn_norm": "8fa1baf44e26d3cabcb7d74081295fa20559e5181c6a8d6f0829e04c526d69bc",
    "ffn_moe_logits": "27ca54a65973d320bf0f69dd82bf4c7d11f8dc7dcd9b78f58d74ad1af7831180",
    "ffn_moe_probs": "d30989bf7e5b6a96ca96fc722cab3f9fddd6cf93ed1075b8ee08dbe20a37df76",
    "ffn_moe_topk": "118c9a3646e374f14bc87a154ddf07761a9b1099b130d7c9004a55051fbbf486",
    "ffn_moe_weights_scaled": "1f707cccbc2495139e25092fa20c6113d1ebe380ff5662d7ed7dde920b9a8a52",
    "ffn_moe_weighted_swiglu": "5a28b7808b552ab58d7d97a50ccb806d24d6a4d170bd9ceb060cf18e40c612e0",
    "ffn_moe_out": "dafdf91e4059b5326685397cff8725eb122ecf0ead6f30a85497b2f16d77d330",
    "ffn_shexp": "ab4de22de03c9e4137267e8908dfe54c8e15124771eadde0ee88a67715c5d408",
    "hc_ffn_post": "b3e73554fcdd72201832fd6520abd22c60bcba863e7e59bdb0daa22caa7267dc",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-12_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-12 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-12 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-12 {hook} capture identity changed")
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
            "layer": 12,
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
        (args.fixtures_root / "prefill-layer11-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer12-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer12-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer12-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer12-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer12-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer12-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer12-q-raw-final-tile.f32le.bin",
        "Qcur": "layer12-q-current-final-tile.f32le.bin",
        "KVrope": "layer12-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer12-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer12_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer12-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer12-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer11-complete-2048"},
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
            f"layer12_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer12-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer12-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer12-qkv-2048"},
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
        filename = f"layer12-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer12_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer12-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer12_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer12-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer12_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer12-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer12-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer12-compressor-2048"},
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
        filename = f"layer12-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer12_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer12-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer12-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer12-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
