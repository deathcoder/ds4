#!/usr/bin/env python3
"""Import repeated full-2K layer-22 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-25T16:40:41Z"
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
    "hc_attn_pre": "e7fa0ae8771c7a2f6c8cd25a301e4c0bcc6484f7a6624c5a773900f4df6b7e1e",
    "attn_norm": "edc9beef89ca49cb68ff7eae08addf1b9ad35bc83fc156952affc03447640665",
    "q_lora": "e2ccde13bee28a9e6edfbc555742cf053d4a0eb4bf31e243f268abab17975ed9",
    "q_lora_norm": "4b48a8d8f60b5386439b7866f8016daf39eec5828157a9e9c5d2989fd1b21719",
    "KVraw": "68ee2ecd971e81c1e2dc1e4f1d3ea3da963139c3b96c0221527f4b6cd6eb60b6",
    "KVnorm": "dd3b35eb59a50de946258832778e4594901e6b9365dd98ed1b989358bea830b9",
    "Qraw": "c38fe2c373b3bf7085cb069cf3c73c463c83fcee43bcbc66d35951c4e6c53713",
    "Qcur": "c5c10eb1096ea1bdd97c7544121b43de4f4fce2961ea6bf905ce0399b59fddb1",
    "KVrope": "1c5834cbe7cb7c1fbd4198bd6f5d41fa594fa9081338f630c3bdc70b8bd7bdab",
    "KVcur": "f93086daaa4369164b9be1a668c744b2807285b3aea36c0f68c522d091ab1216",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "e559e460f04128af254962a84fb6c086e4bfd9d78a12a02a10c5e564a1e613ca"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "b74f92a8201c36219bbcfbbab7c429488abe234095ac1a2cfdc9c1ca76a7c514"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "1189c6a21df63d8d56b4e7ca1b17ef550cb4872cbeaf04afdf7f79071e8b697c"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "0c3d1ae4528ed2998a3ed8567416dbc55f6edf4bdbae0a427ea52f8159b7390c"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "849e3b8f4e96c1dc1c8fa2b587498fa6c19d0d0acefb508440ea9323d0c006c9"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "d585567fc380ceb9e1dc9e93146523a052c23edd2a6df20252946659a8588380"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "21ade0f6fe1ea7ae0cc3d959a89c4624f7c5459eef9ec2c8c440499ead525963",
    "hc_attn_post": "7f6ed3f1741aac51e781dfdb8d0106ca29bb8bf45f7ff393d307c2c356c033eb",
    "kqv_out": "d4288835ecf439700c5de7529275e345e2c82371084bfb88da8f8ab3f215dbc5",
    "kqv_back": "a282c33801433cc2ed2ae802f8bf6da3bb64e5b1d7bb0628958e0c4a392f3846",
    "attn_low": "657d460033e686cdfa755787ec1003faf9c2d9484fbc60c1af1f18f63458ec5e",
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
    "hc_ffn_pre": "4b76d9d5798617db71c1c4084484c36ed1145856133e4ba076cdca3dd4df8557",
    "ffn_norm": "6b373fc531639d78e615bec5fb1c8dca20d284dc3c01ae4ab88101e29b54db00",
    "ffn_moe_logits": "fffecf7a4d95e0ca370fbd98bd5516d2e010a1a7761fdc5fe475a977962cd43a",
    "ffn_moe_probs": "af1f0fd548284095afdf9e604bc946796821c0f980ee18c585548b9a24ce9365",
    "ffn_moe_topk": "206bcc9c39ebd292e0105f9d5a3ad7ddd0ae71933d81a4972c83db37abda29a0",
    "ffn_moe_weights_scaled": "3e55b81354d1fa2f82d06cc40d6b0071c7836d1ec76a60566022686947d477dd",
    "ffn_moe_weighted_swiglu": "572e21e036d8836cec158c78d9e0cfb868f7e6a7f71cd96f808f5ff9312295a3",
    "ffn_moe_out": "5272788acd6aa878fcb65cbaf9a62876005087c1569be8f76a55ea15dc5e884a",
    "ffn_shexp": "14a36250e53fba36e4585a78c62d0c64cef79c9cdc9feb34bd9cafc97f5fed3b",
    "hc_ffn_post": "d6dd0e8480d503dae52730f283ac34e88c57ba3b6b5e4eb8f8f5e7ea7e45c7ce",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-22_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-22 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-22 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-22 {hook} capture identity changed")
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
            "layer": 22,
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
        (args.fixtures_root / "prefill-layer21-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer22-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer22-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer22-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer22-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer22-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer22-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer22-q-raw-final-tile.f32le.bin",
        "Qcur": "layer22-q-current-final-tile.f32le.bin",
        "KVrope": "layer22-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer22-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer22_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer22-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer22-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer21-complete-2048"},
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
            f"layer22_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer22-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer22-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer22-qkv-2048"},
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
        filename = f"layer22-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer22_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer22-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer22_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer22-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer22_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer22-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer22-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer22-compressor-2048"},
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
        filename = f"layer22-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer22_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer22-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer22-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer22-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
