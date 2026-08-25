#!/usr/bin/env python3
"""Import repeated full-2K layer-18 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-25T09:01:27Z"
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
    "hc_attn_pre": "ddab0f4518e9cfa7905f88eb3e493783223f71197c491cba1d3bf1a5c2e4504f",
    "attn_norm": "89261086eb6888b6f97aa59eec717cbabfa449d005f586cdc964ddd8397d3171",
    "q_lora": "495fd1c6dc7a44521dbad6a8707d397395051038345363274c6cb47333074337",
    "q_lora_norm": "10c12fbc4888070b8c27de9e43c33fb9d9a258b6e099249b2b0f4e4f5b83ef64",
    "KVraw": "f38587c1e478c637b1bd2d55d1bdb37d37f0d34e7e77c6b4f9887c133b95accc",
    "KVnorm": "d7f19b7d991cab93b777000404ab91ff5aaaf9887b28a8241b347fa9ba03f98d",
    "Qraw": "07c06eb8f471bca50e13800505b44ba0de3cc9ad4efaaa93bc1a8a020908bc1a",
    "Qcur": "354a889f57426cee1ccc5a67aa0275dfdd04754b6724c9f5802294d2d3230817",
    "KVrope": "76ca466722e5da3e5bd11ce76772ca8a00aceced78203b9a5391db7e2fed5889",
    "KVcur": "eccc28e83f6a4b1f251c55f38cbd34223de7e1f93d63c14b1cd8cd3e13091345",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "7bb7e566e4dc48af6de847590b51dcc3ea6540ea3385618e095c887b4b53c3c0"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "9ae8435227076a1a61dc78a6b3d7f62bb02c7235e29af57575e3a45fa3dc6714"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "69f2f758a030b283a7c66cdd9e58685b954ece8a56c0ba75239ee90469db0700"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "83afcd03648bc6b2a8fbe670d61c2d670d31c9db8ad66db10b3b37fafd2c77a8"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "44985505a0aececb348ee4cf8701483db5208555b1dfb563332d17f9f23e914b"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "e9904b06ba5646ffa5234b516182e4d214f5e7052abc92d7ba4919b641181693"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "7f0bcc6390bc255c04314c24c0b9a184ce75164570f0a4625d4ec6e33976ba96",
    "hc_attn_post": "7dabe6f067db721f48f8a1f6cc556669bc280c1dee0b99021ebd0828a5d860bb",
    "kqv_out": "71a3b2c473a647ceb305db066ce5b6edc5430c398df40cea9a12f3c16346dd54",
    "kqv_back": "b46332ee22c72fc61fdd4954f678e3b8f9d404e6ba6eaf4d3a6a058f7ab7b711",
    "attn_low": "391fde0581c74a5f859611e0eb4cf23f3c38eccfc66635c8c56fb7ac659cb88f",
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
    "hc_ffn_pre": "8f7f586c8ca15d508e4506ff6b1c4012a55163b700f122cb5c6a61e1c33fbca9",
    "ffn_norm": "9f87794a079dc0d108dd13c940cb0c6dd745067640aa500fb31cc62236d65ef9",
    "ffn_moe_logits": "80385e164a1f23807454f6ca6a3cdeabad4675dd90a5958df1405190cdd66250",
    "ffn_moe_probs": "cfe48071bea1cc40b6a2a3a8acecfa74248b04f0b4b1e9cd53a728071427d830",
    "ffn_moe_topk": "eefd165bd6b303f2ad02e4d1b837948cfa3486104894368785f6f5c2a0b45959",
    "ffn_moe_weights_scaled": "53d3bedf8af8d8fdefcde2fc8a9e3a4e742ad6eb59bef35f28d84ddfcf827647",
    "ffn_moe_weighted_swiglu": "911bb48b07e7b059e655a847c0ccfefe2412082caadc9600c2e2eb04bed32acd",
    "ffn_moe_out": "0ab3ea4a4979e4e32f62d31d3663a066e2f6202c1373407743101895aeaecfe9",
    "ffn_shexp": "f7466b52ec1578f532b07806261a850e4e884d56fd6106bc64096793d6c8c47c",
    "hc_ffn_post": "2ef9afc801067194bd4a1cb947041dcfef4b4ce1c92ae81063fab38d434c3a25",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-18_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-18 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-18 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-18 {hook} capture identity changed")
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
            "layer": 18,
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
        (args.fixtures_root / "prefill-layer17-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer18-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer18-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer18-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer18-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer18-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer18-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer18-q-raw-final-tile.f32le.bin",
        "Qcur": "layer18-q-current-final-tile.f32le.bin",
        "KVrope": "layer18-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer18-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer18_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer18-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer18-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer17-complete-2048"},
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
            f"layer18_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer18-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer18-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer18-qkv-2048"},
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
        filename = f"layer18-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer18_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer18-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer18_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer18-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer18_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer18-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer18-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer18-compressor-2048"},
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
        filename = f"layer18-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer18_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer18-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer18-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer18-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
