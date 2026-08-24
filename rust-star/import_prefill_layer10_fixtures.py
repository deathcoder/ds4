#!/usr/bin/env python3
"""Import repeated full-2K layer-10 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-24T05:54:52Z"
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
    "hc_attn_pre": "ebc1cad138ef00f392b60e6842cfa6cc93979d85e118430bd7695f44e9d6cdcc",
    "attn_norm": "f25a2d1d7eba95195080b655d3a31e140a3e0b25d0209804b3d4fb51396d42b7",
    "q_lora": "ab04cc79fb80b792c9c7546a7bf7b20fbd8eaba50bb504b76fef830a399f04f3",
    "q_lora_norm": "e3d7457a9360e4eb5016757457ba8ca78cae27ff2e732221d76c18593f30f113",
    "KVraw": "1bc40e5e3a7c8c1a2bee83eff7ca45b0e549205503cc0906b6eb6763ef508f37",
    "KVnorm": "3c0eb597786d644bb6c9050476a135490a50161728fd5e2ac3b14cb6dd4e4b18",
    "Qraw": "b3360cbf64c24433e298993947ec0144e815e7e5d4acc4fb77cc805d53327202",
    "Qcur": "14ce76bfd6a2eb7c6b8676daf13c6a4c03c3771de1a847390a94a1e9de847109",
    "KVrope": "793f6f0f7d04a4319fa70b36d5cc21cc2e4923eb67314a8bc3609cb1d49e4ae2",
    "KVcur": "8bb8accf4e60c15b261f7a8df3957414a81ab43a88b7d08df992251a14878f30",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "4cf5bdbe6938fa79ff387d1072ae3a7ca3766aee8611ff7abf040c41c0c9586e"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "6fa80f4772166ae4f4c9b792d35bf68cf36b4a7a067a519cf47e758d64d989aa"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "8de2fc7bd809f6139ce340cfe01da39f5ae6b8218d26f42e160b65da0d69fb43"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "21c7887fe553dd85f95e43b5e57e69e4c24dc8586b4421be2b596a6340bef9d0"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "bbe5d5a956ca961fbe51e96e0c74ef01321436cb8f64cb345aa20a5c563c93c7"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "3f2b3a87de0d9082775e39accd9818184fab2b154e781938e82f6ed7f923d259"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "654b98992b3eb3a277ddf8a3d4123137833200551688ee2029f1ce968ee6f55f",
    "hc_attn_post": "c5925f2d8b7e6e1ca916b4f842a0d6072acfbbbcc85915173541e3163c9e5e57",
    "kqv_out": "d482ad736b7289feb5f742deedf5de965486955ffd6a5decbff8fced9788f29b",
    "kqv_back": "e08576b3b6733dc17a73bb483022319c3c2f1518a7fde3e850e3286e3269d1e6",
    "attn_low": "6994ef3b843f36bcdac5d6ec9299ff239aa502f8420112438159d9db5eaebb2b",
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
    "hc_ffn_pre": "d562d0206cfeb41b2019b47c02e3d1937ab4e3dd050ead191ad15427e32817b9",
    "ffn_norm": "c8608df606935a7fcecce91643639e634ac70ece0aaeb37b9f82490e0e4d5f40",
    "ffn_moe_logits": "c0a92657c9305092caa320c9ac1f29ada03ee38b3dd2c5dbc1495f44a8e9e88e",
    "ffn_moe_probs": "70c83b4229b4880c153a69d6966c3c03e732aa03cf31b074f800a34716bdd56e",
    "ffn_moe_topk": "955435b7ffcb01478bc8eadcf579ac72fd9df52a5992ae4bb40775438abde9a1",
    "ffn_moe_weights_scaled": "b5e1e9b331f2a456f2e910faa24171a25bd1f9188246a2d7d505c28deffd24b7",
    "ffn_moe_weighted_swiglu": "9923783db4889895389bb72944bdb9ad07a54531c19c53dd46a40f6ca67f6bb7",
    "ffn_moe_out": "fbc302689836909c61595b165155515b94d26d6be66df2a2980666f2c1fb25f8",
    "ffn_shexp": "a2a532da2912b7597d64580f18b80632d7ff9f5db9c08699cb5b5b7d31ac5b08",
    "hc_ffn_post": "e020c92b574a9e2e283a07f2650a8d74153e1378c3adb97997fad1ae535d7083",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-10_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-10 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-10 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-10 {hook} capture identity changed")
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
            "layer": 10,
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
        (args.fixtures_root / "prefill-layer9-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer10-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer10-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer10-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer10-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer10-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer10-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer10-q-raw-final-tile.f32le.bin",
        "Qcur": "layer10-q-current-final-tile.f32le.bin",
        "KVrope": "layer10-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer10-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer10_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer10-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer10-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer9-complete-2048"},
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
            f"layer10_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer10-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer10-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer10-qkv-2048"},
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
        filename = f"layer10-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer10_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer10-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer10_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer10-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer10_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer10-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer10-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer10-compressor-2048"},
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
        filename = f"layer10-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer10_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer10-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer10-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer10-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
