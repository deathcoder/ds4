#!/usr/bin/env python3
"""Import repeated full-2K layer-26 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-26T05:27:23Z"
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
    "hc_attn_pre": "39ff9895bee6058731b190b81edc3346cbfe48c5d99fb3bbadb8a86c8eb07d74",
    "attn_norm": "48228ebe8d4bea4422654bc3f551bf9e0b1f33c61d3a84a07e1578633664828a",
    "q_lora": "7277dae2f98cfda738a1d73e9d793bf162f37f187a7d063d17021e055a345562",
    "q_lora_norm": "93419a2264095d40a6483938138cc0dd24b70c340ad7a40976a92ea46148bea7",
    "KVraw": "3dd73d2aaa5dfa2aa08c390820d2b5973e8693dea4a04b8c97a1cf11d44d86bf",
    "KVnorm": "306fdb5d5f0e5c6075ad1a0030c90a5cb59da504aa55c4f97ea96f34eeab8b43",
    "Qraw": "5157c94d7f3080e8b3d0ccfd70fb0fa877f1ca36d08a5b5977cbf8d22388a47b",
    "Qcur": "430c308a0f9b0b1a9cbcaec5342a9495b9a703f77da93f2e723d57de6343a6a5",
    "KVrope": "d6d9ae2c8807c88e3ff0864802aaf71cab77c5a54aa3373c6c058660a2458394",
    "KVcur": "429bf1dfe2eac9d51adaad5fd1ffbeaeda49aac641a8b91bccbb6864d149bcba",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "123ca48de9dcdc62d13f137bb87db3368b41d042198ae982a2aeb5e3a40e9e7d"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "b72e945c3af2cf98dc65c5c225ea15208c98b4fccb919a51cf1d75438072dd18"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "130496fecd213b0f3b10095f2798daffdcee01362aae234e4f65416d14a78ced"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "2b21c4ebf8c010fb59a02dd2941f7807f7c8cde47a0723ebdf5d49470b0f39c4"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "8b730f557d3d00107dc9685d514184ae23253b380b4b3703ea650e6c5b7f7aa8"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "106dafb306d972b4497bfa93e830d2f2d921427a9288b1f6da3a58b1428a376c"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "ff44cc0f116a889ee9ad4ca12af793ee0f56219b155f00d4628fbe88938bb493",
    "hc_attn_post": "4e6fb63960c05922c94b0e3d57fce047293f384067d58ebc0389cd2cabcb9ca5",
    "kqv_out": "94e3a8e8b1649c7866276c297d010faadd9da7ef4acbdbf64b84a0a4fdaf035a",
    "kqv_back": "2d41001c8c1d1f2f837edd359482acd85bc60dc6e7afb7bd8e471049bdac5487",
    "attn_low": "ace80ea939f6cdbcac05fc16e342ad6077b1963aed8261e4fc8b4fea11047071",
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
    "hc_ffn_pre": "17e4571ee8a02684e1ab6872294207edf16e37f1a7a1d775015ac8433cb28afa",
    "ffn_norm": "858a05df4bbde1623e7a4138026094478b6d216d2b08b5b5cfb855c14123520c",
    "ffn_moe_logits": "a0cee58ae53b8d054cb36b836247f4714e091abd2bf5462405d6811e4785782d",
    "ffn_moe_probs": "998c833bc5f333221921843a1bc62c045364948382fc89cf2a230f7f7fd37215",
    "ffn_moe_topk": "2a8547eb5157d5760cbc5b927e8d9e4a2a3b1a0709e563724ec4170700633632",
    "ffn_moe_weights_scaled": "790e656c7d74ee974af340d15c68196870efe1e15f27b40535ff6c68dae87d15",
    "ffn_moe_weighted_swiglu": "be8b20ec3198ad1b51daa124e061eaebbc44cc2d697b9a28cb49280eec8648b5",
    "ffn_moe_out": "348007408a12e4da0db10542905b7ea0ded8efe2d9fc3fd055f7a9da3b4496a7",
    "ffn_shexp": "3623dcd7f9e5faa9b27dbb99b1ca850e021f447030ae639f21523bc558716cfc",
    "hc_ffn_post": "e2363df8d98c09eeea5c0ecdbc6f96763f64c2a50fe1bb92a4b8b2490587ae9c",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-26_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-26 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-26 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-26 {hook} capture identity changed")
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
            "layer": 26,
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
        (args.fixtures_root / "prefill-layer25-complete-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
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
    qkv_filenames = {
        "hc_attn_pre": "layer26-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer26-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer26-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer26-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer26-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer26-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer26-q-raw-final-tile.f32le.bin",
        "Qcur": "layer26-q-current-final-tile.f32le.bin",
        "KVrope": "layer26-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer26-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer26_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer26-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer26-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer25-complete-2048"},
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
            f"layer26_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer26-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer26-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer26-qkv-2048"},
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
        filename = f"layer26-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer26_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer26-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer26_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer26-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer26_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer26-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer26-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer26-compressor-2048"},
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
        filename = f"layer26-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer26_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer26-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer26-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer26-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
