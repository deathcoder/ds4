#!/usr/bin/env python3
"""Import repeated full-2K layer-30 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-26T13:41:03Z"
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
    "hc_attn_pre": "784da3a80496901473409ecb8bb97028a16dd3b526957a82ad51616312db1dd1",
    "attn_norm": "b81b774128003f556a76205466cb06760647d6e7e1a2cc1540c81fc5358c24c7",
    "q_lora": "9a754ed4d877650c18c45eea973581639623b8098f4a54b5aea80925b9974aad",
    "q_lora_norm": "ef665e7709d98d24cf5939bc176f79870144639a27398e93f7326a42ae1270fd",
    "KVraw": "18ca28104b0677827c25069ba4fbd257ca1987b9d4e1ab922283231bc65ed5af",
    "KVnorm": "32279e45479423671f159cb672158790147534cb0f1055404424cdd3c418d3ab",
    "Qraw": "973430d77e3ceae097cd4ecc5efb649865b773c58706487c82e1d703b4eec0c5",
    "Qcur": "f2c0bc385d1da9825d59b6d6c6dcf30b006ee57434b244071da8d78eb78eabf6",
    "KVrope": "ff32a94a3fe00359566e01b7e1975a0b2144b2b0de6a271cc4434ffd425476a5",
    "KVcur": "3e29952a6228078bd1c62d34674751a7b394da2d48e42fddd1804ec0facb1905",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "1831cc3ab14d957db130030e0441bd1104fe2627c8327b86bb88ab320e613426"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "957eb99de78a481bcd62a395fb26df03f99fe842ab7e1b95000e572fec3cfc9a"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "a993ac44c3ef507b3ff4057192f5fd76b017200c2a07396d60fa5762b53c6ed2"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "db48c1d925e4189b9ff8b8b1ceeafe5d7f25d36662625fae66fbbbf3094cdb42"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "317c94fb3321941f5af3dfdf36eb8ff6c085b6f00623ce9be9d28d0dbfb6cc20"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "9b1a17a6daff3a1201d72a94deefd26fced96b87f2d8f452ec1098643bbea47a"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "fad57fefaf484ce92a05ce877da5bf69a4f466c8f339b870680c7b5e57f19694",
    "hc_attn_post": "cc9df4b500ddb696408b7e257902fcac4474beb408f0569ab42b453d61d33725",
    "kqv_out": "332d2d91c545a8c03044e36ba945499825d4b76a54f41217b3c96852b34576e4",
    "kqv_back": "bab5abd5318036724a86e04e059656c0586fb9a6703a5e221c1f6c12fd3996ea",
    "attn_low": "9e573d7e7c6dac2ec3972066708b7b4273f0340c93f531a640abe8a93dbaff77",
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
    "hc_ffn_pre": "d546323a37f6f2e1f185e79e9c326b486c6de655c35c4e9eed8a548d2338da3f",
    "ffn_norm": "a2fd91aac95aea74984bea77503167795230bdfc316b6461855896b031380e42",
    "ffn_moe_logits": "7ee0eef7e4007de17e4753ab460eff07c70b7828e2e94efbb5c005c1e1b7c14b",
    "ffn_moe_probs": "f360bb068aadc3d31c2ba19608004fb98c59b46fe3141439f238a06ce4640dfd",
    "ffn_moe_topk": "c5470f2a2ff54d1eef850f1fcb48d76e859f33d6549a0cb3ddf80cac8374c332",
    "ffn_moe_weights_scaled": "6d61848a0f635dc47ac8e12ffb24a76f0c5de8c63352194a3cca0017b0e3c933",
    "ffn_moe_weighted_swiglu": "48f83771beb2f687e5829f5b1dc7cf9aae60cea962e90a8acb315ad7e9164bea",
    "ffn_moe_out": "36032e7cf9d7a22cde69456d4061fe890f6b2960f394479eeccbc07360d8f194",
    "ffn_shexp": "bf9a5feeeb33107f8ef52604beef06dae8b600853ebca2e4047063c17ea1f454",
    "hc_ffn_post": "f2559a72e2ae2568e63bcf88651a954a1124b07f0e0be6d2777accfb3879aaa3",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-30_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-30 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-30 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-30 {hook} capture identity changed")
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
            "layer": 30,
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
        (args.fixtures_root / "prefill-layer29-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer30-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer30-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer30-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer30-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer30-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer30-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer30-q-raw-final-tile.f32le.bin",
        "Qcur": "layer30-q-current-final-tile.f32le.bin",
        "KVrope": "layer30-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer30-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer30_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer30-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer30-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer29-complete-2048"},
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
            f"layer30_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer30-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer30-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer30-qkv-2048"},
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
        filename = f"layer30-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer30_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer30-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer30_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer30-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer30_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer30-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer30-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer30-compressor-2048"},
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
        filename = f"layer30-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer30_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer30-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer30-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer30-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
