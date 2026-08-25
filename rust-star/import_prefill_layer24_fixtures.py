#!/usr/bin/env python3
"""Import repeated full-2K layer-24 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-25T18:38:17Z"
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
    "hc_attn_pre": "79909cdcd612f86148082fdd49486e596224a9ece7d3c44370ff72282242dd88",
    "attn_norm": "090deb43bb11c314d2532c8a8759eb1b86e0491c0360ceafad5b2d53c26c27c4",
    "q_lora": "c866eb4bf325b5904c61b994afb8ad130fc9045ad60522239fc302d3d3bb9d99",
    "q_lora_norm": "7b8fb417e1b944172fea81640939fa88dacfe7b5e8675b5574ef7916f849aa99",
    "KVraw": "eb9edcba0c8387779a7d4fa0d52022e56f0face11267aad2a341b225ad7d1ca6",
    "KVnorm": "8e735b693e12ea0a72eb78f6404df3a154ec28e53e15bc64f8e26af5e0ca6ecd",
    "Qraw": "a2b44573a0334a2451ef5c079da1af6db8563f3f458fe86cd1847e6abd324c52",
    "Qcur": "98d6053e6ffcc32137507537290d0c249318faf41e0de4faaf3a6cc12df29be2",
    "KVrope": "d7fdbbba15c00fccc11bf33fa9d5ff6a5b1054464fdf70b82d06563d3f782794",
    "KVcur": "b5044910135dac64e8b7b5fe2b1394b446512d67ce5b5ed7d72f10d47d2f2bf3",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "206640c435562ce7da443eb289f30bf894cb80ce7bdb5cbf607bb34505d6aeb7"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "5424656820ad1e4eb01fd4fbe446855b6a4669df6c2149bc16db429bf5790c0e"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "3b8caeffa4bedc1709ea48ca8bb2364abbf0c98e89667c1219d9db91c426af60"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "cf8a5734748603595e1aefeb4b92394a654caa6ad54e131b4cbefd1a748044d3"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "ab7487cb14f1139d544b8b41016d08119202d5b9daccf967ed574443e2ab00e3"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "b385000c6fdc47adb54c12a1964c6e8dd5c927719f411ba441db661009191ad9"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "079bc4c52807a11612d7e7e8cf48c7372305414242948882ae95dade67263ad0",
    "hc_attn_post": "800844bf8786b362c6afa62085be1a10bd3c8c41afed008e99c5d8c5d3a8404d",
    "kqv_out": "738e05a80373e368ea937832cfb2d61169752b760b52222e6ccfa9ef33a194b9",
    "kqv_back": "6d32e04e2ff3d41b79ee3377e7c84f07cbb42e4461678650b6e296e1055467b0",
    "attn_low": "5649807a715a3830bf290b9d796390cbc32adf29dca4eed91526ed8417a3355c",
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
    "hc_ffn_pre": "17fbd523a35164b036d41eb027c5910b728b5454d570ec7812bb9a156b96ce94",
    "ffn_norm": "853197c2498b2c183250219a01e57d3a2a42dc46510dc78fed40cdf4a722273c",
    "ffn_moe_logits": "f7bebb3140c130f05d50a43cf55468185ea94db4decf48fb0e4d1efc1485479f",
    "ffn_moe_probs": "0c37b208a5510911c4002308ac27feb4bdb39c068397069b659fd4d450bcd7e2",
    "ffn_moe_topk": "36add37e83dedcb56f506292038c1d87b6799ba5fcc8845df787f95d0b513ab2",
    "ffn_moe_weights_scaled": "454d970e15ad967c0c504f57bd7908ff6943fdf4396fbea0d6c7955c14c5f941",
    "ffn_moe_weighted_swiglu": "5970c0a59d41c7324ff49b89b3bd2b93c0ee1d326aca115dd2aead88f551f3b8",
    "ffn_moe_out": "05405c2b6fc463aed02d932c48526410ec3482d4b3a58c5615990caab6a7d21c",
    "ffn_shexp": "e8ca6f15fc3604f2940d763ed0338c8de9e2026b0e0f99bf035aa84f2e726af0",
    "hc_ffn_post": "62f80d35a46ec16a937ecbd7c58358578ee864f1eeceb2b042ea71b752842f9f",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-24_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-24 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-24 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-24 {hook} capture identity changed")
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
            "layer": 24,
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
        (args.fixtures_root / "prefill-layer23-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer24-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer24-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer24-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer24-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer24-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer24-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer24-q-raw-final-tile.f32le.bin",
        "Qcur": "layer24-q-current-final-tile.f32le.bin",
        "KVrope": "layer24-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer24-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer24_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer24-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer24-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer23-complete-2048"},
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
            f"layer24_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer24-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer24-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer24-qkv-2048"},
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
        filename = f"layer24-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer24_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer24-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer24_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer24-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer24_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer24-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer24-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer24-compressor-2048"},
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
        filename = f"layer24-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer24_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer24-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer24-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer24-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
