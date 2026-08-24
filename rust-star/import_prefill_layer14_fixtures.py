#!/usr/bin/env python3
"""Import repeated full-2K layer-14 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-24T16:30:03Z"
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
    "hc_attn_pre": "420c07ac62980a31249b724c18dc0755d8846e9cbb74e24f8a74c2f38e3c10b7",
    "attn_norm": "15502b0907dfa9b4bac20af47739e4bf0e3896cba8a355068ab9eb120ded6624",
    "q_lora": "fd5fd5a73d255ce0d97c8819ab7f843ea8b5e3504280da713d57440ddc52757c",
    "q_lora_norm": "93cc00813864280e6f661ce55fe4f5072df673f81d75b2d788d88950c47187c1",
    "KVraw": "6dbece9b25902368d5ec03d946ebcf339eff8b47904b0f64ae81e8f8ff1b5892",
    "KVnorm": "7c8214e33633503e6637d83894dca1078b40f6611e14bbf9aaabfb0ae529a2aa",
    "Qraw": "64449f21b11c219b66ee81c6e7ccca235f047e2c69f692a162ee6acbb1fd9597",
    "Qcur": "15bedd3ac4895a19da23d01dabdaa4171e5312f358fabd5776649c75fb9b6299",
    "KVrope": "b5be924f8aab657ce6effca7d1768a329df552ec5181d5c45569a6cb2ea4ef72",
    "KVcur": "a57e01f3374eb40e629bb88c603fdd96c5ed5947004b6b5ff1c3d63abe5416e4",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "a9cf349c3e862c158735244977011293e26f4e143859513939d4a4a85fcde898"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "213758e259259957c8ac292eba47759716d9ad0083011f48c2379b625f8a861b"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "2bfabe524e4ed02c0213ff191fc20894e9d2a409b42e87d89ebb167665a7e05f"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "f418275e39f922087c1bc2d25b5b6008e3d7f49d754c6861a95f152980dbb433"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "df3456e228a15cae186047c76c1daf876b8e6caff022249198fd0e4589507d51"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "1e0609e5ff46296189862c53aeae10b7d7144766222f8516f64571fdd4f2e3d6"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "75a69a49b10ba5250314539efc1147d245d687f1eb6e2a809018cfcb3bad3c7c",
    "hc_attn_post": "cb3c99012bc9f8a15a95ed85e28fb974b0d5aab2a2de4e27aa6ca5ab0eecd1b9",
    "kqv_out": "baa5f6d4b9accfe9889e1a04d1f3927b6968bac5f8d5558ea5d6ee6d85bcd3b8",
    "kqv_back": "8f14625e6406b05035d0de1cfff84cc28ca221595c7fe507119ebada4dfa7b89",
    "attn_low": "eaffc405be8b64329055f83819c1534c090e2445f98c8e53d73c9070e01f6dc4",
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
    "hc_ffn_pre": "d6985b0d8a4738a82a670cb0c77685b434900be7d1f0a8e4bc279de3ca60bac1",
    "ffn_norm": "cfcac66080a8e27cb1dd81022726deb91936c3a43588cae4ebfc6bc9744a3092",
    "ffn_moe_logits": "2cf1f1e5ae3bcd69e51156efbf66a70af1db98a276f0e698f943391982c9c150",
    "ffn_moe_probs": "9bde292df2ded6d963c0845c62da6bccc016ccafa533c9c724f73e1d810343cd",
    "ffn_moe_topk": "fb3f582724e4bee027bbcb437aacfde4289fa484cda96c530d3c9a5ff12549f8",
    "ffn_moe_weights_scaled": "f1ab33ad2e90436e01cb473330b4becfad7a92f366954a02d2d431ea4190ecce",
    "ffn_moe_weighted_swiglu": "edea55d131d1b13fb0018da7fae109f732e04836991c1cbe81b04537d7c60030",
    "ffn_moe_out": "7d9551b840c36a000104fe45f2e1d98ae38addc13a1a3aeda212af8909837402",
    "ffn_shexp": "22aa5f623ca21b71b03a0fad787adfdbbe9c5919a4111c386973b5f86aa6f53e",
    "hc_ffn_post": "a9e54c6cc3cd973cad6e3c2068ba5c25017e83f64462b3b7d5935555af290ecd",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-14_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-14 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-14 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-14 {hook} capture identity changed")
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
            "layer": 14,
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
        (args.fixtures_root / "prefill-layer13-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer14-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer14-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer14-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer14-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer14-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer14-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer14-q-raw-final-tile.f32le.bin",
        "Qcur": "layer14-q-current-final-tile.f32le.bin",
        "KVrope": "layer14-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer14-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer14_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer14-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer14-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer13-complete-2048"},
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
            f"layer14_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer14-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer14-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer14-qkv-2048"},
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
        filename = f"layer14-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer14_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer14-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer14_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer14-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer14_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer14-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer14-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer14-compressor-2048"},
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
        filename = f"layer14-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer14_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer14-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer14-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer14-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
