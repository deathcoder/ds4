#!/usr/bin/env python3
"""Import repeated full-2K layer-28 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROWS = 2_048
TILE_ROWS = 32
CAPTURED_AT_UTC = "2026-08-26T13:46:34Z"
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
    "hc_attn_pre": "6b4c264a655baa201b40ac6083ee821264232c15dfde4898d8f9fa7100827bb8",
    "attn_norm": "75e9fa5e846ba39df7aa6b9016662e3032a63b1fd646b767daa40c4a5828cde7",
    "q_lora": "5c1ce8ff28557627bc81b1e36a07544fa4290cf8fb9e7f81ee8f4c9c0b5d3e19",
    "q_lora_norm": "71f78113f9c07c995d7c9b95cd0a976069895294e8903e8843c012e868bee5da",
    "KVraw": "ca098775590057582565071eebb0bf61d076247cae03f20768252b7196cdc157",
    "KVnorm": "36c8f904004978b4cc8060aa777421099ae894e282a560b7eab3bcf68121d437",
    "Qraw": "143d6e820bf67571524de639ff2dd9da1a2991cded17d75be1aabd8415a86eb9",
    "Qcur": "37353e8073783ef6499726368082a52b5f72f8333abebd262b76d0663c93a7dc",
    "KVrope": "339e8dc8ec1c76d2c4a0cb9f0d6090cc554504b91f79be477160d4898e053303",
    "KVcur": "73b6e21dc3e74674948e29277cc3f1fa125d36fe5a1cbe7a7e69542658b9dc05",
}
COMPRESSOR = {
    "KVcompress": ("attention-compressed-kv.f32le.bin", [512, 512], 1_048_576,
        "e463850ea591e967a682f84872acf16c83cf1a19762d0f502ecf9e237c5130c5"),
    "attn_state_kv": ("attention-state-kv.f32le.bin", [8, 1024], 32_768,
        "ea1cc29879c3c4eb02208ade2316d4840db24fd31682a406236c7ec29a863c04"),
    "attn_state_score": ("attention-state-score.i32le.bin", [8, 1024], 32_768,
        "f89c5c14ccd4a2fd087ad9d192caf8e0c2f389e0ce34ca25b25f27046afad657"),
    "indexer_KVcompress": ("indexer-compressed-kv.f32le.bin", [512, 128], 262_144,
        "6b08393ee5ac5c2f497f9166bcbf90321560a84728ac1585c26798bbf0b01b7c"),
    "indexer_state_kv": ("indexer-state-kv.f32le.bin", [8, 256], 8_192,
        "046c48397e7557ff32ec35783ddf5b91c4179c33689454b5c5ac0240175a478e"),
    "indexer_state_score": ("indexer-state-score.i32le.bin", [8, 256], 8_192,
        "7ee4b988c01f895cf6f6c799e7ed76fbf22992cca4a1fdaa81de48fdd303a8dd"),
}
ATTENTION_WIDTHS = {
    "attn_out": 4_096,
    "hc_attn_post": 16_384,
    "kqv_out": 32_768,
    "kqv_back": 32_768,
    "attn_low": 8_192,
}
ATTENTION_SHA256 = {
    "attn_out": "96fd3ae7e76902eeb95070a2f098bb0856104ee3e24b442d27d505df310a9fdf",
    "hc_attn_post": "dbe412056522dc0a5f3f25dc037476b120d148b1f7129f8fcdd14b627f896fa5",
    "kqv_out": "5823645af0145cd8c1cf57a1a3c69c869707fe06cd07cacb15c1efb4c91ce32c",
    "kqv_back": "0a1221fd3368372ee9e7070f91c08244c9cf98033f6bda40db599675c88c231f",
    "attn_low": "bd7d7e2c0527ddad004784c8530db0702dbd79874f42f039b79103248aaee7e6",
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
    "hc_ffn_pre": "a98627e28356dbaeb8b9fd3102e3b794ee4dd71e3b9eea398dcffd3339e61b20",
    "ffn_norm": "34eb5a65d2081009fd4b370782f289275a762898752a48328bc2121ef92bbed5",
    "ffn_moe_logits": "742a1c1e6fc037b82a89c7a8b0514dd5ebde102602d2aad3f08208e4eabc15bb",
    "ffn_moe_probs": "645d6adf162d3df939c4ac0b13c1e7eced459f5df630f971a8e9c0eef6ad990b",
    "ffn_moe_topk": "68be1aec276f60aa4a7005020e81b9e818ed0b45907f7bc3d97dcf45ec6a1b01",
    "ffn_moe_weights_scaled": "9d8ba0b88de766ab09ef8f78149485b5629169cc23a7516f8606a60d7c3ee687",
    "ffn_moe_weighted_swiglu": "6098cf6ce85ef163505206205798aa7ea73dce391ffa93dd8e8698801124f17a",
    "ffn_moe_out": "20b3ff3f8654747a00f1a9bc16c67011774d701d659c5efb60a687167b50a961",
    "ffn_shexp": "f746907feac13c44b525224afd0d62cefed9dc3890b88d37a7b974cdc41ab774",
    "hc_ffn_post": "6c7bab7cd9890b80c49dad5b03a92e18ff804ed6c5a2a6746cac02d0cc1ffe70",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture(directory: Path, hook: str) -> bytes:
    suffix = "i32" if hook == "ffn_moe_topk" else "bin"
    return (directory / f"{directory.name}_{hook}-28_pos0.{suffix}").read_bytes()


def repeated(
    root: Path, group: str, hook: str, expected_bytes: int, identity: str
) -> bytes:
    first = capture(root / f"{group}_first", hook)
    second = capture(root / f"{group}_second", hook)
    if len(first) != expected_bytes or len(second) != expected_bytes:
        raise SystemExit(f"layer-28 {hook} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process layer-28 {hook} captures differ")
    if sha256(first) != identity:
        raise SystemExit(f"layer-28 {hook} capture identity changed")
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
            "layer": 28,
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
        (args.fixtures_root / "prefill-layer27-complete-2048-v1" / "manifest.json")
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
        "hc_attn_pre": "layer28-hc-attn-pre-final-tile.f32le.bin",
        "attn_norm": "layer28-attn-norm-final-tile.f32le.bin",
        "q_lora": "layer28-q-lora-final-tile.f32le.bin",
        "q_lora_norm": "layer28-q-lora-norm-final-tile.f32le.bin",
        "KVraw": "layer28-kv-raw-final-tile.f32le.bin",
        "KVnorm": "layer28-kv-norm-final-tile.f32le.bin",
        "Qraw": "layer28-q-raw-final-tile.f32le.bin",
        "Qcur": "layer28-q-current-final-tile.f32le.bin",
        "KVrope": "layer28-kv-rope-final-tile.f32le.bin",
        "KVcur": "layer28-kv-current-final-tile.f32le.bin",
    }
    qkv_tensors = []
    for name, width in QKV_WIDTHS.items():
        payload = qkv_payloads[name][-TILE_ROWS * width * 4 :]
        filename = qkv_filenames[name]
        qkv_tensors.append((filename, payload, tensor(
            f"layer28_{name.lower()}_final_tile", name, "f32",
            [TILE_ROWS, width], filename, payload,
            "output" if name == "KVcur" else "intermediate",
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer28-qkv-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer28-qkv-2048",
        template,
        {**common_capture, "fresh_process_captures": 4,
         "full_capture_sha256": QKV_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer27-complete-2048"},
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
            f"layer28_{hook.lower()}", hook, dtype, shape, filename, payload, "output"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer28-compressor-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer28-compressor-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": {hook: spec[3] for hook, spec in COMPRESSOR.items()},
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer28-qkv-2048"},
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
        filename = f"layer28-{name.replace('_', '-')}-row0.f32le.bin"
        attention_tensors.append((filename, payload, tensor(
            f"layer28_{name}_row0", name, "f32", [1, width], filename, payload
        )))
    attention_payload = attention_payloads["attn_out"]
    attention_filename = "layer28-attention-output.f32le.bin"
    attention_tensors.append((attention_filename, attention_payload, tensor(
        "layer28_attention_output", "attn_out", "f32", [ROWS, 4096],
        attention_filename, attention_payload, "output"
    )))
    attention_hc = attention_payloads["hc_attn_post"][-TILE_ROWS * 16384 * 4 :]
    attention_hc_filename = "layer28-hc-attn-post-final-tile.f32le.bin"
    attention_tensors.append((attention_hc_filename, attention_hc, tensor(
        "layer28_hc_attn_post_final_tile", "hc_attn_post", "f32",
        [TILE_ROWS, 16384], attention_hc_filename, attention_hc, "output"
    )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer28-attention-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer28-attention-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": ATTENTION_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer28-compressor-2048"},
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
        filename = f"layer28-{name.replace('_', '-')}-final-tile.{dtype}le.bin"
        ffn_tensors.append((filename, payload, tensor(
            f"layer28_{name}_final_tile", name, dtype, [TILE_ROWS, width],
            filename, payload, "output" if name == "hc_ffn_post" else "intermediate"
        )))
    write_fixture(
        args.fixtures_root,
        "prefill-layer28-complete-2048-v1",
        "dwarfstar-oracle-v1-prefill-layer28-complete-2048",
        template,
        {**common_capture, "fresh_process_captures": 2,
         "full_capture_sha256": FFN_SHA256,
         "input_fixture": "dwarfstar-oracle-v1-prefill-layer28-attention-2048"},
        [
            {"name": "ffn-hc-ingress-and-router", "kernel": "biased top-6 batch"},
            {"name": "routed-shared-experts-and-hc-post", "kernel": "IQ2/Q2 routed plus Q8 shared"},
        ],
        ffn_tensors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
