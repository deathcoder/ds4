#!/usr/bin/env python3
"""Import repeated full-2K layer-2 mixed-attention output captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-16T08:46:37Z"
EXPECTED_BYTES = 2_048 * 4_096 * 4
EXPECTED_SHA256 = "68c2110283b472105f00e192817dbb682ebe0815f2ba4a76cd73ed20f3d97508"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_path(root: Path) -> Path:
    return root / "oracle_attn_out-2_pos0.bin"


def checked_repeated(first: Path, second: Path) -> bytes:
    first_payload = capture_path(first).read_bytes()
    second_payload = capture_path(second).read_bytes()
    if len(first_payload) != EXPECTED_BYTES or len(second_payload) != EXPECTED_BYTES:
        raise SystemExit("layer-2 attn_out capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit("fresh-process layer-2 attn_out captures differ")
    if sha256(first_payload) != EXPECTED_SHA256:
        raise SystemExit("layer-2 attn_out capture identity changed")
    return first_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payload = checked_repeated(args.first, args.second)
    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer2-compressors-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer2-attention-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    filename = "layer2-attention-output.f32le.bin"
    (output / filename).write_bytes(payload)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer2-attention-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 2_048,
            "fresh_process_captures_per_hook": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_out",
            },
            "device_path": (
                "full 2048-row Q projection, dense mixed 2048 raw plus 512 "
                "compressed FlashAttention, inverse compressed RoPE, and "
                "legacy grouped/dense Q8_0 output projections"
            ),
            "indexer_policy": (
                "exactly 512 compressed rows remain dense; sparse top-k begins "
                "only after the compressed cache grows beyond 512 rows"
            ),
            "input_fixture": (
                "dwarfstar-oracle-v1-prefill-layer2-compressors-2048"
            ),
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": 2_047,
            "captured_position_range": [0, 2_047],
        },
        "operations": [
            {
                "name": "layer2-q-projection-and-rope",
                "kernel": "kernel_mul_mm_q8_0_f32 + fused head RMSNorm/YaRN RoPE",
            },
            {
                "name": "layer2-dense-mixed-attention",
                "kernel": "kernel_flash_attn_ext_f16_dk512_dv512",
            },
            {
                "name": "layer2-attention-inverse-rope",
                "kernel": "kernel_rope_norm_f32",
            },
            {
                "name": "layer2-attention-output-projections",
                "kernel": "kernel_mul_mm_id_q8_0_f32 + kernel_mul_mm_q8_0_f32",
            },
        ],
        "tensors": [
            {
                "name": "layer2_attention_output",
                "hook": "attn_out",
                "role": "output",
                "dtype": "f32",
                "shape": [2_048, 4_096],
                "encoding": "little-endian-ieee754-binary32",
                "path": filename,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
