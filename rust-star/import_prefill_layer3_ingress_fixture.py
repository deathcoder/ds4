#!/usr/bin/env python3
"""Import the first full-2K native layer-3 prefill boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PREFILL_ROWS = 2_048
TILE_ROWS = 32
WIDTHS = {
    "hc_attn_pre": 4_096,
    "attn_norm": 4_096,
    "q_lora": 1_024,
}
EXPECTED_FULL_SHA256 = {
    "hc_attn_pre": "615f67cb9738b583263e1a7abd3970ad4df818f0bc9220053be4cbc6ba7d6cab",
    "attn_norm": "1348e3368a4c6b7185e730f11545c74a7fe8fa1c9877fcc9200a5de405f2b818",
    "q_lora": "bafe82a1535457caec52278bdb3c95e317aacb128c94f704fc523ca9581d6265",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first_dir: Path, second_dir: Path) -> bytes:
    filename = f"{name}-3_pos0.bin"
    first = (first_dir / f"first_{filename}").read_bytes()
    second = (second_dir / f"second_{filename}").read_bytes()
    expected_size = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first) != expected_size or len(second) != expected_size:
        raise SystemExit(f"{name} capture has the wrong size")
    if first != second:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first) != EXPECTED_FULL_SHA256[name]:
        raise SystemExit(f"{name} capture identity changed")
    return first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    full = {
        name: checked_repeated(name, args.first_dir, args.second_dir)
        for name in WIDTHS
    }
    template = json.loads(
        (args.fixtures_root / "prefill-layer2-complete-2048-v1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer3-ingress-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    tensors = []
    for name, width in WIDTHS.items():
        payload = full[name][-TILE_ROWS * width * 4 :]
        filename = f"layer3-{name.replace('_', '-')}-final-tile.f32le.bin"
        (output / filename).write_bytes(payload)
        tensors.append({
            "name": f"layer3_{name}_final_tile",
            "hook": name,
            "role": "output" if name == "q_lora" else "intermediate",
            "dtype": "f32",
            "shape": [TILE_ROWS, width],
            "encoding": "little-endian-ieee754-binary32",
            "path": filename,
            "bytes": len(payload),
            "sha256": sha256(payload),
        })

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer3-ingress-2048",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "3",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": ",".join(WIDTHS),
            },
            "full_capture_sha256": EXPECTED_FULL_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer2-complete-2048",
            "storage_note": "Exact final tiles are retained; complete 2K identities are pinned by SHA-256.",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 3,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [PREFILL_ROWS - TILE_ROWS, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "attention-hc-ingress-and-norm",
                "kernel": "kernel_rms_norm_f32_4 plus kernel_mul_mm_f16_f32 plus kernel_dsv4_hc_split_weighted_sum_norm4",
            },
            {"name": "q-lora-projection", "kernel": "kernel_mul_mm_q8_0_f32"},
        ],
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
