#!/usr/bin/env python3
"""Import compact positions-1984..2015 evidence for complete layers 0 and 1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-16T05:34:30Z"
PREFILL_ROWS = 2_048
TILE_ROWS = 32
TILE_START = 1_984
TILE_END = TILE_START + TILE_ROWS
FULL_SHA256 = {
    "layer0_kv_current": "bef8d14d805a482960cbf7315ad0efccf211516a27bd767d0884b81f3ad33893",
    "layer0_hc_ffn_post": "5540a37fd14e8d3f9eb5d2b2ac0e515366410ea7ae66530c2eb7ccf6fdf12930",
    "layer0_selected": "2af6c8b65918ddbfa1b06afba3be9b8820450cfa6027865fbd8973a8dc526bf1",
    "layer1_kv_current": "a008066c234ef8b9bf162b38fb55e1593302a7d517138fc6c36df13595449cbb",
    "layer1_hc_ffn_post": "2ed3f6203eed280bef18380e77511a6fc2122f25d1cb96dc1cf4921b95505e99",
    "layer1_selected": "e727333d04bf36fdaa50fbf931cae10088195cfe69e57518363535f9466240a5",
}
WIDTHS = {
    "layer0_kv_current": 512,
    "layer0_hc_ffn_post": 16_384,
    "layer0_selected": 6,
    "layer1_kv_current": 512,
    "layer1_hc_ffn_post": 16_384,
    "layer1_selected": 6,
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    expected_bytes = PREFILL_ROWS * WIDTHS[name] * 4
    if len(first_payload) != expected_bytes or len(second_payload) != expected_bytes:
        raise SystemExit(f"{name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process {name} captures differ")
    if sha256(first_payload) != FULL_SHA256[name]:
        raise SystemExit(f"{name} full-capture identity changed")
    return first_payload


def tile(payload: bytes, width: int) -> bytes:
    row_bytes = width * 4
    return payload[TILE_START * row_bytes : TILE_END * row_bytes]


def tensor(name: str, hook: str, dtype: str, payload: bytes, width: int) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": "output",
        "dtype": dtype,
        "shape": [TILE_ROWS, width],
        "encoding": (
            "little-endian-signed-integer32"
            if dtype == "i32"
            else "little-endian-ieee754-binary32"
        ),
        "path": f"{name.replace('_', '-')}.{dtype}le.bin",
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in WIDTHS:
        parser.add_argument(f"--{name.replace('_', '-')}-first", type=Path, required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payloads = {}
    for name, width in WIDTHS.items():
        full = repeated(
            name,
            getattr(args, f"{name}_first"),
            getattr(args, f"{name}_second"),
        )
        payloads[name] = tile(full, width)

    output = args.fixtures_root / "prefill-layers01-previous-tile-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()

    entries = []
    for name, payload in payloads.items():
        dtype = "i32" if name.endswith("selected") else "f32"
        hook = "ffn_moe_topk" if dtype == "i32" else (
            "KVcur" if name.endswith("kv_current") else "hc_ffn_post"
        )
        entry = tensor(name, hook, dtype, payload, WIDTHS[name])
        (output / entry["path"]).write_bytes(payload)
        entries.append(entry)

    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer1-complete-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layers01-previous-tile-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
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
                "DS4_METAL_GRAPH_DUMP_LAYER": "0 and 1",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "KVcur,hc_ffn_post,ffn_moe_topk",
            },
            "device_path": "complete uncompressed layers 0 and 1 over the previous M1 tile",
            "full_capture_sha256": FULL_SHA256,
            "input_fixtures": [
                "dwarfstar-oracle-v1-prefill-frontier-2048",
                "dwarfstar-oracle-v1-prefill-attention-read-2048",
                "dwarfstar-oracle-v1-prefill-layer1-complete-2048",
            ],
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layers": [0, 1],
            "position": TILE_END - 1,
            "captured_position_range": [TILE_START, TILE_END - 1],
            "kv_position_range": [0, TILE_END - 1],
        },
        "operations": [
            {"name": "layer0-complete", "kernel": "native M1 43-dispatch tile schedule"},
            {"name": "layer1-complete", "kernel": "native M1 41-dispatch tile schedule"},
        ],
        "tensors": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
