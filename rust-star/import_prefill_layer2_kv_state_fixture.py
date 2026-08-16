#!/usr/bin/env python3
"""Import repeated full-2K layer-2 compressed-RoPE KV captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-16T07:25:47Z"
PREFILL_ROWS = 2_048
KV_WIDTH = 512
EXPECTED_BYTES = PREFILL_ROWS * KV_WIDTH * 4
EXPECTED_SHA256 = {
    "KVrope": "d46da14951b304fb4a19be43b82d350b273337de042da2308530438e431e117d",
    "KVcur": "07f19c5197442f3c85350b32d0661e81b3f105a0e8640d3b3bced6c333267135",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_repeated(name: str, first: Path, second: Path) -> bytes:
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    if len(first_payload) != EXPECTED_BYTES or len(second_payload) != EXPECTED_BYTES:
        raise SystemExit(f"layer-2 {name} capture has the wrong size")
    if first_payload != second_payload:
        raise SystemExit(f"fresh-process layer-2 {name} captures differ")
    if sha256(first_payload) != EXPECTED_SHA256[name]:
        raise SystemExit(f"layer-2 {name} capture identity changed")
    return first_payload


def tensor(name: str, hook: str, path: str, payload: bytes) -> dict:
    return {
        "name": name,
        "hook": hook,
        "role": "output",
        "dtype": "f32",
        "shape": [PREFILL_ROWS, KV_WIDTH],
        "encoding": "little-endian-ieee754-binary32",
        "path": path,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("kv-rope", "kv-current"):
        parser.add_argument(f"--{name}-first", type=Path, required=True)
        parser.add_argument(f"--{name}-second", type=Path, required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    captures = {
        "KVrope": checked_repeated(
            "KVrope", args.kv_rope_first, args.kv_rope_second
        ),
        "KVcur": checked_repeated(
            "KVcur", args.kv_current_first, args.kv_current_second
        ),
    }
    template = json.loads(
        (
            args.fixtures_root
            / "prefill-layer2-kvnorm-2048-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    output = args.fixtures_root / "prefill-layer2-kv-state-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    names = {
        "KVrope": "layer2-kv-rope.f32le.bin",
        "KVcur": "layer2-kv-current.f32le.bin",
    }
    for hook, payload in captures.items():
        (output / names[hook]).write_bytes(payload)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-layer2-kv-state-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": PREFILL_ROWS,
            "fresh_process_captures_per_hook": 2,
            "fresh_process_bitwise_match": True,
            "command": template["capture"]["command"],
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "2",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "KVrope or KVcur",
            },
            "device_path": (
                "layer-2 YaRN-scaled compressed-attention KV RoPE followed by "
                "E4M3FN simulation on the M1 Ultra"
            ),
            "full_capture_sha256": EXPECTED_SHA256,
            "input_fixture": "dwarfstar-oracle-v1-prefill-layer2-kvnorm-2048",
        },
        "scope": {
            "kind": "layer-segment",
            "phase": "prefill",
            "layer": 2,
            "position": PREFILL_ROWS - 1,
            "captured_position_range": [0, PREFILL_ROWS - 1],
        },
        "operations": [
            {
                "name": "layer2-kv-rope",
                "kernel": "kernel_dsv4_rope_tail_f32",
            },
            {
                "name": "layer2-kv-finalize",
                "kernel": "kernel_dsv4_fp8_kv_quantize_f32",
            },
        ],
        "tensors": [
            tensor(
                "layer2_kv_rope",
                "KVrope",
                names["KVrope"],
                captures["KVrope"],
            ),
            tensor(
                "layer2_kv_current",
                "KVcur",
                names["KVcur"],
                captures["KVcur"],
            ),
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
