#!/usr/bin/env python3
"""Import independently repeated layer-8 through -42 decode fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import import_layer45_fixtures as common


CAPTURED_AT_UTC = "2026-08-15T10:38:14Z"
LAYERS = range(8, 43)
POSITIONS = (1, 2, 3)


def record_batch_capture(output: Path, position: int) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["captured_at_utc"] = CAPTURED_AT_UTC
    capture = manifest["capture"]
    capture["batch_capture_layers"] = [0, 42]
    capture["environment"]["DS4_METAL_GRAPH_DUMP_LAYER"] = "all"
    if position == 0:
        capture["environment"]["DS4_METAL_GRAPH_DUMP_NAME"] = "attn_norm,KVcur"
    else:
        hooks = [
            tensor["hook"]
            for tensor in manifest["tensors"]
            if tensor["hook"] not in {"derived-from-position0-KVcur", "KVcompress"}
        ]
        capture["environment"]["DS4_METAL_GRAPH_DUMP_NAME"] = ",".join(
            hooks + ["KVcompress"]
        )
        if position == 1:
            capture["position0_environment"] = {
                "DS4_METAL_GRAPH_DUMP_LAYER": "all",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "attn_norm,KVcur",
            }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    common.CAPTURED_AT_UTC = CAPTURED_AT_UTC
    for layer in LAYERS:
        for position in POSITIONS:
            output = common.import_complete(
                args.capture_root,
                args.fixtures_root,
                layer,
                position,
            )
            record_batch_capture(output, position)
            print(output)
        output = common.import_prime(args.capture_root, args.fixtures_root, layer)
        record_batch_capture(output, 0)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
