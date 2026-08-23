#!/usr/bin/env python3
"""Augment the row-2,049 fixture with live layers-0/1 predecessor evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POSITION = 8195
RAW_CAPTURE_ROWS = 4352
PRIOR_RAW_ROWS = 127
RAW_CAPTURE_EXECUTABLE_SHA256 = (
    "a8206cf77be4e903a1c6a56fc39f794a348a9e5fdb907236f62c49f6d6d4da53"
)
HC_CAPTURE_EXECUTABLE_SHA256 = (
    "d32bff9535405147fa58eba8b505bfadccd15adb2a2eb8ff2549ff983e775b18"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repeated(first: Path, second: Path, name: str, expected_bytes: int) -> bytes:
    left = (first / name).read_bytes()
    right = (second / name).read_bytes()
    if len(left) != expected_bytes:
        raise SystemExit(f"{name} has {len(left)} bytes, expected {expected_bytes}")
    if left != right:
        raise SystemExit(f"fresh-process predecessor captures differ: {name}")
    return left


def tensor(
    fixture: Path,
    *,
    name: str,
    hook: str,
    role: str,
    shape: list[int],
    payload: bytes,
) -> dict:
    destination = fixture / f"{name.replace('_', '-')}.f32le.bin"
    destination.write_bytes(payload)
    return {
        "name": name,
        "hook": hook,
        "role": role,
        "dtype": "f32",
        "shape": shape,
        "encoding": "little-endian-ieee754-binary32",
        "path": destination.name,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_first", type=Path)
    parser.add_argument("raw_second", type=Path)
    parser.add_argument("layer0_hc_first", type=Path)
    parser.add_argument("layer0_hc_second", type=Path)
    parser.add_argument("layer1_hc_first", type=Path)
    parser.add_argument("layer1_hc_second", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "fixtures"
            / "retained-sparse-layer2-pos8195-v1"
        ),
    )
    args = parser.parse_args()

    manifest_path = args.fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fixture = (
        "dwarfstar-oracle-v1-retained-layer2-pos8195-sparse-multimerge"
    )
    if manifest.get("fixture_id") != expected_fixture:
        raise SystemExit("predecessor evidence targets the wrong fixture")
    added_names = {
        "layer0_raw_cache_prior",
        "layer1_raw_cache_prior",
        "layer0_raw_cache_current",
        "layer1_raw_cache_current",
        "layer0_hc_ffn_post",
    }
    existing_names = {entry["name"] for entry in manifest["tensors"]}
    if existing_names & added_names:
        raise SystemExit("predecessor evidence is already present")

    raw_captures: dict[int, bytes] = {}
    for layer in (0, 1):
        raw_captures[layer] = repeated(
            args.raw_first,
            args.raw_second,
            f"capture_raw_cache_decode-{layer}_pos{POSITION}.bin",
            RAW_CAPTURE_ROWS * 512 * 4,
        )
    layer0_hc = repeated(
        args.layer0_hc_first,
        args.layer0_hc_second,
        f"capture_hc_ffn_post-0_pos{POSITION}.bin",
        4 * 4096 * 4,
    )
    layer1_hc = repeated(
        args.layer1_hc_first,
        args.layer1_hc_second,
        f"capture_hc_ffn_post-1_pos{POSITION}.bin",
        4 * 4096 * 4,
    )
    if layer1_hc != (args.fixture / "retained-input-hc.f32le.bin").read_bytes():
        raise SystemExit("captured layer-1 HC does not match the pinned layer-2 handoff")

    additions = []
    for layer, raw_capture in raw_captures.items():
        raw_prior = bytearray()
        for logical_position in range(POSITION - PRIOR_RAW_ROWS, POSITION):
            slot = logical_position % RAW_CAPTURE_ROWS
            start = slot * 512 * 4
            raw_prior.extend(raw_capture[start : start + 512 * 4])
        current_slot = POSITION % RAW_CAPTURE_ROWS
        current_start = current_slot * 512 * 4
        current = raw_capture[current_start : current_start + 512 * 4]
        additions.extend(
            [
                tensor(
                    args.fixture,
                    name=f"layer{layer}_raw_cache_prior",
                    hook="raw_cache_decode",
                    role="input",
                    shape=[PRIOR_RAW_ROWS, 512],
                    payload=bytes(raw_prior),
                ),
                tensor(
                    args.fixture,
                    name=f"layer{layer}_raw_cache_current",
                    hook="raw_cache_decode",
                    role="intermediate",
                    shape=[512],
                    payload=current,
                ),
            ]
        )
    additions.append(
        tensor(
            args.fixture,
            name="layer0_hc_ffn_post",
            hook="hc_ffn_post",
            role="intermediate",
            shape=[4, 4096],
            payload=layer0_hc,
        )
    )

    manifest["capture"].update(
        {
            "predecessor_layers": [0, 1],
            "predecessor_fresh_process_captures": 2,
            "predecessor_fresh_process_bitwise_match": True,
            "predecessor_layer1_handoff_match": True,
            "predecessor_raw_capture_executable_sha256": (
                RAW_CAPTURE_EXECUTABLE_SHA256
            ),
            "predecessor_hc_capture_executable_sha256": (
                HC_CAPTURE_EXECUTABLE_SHA256
            ),
            "temporary_predecessor_raw_hook_removed_after_capture": True,
        }
    )
    manifest["scope"].update(
        {
            "executed_predecessor_layers": [0, 1],
            "predecessor_prior_raw_rows": PRIOR_RAW_ROWS,
        }
    )
    manifest["claims"].update(
        {
            "complete_layer": True,
            "preceding_layers_execution": True,
            "preceding_layer_raw_history_seeded": True,
        }
    )
    manifest["tensors"].extend(additions)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
