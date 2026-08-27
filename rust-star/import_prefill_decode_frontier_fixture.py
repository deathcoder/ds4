#!/usr/bin/env python3
"""Import the repeated 2K-batched-prefill closed-loop continuation oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-27T13:24:48Z"
PREFILL_TOKENS = 2_048
FINAL_POSITION = 4_099
CONTINUATION_TOKENS = FINAL_POSITION - PREFILL_TOKENS + 1
VOCAB = 129_280
EXPECTED_CAPTURE_EXECUTABLE_SHA256 = (
    "3a2e2f64dc70341b5729bf4c29713c4212e1658f81555492fffeba616573167a"
)
EXPECTED_PREFILL_PAYLOAD_SHA256 = (
    "296554d691a66bc9869cf670d93320a645b4977c622c0c94321473d485532654"
)
EXPECTED_INPUT_TOKENS_SHA256 = (
    "ebcfa0c483d2ce202c3ee425047fdd41b2dfc7f9f56bd47318cfe8a5ddb104fe"
)
EXPECTED_FIRST_LOGITS_SHA256 = (
    "ba543a4eb2fd2cef42d2976eb6a0ea408feeb30377c6e6e77522b5f983e730a3"
)
EXPECTED_FINAL_LOGITS_SHA256 = (
    "b0d3d11d08f336d4a40e10eea5c5d2018d205aae61068c76d8ec8ce824fa5f8c"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_repeated(first: Path, second: Path, name: str, expected: str) -> bytes:
    first_payload = (first / name).read_bytes()
    second_payload = (second / name).read_bytes()
    if first_payload != second_payload:
        raise SystemExit(f"independent {name} captures differ")
    if sha256(first_payload) != expected:
        raise SystemExit(f"{name} identity changed")
    return first_payload


def lowest_argmax(values: tuple[float, ...]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_capture", type=Path)
    parser.add_argument("second_capture", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    payload = read_repeated(
        args.first_capture,
        args.second_capture,
        "prefill-layer-payload.bin",
        EXPECTED_PREFILL_PAYLOAD_SHA256,
    )
    input_tokens = read_repeated(
        args.first_capture,
        args.second_capture,
        "input-tokens.u32le.bin",
        EXPECTED_INPUT_TOKENS_SHA256,
    )
    first_logits = read_repeated(
        args.first_capture,
        args.second_capture,
        "first-logits.f32le.bin",
        EXPECTED_FIRST_LOGITS_SHA256,
    )
    final_logits = read_repeated(
        args.first_capture,
        args.second_capture,
        "final-logits.f32le.bin",
        EXPECTED_FINAL_LOGITS_SHA256,
    )
    if len(payload) != 51_659_152:
        raise SystemExit("retained prefill payload has an invalid size")
    if len(input_tokens) != CONTINUATION_TOKENS * 4:
        raise SystemExit("continuation transcript has an invalid size")
    if len(first_logits) != VOCAB * 4 or len(final_logits) != VOCAB * 4:
        raise SystemExit("continuation logits have an invalid size")

    tokens = struct.unpack(f"<{CONTINUATION_TOKENS}I", input_tokens)
    first_values = struct.unpack(f"<{VOCAB}f", first_logits)
    final_values = struct.unpack(f"<{VOCAB}f", final_logits)
    first_selected = lowest_argmax(first_values)
    final_selected = lowest_argmax(final_values)
    if tokens[0] != 15342 or tokens[1] != first_selected or first_selected != 201:
        raise SystemExit("the immediate post-prefill selection chain changed")
    if tokens[-1] != 312 or final_selected != 2538:
        raise SystemExit("the first sparse-boundary selection changed")

    template = json.loads(
        (args.fixtures_root / "prefill-frontier-2048-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-decode-frontier-4099-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    shutil.copyfile(
        args.first_capture / "input-tokens.u32le.bin",
        output / "input-tokens.u32le.bin",
    )
    shutil.copyfile(
        args.first_capture / "first-logits.f32le.bin",
        output / "first-logits.f32le.bin",
    )
    shutil.copyfile(
        args.first_capture / "final-logits.f32le.bin",
        output / "final-logits.f32le.bin",
    )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-decode-frontier-4099",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "construction": "exact 2048-token batched prefill followed by greedy ds4_session_eval calls through position 4099",
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "capture_executable_sha256": EXPECTED_CAPTURE_EXECUTABLE_SHA256,
            "command": "capture-prefill-decode-frontier MODEL token-ids.u32le.bin 4099 ...",
            "retained_prefill_payload": {
                "bytes": len(payload),
                "sha256": EXPECTED_PREFILL_PAYLOAD_SHA256,
                "imported": False,
                "purpose": "repeatability evidence and local diagnosis only",
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "position": FINAL_POSITION,
            "prefill_tokens": PREFILL_TOKENS,
            "position_start": PREFILL_TOKENS,
            "position_end": FINAL_POSITION,
            "evaluated_positions": CONTINUATION_TOKENS,
            "first_default_sparse_position": FINAL_POSITION,
            "ratio4_compressed_rows": 1_025,
        },
        "operations": [
            {
                "name": "exact_batched_prefill",
                "kernel": "ds4_session_sync",
            },
            {
                "name": "greedy_decode_positions_2048_4099",
                "kernel": "ds4_session_eval",
            },
        ],
        "selection": {
            "method": "lowest-token-id-argmax",
            "prefill_selected_token": tokens[0],
            "position_2048_selected_token": first_selected,
            "position_4099_input_token": tokens[-1],
            "position_4099_selected_token": final_selected,
        },
        "tensors": [
            {
                "name": "decode_input_token_ids",
                "hook": "closed_loop_input",
                "role": "input",
                "dtype": "i32",
                "shape": [CONTINUATION_TOKENS],
                "encoding": "little-endian-signed-integer32",
                "path": "input-tokens.u32le.bin",
                "bytes": len(input_tokens),
                "sha256": EXPECTED_INPUT_TOKENS_SHA256,
            },
            {
                "name": "position_2048_logits",
                "hook": "session_decode_logits",
                "role": "output",
                "dtype": "f32",
                "shape": [VOCAB],
                "encoding": "little-endian-ieee754-binary32",
                "path": "first-logits.f32le.bin",
                "bytes": len(first_logits),
                "sha256": EXPECTED_FIRST_LOGITS_SHA256,
            },
            {
                "name": "position_4099_logits",
                "hook": "session_decode_logits",
                "role": "output",
                "dtype": "f32",
                "shape": [VOCAB],
                "encoding": "little-endian-ieee754-binary32",
                "path": "final-logits.f32le.bin",
                "bytes": len(final_logits),
                "sha256": EXPECTED_FINAL_LOGITS_SHA256,
            },
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
