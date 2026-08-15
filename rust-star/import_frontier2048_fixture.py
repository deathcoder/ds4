#!/usr/bin/env python3
"""Import repeated oracle logits and token IDs for the 2K prompt frontier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T13:29:01Z"
FRONTIER = 2_048
VOCAB = 129_280
EXPECTED_TOKEN_SHA256 = (
    "5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380"
)
EXPECTED_BATCH_LOGITS_SHA256 = (
    "7b5e851884bbb0aa8c2a249c8497af0feccb267cbd0a40e0a4a5aee584ecbfaf"
)
EXPECTED_DECODE_REPLAY_LOGITS_SHA256 = (
    "aa657efb7a5cb7108ee639ea797eb4f1c223f36360d356f96674499935d1f405"
)
PROMPT_SHA256 = "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_tokens(path: Path) -> bytes:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    tokens = json.loads(first_line)
    if len(tokens) < FRONTIER:
        raise SystemExit(f"token dump has {len(tokens)} tokens, need {FRONTIER}")
    tokens = tokens[:FRONTIER]
    if any(not isinstance(token, int) or not 0 <= token < VOCAB for token in tokens):
        raise SystemExit("token dump contains an invalid token ID")
    payload = struct.pack(f"<{FRONTIER}I", *tokens)
    if sha256(payload) != EXPECTED_TOKEN_SHA256:
        raise SystemExit("2K prompt token identity changed")
    return payload


def read_logits(path: Path) -> tuple[bytes, int]:
    capture = json.loads(path.read_text(encoding="utf-8"))
    if (
        capture.get("source") != "ds4-bench"
        or capture.get("prompt_tokens") != FRONTIER
        or capture.get("frontier_tokens") != FRONTIER
        or len(capture.get("logits", [])) != VOCAB
    ):
        raise SystemExit(f"invalid 2K frontier capture: {path}")
    logits = struct.pack(f"<{VOCAB}f", *capture["logits"])
    selected = max(range(VOCAB), key=struct.unpack(f"<{VOCAB}f", logits).__getitem__)
    if selected != capture.get("argmax_id"):
        raise SystemExit(f"frontier argmax disagrees with logits: {path}")
    return logits, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("token_dump", type=Path)
    parser.add_argument("first_capture", type=Path)
    parser.add_argument("second_capture", type=Path)
    parser.add_argument("first_decode_replay", type=Path)
    parser.add_argument("second_decode_replay", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    token_payload = read_tokens(args.token_dump)
    first_logits, first_selected = read_logits(args.first_capture)
    second_logits, second_selected = read_logits(args.second_capture)
    if first_logits != second_logits or first_selected != second_selected:
        raise SystemExit("independent 2K frontier captures differ")
    if sha256(first_logits) != EXPECTED_BATCH_LOGITS_SHA256:
        raise SystemExit("2K frontier logits identity changed")
    first_replay = args.first_decode_replay.read_bytes()
    second_replay = args.second_decode_replay.read_bytes()
    if len(first_replay) != VOCAB * 4 or first_replay != second_replay:
        raise SystemExit("independent 2K decode replays differ or have invalid size")
    if sha256(first_replay) != EXPECTED_DECODE_REPLAY_LOGITS_SHA256:
        raise SystemExit("2K decode-replay logits identity changed")
    if first_replay == first_logits:
        raise SystemExit("decode replay unexpectedly equals batched prefill")

    template = json.loads(
        (args.fixtures_root / "output-head-pos4-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "prefill-frontier-2048-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    (output / "token-ids.u32le.bin").write_bytes(token_payload)
    (output / "batch-prefill-logits.f32le.bin").write_bytes(first_logits)
    (output / "decode-replay-logits.f32le.bin").write_bytes(first_replay)

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-prefill-frontier-2048",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": "first 2048 raw tokens of speed-bench/promessi_sposi.txt",
            "prompt_sha256": PROMPT_SHA256,
            "prompt_token_ids_sha256": EXPECTED_TOKEN_SHA256,
            "prefill_tokens": FRONTIER,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "command": "ds4-bench --metal --ctx-start 2048 --ctx-max 2048 --gen-tokens 0 --warm-weights --dump-frontier-logits-dir",
            "decode_replay": {
                "construction": "one-token cold prefill followed by 2047 ds4_session_eval calls",
                "fresh_process_captures": 2,
                "fresh_process_bitwise_match": True,
                "logits_sha256": EXPECTED_DECODE_REPLAY_LOGITS_SHA256,
                "equals_batched_prefill": False,
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "prefill",
            "layer": 43,
            "position": FRONTIER - 1,
        },
        "operations": [
            {
                "name": "sequential-decode-replay",
                "kernel": "complete-decoder",
                "weights": [],
            },
            {
                "name": "batched-prefill-frontier-full-logits",
                "kernel": "kernel_mul_mv_q8_0_f32",
                "weights": ["output.weight"],
            },
        ],
        "selection": {
            "method": "lowest-token-id-argmax",
            "token_id": first_selected,
        },
        "tensors": [
            {
                "name": "prompt_token_ids",
                "hook": "tokenizer",
                "role": "input",
                "dtype": "i32",
                "shape": [FRONTIER],
                "encoding": "little-endian-signed-integer32",
                "path": "token-ids.u32le.bin",
                "bytes": len(token_payload),
                "sha256": EXPECTED_TOKEN_SHA256,
            },
            {
                "name": "batch_prefill_logits",
                "hook": "frontier_logits",
                "role": "output",
                "dtype": "f32",
                "shape": [VOCAB],
                "encoding": "little-endian-ieee754-binary32",
                "path": "batch-prefill-logits.f32le.bin",
                "bytes": len(first_logits),
                "sha256": EXPECTED_BATCH_LOGITS_SHA256,
            },
            {
                "name": "decode_replay_logits",
                "hook": "session_decode_logits",
                "role": "output",
                "dtype": "f32",
                "shape": [VOCAB],
                "encoding": "little-endian-ieee754-binary32",
                "path": "decode-replay-logits.f32le.bin",
                "bytes": len(first_replay),
                "sha256": EXPECTED_DECODE_REPLAY_LOGITS_SHA256,
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
