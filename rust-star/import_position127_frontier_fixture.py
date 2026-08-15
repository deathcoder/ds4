#!/usr/bin/env python3
"""Import a repeated position-127 closed-loop frontier capture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path


CAPTURED_AT_UTC = "2026-08-15T14:51:00Z"
EXPECTED_TOKENS = 128
EXPECTED_TOKEN_SHA256 = (
    "95aedd05c1843ed9638bcd24e93b9ed4cde360a341f8c720c7efc908c3586697"
)
EXPECTED_LOGITS_SHA256 = (
    "1258d4c3ab662f72ac1f70be80ddbc8f9cf72db35b4450231369f3b6ae07c895"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_capture(run: Path) -> tuple[list[int], bytes]:
    logprobs = json.loads((run / "logprobs.json").read_text(encoding="utf-8"))
    if logprobs.get("prompt_tokens") != 1 or len(logprobs.get("steps", [])) != 127:
        raise SystemExit(f"invalid one-token/127-step capture in {run}")
    tokens = [step["selected"]["id"] for step in logprobs["steps"]]
    logits = (run / "oracle_result_output-43_pos0.bin").read_bytes()
    if len(logits) != 129280 * 4:
        raise SystemExit(f"invalid final logits size in {run}")
    values = struct.unpack(f"<{len(logits) // 4}f", logits)
    tokens.append(max(range(len(values)), key=values.__getitem__))
    return tokens, logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    first_tokens, first_logits = read_capture(args.capture_root / "a")
    second_tokens, second_logits = read_capture(args.capture_root / "b")
    if first_tokens != second_tokens or first_logits != second_logits:
        raise SystemExit("independent position-127 captures differ")
    if len(first_tokens) != EXPECTED_TOKENS:
        raise SystemExit("position-127 transcript does not contain 128 tokens")

    token_payload = b"".join(token.to_bytes(4, "little") for token in first_tokens)
    if sha256(token_payload) != EXPECTED_TOKEN_SHA256:
        raise SystemExit("position-127 token transcript identity changed")
    if sha256(first_logits) != EXPECTED_LOGITS_SHA256:
        raise SystemExit("position-127 logits identity changed")

    template = json.loads(
        (args.fixtures_root / "output-head-pos4-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.fixtures_root / "decoder-frontier-pos127-v1"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    (output / "token-ids.u32le.bin").write_bytes(token_payload)
    shutil.copyfile(
        args.capture_root / "a" / "oracle_result_output-43_pos0.bin",
        output / "logits.f32le.bin",
    )

    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-decoder-frontier-pos127",
        "captured_at_utc": CAPTURED_AT_UTC,
        "oracle": copy.deepcopy(template["oracle"]),
        "model": copy.deepcopy(template["model"]),
        "capture": {
            "backend": "metal",
            "machine": template["capture"]["machine"],
            "prompt": template["capture"]["prompt"],
            "prompt_sha256": template["capture"]["prompt_sha256"],
            "prefill_tokens": 1,
            "bootstrap_prompt_token_id": 36662,
            "decode_positions": 127,
            "committed_tokens": 128,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "environment": {
                "DS4_METAL_GRAPH_DUMP_LAYER": "43",
                "DS4_METAL_GRAPH_DUMP_POS": "0",
                "DS4_METAL_GRAPH_DUMP_NAME": "result_output",
            },
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "layer": 43,
            "position": 127,
        },
        "operations": [
            {
                "name": "closed-loop-greedy-transcript",
                "kernel": "lowest-token-id-argmax",
                "weights": [],
            },
            {
                "name": "position-127-full-logits",
                "kernel": "kernel_mul_mv_q8_0_f32",
                "weights": ["output.weight"],
            },
        ],
        "selection": {
            "method": "lowest-token-id-argmax",
            "token_id": first_tokens[-1],
            "transcript_tokens": len(first_tokens),
            "transcript_sha256": EXPECTED_TOKEN_SHA256,
        },
        "tensors": [
            {
                "name": "committed_token_ids",
                "hook": "greedy_selected_tokens",
                "role": "intermediate",
                "dtype": "i32",
                "shape": [EXPECTED_TOKENS],
                "encoding": "little-endian-signed-integer32",
                "path": "token-ids.u32le.bin",
                "bytes": len(token_payload),
                "sha256": EXPECTED_TOKEN_SHA256,
            },
            {
                "name": "logits",
                "hook": "result_output",
                "role": "output",
                "dtype": "f32",
                "shape": [129280],
                "encoding": "little-endian-ieee754-binary32",
                "path": "logits.f32le.bin",
                "bytes": len(first_logits),
                "sha256": EXPECTED_LOGITS_SHA256,
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
