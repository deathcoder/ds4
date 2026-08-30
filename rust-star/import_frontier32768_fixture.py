#!/usr/bin/env python3
"""Import the accepted four-repeat oracle-v3 32K prefill frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


FRONTIER = 32_768
VOCAB = 129_280
REPETITIONS = 4
EXPECTED_SOURCE_COMMIT = "d35fb12d01d500b9cefcef24092c295687ceaf7e"
EXPECTED_SOURCE_TREE = "617415ee9f8ea7dc176d63dada1d5a7582063824"
EXPECTED_MODEL_SHA256 = (
    "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
)
EXPECTED_PROMPT_SHA256 = (
    "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f"
)
EXPECTED_TOKEN_SHA256 = (
    "874c479e2cde7b231dfa4a963d5f1fbe9dd9e60c2842f869e2a117d4860902a0"
)
EXPECTED_JSON_LOGITS_SHA256 = (
    "269f9529f0e649d6e4f5ecd125ef7c4808f825e65bf9d503c512ef611c958e12"
)
EXPECTED_PACKED_LOGITS_SHA256 = (
    "603430b4b4b47f14b520f1977770bc292d1d136488f254116eb214766609c547"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_tokens(path: Path) -> bytes:
    payload = path.read_bytes()
    if len(payload) != FRONTIER * 4 or sha256(payload) != EXPECTED_TOKEN_SHA256:
        raise SystemExit("32K prompt token identity changed")
    tokens = struct.unpack(f"<{FRONTIER}I", payload)
    if any(token >= VOCAB for token in tokens):
        raise SystemExit("32K prompt contains an invalid token ID")
    return payload


def read_frontier(path: Path) -> tuple[bytes, int]:
    raw = path.read_bytes()
    if sha256(raw) != EXPECTED_JSON_LOGITS_SHA256:
        raise SystemExit(f"32K JSON logits identity changed: {path}")
    capture = json.loads(raw)
    if (
        capture.get("source") != "ds4-bench"
        or capture.get("prompt_tokens") != FRONTIER
        or capture.get("frontier_tokens") != FRONTIER
        or capture.get("vocab") != VOCAB
        or len(capture.get("logits", [])) != VOCAB
    ):
        raise SystemExit(f"invalid 32K frontier capture: {path}")
    logits = struct.pack(f"<{VOCAB}f", *capture["logits"])
    if sha256(logits) != EXPECTED_PACKED_LOGITS_SHA256:
        raise SystemExit(f"packed 32K logits identity changed: {path}")
    values = struct.unpack(f"<{VOCAB}f", logits)
    selected = max(range(VOCAB), key=values.__getitem__)
    if selected != capture.get("argmax_id"):
        raise SystemExit(f"frontier argmax disagrees with logits: {path}")
    return logits, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("token_ids", type=Path)
    parser.add_argument("oracle_bundle", type=Path)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    manifest_path = args.oracle_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rust-star-oracle-manifest-v1"
        or manifest.get("oracle_id") != "oracle-v3"
        or manifest.get("status") != "complete"
        or manifest.get("conformance", {}).get("status") != "passed"
        or manifest.get("source", {}).get("commit") != EXPECTED_SOURCE_COMMIT
        or manifest.get("source", {}).get("tree") != EXPECTED_SOURCE_TREE
        or manifest.get("model", {}).get("sha256") != EXPECTED_MODEL_SHA256
        or manifest.get("prompt", {}).get("expanded_sha256") != EXPECTED_PROMPT_SHA256
        or manifest.get("configuration", {}).get("conformance_repetitions")
        != REPETITIONS
    ):
        raise SystemExit("oracle-v3 manifest identity or acceptance state changed")

    token_payload = read_tokens(args.token_ids)
    runs = [
        run
        for run in manifest["conformance"]["runs"]
        if run.get("context") == FRONTIER
    ]
    if sorted(run.get("repetition") for run in runs) != list(range(1, REPETITIONS + 1)):
        raise SystemExit("oracle-v3 does not contain all four 32K repetitions")
    captures = []
    for run in runs:
        if run.get("logits", {}).get("sha256") != EXPECTED_JSON_LOGITS_SHA256:
            raise SystemExit("oracle-v3 manifest records 32K repetition drift")
        captures.append(read_frontier(args.oracle_bundle / run["logits"]["path"]))
    if any(capture != captures[0] for capture in captures[1:]):
        raise SystemExit("oracle-v3 32K repetitions are not bit-identical")
    logits, selected = captures[0]

    output = args.fixtures_root / "prefill-frontier-32768-v3"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    (output / "token-ids.u32le.bin").write_bytes(token_payload)
    (output / "batch-prefill-logits.f32le.bin").write_bytes(logits)

    fixture = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-frontier-32768",
        "captured_at_utc": manifest["completed_at_utc"],
        "oracle": {
            "id": "oracle-v3",
            "repository": manifest["source"]["repository"],
            "commit": EXPECTED_SOURCE_COMMIT,
            "tree": EXPECTED_SOURCE_TREE,
            "capture_executable_sha256": manifest["build"]["executables"]["ds4-bench"]["sha256"],
            "capture_kit_commit": manifest["capture_kit"]["commit"],
            "archive_sha256": "be77e20c42875f370169c777e2cb26d090fbb516826dc215aa33488d4b4e37bc",
        },
        "model": {
            "family": manifest["configuration"]["model_family"],
            "sha256": EXPECTED_MODEL_SHA256,
        },
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "first 32768 raw tokens of speed-bench/promessi_sposi.txt",
            "prompt_sha256": EXPECTED_PROMPT_SHA256,
            "prompt_token_ids_sha256": EXPECTED_TOKEN_SHA256,
            "prefill_tokens": FRONTIER,
            "fresh_process_captures": REPETITIONS,
            "fresh_process_bitwise_match": True,
            "json_logits_sha256": EXPECTED_JSON_LOGITS_SHA256,
            "performance_enabled": False,
        },
        "scope": {
            "kind": "decode-step",
            "phase": "prefill",
            "layer": 43,
            "position": FRONTIER - 1,
        },
        "operations": [
            {
                "name": "chunked-batched-prefill-frontier-full-logits",
                "kernel": "complete-decoder",
                "weights": ["output.weight"],
            }
        ],
        "selection": {
            "method": "lowest-token-id-argmax",
            "token_id": selected,
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
                "bytes": len(logits),
                "sha256": EXPECTED_PACKED_LOGITS_SHA256,
            },
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
