#!/usr/bin/env python3
"""Import the four-repeat supplemental oracle-v3 8K prefill frontier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import subprocess
from pathlib import Path


FRONTIER = 8_192
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
EXPECTED_CAPTURE_EXECUTABLE_SHA256 = (
    "8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3"
)
EXPECTED_JSON_LOGITS_SHA256 = (
    "791ee1ea8129889e3adaf4ce6e042156b85323e46df32581e4df275055848f94"
)
EXPECTED_PACKED_LOGITS_SHA256 = (
    "626454dd1d12717abe29c9fe7d4140bcb265046bf8253bc46ea02575a9c53a1a"
)
EXPECTED_TOKEN_SHA256 = (
    "0a1625377cb917e0a62c328909f01c68c4fe6256a3af37f213a071af170b7e17"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(source: Path) -> tuple[str, str]:
    def resolve(revision: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", revision],
            cwd=source,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout.strip()

    return resolve("HEAD^{commit}"), resolve("HEAD^{tree}")


def read_tokens(fixture_32k: Path) -> bytes:
    manifest = json.loads((fixture_32k / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("fixture_id")
        != "dwarfstar-oracle-v3-prefill-frontier-32768"
        or manifest.get("oracle", {}).get("commit") != EXPECTED_SOURCE_COMMIT
        or manifest.get("oracle", {}).get("tree") != EXPECTED_SOURCE_TREE
        or manifest.get("model", {}).get("sha256") != EXPECTED_MODEL_SHA256
        or manifest.get("capture", {}).get("prompt_sha256") != EXPECTED_PROMPT_SHA256
    ):
        raise SystemExit("accepted 32K oracle-v3 fixture identity changed")
    payload = (fixture_32k / "token-ids.u32le.bin").read_bytes()[: FRONTIER * 4]
    if len(payload) != FRONTIER * 4 or sha256(payload) != EXPECTED_TOKEN_SHA256:
        raise SystemExit("8K prompt-token prefix identity changed")
    tokens = struct.unpack(f"<{FRONTIER}I", payload)
    if any(token >= VOCAB for token in tokens):
        raise SystemExit("8K prompt contains an invalid token ID")
    return payload


def read_capture(path: Path) -> tuple[bytes, int]:
    raw = path.read_bytes()
    if sha256(raw) != EXPECTED_JSON_LOGITS_SHA256:
        raise SystemExit(f"8K JSON logits identity changed: {path}")
    capture = json.loads(raw)
    if (
        capture.get("source") != "ds4-bench"
        or capture.get("prompt_tokens") != FRONTIER
        or capture.get("frontier_tokens") != FRONTIER
        or capture.get("vocab") != VOCAB
        or len(capture.get("logits", [])) != VOCAB
    ):
        raise SystemExit(f"invalid 8K frontier capture: {path}")
    logits = struct.pack(f"<{VOCAB}f", *capture["logits"])
    if sha256(logits) != EXPECTED_PACKED_LOGITS_SHA256:
        raise SystemExit(f"packed 8K logits identity changed: {path}")
    values = struct.unpack(f"<{VOCAB}f", logits)
    selected = max(range(VOCAB), key=values.__getitem__)
    if selected != capture.get("argmax_id") or selected != 77_179:
        raise SystemExit(f"8K frontier argmax disagrees with logits: {path}")
    return logits, selected


def read_prefill_observation(path: Path) -> tuple[float, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise SystemExit(f"expected one 8K CSV row: {path}")
    row = rows[0]
    if (
        row.get("ctx_tokens") != str(FRONTIER)
        or row.get("prefill_tokens") != str(FRONTIER)
        or row.get("gen_tokens") != "0"
    ):
        raise SystemExit(f"invalid 8K CSV row: {path}")
    return float(row["prefill_tps"]), sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("producer_source", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("fixture_32k", type=Path)
    parser.add_argument("--captured-at-utc", required=True)
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()

    commit, tree = git_identity(args.producer_source)
    if (commit, tree) != (EXPECTED_SOURCE_COMMIT, EXPECTED_SOURCE_TREE):
        raise SystemExit("8K producer source identity changed")
    if sha256_file(args.producer_source / "ds4-bench") != EXPECTED_CAPTURE_EXECUTABLE_SHA256:
        raise SystemExit("8K capture executable identity changed")
    if sha256_file(args.producer_source / "rust_star_oracle_prompt.txt") != EXPECTED_PROMPT_SHA256:
        raise SystemExit("8K capture prompt identity changed")
    if sha256_file(args.producer_source / "model.gguf") != EXPECTED_MODEL_SHA256:
        raise SystemExit("8K capture model identity changed")

    token_payload = read_tokens(args.fixture_32k)
    captures: list[tuple[bytes, int]] = []
    observations: list[dict[str, object]] = []
    for repetition in range(1, REPETITIONS + 1):
        run = args.capture_root / f"run_{repetition:02d}"
        logits_path = run / "logits" / "frontier_008192.logits.json"
        captures.append(read_capture(logits_path))
        prefill_tps, csv_sha = read_prefill_observation(run / "prefill.csv")
        observations.append(
            {
                "repetition": repetition,
                "json_logits_sha256": sha256_file(logits_path),
                "csv_sha256": csv_sha,
                "prefill_tokens_per_second": prefill_tps,
            }
        )
    if any(capture != captures[0] for capture in captures[1:]):
        raise SystemExit("8K fresh-process captures are not bit-identical")
    logits, selected = captures[0]

    output = args.fixtures_root / "prefill-frontier-8192-v3"
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir()
    (output / "token-ids.u32le.bin").write_bytes(token_payload)
    (output / "batch-prefill-logits.f32le.bin").write_bytes(logits)

    fixture = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v3-prefill-frontier-8192",
        "captured_at_utc": args.captured_at_utc,
        "oracle": {
            "id": "oracle-v3",
            "repository": "https://github.com/deathcoder/ds4.git",
            "commit": EXPECTED_SOURCE_COMMIT,
            "tree": EXPECTED_SOURCE_TREE,
            "capture_executable_sha256": EXPECTED_CAPTURE_EXECUTABLE_SHA256,
            "relationship": "supplemental 8K frontier from the accepted synchronized producer",
        },
        "model": {
            "family": "DeepSeek-V4-Flash-0731",
            "sha256": EXPECTED_MODEL_SHA256,
        },
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "first 8192 raw tokens of speed-bench/promessi_sposi.txt",
            "prompt_sha256": EXPECTED_PROMPT_SHA256,
            "prompt_token_ids_sha256": EXPECTED_TOKEN_SHA256,
            "prefill_tokens": FRONTIER,
            "prefill_chunk_tokens": 4096,
            "fresh_process_captures": REPETITIONS,
            "fresh_process_bitwise_match": True,
            "json_logits_sha256": EXPECTED_JSON_LOGITS_SHA256,
            "observations": observations,
            "performance_enabled": False,
            "timing_classification": "conformance observations; not paired benchmark evidence",
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
