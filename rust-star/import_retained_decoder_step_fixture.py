#!/usr/bin/env python3
"""Import a compact, exact position-8195 full decoder-step fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


POSITION = 8195
LAYERS = 43
RAW_ROWS = 128
PRIOR_RAW_ROWS = 127
HEAD_DIM = 512
INDEX_DIM = 128
EVEN_LAYERS = list(range(2, 43, 2))
ODD_LAYERS = list(range(3, 43, 2))
MODEL_SHA256 = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"


@dataclass
class LayerState:
    raw: memoryview
    attention: memoryview | None = None
    attention_state_kv: memoryview | None = None
    attention_state_score: memoryview | None = None
    indexer: memoryview | None = None
    indexer_state_kv: memoryview | None = None
    indexer_state_score: memoryview | None = None


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_repeated(first: Path, second: Path, label: str) -> bytes:
    left = first.read_bytes()
    right = second.read_bytes()
    if left != right:
        raise SystemExit(f"fresh-process captures differ: {label}")
    return left


def parse_payload(payload: bytes) -> list[LayerState]:
    view = memoryview(payload)
    header = struct.unpack_from("<14I", view)
    expected = (
        0x4C565344,
        1,
        8196,
        4096,
        4352,
        128,
        2051,
        8195,
        43,
        512,
        128,
        0,
        42,
        128,
    )
    if header != expected:
        raise SystemExit(f"unexpected retained layer-payload header: {header!r}")
    offset = 14 * 4
    n_comp = struct.unpack_from("<43I", view, offset)
    offset += 43 * 4
    n_index = struct.unpack_from("<43I", view, offset)
    offset += 43 * 4
    layers: list[LayerState] = []

    def take(elements: int) -> memoryview:
        nonlocal offset
        byte_count = elements * 4
        result = view[offset : offset + byte_count]
        if len(result) != byte_count:
            raise SystemExit("retained layer payload is truncated")
        offset += byte_count
        return result

    for layer in range(LAYERS):
        raw = take(RAW_ROWS * HEAD_DIM)
        state = LayerState(raw=raw)
        if layer >= 2:
            ratio = 4 if layer % 2 == 0 else 128
            expected_rows = 2048 if ratio == 4 else 64
            if n_comp[layer] != expected_rows:
                raise SystemExit(f"layer {layer} has unexpected compressed row count")
            state.attention = take(expected_rows * HEAD_DIM)
            state_elements = 8192 if ratio == 4 else 65536
            state.attention_state_kv = take(state_elements)
            state.attention_state_score = take(state_elements)
            if ratio == 4:
                if n_index[layer] != 2048:
                    raise SystemExit(f"layer {layer} has unexpected indexer row count")
                state.indexer = take(2048 * INDEX_DIM)
                state.indexer_state_kv = take(2048)
                state.indexer_state_score = take(2048)
            elif n_index[layer] != 0:
                raise SystemExit(f"layer {layer} unexpectedly has indexer rows")
        elif n_comp[layer] != 0 or n_index[layer] != 0:
            raise SystemExit(f"uncompressed layer {layer} has compressed rows")
        layers.append(state)
    if offset != len(view):
        raise SystemExit(
            f"retained layer payload has trailing bytes: parsed={offset} total={len(view)}"
        )
    return layers


def pack_exact_f16(chunks: list[memoryview], label: str) -> bytes:
    packed = bytearray()
    for chunk_index, chunk in enumerate(chunks):
        values = np.frombuffer(chunk, dtype="<f4")
        halves = values.astype("<f2")
        restored = halves.astype("<f4")
        mismatch = np.flatnonzero(values.view("<u4") != restored.view("<u4"))
        if mismatch.size:
            raise SystemExit(
                f"{label} is not exactly f16 at chunk {chunk_index} element {int(mismatch[0])}"
            )
        packed.extend(halves.tobytes())
    return bytes(packed)


def tensor(
    fixture: Path,
    *,
    name: str,
    role: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> dict:
    suffix = "f16le" if dtype == "f16" else f"{dtype}le"
    path = fixture / f"{name.replace('_', '-')}.{suffix}.bin"
    path.write_bytes(payload)
    encoding = {
        "f16": "little-endian-ieee754-binary16",
        "f32": "little-endian-ieee754-binary32",
        "i32": "little-endian-signed-integer32",
    }[dtype]
    return {
        "name": name,
        "hook": "session_layer_payload" if role == "input" else name,
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "encoding": encoding,
        "path": path.name,
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state_first", type=Path)
    parser.add_argument("state_second", type=Path)
    parser.add_argument("capture_first", type=Path)
    parser.add_argument("capture_second", type=Path)
    parser.add_argument("topk_first", type=Path)
    parser.add_argument("topk_second", type=Path)
    parser.add_argument("--capture-executable-sha256", required=True)
    parser.add_argument("--topk-executable-sha256", required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parent
        / "fixtures"
        / "retained-decoder-step-pos8195-v1",
    )
    args = parser.parse_args()
    for value in (args.capture_executable_sha256, args.topk_executable_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise SystemExit("capture executable SHA-256 is invalid")

    state_payload = read_repeated(args.state_first, args.state_second, "layer payload")
    layers = parse_payload(state_payload)
    if (args.fixture / "manifest.json").exists():
        raise SystemExit(f"fixture is already complete: {args.fixture}")
    args.fixture.mkdir(parents=True, exist_ok=True)

    topk_payloads: list[bytes] = []
    topk_indices: list[tuple[int, ...]] = []
    for layer in EVEN_LAYERS:
        name = f"capture_indexer_topk-{layer}_pos{POSITION}.i32"
        payload = read_repeated(args.topk_first / name, args.topk_second / name, name)
        if len(payload) != 512 * 4:
            raise SystemExit(f"{name} has the wrong byte count")
        indices = struct.unpack("<512i", payload)
        if len(set(indices)) != 512 or min(indices) < 0 or max(indices) > 2048:
            raise SystemExit(f"{name} contains invalid selected rows")
        topk_payloads.append(payload)
        topk_indices.append(indices)

    raw_chunks = [state.raw[HEAD_DIM * 4 :] for state in layers]
    even_attention_chunks: list[memoryview] = []
    for layer, indices in zip(EVEN_LAYERS, topk_indices):
        attention = layers[layer].attention
        assert attention is not None
        zero = memoryview(bytes(HEAD_DIM * 4))
        for index in indices:
            start = index * HEAD_DIM * 4
            even_attention_chunks.append(
                zero if index == 2048 else attention[start : start + HEAD_DIM * 4]
            )
    odd_attention_chunks = [layers[layer].attention for layer in ODD_LAYERS]
    even_index_chunks = [layers[layer].indexer for layer in EVEN_LAYERS]
    if any(chunk is None for chunk in odd_attention_chunks + even_index_chunks):
        raise SystemExit("retained payload is missing compressed cache state")

    def joined(attribute: str, selected_layers: list[int]) -> bytes:
        result = bytearray()
        for layer in selected_layers:
            chunk = getattr(layers[layer], attribute)
            if chunk is None:
                raise SystemExit(f"layer {layer} is missing {attribute}")
            result.extend(chunk)
        return bytes(result)

    hc_outputs = []
    for layer in range(LAYERS):
        name = f"capture_hc_ffn_post-{layer}_pos{POSITION}.bin"
        hc_outputs.append(
            read_repeated(args.capture_first / name, args.capture_second / name, name)
        )
        if len(hc_outputs[-1]) != 4 * 4096 * 4:
            raise SystemExit(f"{name} has the wrong byte count")
    logits = read_repeated(
        args.capture_first / "logits.bin",
        args.capture_second / "logits.bin",
        "output logits",
    )
    if len(logits) != 129280 * 4:
        raise SystemExit("output logits have the wrong byte count")

    tensors = [
        tensor(
            args.fixture,
            name="raw_cache_prior",
            role="input",
            dtype="f16",
            shape=[43, 127, 512],
            payload=pack_exact_f16(raw_chunks, "raw cache"),
        ),
        tensor(
            args.fixture,
            name="even_attention_selected_prior",
            role="input",
            dtype="f16",
            shape=[21, 512, 512],
            payload=pack_exact_f16(even_attention_chunks, "even attention cache"),
        ),
        tensor(
            args.fixture,
            name="even_attention_selected_indices",
            role="input",
            dtype="i32",
            shape=[21, 512],
            payload=b"".join(topk_payloads),
        ),
        tensor(
            args.fixture,
            name="odd_attention_prior",
            role="input",
            dtype="f16",
            shape=[20, 64, 512],
            payload=pack_exact_f16(
                [chunk for chunk in odd_attention_chunks if chunk is not None],
                "odd attention cache",
            ),
        ),
        tensor(
            args.fixture,
            name="even_indexer_prior",
            role="input",
            dtype="f32",
            shape=[21, 2048, 128],
            payload=b"".join(
                bytes(chunk) for chunk in even_index_chunks if chunk is not None
            ),
        ),
        tensor(args.fixture, name="even_attention_state_kv", role="input", dtype="f32", shape=[21, 8192], payload=joined("attention_state_kv", EVEN_LAYERS)),
        tensor(args.fixture, name="even_attention_state_score_bits", role="input", dtype="i32", shape=[21, 8192], payload=joined("attention_state_score", EVEN_LAYERS)),
        tensor(args.fixture, name="odd_attention_state_kv", role="input", dtype="f32", shape=[20, 65536], payload=joined("attention_state_kv", ODD_LAYERS)),
        tensor(args.fixture, name="odd_attention_state_score_bits", role="input", dtype="i32", shape=[20, 65536], payload=joined("attention_state_score", ODD_LAYERS)),
        tensor(args.fixture, name="even_indexer_state_kv", role="input", dtype="f32", shape=[21, 2048], payload=joined("indexer_state_kv", EVEN_LAYERS)),
        tensor(args.fixture, name="even_indexer_state_score_bits", role="input", dtype="i32", shape=[21, 2048], payload=joined("indexer_state_score", EVEN_LAYERS)),
        tensor(args.fixture, name="layer_hc_ffn_post", role="intermediate", dtype="f32", shape=[43, 4, 4096], payload=b"".join(hc_outputs)),
        tensor(args.fixture, name="output_logits", role="output", dtype="f32", shape=[129280], payload=logits),
    ]
    selected_token = max(
        range(129280), key=lambda index: struct.unpack_from("<f", logits, index * 4)[0]
    )
    manifest = {
        "schema": "rust-star-differential-fixture-v1",
        "fixture_id": "dwarfstar-oracle-v1-retained-decoder-step-pos8195",
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "oracle": {
            "id": "oracle-v1",
            "repository": "https://github.com/antirez/ds4.git",
            "commit": "b0309611041655f4e45671cfd9c9886aff161406",
            "tree": "20c11af22f90a0bdf25da860da5ef06de4064060",
            "capture_executable_sha256": args.capture_executable_sha256,
        },
        "model": {"family": "DeepSeek-V4-Flash-0731", "sha256": MODEL_SHA256},
        "capture": {
            "backend": "metal",
            "machine": "Apple M1 Ultra, 48 GPU cores, 128 GB unified memory",
            "prompt": "speed-bench/promessi_sposi.txt",
            "prompt_sha256": "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f",
            "prefix_tokens": 8195,
            "captured_position": 8195,
            "captured_token": 381,
            "fresh_process_captures": 2,
            "fresh_process_bitwise_match": True,
            "session_layer_payload_bytes": len(state_payload),
            "session_layer_payload_sha256": sha256(state_payload),
            "topk_replay_executable_sha256": args.topk_executable_sha256,
            "topk_replay_bitwise_match": True,
            "temporary_sparse_topk_hook_removed_after_capture": True,
        },
        "scope": {
            "kind": "decode-step",
            "phase": "decode",
            "position": 8195,
            "layers": 43,
            "prior_raw_rows": 127,
            "ratio4_prior_compressed_rows": 2048,
            "ratio128_prior_compressed_rows": 64,
            "sparse_top_k": 512,
            "selected_token": selected_token,
        },
        "operations": [
            {"name": "retained-state-seed", "kernel": "host-to-shared-Metal-state"},
            {"name": "transformer-layers-0-through-42", "kernel": "Rust-Star-Metal-decoder-layer-chain"},
            {"name": "output-head", "kernel": "Rust-Star-Metal-output-head"},
        ],
        "claims": {
            "preceding_layers_execution": True,
            "preceding_layer_history_seeded": True,
            "complete_decoder_step": True,
            "output_logits": True,
            "native_prefill": False,
            "throughput": False,
        },
        "tensors": tensors,
    }
    (args.fixture / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.fixture)
    print(f"selected_token={selected_token} fixture_bytes={sum(t['bytes'] for t in tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
