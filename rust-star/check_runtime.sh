#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
manifest="$repo_dir/rust-star/runtime/Cargo.toml"
target_dir="$repo_dir/rust-star/.work/runtime-target"

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: Cargo is required (Rust 1.74 or newer)" >&2
    exit 2
fi

export CARGO_TARGET_DIR="$target_dir"

echo "==> formatting contract"
cargo fmt --manifest-path "$manifest" --check

echo "==> platform-independent unit tests"
cargo test --manifest-path "$manifest"

echo "==> optimized host build"
cargo build --release --manifest-path "$manifest"

if [ "$(uname -s)" = "Darwin" ]; then
    echo "==> Metal ownership and dispatch probe"
    "$target_dir/release/rust-star" metal-probe \
        --elements 4096 \
        --iterations 100 \
        --json "$target_dir/metal-dispatch-probe.json"
fi

echo "==> existing Python artifact tests"
python3 -m unittest discover -s "$repo_dir/rust-star/tests" -v

echo "==> pinned differential fixtures"
for fixture_manifest in "$repo_dir"/rust-star/fixtures/*/manifest.json; do
    python3 "$repo_dir/rust-star/verify_differential_fixture.py" \
        "$(dirname "$fixture_manifest")"
done

echo "==> Rust-writer/Python-reader artifact contract"
fixture="$target_dir/candidate-logit-fixture.json"
comparison="$target_dir/candidate-logit-fixture-comparison.json"
cargo run --quiet --manifest-path "$manifest" \
    --example write_logit_fixture -- "$fixture"
python3 "$repo_dir/rust-star/compare_logits.py" \
    "$fixture" "$fixture" --json "$comparison"

if [ "$#" -eq 0 ]; then
    echo "==> model inspection skipped (pass an absolute GGUF path to enable it)"
elif [ "$#" -eq 1 ]; then
    case "$1" in
        /*) ;;
        *)
            echo "error: model path must be absolute" >&2
            exit 2
            ;;
    esac
    echo "==> strict target model inspection"
    "$target_dir/release/rust-star" inspect "$1"
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "==> no-copy F16 embedding gather"
        "$target_dir/release/rust-star" embedding-probe "$1" \
            --json "$target_dir/f16-embedding-probe.json"
        echo "==> no-copy Q8_0 decode projection"
        "$target_dir/release/rust-star" projection-probe "$1" \
            --json "$target_dir/q8-projection-probe.json"
        echo "==> exact M1 batched-Q8 prefill arithmetic boundary"
        "$target_dir/release/rust-star" prefill-q8-boundary-probe "$1" \
            --json "$target_dir/prefill-q8-boundary-probe.json"
        "$target_dir/release/rust-star" prefill-qkv-boundary-probe "$1" \
            --json "$target_dir/prefill-qkv-boundary-probe.json"
        echo "==> continuous complete M1 layer-0 prefill final tile"
        "$target_dir/release/rust-star" prefill-layer0-boundary-probe "$1" \
            --json "$target_dir/prefill-layer0-boundary-probe.json"
        echo "==> direct native M1 layer-0 to layer-1 prefill handoff"
        "$target_dir/release/rust-star" prefill-layers01-boundary-probe "$1" \
            --json "$target_dir/prefill-layers01-boundary-probe.json"
        echo "==> continuous complete M1 layers-0/1 prefill final tile"
        "$target_dir/release/rust-star" prefill-layers01-complete-boundary-probe "$1" \
            --json "$target_dir/prefill-layers01-complete-boundary-probe.json"
        echo "==> arbitrary-position M1 layers-0/1 prefill row coverage"
        "$target_dir/release/rust-star" prefill-layers01-row-coverage-probe "$1" \
            --json "$target_dir/prefill-layers01-row-coverage-probe.json"
        echo "==> persistent live-KV chain across two M1 layers-0/1 prefill tiles"
        "$target_dir/release/rust-star" prefill-layers01-live-kv-chain-probe "$1" \
            --json "$target_dir/prefill-layers01-live-kv-chain-probe.json"
        echo "==> empty-seed full-2K live-KV loop across native M1 layers 0/1"
        "$target_dir/release/rust-star" prefill-layers01-live-kv-loop-probe "$1" \
            --json "$target_dir/prefill-layers01-live-kv-loop-probe.json"
        echo "==> downstream layer-2 KVnorm validation across the full native 2K layers-0/1 loop"
        "$target_dir/release/rust-star" prefill-layers012-kvnorm-loop-probe "$1" \
            --json "$target_dir/prefill-layers012-kvnorm-loop-probe.json"
        echo "==> full native layer-2 rotated/finalized KV-state ownership"
        "$target_dir/release/rust-star" prefill-layers012-kv-state-loop-probe "$1" \
            --json "$target_dir/prefill-layers012-kv-state-loop-probe.json"
        echo "==> paired layer-2 ratio-4 compressors across the full native 2K loop"
        "$target_dir/release/rust-star" prefill-layers012-compressor-loop-probe "$1" \
            --json "$target_dir/prefill-layers012-compressor-loop-probe.json"
        echo "==> exact complete layers 0-5 plus layer-6 Q/KV across the full native 2K loop"
        "$target_dir/release/rust-star" prefill-layers012-attention-loop-probe "$1" \
            --json "$target_dir/prefill-layers012-attention-loop-probe.json"
        echo "==> no-copy layer-0 attention ingress"
        "$target_dir/release/rust-star" attention-ingress-probe "$1" \
            --json "$target_dir/attention-ingress-probe.json"
        echo "==> no-copy layer-0 attention projection setup"
        "$target_dir/release/rust-star" attention-setup-probe "$1" \
            --json "$target_dir/attention-setup-probe.json"
        echo "==> layer-0 Q/K RoPE and guarded KV-cache store"
        "$target_dir/release/rust-star" rope-kv-store-probe "$1" \
            --json "$target_dir/rope-kv-store-probe.json"
        echo "==> layer-0 raw-cache FlashAttention read"
        "$target_dir/release/rust-star" attention-read-probe "$1" \
            --json "$target_dir/attention-read-probe.json"
        echo "==> layer-0 grouped attention output and HC post-update"
        "$target_dir/release/rust-star" attention-output-probe "$1" \
            --json "$target_dir/attention-output-probe.json"
        echo "==> layer-0 FFN HC ingress and hash router"
        "$target_dir/release/rust-star" ffn-router-probe "$1" \
            --json "$target_dir/ffn-router-probe.json"
        echo "==> layer-0 routed/shared experts and FFN HC post-update"
        "$target_dir/release/rust-star" moe-output-probe "$1" \
            --json "$target_dir/moe-output-probe.json"
        echo "==> continuous complete layer-0 command chain"
        "$target_dir/release/rust-star" layer0-probe "$1" \
            --json "$target_dir/layer0-probe.json"
        echo "==> persistent layers 0-3 with direct HC-state handoffs and per-layer KV caches"
        "$target_dir/release/rust-star" layers0123-probe "$1" \
            --json "$target_dir/layers0123-probe.json"
        echo "==> chained layers 0-3 with one tail wait"
        "$target_dir/release/rust-star" layers0123-chained-probe "$1" \
            --json "$target_dir/layers0123-chained-probe.json"
        echo "==> position-advancing layers 0-3 with persistent per-layer KV caches"
        "$target_dir/release/rust-star" layers0123-decode-probe "$1" \
            --json "$target_dir/layers0123-decode-probe.json"
        echo "==> position-advancing layers 0-5 with generalized compressor ownership"
        "$target_dir/release/rust-star" layers012345-decode-probe "$1" \
            --json "$target_dir/layers012345-decode-probe.json"
        echo "==> position-advancing layers 0-7 with generalized compressor ownership"
        "$target_dir/release/rust-star" layers01234567-decode-probe "$1" \
            --json "$target_dir/layers01234567-decode-probe.json"
        echo "==> position-advancing layers 0-42 with generalized compressor ownership"
        "$target_dir/release/rust-star" layers0-42-decode-probe "$1" \
            --json "$target_dir/layers0-42-decode-probe.json"
        echo "==> exact decoder output head with full logits and greedy selection"
        "$target_dir/release/rust-star" decoder-output-probe "$1" \
            --json "$target_dir/decoder-output-probe.json"
        echo "==> C0 closed-loop decoder and readback-free timed diagnostic"
        "$target_dir/release/rust-star" closed-loop-decoder-probe "$1" \
            --json "$target_dir/closed-loop-decoder-probe.json"
        echo "==> integrated position-127 closed-loop decoder frontier"
        "$target_dir/release/rust-star" position127-decoder-probe "$1" \
            --json "$target_dir/position127-decoder-probe.json"
        echo "==> cold one-token prefill and integrated position-127 decoder"
        "$target_dir/release/rust-star" cold-prefill-decoder-probe "$1" \
            --json "$target_dir/cold-prefill-decoder-probe.json"
        echo "==> 2K sequential frontier and batched-prefill boundary"
        "$target_dir/release/rust-star" prefill-frontier-probe "$1" \
            --json "$target_dir/prefill-frontier-probe.json"
        echo "==> bounded position-127 ratio-128 compressor replay"
        "$target_dir/release/rust-star" ratio128-compressor-replay-probe "$1" \
            --json "$target_dir/ratio128-compressor-replay-probe.json"
        echo "==> repeated layers 0-3 with setup and correctness readback outside timing"
        "$target_dir/release/rust-star" layers0123-bench "$1" \
            --warmup 5 --iterations 20 \
            --json "$target_dir/layers0123-bench.json"
        echo "==> persistent layer-0 steady-state execution"
        "$target_dir/release/rust-star" layer0-bench "$1" \
            --warmup 10 --iterations 30 \
            --json "$target_dir/layer0-bench.json"
    fi
else
    echo "usage: $0 [/absolute/path/to/model.gguf]" >&2
    exit 2
fi

echo "==> Rust Star host-runtime checks passed"
