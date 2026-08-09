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

echo "==> existing Python artifact tests"
python3 -m unittest discover -s "$repo_dir/rust-star/tests" -v

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
else
    echo "usage: $0 [/absolute/path/to/model.gguf]" >&2
    exit 2
fi

echo "==> Rust Star host-runtime checks passed"
