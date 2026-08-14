# Rust Star host-runtime scaffold

This crate is the smallest executable boundary for the new engine. It is
model-specific by design and currently contains one connected six-operation
layer segment, not a decoder.

Implemented contracts:

- bounded, dependency-free GGUF v3 metadata/tensor-directory parsing;
- no tensor-payload reads during inspection;
- duplicate, overflow, alignment, overlap, UTF-8, type, and file-bound checks;
- exact DeepSeek V4 Flash shape validation derived from DwarfStar;
- exact resident imatrix-Q2 tensor recipe validation: IQ2_XXS routed gate/up,
  Q2_K routed down, Q8_0 attention/shared/output, and F16 HC/compressor/indexer;
- a full-FP32-logit JSON writer compatible with `../ARTIFACT_FORMAT.md`.
- a macOS-only Rust/Objective-C/Metal ownership boundary and correctness-checked
  command-dispatch probe, documented in `METAL.md`.
- a Rust-owned read-only shared mmap whose page-aligned F16 embedding span is
  wrapped by Metal with `newBufferWithBytesNoCopy`;
- DwarfStar's `kernel_get_rows_f16` token-embedding gather, validated bit-for-bit
  against a dependency-free CPU F16-to-F32 reference on selected real rows.
- DwarfStar's decode-time `kernel_mul_mv_q8_0_f32`, consuming a real layer-0
  decode activation and the no-copy `blk.0.attn_q_a.weight` span, validated
  bit-for-bit against a pinned DwarfStar layer/step fixture.
- the connected layer-0 attention ingress from token 201 through the Q-A
  projection, with six no-copy model ranges and bitwise checks at mixer, HC
  split, collapsed HC, learned norm, and Q-Lora boundaries.

The validator proves model shape and quantization-recipe identity. It does not
pretend that a mutable GGUF name proves the `0731` checkpoint. The completed
`oracle-v1` manifest's whole-file model SHA-256 will be authoritative.

## Build and test

Rust 1.74 or newer is required. No crates are downloaded.

```sh
cargo fmt --manifest-path rust-star/runtime/Cargo.toml --check
cargo test --manifest-path rust-star/runtime/Cargo.toml
cargo build --release --manifest-path rust-star/runtime/Cargo.toml
```

On macOS, Cargo uses `xcrun clang` to compile the small ARC Objective-C shim and
links Foundation and Metal. Other platforms build the host contracts with an
explicit unsupported Metal stub.

`check_runtime.sh` additionally writes a synthetic candidate artifact with the
Rust writer and makes the existing Python comparator accept it as C0 against
itself. This catches cross-language JSON/FP32 contract mistakes.

On the target Mac, run all host checks and inspect the real model with:

```sh
./rust-star/check_runtime.sh \
  /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf
```

Inspection reads the header, metadata, and tensor directory. It seeks over
large tokenizer arrays and does not hash or read tensor payloads, so this check
is not a substitute for `capture_oracle_v1.py`'s whole-model hash.

For a structural diagnosis that intentionally skips target identity:

```sh
rust-star/.work/runtime-target/release/rust-star gguf MODEL.gguf
```

Do not use the structural-only result as evidence that an inference model is
supported.

To run only the initial Metal dispatch probe:

```sh
rust-star/.work/runtime-target/release/rust-star metal-probe \
  --elements 4096 \
  --iterations 100 \
  --json rust-star/.work/runtime-target/metal-dispatch-probe.json
```

To run only the real-model no-copy/kernel boundary:

```sh
rust-star/.work/runtime-target/release/rust-star embedding-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/f16-embedding-probe.json
```

This command first applies the strict target validator. It maps the GGUF
read-only with `MAP_SHARED`, wraps only the page-aligned embedding tensor range,
checks that Metal retained the exact mmap pointer, gathers five rows spanning
the vocabulary, and requires every FP32 output bit to match the CPU reference.

To run the first decode projection boundary:

```sh
rust-star/.work/runtime-target/release/rust-star projection-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/q8-projection-probe.json
```

The fixture under `../fixtures/q8-attn-q-a-v1/` was captured from pinned
DwarfStar at layer 0, decode position 1. The command imports DwarfStar's default
four-simdgroup/two-row Q8_0 matvec, wraps only the real weight span without a
copy, and requires all 1,024 output FP32 bit patterns to match the fixture.

To run the connected layer-0 ingress gate:

```sh
rust-star/.work/runtime-target/release/rust-star attention-ingress-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/attention-ingress-probe.json
```

The command derives the projection input from the real token embedding in one
six-dispatch command buffer. It requires all six mmap-backed weight views and
every retained intermediate in `../fixtures/layer0-attention-ingress-v1/` to
match the pinned DwarfStar capture bit-for-bit.
