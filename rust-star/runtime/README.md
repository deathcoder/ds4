# Rust Star host-runtime scaffold

This crate is the smallest executable boundary for the new engine. It is
model-specific by design and currently contains one connected nineteen-dispatch
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
- the continuation through KV projection, fused Q-Lora/KV learned RMSNorm, and
  Q-B, producing the full 64×512 raw Q tensor and normalized KV row in the same
  command buffer.
- the first stateful boundary: fused Q head RMSNorm/RoPE, KV RoPE, E4M3FN KV
  finalization, and an FP16-rounded cache-row write with untouched guard rows.
- the first stateful read: exact raw-cache F16 staging, padded 512-wide
  FlashAttention/reduction over positions 0-1, and inverse RoPE, checked against
  DwarfStar's complete 64×512 `kqv_back` boundary.
- DwarfStar's grouped Q8 attention output projection and fused four-stream HC
  post-update, checked at the 8×1,024 low-rank, 4,096 output, and 4×4,096
  updated-state boundaries.

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

To extend that chain through the Q/K projection setup:

```sh
rust-star/.work/runtime-target/release/rust-star attention-setup-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/attention-setup-probe.json
```

This runs nine dispatches over ten no-copy model views and verifies Q-Lora
norm, KV raw/norm, and all 32,768 raw Q values against
`../fixtures/layer0-qkv-setup-v1/`.

To cross the first stateful attention boundary:

```sh
rust-star/.work/runtime-target/release/rust-star rope-kv-store-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/rope-kv-store-probe.json
```

This extends the same path to twelve dispatches with DwarfStar's fused Q head
RMSNorm/RoPE, the layer-0 KV RoPE, and its fused E4M3FN KV finalizer/cache
store. It matches `Qcur`, `KVrope`, and `KVcur` bit-for-bit, then verifies the
FP16-rounded target cache row while two sentinel neighbor rows remain intact.

To validate the first cache read and attention result:

```sh
rust-star/.work/runtime-target/release/rust-star attention-read-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/attention-read-probe.json
```

This seventeen-dispatch checkpoint stages cache rows 0-1 to F16, executes
DwarfStar's padded 512-wide FlashAttention and reduction, then applies inverse
RoPE. It preserves the first cache row and guard row and requires the complete
64×512 `kqv_back` result to match the pinned DwarfStar oracle bit-for-bit.

To validate the attention output and HC state update:

```sh
rust-star/.work/runtime-target/release/rust-star attention-output-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/attention-output-probe.json
```

This adds the two release Q8 kernels used by DwarfStar: the eight-group
block-diagonal projection to 8×1,024 low-rank values, followed by the fused
8,192-to-4,096 expansion and four-stream HC post-update. All three retained
boundaries must match the oracle bit-for-bit.
