# Rust Star host-runtime scaffold

This crate is the smallest executable boundary for the new engine. It is
model-specific by design and currently contains the complete thirty-dispatch
layer-0 decode path in one Metal command buffer, not a decoder.

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
- the layer-0 FFN HC collapse/learned norm and F16 router projection, followed
  by the model's exact token-ID hash selection and probability-derived route
  weights; six mmap-backed model views and every retained boundary match C0.
- the four-dispatch release MoE suffix: paired IQ2_XXS routed gate/up with
  weighted SwiGLU, Q2_K six-expert down reduction, paired Q8_0 shared gate/up
  with SwiGLU, and Q8_0 shared down fused with the four-stream FFN HC update.
  Six model ranges remain mmap-backed and all retained boundaries match C0.
- a connected layer-0 gate that passes the live attention HC state into the
  FFN/router kernels and their live selected IDs and weights into the expert
  kernels without fixture handoffs. All thirty dispatches share one command
  buffer, all 25 model ranges remain mmap-backed, and every retained boundary
  through the final four-stream HC state matches C0 bit-for-bit.
- a persistent steady-state gate that creates those 25 model views and all
  activation buffers once, excludes configurable warm-up iterations, retains
  every raw wall/GPU timing sample, and rejects any measured iteration whose
  outputs differ by one bit from the first or from the pinned oracle.

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

To validate the layer-0 FFN ingress and router boundary:

```sh
rust-star/.work/runtime-target/release/rust-star ffn-router-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/ffn-router-probe.json
```

This seven-dispatch continuation begins with the pinned `hc_attn_post` state,
applies FFN HC mixing/collapse and learned RMSNorm, projects 256 router logits,
then follows DwarfStar's early-layer token-ID hash route for token 201. It
requires bitwise equality for the HC intermediates, logits, probabilities,
six selected experts, and scaled weights, with all six model ranges zero-copy.

To validate the routed/shared experts and finish the layer-0 state update:

```sh
rust-star/.work/runtime-target/release/rust-star moe-output-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/moe-output-probe.json
```

This four-dispatch continuation starts from the pinned normalized FFN row,
selected expert IDs, scaled route weights, HC residual, and HC split. It wraps
the three full routed-expert tensors and three shared-expert tensors directly
from the model mmap, executes DwarfStar's exact release fusions, and requires
bitwise equality for all 6×2,048 routed activations, both 4,096-wide outputs,
and the final 4×4,096 HC state.

To validate the complete connected layer-0 path:

```sh
rust-star/.work/runtime-target/release/rust-star layer0-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layer0-probe.json
```

This is the first probe with no fixture handoff between attention, routing,
and expert execution. The GPU-produced `hc_attn_post` buffer feeds the FFN
continuation directly, and the GPU-produced expert IDs and route weights feed
the routed kernels directly. The report requires one command buffer, thirty
dispatches, exact pointer identity for all 25 model views, the expected six
experts, and bitwise equality at every retained checkpoint through
`hc_ffn_post`.

Its wall and GPU intervals include fresh-process mapping, Metal view creation,
pipeline setup, and fixture readback. They are diagnostic only and must not be
used as layer latency or decoder-throughput evidence.

To measure repeated execution after that setup is complete:

```sh
rust-star/.work/runtime-target/release/rust-star layer0-bench \
  /absolute/path/to/model.gguf \
  --warmup 10 \
  --iterations 30 \
  --json rust-star/.work/runtime-target/layer0-bench.json
```

`layer0-bench` keeps the Metal context, compiled pipelines, 25 no-copy model
views, cache scratch, and activation buffers alive across all iterations. Each
iteration still uses one synchronized command buffer containing thirty
dispatches. Warm-ups are excluded; JSON retains every measured wall and GPU
sample plus median, MAD, minimum, and maximum. The command requires all
measured outputs to be mutually bit-identical and the final outputs to match
the same C0 fixtures as `layer0-probe`.

This is credible steady-state evidence for the isolated layer-0 path at the
pinned two-row attention geometry. It is not a full-decoder benchmark and must
not be multiplied by the layer count to claim model throughput: later layers,
the recurrent decoder state, cache growth, final normalization, sampling, and
host scheduling are not represented yet.

To validate the first cross-layer scheduler boundary:

```sh
rust-star/.work/runtime-target/release/rust-star layers01-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers01-probe.json
```

`layers01-probe` creates one Rust-owned executor, binds it to one immutable
model mapping, and runs complete layers 0 and 1 in order. Exact model views
and activation buffers remain cached for that executor's lifetime. Layer 1
skips embedding/repeat and reads layer 0's final four-stream HC state directly
from the retained Metal buffer; no host upload or fixture handoff occurs at
the layer seam. Each layer uses one synchronized command buffer, and both
layers must match their independently captured DwarfStar checkpoints
bit-for-bit. This proves the two-layer ownership and state-handoff contract,
not yet a full decoder loop.
