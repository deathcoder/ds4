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
- DwarfStar's native M1 `kernel_mul_mm_q8_0_f32` prefill projection over a
  captured 128-row layer-0 tile, plus a one-row decode control sharing the
  final normalized input. Both match their oracle tensors bit-for-bit and
  preserve the expected cross-schedule arithmetic divergence.
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

To validate the first native batched-prefill arithmetic boundary:

```sh
rust-star/.work/runtime-target/release/rust-star prefill-q8-boundary-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-q8-boundary-probe.json
```

The M1 Ultra disables DwarfStar's Metal 4 TensorOps path, so this command uses
the actual legacy `kernel_mul_mm_q8_0_f32` batch kernel with four 32-row tiles,
sixteen output groups, 128 threads per group, and 6,144 bytes of threadgroup
memory. It requires all 131,072 batch outputs and the final one-row decode
control to match the repeated oracle captures. The two outputs intentionally
differ in all 1,024 final-row values; this localizes the first 2K schedule
divergence but does not claim a complete native prefill implementation.

To extend that native boundary through layer-0 Q/KV setup:

```sh
rust-star/.work/runtime-target/release/rust-star prefill-qkv-boundary-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-qkv-boundary-probe.json
```

This command runs the final 32 prompt rows through Q-A, KV-A, fused Q/KV
learned RMSNorm, Q-B, and Q head RMSNorm/RoPE. It wraps five GGUF ranges without
copying and requires seven independently captured boundaries to match
bit-for-bit. It is the native M1 arithmetic schedule for that layer segment,
not a complete prefill implementation or throughput result.

To move the same final tile's input seam back to token IDs:

```sh
rust-star/.work/runtime-target/release/rust-star prefill-layer0-boundary-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-layer0-boundary-probe.json
```

This command gathers the 32 F16 embeddings, constructs and normalizes the four
HC streams, applies the legacy F16 mixer and fused HC collapse, then continues
through Q/KV setup, KV finalization, guarded raw-cache storage, rectangular
zero-prefix FlashAttention over all 2,048 KV rows, inverse RoPE, grouped
attention output, FFN routing, routed and shared experts, and the additive FFN
HC update in one 43-dispatch command buffer. It wraps 25 GGUF ranges without
copying and requires 6,979,776 produced FP32 values plus 192 selected expert
IDs to match repeated DwarfStar captures bit-for-bit. It completes layer 0 for
the isolated final tile, not the full 2K prefill or a throughput result.

To validate the direct layer-0 to layer-1 HC handoff:

```sh
rust-star/.work/runtime-target/release/rust-star prefill-layers01-boundary-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-layers01-boundary-probe.json
```

This separate control preserves the complete 43-dispatch layer-0 command and
continues in the same command buffer through layer 1's plain four-stream
RMSNorm, F16 HC mixer, fused HC collapse/learned norm, and Q8_0 Q-A projection.
It wraps five additional model ranges without copying and requires the final
`32×4096` HC/norm tiles and `32×1024` Q-Lora tile to match two fresh DwarfStar
captures bit-for-bit. It is not a complete layer-1 or full-prefill claim.

To validate every native layer-1 output row through layer 2's normalized KV
boundary:

```sh
rust-star/.work/runtime-target/release/rust-star \
  prefill-layers012-kvnorm-loop-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-layers012-kvnorm-loop-probe.json
```

This command runs all 2,048 prompt rows as 64 complete native layers-0/1 tiles
with empty-seed live KV ownership. Each live layer-1 output feeds six native
layer-2 dispatches through fused Q/KV learned norm. All 1,048,576 layer-2
`KVnorm` values must match the repeated DwarfStar fixture bit-for-bit, and every
tile must preserve 57/57 no-copy model views. Layer-2 compressed attention and
FFN, later layers, output logits, and throughput remain outside this boundary.

To continue that boundary through compressed-attention RoPE and finalized raw
KV ownership:

```sh
rust-star/.work/runtime-target/release/rust-star \
  prefill-layers012-kv-state-loop-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-layers012-kv-state-loop-probe.json
```

The extended 92-dispatch tile applies layer 2's YaRN parameters, E4M3FN
finalization, and a GPU append into one persistent full-2K layer-2 KV buffer.
Every accumulated prefix plus every `KVnorm`, `KVrope`, and `KVcur` tile must
match the repeated DwarfStar captures bit-for-bit. The attention/indexer
compressors, mixed attention, layer-2 FFN, and later model remain pending.

To complete native 2K model prefill through full logits:

```sh
rust-star/.work/runtime-target/release/rust-star \
  prefill-layers012-attention-loop-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-layers012-attention-loop-probe.json
```

This command retains the same 64-tile context, first requires all 512 layer-2
attention/indexer compressed rows and final recurrent states to match, then
executes full layer-2 Q-B, compressed YaRN, dense mixed FlashAttention over
2,048 raw plus 512 compressed rows, inverse RoPE, both Q8 output projections,
the attention HC update, and the complete routed/shared FFN. It hands the exact
final HC buffer directly into layer 3 without a host upload. Layer 3 then runs
HC ingress, Q/KV setup, compressed RoPE, FP8 raw-KV finalization, its full
ratio-128 attention compressor, dense mixed attention over 2,048 raw plus 16
compressed rows, output projections, additive attention HC post, and the
complete FFN. Its router selects top-6 experts from biased probabilities but
normalizes expert weights from the unbiased probabilities. All compressed
rows, recurrent states, raw-KV rows, attention output, FFN intermediates, final
HC identity, and every retained prior boundary must be bit-identical to
repeated DwarfStar captures. It then completes layer 4 through its paired
ratio-4 compressors, dense mixed attention, biased top-6 routed/shared FFN,
and both additive HC updates. The retained final HC flows directly into layer
5's HC ingress, Q-A/KV projections, fused learned normalization, Q-B,
compressed RoPE, and FP8 KV finalization. All ten layer-5 final-tile boundaries
match four fresh DwarfStar captures bit-for-bit, and the full layer-5 Q/KV
state remains in the persistent Metal context. The same command continues
through layer 5's ratio-128 compressor, dense mixed attention, biased top-6
routed/shared FFN, and final HC update. It then executes layer 6's full Q/KV
state and paired ratio-4 attention/indexer compressors. All 512 layer-6
compressed rows and all four recurrent-state tensors match repeated DwarfStar
captures bit-for-bit. The retained state then drives layer 6's dense mixed
attention, inverse RoPE, grouped/dense output projections, and additive
attention HC post. The complete attention output and full HC identity match
repeated DwarfStar captures exactly. The command then completes layer 6's
learned FFN ingress, biased top-6 routing, routed/shared experts, and additive
final HC update. Every retained FFN boundary and the full final HC identity
match repeated DwarfStar captures exactly. Its retained final HC then flows
directly through layer 7's full Q/KV state, ratio-128 compressor, dense mixed
attention, biased top-6 routed/shared FFN, and additive final HC update. Layer
7's final HC continues directly through layer 8's full Q/KV state, paired
ratio-4 attention/indexer compressors, dense mixed attention, routed/shared
FFN, and additive final HC update. Layer 8's final HC then continues through
layer 9's full Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and additive final HC update, then through layer 10's full
Q/KV state, paired ratio-4 attention/indexer compressors, dense mixed
attention, routed/shared FFN, and additive final HC update, then through layer
11's full Q/KV state, ratio-128 compressor, dense mixed attention,
routed/shared FFN, and additive final HC update. Layer 11's retained final HC
then flows through layer 12's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, then through layer 13's full Q/KV state, ratio-128
compressor, dense mixed attention, routed/shared FFN, and additive final HC
update, then through layer 14's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, then through layer 15's full Q/KV state, ratio-128
compressor, dense mixed attention, routed/shared FFN, and additive final HC
update, and then through layer 16's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 17's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 18's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 19's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 20's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 21's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 22's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 23's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 24's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 25's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 26's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 27's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 28's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 29's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 30's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 31's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, then through layer 32's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 33's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, then through layer 34's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 35's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 36's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 37's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 38's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 39's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 40's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update, and then through layer 41's full Q/KV state,
ratio-128 compressor, dense mixed attention, routed/shared FFN, and additive
final HC update, and then through layer 42's full Q/KV state, paired ratio-4
attention/indexer compressors, dense mixed attention, routed/shared FFN, and
additive final HC update. Every retained layer-7 through layer-42 boundary, full
attention output, compressor state, and full HC identity matches fresh
DwarfStar processes exactly. The combined terminal schedule uses 2,372
dispatches and 1,216/1,216 no-copy model views. The retained final layer-42 HC
row then feeds the exact five-dispatch output head in the same persistent
context. Its five additional model views preserve pointer identity, all
129,280 logits match the independently repeated DwarfStar batched-prefill
frontier bit-for-bit, and lowest-ID argmax selects token 15342. The complete
native model schedule therefore uses 2,377 dispatches and 1,221 no-copy views.

Exactly 512 compressed rows for each even layer from 2 through 42 still use the
dense path; each odd compressed layer through layer 23 retains 16 ratio-128
rows, as do layers 25, 27, 29, 31, 33, 35, 37, 39, and 41. The pinned default
remains dense through 1,024 rows; sparse top-k first
applies at 1,025 rows. This command claims complete-model native batched
prefill, exact output logits, and exact greedy selection. Sparse post-prompt
integration and throughput remain outside its scope.

To exercise the first 4K transformer chunk from the accepted 32K token stream:

```sh
rust-star/.work/runtime-target/release/rust-star \
  long-prefill-transformer-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/long-prefill-transformer-4096.json
```

This timing-only command uses 64 64-row bootstrap tiles, then runs the ordinary
layers-2-through-42 transformer and output head with prompt-derived buffer and
attention extents. It requires 7,556 bootstrap dispatches with 4,160/4,160
no-copy mappings and 2,372 transformer dispatches with 1,216/1,216 no-copy
mappings. The retained context contains 4,096 raw rows per layer, 1,024 rows
for each ratio-4 path, 32 rows for each ratio-128 path, and all compressor
recurrent tails. The initial M1 Ultra run selected token 565 and wrote
`rust-star-long-prefill-transformer-probe-v1` evidence. The accepted 32K token
stream identifies the input only: this command does not claim intermediate C0,
selected-token oracle equality, complete 32K prefill, or throughput. Continue
to use `prefill-layers012-attention-loop-probe` as the exhaustive 2K C0 control.

The next continuation gate is the supplemental
`dwarfstar-oracle-v3-prefill-frontier-8192` fixture. Four fresh processes from
the same accepted synchronized producer agree byte-for-byte on its complete
129,280-logit tensor and select token 77179. The fixture is a C0 target for the
second 4K chunk; its recorded prefill observations are explicitly not benchmark
evidence.

To exercise the retained second chunk and its first exact layer-2 tail:

```sh
rust-star/.work/runtime-target/release/rust-star \
  long-prefill-continuation-bootstrap-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/long-prefill-continuation-8192.json
```

The command completes the first 4K transformer, appends positions
4,096--8,191 through layers 0/1 and the paired layer-2 compressors, and checks
the true position-4,099 sparse-attention boundary. The first 32-row layer-2
tile is fully live and exact through its final HC update. That retained Metal
buffer feeds exact layer-3 ingress and QKV. The native first-4K transformer now
retains the exact pre-4,096 layer-3 attention history: the final 128 raw KV rows
and 32 ratio-128 compressed KV rows. The executor advances that live state,
runs dense mixed attention, inverse RoPE, attention output, biased
routed/shared FFN, and final HC bit-for-bit for positions 4,096--4,127. The
connected layer-3 tile uses 76 dispatches, preserves 28/28 no-copy mappings,
and checks 17 state and downstream outputs. The report schema is
`rust-star-long-prefill-continuation-bootstrap-probe-v8`; it claims an unseeded
complete layer-3 tile, but not the complete 8K transformer, output-logit C0, or
throughput.

The repair has two parts. At 4K, layer 2 has 1,024 ratio-4 compressed rows and
must use DwarfStar's indexed top-512 path rather than dense mixed attention.
The decomposed router kernels must also cover all 4,096 batch rows; their old
2,048-row ceiling silently left the second half of every first-chunk FFN
untouched. The 2K regression remains on the dense 512-row path.

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

This runs eight dispatches over ten no-copy model views and verifies Q-Lora
norm, KV raw/norm, and all 32,768 raw Q values against
`../fixtures/layer0-qkv-setup-v1/`.

The production Q-B projection processes four output rows per workgroup while
retaining the exact per-row Q8_0 accumulation and reduction order. To compare
it with the original two-row geometry or the eight-row control:

```sh
rust-star/.work/runtime-target/release/rust-star qb-rows-bench \
  /absolute/path/to/model.gguf \
  --rows 4 --warmup 10 --iterations 100 \
  --json rust-star/.work/runtime-target/qb-rows-bench.json
```

The command alternates both paths in one Metal context and checks Q-Lora norm,
KV raw/norm, and every raw Q value after each execution. Its timing is a focused
diagnostic rather than a paired full-engine throughput claim.

Production decode also folds the single KV RoPE workgroup into the 64-head Q
RMSNorm/RoPE launch. To replay the separate and fused kernels directly:

```sh
rust-star/.work/runtime-target/release/rust-star q-head-kv-fusion-bench \
  --warmup 10 --iterations 500 \
  --json rust-star/.work/runtime-target/q-head-kv-fusion-bench.json
```

The focused command alternates both paths in one Metal context and checks all
32,768 Q values plus all 512 KV values after every run. The retained
`q-head-threads-bench` control records why the exact 128-thread Q launch was
rejected in favor of the original 256-thread reduction geometry.

To cross the first stateful attention boundary:

```sh
rust-star/.work/runtime-target/release/rust-star rope-kv-store-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/rope-kv-store-probe.json
```

This extends the same path to ten dispatches with Rust Star's combined Q-head
RMSNorm/Q-RoPE/KV-RoPE launch and the fused E4M3FN KV finalizer/cache
store. It matches `Qcur`, `KVrope`, and `KVcur` bit-for-bit, then verifies the
FP16-rounded target cache row while two sentinel neighbor rows remain intact.

To validate the first cache read and attention result:

```sh
rust-star/.work/runtime-target/release/rust-star attention-read-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/attention-read-probe.json
```

This fourteen-dispatch checkpoint stages cache rows 0-1 to F16, executes
DwarfStar's padded 512-wide FlashAttention, and folds inverse RoPE into the
split-K reduction. It preserves the first cache row and guard row and requires the complete
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

To replay the accepted reduction/inverse-RoPE optimization against its
separate-launch control in one Metal context:

```sh
rust-star/.work/runtime-target/release/rust-star attention-rope-fusion-bench \
  /absolute/path/to/model.gguf \
  --warmup 10 --iterations 100 \
  --json rust-star/.work/runtime-target/attention-rope-fusion-bench.json
```

The benchmark alternates the 17-dispatch control and 16-dispatch fused chain,
preserves both the pre- and post-RoPE diagnostic boundaries, and rejects either
path on the first non-bitwise output. Its timings are diagnostic rather than a
paired full-engine throughput claim.

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

To cross the first compressed-attention and router-mode boundaries with
explicit cache ownership:

```sh
rust-star/.work/runtime-target/release/rust-star layers0123-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers0123-probe.json
```

`layers0123-probe` extends the same live HC chain through layer 3 and assigns a
distinct persistent KV-cache allocation to each executed layer. Layers 2–3 use
the model's compressed-attention RoPE parameters (base 160,000, scale 1/16,
65,536-token original context, and YaRN interpolation). Layer 3 also replaces
the first three layers' token-hash router with biased top-k selection using
`blk.3.exp_probs_b.bias`. All four layers must match their independently
captured position-1 DwarfStar checkpoints bit-for-bit.

The exact chained scheduler variant is available separately:

```sh
rust-star/.work/runtime-target/release/rust-star layers0123-chained-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers0123-chained-probe.json
```

It commits the same four per-layer command buffers to one queue without
waiting between layers, then waits once after layer 3. Layer-scoped activation
storage keeps every intermediate boundary alive for post-chain C0 comparison;
the HC dependency remains a direct Metal-buffer edge and KV storage remains
distinct by layer. Its `chain_wall_ms` spans the first submission through the
single tail wait, while `summed_command_gpu_ms` sums the four Metal command
intervals. These are narrow scheduler diagnostics, not decoder throughput.

To validate position advancement and persistent cache growth:

```sh
rust-star/.work/runtime-target/release/rust-star layers0123-decode-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers0123-decode-probe.json
```

`layers0123-decode-probe` prepares layers 0–3 once, executes token 201 at
position 1, token 361 at position 2, and token 1915 at position 3 while reusing
the same four layer-scoped KV allocations. Each step commits four ordered
command buffers and waits once at the tail. The probe requires every retained
attention, router, expert, and HC boundary to match independently repeated
DwarfStar captures bit-for-bit; it also verifies that each raw KV cache
preserves its earlier rows and grows from two visible rows to four. Layer 2
advances its ratio-4 attention and indexer compressor states, emits and checks
its first compressed KV row, and appends that row to its position-3 attention
input. Layer 3 advances its ratio-128 attention compressor without emitting.
Layer 3's final 16,384-element HC state is the explicit output handoff.

This is the first stateful decode slice, but not a complete decoder or a
throughput benchmark. It covers four of 43 layers and does not produce logits
or sample a next token. The compression schedule is layer-specific: layer 2
uses ratio 4 and emits at position 3, while layer 3 uses ratio 128 and will not
emit until position 127. This corrects the earlier documentation statement
that both layers used ratio 4.

To exercise generalized compressor ownership through the next schedule pair:

```sh
rust-star/.work/runtime-target/release/rust-star layers012345-decode-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers012345-decode-probe.json
```

`layers012345-decode-probe` retains the same positions 1–3 and exact four-layer
regression boundary, then continues the live HC handoff through layers 4 and 5.
Layer 4 owns a second ratio-4 attention/indexer compressor pair and validates
its emitted position-3 KV row; layer 5 owns non-emitting ratio-128 attention
state. The six command buffers share one queue and one tail wait per step, and
all six raw caches and compressor states remain layer-scoped. This still covers
only six of 43 layers and is a correctness diagnostic, not token throughput.

To extend the same exact stateful boundary through the next compressor pair:

```sh
rust-star/.work/runtime-target/release/rust-star \
  layers01234567-decode-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers01234567-decode-probe.json
```

`layers01234567-decode-probe` preserves the four-layer and six-layer controls,
then continues the live HC handoff through layers 6 and 7. Layer 6 owns the
third ratio-4 attention/indexer compressor pair and validates its position-3
emission; layer 7 advances non-emitting ratio-128 attention state. Eight raw KV
caches and all compressor allocations remain layer-scoped, with eight ordered
command buffers and one tail wait per position. The command covers eight of 43
layers and remains a correctness diagnostic rather than token throughput.

To execute the same exact boundary through the complete transformer stack:

```sh
rust-star/.work/runtime-target/release/rust-star \
  layers0-42-decode-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/layers0-42-decode-probe.json
```

`layers0-42-decode-probe` uses one checked fixture registry for layers 4–42
and preserves the four-, six-, and eight-layer commands as separate controls.
It executes 43 ordered command buffers with one tail wait per position, retains
43 raw KV caches plus layer-scoped compressor state, validates every even-layer
position-3 compressed KV emission through layer 42, and exposes layer 42's
exact 16,384-element HC state. Two fresh oracle processes produced 3,448
byte-identical payload pairs for the newly added layers 8–42 before fixture
import. This remains the transformer-stack regression control; the separate
command below extends it through the exact output head.

To validate output normalization, the complete vocabulary projection, and
deterministic next-token selection:

```sh
rust-star/.work/runtime-target/release/rust-star \
  decoder-output-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/decoder-output-probe.json
```

`decoder-output-probe` retains the same 43 layer-scoped KV caches, then runs
DwarfStar's plain HC RMSNorm, F16 four-way HC projection, output-HC weights,
fused HC collapse/learned RMSNorm, and full 129,280-row Q8_0 vocabulary
projection. It compares all five output tensors by FP32 bit pattern at
positions 1–4 and applies lowest-token-ID argmax on the CPU. The fixed input
tokens `[201, 361, 1915, 262]` select `[361, 1915, 262, 1554]`; the final step
is the first to consume persistent compressed rows emitted by the prior step. The correctness schedule
uses 44 command buffers and two host waits per step. Inputs are still supplied
externally, so this is not a closed-loop generator or a token-throughput
benchmark.

To verify the feedback edge and exercise the corresponding readback-free timed
path:

```sh
rust-star/.work/runtime-target/release/rust-star \
  closed-loop-decoder-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/closed-loop-decoder-probe.json
```

`closed-loop-decoder-probe` first performs exhaustive C0 collection while
feeding each lowest-ID argmax result into the next position. It then reuses the
prepared model views, Metal pipelines, and retained allocations for a separate
timed pass. Those intervals include all 44 command buffers, both required tail
waits, the 129,280-logit transfer used for CPU argmax, and the argmax itself;
they exclude tensor-fixture readback and comparison. Its four-token rate is a
diagnostic only. The command reports `paired_protocol_eligible: false` because
the captured slice does not yet implement cold prefill or arbitrary frontiers.
The longer command below now covers the protocol-length transcript.

To exercise the integrated decoder through its first ratio-128 emissions:

```sh
rust-star/.work/runtime-target/release/rust-star \
  position127-decoder-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/position127-decoder-probe.json
```

`position127-decoder-probe` owns one prepared 43-layer executor and advances it
through positions 1–127. Together with initial committed token 201, the 127
greedy selections must reproduce the complete 128-token DwarfStar transcript.
After the last synchronized step it compares all 129,280 final logits and reads the
first persistent ratio-128 compressed-cache rows from layers 3 and 5; all three
boundaries must be FP32 bit-identical to independently repeated oracle data.

The timed interval ends before those correctness readbacks and comparisons and
reports evaluated positions per second. It is still diagnostic and sets
`paired_protocol_eligible: false`: the starting raw caches and compressor state
are captured fixtures rather than the result of cold prompt prefill, and an
arbitrary benchmark frontier cannot yet be initialized.

To replace that captured initial state with a live one-token prefill:

```sh
rust-star/.work/runtime-target/release/rust-star \
  cold-prefill-decoder-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/cold-prefill-decoder-probe.json
```

`cold-prefill-decoder-probe` initializes empty raw KV, attention-compressor,
indexer-compressor, and compressed-cache storage. It evaluates raw prompt token
36662 at position 0, requires every one of its 129,280 logits and selected token
201 to match two fresh DwarfStar captures, and retains the resulting live state
for positions 1–127. The complete 128-token transcript, final logits, and
layer-3/layer-5 ratio-128 rows must remain exact.

This command never copies the captured position-0 cache rows or compressor
activations into Metal. It remains `paired_protocol_eligible: false` because it
only prefills the one-token correctness frontier; the paired protocol requires
fresh multi-token prefill at arbitrary 2K–256K and later context frontiers.
Its diagnostic prefill interval includes full-logit projection and CPU greedy
selection and therefore is not yet the engine-measurement `prefill_ms` field.

To exercise context-sized state through the first protocol frontier:

```sh
rust-star/.work/runtime-target/release/rust-star \
  prefill-frontier-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-frontier-probe.json
```

`prefill-frontier-probe` evaluates the canonical 2,048 token IDs from empty
state, uses a 128-row raw-KV ring, and allocates 514 ratio-4 or 18 ratio-128
compressed rows per applicable layer. Its final logits match two independent
DwarfStar replays built from one-token prefill plus 2,047 ordinary decode calls
bit-for-bit. The retained `capture_decode_replay.c` helper defines that
diagnostic oracle boundary.

From the repository root, rebuild that helper against the same DwarfStar
objects used by `ds4-bench` with:

```sh
cc -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -I. \
  -o rust-star/.work/capture-decode-replay \
  rust-star/capture_decode_replay.c \
  ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o \
  ds4_layer_pack.o -lm -pthread -framework Foundation -framework Metal

rust-star/.work/capture-decode-replay \
  /absolute/path/to/model.gguf \
  speed-bench/promessi_sposi.txt \
  2048 \
  rust-star/.work/decode-replay-logits.f32le.bin
```

This is not batched-prefill C0. Two independent ordinary `ds4-bench` captures
also agree with each other, but all 129,280 batch-frontier logits differ from
the decode replay (maximum absolute error 2.325326) while both select token
15342. The JSON records both facts and remains `paired_protocol_eligible:
false`. The next inference boundary is native batched prefill; the following
decode boundary is sparse indexer top-k once ratio-4 memory exceeds the pinned
default threshold of 1,024 rows. The model top-k itself remains fixed at 512.

To validate that mechanism at both the small diagnostic boundary and the
production default's first sparse row:

```sh
rust-star/.work/runtime-target/release/rust-star \
  sparse-indexed-attention-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/sparse-indexed-attention-probe.json
```

The command first preserves the repeated layer-2 position-2051 oracle control
made with `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=512`, then validates two
fresh production-default captures at position 4099 and 1,025 compressed rows.
It reproduces the F16 indexer projections, compressed RoPE, QAT, direct scores,
the two-block argsort plus merge, exact descending top-512, 12-way indexed mixed
attention, reduction, and inverse RoPE bit-for-bit while preserving all three
mmap model pointers. This proves the default switch layer segment, not complete
decode, output logits, or throughput. The retained even-layer decoder now owns
the same schedule. Its top-k scratch is fixed from context capacity, while the
active sort blocks, compact work width, ping-pong offsets, merge count, and
final top-512 dispatch are derived from the visible compressed rows.

To prove the same boundary through the general retained executor:

```sh
rust-star/.work/runtime-target/release/rust-star \
  retained-sparse-boundary-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/retained-sparse-boundary-probe.json
```

This diagnostic seeds the captured layer-2 HC, 127 prior raw-ring rows, 1,024
prior attention/indexer compressed rows, and both pre-update recurrent states.
It then runs the ordinary retained position-4099 layer schedule, commits row
1,025, and matches 30 tensors by bit pattern across 52 dispatches with 35/35
no-copy model mappings. The captured router table uniquely recovers oracle
input token 7129, so token-hash selection, routed/shared experts, and final HC
are all inside the C0 gate. Empty queue predecessors preserve declared
scheduler ownership without claiming layers 0 and 1 executed. The complete
retained layer is proven; complete-decoder, logits, and throughput claims
remain false.

To validate the first schedule that needs repeated merge passes:

```sh
rust-star/.work/runtime-target/release/rust-star \
  retained-sparse-multimerge-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/retained-sparse-multimerge-probe.json
```

This control seeds the prior 127-row raw-ring histories for layers 0-2 and the
2,048 prior rows plus recurrent states in both layer-2 compressed paths. It does
not seed incoming HC. Layers 0 and 1 execute normally for token 381, their live
HC handoff feeds layer 2, and the ordinary retained schedule commits row 2,049,
emits three initial sort blocks, and performs two ping-pong merge passes. Two
fresh DwarfStar processes independently pin both predecessor cache histories
and HC checkpoints. All 44 checked boundaries match by bit pattern across 105
total dispatches with 85/85 no-copy mappings; layer 2 accounts for 53 dispatches
and 35/35 mappings. The exact checks continue through selected experts, weighted
SwiGLU, routed/shared outputs, and final layer HC. The workspace allocation is
based on context capacity so its identity remains stable as visible rows grow.
Preceding layers 0 and 1 and the complete retained layer 2 are now claimed; the
seeded prior histories, complete decoder, logits, and throughput remain outside
the claim.

To validate the exact native-prefill handoff and closed-loop sparse frontier:

```sh
rust-star/.work/runtime-target/release/rust-star \
  prefill-decode-frontier-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/prefill-decode-frontier-probe.json
```

This command starts with the complete exact 2K batched-prefill schedule and
keeps its raw KV, compressed caches, recurrent compressor states, and final HC
resident in the same Metal context. One blit command performs 229 GPU-to-GPU
copies: the final 128 raw rows for all 43 layers, attention compressed/state
storage for layers 2--42, and indexer compressed/state storage for every even
layer. No state is read through the host during adoption.

The resulting decoder greedily evaluates positions 2048--4099 against a
2,052-token transcript captured bit-identically in two fresh DwarfStar
processes. It requires the complete position-2048 and position-4099 logits to
match by bit pattern. Position 4099 commits compressed row 1,025 before the
same step's fixed top-512 lookup and therefore exercises all 21 even layers at
the production-default sparse threshold. Per-position synchronization and
correctness checks remain enabled, so the reported 16.563 positions/s from the
first focused run is diagnostic and not a throughput result.

To exercise the first engine-measurement process boundary at the exact 2K/128
workload:

```sh
rust-star/.work/runtime-target/release/rust-star \
  engine-measure \
  /absolute/path/to/model.gguf \
  --context 2048 \
  --gen-tokens 128 \
  --json rust-star/.work/runtime-target/rust-star-engine-run.json
```

The prefill interval follows the same native Metal schedule as the exact C0
command, but does not decode diagnostic output fixtures, allocate host boundary
tensors, or copy transformer boundaries to the host. It retains only the GPU
state required by the decoder and transfers the final logits for lowest-ID
argmax. Before that interval closes, all established no-copy model views are
registered in a Metal residency set attached to the inference queue and receive
a synchronized one-mebibyte-stride GPU touch. The raw timing record accounts
for the view, allocation, and touch counts plus wall/GPU time; that cost remains
inside `prefill_ms`. The generation intervals include command encoding,
synchronized transformer and output-head execution, CPU argmax, and token
commitment. They
do not collect diagnostic tensors; the complete 128-token selection transcript
is compared with the pinned oracle after timing. The raw producer is
`paired_protocol_eligible: true` only when both collection paths are disabled.
Run it through `measure_ruststar.py` for normalized, checksummed evidence.

Timing-only prefill uses 32 64-row bootstrap command buffers, submits them
continuously to the serial inference queue, and synchronizes once at the final
tile. Matrix and FlashAttention grids plus grouped-output routing storage are
derived from the row count; this corrects the fixed-32 launch geometry that
invalidated the first wide-row experiment. Queue order preserves live KV and
compressor dependencies without 31 inter-tile host waits. Diagnostic commands
retain their independently synchronized 64-by-32-row C0 schedule. The raw
record separately reports aggregate bootstrap wall/GPU span,
remaining-transformer wall/GPU time, output-head time, handoff time, residency
time, and residual host overhead. Passive timers further split host work around
context/bootstrap/transformer/output setup, the decoder-state handoff, and
model residency, leaving an explicit unattributed remainder. A full 2,048-row
bootstrap remains rejected because its raw-cache append is not a valid
unwrapped 2K store.

`engine-profile` is an explicitly paired-ineligible diagnostic. In addition to
its decode attribution, it splits the otherwise single layers-2-through-42
prefill transformer command into synchronized per-layer command buffers and
records each completed command buffer's Metal GPU interval. Representative
even layer 4 and odd layer 5 receive an attention-to-FFN split plus five
attention-internal intervals: QKV/RoPE, compressor/staging, Flash block
preparation, FlashAttention, and output/HC. The JSON report labels all 52
diagnostic command buffers and host waits, requires the 41 layer intervals to
sum to the reported transformer GPU time, requires each representative
attention breakdown to sum to its attention interval and both attention/FFN
pairs to sum to their layer totals, and blocks paired use. Normal
`engine-measure` neither enables this state nor changes the production prefill
command schedule.

The production 512-wide non-vector FlashAttention specialization uses four
SIMD groups per 32-thread-wide launch. The function constant and all 43 matching
prefill/decode dispatches share `kRustStarFlashNonvecNsg`, preventing a pipeline
and launch-geometry mismatch. Three exact M1 Ultra profiles reduced the median
representative Flash intervals by 63.4% and 65.4% relative to the earlier
eight-group geometry; three eligible 2K/128 controls improved median prefill
from 138.083 to 171.682 tok/s.

To validate the complete retained step at the same position:

```sh
rust-star/.work/runtime-target/release/rust-star \
  retained-decoder-step-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/retained-decoder-step-probe.json
```

Two fresh DwarfStar processes pin identical pre-step state, every layer's final
HC, and the full output vector. Rust seeds 127 prior raw rows in all 43 layers,
the 2,048/64 compressed histories and recurrent state in the alternating
ratio-4/ratio-128 layers, and the complete ratio-4 indexer histories. It then
executes layers 0 through 42 with live HC handoffs and runs the output head.
All 43 HCs and all 129,280 logits match by bit pattern, selecting token 35,597
across 1,813 transformer dispatches and 1,370/1,370 no-copy model mappings.
This is a complete retained decoder step with seeded history, not native
prefill or a throughput result.

To cross the first ratio-128 emission boundary without overstating decoder
coverage:

```sh
rust-star/.work/runtime-target/release/rust-star \
  ratio128-compressor-replay-probe \
  /absolute/path/to/model.gguf \
  --json rust-star/.work/runtime-target/ratio128-compressor-replay-probe.json
```

`ratio128-compressor-replay-probe` consumes 128 pinned DwarfStar `attn_norm`
rows for each of layers 3 and 5. It replays the exact paired F16 projection,
APE state store, legacy single-row softmax/multiply/sum reduction, learned
normalization, compressed RoPE, and FP8 finalization. Each layer owns its own
128-row recurrent state, uses four mmap-backed model ranges without copying,
and must reproduce its position-127 `KVcompress` row bit-for-bit.

The intermediate activations are externally supplied oracle evidence. The
probe does not execute layers 0–5 for positions 4–127, produce logits, or sample
tokens, so its timing is a compressor correctness diagnostic rather than token
throughput.

For repeated fixed-position execution with preparation and correctness
collection outside the measured interval:

```sh
rust-star/.work/runtime-target/release/rust-star layers0123-bench \
  /absolute/path/to/model.gguf \
  --warmup 5 \
  --iterations 20 \
  --json rust-star/.work/runtime-target/layers0123-bench.json
```

`layers0123-bench` resolves all 100 layer-specific model spans, decodes the four
pinned fixtures, and allocates host result storage once. Each timed iteration
then commits four already-prepared layer calls, waits once at the tail, and
reads only Metal command-buffer timing metadata. No activation, router, expert,
cache, or HC boundary is copied to the host inside a measured interval. After
the final sample, the normal collector performs one exhaustive comparison of
all four layers against the pinned DwarfStar fixtures.

This is a fixed token-201, position-1 steady-state replay. It isolates the
current four-layer execution/scheduler cost but does not advance a decoder,
sample tokens, or establish end-to-end token throughput.
