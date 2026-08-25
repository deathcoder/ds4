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

To complete native 2K prefill through layer 19:

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
final HC update. Every retained layer-7 through layer-19 boundary,
full attention output, compressor state, and full HC identity matches fresh
DwarfStar processes exactly. The combined terminal schedule uses 1,015
dispatches and 524/524 no-copy model views.

Exactly 512 layer-2/layer-4/layer-6/layer-8/layer-10/layer-12/layer-14/layer-16/layer-18
compressed rows still use the dense path; each odd compressed layer through
layer 17 retains 16 ratio-128 rows. The pinned default
remains dense through 1,024 rows; sparse top-k first
applies at 1,025 rows. Layer-20 prefill, sparse post-prompt integration,
complete-model batched prefill, output logits, and throughput remain outside
this command.

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
1,025, and matches 16 tensors by bit pattern across 54 dispatches with 35/35
no-copy model mappings. Empty queue predecessors preserve declared scheduler
ownership without claiming layers 0 and 1 executed. The placeholder token also
means the token-dependent FFN is outside the oracle; complete-layer,
complete-decoder, logits, and throughput claims remain false.

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
and HC checkpoints. All 44 checked boundaries match by bit pattern across 113
total dispatches with 85/85 no-copy mappings; layer 2 accounts for 55 dispatches
and 35/35 mappings. The exact checks continue through selected experts, weighted
SwiGLU, routed/shared outputs, and final layer HC. The workspace allocation is
based on context capacity so its identity remains stable as visible rows grow.
Preceding layers 0 and 1 and the complete retained layer 2 are now claimed; the
seeded prior histories, complete decoder, logits, and throughput remain outside
the claim.

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
