# Initial Rust/Metal Boundary

Schema: `rust-star-metal-dispatch-probe-v1`.

Rust Star owns device-context lifetime, configuration validation, reporting,
and the future scheduler in Rust. A small Objective-C file owns only the Metal
objects that cannot be expressed through a stable C interface:

- `MTLDevice`, command queue, runtime library, and compute pipeline creation;
- shared-buffer allocation;
- command-buffer/encoder creation, dispatch, commit, and synchronization; and
- Metal GPU start/end timestamps and error extraction.

The shim exposes a narrow C ABI for context lifetime and the probe entry
points. It is
compiled by `build.rs` with the platform Xcode toolchain and has no third-party
Rust dependency. Non-macOS builds compile an explicit unsupported stub, keeping
the GGUF and artifact contracts portable.

The first model boundary extends that ABI without transferring ownership:

- Rust opens and parses the GGUF, then owns one read-only `MAP_SHARED` mapping
  for the complete file.
- The Objective-C shim rounds the validated tensor range to host pages exactly
  as DwarfStar does and creates a shared `MTLBuffer` with
  `newBufferWithBytesNoCopy` and no deallocator.
- The buffer must expose the same page pointer before dispatch. The shader sees
  the tensor only as `device const` memory, and the OS mapping is not writable.
- The Metal buffer is released before control returns to Rust, so the mmap
  cannot be dropped while GPU work or an Objective-C object still references
  it.

## Probe

```sh
cargo run --release --manifest-path rust-star/runtime/Cargo.toml -- \
  metal-probe \
  --elements 4096 \
  --iterations 100 \
  --json /tmp/rust-star-metal-probe.json
```

The probe compiles one integer kernel, validates a shared buffer against a CPU
reference, and measures two submission shapes after a correctness warm-up that
is recorded but excluded from both measurements:

- `roundtrip`: one command buffer, dispatch, commit, and wait per iteration;
- `batched`: all dispatches encoded into one command buffer and synchronized
  once.

Both wall-clock and Metal-reported GPU intervals are retained. The ratio exposes
how much host synchronization/command construction costs relative to batching.
It is not a model throughput benchmark and cannot support an inference speedup
claim.

The JSON records device name, unified-memory support, recommended working-set
size, pipeline width, configuration, timings, rates, and a validated checksum.
It deliberately excludes registry IDs, serial numbers, UUIDs, environment
dumps, and local paths.

## Why this boundary comes first

This resolves the Rust-versus-C++ question empirically at the relevant layer:
Rust is not expected to change GPU kernel throughput. The probe determines
whether a narrow Rust host can create and batch Metal work without hidden
framework overhead, and provides the first target-Mac measurement before the
runtime takes ownership of model buffers or DwarfStar kernels.

The target-Mac dispatch result confirmed that the ownership split is viable;
the no-copy model boundary below is the next completed gate.

## F16 embedding gather

Schema: `rust-star-f16-embedding-probe-v1`.

The first imported kernel is DwarfStar's `kernel_get_rows_f16`, including its
argument layout and threadgroup geometry. It is used for the model's initial
token embedding and therefore lies on both prefill and decode paths.

`embedding-probe` wraps the real `token_embd.weight` payload without copying,
gathers rows at the start, middle, and end of the vocabulary, and compares all
FP32 results by bit pattern against Rust's exact F16 decoder. Its JSON records
page/buffer offsets, pointer identity, output checksum, and dispatch timing but
no mmap address or local model path.

That first gate established mmap and Metal lifetime ownership. The projection
boundary below adds runtime activation arithmetic against a differential
fixture before any whole-model scheduler is introduced.

## Q8_0 decode projection

Schema: `rust-star-q8-0-projection-probe-v1`.

The second imported model kernel is DwarfStar's
`kernel_mul_mv_q8_0_f32`, including its two-stage simdgroup reduction and the
target's default four-simdgroup/two-output-row dispatch. It reads
`blk.0.attn_q_a.weight` directly from the Rust-owned read-only mmap and receives
the activation through one small shared runtime buffer.

The fixture in `../fixtures/q8-attn-q-a-v1/` was captured from the pinned
DwarfStar source and model on the M1 Ultra. DwarfStar's existing graph dump
hooks recorded layer 0 `attn_norm` immediately before this projection and
`q_lora` immediately after it at decode position 1. Its manifest binds the
source tree, model SHA-256, machine, prompt, tensor, layer/step, dispatch
geometry, payload sizes, and artifact SHA-256 values.

`projection-probe` checks the target recipe, fixture dimensions, no-copy
pointer identity, dispatch geometry, every output FP32 bit pattern, and stable
input/output checksums. The JSON deliberately excludes fixture contents, mmap
addresses, and local paths. Its standalone timing includes cold pipeline and
boundary costs and is not a decode-throughput measurement.

## Layer-0 attention ingress

Schema: `rust-star-layer0-attention-ingress-probe-v1`.

`attention-ingress-probe` runs the first connected decode segment in one Metal
command buffer: F16 embedding gather, HC repeat, plain RMSNorm, F16 HC mixer,
fused HC split/collapse plus learned attention norm, and the Q8_0 Q-A
projection. Six independently page-aligned model ranges are wrapped from the
Rust-owned mmap; all six must preserve pointer identity.

The `layer0-attention-ingress-v1` fixture was captured twice in fresh pinned
DwarfStar processes on the M1 Ultra. It retains the 24 mixer values, all 24 HC
split coefficients, the 4,096-value collapsed HC row, the 4,096-value learned
attention norm, and the 1,024-value Q-Lora output. The probe requires every
retained FP32 bit pattern to match. This validates a real token-to-projection
chain without introducing a general graph or allocator framework.

Its standalone timing includes lazy pipeline compilation, cold model pages,
and fixture readback. It is correctness and ownership evidence, not a decoder
throughput result.

## Layer-0 attention projection setup

Schema: `rust-star-layer0-attention-setup-probe-v1`.

`attention-setup-probe` extends the connected command buffer with the Q8_0 KV
projection, DwarfStar's fused Q-Lora/KV learned RMSNorm, and the Q8_0 Q-B
projection. The full path now contains nine dispatches and ten independently
wrapped mmap-backed model ranges.

The `layer0-qkv-setup-v1` fixture preserves Q-Lora and attention-norm inputs,
the raw and learned-normalized 512-value KV row, the learned-normalized
1,024-value Q-Lora row, and the complete 64×512 raw Q tensor. Two fresh pinned
DwarfStar processes produced byte-identical artifacts. The probe requires all
four new runtime boundaries to match bit-for-bit before reporting success.

## Layer-0 RoPE and guarded KV-cache store

Schema: `rust-star-layer0-rope-kv-store-probe-v1`.

`rope-kv-store-probe` adds the three DwarfStar release kernels immediately
before attention: fused per-head Q RMSNorm/RoPE, the in-place KV RoPE, and the
fused E4M3FN KV finalizer/raw-cache store. The build embeds the pinned
`metal/dsv4_rope.metal` and `metal/dsv4_kv.metal` sources directly instead of
maintaining approximate copies.

The `layer0-rope-kv-store-v1` fixture contains two fresh-process captures of
both DwarfStar paths. The diagnostic path exposes standalone `Qnorm`; the
release path uses the fused kernel. Their final `Qcur` artifacts are
byte-identical. `KVrope` and post-FP8 `KVcur` are also captured directly. The
expected cache row is the documented IEEE-754 binary16 round-to-nearest-even
expansion of `KVcur`, matching DwarfStar's half-typed FlashAttention cache.

The runtime writes physical row 1 of a three-row shared cache initialized with
a finite sentinel. It requires the target row to match every expected FP32 bit
and both neighboring rows to retain the sentinel. This proves the first
stateful write boundary.

## Layer-0 raw-cache attention read

Schema: `rust-star-layer0-attention-read-probe-v1`.

`attention-read-probe` extends the connected path to seventeen dispatches. It
imports DwarfStar's contiguous F32-to-F16 cache staging, partial-block padding,
512-wide vector FlashAttention, reduction, and inverse-RoPE kernels. The exact
decode geometry is two raw cache rows, 64 heads, 512 values per head, one
simdgroup per FlashAttention threadgroup, and 32 reduction workgroups.

The `layer0-attention-read-v1` fixture combines fresh-process graph captures at
positions 0 and 1. Cache rows are the documented F16 round/expand forms of the
captured post-FP8 `KVcur` values; `Qcur` and the final `kqv_back` tensor are
captured directly. Independent captures were byte-identical.

The probe initializes row 0 from the oracle, writes row 1 through the already
validated KV-store kernel, and leaves row 2 as a guard. It then requires row 0
to remain unchanged, row 1 to match its oracle, row 2 to retain its sentinel,
and all 32,768 post-inverse-RoPE attention values to match DwarfStar bit-for-bit.

## Layer-0 grouped attention output and HC post-update

Schema: `rust-star-layer0-attention-output-probe-v1`.

`attention-output-probe` extends the connected path to nineteen dispatches and
thirteen independently wrapped mmap-backed model ranges. It imports the two
exact release kernels after inverse RoPE. The first treats the 64 attention
heads as eight fixed groups and applies the block-diagonal Q8_0 output-A
projection to an 8×1,024 low-rank tensor. The second applies output-B and folds
the learned HC post weights plus the four residual streams into the same Q8_0
dispatch, while retaining the 4,096-value block output for diagnostics.

The `layer0-attention-output-v1` fixture contains two byte-identical
fresh-process DwarfStar captures of `attn_low`, `attn_out`, and
`hc_attn_post`. The runtime requires all 8,192 low-rank values, all 4,096
attention output values, and all 16,384 updated HC-state values to match their
FP32 bit patterns exactly.

## Layer-0 FFN HC ingress and hash router

Schema: `rust-star-layer0-ffn-router-probe-v1`.

`ffn-router-probe` continues from the pinned `hc_attn_post` boundary through
seven exact DwarfStar dispatches: plain 16,384-wide RMSNorm, F16 HC mixing,
fused split/collapse plus learned FFN norm, F16 projection to 256 logits,
softplus-plus-square-root probabilities, hash-table expert selection, and
selected-probability normalization with the 1.5 route-weight scale.

Layer 0 is intentionally not probability top-k. The target model contains
`blk.0.ffn_gate_tid2eid.weight`, so token 201 selects experts
`[25, 174, 215, 58, 48, 60]`; probabilities still determine their normalized
weights. The probe wraps the five learned FFN ranges and the I32 hash table
directly from the Rust-owned mmap. All six pointer identities and every
captured FP32/I32 value must match the two byte-identical oracle captures.

## Layer-0 routed/shared experts and FFN HC post-update

Schema: `rust-star-layer0-moe-output-probe-v1`.

`moe-output-probe` consumes the validated router fixture and executes the four
fusions used by the normal M1 release path:

1. IQ2_XXS routed gate/up plus weighted SwiGLU;
2. Q2_K down projection with direct six-expert reduction;
3. Q8_0 shared gate/up plus SwiGLU; and
4. Q8_0 shared down plus routed add and four-stream HC expansion.

The three routed and three shared weight tensors are separate bytes-no-copy
Metal views over the Rust-owned model mmap. The probe checks all six pointer
identities and compares the routed activation, routed output, shared output,
and final HC state bit-for-bit against two fresh-process DwarfStar captures.

## Complete connected layer 0

Schema: `rust-star-layer0-complete-probe-v1`.

`layer0-probe` joins the validated attention, FFN/router, and MoE segments into
one thirty-dispatch Metal command buffer. It does not upload fixture data at
either internal seam: the attention HC output remains a live GPU buffer for
FFN ingress, while the router's selected expert IDs and normalized weights
remain live GPU buffers for the routed-expert kernels.

The command wraps all 25 learned tensor ranges independently from the
Rust-owned read-only mmap and requires every Metal view to preserve its source
pointer. It checks the attention HC state, FFN normalization, route weights,
routed activation and output, shared output, and final four-stream HC state by
FP32 bit pattern. The layer-0 token-ID route must remain
`[25, 174, 215, 58, 48, 60]`.

This establishes the first continuous model layer and its ownership boundary;
it is not a reusable decoder loop yet. The standalone report's timestamps
include fresh-process model/view and pipeline setup plus correctness readback,
so they are deliberately excluded from performance claims.

## Persistent layer-0 steady state

Schema: `rust-star-layer0-steady-state-v1`.

`layer0-bench` separates persistent setup from repeated command execution. It
creates the Metal context, pipelines, all 25 model views, activation buffers,
and cache scratch once, then submits one complete layer command buffer per
iteration. Configurable warm-ups are synchronized but excluded from all
reported samples.

Every measured wall and Metal GPU interval is retained. The JSON also reports
median, median absolute deviation, minimum, and maximum rather than selecting
a best run. Cache inputs are restored before each iteration, the first measured
output becomes the repeat reference, and all later measured checkpoints must
match it byte-for-byte. The final output must also pass the full C0 fixture
comparison in Rust.

This gate measures only the already validated layer-0, position-1 execution
shape. It does not include view/pipeline allocation, model parsing, or fixture
comparison in the samples, but it also does not represent the later-layer
decoder, growing contexts, sampling, or end-to-end token latency.

## Persistent layers 0–1

Schema: `rust-star-layers01-continuous-probe-v1`.

`layers01-probe` moves setup ownership into a narrow Rust `LayerExecutor`.
The executor owns one Metal context, binds it permanently to one Rust model
mmap, caches exact bytes-no-copy model views and activation buffers, and
requires monotonically ordered layer calls.

Layer 0 executes the established thirty-dispatch chain. After its command
buffer completes, layer 1 executes twenty-eight dispatches: it omits token
embedding and HC repeat, and consumes layer 0's retained final HC Metal buffer
directly. Both layers restore their independently captured position-0 cache
row, select their expected experts, and pass bitwise comparisons at every
retained attention, router, expert, and final-HC boundary. The synchronization
between the two command buffers is currently intentional; removing that
boundary belongs to later scheduler work.

## Persistent layers 0–3 and per-layer KV ownership

Schema: `rust-star-layers0123-continuous-probe-v1`.

`layers0123-probe` retains the ordered `LayerExecutor` lifetime and extends it
through layer 3. Cache storage is keyed by layer identity, so layers 0 through
3 own distinct persistent Metal allocations while the four-stream HC state is
the single live cross-layer handoff.

Layer 2 is the first compressed-attention layer. Layers 2 and 3 use
the model's compressed base, scaling, original-context, and YaRN parameters
instead of the dense-layer values used by layers 0 and 1. Layer 3 is also the
first biased-top-k router layer; its 256-value bias replaces the token-indexed
hash table used by layers 0 through 2. The probe compares every retained
attention and MoE boundary against two byte-identical fresh-process captures
and requires the uninterrupted layers 0→1→2→3 chain to remain C0 exact.

## Chained layers 0–3

Schema: `rust-star-layers0123-chained-probe-v1`.

`layers0123-chained-probe` uses the same four command buffers and operation
order as the synchronized control, but commits all four to one Metal queue
before issuing a single tail wait. The queue order supplies the HC dependency;
there is no host upload, event, or fixture seam between layers.

Because correctness readback occurs only after layer 3 completes, this path
keys transient activation buffers by layer as well as retaining the existing
per-layer KV allocations. That lifetime rule prevents later layers from
overwriting earlier C0 checkpoints before collection. Every retained boundary
is compared after the chain completes, and the chained path is rejected on the
first FP32 bit mismatch.

The report records one host wait, total wall time from the first submission to
tail completion, and the sum of the four command-buffer GPU intervals. It
does not promote this four-layer diagnostic to decoder throughput.

## Position-advancing layers 0–3

Schema: `rust-star-layers0123-position-advancing-probe-v1`.

`layers0123-decode-probe` retains one prepared execution object and one raw KV
allocation for each of layers 0–3 across three real decode steps. Token 201 at
position 1 writes raw row 1 and reads rows 0–1. Token 361 at position 2
preserves those rows, writes raw row 2, and reads rows 0–2. Token 1915 at
position 3 writes raw row 3; layer 2 also emits a compressed row, so its
attention reads five rows while the other layers read four. Every step submits
four layer command buffers to the same queue and performs one tail wait.

The ABI carries token and position explicitly. RoPE, inverse RoPE, raw-cache
targeting, visible attention geometry, and hash routing all derive from those
values. Raw KV values are validated after the same FP16 store rounding used by
DwarfStar, including all retained history. The FP32→FP16 attention staging
dispatch scales with the mixed visible row count.

All retained position-2 and position-3 tensor boundaries come from two
byte-identical fresh-process DwarfStar captures. Layer 2 owns ratio-4 attention
and indexer compressor state, reproduces the M1 separate paired-projection and
state-store path, and validates its first 512-value compressed KV emission.
Layer 3 owns ratio-128 attention state and does not emit at this frontier. This
corrects the earlier assumption that both layers used ratio four. The final
layer-3 HC buffer has 16,384 FP32 elements and is the declared output handoff.
The probe still omits the remaining 39 layers, output head, logits, and
sampling, so its timing is not model token throughput.

## Position-advancing layers 0–5

Schema: `rust-star-layers012345-position-advancing-probe-v1`.

`layers012345-decode-probe` generalizes the prepared submission from a fixed
four-layer tail to a checked contiguous tail of six layers. The Objective-C
boundary derives compressor ratio, width, emission, and indexer ownership from
layer parity: even compressed layers use ratio 4 with indexer state; odd layers
use ratio 128 without indexer state. Every compressor, cache, activation, and HC
allocation remains keyed by layer identity.

Two fresh DwarfStar processes produced byte-identical fixtures for every layer
4/5 boundary at positions 0–3. Layer 4 emits and validates its first 512-value
compressed KV row at position 3; layer 5 accumulates ratio-128 state without
emitting. The six-layer path preserves the original four-layer probe as a
separate regression command and remains a partial correctness slice rather than
a decoder-throughput measurement.

## Position-127 ratio-128 compressor replay

Schema: `rust-star-ratio128-compressor-replay-probe-v1`.

`ratio128-compressor-replay-probe` is a deliberately narrower ownership
boundary than the position-advancing layer executor. Two fresh oracle processes
captured all 128 layer-3 and layer-5 `attn_norm` rows plus the first compressed
row emitted by each layer. The fixture importer requires all 258 corresponding
payload pairs to be byte-identical before it packs each activation sequence.

Rust owns one 128-row KV state and score state per replayed layer. Objective-C
wraps the APE, paired projection, and learned-norm weights directly from the
model mmap, submits all 128 recurrent updates in one command buffer, and waits
once. At position 127 it preserves DwarfStar's single-compressed-row operation
boundary: strided state is packed contiguously, then processed by the legacy
softmax, multiply, sum-rows, learned norm, compressed RoPE, and FP8 kernels.
Both 512-value outputs must match their pinned DwarfStar rows by FP32 bit
pattern.

This proves the ratio-128 compressor schedule and state lifetime only. The 128
`attn_norm` rows are external oracle inputs; no token sampling, preceding-layer
execution, logits, or full decoder is part of this command.

## Repeated layers 0–3 steady-state replay

Schema: `rust-star-layers0123-steady-state-v1`.

`layers0123-bench` prepares the four complete layer bindings and host fixture
buffers once, then reuses the same Rust-owned context, no-copy model views,
activation allocations, and per-layer KV storage across warmup and measured
chains. The timing-only collection mode reads the completed command-buffer
timestamps without copying any tensor boundary to the host.

After the last measured chain, one ordinary collection compares every retained
boundary in all four layers. The JSON records all wall and summed-GPU samples,
median/MAD/min/max summaries, the four final fixture identities and routes, and
the post-measurement C0 result.

The replay always executes token 201 at decode position 1 and restores the
pinned cache inputs. It is an execution/scheduler microbenchmark, not a decoder
loop or token-throughput result.
