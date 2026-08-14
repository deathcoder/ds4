# Initial Rust/Metal Boundary

Schema: `rust-star-metal-dispatch-probe-v1`.

Rust Star owns device-context lifetime, configuration validation, reporting,
and the future scheduler in Rust. A small Objective-C file owns only the Metal
objects that cannot be expressed through a stable C interface:

- `MTLDevice`, command queue, runtime library, and compute pipeline creation;
- shared-buffer allocation;
- command-buffer/encoder creation, dispatch, commit, and synchronization; and
- Metal GPU start/end timestamps and error extraction.

The shim exposes a narrow C ABI for context lifetime and the six probes. It is
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
