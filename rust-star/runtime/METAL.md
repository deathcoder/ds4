# Initial Rust/Metal Boundary

Schema: `rust-star-metal-dispatch-probe-v1`.

Rust Star owns device-context lifetime, configuration validation, reporting,
and the future scheduler in Rust. A small Objective-C file owns only the Metal
objects that cannot be expressed through a stable C interface:

- `MTLDevice`, command queue, runtime library, and compute pipeline creation;
- shared-buffer allocation;
- command-buffer/encoder creation, dispatch, commit, and synchronization; and
- Metal GPU start/end timestamps and error extraction.

The shim exposes a narrow C ABI for context lifetime and the two probes. It is
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

With this gate complete, the next increment is a decode projection/matvec that
consumes both model weights and an activation. Whole-model scheduling remains
out of scope until that second arithmetic boundary has a differential fixture.
