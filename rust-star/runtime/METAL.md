# Initial Rust/Metal Boundary

Schema: `rust-star-metal-dispatch-probe-v1`.

Rust Star owns device-context lifetime, configuration validation, reporting,
and the future scheduler in Rust. A small Objective-C file owns only the Metal
objects that cannot be expressed through a stable C interface:

- `MTLDevice`, command queue, runtime library, and compute pipeline creation;
- shared-buffer allocation;
- command-buffer/encoder creation, dispatch, commit, and synchronization; and
- Metal GPU start/end timestamps and error extraction.

The shim exposes three C-ABI calls: create, run the probe, and destroy. It is
compiled by `build.rs` with the platform Xcode toolchain and has no third-party
Rust dependency. Non-macOS builds compile an explicit unsupported stub, keeping
the GGUF and artifact contracts portable.

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

The next Metal increment should wrap a read-only page-aligned model span as a
no-copy buffer and execute one imported, independently testable kernel. Whole
model loading and graph scheduling should not be attempted until that memory
ownership boundary is measured and validated.
