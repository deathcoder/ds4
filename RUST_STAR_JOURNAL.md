# Rust Star Work Journal

This is the operational source of truth for current state, evidence, blockers,
and next actions. Stable scope and contracts live in `RUST_STAR_PROJECT.md`.
User/hardware/account-dependent work lives in `RUST_STAR_MANUAL_TASKS.md`.

## Resume Protocol

Before changing code:

1. Read `RUST_STAR_PROJECT.md` and this file completely.
2. Run `git status --short --branch`, inspect `git log -1`, and verify remotes.
3. Confirm that the current-state summary still matches the repository.
4. Check whether the active oracle manifest and benchmark inputs are pinned.
5. Update this journal before handoff, context compaction, or the end of a
   meaningful session.

Journal entries are reverse chronological. Do not edit old entries to change
history; add a correction and update the current-state summary.

## Current State

- Phase: target-Mac bootstrap, quick `oracle-v1`, no-copy model mapping, the
  canonical differential-fixture envelope, and the connected layer-0
  attention path through Q/K RoPE and its first exact KV-cache row write are
  complete; the next increment is the first layer-0 attention scan/readback.
- Working branch: `agent/rust-star-bootstrap`.
- Branch base: upstream `antirez/ds4` commit
  `b0309611041655f4e45671cfd9c9886aff161406`.
- Local `main`: fast-forwarded to the same upstream commit.
- Fork `origin/main`: synchronized to the same upstream commit through the
  GitHub app.
- Oracle: `oracle-v1` is complete and independently verified. It binds source
  `b030961`, model SHA-256 `ca22ae2f...6261c0`, the target toolchain, 2K/32K
  full-vocabulary logits, correctness logs, and repeated performance evidence.
- Capture kit: `rust-star/capture_oracle_v1.py` prepares a privacy-filtered,
  checksummed result bundle from an isolated build of the pinned oracle source.
- Differential tooling: the initial bundle/full-logit format is stable and has
  cross-platform verification, exact C0 comparison, and drift diagnostics. The
  `rust-star-differential-fixture-v1` envelope now covers kernel,
  layer-segment, and decode-step scopes with strict tensor shape, path,
  finiteness, size, and hash verification.
- Benchmarking: `rust-star/BENCHMARK_PROTOCOL.md` fixes the v1 paired workload,
  eligibility, run ordering, aggregation, and capacity semantics; execution
  awaits the target Mac and a runnable candidate. The paired raw/summary JSON
  contract and offline validator/aggregator are implemented and synthetically
  tested. A fresh-process DwarfStar adapter now normalizes one frontier while
  externally recording wall time and peak RSS. The checkpointed paired runner
  enforces warm-up, A/B and context ordering, explicit retries, and final
  validated export through an engine-neutral contract.
- Implementation: dependency-free Rust host scaffold under `rust-star/runtime/`
  strictly parses GGUF v3 directories, validates the Flash resident-Q2
  shape/recipe, and writes candidate full-logit artifacts. A macOS-only
  Rust/Objective-C/Metal boundary now owns a device, queue, pipeline, and
  correctness-checked shared-buffer dispatch probe. The complete Objective-C
  shim, optimized Rust build, shared-buffer probe, and real-model validator now
  pass on the M1 Ultra. The real GGUF exposed and fixed an incorrect Q8_0
  expectation for the F16 indexer Q projection. Rust now owns a read-only
  shared GGUF mmap; Metal wraps the page-aligned embedding tensor without a
  copy and runs the imported DwarfStar F16 embedding gather behind a bitwise
  CPU reference. The imported decode-time Q8_0 matvec now consumes a real
  layer-0 activation and the no-copy `blk.0.attn_q_a.weight` span, reproducing
  all 1,024 FP32 DwarfStar fixture outputs by bit pattern. The first connected
  chain now runs six imported operations in one command buffer from token 201's
  embedding through Q-Lora, with six mmap-backed model spans and bitwise checks
  at mixer, HC split, collapsed HC, learned attention norm, and Q-Lora.
  The continuation adds Q8 KV projection, fused Q-Lora/KV learned RMSNorm, and
  Q-B in the same command buffer, yielding full raw Q and normalized KV with
  ten mmap-backed model spans. The twelve-dispatch continuation imports the
  exact pinned RoPE and KV-finalizer sources, produces final Q/K, FP8-rounds KV,
  and writes an FP16-rounded physical cache row while preserving guard rows.
- Measurements: Metal batching was 42.861x faster than synchronized submission
  in the retained M-002 probe. DwarfStar medians are 164.86 prefill / 19.90
  generation tok/s at 2K and 161.05 prefill / 17.36 generation tok/s at 32K.
  The first real-model kernel matched all 20,480 checked FP32 values; its final
  validation dispatch reported 0.020 ms GPU time and is not an inference-speed
  claim. The Q8_0 projection matched its 1,024-value decode fixture and reported
  0.031 ms GPU time in its completion gate, also not a throughput claim. The
  connected six-dispatch layer segment matched 9,264 retained FP32 values and
  reported 0.102 ms GPU time in the final gate; this remains correctness
  evidence rather than decoder throughput. The nine-dispatch Q/K projection
  setup matched another 34,816 FP32 values and reported 0.473 ms GPU time in
  its final gate, also not a throughput claim. The twelve-dispatch RoPE/cache
  gate matched 34,816 direct outputs plus the 512-value stored row, preserved
  two 512-value guard rows, and reported 0.949 ms GPU time; this is likewise
  correctness evidence rather than decoder throughput.
- Manual handoff: `RUST_STAR_MANUAL_TASKS.md` records Actions approval, target
  compilation/model inspection, quick/extended oracle capture, and the deferred
  secure-access decision with exact evidence requirements.
- Parallel research: DSpark remains separate on
  `origin/codex/dspark-observability-0` and is not on this phase's critical path.
- Publication: the connected GitHub app is installed on both `deathcoder` and
  `dion-labs`. Remote branch `agent/rust-star-bootstrap` is published and the
  validated checkpoints are pushed. Prefer the app for future remote writes
  from this environment.

## Immediate Next Actions

1. Extend the connected layer-0 path through the uncompressed layer-0
   FlashAttention scan over the now-validated raw cache and validate its
   attention output before inverse RoPE.
2. Define the smallest reusable Rust buffer/scheduling abstraction justified by
   the now-connected path; do not introduce a general graph framework yet.
3. Run the extended 2K--1M frontier capture when the Mac can be dedicated to a
   long benchmark; preserve any 512K/1M capacity failure as evidence.
4. Run or approve the fork's GitHub Actions workflow and retain its URL.

## Entries

### 2026-08-14 — Layer-0 RoPE and first KV-cache write matched DwarfStar

Objective:

- Cross the first stateful decode boundary by finalizing Q/K position encoding
  and proving an exact, isolated cache-row mutation.

Oracle fixture:

- Captured layer 0, decode position 1 in four fresh pinned DwarfStar processes:
  two with standalone diagnostic Q normalization and two with the release
  fused Q-normalization/RoPE path.
- Every repeated artifact was byte-identical. The fused and diagnostic paths
  also produced byte-identical final `Qcur` tensors.
- Added `rust-star/fixtures/layer0-rope-kv-store-v1/` with raw Q/normalized KV
  inputs, diagnostic `Qnorm`, release `Qcur`, pre-store `KVrope`, post-E4M3FN
  `KVcur`, and the documented FP16-rounded cache-row derivation.

Implementation:

- Embedded the pinned DwarfStar `metal/dsv4_rope.metal` and
  `metal/dsv4_kv.metal` sources directly in the runtime Metal library.
- Extended the connected command buffer with fused per-head Q RMSNorm/RoPE,
  layer-0 KV RoPE, and the fused E4M3FN KV finalizer/raw-cache store.
- Added a three-row cache probe initialized with a finite sentinel. The kernel
  targets physical row 1; Rust checks every target-row bit and both neighboring
  guard rows.
- Added a stable CLI/JSON schema, fixture and unit coverage, documentation, and
  automatic execution in `check_runtime.sh`.

Target-Mac evidence:

- All ten model views retained exact mmap pointer identity.
- All 32,768 `Qcur`, 512 `KVrope`, 512 post-FP8 `KVcur`, and 512 cache-row
  values matched the pinned evidence by FP32 bit pattern. Both 512-value guard
  rows remained unchanged.
- The final gate reported 40.305 ms wall and 0.949 ms Metal GPU time. This is a
  cold standalone correctness boundary, not decoder throughput.
- The complete gate passed: 24 Rust tests, optimized macOS build, 30 Python
  tests, all four fixture verifiers, cross-language C0 smoke, strict validation
  of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept the first exact layer-0 cache mutation as the stateful attention
  checkpoint. Add the layer-0 attention scan next, without compression or
  output projection yet.

Next:

- Capture/import the uncompressed layer-0 FlashAttention read over positions
  0..1 and validate the pre-inverse-RoPE attention output.

### 2026-08-14 — Layer-0 Q/K projection setup matched DwarfStar

Objective:

- Continue the real token path beyond Q-Lora through the complete raw Q and
  learned-normalized KV projection setup.

Oracle fixture:

- Captured layer 0, decode position 1 `q_lora_norm`, `KVraw`, `KVnorm`, and
  `Qraw` from the pinned DwarfStar executable in two fresh processes; every
  payload was byte-identical across captures.
- Added `rust-star/fixtures/layer0-qkv-setup-v1/`, retaining the independent
  attention-norm/Q-Lora inputs and all four new boundaries. The fixture contains
  six tensors totaling 159,744 bytes and binds the same source, executable,
  model, prompt, machine, token, layer, and position identities.

Implementation:

- Imported DwarfStar's fused Q-Lora/KV learned RMSNorm, preserving the default
  norm-unification formula and dispatch geometry.
- Extended the connected command buffer with Q8_0 KV projection, fused Q/K
  norm, and Q8_0 Q-B projection. The path now has nine dispatches and ten
  independently page-aligned mmap-backed model views.
- Added a separate stable probe schema/CLI/JSON report, fixture verification,
  unit coverage, documentation, and automatic target-model gate execution.

Target-Mac evidence:

- All ten model views retained exact mmap pointer identity.
- All 1,024 Q-Lora norm values, 512 KV raw values, 512 KV norm values, and
  32,768 raw Q values matched DwarfStar by FP32 bit pattern.
- The final gate reported 38.399 ms wall and 0.473 ms Metal GPU time. This is a
  cold standalone correctness boundary, not decoder throughput.
- The complete gate passed: 22 Rust tests, optimized macOS build, 29 Python
  tests, all three fixture verifiers, cross-language C0 smoke, strict validation
  of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept raw Q plus normalized KV as the next connected layer checkpoint. Keep
  head normalization/RoPE and cache mutation separate so the first stateful
  boundary has its own exact evidence.

Next:

- Capture/import Q head RMSNorm and RoPE plus KV RoPE, then validate the first
  layer-0 cache-row write without adding the attention scan yet.

### 2026-08-14 — Connected layer-0 attention ingress matched DwarfStar

Objective:

- Replace the isolated projection input with a real token embedding and make
  kernel/layer/decode-step evidence independently verifiable.

Artifact contract and oracle fixture:

- Added `rust-star-differential-fixture-v1` with kernel, layer-segment, and
  decode-step scopes, ordered operations, uniquely named tensor boundaries,
  safe relative paths, exact FP32 shapes/encodings, finite-value checks, byte
  counts, and SHA-256 verification.
- Migrated the Q8 projection fixture to the new schema and added a standalone
  verifier that the full runtime gate applies to every committed fixture.
- Captured layer 0, decode position 1 from the pinned DwarfStar executable in
  two fresh processes. All mixer, HC split, collapsed HC, attention norm, and
  Q-Lora payloads were byte-identical across captures.
- Retained seven tensor artifacts totaling 37,056 bytes under
  `rust-star/fixtures/layer0-attention-ingress-v1/`. The fixture records token
  201 and binds the pinned source/tree, executable, model, prompt, machine, and
  graph-dump selection.

Implementation:

- Imported DwarfStar's F16 gather, HC repeat, plain RMSNorm, vectorized F16
  matvec, fused HC split/collapse/weighted norm, and Q8_0 matvec.
- Generated the Objective-C NSString representation of the auditable Metal
  source at build time, keeping the source readable while retaining a
  dependency-free runtime binary.
- Encoded all six operations into one command buffer. Rust retains ownership
  of the read-only whole-model mmap; the shim independently wraps the six
  page-aligned weight ranges with `newBufferWithBytesNoCopy`.
- Added a stable JSON report, CLI command, tests, documentation, and automatic
  execution in the target-model runtime gate.

Target-Mac evidence:

- All six model views preserved exact mmap pointer identity.
- All 24 HC mixer values, 24 split coefficients, 4,096 collapsed HC values,
  4,096 learned-norm values, and 1,024 Q-Lora values matched DwarfStar by FP32
  bit pattern.
- The final gate reported 37.782 ms wall and 0.102 ms Metal GPU time. It
  includes a standalone correctness boundary and is not a decode-throughput
  claim.
- The complete gate passed: 21 Rust tests, optimized macOS build, 28 Python
  tests, both committed fixture verifiers, cross-language C0 artifact smoke,
  strict validation of all 1,288 required GGUF tensors, and every existing and
  new Metal probe.

Decision:

- Accept the connected attention-ingress segment as the first layer-level
  execution checkpoint. Continue immediately after Q-Lora while preserving
  narrow artifacts and avoiding a premature graph framework.

Next:

- Capture and import Q-Lora normalization/Q-B plus the paired KV setup needed
  for the layer-0 attention core.

### 2026-08-14 — First quantized decode projection matched DwarfStar

Objective:

- Validate quantized model bytes, a real runtime activation, DwarfStar's
  decode reduction order, and Rust/Metal ownership in one isolated operation.

Oracle fixture:

- Ran pinned DwarfStar source `b030961` on the M1 Ultra with one prefill token
  and one greedy decode token, filtering its existing graph dump hooks to layer
  0, position 1, `attn_norm` and `q_lora`.
- Retained the 4,096-value input and 1,024-value output under
  `rust-star/fixtures/q8-attn-q-a-v1/`. The manifest binds source tree, model
  SHA-256, machine, prompt, layer/step, tensor, dispatch geometry, and artifact
  SHA-256 values.
- Repeated the complete one-token DwarfStar capture in a fresh process; both
  activation and output files were byte-identical to the retained fixture.

Implementation:

- Imported DwarfStar's `kernel_mul_mv_q8_0_f32` arithmetic and its default four
  simdgroup/two-row dispatch behind the narrow Objective-C ABI.
- Reused the Rust-owned read-only `MAP_SHARED` model mapping and wrapped only
  the page-aligned `blk.0.attn_q_a.weight` span with
  `newBufferWithBytesNoCopy`; the runtime activation uses one small shared
  buffer.
- Added strict fixture/tensor/dispatch validation, bitwise output comparison,
  stable privacy-limited JSON, CLI support, fixture hash tests, documentation,
  and automatic target-model execution in `check_runtime.sh`.

Target-Mac evidence:

- The Q8_0 tensor begins at byte 79,060,632,640 and contains 4,456,448 bytes.
  Metal wrapped a 4,472,832-byte page-aligned view with a 1,088-byte inner
  offset and exact pointer identity.
- All 1,024 output FP32 bit patterns matched DwarfStar. Input/output checksums
  are `6001855774483604828` and `13770952831385691371`.
- The final standalone dispatch reported 3.990 ms wall and 0.031 ms GPU time;
  pipeline/boundary setup makes this correctness evidence, not decode
  throughput.
- The complete gate passed: 20 Rust tests, optimized macOS build, both no-copy
  model kernels, Metal dispatch/shared-buffer validation, 26 Python tests,
  cross-language C0 artifact smoke, and strict validation of all 1,288 required
  GGUF tensors.

Decision:

- Accept the no-copy Q8_0 matvec as the first runtime-activation arithmetic
  boundary. Build the next increment toward a minimal layer-0 chain and extend
  the artifact format at each new boundary.

Next:

- Specify the kernel/layer artifact envelope around this fixture, then derive
  layer-0 `attn_norm` from the token embedding and feed it into the validated
  projection.

### 2026-08-14 — No-copy model span and first DwarfStar kernel validated

Objective:

- Cross the first real-model execution boundary without adding a graph,
  allocator framework, or whole-model scheduler.

Changes:

- Added a dependency-free Rust `MappedModel` that parses the GGUF and owns one
  read-only `MAP_SHARED` mapping for its complete lifetime.
- Added a page-aligned Objective-C wrapper using
  `newBufferWithBytesNoCopy`; it requires the returned Metal contents pointer
  to equal the mmap pointer and releases the buffer before Rust can unmap.
- Imported DwarfStar's `kernel_get_rows_f16` argument layout, kernel body, and
  threadgroup geometry. Its pipeline is compiled lazily so the original
  dispatch probe's compile-time metric remains comparable.
- Added exact Rust F16-to-F32 reference conversion, bitwise comparison, a stable
  `rust-star-f16-embedding-probe-v1` JSON artifact, CLI support, documentation,
  and automatic target-model execution in `check_runtime.sh`.

Target-Mac evidence:

- Strict validation selected `token_embd.weight` at byte 77,928,033,088 with
  1,059,061,760 payload bytes. Metal wrapped a 1,059,078,144-byte page-aligned
  view with an 11,072-byte inner offset and exact pointer identity.
- Five rows spanning token IDs 0 through 129,279 produced 20,480 FP32 values;
  every bit matched the CPU reference. The retained checksum is
  `4028716204944876325`.
- The final validation dispatch reported 17.632 ms wall and 0.020 ms GPU time.
  This includes a cold standalone boundary and is correctness evidence, not a
  throughput claim.
- The complete runtime gate passed: 18 Rust tests, optimized macOS build, Metal
  dispatch/shared-buffer validation, 25 Python tests, cross-language C0 smoke,
  strict real-GGUF inspection, and the new no-copy gather.

Decision:

- Accept this mmap/Metal lifetime split as the first model ownership boundary.
  Keep the next increment to one decode projection/matvec before designing the
  graph scheduler.

Next:

- Capture or derive an exact DwarfStar fixture for one Q8_0 decode projection,
  then run the same weight span and activation through Rust Star.

### 2026-08-14 — Target-Mac bootstrap and oracle-v1 completed

Objective:

- Validate the Rust/Metal boundary and pin the DwarfStar oracle on the target
  M1 Ultra before implementing model execution.

Actions and evidence:

- Created a dedicated `agent/rust-star-bootstrap` worktree so the DSpark
  observability branch can continue in a separate Codex task.
- Installed the missing standard `rustfmt` component and ran the full M-002
  gate on the M1 Ultra.
- Fixed the target recipe for `indexer.attn_q_b.weight` from Q8_0 to F16 and
  added a regression test covering every ratio-4 indexer layer.
- Added `pipefail` to the documented evidence command after the first strict
  inspection failure was masked by `tee`.
- Passed 15 Rust tests, 25 Python tests, the optimized macOS build, the C0
  cross-language artifact check, shared-buffer validation, and strict real-GGUF
  inspection of all 1,288 required tensors.
- Captured and independently verified the complete 2K/32K `oracle-v1` bundle.
  Correctness, full-vocabulary frontier logits, and all six timed measurements
  passed. The archive SHA-256 is `5115f445...a09ee3c`.

Decision:

- Accept the quick capture as `oracle-v1`. Keep the extended context capture
  ready but move the critical implementation path to no-copy model mapping and
  the first isolated kernel.

Next:

- Implement and test the no-copy, page-aligned read-only GGUF span boundary,
  then import one decode-critical DwarfStar kernel behind C0 fixtures.

### 2026-08-09 — Initial Rust/Metal ownership and dispatch boundary added

Objective:

- Turn the host-only scaffold into the smallest testable Metal runtime boundary
  without claiming model inference or performance before the M1 Ultra run.

Changes:

- Added a macOS-only Objective-C ARC shim, compiled by Cargo through the Xcode
  toolchain, that owns the Metal device, queue, runtime-compiled probe pipeline,
  shared buffer, command encoders, synchronization, and GPU timestamps.
- Added a safe Rust lifecycle/configuration/reporting layer and `metal-probe`
  command. It compares synchronized per-dispatch command buffers with one
  batched command buffer, validates every resulting element against a CPU
  reference, and can atomically emit a privacy-limited stable JSON artifact.
- Kept non-macOS host contracts buildable through an explicit unsupported stub,
  added a macOS CI compilation job, and made the target-Mac check script run the
  probe before model inspection.
- Applied rustfmt to the complete pre-existing Rust crate when a temporary
  formatter became available; the GGUF/target diffs are formatting-only.
- Documented the ownership boundary in `rust-star/runtime/METAL.md` and extended
  manual task M-002 to request the resulting probe artifact.

Validation in this workspace:

- `check_runtime.sh` passed without a model using a temporary Rust 1.85
  toolchain: rustfmt, 15 Rust unit tests, optimized host build, all 25 Python
  contract tests, and the Rust-writer/Python-reader C0 smoke check succeeded.
- Python compilation, workflow YAML parsing, shell syntax, an independent Rust
  grammar parse, and `git diff --check` passed. The non-macOS `metal-probe`
  command failed explicitly as designed.
- One initial optimized link in the generated in-repository target directory
  reported unresolved thin-LTO symbols; an immediate rebuild and the complete
  check script then passed. Preserve macOS CI and clean-target evidence rather
  than treating this temporary-toolchain anomaly as target evidence.
- This environment still has no Xcode/Metal device or model. The Objective-C
  shim has not been compiled, and no Metal execution, model correctness run, or
  performance measurement has been performed here.

Next:

- Compile and run the complete check script on the M1 Ultra. If the boundary
  validates, add a no-copy read-only model span and one isolated DwarfStar
  kernel while preserving C0 differential hooks.

### 2026-08-09 — Checkpointed paired runner and Rust-side contract added

Objective:

- Complete the engine-neutral A/B orchestration before a Rust Star decoder is
  available, so target-Mac measurements cannot drift into an ad hoc procedure.

Changes:

- Added `rust-star/ENGINE_MEASUREMENT_FORMAT.md` as the shared fresh-process
  contract. DwarfStar implements it now; the Rust Star benchmark must emit the
  same metric semantics directly or through a thin adapter.
- Added `rust-star/paired_runner_lib.py` and `run_paired_benchmark.py` with an
  immutable SHA-bound plan, atomic `state.json` checkpointing after every engine
  process, untimed per-engine warm-ups, alternating A/B and context ordering,
  and optional one-pair-at-a-time execution.
- Timed failures block without automatic retries. Retrying requires an explicit
  reason, reruns the complete pair, retains the old attempt/evidence, and
  preserves already-finalized outputs under `superseded/`.
- Adapter invocations use argument arrays without a shell and fresh process
  groups. Timeouts and interrupts terminate the process group before control
  returns; a hard-interruption checkpoint remains blocked for deliberate review.
- Finalization occurs only after every predeclared pair has one final valid
  attempt. It re-hashes and revalidates every successful engine measurement,
  emits the strict raw artifact, revalidates that artifact, and writes the
  paired summary.
- Added `rust-star/PAIRED_RUNNER.md` and manual task M-005 for the eventual first
  target-Mac paired run. Local plans may contain private paths and are never
  copied into the shared result directory.

Validation:

- All 25 Python contract tests passed. Synthetic end-to-end tests cover pause
  and resume, engine/context ordering, blocked failures, explicit full-pair
  retry, plan immutability, measurement-tamper rejection, validated final
  export, and preservation of superseded results.
- Python compilation, both new CLI help paths, workflow/shell static checks,
  and `git diff --check` passed.
- No real engine, model, Metal, correctness, or performance run was possible in
  this environment.

Next:

- Implement the Rust Star measurement producer alongside the first runnable
  decoder, then instantiate a private paired plan from the completed oracle
  manifest rather than inventing identities in advance.

### 2026-08-09 — Isolated DwarfStar measurement boundary added

Objective:

- Make the pinned oracle measurable through the future paired-runner boundary
  without modifying DwarfStar or waiting for a Rust Star decoder.

Changes:

- Added `rust-star/measure_dwarfstar.py` and
  `dwarfstar_measurement_lib.py`. Each invocation runs one fresh `ds4-bench`
  process at one frontier, preserves its CSV and sanitized logs, and emits a
  checksummed `rust-star-engine-measurement-v1` record.
- Added external complete-process wall time and per-child peak RSS collection.
  Prefill, full-generation, and steady-generation durations are reconstructed
  from DwarfStar's reported counts/rates; the residual is explicitly named
  process overhead rather than inaccurately calling it model-load time.
- Documented a DwarfStar semantic trap: at a terminal single frontier,
  `kvcache_bytes=0` means no serialized session snapshot was created. The
  normalized contract now records `null`, not zero live KV usage.
- Hardened the adapter against missing/partial CSV output, partial generation,
  non-finite metrics, non-empty result directories, timeouts, and private path
  leakage in captured logs.
- Added `rust-star/MEASUREMENT_ADAPTER.md` and included the adapter in the host
  workflow path filters.

Validation:

- Python compilation, CLI help, shell syntax, workflow YAML parsing, and
  `git diff --check` passed.
- All 20 Python contract tests passed. Synthetic child-process tests cover CSV
  normalization, per-process measurements, path redaction, and persisted
  adapter-validation failures.
- No real DwarfStar/model/Metal measurement was run in this environment.

Next:

- Define the Rust Star adapter against the same single-process schema, then add
  a checkpointed paired orchestrator that enforces A/B and context order.

### 2026-08-09 — Manual ledger and paired benchmark v1 defined

Objective:

- Preserve every user/hardware/account-dependent handoff in one actionable
  ledger and remove benchmark-method ambiguity before measurements begin.

Changes:

- Added `RUST_STAR_MANUAL_TASKS.md` with status rules, security boundaries,
  exact commands, dependencies, evidence requirements, and success conditions
  for GitHub Actions approval, target-Mac runtime/model inspection, quick and
  extended oracle capture, and the deferred secure remote-access decision.
- Added `workflow_dispatch` to the host-contract workflow so it can be run from
  GitHub after fork Actions are enabled or approved.
- Added `rust-star/BENCHMARK_PROTOCOL.md` as
  `rust-star-paired-benchmark-v1`. It fixes C0 eligibility, batch-one closed-loop
  decode semantics, 256K primary context, development/full context sets,
  committed-token metrics, alternating paired order, repetition counts,
  thermal controls, raw-data retention, pairwise speedup aggregation, and
  capacity/failure treatment.
- Added `rust-star/PAIRED_RESULT_FORMAT.md`, a strict standard-library parser,
  `summarize_paired_benchmark.py`, and synthetic tests for machine-readable raw
  pairs, host/correctness manifests, exact build/runtime identities,
  predeclared coverage/order, invalid-pair retention, operational metrics, C0
  headline eligibility, and within-pair speedup aggregation.
- Updated the durable project open item to distinguish defining the protocol
  from executing it and collecting the first DwarfStar numbers.

Validation:

- Python compilation passed for the paired parser, CLI, and tests.
- All 16 Python contract tests passed, including cross-context schedule and
  invalid-retry cases.
- `git diff --check` and workflow/shell static checks passed.
- No model, Metal, correctness, or performance run was possible in this
  environment.

Next:

- Define the narrow benchmark-runner interface that will produce paired raw
  rows from DwarfStar and the future Rust executable without coupling either
  engine to the aggregator.

### 2026-08-09 — Strict Rust host-runtime scaffold added

Objective:

- Establish a narrow, executable Rust boundary that can be prepared without
  Metal or model access and prevents unsupported models from entering the
  future engine accidentally.

Changes:

- Added a dependency-free Rust 2021 crate at `rust-star/runtime/` with a
  committed lockfile and a minimum Rust version of 1.74.
- Added a bounded GGUF v3 parser that reads metadata/tensor directories without
  reading tensor payloads. It rejects duplicate names, unknown types, malformed
  booleans/UTF-8, excessive counts/strings, rank/size arithmetic overflow,
  invalid alignment, payload overlap, and out-of-file ranges.
- Derived and encoded the exact DwarfStar Flash shape: 43 layers, width 4096,
  vocabulary 129280, 64 attention heads, 256 experts with 6 active, and the
  model's compression schedule and semantic metadata.
- Added the initial resident imatrix-Q2 recipe validator: IQ2_XXS routed gate/up,
  Q2_K routed down, Q8_0 attention/shared/output, and F16 HC/compressor/indexer.
- Kept checkpoint identity honest: GGUF name metadata is informational; the
  completed oracle's whole-model SHA-256 remains necessary to prove `0731`.
- Added a full-FP32-logit writer with finite-value checks, nine-significant-digit
  output, signed-zero preservation, and lowest-token-ID argmax tie semantics.
- Added `rust-star/check_runtime.sh` to run formatting, Rust unit tests, release
  build, existing Python artifact tests, a Rust-writer/Python-reader C0 smoke
  check, and optional strict model inspection on the target Mac.
- Added a path-filtered GitHub Actions job pinned to Rust 1.74 so formatting,
  compilation, unit tests, release build, and the cross-language artifact smoke
  check can be validated before target-Mac access is available.

Validation in this workspace:

- All Rust files passed an independent Rust grammar parse.
- Shell syntax and `git diff --check` passed.
- All nine existing Python artifact/bundle tests passed.
- The check script correctly exits with a clear Rust-toolchain prerequisite in
  this container.

Not validated here:

- This environment has neither `rustc` nor Cargo, so type-checking, rustfmt, and
  the new Rust unit tests are pending the target-Mac run.
- No real GGUF was available. The first strict inspection may expose a justified
  difference between the actual oracle recipe and the currently derived
  DwarfStar template policy; any change must be evidence-driven and journaled.
- No Metal code, inference, correctness run, or performance measurement exists
  in the scaffold yet.

Next:

- Run `./rust-star/check_runtime.sh /absolute/path/to/model.gguf` on the M1
  Ultra, then run the quick `oracle-v1` capture.

### 2026-08-09 — Offline oracle validation and C0 comparator added

Objective:

- Progress the correctness infrastructure without access to the M1 Ultra or
  model, and give the future Rust runtime an exact output contract.

Changes:

- Added `rust-star/ARTIFACT_FORMAT.md` as the stable initial bundle and
  post-prefill full-logit contract.
- Added `rust-star/artifact_lib.py` with safe bundle extraction, artifact hash
  verification, strict finite-FP32 parsing, bit-pattern comparison, and drift
  metrics.
- Added `rust-star/verify_oracle_bundle.py` to validate a result directory or
  `.tar.gz`, including its sibling/explicit archive SHA-256.
- Added `rust-star/compare_logits.py`; it exits zero only for C0 by default and
  reports mismatch/ULP/error, cosine, KL, argmax, and top-k diagnostics for
  valid non-C0 artifacts.
- Added `rust-star/tests/test_artifact_tools.py` with synthetic fixtures for
  exact equality, signed zero, drift, metadata, tampering, partial manifests,
  and archive traversal.
- Updated the capture README and made tar extraction behavior explicit across
  supported Python versions.

Validation:

- Python compilation succeeded for the capture, library, both CLIs, and tests.
- Nine unit tests passed with deprecation warnings promoted to errors.
- CLI integration checks confirmed C0 exit 0, drift exit 1, and bundle/archive
  verification with the sibling checksum.
- `git diff --check` passed.
- No Metal, model, or performance run was attempted in this environment.

Next:

- Begin a strict, platform-independent Rust runtime scaffold while target-Mac
  capture remains pending.

### 2026-08-09 — Target-Mac oracle capture kit prepared

Objective:

- Prepare everything that can be cloned and executed on the M1 Ultra without
  granting this environment remote access to the machine.

Changes:

- Added `rust-star/capture_oracle_v1.py`, a one-command Python standard-library
  capture tool.
- Added `rust-star/README.md` with quick, extended, custom-context, reporting,
  and security instructions.
- Git-ignored local `rust-star/results/` and `rust-star/.work/` outputs.
- The tool hashes the entire GGUF; captures allowlisted hardware, macOS,
  compiler, Metal, power, and load metadata; exports pinned upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406`; and builds it in a temporary
  directory.
- It runs `--metal-kernels` and `--logprob-vectors`, captures full post-prefill
  FP32 frontier logits separately from timed runs, and aggregates repeated
  batch-one decode/prefill measurements with median, MAD, minimum, and maximum.
- The default quick set is 2K/32K with three repetitions. The opt-in full set is
  2K, 32K, 128K, 256K, 512K, and 1M.
- Result manifests are updated after each completed gate/run so partial evidence
  survives interruption or a long-context capacity failure.
- The shareable archive excludes the model, full environment, unrelated process
  listings, credentials, absolute model path, and Mac serial/UUID identifiers.

Validation in this Linux workspace:

- `python3 -m py_compile rust-star/capture_oracle_v1.py`
- CLI help and default/full dry-run plans.
- Invalid duplicate-context rejection.
- Median/MAD aggregation helper checks against synthetic rows.
- Pinned source commit/tree validation, offline `git archive` extraction,
  deterministic 1M prompt expansion, and command-path sanitization.
- `git diff --check` and ignore-rule verification.

Not validated here:

- No Metal build, GGUF load, correctness gate, logit dump, or performance run
  was possible because this workspace is not the target Mac and has no model.

Next:

- Run the documented default capture on the M1 Ultra and return its archive and
  SHA-256 for inspection before attempting the extended context set.

### 2026-08-09 — Fork synchronized and research branch published

Objective:

- Publish the bootstrap work after correcting the GitHub App installation
  scope.

Actions and evidence:

- Confirmed a new personal-account installation for `deathcoder` was visible
  alongside the existing `dion-labs` installation.
- Fast-forwarded `deathcoder/ds4:main` to upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406` without force.
- Created remote branch `agent/rust-star-bootstrap` from that commit.
- Recreated the three-file local documentation tree atomically through GitHub's
  blob, tree, commit, and ref APIs. The initial published commit was
  `6daf28c3faee652463fe0957397e023e4ae7fa98`; its tree
  `e9761869fc0a45fc3880ff42d133deb9c0da2371` exactly matched the local tree.
- No pull request was opened because branch publication, not upstream review,
  was requested.

Next:

- Capture the oracle manifest and baseline on the target M1 Ultra, then define
  the differential golden-artifact format.

### 2026-08-09 — Publication 403 root cause identified

Objective:

- Determine why other GitHub-connected sessions can publish while writes to
  this fork consistently fail.

Evidence:

- The authenticated GitHub user is `deathcoder`.
- The app installation/account inventory contains only organization
  `dion-labs` (installation `152207005`).
- There is no GitHub App installation for the personal `deathcoder` account.
- `deathcoder/ds4` repository metadata reports the user's collaborator
  permission (`push: true`), but Git database writes execute through an app
  installation token. No installation token is scoped to this personal fork.

Conclusion:

- The 403 is not caused by the branch name, upstream commit, or use of the
  low-level ref endpoint. The app is installed on the wrong owning account for
  this repository. Successful sessions likely target repositories covered by
  the `dion-labs` installation or use separately authenticated local git.

Next:

- Install the GitHub app on the `deathcoder` personal account and grant it
  access to `deathcoder/ds4`, then retry ref creation/update through the app.

### 2026-08-09 — GitHub app publication retry

Objective:

- Retry fork synchronization and branch publication through the connected
  GitHub app, and preserve that app as the preferred publication mechanism.

Evidence:

- The app successfully read `deathcoder/ds4` and reported `admin: true` and
  `push: true` repository permissions.
- It successfully resolved upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406` through the fork.
- Updating `refs/heads/main` to that commit failed with GitHub HTTP 403:
  `Resource not accessible by integration`.
- Creating `agent/rust-star-bootstrap` from that commit failed with the same
  GitHub HTTP 403 from the create-reference endpoint.

Decision:

- Future sessions should try the GitHub app first, but must confirm an actual
  ref write succeeds. Reported repository permission metadata alone is not
  sufficient evidence that the installation token can publish.

Next:

- Refresh or reconnect the GitHub app installation with repository Contents
  write access, then retry the two ref operations before recreating the local
  documentation commit on the remote branch.

### 2026-08-09 — Bootstrap branch and durable project memory

Objective:

- Synchronize the fork work with upstream, create an isolated research branch,
  and preserve the brainstorming decisions across handoffs and compaction.

Repository actions:

- Cloned `deathcoder/ds4` as `origin`.
- Added `antirez/ds4` as `upstream`.
- Verified the fork's `main` was 137 commits behind and 0 ahead of upstream.
- Fast-forwarded local `main` from `80ebbc396aee40eedc1d829222f3362d10fa4c6c`
  to upstream `b0309611041655f4e45671cfd9c9886aff161406`.
- Created local branch `agent/rust-star-bootstrap` at that upstream commit.
- Added `RUST_STAR_PROJECT.md`, this journal, and an `AGENT.md` pointer that
  requires future Rust Star sessions to read and update them.

Decisions recorded:

- Target only the 48-GPU-core, 128 GB M1 Ultra Mac Studio and
  DeepSeek-V4-Flash-0731 initially.
- Start from the resident imatrix Q2 model and a minimal Rust decoder/benchmark.
- Require C0 full-FP32-logit bit equivalence in the default path.
- Version the upstream DwarfStar oracle rather than treating a moving branch as
  immutable truth.
- Make batch-one committed decode t/s primary; retain prefill, TTFT, memory,
  long-context, low-concurrency, and agent-loop metrics as research scope.
- Deprioritize speculative decoding while retaining the existing DSpark branch
  as evidence and a possible later source of proven mechanisms.
- Use 256K as an important operating point and test correct scaling toward 1M
  rather than creating an undefined separate capacity mode.

Validation:

- No build, model correctness test, or performance benchmark was run. This
  change contains project documentation only.

Blockers/notes:

- The connected GitHub integration could read the repositories but returned
  HTTP 403 for branch/ref writes. Local git work proceeded instead.
- A non-interactive `git push origin main:main` also failed because this
  workspace has no HTTPS GitHub username/token configured. Remote fork
  synchronization and branch publication remain pending write credentials.
- SSH was not a viable fallback: no authentication agent was available and the
  workspace's restricted network could not resolve GitHub on the SSH path.

Next:

- Publish the already committed local state from a GitHub-authenticated
  environment, then capture the oracle manifest and baseline on the target Mac.
