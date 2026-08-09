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

- Phase: initial Rust/Metal ownership boundary ready for target-Mac compilation,
  dispatch measurement, and model inspection; `oracle-v1` capture remains
  pending.
- Working branch: `agent/rust-star-bootstrap`.
- Branch base: upstream `antirez/ds4` commit
  `b0309611041655f4e45671cfd9c9886aff161406`.
- Local `main`: fast-forwarded to the same upstream commit.
- Fork `origin/main`: synchronized to the same upstream commit through the
  GitHub app.
- Oracle: candidate source commit selected, but `oracle-v1` is incomplete until
  its model SHA, target-machine toolchain/configuration, and golden artifacts
  are captured.
- Capture kit: `rust-star/capture_oracle_v1.py` prepares a privacy-filtered,
  checksummed result bundle from an isolated build of the pinned oracle source.
- Differential tooling: the initial bundle/full-logit format is stable and has
  cross-platform verification, exact C0 comparison, and drift diagnostics.
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
  correctness-checked shared-buffer dispatch probe. The portable Rust side has
  been compiled and tested with a temporary Rust 1.85 toolchain; the Objective-C
  shim has not been compiled because this workspace has no Apple toolchain.
- Measurements: no model/Metal correctness or performance runs have been made.
- Manual handoff: `RUST_STAR_MANUAL_TASKS.md` records Actions approval, target
  compilation/model inspection, quick/extended oracle capture, and the deferred
  secure-access decision with exact evidence requirements.
- Parallel research: DSpark remains separate on
  `origin/codex/dspark-observability-0` and is not on this phase's critical path.
- Publication: the connected GitHub app is installed on both `deathcoder` and
  `dion-labs`. Remote branch `agent/rust-star-bootstrap` is published. Prefer
  the app for future remote writes from this environment.

## Immediate Next Actions

1. Clone this branch on the M1 Ultra and run `rust-star/check_runtime.sh` with
   the absolute model path; report the complete output before changing the
   validator policy.
2. Run the default 2K/32K oracle capture and return the generated `.tar.gz`
   bundle plus its printed SHA-256.
3. Inspect the quick capture, resolve machine/model-specific failures, and bind
   the validator to the completed `oracle-v1` model SHA-256.
4. Run the extended 2K--1M frontier set only after the quick capture succeeds.
5. Inspect the Metal probe artifact and use the observed roundtrip/batched
   submission ratio to confirm or adjust the ownership boundary.
6. Add a no-copy, read-only model span and one independently testable imported
   DwarfStar kernel before attempting whole-model graph scheduling.
7. Add kernel/layer/decode-step artifact extensions alongside the relevant
   runtime hooks.

## Entries

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
