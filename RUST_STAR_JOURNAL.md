# Rust Star Work Journal

This is the operational source of truth for current state, evidence, blockers,
and next actions. Stable scope and contracts live in `RUST_STAR_PROJECT.md`.

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

- Phase: `oracle-v1` capture tooling ready; target-Mac capture pending.
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
- Implementation: no Rust runtime or inference code has been added yet.
- Measurements: no Rust Star correctness or performance runs have been made.
- Parallel research: DSpark remains separate on
  `origin/codex/dspark-observability-0` and is not on this phase's critical path.
- Publication: the connected GitHub app is installed on both `deathcoder` and
  `dion-labs`. Remote branch `agent/rust-star-bootstrap` is published. Prefer
  the app for future remote writes from this environment.

## Immediate Next Actions

1. Clone this branch on the M1 Ultra and run the default 2K/32K capture; return
   the generated `.tar.gz` bundle and its printed SHA-256.
2. Inspect the quick capture, resolve any machine-specific failures, and accept
   its model/toolchain fields as the completed `oracle-v1` identity.
3. Run the extended 2K--1M frontier set only after the quick capture succeeds.
4. Create the smallest Rust host/runtime scaffold that can be validated without
   Metal, beginning with strict target/model identity and artifact contracts.
5. Add kernel/layer/decode-step artifact extensions alongside the relevant
   runtime hooks.
6. On the target Mac, measure interop and command-submission overhead with the
   smallest Metal dispatch prototype before choosing more architecture.

## Entries

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
