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

- Phase: project bootstrap and experimental-contract definition.
- Working branch: `agent/rust-star-bootstrap`.
- Branch base: upstream `antirez/ds4` commit
  `b0309611041655f4e45671cfd9c9886aff161406`.
- Local `main`: fast-forwarded to the same upstream commit.
- Fork `origin/main`: still at
  `80ebbc396aee40eedc1d829222f3362d10fa4c6c` until a successful push; it was
  137 commits behind and 0 ahead of upstream at bootstrap.
- Oracle: candidate source commit selected, but `oracle-v1` is incomplete until
  its model SHA, target-machine toolchain/configuration, and golden artifacts
  are captured.
- Implementation: no Rust runtime or inference code has been added yet.
- Measurements: no Rust Star correctness or performance runs have been made.
- Parallel research: DSpark remains separate on
  `origin/codex/dspark-observability-0` and is not on this phase's critical path.
- Publication: prefer the connected GitHub app for remote writes. Its repository
  metadata reports the authenticated user's `push: true`, but the app is
  installed only on `dion-labs`, not the personal `deathcoder` account that owns
  this fork. Ref-update and branch-create calls therefore return GitHub HTTP 403.

## Immediate Next Actions

1. Install/authorize the GitHub app on the personal `deathcoder` account with
   access to `deathcoder/ds4`, then use it to update fork `main` and publish
   `agent/rust-star-bootstrap`. Fall back to authenticated local git only if the
   correctly scoped app installation still cannot write refs.
2. On the M1 Ultra, capture the complete oracle manifest and a clean DwarfStar
   baseline at agreed context frontiers.
3. Specify a versioned golden-artifact format for kernel, layer, and full-logit
   comparisons.
4. Create the smallest Rust host/Metal dispatch prototype needed to measure
   interop and command-submission overhead before choosing more architecture.

## Entries

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
