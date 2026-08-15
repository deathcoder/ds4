# Rust Star Manual Task Ledger

This file is the source of truth for work that cannot be completed by the
current coding environment because it requires the M1 Ultra, the real model,
account-level UI access, or a deliberate user security decision.

Keep engineering history in `RUST_STAR_JOURNAL.md`. Keep stable scope and
correctness rules in `RUST_STAR_PROJECT.md`. Add tasks here only when a person
must perform or authorize something, and remove nothing merely because it is
inconvenient: mark it complete with evidence instead.

## Status Rules

- `READY`: can be performed now when the named access is available.
- `BLOCKED`: another task or missing resource must be resolved first.
- `DEFERRED`: intentionally outside the current phase.
- `DONE`: completed; retain the evidence summary and date.

Do not paste credentials, access tokens, private keys, environment dumps, Mac
serial/UUID values, or the model into this repository or chat. Model paths in
shared output should be reduced to the filename where possible.

## M-001 — Confirm GitHub Actions for the research branch

Status: `READY`

Why manual:

- The workflow is published, but no run was visible after the first branch
  push. Fork Actions may need to be enabled or approved in the GitHub UI.

Procedure:

1. Open the repository's **Actions** tab.
2. If GitHub presents an enable/approve button for fork workflows, review and
   enable Actions for `deathcoder/ds4`.
3. Select **Rust Star host contracts**.
4. Use **Run workflow**, choose branch `agent/rust-star-bootstrap`, and run it.
5. Do not alter repository secrets; this workflow needs none.

Return evidence:

- The workflow run URL and final status.
- If it fails, the failing step name and its plain-text log. Review the log for
  local paths or other private data before sharing.

Success condition:

- Formatting, Rust tests, release build, Python tests, and the cross-language
  artifact check all pass on pinned Rust 1.74.

## M-002 — Compile and inspect the real GGUF on the M1 Ultra

Status: `DONE` (2026-08-14)

Prerequisites:

- Access to the M1 Ultra Mac Studio.
- The DeepSeek-V4-Flash-0731 resident imatrix-Q2 GGUF.
- Rust 1.74 or newer and Python 3 available on the Mac.

Procedure from a clean checkout:

```sh
git switch agent/rust-star-bootstrap
git pull --ff-only

set -o pipefail
./rust-star/check_runtime.sh \
  /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf \
  2>&1 | tee rust-star-runtime-check.txt
```

`pipefail` is required so a failed check remains a failed command when its
output is also written by `tee`.

The model path must remain outside the repository. The inspector reads the
GGUF metadata and tensor directory but does not read tensor payloads or compute
the model hash.

Return evidence:

- `rust-star-runtime-check.txt` after checking that it contains no private path
  you do not want to share. Redacting only the absolute path is acceptable.
- `rust-star/.work/runtime-target/metal-dispatch-probe.json`.
- The output of `rustc --version` and `cargo --version` if the script fails
  before compilation.

Success condition:

- Host tests and cross-language artifact checks pass.
- The Metal probe validates its shared buffer and reports both roundtrip and
  batched dispatch timing.
- Strict inspection ends with `result: target shape and Q2 recipe valid`.

If strict inspection fails, do not weaken the validator yet. Return the exact
error so the expected recipe can be reconciled against the real oracle model.

Evidence:

- Apple M1 Ultra Metal shared-buffer validation passed and reported both
  roundtrip and batched dispatch timing.
- 15 Rust tests, 25 Python tests, the optimized macOS build, and the
  cross-language C0 artifact check passed.
- Strict inspection validated all 1,288 required tensors in the resident 0731
  imatrix-Q2 model after correcting the indexer Q projection recipe to F16.
- `rust-star/.work/runtime-target/metal-dispatch-probe.json` and the local
  `rust-star-runtime-check.txt` retain the machine-readable and console
  evidence; generated evidence remains uncommitted.

## M-003 — Capture the quick `oracle-v1` bundle

Status: `DONE` (2026-08-14)

Procedure:

```sh
python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf
```

This runs the default 2K/32K capture with three repetitions. It hashes the full
model, so it will take longer than M-002.

Return evidence:

- The generated `oracle-v1-*.tar.gz` archive.
- Its printed SHA-256 or sibling `.sha256` file.
- Any failure summary printed by the tool.

Success condition:

- `verify_oracle_bundle.py` accepts the returned archive.
- The manifest is complete and binds the source, model SHA, toolchain, target
  machine configuration, correctness gates, logits, and performance runs.

Evidence:

- The verifier accepted a complete 2K/32K bundle containing 13 checksummed
  artifacts and both full-vocabulary conformance tensors.
- Oracle model SHA-256:
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
- Archive SHA-256:
  `5115f445db651de8777af64c708df81e7a708e5e950fbc5502d56fea9a09ee3c`.
- Median DwarfStar performance was 164.86 prefill / 19.90 generation tok/s at
  2K and 161.05 prefill / 17.36 generation tok/s at 32K.

## M-004 — Capture extended context frontiers

Status: `READY`

Procedure:

```sh
python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf \
  --full \
  --repetitions 5
```

This covers 2K, 32K, 128K, 256K, 512K, and 1M. Close memory-heavy applications
first. A 512K/1M capacity failure is evidence to preserve, not a reason to hide
or retry it until it happens to pass.

Return evidence:

- The generated archive and SHA-256.
- The last completed context if the run stops for capacity.

## M-005 — Execute the first paired DwarfStar/Rust Star run

Status: `BLOCKED` on native batched prefill, ratio-4 sparse indexed attention,
and the engine-measurement producer. The 2K sequential initializer now owns a
128-row raw ring plus context-sized compressed state and exactly matches two
fresh DwarfStar one-token decode replays. It deliberately records a full-logit
mismatch against DwarfStar's batched-prefill oracle, so it is not eligible for
the paired protocol even though both paths select the same token.

Prerequisites:

- A complete accepted oracle bundle and correctness-manifest SHA-256.
- Pinned DwarfStar and Rust Star executables with SHA-256 values.
- A private local plan following `rust-star/PAIRED_RUNNER.md`.

Start with one observed pair rather than the full schedule:

```sh
python3 rust-star/run_paired_benchmark.py run paired-plan.json \
  --output /absolute/path/to/paired-results \
  --max-new-pairs 1
```

The plan can contain private model/source paths and must remain outside the
repository. Do not share it without review. The result directory binds the plan
by SHA-256 without copying it.

Return evidence:

- `state.json`, `paired-raw.json`, and `paired-summary.json` once complete.
- The referenced engine measurement directories after reviewing their logs.
- Any blocked attempt unchanged; do not manufacture a retry reason for a real
  engine, correctness, or capacity failure.

Success condition:

- Every predeclared pair ends in exactly one valid attempt, retained invalid
  attempts have explicit external-event reasons, and the generated raw file
  passes the paired validator.

## M-006 — Decide on secure remote Mac access

Status: `DEFERRED`

This is not required for bootstrap. Before enabling remote execution, make a
deliberate choice that provides all of the following:

- private network reachability rather than a public inbound service;
- a dedicated, revocable, short-lived identity;
- command and directory restrictions appropriate to this repository;
- no credentials committed to Git or pasted into chat;
- no access to unrelated personal files or keychains;
- an audit trail and a simple kill switch.

Document the chosen boundary here before configuring it. Do not improvise
remote access merely to save a manual benchmark run.

## Completed Tasks

None yet.
