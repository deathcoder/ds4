# Rust Star Oracle Capture

This directory contains the reproducible capture kit for `oracle-v1`. Run it on
the target M1 Ultra before implementing or benchmarking the Rust runtime.

The capture does not need network access, GitHub credentials, SSH keys, or any
other secret. It records only an allowlist of hardware and toolchain fields; it
does not dump the process environment or Mac serial identifiers.

The platform-independent Rust host scaffold now lives in `rust-star/runtime/`.
Before the oracle capture, compile its tests and inspect the model directory:

```sh
./rust-star/check_runtime.sh \
  /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf
```

See `rust-star/runtime/README.md` for its exact scope. It now has the minimal
Metal ownership/dispatch probe, a no-copy real-model F16 embedding gather, and
a no-copy Q8_0 decode projection checked against a pinned DwarfStar fixture; it
is not yet a decoder.

Project controls and benchmark contracts:

- `RUST_STAR_MANUAL_TASKS.md` is the canonical ledger for work that needs the
  target Mac, model, GitHub UI, or a deliberate access decision.
- `rust-star/BENCHMARK_PROTOCOL.md` defines the paired DwarfStar/Rust Star
  comparison.
- `rust-star/PAIRED_RESULT_FORMAT.md` defines the validated raw and summary JSON
  boundary for future benchmark runners.
- `rust-star/MEASUREMENT_ADAPTER.md` defines the isolated DwarfStar measurement
  boundary used to populate those paired records.
- `rust-star/ENGINE_MEASUREMENT_FORMAT.md` defines the equivalent Rust Star
  boundary, and `rust-star/PAIRED_RUNNER.md` documents resumable A/B execution.
- `rust-star/runtime/METAL.md` defines the initial Rust/Objective-C ownership
  split and correctness-checked command-dispatch probe.

## Quick capture

Clone the published research branch on the Mac Studio, then run:

```sh
git clone --branch agent/rust-star-bootstrap \
  https://github.com/deathcoder/ds4.git rust-star-ds4
cd rust-star-ds4

python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/DeepSeek-V4-Flash-0731-Q2-imatrix.gguf
```

The default capture uses 2K and 32K context frontiers with three timed
repetitions each. It:

1. hashes the complete model;
2. records a privacy-filtered hardware, macOS, and toolchain manifest;
3. exports and builds an isolated copy of pinned upstream commit `b030961`;
4. runs the existing Metal-kernel and official logprob-vector correctness gates;
5. captures full FP32 frontier logits in separate, untimed conformance runs;
6. runs repeated batch-one DwarfStar benchmarks; and
7. writes raw CSVs, median/MAD summaries, logs, hashes, and a shareable archive.

Model hashing reads the entire GGUF and will take a while. The model is never
copied into the results or archive. The build uses a temporary source directory
and a generic `model.gguf` symlink so local account paths do not appear in the
logit artifacts.

At completion the script prints paths similar to:

```text
Results: rust-star/results/oracle-v1-YYYYMMDDTHHMMSSZ
Archive: rust-star/results/oracle-v1-YYYYMMDDTHHMMSSZ.tar.gz
SHA-256: ...
```

Send back the `.tar.gz` archive and its printed SHA-256. Capture outputs are
git-ignored; do not force-add them to the repository.

You can verify the completed archive before sending it:

```sh
python3 rust-star/verify_oracle_bundle.py \
  rust-star/results/oracle-v1-TIMESTAMP.tar.gz
```

The verifier automatically uses the sibling `.sha256` file. See
`rust-star/ARTIFACT_FORMAT.md` for bundle and exact-logit comparison semantics.

## Extended contexts

After the quick capture succeeds, run the extended frontier set with five
repetitions:

```sh
python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/model.gguf \
  --full \
  --repetitions 5
```

`--full` covers 2K, 32K, 128K, 256K, 512K, and 1M. The 512K and 1M runs may
approach the safe working-set limit on a 128 GB machine. Run the quick capture
first and close other memory-heavy applications before the extended capture.

For a custom subset:

```sh
python3 rust-star/capture_oracle_v1.py \
  --model /absolute/path/to/model.gguf \
  --contexts 2048,32768,131072,262144 \
  --repetitions 5 \
  --notes "Room 22C; normal background services; no other model processes"
```

Useful options:

- `--output DIR`: choose an empty output directory.
- `--gen-tokens N`: change the timed greedy decode length; default 128.
- `--skip-correctness`: omit the two existing correctness gates.
- `--skip-conformance`: omit full-logit artifacts.
- `--skip-performance`: collect only manifest/correctness/conformance data.
- `--dry-run`: print the plan without hashing, building, or running the model.

Skipping a section is recorded in the manifest and does not produce a complete
`oracle-v1` capture.

## Artifact semantics

Conformance and performance runs are deliberately separate. The headline
throughput CSVs are produced without logit dumping. Full frontier-logit JSON
uses DwarfStar's nine-significant-digit FP32 encoding, which round-trips finite
FP32 values exactly.

The first capture covers full logits immediately after prefill at each selected
context. A later differential harness will extend this to kernel/layer
boundaries and every decode step. The manifest states this scope explicitly so
the initial artifact is not mistaken for complete engine conformance coverage.

Performance runs use greedy decode while excluding EOS, matching `ds4-bench`.
Each selected context runs in its own process and therefore measures prefill
from zero to that frontier. Repetition order alternates ascending and descending
contexts to reduce systematic thermal-order bias.

## Security boundary

The archive contains model filename, byte size, and SHA-256, but never model
contents. It excludes:

- environment variables not on the small compiler/runtime allowlist;
- GitHub tokens, SSH configuration, and credentials;
- process listings and command lines from unrelated applications;
- hardware serial number, UUID, and provisioning identifiers; and
- the absolute model path.

Review free-form `--notes` before sharing because their content is supplied by
you. A future remote-execution setup should use a private network and a
short-lived, command-restricted credential; it is intentionally outside this
capture kit.
