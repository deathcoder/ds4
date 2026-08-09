# Rust Star Artifact Contract

Version: `rust-star-oracle-manifest-v1` and
`rust-star-logit-comparison-v1`.

This document defines the first stable boundary between the pinned DwarfStar
oracle and candidate Rust Star implementations. It covers result bundles and
post-prefill full-vocabulary logits. Kernel, layer, and decode-step artifacts
will extend this contract without changing the C0 meaning below.

## Oracle bundle

An oracle bundle is a directory, optionally packaged as `.tar.gz`, whose root
contains `manifest.json`. The manifest binds together:

- the immutable upstream source commit and tree;
- the capture-kit commit and clean-worktree state;
- model filename, byte size, and SHA-256 without model contents or absolute path;
- privacy-filtered host and toolchain identity;
- prompt reconstruction and hash;
- inference and benchmark configuration;
- correctness-gate results;
- conformance artifact descriptors; and
- raw and aggregated performance results.

Every descriptor with `path`, `bytes`, and `sha256` names a regular file inside
the bundle. Paths must be relative and cannot escape the bundle. The archive's
separate SHA-256 protects the manifest and unhashed diagnostic logs as well.

`oracle-v1` is complete only when `manifest.status` is `complete` and every
enabled correctness, conformance, and performance section has status `passed`.
A partial bundle is useful for diagnosing a capacity failure but is not an
accepted oracle.

## Full-logit JSON

The initial tensor artifact is DwarfStar's
`frontier_NNNNNN.logits.json`. Required semantic fields are:

| Field | Meaning |
| --- | --- |
| `vocab` | Number of FP32 elements. Must equal `len(logits)`. |
| `prompt_tokens` | Tokenized prompt length at capture. |
| `frontier_tokens` | Post-prefill frontier represented by this tensor. |
| `ctx` | Allocated context size. |
| `quant_bits` | Routed-expert quantization; `oracle-v1` requires Q2. |
| `quality` | DwarfStar quality-path state. |
| `argmax_id` | Lowest-index maximum, verified against the tensor. |
| `logits` | Full pre-softmax vocabulary vector in token-ID order. |

All logits must be finite. JSON `null`, NaN, and infinities are invalid.
DwarfStar writes each value with nine significant decimal digits. Nine digits
are sufficient to round-trip a finite IEEE-754 binary32 value.

## C0 comparison

C0 is evaluated as follows:

1. Parse each JSON number.
2. Round it to IEEE-754 binary32 using round-to-nearest, ties-to-even.
3. Compare all 32 bits at every vocabulary index.
4. Compare `vocab`, `prompt_tokens`, `frontier_tokens`, `ctx`, `quant_bits`, and
   `quality` metadata.

Signed zero is significant: `+0.0` and `-0.0` are not C0-equal. NaN
canonicalization is irrelevant because non-finite values are forbidden. Source
and backend labels may differ between implementations and are informational.

Only zero bit mismatches and zero required-metadata mismatches produce `C0`.
Argmax or generated-token agreement alone never does.

Run:

```sh
python3 rust-star/compare_logits.py ORACLE.json CANDIDATE.json \
  --json comparison.json
```

The command exits zero for C0, one for valid but non-C0 tensors, and two for an
invalid artifact. `--allow-drift` changes a valid non-C0 result to exit zero for
experimental pipelines; it does not change the reported classification.

## Drift diagnostics

For non-C0 experimental paths the comparator reports:

- mismatch count and rate;
- representative differing FP32 bit patterns;
- maximum and mean absolute error;
- RMSE and maximum relative error;
- maximum ordered-FP32 ULP distance;
- raw-logit cosine similarity;
- `KL(P_reference || P_candidate)` after stable softmax;
- argmax agreement; and
- top-1, top-10, and top-50 Jaccard overlap where defined.

These metrics characterize divergence but do not establish a universal C2
acceptance threshold. Each experimental optimization must declare its own
thresholds and task-quality evidence before results are interpreted.

## Bundle verification

Verify a returned capture before using it:

```sh
python3 rust-star/verify_oracle_bundle.py \
  rust-star/results/oracle-v1-TIMESTAMP.tar.gz \
  --sha256 PRINTED_SHA256
```

If the sibling `.sha256` file is present, `--sha256` may be omitted. Verification
checks safe archive extraction, pinned source identity, manifest state, artifact
size/hash, context coverage, Q2 conformance tensors, and performance repetition
counts. `--allow-partial` is diagnostic only.
