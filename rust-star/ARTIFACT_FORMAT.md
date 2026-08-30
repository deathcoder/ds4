# Rust Star Artifact Contract

Version: `rust-star-oracle-manifest-v1`,
`rust-star-differential-fixture-v1`, and `rust-star-logit-comparison-v1`.

This document defines the first stable boundary between the pinned DwarfStar
oracle and candidate Rust Star implementations. It covers result bundles and
post-prefill full-vocabulary logits, plus reusable differential fixtures at
kernel, layer-segment, and decode-step boundaries.

## Differential fixtures

A differential fixture is a directory whose root contains `manifest.json`.
Its `scope.kind` is one of `kernel`, `layer-segment`, or `decode-step`, and its
`operations` array records the ordered oracle operations represented by the
fixture. Every tensor descriptor records a unique name and relative path, its
boundary role, shape and encoding, byte count, and SHA-256. Payloads may use
little-endian finite IEEE-754 binary16 (`f16`), binary32 (`f32`), or
little-endian signed 32-bit integers (`i32`); the latter preserves discrete
boundaries and exact FP32 bit patterns such as selected expert IDs or compressor
score states. The verifier requires byte counts to agree with the declared shape
and all floating-point values to be finite.

The manifest pins the DwarfStar oracle commit/tree and capture executable, the
model SHA-256, execution phase, layer where applicable, and token position.
This makes a fixture independently auditable rather than an unnamed pair of
binary blobs. The tensor list can preserve any number of intermediate
boundaries, so the same schema covers one kernel or a complete decode step.

Verify a fixture before consuming it:

```sh
python3 rust-star/verify_differential_fixture.py \
  rust-star/fixtures/q8-attn-q-a-v1
```

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

An oracle bundle is complete only when `manifest.status` is `complete` and every
enabled correctness, conformance, and performance section has status `passed`.
A partial bundle is useful for diagnosing a capacity failure but is not an
accepted oracle.

`oracle-v1` retains its historical one-capture-per-context contract. The
accepted `oracle-v2` pins repaired producer commit `b81c099`, must cover 2K
and 32K, and requires at least two fresh-process conformance captures per
frontier. Every repeated full-logit tensor and required C0 metadata field must
be bit-identical; the verifier rejects the bundle otherwise. Performance may
remain a separate disabled section because oracle identity and benchmark
eligibility are independent contracts.

The accepted `oracle-v3` pins host-synchronized producer commit `d35fb12`, covers the same 2K
and 32K frontiers, and raises the minimum to four fresh-process conformance
captures per frontier after a post-acceptance v2 audit exposed a rarer residual
dependency hazard. It must preserve the accepted v2 tensors bit-for-bit before
advancement; source identity and repeatability are both part of acceptance.

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
