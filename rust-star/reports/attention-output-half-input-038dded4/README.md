# Attention-output FP16 input staging: exact, promising, timing control incomplete

**Do not integrate or select this candidate for full-model validation yet.**
All correctness checks pass and the candidate is consistently much quicker in
this synthetic complete-chain experiment, but the contemporaneous A/A controls
fail the predeclared neutrality criteria. The overall screening decision is
inconclusive; there is no accepted engine speedup.

All **19,197,968,384 FP32 comparisons**
pass across edge cases and both fresh processes. All **72 candidate pairs**
improve. Candidate/original medians are **34.29–35.09%
lower**, separately in AB and BA order across all six case/session combinations.
A/A measurements reveal a **1.99–2.59% difference associated with physical buffer
set identity**, with 2.07–2.56% first/second median absolute deviation. Those
exceed the inherited 1% limits. No threshold was relaxed or run discarded.

## Target and change

Starting commit: `038dded4a875932ca8b0c9123eeeb3e1fe844f6a`. Runtime remains unchanged at
`9d35953f54d4c1c5c8c85ceebf696d1432025c92`. The preceding real-model profile attributed 35.076s
(15.12% of recorded GPU stage time) to attention output. Its native 4K path is:

1. Build the eight-group map.
2. Q8 grouped projection: eight 4096-input / 1024-output groups per token.
3. Q8 dense projection: 8192 inputs / 4096 outputs per token.
4. Expand the output into four HC residual streams.

Both matrix kernels already cast FP32 RHS loads to FP16 before their matrix
multiply. The candidate performs that conversion once per tensor instead of
repeating it for each matrix tile. It adds a heads-to-FP16 pass before the first
projection and a low-output-to-FP16 pass before the second. Group mapping and
HC expansion stay unchanged. The complete timed command includes all six
candidate dispatches; conversion costs are not subtracted.

No imported kernel body is edited. The grouped stage uses the existing
`kernel_mul_mm_id_q8_0_f16` export. The dense stage adds only a half-RHS
instantiation of the same `kernel_mul_mm` template, keeping its matrix and
accumulator types. The grouped half export also changes dormant weight-load
parameter types; their alternate branch is disabled for Q8 weights and constant
700=false. The executed dequantization and accumulation paths remain the same.
`verify.py` checks both instantiations and exact source assembly. This is a
change in where existing FP16 rounding happens, not a new quantization format.

At 4096 rows, the two extra reusable scratch tensors total **320 MiB**. Logical
RHS tile loads plus conversion traffic change from **16 GiB to 8.9375 GiB** per
attention-output chain. This is source-derived traffic accounting, not measured
DRAM traffic: caches and weight/output traffic are excluded. Buffer allocation
is outside repeated GPU timing, and each physical test buffer set allocates
candidate scratch even during A/A. A future integration must assess allocation
and memory costs separately.

## Prospective method and exactness

The plan and source/binary freeze predate GPU execution. The host executable
builds with Clang and unchanged Metal source bodies; all four Metal argument
structs are extracted from that source and statically checked at 48/104/88/152
bytes. Production encoding and argument excerpts are included for review.
There is no runtime build/test or model inference in this step because all 41
runtime/source inputs remain unchanged; the new standalone host and Metal
pipelines were built and exercised here.

Sixteen edge cases cover rows 1, 31, 32, 33, 127, 128, 129 and 1024 with random
and FP16 rounding-boundary mixtures. The latter includes signed zero, subnormal,
normal-boundary and tie values. Two opposite-order pairs compare all grouped,
dense and HC FP32 outputs; CPU checks verify every FP16 conversion and map entry.
All guards, finite/nondegenerate checks and immutable input hashes pass:
**172,605,440 FP32 words** in a 9.457s edge process.
Partial-row edge tests enable the original dense output bounds handling and
8192-byte scratch; the 4096-row timed path uses production constants 700=false,
701=false and its original 6144-byte dense / 8192-byte grouped scratch.

The main screen uses exactly two fresh processes, three 4096-row seeded inputs
per process, and reversed case order in the second process. Each case has two
initial opposite-order exactness pairs, one excluded per-stage diagnostic chain
per variant, and 12 candidate plus 12 A/A pairs. Every slot queues four
conditioning commands and averages three measured commands. AB/BA, physical
buffer assignment and protocol order are balanced. A/A uses the identical
original pipeline objects and original four-dispatch sequence in both roles.

Every pair compares complete final grouped, dense and HC outputs, checksums and
map entries; guards and immutable input hashes also pass. Initial and diagnostic
pairs include full CPU half-conversion checks. All outputs and inputs repeat
across processes. Intermediate results overwritten by a seven-command slot
are not separately retained. Main total: **19,025,362,944 FP32 words**,
144 measured pairs and 2,016 audited GPU command intervals. Each interval is
finite, positive and ordered; the audit recomputes three-sample means, schedule,
ratios and every acceptance gate.

## Candidate observations

Candidate criteria require each AB and BA median below 0.99 and at least 5/6
faster pairs per order at every case/session. All these candidate-specific
criteria pass. The values below are observations, not an accepted speed claim.

| Session | Input case | AB ratio | BA ratio | Faster AB | Faster BA |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.657006 | 0.651066 | 6/6 | 6/6 |
| 0 | 1 | 0.649104 | 0.654232 | 6/6 | 6/6 |
| 0 | 2 | 0.652379 | 0.656334 | 6/6 | 6/6 |
| 1 | 2 | 0.654569 | 0.653937 | 6/6 | 6/6 |
| 1 | 1 | 0.651289 | 0.653237 | 6/6 | 6/6 |
| 1 | 0 | 0.657148 | 0.656467 | 6/6 | 6/6 |

## A/A failure and interpretation

Every case/session must also pass the fixed A/A gates: first/second, AB/BA and
physical-buffer medians within 1% of parity; first/second MAD at most 1%; at least
10/12 ratios within 3%. All six cases fail physical-buffer and dispersion-MAD
limits. Session 1 / case 2 additionally fails AB/BA and within-3% count gates.

| Session | Case | First/second median | AB / BA medians | Physical A/B median | MAD | Within 3% | Failed gates |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0.998921 | 1.006962 / 0.999525 | 1.022085 | 2.194% | 10/12 | physical_buffer, mad |
| 0 | 1 | 0.999060 | 1.003630 / 1.002705 | 1.019910 | 2.085% | 12/12 | physical_buffer, mad |
| 0 | 2 | 0.998240 | 1.002651 / 1.004334 | 1.021418 | 2.133% | 10/12 | physical_buffer, mad |
| 1 | 2 | 0.998100 | 0.994463 / 0.989551 | 1.025930 | 2.555% | 9/12 | ab_ba, physical_buffer, mad, pair_dispersion |
| 1 | 1 | 1.000382 | 1.000118 / 1.000893 | 1.025509 | 2.520% | 10/12 | physical_buffer, mad |
| 1 | 0 | 0.998265 | 0.997976 / 0.995056 | 1.022681 | 2.067% | 11/12 | physical_buffer, mad |

The large candidate signal is much greater than the observed null difference,
but that does not satisfy the preset conjunction of candidate and A/A gates.
Keep the candidate and all data for follow-up; do not retroactively normalize
or waive the failed controls. The result does not show that the kernel is slow
or incorrect. It shows that this complete-chain harness has a buffer-associated
timing effect that the prior compressed-attention A/A result did not establish
neutrality for. Cache behavior, page placement and GPU clocks are unmeasured;
none is identified as the cause.

## Excluded local stage diagnostics

The following are medians of six single stage observations per variant,
collected in separate, host-waited command buffers before timing. They help
locate local work and are explicitly excluded from candidate acceptance.

| Stage | Original ms | Candidate ms |
|---|---:|---:|
| convert_heads_f16 | — | 1.314 |
| group_map | 1.626 | 0.871 |
| grouped_projection | 83.782 | 48.602 |
| convert_low_f16 | — | 0.385 |
| dense_projection | 37.392 | 23.458 |
| hc_expand4 | 4.088 | 4.294 |

These support continuing to investigate both projections rather than removing
the map or HC step. The copy passes cost little in this fixture relative to the
observed projection reduction. If a similar 35% reduction later survives real
inputs, its approximately 15% share of recorded GPU time would correspond to
about 5% of that stage sum; this is hypothetical, not an engine prediction.

## Next bounded step

Diagnose the buffer-associated A/A effect before changing runtime or loading
the model. A prospective same-buffer crossover with a protected exact reference
is a reasonable next test; preserve full output checks and both execution
orders. First establish neutrality using identical original pipelines, then
repeat the candidate only under a new fixed plan. Do not modify these recorded
criteria or launch an automatic broad timing campaign. 256K stays paused.

Both main processes finish normally in 142.612s and 143.597s. Main host swap is
`vm.swapusage: total = 1024.00M  used = 33.25M  free = 990.75M  (encrypted)` initially and `vm.swapusage: total = 1024.00M  used = 33.25M  free = 990.75M  (encrypted)` finally; periodic thermal,
VM and load observations are preserved. Hardware was checked as M1 Ultra with
48 GPU cores. Final hashes verify all frozen study inputs/executable and all
41 unchanged runtime/source inputs. All GPU work is finished. No push or release.

## Portable evidence

Run `python3 verify.py` in this folder. It validates the artifact manifest,
original/stored path-redaction bindings, source assembly, template instantiations,
ABIs, freezes and chronology, edge exactness and all 144 timed-pair receipts and
gates. The executable is omitted with its SHA-256 in `binary-identity.json`;
no model weights are used or included. Private user paths are replaced with
`<repo>` / `<user>`, with original and stored hashes preserved. The included
execution scripts are provenance and require a checkout to rerun; the verifier
itself needs only Python's standard library.
