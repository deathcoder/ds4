# Compressed-attention empty-block run skipping

The run-skip candidate passes the predeclared isolated selection screen.
All 72 candidate pairs improve; order-specific median GPU-time reductions
range from 1.14% to 6.08% across three timed shapes in two fresh processes.
Every contemporary A/A gate and every integer/GPU correctness check passes.

Retain the frozen candidate for later full-model validation. Runtime remains
`1711856e`; this report establishes no normal-engine or total-prefill speedup.

## Change and rationale

The original compressed FlashAttention loop visits every cache-block marker.
For an empty block it advances each query mask pointer and continues without
attention arithmetic. In the seven existing production-shape fixtures,
90.00–94.03% of these block markers are zero.

The isolated candidate checks complete blocks in groups of four, skips all-zero
runs, then handles remaining zero blocks individually. It advances mask pointers
once for the whole skipped run. The original loop body still processes active
blocks in the same order, and the incomplete padded tail follows the original
path. No extra buffer, preprocessing kernel, dispatch-grid change or floating-
point arithmetic change is introduced.

The source proof restores the complete original template byte-for-byte by
removing the integer run-skip insertion and reversing symbol renames. This
proves source-level arithmetic preservation; GPU comparisons provide separate
evidence that the compiled candidate is exact on the covered inputs.

Skipping adds metadata reads near active blocks. The benefit under test is
fewer outer-loop/padding checks and mask-pointer increments, not fewer bytes
loaded in every mask distribution. The CPU model records this tradeoff.

## Fixed study

- Starting commit: `fbbea533`; all 40 runtime/imported source pins remain
  unchanged before and after the study.
- Target rechecked: Apple M1 Ultra, 48 GPU cores, 128GB unified memory.
- Original and candidate use the same arguments, original grid, function
  constants and 28,672 bytes of threadgroup memory.
- Current GPU coverage is the 512-wide F16, eight-SIMD-group masked path.
  Other generic template variants were not separately GPU-validated.
- The candidate, plan and edge-test sources were frozen before execution.
  Successful edge receipts and the timed harness were frozen again before
  either performance process.
- All previous failed grid-order experiments remain unchanged.

## Correctness gates before timing

**Integer traversal:** 119,150 exhaustive/seeded cases compare active-block
order, mask-pointer state and padded K/V addressing. Coverage includes every
0/1/2 marker sequence through length nine, masked/unmasked control paths, both
padding states and seeded cases through 4,096 blocks. Any out-of-bounds marker
read fails the check. This models integer control flow, not GPU arithmetic.

**GPU edges:** 120 small cases use 16 queries, two heads and 512-wide data.
Fifteen key lengths bracket cache-block and four-block boundaries, from one
key through 4,384. Eight mask patterns cover empty/full masks, first/last live
blocks, four-block groups, seven-zero-block runs, partially visible gaps and
isolated endpoint keys. Two opposite-order pairs per case pass 3,932,160 FP32
comparisons, including 15 all-masked positive-zero outputs.

Independent CPU/GPU block-map and padded mask/KV checks, finite outputs,
destination guards and full input hashes also pass. Edge execution took
1.911s including initialization/compilation. No failed edge attempt occurred.

## Timed protocol and preset selection

Two fresh processes cover all seven production chunk shapes for correctness,
in forward/reverse order. Three shapes (chunk starts 4096, 16384 and 28672)
receive 12 candidate and 12 A/A pairs per process. AB/BA execution, physical
output-buffer assignment and candidate/control order are balanced.

Every slot queues four conditioning dispatches and three measured dispatches
before waiting. The slot time averages the three measured GPU intervals.
Both output buffers are poisoned before each pair; full validation follows
both roles. A/A dispatches the same original pipeline object in both slots.
Candidate pairs dispatch original A and run-skip B on the original grid.

Selection required all correctness gates and every contemporary A/A case
to pass. The inherited null thresholds require median first/second, AB/BA and
physical-buffer ratios within 1%, ratio MAD at most 1%, and at least 10/12
first/second ratios within 3%. Each candidate shape/session also needed AB and
BA median ratios below 0.99 and at least 5/6 faster pairs per order.

Every required condition passes. No criterion was changed, pair discarded,
timing normalized against A/A or additional process run.

## Candidate results

Ratios are candidate GPU time divided by original GPU time. Each order has
six paired observations. Values below 1 mean lower candidate latency.

| Session | Chunk start | AB median | BA median | Faster AB | Faster BA | Selection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4096 | 0.960509 | 0.970953 | 6/6 | 6/6 | Pass |
| 0 | 16384 | 0.947580 | 0.942933 | 6/6 | 6/6 | Pass |
| 0 | 28672 | 0.988563 | 0.986260 | 6/6 | 6/6 | Pass |
| 1 | 28672 | 0.985507 | 0.987819 | 6/6 | 6/6 | Pass |
| 1 | 16384 | 0.946539 | 0.939226 | 6/6 | 6/6 | Pass |
| 1 | 4096 | 0.965066 | 0.961747 | 6/6 | 6/6 | Pass |

Both timed processes finish normally, in 192.964s and 191.966s including
initialization, conditioning and CPU checks. These are harness process times.

Across the six A/A cases, maximum first/second median deviation is 0.322%,
maximum AB/BA deviation 0.454%, maximum physical-buffer deviation 0.301%, and
maximum ratio MAD 0.460%. All 72 A/A ratios fall within 3% of parity.

The main harness passes 23,085,449,216 final-output FP32 comparisons over
144 measured pairs and 28 initial warm-up pairs. Together with the edge cases,
23,089,381,376 FP32 comparisons pass. Intermediate repeated dispatch outputs
are overwritten; the final output of each role is compared after each pair.
All 2,016 recorded GPU intervals and the complete balanced schedule audit pass.

Host snapshots show unchanged 49.25MB swap usage and no recorded thermal or
performance warning. GPU clocks were not measured. No other model/probe process
was present at either timed-process preflight.

## Limits and next step

This measures a synthetic, warm repeated-kernel regime. It is not a fresh
full-model C0 run or a production prefill benchmark. Exactness on all seven
shapes does not establish behavior for arbitrary masks, function constants,
model configurations or larger frontiers.

The [32K diagnostic](../32k-attention-1711856e/README.md) attributed approximately
12.04% of summed recorded GPU stage time to compressed attention. For scale,
a hypothetical uniform 4% reduction in that component corresponds to only
about 0.48% of that recorded sum. This is planning arithmetic on a perturbed
profile, not a measured engine benefit. The isolated percentages must not be
reported as total prefill gains.

The next concrete step is a narrow Rust-owned integration for native compressed
prefill, then a pinned full-output 32K/128 C0 check and bounded real-component
validation. Decide whether normal engine pairs are worth their cost from that
evidence; do not automatically launch a broad timing campaign. Keep the 256K
campaign paused. No runtime edit, model load, push or release occurred here.

## Verify and reproduce

Verify the portable evidence without a model or GPU:

```sh
python3 rust-star/reports/compressed-skip-runs-fbbea533/verify.py
```

The verifier checks all artifact hashes, original/stored path redactions,
source and executable bindings, source/library/ABI reconstruction, inherited
host code, edge receipts, integer traces, timestamp ordering, exactness checks
and independently recalculated null/candidate statistics. Binaries and weights
are omitted; executable hashes remain in the freeze manifests.

For a deliberate new experiment on the target Mac, copy the shader/header and
host sources into an empty directory, then build:

```sh
clang -O3 -fobjc-arc -framework Foundation -framework Metal probe.m -o probe
clang -O3 -fobjc-arc -framework Foundation -framework Metal edges/probe.m -o edges/edge-probe
./edges/edge-probe combined.metal > edges.jsonl
./probe combined.metal 0 > session-0.jsonl
./probe combined.metal 1 > session-1.jsonl
```

These commands are a reproduction sketch. The saved runners additionally
validate pins, preserve failures and record host conditions. Create a new plan
and freeze with empty session directories; never overwrite committed evidence.
