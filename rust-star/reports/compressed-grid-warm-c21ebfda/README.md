# Fresh compressed-attention grid screen with A/A controls

Reject the head-first scheduling candidate under this predeclared screen.
All six shape/session cases fail the speed criteria: candidate/original
medians are 1.36–2.83% slower separately in AB and BA order, and only 2 of 72
candidate pairs improve. Every contemporary A/A null gate passes.

All 23,085,449,216 final-output FP32 comparisons pass. No runtime integration
or new engine/model run is justified by this result. Runtime remains 1711856e.

## Why this was a new experiment

The earlier head-first screen failed its criteria and exposed substantial
run-order sensitivity. That report remains immutable. The subsequent
[A/A protocol study](../compressed-aa-protocol-a6bf9e24/README.md) validated
equal conditioning, queued submission, averaging and balanced buffer/order
assignment. This study prospectively tests the same source-level candidate
with that method and interleaved contemporary A/A controls. It does not
normalize, rewrite or retroactively accept the earlier measurement.

The component choice came from the [32K attention diagnostic](../32k-attention-1711856e/README.md): compressed
attention accounted for 26.158s, approximately 12.04% of summed recorded GPU
stage duration. That profile is perturbed and cannot establish normal engine
throughput; it identifies a component worth screening.

## Fixed scope and criteria

- Starting commit: `c21ebfda`; all 40 runtime/imported source hashes match
  runtime `1711856e` before and after both processes.
- Hardware recheck: Apple M1 Ultra, 48 GPU cores, 128GB unified memory.
- Candidate: the existing head-first wrapper forwards swapped threadgroup
  coordinates into the unchanged original FlashAttention implementation.
  The host transposes the grid. Arguments, masks and arithmetic are unchanged.
- A/A: both roles dispatch the same original pipeline object and original grid.
- Correctness: all seven production chunk shapes, with deterministic original
  inputs and both padded/unpadded KV paths, in both fresh processes.
- Timing: chunk starts 4096, 16384 and 28672; 12 pairs per protocol per shape
  per process. AB/BA order, physical buffers and protocol order are balanced.

Each role receives four conditioning dispatches followed by three measured
dispatches, all encoded and queued before waiting for the final command.
The reported slot time averages those three GPU command-buffer intervals.
Both output buffers are poisoned before a pair; CPU validation follows both
roles. All conditioning/sample intervals are retained. They describe a warm
repeated-kernel regime, not production engine scheduling.

Selection required all correctness checks and all six contemporary A/A cases
to pass the inherited null rules: first/second, AB/BA and physical-buffer
median ratios within 1%, ratio MAD at most 1%, and at least 10/12 pairs within
3% of parity. Each candidate shape/session additionally needed a median
candidate/original ratio below 0.99 separately in AB and BA order, with at
least 5/6 faster pairs in each order. These criteria were frozen before either
process; no threshold changed, pair was discarded or extra run was added.

## Candidate results

Every ratio is candidate GPU time divided by original GPU time; values above
1 mean the candidate is slower. Each order has six pairs.

| Session | Chunk start | AB median | BA median | Faster AB | Faster BA | Selection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4096 | 1.025025 | 1.025329 | 0/6 | 0/6 | Fail |
| 0 | 16384 | 1.016950 | 1.013566 | 0/6 | 0/6 | Fail |
| 0 | 28672 | 1.022825 | 1.025588 | 0/6 | 0/6 | Fail |
| 1 | 28672 | 1.024308 | 1.027217 | 1/6 | 1/6 | Fail |
| 1 | 16384 | 1.019955 | 1.014394 | 0/6 | 0/6 | Fail |
| 1 | 4096 | 1.028309 | 1.024560 | 0/6 | 0/6 | Fail |

The two faster pairs at chunk start 28672 in session 1 remain in the raw
data and statistics. They do not rescue either order median or the declared
consistency threshold. This is a scoped negative kernel result; it does not
measure an engine regression because the candidate was never integrated.

## Contemporary null controls

Every one of the six A/A cases passes every predeclared condition:

- Maximum first/second median deviation: 0.170%.
- Maximum AB/BA role-median deviation: 0.398%.
- Maximum physical-buffer median deviation: 0.561%.
- Maximum first/second ratio MAD: 0.641%.
- 71/72 ratios lie within 3%; each individual case meets the 10/12 minimum.

This supports the screen decision in this experiment. It does not prove
neutrality for arbitrary kernels, workload distributions or host conditions.
No candidate timings are divided by the A/A timings or otherwise corrected.

## Correctness and execution evidence

There are 144 measured pairs (72 candidate, 72 A/A), plus 28 initial warm-up
pairs comparing original/candidate over all seven shapes in two processes.
Each checked pair compares all 134,217,728 final FP32 words bitwise, checks
finite/nondegenerate outputs, and matches the immutable original checksum.
The harness also checks output guards, full input immutability and independent
CPU/GPU block-map, mask-padding and KV-padding preparation.

Intermediate repeated dispatch outputs are overwritten; only each role's
final output is compared after a pair. This remains isolated synthetic
correctness evidence, not a new full-model C0 run. The new Objective-C harness
builds, its library/ABI/arithmetic source proof passes, and both live runs
finish normally. Runtime sources are unchanged; no new Rust test/build or
model/oracle inference campaign was needed.

The processes took 196.957s and 193.930s including initialization, compilation,
conditioning and CPU checks. These are harness process times. Host snapshots
show 57.25MB swap usage unchanged throughout and no recorded thermal/performance
warning. GPU clocks were not measured. No other model/probe process was
present at either preflight. There were no failed executions or retries.

## Verify or reproduce

Verify the portable package without a model or GPU:

```sh
python3 rust-star/reports/compressed-grid-warm-c21ebfda/verify.py
```

The verifier checks the artifact inventory, frozen plan/source/binary bindings,
path-redaction original/stored hashes, exact library assembly and ABI, original
wrapper restoration, unchanged preparation/validation and timing-slot bodies,
all 2,016 recorded GPU intervals, exactness receipts, balanced schedule and
independently recomputed null/candidate criteria.

To deliberately reproduce in a fresh directory on the target machine, copy
`probe.m`, `arguments.h` and `combined.metal`, then build and run:

```sh
clang -O3 -fobjc-arc -framework Foundation -framework Metal probe.m -o probe
./probe combined.metal 0 > session-0.jsonl
./probe combined.metal 1 > session-1.jsonl
```

This is a manual reproduction sketch. `run.py` additionally performs source
and binary pin checks, records host observations and preserves failures.
Create a new plan/freeze and empty session directories for any future study.
Do not overwrite committed evidence. The executable is omitted from this
portable bundle; its original SHA-256 remains in `freeze.json`.

## Decision and next action

Keep the head-first candidate out of the runtime. Retain the validated timing
method and both negative screens. A useful next investigation is the cost of
traversing fully masked cache blocks in compressed attention: the current
kernel walks their block markers and advances mask pointers before skipping
arithmetic. Any change must preserve active-block order, padding behavior and
exact outputs, and earn selection with a bounded paired screen plus A/A.
That investigation has not been implemented or measured here.

No push, release, new frontier or 256K resumption is part of this task.
