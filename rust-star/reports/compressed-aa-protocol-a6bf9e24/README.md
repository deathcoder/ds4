# Compressed-attention A/A timing protocol

The predeclared equal-warm protocol passes all six shape/session null screens.
Median first/second GPU-time ratios stay within 0.487% of parity, compared
with 6.60–15.28% first-slot penalties in the contemporary balanced legacy-style
control. All 20,937,965,568 final-output FP32 comparisons pass.

Retain this as a usable isolated measurement protocol for this kernel and
these shapes, with a contemporaneous A/A control required for future candidate
studies. This is measurement-method evidence: no runtime optimization, engine
speedup, new C0 frontier or Rust-versus-DwarfStar baseline is established.

## Scope and immutable inputs

- Starting commit: `a6bf9e24`; runtime remains `1711856e`.
- Target: Apple M1 Ultra. Both timed roles use the same original compressed
  FlashAttention pipeline object, function constants, arguments and grid.
- The original combined Metal library and ABI header are unchanged. Its old
  head-first wrapper is compiled during setup but never dispatched in this study.
- Synthetic queries, shared K/V, sinks and causal masks match the prior probe
  for chunk starts 4096, 16384 and 28672, including padded and unpadded KV.
- Plan, harness, executable, audit and runner were hashed before execution.
  All 40 runtime/imported sources match before each process and after it.
- The previous failed candidate and earlier A/A evidence remain unchanged in
  [the attention report](../32k-attention-1711856e/README.md).

## Predeclared experiment

Two fresh processes execute shapes in forward and reverse order. Each shape
has 12 pairs per protocol, with six AB and six BA pairs and balanced assignment
of roles to physical output buffers. Protocol order alternates within pairs
and is reversed in the second process. There are 144 measured pairs in total.

Both protocols poison two guarded 512 MiB output buffers before each pair.
CPU validation occurs after both roles. The balanced legacy-style control
uses one command buffer per role with no immediate conditioning. Equal-warm
pre-encodes and queues four identical conditioning dispatches followed by
three measured dispatches for each role, then waits for the last command.
Its slot time is the arithmetic mean of the three measured GPU intervals.
Every command buffer contains one attention dispatch; warm-up and queue gaps
are excluded from the reported slot average and all raw intervals are retained.

This control differs from the older seven-pair experiment through balanced
order, buffer crossover and interleaving with the warm protocol. Its absolute
bias is not a directly comparable historical metric.

The acceptance rule required every warmed shape in both sessions to have:

- First/second, AB-role, BA-role and physical-buffer median ratios within 1%.
- First/second ratio median absolute deviation no greater than 1%.
- At least 10 of 12 first/second ratios within 3% of parity.

Both sessions were fixed in the original plan. There were no failed executions,
retuned thresholds, discarded pairs or extra timing repeats.

## Results

Ratios below are first-slot time divided by second-slot time. Each cell is
the median of 12 pairs. Values above 1 mean the first slot took longer.

| Session | Chunk start | Legacy-style | Equal-warm | Warm ratio MAD | Within 3% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4096 | 1.088620 | 0.998509 | 0.003891 | 12/12 |
| 0 | 16384 | 1.107297 | 1.004863 | 0.006017 | 12/12 |
| 0 | 28672 | 1.074229 | 0.998895 | 0.004332 | 12/12 |
| 1 | 28672 | 1.066019 | 1.001503 | 0.002417 | 12/12 |
| 1 | 16384 | 1.090599 | 0.998989 | 0.003466 | 12/12 |
| 1 | 4096 | 1.152751 | 1.004433 | 0.004377 | 12/12 |

Every warmed AB/BA median also stays within 0.487% of parity; every physical
buffer median stays within 0.324%. All 72 warmed pairs fall within 3%. All
six legacy-style cases fail the same null screen. See `audit.json` for every
gate, sample range, role median and dispersion; raw command intervals are in
the two `session-*/probe.jsonl` files.

The processes took 117.988s and 116.902s including initialization, compilation,
CPU preparation, conditioning and validation. These are harness process times.
Host observations report zero swap usage and no recorded thermal/performance
warning; they do not measure GPU clocks. No other model/probe process was
present at either preflight.

## Correctness and limits

Each initial warm-up pair and measured pair checks all 134,217,728 FP32 output
words for bitwise equality, finite/nondegenerate values and the previous
immutable output checksum. The harness checks destination guards, unchanged
full input hashes, independently prepared CPU/GPU mask blocks and padded KV.
The portable audit verifies these receipts and reconstructs all timing stats.

Intermediate repeated dispatch outputs are overwritten; only the final output
for each role in a pair is compared. The arithmetic and dispatch are identical
throughout. This is isolated synthetic kernel evidence, not a new full-model
C0 run. Runtime code is unchanged, so no new Rust test/build campaign or model
load was performed. The new Objective-C harness builds and both GPU runs pass.

Equal conditioning, queued submission and three-sample averaging changed
together. This shows that the combined protocol controls the observed order
effect under this experiment. It does not distinguish CPU-gap/cache effects
from GPU clock transitions, prove neutrality for other kernels or justify
retroactive correction of the failed candidate. Conditioning also creates a
warm repeated-kernel regime, which may differ from normal engine scheduling.

## Verification and reproduction

Verify all portable artifacts without loading the model or using the GPU:

```sh
python3 rust-star/reports/compressed-aa-protocol-a6bf9e24/verify.py
```

The verifier checks the complete artifact inventory, original/stored hashes
for path redactions, frozen plan/source/run bindings, library assembly and ABI,
unchanged dispatch/preparation/validation bodies, timestamp ordering, exactness
receipts, schedule balance and the preset null criteria.

For a deliberate new experiment on the target Mac, copy `probe.m`,
`arguments.h` and `combined.metal` into a fresh output directory and build:

```sh
clang -O3 -fobjc-arc -framework Foundation -framework Metal probe.m -o probe
./probe combined.metal 0 > session-0.jsonl
./probe combined.metal 1 > session-1.jsonl
```

Those commands are a manual reproduction sketch; recorded host observations,
source/binary pin checks and failure preservation are implemented in `run.py`.
Create a new plan/freeze and use empty result directories for new evidence.
Never overwrite the committed results. The original executable is omitted
from this portable bundle but its hash remains bound in `freeze.json`.

## Next action

Use the validated conditioning protocol in a newly predeclared compressed-
attention candidate screen with an accompanying A/A null control. Require
improvement in both AB and BA orders and across shapes; do not turn the old
candidate into a win through normalization. Only a successful new screen
justifies runtime integration, full-output C0 and unprofiled engine pairs.
Keep the 256K campaign paused. No push or release is part of this study.
