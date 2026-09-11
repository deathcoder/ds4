# Routed-prefill grid-order experiment

The isolated candidate passes the predeclared screening criteria. Transposing
only the work/output grid axes reduces median paired fused gate/up GPU latency
by 8.72% for a balanced 4K token batch and 1.91% for a skewed 4K batch. All fourteen
4K pairs improve; every output comparison is exact. Production remains at
d6c14fa0. This report establishes a synthetic kernel result, not full-model C0
or an engine throughput improvement.

## Why this experiment

The latest accepted 32K profile from `128k-score64-b6b0f125` records 86.851s of
GPU time in native tails. Its component attribution assigns 32.201s to attention
output and 30.598s to routed gate/up. Routed gate/up accounts for 35.23% of the
tail and 14.52% of all recorded stage GPU time. These are perturbed diagnostic
estimates: component times partition their parent and must not be added to it.

The earlier dense attention-output experiment was already rejected. This study
instead tests locality in the unchanged fused IQ2_XXS gate/up kernel. The
baseline dispatch uses `(work_capacity, 32, 1)` and reads work from grid x and
output tile from grid y. The candidate uses `(32, work_capacity, 1)` and swaps
those two coordinate reads. The 64x32 MMA tile, 128 threads, 16KiB shared memory,
K accumulation order, dequantization, SwiGLU, route weighting and output stores
are identical. Better data reuse is a hypothesis; hardware cache counters were
not collected, so this study does not prove the causal mechanism.

## Results

| Batch rows / routing | Baseline median GPU ms | Candidate median GPU ms | Median paired latency ratio | Faster pairs |
|---|---:|---:|---:|---:|
| 2,048 / balanced | 56.395 | 55.792 | 0.992044 | 5/7 |
| 4,096 / balanced | 91.906 | 84.194 | 0.912801 | 7/7 |
| 4,096 / skewed | 93.963 | 91.252 | 0.980903 | 7/7 |

Ratios are computed within each pair; ratios of the separate medians differ.
`result.json` retains all 21 pairs plus median, MAD, minimum and maximum for
both variants and paired ratios. `probe.jsonl` preserves the first baseline
execution, all warmups, controls and measured samples. No run was retried or
sample discarded. The complete probe took 10.138s including shader compilation,
allocation, data generation and verification; peak process RSS was 1.555GiB.

At the recorded 14.52% stage share, transferring these 4K kernel savings unchanged
would reduce summed stage GPU time by only approximately 0.28–1.27%. That is a
rough arithmetic extrapolation across diagnostic and synthetic data, not a
prediction or measurement of end-to-end prefill latency. It does not explain
most of the historical Rust/reference gap.

## Exactness and limits

- The baseline imported MoE source is byte-identical to accepted native b6b0f125.
  Current production sources remain unchanged. Oracle-v5 stays independently
  pinned at ccbd16d36895ef9fe8edf7f793014988ff1ab7f4.
- Eight shapes cover empty experts, partial and full 32-row expert tiles,
  over-dispatched capacity slots, balanced routing and skew toward 16 experts.
  Each token has six distinct routes among 256 experts; input width is 4096
  and output width is 2048. Batch sizes describe kernel inputs, not measured
  complete-model context frontiers.
- Weights, normalized-input-like values and route weights are deterministic
  synthetic data. The probe uses the real production IQ2_XXS format and GPU
  mapping kernel; it independently verifies that GPU map against CPU route
  enumeration before comparing variants. It does not load the real model.
- All 64 post-baseline executions match all output FP16 bit patterns and retain
  guard bytes: 2,270,429,184 word comparisons. Baseline outputs are finite and
  nondegenerate; hashes and nonzero/negative counts are retained. This checks
  the fused kernel boundary, not final FP32 logits or downstream model C0.
- Each timed case has two warmups per variant followed by seven alternating
  AB/BA pairs. GPU timestamps cover one completed kernel command. Allocation,
  routing-map preparation, CPU output checks and sentinel resets are excluded
  from those kernel intervals. This is not the fresh-process engine protocol.
- Before/after host load, thermal, compression and swap observations are included;
  this is one short run on an interactive M1 Ultra, without continuous observation
  or hardware cache profiling. No full-model job overlapped this probe.

An initial preparation assertion incorrectly expected textual identity between
native and oracle-v5 template bodies. It failed before compilation or GPU work.
The native comparator was then bound to its actual accepted source; no numerical
criterion was relaxed. `setup-failure.json` preserves this preparation failure.
There were no compiler or GPU failures in the frozen experiment.

## Reproduction and disposition

On any host with Python 3, verify the complete package without the model:

```sh
python3 verify.py
```

On the target M1 Ultra, compile and execute the included standalone shaders:

```sh
clang -fobjc-arc -O2 -framework Foundation -framework Metal route-order-probe.m -o route-order-probe
./route-order-probe baseline.metal candidate.metal > reproduction.jsonl 2> reproduction.stderr
```

Preserve the original artifacts. `prepare.py` records source extraction from the
repository; it is not needed to run the self-contained probe. Build and shader
hashes, predeclared hypothesis and all raw samples are included. The binary
remains in local ignored evidence and is not a product release.

The candidate is eligible for one bounded real-model validation step: apply
only this coordinate remapping to a separate candidate shader, run the existing
32K full-output C0 gate, and inspect a matched lower-context diagnostic before
considering a larger engine timing campaign. The kernel result alone does not
justify accepting a production optimization. The 256K campaign remains paused.
