# Rust Star: complete-chunk odd-layer ingress

Native odd-layer ingress now executes all 4096 independent query rows in one call per layer. Projection kernels retain their 32-row geometry and original reductions. Input HC aliases the immutable predecessor buffer instead of staging 128 copies. The oracle and all golden tensors are unchanged.

## Five-pair result

Five alternating AB/BA pairs and one warmup per engine passed on their first attempt. Initial selection belongs to prefill; decode timing includes evaluation, selection and commitment. Correctness collection and profiling are disabled during the paired block. These are 32K/128 development results.

| Metric | Rust median | Oracle-v5 median |
|---|---:|---:|
| Prefill | 371.601 s | 206.136 s |
| Steady decode | 17.264 tokens/s | 17.075 tokens/s |
| First decode step | 67.914 ms | 58.253 ms |
| Peak process memory | 92.250 GiB | 80.895 GiB |

| Pair | Order | Rust prefill s | Reference prefill s | Rust/reference |
|---|---|---:|---:|---:|
| 1 | AB | 370.708 | 206.276 | 1.79715 |
| 2 | BA | 367.192 | 205.494 | 1.78687 |
| 3 | AB | 371.601 | 205.700 | 1.80652 |
| 4 | BA | 372.663 | 206.136 | 1.80785 |
| 5 | AB | 373.212 | 206.192 | 1.81002 |

OBS/camera/video activity appears in the process observations from 2026-09-07T18:04:30.232659+00:00 through 2026-09-07T19:15:10.142323+00:00 (the observer also covers the separate profile). Background load varies across the observations. All pairs are retained; the exact latency improvement cannot be attributed solely to the code change under these varying background conditions.

Host swap before the campaign: `vm.swapusage: total = 3072.00M  used = 1823.19M  free = 1248.81M  (encrypted)`; after the timed block: `vm.swapusage: total = 3072.00M  used = 1791.19M  free = 1280.81M  (encrypted)`. Peak process memory is an observed residency measure under these host conditions, not proof of a lower allocation requirement.

Paired Rust/reference prefill latency ratio: 1.80652. Steady decode ratio: 1.00776. Complete median/MAD/range distributions and independent recomputation are included.

## Change from the attention-batching baseline

- Candidate median prefill: 373.843 → 371.601 s (-0.60% latency).
- Oracle median prefill: 177.723 → 206.136 s (+15.99% latency).
- Rust steady decode: 19.933 → 17.264 tokens/s; paired Rust/reference ratio 1.02630 → 1.00776.
- Reference-normalized prefill improvement across campaigns: 1.162× (paired latency ratio 2.09918 → 1.80652).
- Rust peak memory: 93.229 → 92.250 GiB.

Both campaigns use the same pinned oracle-v5, model, prompt, full-output targets and greedy transcript. They are separate host sessions; the normalized comparison is descriptive. Background applications remained active, and full load/thermal observations are included.
The audit records 0 unusual thermal observations.

## Validation

- Initial candidate 05447277 stopped on a stale schedule-count assertion at layer 3 before HC comparison. The original failed attempt is retained; the corrected candidate completed all gates. No tensor target was changed.
- 310 Rust tests pass. The unchanged Python toolkit retains its 93-test acceptance evidence.
- Fresh complete 32K output-only and diagnostic runs match every prefill logit and all 128 full decode outputs to oracle-v5. Every accepted internal HC target through 24K matches.
- The 12K native control, detailed 8K legacy control, 2K control and two retained reset/reuse cycles pass. No fixture was updated or tolerance introduced.
- The full v5 tensors, reference repeatability and tokenizer evidence are reused unchanged from the audited baseline report. Historical failed oracle/diagnostic attempts remain preserved and explicitly separate from this candidate.
- Existing decoder, prefix, SSD and distributed implementations are unchanged. The 32-row standalone ingress controls remain available; the native chunk uses one complete ingress batch.

## Isolated profile

Host elapsed time includes command waits and is not additive with GPU time.

| Stage group | GPU seconds | Host elapsed seconds | Calls |
|---|---:|---:|---:|
| even/tail | 49.919 | 66.497 | 147 |
| odd/tail | 47.579 | 63.637 | 140 |
| even/attention | 41.213 | 44.582 | 147 |
| bootstrap/bootstrap_ingress | 36.508 | 39.336 | 7 |
| prefix/initial_bootstrap | 28.205 | 28.458 | 1 |
| odd/attention | 27.612 | 31.270 | 140 |
| even/boundary | 23.879 | 32.557 | 147 |
| odd/ingress | 22.709 | 29.797 | 140 |
| prefix/prefix_layer | 20.982 | 29.076 | 40 |
| odd/compressor | 0.546 | 4.091 | 140 |
| even/bind | 0.000 | 3.106 | 147 |
| even/host_scratch | 0.000 | 1.224 | 147 |
| odd/bind | 0.000 | 0.333 | 140 |

Odd-layer ingress API calls fall from 17,920 to 140 (the profile aggregates one ingress stage per layer in both campaigns); GPU time changes from 54.263 to 22.709 s and host time from 81.355 to 29.797 s. This separate instrumented run is ineligible for throughput claims.

The optimization is retained: exact outputs and lifecycle gates pass, and ingress cost falls in the separate profile while the paired latency ratio improves. Absolute paired prefill latency changes by only 0.60% under different host conditions; a precise isolated end-to-end speedup is not established.

The next measured stage group is `even/tail`. Target that cost while preserving full v5 C0 and lifecycle; track peak memory alongside speed. Broader contexts and the 256K headline workload remain future work.

## Reproduction

Candidate source `ecf529a9743baac5e7c182a8c14dfb34d3276984`, executable SHA `adf11352e40be3b006ce3ade1e31bdee687fe81464a83afaed5f17a1e7e67120`.
Oracle source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, executable SHA `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`.
Frozen plan SHA `ec570b76290ba2b994c25bf62956844230d6fa37ce0e482ed1c09e58c7db2f64`.

Run `python3 verify.py` here without the model. It verifies all artifact hashes, raw paired statistics, transcripts, complete v5 tensors, candidate gates and profile coverage. `implementation.patch` records the runtime changes; `inputs/plan-template.json` and `portability.json` retain the frozen configuration and path-redaction provenance.
