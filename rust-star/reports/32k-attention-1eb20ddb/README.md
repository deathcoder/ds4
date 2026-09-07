# Rust Star: complete-chunk even-layer attention

Native continuation attention now executes all 4096 query rows in one call per even layer. Projection kernels retain their 32-row geometry, score kernels retain their 8-query geometry, and attention retains its original per-query reduction. Raw-ring reconstruction and compressed-KV conversion happen once per chunk, and attention writes directly into the existing tail-input buffer. The oracle and all golden tensors are unchanged.

## Five-pair result

Five alternating AB/BA pairs and one warmup per engine passed on their first attempt. Initial selection belongs to prefill; decode timing includes evaluation, selection and commitment. Correctness collection and profiling are disabled during the paired block. These are 32K/128 development results.

| Metric | Rust median | Oracle-v5 median |
|---|---:|---:|
| Prefill | 373.843 s | 177.723 s |
| Steady decode | 19.933 tokens/s | 19.386 tokens/s |
| First decode step | 61.602 ms | 56.158 ms |
| Peak process memory | 93.229 GiB | 80.895 GiB |

| Pair | Order | Rust prefill s | Reference prefill s | Rust/reference |
|---|---|---:|---:|---:|
| 1 | AB | 373.843 | 177.109 | 2.11081 |
| 2 | BA | 372.440 | 177.422 | 2.09918 |
| 3 | AB | 373.280 | 177.723 | 2.10035 |
| 4 | BA | 406.768 | 195.391 | 2.08182 |
| 5 | AB | 419.287 | 208.799 | 2.00809 |

OBS/camera/video activity appears in the process observations from 2026-09-07T16:30:25.538949+00:00 through 2026-09-07T16:58:59.831532+00:00 (the observer also covers the separate profile). Later sample conditions therefore differ from earlier ones. All pairs are retained; the exact latency improvement cannot be attributed solely to the code change under these varying background conditions.

Paired Rust/reference prefill latency ratio: 2.09918. Steady decode ratio: 1.02630. Complete median/MAD/range distributions and independent recomputation are included.

## Change from the frozen v5 baseline

- Candidate median prefill: 423.152 → 373.843 s (-11.65% latency).
- Oracle median prefill: 172.875 → 177.723 s (+2.80% latency).
- Rust steady decode: 20.484 → 19.933 tokens/s; paired Rust/reference ratio 1.04952 → 1.02630.
- Reference-normalized prefill improvement across campaigns: 1.162× (paired latency ratio 2.43919 → 2.09918).
- Rust peak memory: 92.300 → 93.229 GiB.

Both campaigns use the same pinned oracle-v5, model, prompt, full-output targets and greedy transcript. They are separate host sessions; the normalized comparison is descriptive. Background applications remained active, and full load/thermal observations are included.
The audit records 0 unusual thermal observations.

## Validation

- 310 Rust tests pass. The unchanged Python toolkit retains its 93-test acceptance evidence.
- Fresh complete 32K output-only and diagnostic runs match every prefill logit and all 128 full decode outputs to oracle-v5. Every accepted internal HC target through 24K matches.
- The 12K native control, detailed 8K legacy control, 2K control and two retained reset/reuse cycles pass. No fixture was updated or tolerance introduced.
- The full v5 tensors, reference repeatability and tokenizer evidence are reused unchanged from the audited baseline report. Historical failed oracle/diagnostic attempts remain preserved and explicitly separate from this candidate.
- Existing decoder, prefix, SSD and distributed implementations are unchanged. The 32-row standalone attention controls remain available; the native chunk uses one complete attention batch.

## Isolated profile

Host elapsed time includes command waits and is not additive with GPU time.

| Stage group | GPU seconds | Host elapsed seconds | Calls |
|---|---:|---:|---:|
| odd/ingress | 54.263 | 81.355 | 140 |
| even/tail | 49.799 | 64.446 | 147 |
| odd/tail | 47.440 | 62.212 | 140 |
| even/attention | 41.339 | 45.620 | 147 |
| bootstrap/bootstrap_ingress | 36.161 | 40.451 | 7 |
| odd/attention | 29.423 | 33.427 | 140 |
| prefix/initial_bootstrap | 27.930 | 28.189 | 1 |
| even/boundary | 24.744 | 33.143 | 147 |
| prefix/prefix_layer | 20.428 | 28.297 | 40 |
| odd/compressor | 0.717 | 4.405 | 140 |
| even/bind | 0.000 | 4.051 | 147 |
| even/host_scratch | 0.000 | 1.235 | 147 |
| odd/bind | 0.000 | 0.313 | 140 |

Even-layer attention calls fall from 18,816 to 147; GPU time changes from 66.470 to 41.339 s and host time from 110.698 to 45.620 s. This separate instrumented run is ineligible for throughput claims.

The next measured stage group is `odd/ingress`. Target that cost while preserving full v5 C0 and lifecycle; track peak memory alongside speed. Broader contexts and the 256K headline workload remain future work.

## Reproduction

Candidate source `1eb20ddbfa29b893dca8865494e0b442e39e92e4`, executable SHA `a13b8d518c3715ed35f54deecb7845d0dfd3f075545527970c1b69bed1eeb908`.
Oracle source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, executable SHA `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`.
Frozen plan SHA `2ca644b554465dc38d16655872f3f28de184c60003c89e1044e60c0f90df6ed7`.

Run `python3 verify.py` here without the model. It verifies all artifact hashes, raw paired statistics, transcripts, complete v5 tensors, candidate gates and profile coverage. `implementation.patch` records the runtime changes; `inputs/plan-template.json` and `portability.json` retain the frozen configuration and path-redaction provenance.
