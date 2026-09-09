# 128K fresh and retained engine characterization

**Complete:** the timed engine now supports the pinned 131,072-token prompt with 128 greedy decode steps, in fresh processes and two-cycle retained-context diagnostics. The candidate passes a new complete full-logit 128K gate and shorter controls. All 317 Rust tests, 94 Python tests and the release build pass.

Five predeclared alternating pairs and one warm-up per engine complete against unchanged oracle-v5, with every native timed transcript exact. This is 128K characterization under the existing protocol, not the 256K primary headline or a coding-agent workload. No required speedup was imposed.

The median paired comparison shows Rust Star prefill latency 34.28% higher, steady decode throughput 2.82% lower, peak RSS 14.15% higher. These measurements characterize the current implementation; correctness and measurement support do not imply a performance advantage.

| Metric | Rust median | Oracle median | Paired Rust/oracle median | Ratio MAD |
|---|---:|---:|---:|---:|
| Prefill (s) | 1494.551 | 1079.569 | 1.342774 | 0.013303 |
| Steady decode (tokens/s) | 14.578 | 15.618 | 0.971815 | 0.000493 |
| All decode (tokens/s) | 14.551 | 15.612 | 0.968691 | 0.002273 |
| First decode step (ms) | 84.848 | 65.880 | 1.287912 | 0.180601 |
| Peak RSS (GiB) | 92.345 | 80.895 | 1.141530 | 0.017045 |
| Process wall (s) | 1525.178 | 1103.056 | 1.342933 | 0.013185 |

Ratios are computed within each pair, then summarized. Lower latency/memory ratios and higher throughput ratios are better. First-step latency is the first decode evaluation and selection; it is not full prompt-to-first-token latency. The unchanged protocol summary is `results/paired-summary.json`. The independently recomputed `extended-summary.json` adds all numeric metrics and paired latency/memory ratios, including min/max and MAD; raw samples remain in `results/paired-raw.json`.

| Pair | Order | Rust prefill (s) | Oracle prefill (s) | Rust steady tokens/s | Oracle steady tokens/s | Rust RSS (GiB) | Oracle RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | AB | 1494.551 | 1079.569 | 14.578 | 15.639 | 88.434 | 80.897 |
| 2 | BA | 1536.733 | 1133.219 | 14.190 | 14.602 | 92.181 | 80.895 |
| 3 | AB | 1524.108 | 1135.044 | 14.187 | 14.590 | 92.345 | 80.896 |
| 4 | BA | 1380.256 | 1030.604 | 15.170 | 15.618 | 94.509 | 80.895 |
| 5 | AB | 1333.170 | 1031.931 | 15.210 | 15.646 | 93.723 | 80.895 |

A is oracle-v5 and B is Rust Star. Both engines warm mapped weights before internal timing; Rust charges Metal residency establishment to prefill. Process peak RSS is the normalized macOS child-process high-water measurement. These warmed measurements are comparable within this paired campaign and must not be compared directly with the unwarmed diagnostic RSS in the preceding correctness report. They are not universal total Metal allocation bounds.

Passive timing attribution across the five native runs places a median 85.64% of prefill wall time in the remaining transformer chain, 11.01% in bootstrap tile execution, and 3.39% in host overhead. These separately summarized shares need not add exactly to 100%; smaller output/handoff/residency intervals also remain. This identifies where to profile next but does not establish which 128K kernel dominates.

Native first-step latency spans 71.871–255.933ms, versus 64.531–69.778ms for the oracle; the slowest native sample is retained. Native peak RSS spans 88.434–94.509GiB. The shared-host setting and these variations limit attribution beyond this paired campaign.

## Retained engine

The same-process 128K diagnostic completes two exact 128-token transcripts after resetting request state. First-load prompt cost including model-page warming is 1386.939s; the retained prompt costs 1377.238s. Model mapping, Metal context, views and residency are reused; KV/compressor/activation state is rebuilt. This single diagnostic is explicitly paired-ineligible and does not establish a retained-service performance distribution.

Retained two-cycle process peak RSS is 94.571GiB. This diagnostic high-water mark is separate from the fresh-process paired distribution and the earlier allocation planning estimate.

Retained correctness checks the full greedy transcript and lifecycle invariants. Independent fresh C0 checks every one of 129,280 prefill and 16,547,840 decode FP32 words. No full-logit collection is performed in timed or retained measurement mode.

## Implementation, validation and limits

Frozen candidate `ef37e56610ef813ed4df3f67f013d62e34995170`, executable SHA-256 `50f59a30144d106564d776a8b7a966d06a3d3b3dbe6c6bc3bf30ad69c81fa752`. A small prompt/selection fixture is imported from the previously repeated 128K reference. Timed prefill permits exactly 128K in addition to 2K/32K; the engine adopts the corresponding decode position/capacity. Sampling, timers, residency/reset rules, kernel arithmetic and chunk-local activation storage are preserved. Built-in conformance commands keep their explicit earlier limits; the external-reference probe validates 128K.

Six pre-pair gates pass: full 128K C0, retained 128K, 32K full output/decode, 2K output, retained 32K and profiled 32K. The component audit accepts 287 native tails and 2,583 intervals. Internal HC acceptance still ends at 24K. Oracle-v5 source/executable and model/prompt pins are unchanged and reverified. An earlier candidate’s retained run was rejected by the old report writer after both execution cycles returned. Its preserved log and metadata are under `rejected/`. The corrected candidate passes all six gates. No timed pair failed or was retried.

Host observations: 472 samples; all thermal/performance queries report no recorded warnings: True; observed swap 35.94–42.19MiB. The machine remained shared with interactive applications. Fresh filtered host snapshots bracket the paired block. These observations do not establish an idle laboratory host.

The paired block completed at 22:02 UTC on September 8. The subsequent read-only input hash verification was interrupted before producing a result; it was resumed on September 9 and passed. No paired sample was rerun or altered. `closeout-resumption.json` records the distinction.

Run `python3 verify.py` from this report directory to verify the portable evidence.

The portable verifier rehashes the artifact inventory, checks reference vectors/provenance, native and lifecycle controls, source snapshots, every paired measurement and both warm-ups, and independently recomputes all summary statistics. Model files, executables and the private local plan are excluded; sanitized command/log metadata and the immutable plan hash are retained.

Proposed next milestone: obtain and validate a 128K component profile of the remaining transformer chain, choose a focused optimization from that evidence, and evaluate it with full C0 controls and pinned paired measurements. This next milestone has not started; 256K and arbitrary coding-agent workloads remain future work.
