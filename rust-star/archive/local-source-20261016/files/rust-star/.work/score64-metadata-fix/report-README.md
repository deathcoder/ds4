# Rust Star: 128K indexer score tiling — retained

The candidate meets the prebound criteria: median paired prefill latency is 8.28% lower, with 5/5 pairs faster. The score tiling change is retained. This is a Rust-versus-Rust development comparison against frozen accepted `ef37e566`. Oracle-v5 remains the independent correctness oracle; this study does not update the Rust-versus-DwarfStar baseline.

The implementation widens prefill indexer score tiles from 8×32 to 8×64. Eight SIMD groups share query staging across twice as many compressed columns, using 256 threads and 20KiB of shared memory. Half conversion, each 8×8 dot product and ordered FP32 accumulation over heads are preserved. One geometry helper covers all three prefix/continuation call sites. No runtime selector flag or alternate numerical contract was added. Decoder scoring is unchanged.

## Five-pair engine evidence

A denotes the accepted Rust baseline and B the candidate. All five alternating pairs and one warm-up per engine use fresh processes, the same 131,072-token prompt and 128 decode steps. Timed runs disable full-output collection and profiling, then require the exact pinned greedy transcript. Every attempt and all outliers are retained. The unchanged runner schema calls steady decode its generic primary metric; this study's separately frozen acceptance plan evaluates prefill latency and guards decode and RSS. Both scopes are recorded in `inputs/protocol-scope.json`.

| Metric | Candidate median | Accepted Rust median | Median paired ratio | Candidate / baseline MAD |
|---|---:|---:|---:|---:|
| Prefill | 1216.402 s | 1358.855 s | 0.917193 | 18.364 / 11.225 |
| Steady decode | 15.145 tokens/s | 14.860 tokens/s | 1.010612 | 0.113 / 0.041 |
| First decode step | 73.073 ms | 74.260 ms | 0.960345 | 2.878 / 4.596 |
| Fresh peak RSS | 94.513 GiB | 94.511 GiB | 1.000015 | 0.001 / 0.001 |
| Process wall time | 1245.752 s | 1388.454 s | 0.918760 | 18.626 / 11.213 |

Steady throughput changes +1.06%, first-step latency -3.97% and fresh peak RSS +0.00% on paired medians. Ratios are computed within pairs; a ratio of separate medians is a different statistic. Full minima, maxima, MADs and all numeric metrics are in `extended-summary.json`.

| Pair | Order | Candidate prefill s | Baseline prefill s | Latency ratio | Candidate steady tokens/s | Baseline steady tokens/s |
|---|---|---:|---:|---:|---:|---:|
| 1 | AB | 1255.312 | 1368.645 | 0.917193 | 15.129 | 14.832 |
| 2 | BA | 1226.875 | 1358.855 | 0.902874 | 14.976 | 14.819 |
| 3 | AB | 1216.402 | 1370.080 | 0.887833 | 15.258 | 14.860 |
| 4 | BA | 1179.245 | 1280.778 | 0.920725 | 15.262 | 15.234 |
| 5 | AB | 1198.037 | 1280.431 | 0.935651 | 15.145 | 15.226 |

| Pair | Candidate first step ms | Baseline first step ms | Latency ratio |
|---|---:|---:|---:|
| 1 | 73.072917 | 79.079958 | 0.924038 |
| 2 | 77.195709 | 78.855875 | 0.978947 |
| 3 | 71.315292 | 74.260083 | 0.960345 |
| 4 | 68.050125 | 70.877791 | 0.960105 |
| 5 | 75.950708 | 68.515500 | 1.108519 |

First-step paired ratios range from 0.924038 to 1.108519; the slower pair is retained in the distribution. Five pairs on one interactive host establish this study's acceptance result, not a universal speedup or confidence bound.

Acceptance was frozen before measurement: median paired prefill ratio below 1.0, at least four faster pairs, steady throughput ratio at least 0.97 and peak-RSS ratio at most 1.03. First-step latency is reported as a tradeoff. `optimization-verdict.json` records each criterion independently. No favorable retry is permitted.

## Correctness and lifecycle

All 317 Rust tests, 94 Python tests and the release build pass. The ten frozen-candidate live gates below pass. Fresh normal and profiled 128K executions each match all 16,677,120 FP32 words across prefill and 128 decode rows. The 32K modes match the complete vocabulary and decode trace; accepted internal HC targets remain limited to 24K. The 12K gate is an internal-state control, not a new complete-output claim.

| Gate | Result | Diagnostic process seconds |
|---|---|---:|
| batch-2k | pass | 70.196 |
| batch-8k | pass | 197.156 |
| batch-12k | pass | 132.017 |
| batch-32k-output | pass | 260.282 |
| batch-32k-internal | pass | 363.135 |
| batch-32k-profiled | pass | 254.531 |
| retained-32k | pass | 513.820 |
| c0-128k | pass | 1267.701 |
| c0-128k-profiled | pass | 1239.080 |
| retained-128k | pass | 2528.817 |

Retained 32K and 128K checks each verify two exact transcripts, request-state reset and model/context/residency reuse. They do not collect every logit or count as paired samples; fresh full-logit C0 remains independent. Their timings and diagnostic RSS do not replace warmed fresh-process engine measurements.

## Validated 128K profiling

The original baseline profile stopped at chunk 57,344 / layer 41 with a non-increasing timestamp. All nine individual component spans were positive; the shared-expert span lay inside routed-down. A nine-pass Metal probe produced overlap in 100/100 unfenced runs and reported 100/100 ordered packets with fences. Raw successful fenced microprobe packets were not emitted; the later full-model profiles do retain all their raw intervals. Apple documents that independent passes may overlap ([Metal 3 session](https://developer.apple.com/videos/play/wwdc2022/10101/?time=413)).

Diagnostic `ec5db25e` adds fences only between the nine tail profile encoders and keeps the strict timestamp guard. It passes full 128K output C0 plus all 1,271 packets / 11,439 intervals. Counter-span coverage is 99.793–99.943% (median 99.896%). The score candidate also passes the full 128K profile and the 32K 287-packet / 2,583-interval control. No 128K decoder-stage profile is claimed.

The baseline diagnostic assigns 49.04% of recorded stage GPU time to attention, with even-layer indexed attention the largest subgroup at 383.761s. This motivated the score-tile experiment. The candidate diagnostic records 284.925s for even-layer attention and 199.100s for odd-layer attention. These are separate perturbed diagnostic runs; scaled tail component shares must not be added to their parent times or used as headline kernel latency.

| Stage | Baseline diagnostic GPU s | Candidate diagnostic GPU s |
|---|---:|---:|
| attention | 581.775 | 484.024 |
| tail | 376.485 | 381.829 |
| bootstrap_ingress | 148.347 | 149.287 |
| ingress | 29.396 | 28.393 |
| boundary | 26.414 | 26.274 |
| prefix_layer | 17.179 | 18.141 |
| initial_bootstrap | 4.962 | 5.106 |
| compressor | 1.812 | 1.820 |

## Isolated score control

Seven shapes, including partial tiles and causal masks, match every one of 167,815,044 score words. Seven alternating timing pairs at each large shape all improve. These synthetic kernel measurements justified full-model testing; they do not establish an engine speedup by themselves. `profile-study/probe-commands.json` records standalone reproduction commands for macOS/Metal.

| Shape | Original kernel median ms | Wider kernel median ms | Median paired ratio |
|---|---:|---:|---:|
| 32K | 178.378 | 117.457 | 0.660566 |
| 128K | 784.685 | 485.343 | 0.619380 |

## Integrity, host conditions and disposition

The first freeze-helper attempt rejected different numbers of empty lines left after removing comments. It occurred before model execution; nonempty kernel code lines were identical. Its record is preserved alongside the original rejected profile. Neither numerical nor timestamp validity was relaxed.

Candidate `7b1a5b58` passed six shorter numerical gates before review caught two stale score-kernel names in its 8K JSON. Validation stopped during retained-32K; that partial gate is excluded and no timed samples had begun. A strengthened writer regression reproduced the defect. Both labels were corrected, a new executable was frozen, and all ten acceptance gates were rerun. `stopped-metadata-attempt/`, `stopped-source/` and `metadata-writer-fix.patch` preserve that history; the six earlier gates do not substitute for final-candidate validation.

There are 625 host observations spanning profiling, validation and measurement, including 0 unusual thermal-query observations. Observed swap use ranges from 0.00 to 32.19MiB. This was an interactive host; isolated probes and model jobs ran sequentially. Host-side build/audit/report work during ineligible diagnostics is distinct from the paired block. RSS values are observations under the recorded warming policy, not universal model/Metal memory bounds.

Candidate source: `b6b0f12530b83caaeb8dfe5310c4dd346368404d`; executable SHA-256 `99f4971e684ed455841065ca597e9ee10055e3a937bbaeb416a9bb9b70800c4c`. Timed Rust baseline: `ef37e56610ef813ed4df3f67f013d62e34995170`; executable SHA-256 `50f59a30144d106564d776a8b7a966d06a3d3b3dbe6c6bc3bf30ad69c81fa752`. Oracle-v5 remains `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. The final model, prompt and oracle hash checks pass.

Run `python3 verify.py` without the model to rehash the complete package, validate oracle row hashes and full-output receipts, recompute profile attribution and all paired distributions, and check the prebound verdict. Kernel probes require the documented Metal device. `candidate-source/`, `diagnostic-source/` and the two implementation patches preserve the exact source states; compiled binaries, the model and private absolute-path plan are excluded. Hash-bound measurement JSON is unchanged. `redactions.json` records raw/stored digests for redacted local paths elsewhere.

Stop at this milestone. 256K measurements and the coding-agent corpus remain future work.
