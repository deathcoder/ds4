# Native wide bootstrap: complete 32K milestone

Decision: **reject the candidate as an engine improvement** under the criteria fixed before measurement. This compares Rust `c7adfb22` with frozen accepted Rust `4ced6ee8`; it does not measure a DwarfStar advantage.

Median paired prefill latency changes by -1.127%. Steady decode changes by +0.556% and peak RSS by -0.997%. The separate macOS peak-memory-footprint metric changes by +52.484% (baseline 11.539 GiB, candidate 17.593 GiB). Every raw pair and both order subsets are retained below.

## Implementation and scope

In this candidate, native continuation at overall frontiers up to 32K processes the layers-0/1/2 bootstrap in one 4,096-row batch per chunk instead of sixty-four 64-row calls. Initial prefix, 2K, legacy callers and overall frontiers above 32K keep their existing schedules. Position arrays cover the larger batch; diagnostic raw-ring snapshots retain the final 128 rows. No shader arithmetic changes. Three complete source files restore byte-for-byte to 4ced6ee8 by reversing the recorded edits; the other 39 pinned sources are unchanged.

The first candidate, 7b4b76ac, failed its first 12K internal gate at chunk 4096, layer 0 final HC; every initial prefix layer was exact. Its later correctness and timing runs did not execute. Inspection found four router launches with 4,096 threads per group although both compiled pipelines permit only 1,024. The repaired candidate uses MIN(rows,256) for those four launches, preserving short widths. The actual unchanged Metal kernels pass 29,344 output-word comparisons and 96 guards at 32/64/4,096 rows. The repaired full-model checks subsequently pass the original failure site and all remaining gates. The first compile-only failure and first model failure are preserved.

## Correctness and diagnostic selection

All 320 release host tests pass. Forced native release build, formatting, executable/source freeze, complete inverse-edit proof and final full model/prompt/oracle/binary/source hashes pass. The 12K internal run checks all 43 layers of the initial prefix and both native continuation chunks. Separate normal and profiled 32K/128 C0 runs each check all 16,677,120 FP32 logit words. The fresh baseline profile checks the same full output, making 50,031,360 FP32 logit-word comparisons across these three full runs. Token agreement alone is not C0.

Strict counter audits validate every attention/tail packet, interval, order and attribution. Bootstrap-ingress recorded GPU time changes from 31.943s to 6.701s: raw ratio 0.209769, control-normalized ratio 0.202812. Both meet the prospectively fixed <0.90 diagnostic gate. This is selection evidence; whole-engine results come from the separate unprofiled runs.

## Whole-engine results

Two fresh-process warm-ups are preserved and excluded. Five complete pairs use orders AB, BA, AB, BA, AB with fixed 32K input and 128 greedy steps. All 12 exact transcripts, raw engine metadata, timing-off checks, host observations and process exit records pass. No favorable early stopping or discretionary reruns.

| Pair | Order | Baseline prefill s | Candidate prefill s | Prefill change | Steady decode change | RSS change |
|---|---|---:|---:|---:|---:|---:|
| 1 | AB | 234.538 | 250.461 | +6.789% | -0.571% | -0.997% |
| 2 | BA | 242.149 | 239.420 | -1.127% | +1.450% | -0.663% |
| 3 | AB | 238.811 | 225.593 | -5.535% | +3.330% | -1.528% |
| 4 | BA | 244.422 | 230.161 | -5.834% | +0.043% | -1.527% |
| 5 | AB | 262.539 | 264.022 | +0.565% | +0.556% | -0.000% |

| Metric | Baseline median | Candidate median | Paired ratio median | Ratio MAD | Ratio range |
|---|---:|---:|---:|---:|---|
| prefill_ms | 242148.773 | 239419.839 | 0.988730 | 0.044078 | 0.941655–1.067889 |
| prefill_tps | 135.322 | 136.864 | 1.011398 | 0.047192 | 0.936427–1.061960 |
| gen_steady_tps | 20.043 | 20.092 | 1.005555 | 0.008948 | 0.994295–1.033300 |
| gen_tps | 20.006 | 20.057 | 1.005472 | 0.009190 | 0.994289–1.033553 |
| gen_first_ms | 60.698 | 60.041 | 0.969438 | 0.026226 | 0.943212–1.006316 |
| process_wall_seconds | 270.110 | 268.333 | 0.993420 | 0.042236 | 0.945009–1.068686 |
| peak_rss_bytes | 89553108992.000 | 88656986112.000 | 0.990034 | 0.005301 | 0.984719–0.999999 |
| peak_memory_footprint_bytes | 12389959536.000 | 18890639240.000 | 1.524844 | 0.000250 | 1.524572–1.525584 |

## Passive phase breakdown

These timings were already recorded by the unprofiled engine. They are descriptive and do not alter the decision. Command wall minus GPU is an aggregate timing difference; it does not identify a specific stall or allocation cause. The bootstrap tile GPU metric spans the first command start through the last command end, including any gaps between commands; it is not a sum of isolated kernel durations. Raw per-pair and warm-up values are in [passive-phase-audit.json](passive-phase-audit.json).

| Phase | Baseline median s | Candidate median s | Median paired difference s |
|---|---:|---:|---:|
| prefill_bootstrap_setup_ms | 0.152 | 1.562 | +1.411 |
| prefill_tile_wall_ms | 37.707 | 20.610 | -16.477 |
| prefill_tile_gpu_ms | 37.505 | 20.020 | -16.852 |
| prefill_transformer_setup_ms | 5.190 | 10.183 | +3.833 |
| prefill_transformer_wall_ms | 196.685 | 205.538 | +6.710 |
| prefill_transformer_gpu_ms | 160.328 | 159.507 | -3.339 |
| transformer_command_wall_minus_gpu_ms | 35.345 | 46.547 | +10.018 |
| prefill_host_overhead_ms | 7.194 | 13.130 | +4.960 |

## Preset decision rules

All five prefill pairs must be faster; separate AB and BA median prefill ratios must be below 0.99. Steady decode median must be at least 0.97, with each order subset at least 0.95. Median peak-RSS ratio must be at most 1.02. Exactness and valid timing are required for every run. Thresholds were not adjusted after observing results. The separate peak memory footprint is reported descriptively from the already captured time logs; it is not substituted for RSS or used to invent a new acceptance threshold.

| Gate | Result |
|---|---|
| all exact transcripts | PASS |
| all five prefill faster | FAIL |
| both order prefill medians | FAIL |
| steady decode median | PASS |
| steady decode both orders | PASS |
| peak rss median | PASS |

All role and paired distributions, warm-ups, order subsets and raw measurements are in [engine/audit.json](engine/audit.json), with the distinct memory-footprint metric in [memory-footprint-audit.json](memory-footprint-audit.json). This small paired study does not establish statistical significance.

## Reproduce and verify

Run `python3 rust-star/reports/wide-bootstrap-c7adfb22/verify-report.py` from the checkout. The verifier checks the artifact manifest and original/stored hashes, all three complete source restorations, preserved failed attempt, dispatch proof, strict profiles, full C0 records, every engine run and all decision rules. It needs Python 3 and the included standard-library adapters; it does not load the model or rerun GPU inference.

The original scripts and plans are preserved. `portable-engine-audit.py` changes only the unused repository-root assignment, adapter import and manifest-verified raw-hash lookup. Both portable profile audits change only hash lookup. Private machine paths are replaced by explicit placeholders, with original and stored hashes recorded separately. Executables and model are omitted; executable identities and complete source inputs are retained.

To rerun on the fixed target, build runtime commit `c7adfb22706d5a5883d8a251a3a9046a2c8e587b` using the recorded toolchain, then follow the frozen plan and runners with the pinned model/oracle artifacts. A new execution must use a fresh directory and preserve all attempts. Do not overwrite this evidence.

The new batching path has full-model evidence at 12K internal checkpoints and 32K/128 outputs only. No fresh 128K/256K claim follows; the old 256K attempt remains paused. Current-source higher-frontier correctness and a refreshed direct DwarfStar comparison remain separate project milestones. No publication was performed.
