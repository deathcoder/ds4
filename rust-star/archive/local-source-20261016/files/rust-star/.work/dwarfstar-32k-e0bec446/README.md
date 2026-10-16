# Accepted Rust Star versus pinned DwarfStar at 32K

Measurement milestone complete. Five fresh pairs compare accepted Rust `e0bec446` with DwarfStar oracle-v5 `ccbd16d3`, using the same model and 32K prompt plus 128 decode steps. Median paired Rust/DwarfStar prefill latency is **1.168532** (+16.85%). Steady decode throughput is **+2.00%** on the paired median. Runtime sources and the accepted executable are unchanged.

## Protocol and identity

One excluded warmup per engine precedes five AB/BA/AB/BA/AB pairs; A is DwarfStar and B is Rust. All twelve fresh processes finish in the frozen order. There are no failed engine attempts, repeats or speed-based exclusions. Built-in weight warming is enabled, profiling and correctness collection are disabled, and both adapters require sampling-inclusive timing. Prefill includes initial selection; each decode interval includes evaluation and the resulting token selection. First decode latency is not time to first output.

Rust executable SHA-256: `7590af7ade11d5b07d65f0fefc9e01c02668318896fc4e955307888d85cdaa35`. DwarfStar: `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`. The reference uses its pinned original benchmark driver without an overlay or rebuild. The accepted Rust build uses Rust 1.74. Full model, prompt, oracle, 7,669 fixture and 42 runtime-source hashes match before and after; smaller frozen source/executable/protocol checks pass before every process.

## Results

Values are median ± median absolute deviation across five timed runs per engine. Full ranges, each pair and order subsets are in `evidence/audit.json`. Five pairs do not establish statistical significance.

| Metric | Rust Star | DwarfStar |
|---|---:|---:|
| Prefill | 205.021 ± 0.256 s | 175.261 ± 0.280 s |
| Prefill throughput | 159.827 ± 0.200 tokens/s | 186.967 ± 0.298 tokens/s |
| Steady decode | 20.137 ± 0.002 tokens/s | 19.751 ± 0.037 tokens/s |
| All 128 decode steps | 20.093 ± 0.013 tokens/s | 19.739 ± 0.032 tokens/s |
| First decode step | 63.342 ± 1.012 ms | 53.010 ± 2.298 ms |
| Whole process | 233.042 ± 0.108 s | 195.572 ± 0.154 s |
| Peak process RSS | 85.366 ± 0.004 GiB | 80.895 ± 0.000 GiB |

| Pair | Order | Rust/Dwarf prefill latency | Rust/Dwarf steady decode |
|---|---|---:|---:|
| 1 | AB | 1.123543 | 1.019417 |
| 2 | BA | 1.174759 | 1.022232 |
| 3 | AB | 1.169805 | 1.015818 |
| 4 | BA | 1.168532 | 1.020912 |
| 5 | AB | 1.167928 | 1.020016 |

AB and BA median paired prefill ratios are 1.167928 and 1.171646; steady-decode ratios are 1.019417 and 1.021572. The ratio of prefill latency medians is 1.169805; it is a different statistic from the median paired ratio.

There are 105 passive host observations and 0 unusual thermal/performance records. Desktop applications remained open; this is not a dedicated idle-host experiment. Raw load, swap and VM observations are preserved. RSS is measured per isolated child process; it does not attribute all memory to live model or KV allocations. Reference zero snapshot bytes means unavailable snapshot size at the terminal frontier, not zero KV memory. No comparable DwarfStar peak-footprint metric is supplied by this adapter.

## Correctness and evidence limits

This milestone reuses the [accepted full-logit C0 evidence](../selective-clear-db1de4a9/README.md) for unchanged code and binaries. The prior normal and profiled production runs compared 33,354,240 FP32 words, with another 16,677,120 in the separate ownership-observer run. Rust timing runs additionally check the fixed 128-step transcript. The DwarfStar CSV does not expose per-run token IDs: no new DwarfStar transcript or full-logit claim is made from these measurements. The pinned oracle-v5 producer acceptance remains the correctness reference.

The saved accepted profile is replayed from raw records: all 4,466 intervals regenerate byte-identically. An initial replay assertion compared Python tuples to JSON lists in the existing-file branch. Replaying the unchanged audit into an empty temporary directory reproduces every saved byte. The original failed replay and its diagnosis are preserved; no inference run or metric was changed.

## Next milestone — not started

Investigate routed gate/up projection in the accepted 32K tail. It remains the largest individual recorded GPU component at 30.150 s, followed by compressed attention at 24.503 s and attention output at 21.238 s. Tail totals 75.379 s and attention totals 56.999 s in the separate perturbed profile. These component measurements do not partition unprofiled engine wall time and do not identify the cause of the Rust/reference gap by themselves.

The next bounded milestone should audit whether wider token tiles can reuse dequantized expert weights while preserving each output accumulation order, then screen one bounded layout or scheduling hypothesis with exact intermediate comparisons and contemporaneous A/A controls. Preserve the already failed routed-input-half and longer-median screens; do not repeat them or reinterpret their failed controls as gains. Freeze the hypothesis and selection criteria before timing, stop if neutrality fails, and require separate full C0 plus whole-engine evidence before retaining an optimization. No candidate is implemented in this comparison milestone.

## Verification

Run `python3 rust-star/reports/dwarfstar-32k-e0bec446/verify.py`. It verifies every packaged artifact, all twelve measurement bundles, timing and transcript parser contracts, frozen identities and before/after inputs; recomputes all paired statistics; and replays the accepted profile. Negative controls corrupt DwarfStar sampling metadata and Rust transcript metadata while updating their artifact hashes, and must still be rejected. No model, GPU or compiler is needed.

This is a local 32K development result, not a project completion or larger-frontier claim. Oracle-v5 remains immutable, 256K remains paused and publication remains deferred.
