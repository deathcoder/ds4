# Explicit initialized bootstrap scratch reuse

Rejected: peak RSS increased 2.415%, above the unchanged 2% limit. Accepted source/default restored.

One distinct buffer per scratch name is owned across each four-tile native 1K
bootstrap chunk. Every checkout restores the complete original zero/copy initial
state after checked GPU completion; existing CPU initialization follows unchanged.
The owner clears at final completion/error, and persistent KV/compressor state
stays outside the pool. Scope remains the selected 32K path; legacy/diagnostic
widths, higher-frontier selection, decode and shader arithmetic stay unchanged.
See [ownership proof](scope-proof.md) and [source audit](ownership-source-audit.json).

The actual GPU helper proof passes 62,043,240 byte comparisons, 192,360 guard
bytes, 2,205 identity reuses and 84 rejected invalid checkouts. All 320 Rust tests,
optimized production/diagnostic builds, formatting, 1,567 ring cases and the
7,840-word/96-guard router probe pass. 12K internal checks and three fresh 32K/128
full-C0 runs pass 50,031,360 exact FP32 comparisons, including every logit.
All 4,466 profile intervals pass. Full model/oracle/prompt / 7,669-fixture hashes
match before and after inference. No inference run was repeated.

The pool trace passes all 28 tiles / 119 snapshots / 2,940 checkouts in 7 chunks:
105 distinct buffer identities remain stable within each chunk; prior GPU work
is completed before reuse, no old chunk survives into the next entry, capacities
match exactly, and at most 1 command is unfinished. Maximum sampled logical
scratch is 2,493,173,233 bytes. The context owner clears after every final tile;
all final weak samples happen to be zero. Logical lengths are not OS footprint.

**Preserved audit correction.** The original observer writes a nil-command
sentinel as unsigned NSUIntegerMax (18446744073709551615), while the frozen
auditor expected signed -1. A standalone CPU-only Foundation probe confirmed
that representation before pool results were examined. The original assertion
and failed driver statuses are preserved. audit-pool-v2.py changes only that
first-checkout sentinel: every reuse still requires status 4 (Completed). No
runtime/observer change, capacity/performance threshold change or inference
retry occurred. Three source/observer generator assertion failures before builds
are also preserved with their corrections.

**Whole-engine result.** Current Rust 1.74 baseline versus candidate on the
M1 Ultra, 32K prefill plus 128 decode tokens. Two excluded warmups followed by
five fixed AB/BA/AB/BA/AB pairs; all 12 fresh processes preserve the exact oracle
transcript. Profiling/correctness collection is disabled for these timings.

| Metric | Baseline median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Prefill | 228.532 s | 205.402 s | -10.243% |
| Steady decode | 20.185 tok/s | 20.187 tok/s | -0.005% |
| First decode token | 61.476 ms | 62.428 ms | -0.093% |
| Peak RSS | 90.035 GB | 92.210 GB | +2.415% |
| Peak memory footprint | 12.385 GB | 13.841 GB | +11.663% |

| Pair | Order | Prefill B/A | RSS B/A | Footprint B/A |
|---|---|---:|---:|---:|
| 1 | AB | 0.886637 | 1.024184 | 1.116627 |
| 2 | BA | 0.915438 | 1.024157 | 1.058980 |
| 3 | AB | 0.899217 | 1.024152 | 1.094404 |
| 4 | BA | 0.896156 | 1.024084 | 1.118020 |
| 5 | AB | 0.897575 | 1.023967 | 1.117519 |

Paired changes use each matched pair; they are not ratios of the separate role medians.

Every original gate remains unchanged, including separate AB/BA prefill medians
< 0.99, every pair faster, steady-decode median >= 0.97 and each order >= 0.95,
RSS median <= 1.02 and peak-footprint median <= 1.15. Gate results:

- all_exact_transcripts: PASS.
- all_five_prefill_faster: PASS.
- both_order_prefill_medians: PASS.
- steady_decode_median: PASS.
- steady_decode_both_orders: PASS.
- peak_rss_median: FAIL.
- peak_memory_footprint_median: PASS.

Full medians/MAD/ranges, order subsets and raw runs are in the engine audit
and manifest. Passive phase accounting is separately verified. These are
comparisons between Rust versions, not statistical significance or DwarfStar claims.

The perturbed profile records seven bootstrap spans: GPU-span total 6847.582ms, host elapsed 8480.624ms and command wall 7036.091ms. These include scheduling gaps and are not kernel sums or engine acceptance measurements.

Production executable SHA-256: `eb50db2862c3fed2bb4e0c6245ec1095a75bb4e38a2e8a38516833ce5333c3ea`.
Observed executable: `5bc5f84800b2aad2043b429367f86a3f6990752267efcf3b864fe4dbf40200f9`.
Final default executable: `2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4`.
All 42 final source hashes are verified; the previous default and both candidates
remain available in the ignored local study directory. The reused build target
contains the observed binary and is not the production default.

Run `python3 verify-report.py` from this report directory with Python 3.9+.
It checks stored/uncompressed hashes, source and observer inverses, full C0,
profile/ring/pool audits, reproduction of the original sentinel failure, the
corrected audit plus a rejected corrupted completion status, engine gates and
final runtime state. No model, GPU, Rust compiler or network is needed. Binaries
are identified by hashes rather than included; large raw artifacts are losslessly
gzipped. The manifest records original and stored identities.

The zero-initialization and completion contracts were verified against primary
Apple documentation: [buffer creation](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28length%3Aoptions%3A%29?language=objc), [byte copying](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytes%3Alength%3Aoptions%3A%29?language=objc), [checked completion](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29?language=objc).

Next milestone, **not started**: Diagnose the pooled scratch RSS cost at 32K, then establish proofs of complete writes before first reads for the large buffers that might safely avoid unconditional clearing. Treat clearing as a hypothesis, not a proved cause. Use a bounded synthetic residency/initialization experiment before any new model candidate, preserve exact C0 and the unchanged 2% RSS/15% footprint gates, and keep 256K paused.

No publication, remote workflow dispatch or 256K work occurred. Stop after the
current milestone update.
