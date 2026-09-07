# Native32K prefill tail batching

The prefill optimization milestone is complete for candidate `dd0d26c5424acd3e8b27606771c46ed3c71e715a` against unchanged oracle-v4 `df99eba7f722f8170dc2960b8b4666f73818a5ba`. Five complete pairs and both warmups pass; exact full-logit gates, retained reset/reuse, and independent audit pass.

Median Rust prefill falls from **813.107 s to 521.130 s**: **35.91% less latency (1.560× throughput)** relative to the previous audited baseline. The current reference median is 184.333 s. These are fixed 32K/128 development results on the M1 Ultra (48 GPU cores, 128 GB memory), not the project's 256K headline.

## Measured problem and change

A profile of the unoptimized scheduler reconciled 1,483 stage rows with aggregate GPU/command timers. It identified even-layer tails as the largest GPU cost: 294.625 s across 18,816 calls, versus 40.337 s across 140 odd-layer batched calls. Binding itself took only 3.875 s, so binding allocation was not the explanation for the aggregate setup cost.

Each even layer previously ran attention and its output/FFN tail in 128 separate 32-token tiles per chunk. The new schedule retains each tile's inverse-RoPE attention heads with a blit in the existing command, then executes the same tail kernels once on all 4,096 rows. This reduces even-tail calls to 147 across the full request. Guards require consecutive tiles and a complete chunk before the tail can execute. The shared 512 MiB attention buffer participates in the existing prefill ownership and reset cleanup.

Shared Metal kernels, arithmetic kernels, oracle source, model and immutable targets are unchanged. Batch grouping changes scheduling; exactness was established by full-output comparison rather than assumed from this design.

## Correctness

The candidate passed an early internal12K localization gate; two fresh full32K gates (output-only and internal) matching all 129 full-logit rows and GPU selections; exact 2K/8K controls; and both 32K/128 cycles in a retained Metal context. Established internal checkpoints through 24K pass; later internal hashes remain diagnostic. All 310 Rust tests and 91 Python tests pass.

Oracle-v4 acceptance and complete trace evidence remain pinned in [the baseline report](../32k-baseline-cbb078bb/README.md). The exact same oracle executable is used here. Model and prompt were freshly rehashed before freezing the new plan. Sampling remains included in both engines' intervals; the fixed full-logit trace establishes identical non-EOS selections despite their different general EOS policies.

## Paired results

One fresh-process warmup per engine precedes five pairs in AB/BA/AB/BA/AB order. Pairs 1–4 passed on first attempts. A host restart interrupted pair 5, leaving its candidate without a completed result. Uptime reset from over 16 hours to minutes and swap reset to zero on resumption; the restart trigger is unavailable. The original state and reference measurement are retained; the complete fifth pair was rerun under the protocol’s documented external-interruption rule. No result was excluded for speed. The campaign therefore spans an interruption and is not a continuously controlled host session. Values are median ± median absolute deviation (MAD), not confidence intervals.

| Metric | Optimized Rust | Oracle-v4 |
| --- | ---: | ---: |
| Steady decode | 19.752 ± 0.201 tokens/s | 19.595 ± 0.363 tokens/s |
| All 128 decode steps | 19.720 ± 0.192 tokens/s | 19.567 ± 0.361 tokens/s |
| First decode step | 61.030 ± 2.162 ms | 60.657 ± 0.718 ms |
| Prefill | 521.130 ± 21.751 s | 184.333 ± 5.872 s |
| Whole process | 549.959 ± 24.241 s | 211.460 ± 5.708 s |
| Peak process memory | 87.585 ± 1.675 GiB | 80.893 ± 0.000 GiB |

Median Rust peak process memory is 4.060 GiB below the previous campaign’s 91.645 GiB. This is a process measurement, not an attribution of all allocation savings to the new buffer. Decode remains near parity; the observed paired range does not establish a robust decode speedup.

Restart sensitivity: the first four pairs have median paired prefill-throughput ratio 0.34065; the complete post-restart pair has ratio 0.35696. All five valid pairs remain in the declared aggregate.

The median paired Rust/reference steady-decode ratio is 1.00368, range 0.96658–1.05343. The median paired prefill-throughput ratio is 0.35133, range 0.32646–0.36619. Rust prefill remains 2.827× slower by the ratio of latency medians.

The old and new Rust runs are separate benchmark campaigns, not direct old/new pairs. The reference-normalized prefill-throughput improvement is 1.521×, compared with 1.560× from Rust latency medians. The reference prefill median changed from 187.890 s to 184.333 s; retained host/load observations allow that comparison to be assessed. First-decode-step latency excludes prefill and is not time to first output.

## Separate profiles

Both profiles match the fixed transcript and are paired-ineligible. The before profile was collected before choosing the change; the after profile was collected after all timed pairs. Existing command timings and host timers add no prefill GPU waits. Bootstrap work is explicitly grouped by layer range, prefix work by layer, and continuation work by stage. These are stage observations rather than per-kernel attribution. Nested intervals and independently sampled medians must not be summed into an exact wall-time partition.

| Continuation stage | Before GPU seconds | After GPU seconds | Before / after calls |
| --- | ---: | ---: | ---: |
| even/attention | 73.932 | 77.645 | 18816 / 18816 |
| odd/ingress | 61.540 | 55.070 | 140 / 140 |
| even/tail | 294.625 | 43.453 | 18816 / 147 |
| odd/tail | 40.337 | 40.746 | 140 / 140 |
| odd/attention | 22.941 | 23.474 | 140 / 140 |
| even/boundary | 23.540 | 22.438 | 147 / 147 |
| odd/compressor | 0.797 | 0.745 | 140 / 140 |
| even/bind | 0.000 | 0.000 | 147 / 147 |
| even/host_scratch | 0.000 | 0.000 | 147 / 147 |
| odd/bind | 0.000 | 0.000 | 140 / 140 |

The largest remaining measured GPU stage is `even/attention` at 77.645 s in the after profile. The next milestone should investigate that stage's batching, allocation and command costs while preserving the same exact gates, then validate any change with this paired protocol. Broader native contexts and agent co-design remain future work.

## Evidence and audit

`verify.py` rehashes the portable evidence, checks all twelve accepted warmup/timed measurement bundles plus the interrupted pair’s completed reference bundle and recomputes the paired statistics. `independent-audit.json` also records verification of the frozen executable hashes, exact transcripts and byte-identical official summary regeneration. `profile32k-analysis.json` and `correctness/profile-before-analysis.json` reconcile profile coverage and interval sums. `correctness/` retains the full live gates and build identities.

There are 116 timed-block host observations and 0 unusual thermal observations. Desktop applications remained open; this is not a dedicated idle-host claim. Profiling remains excluded from throughput samples. The summary's `headline_eligible` flag means protocol eligibility, not completion of the 256K project target.

`portability.json` records the plan template and test-log path rendering; raw measurements retain original bytes. Run `python3 rust-star/reports/32k-prefill-dd0d26c5/verify.py` from the repository root.
