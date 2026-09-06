# Rust Star 32K/128 development baseline

This report compares the frozen Rust candidate `cbb078bb79871169f6a1dd367f96c141969e917e` with repaired DwarfStar `oracle-v4`, source `df99eba7f722f8170dc2960b8b4666f73818a5ba`, on the target M1 Ultra (48 GPU cores, 128 GB unified memory). It covers the fixed 32,768-token prompt and 128 committed greedy decode steps. It does not establish the project's 256K primary headline or general task quality.

## Measurement protocol

One fresh-process warmup per engine precedes five fresh-process pairs in AB/BA/AB/BA/AB order. Each process loads the same pinned model and prompt. Sampling is included in both decode timers and initial selection is included in prefill. The fixed full-logit trace proves the same non-EOS selections despite the reference explicitly excluding EOS and Rust using global lowest-ID argmax. That equivalence is scoped to this workload.

The table reports medians, median absolute deviations, and pairwise ratios. The candidate profile is collected after the paired block and is ineligible as a throughput sample. Other desktop applications remain open; retained host observations describe the conditions rather than claiming dedicated idle hardware.

## Correctness and repairs

Before measuring, the candidate passed four fresh complete 32K/128 full-logit traces and both 2K/8K regression controls. Two of the 32K runs also checked established internal checkpoints through 24K. Later internal hashes remain diagnostic. The production GPU argmax was compared with the CPU argmax of the same completed full-logit row throughout those gates.

Oracle-v4 passed official Metal kernel and logprob tests, four fresh captures at each of 2K and 32K, and two additional complete 129-row benchmark-executable traces. Every accepted output word matches the existing immutable targets. Historical fixture producer identities remain unchanged.

Preparation exposed two defects, and all rejected attempts remain evidence:

1. The reference benchmark excluded greedy selection from per-step timing. The driver now includes it; a compiled-driver mock-clock regression checks the actual timing loops.
2. A streaming top-k kernel let faster SIMD groups change a shared count before slower groups sampled it. A standalone test reproduced incorrect indices under uneven scheduling. An unconditional barrier after all threads snapshot the count fixes this race. Independent CPU-sort GPU regressions cover four shapes and normal/uneven scheduling; an extended adversarial run checks 32,768,000 selected indices.

An intermediate projection-wait hypothesis passed two full runs but failed an internal checkpoint in a third. That experiment was reverted. Neither the legacy-timer plan nor the subsequent failed C0 plan contributes samples here.

## Evidence

`results/` retains measurements and raw paired data. `independent-audit.json` independently rehashes artifacts and recomputes statistics. `correctness/` retains acceptance and full-run evidence. `rejected/` preserves negative results. `inputs/plan-template.json` is a portable rendering; its bytes differ from the private frozen plan, whose original hash remains recorded. `artifact-hashes.json` binds the exported files. `portability.json` records the two derived files: the plan template and a test log whose repository prefix was replaced with `$REPO`. All raw measurements and oracle artifacts retain their original bytes. Run `python3 rust-star/reports/32k-baseline-cbb078bb/verify.py` from the repository root to verify the portable bundle.

## Results

Five valid first-attempt pairs; no invalid attempts. Values below are median ± median absolute deviation (MAD), not confidence intervals.

| Metric | Rust Star | Oracle-v4 |
| --- | ---: | ---: |
| Steady decode | 19.574 ± 0.144 tokens/s | 19.454 ± 0.076 tokens/s |
| All 128 decode steps | 19.538 ± 0.143 tokens/s | 19.431 ± 0.066 tokens/s |
| First decode step | 63.094 ± 0.766 ms | 55.649 ± 1.828 ms |
| Prefill | 813.107 ± 9.854 s | 187.890 ± 0.609 s |
| Whole process | 842.116 ± 9.777 s | 209.095 ± 0.689 s |
| Peak process memory | 91.645 ± 0.337 GiB | 80.893 ± 0.000 GiB |

The median paired steady-decode ratio is **1.00616×** Rust/reference, with MAD 0.00352 and range 0.98766–1.02341. The approximately 0.62% median difference is small and the pairwise range crosses parity. These five runs do not establish a robust decode speed advantage.

Prefill is **4.33× slower** by the ratio of latency medians. Rust also uses about 10.75 GiB more peak process memory. The first-decode-step metric excludes prefill; it is not end-to-end time to first output. Process wall time includes initialization and other overhead outside the internal timers.

## Separate profile

The profile matches the fixed transcript and is excluded from paired throughput. Its eight decode stage shares use Metal timestamp counters scaled to completed layer GPU intervals. Encoder splitting perturbs execution; these figures guide investigation rather than measuring uninstrumented throughput.

| Decode stage | GPU ms / steady token | Share of transformer GPU interval |
| --- | ---: | ---: |
| routed_experts | 11.778 | 21.79% |
| attention_cache_and_routing | 10.077 | 18.64% |
| attention_ingress_qkv | 8.760 | 16.21% |
| attention_output_hc | 6.841 | 12.66% |
| rope_kv_compressors | 4.794 | 8.87% |
| attention | 4.584 | 8.48% |
| ffn_ingress_router | 3.852 | 7.13% |
| shared_expert_ffn_hc | 3.361 | 6.22% |

At 32K this mode does not collect per-layer or per-kernel prefill attribution. The uninstrumented paired candidate runs do expose these aggregate medians:

| Prefill interval | Median seconds |
| --- | ---: |
| Transformer setup | 94.357 |
| Transformer execution wall | 675.449 |
| Transformer GPU | 568.313 |
| Tile GPU | 38.464 |
| Total reported host overhead | 98.617 |

These intervals overlap or nest and their independent medians must not be summed into an exact partition. GPU execution is the largest measured component; setup is also material. The measurements do not identify a particular slow prefill kernel.

## Next milestone

Close the 32K prefill gap: first add explicitly ineligible per-chunk/per-stage profiling across all 43 layers, accounting separately for setup, encoding/waits, and GPU work. Use it to select and implement the largest measured C0-safe improvement, preserving full-logit checks and shorter controls, then repeat this five-pair protocol. Reusing prefill resources may reduce the measured setup cost, but its payoff is a hypothesis until measured. Broader native contexts and the 256K primary result remain future work.

## Audit and limits

The independent audit rehashes all twelve warmup/timed measurement bundles, checks both frozen executable hashes, verifies candidate transcripts, recomputes all summary distributions and paired ratios, regenerates the official summary byte-for-byte, and checks profile scope and stage shares. It records 140 host observations and 0 unusual thermal observations. Background applications and load observations remain in the evidence. Five pairs describe this fixed workload on this host; they do not establish general model quality or broad speed superiority.

The summary's `headline_eligible` flag means schema/protocol eligibility only. This is a 32K development baseline, not the charter's 256K headline.
