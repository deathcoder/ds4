# Rust Star 32K/128 development baseline

This report compares the frozen Rust candidate `cbb078bb79871169f6a1dd367f96c141969e917e` with repaired DwarfStar `oracle-v4`, source `df99eba7f722f8170dc2960b8b4666f73818a5ba`, on the target M1 Ultra (48 GPU cores, 128 GB unified memory). It covers the fixed 32,768-token prompt and 128 committed greedy decode steps. It does not establish the project's 256K primary headline or general task quality.

## Measurement protocol

One fresh-process warmup per engine precedes five fresh-process pairs in AB/BA/AB/BA/AB order. Each process loads the same pinned model and prompt. Sampling is included in both decode timers and initial selection is included in prefill. The fixed full-logit trace proves the same non-EOS selections despite the reference explicitly excluding EOS and Rust using global lowest-ID argmax. That equivalence is scoped to this workload.

The table will report medians, median absolute deviations, and pairwise ratios. The candidate profile is collected after the paired block and is ineligible as a throughput sample. Other desktop applications remain open; retained host observations describe the conditions rather than claiming dedicated idle hardware.

## Correctness and repairs

Before measuring, the candidate passed four fresh complete 32K/128 full-logit traces and both 2K/8K regression controls. Two of the 32K runs also checked established internal checkpoints through 24K. Later internal hashes remain diagnostic. The production GPU argmax was compared with the CPU argmax of the same completed full-logit row throughout those gates.

Oracle-v4 passed official Metal kernel and logprob tests, four fresh captures at each of 2K and 32K, and two additional complete 129-row benchmark-executable traces. Every accepted output word matches the existing immutable targets. Historical fixture producer identities remain unchanged.

Preparation exposed two defects, and all rejected attempts remain evidence:

1. The reference benchmark excluded greedy selection from per-step timing. The driver now includes it; a compiled-driver mock-clock regression checks the actual timing loops.
2. A streaming top-k kernel let faster SIMD groups change a shared count before slower groups sampled it. A standalone test reproduced incorrect indices under uneven scheduling. An unconditional barrier after all threads snapshot the count fixes this race. Independent CPU-sort GPU regressions cover four shapes and normal/uneven scheduling; an extended adversarial run checks 32,768,000 selected indices.

An intermediate projection-wait hypothesis passed two full runs but failed an internal checkpoint in a third. That experiment was reverted. Neither the legacy-timer plan nor the subsequent failed C0 plan contributes samples here.

## Evidence

`results/` retains measurements and raw paired data. `independent-audit.json` independently rehashes artifacts and recomputes statistics. `correctness/` retains acceptance and full-run evidence. `rejected/` preserves negative results. `inputs/plan-template.json` is a portable rendering; its bytes differ from the private frozen plan, whose original hash remains recorded. `artifact-hashes.json` binds the exported files.

## Results and next milestone

Populate only after all five pairs, the separate profile, and independent audit pass. Select the next optimization from measured costs. Decode-stage instrumentation cannot establish a specific prefill-kernel bottleneck; prefill attribution in this profile is aggregate.
