# Attention-output half inputs: 32K whole-engine decision

Decision: **retain the integrated candidate** under the prospective rules. This compares Rust `4ced6ee8` with frozen Rust `9d35953f` on the same M1 Ultra. It does not measure a DwarfStar advantage.

Median paired prefill latency changes by -4.329%; role medians are 262.632s baseline and 251.174s candidate. Steady decode changes by -0.190% on paired medians; peak RSS changes by +0.001%. First-step latency changes by -1.966%.

## Completed milestones

1. Same-arena A/A control: two fresh processes, 72 pairs, 18,320,719,872 exact FP32 comparisons. Every case passes the prospectively declared temporal, AB/BA, active-arena, MAD and dispersion criteria.
2. Candidate plus contemporary A/A: two fresh processes, 144 total pairs and 35,232,153,600 exact FP32 comparisons. All 72 candidate pairs improve. Separate AB/BA medians improve 33.919–34.902%, including both added conversions; all current A/A gates pass.
3. Integration and full model: 320 release host tests, formatting, release build, exact source restoration and production Metal compilation pass. Full 32K prefill plus 128 decode steps match all 16,677,120 FP32 words. The correctness process takes 292.181s; that elapsed time is not a performance result.
4. Whole-engine decision: two preserved warm-ups and five complete alternating fresh-process pairs, using the existing unprofiled `engine-measure` command and strict measurement adapter. All 12 normal-engine transcripts match the pinned oracle, and every raw interval/metadata contract passes.

## Whole-engine pairs

| Pair | Order | Baseline prefill (s) | Candidate prefill (s) | Latency change | Steady decode change | RSS change |
|---|---|---:|---:|---:|---:|---:|
| 1 | AB | 262.632 | 251.745 | -4.145% | -1.247% | +0.001% |
| 2 | BA | 263.650 | 250.759 | -4.889% | -2.824% | +0.001% |
| 3 | AB | 262.209 | 251.531 | -4.072% | +0.685% | -0.002% |
| 4 | BA | 262.646 | 251.044 | -4.417% | -0.190% | -0.001% |
| 5 | AB | 262.538 | 251.174 | -4.329% | -0.175% | +0.003% |

A is frozen Rust; B is the integrated candidate. Warm-ups are retained in the raw data but excluded from the table and statistics. No timed pair was discarded or repeated. All ratios are computed within pairs; the ratio of independent role medians is not substituted.

| Metric | Baseline median | Candidate median | Paired ratio median | Ratio MAD | Ratio min–max |
|---|---:|---:|---:|---:|---:|
| Prefill seconds | 262.632 | 251.174 | 0.956713 | 0.001833 | 0.951109–0.959276 |
| Steady decode tokens/s | 18.498 | 18.385 | 0.998104 | 0.008742 | 0.971759–1.006846 |
| First step milliseconds | 67.195 | 63.976 | 0.980336 | 0.012623 | 0.952096–1.021470 |
| Process seconds | 292.113 | 280.996 | 0.961942 | 0.003208 | 0.955648–0.965318 |
| Peak RSS GiB | 83.850 | 83.850 | 1.000005 | 0.000012 | 0.999978–1.000031 |

The full audit includes each role’s MAD/min/max, all observations and separate order subsets. Five pairs support this bounded local decision; no significance claim is made.

## Prospectively bound retention criteria

| Gate | Result |
|---|---|
| all exact transcripts | PASS |
| all five prefill faster | PASS |
| both order prefill medians | PASS |
| steady decode median | PASS |
| steady decode both orders | PASS |
| peak rss median | PASS |

AB prefill ratio median: 0.958546; BA: 0.953468. The preset screen requires all five prefill pairs faster, both order medians below 0.99, steady decode median ratio at least 0.97 and each order at least 0.95, and peak-RSS median ratio at most 1.02. Thresholds and run count were written before whole-model timing.

## Change and scope

The original two Q8 projections repeatedly round their FP32 right-hand inputs to FP16 inside tiled loads. The candidate performs each conversion once, then uses the existing grouped half-input kernel and a half-input instantiation of the unchanged dense template. Accumulators, arithmetic order, group mapping and HC expansion remain intact. Both conversion dispatches are included in isolated chain timing and normal engine execution.

Two reusable transient scratch buffers add 320 MiB at 4096 rows and are released under existing prefill ownership. Selection is limited to native continuation chunks with nonzero chunk start. Prefix, legacy tiled prefill, decode, imported kernels and the rest of the host restore byte-for-byte to the baseline after the recorded edits are inverted.

New full-model correctness and timing evidence is limited to 32K/128 on this model and machine. Larger frontiers have not been revalidated for this change. The 256K campaign remains paused. No push or release is performed.

The previous two-arena screen remains a failed control study. This report does not normalize or relax it retroactively. The fresh same-arena design uses a protected baseline reference, poisons only the active arena before each role, and compares every final low/dense/HC output word after each role before overwrite. Both physical arenas and both role orders are balanced. Four conditioning command buffers precede three measured samples per role. Intermediate overwritten samples are not individually retained.

The earlier 16 edge cases and CPU rounding-boundary checks are reused unchanged from the [prior source-identical study](../attention-output-half-input-038dded4/README.md); no new edge campaign is claimed. The current fresh processes also verify complete CPU conversions during initial checks, input hashes, guards and maps. The observed prior buffer-identity effect has no established cache/clock/page-allocation cause.

## Evidence and verification

Run `python3 verify.py` from any copy of this report. It verifies stored and original artifact hashes, exact frozen sources, both shader assemblies, complete-host restoration, immutable-reference checksums, every isolated interval and decision gate, full C0 receipts, all 12 normal-engine records and complete paired statistics. The portable engine auditor changes only the adapter import path and manifest-verified original-hash lookup; its source transformation is checked explicitly.

All 42 candidate source inputs are included, with the two modified baseline files for complete restoration. Executables and model data are omitted; their SHA-256 identities, build and full-input verification receipts are retained. Local paths are redacted in portable text, with original/stored hashes recorded in `artifacts.json`. Raw work evidence remains unchanged.

Key files: [umbrella plan](plan.json), [whole-engine plan](engine-plan.json), [engine audit](engine/audit.json), [source/build freeze](integration/candidate-build.json), [source patch](integration/implementation.patch), [full C0 receipt](integration/validation-acceptance.json), [final pins](completion-pins.json).

Host observations: 133 samples span the 12 runs. Recorded swap use ranges from 33.25 to 33.25 MiB. macOS reports no thermal or performance warning in these samples. These observations are context, not a clock-frequency or GPU-isolation guarantee.
