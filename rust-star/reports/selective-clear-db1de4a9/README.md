# Retained: bounded scratch reuse with selective initialization

The milestone is closed. **Every unchanged engine and memory gate passes.** The missing guarded mutation
proof is now covered by new evidence, and the previous incomplete diagnostic
remains unchanged. See `decision.json` for the authoritative outcome.

All fresh source/GPU/build, full-C0 and ownership prerequisites pass before
the fixed engine block. Twelve fresh processes complete: two excluded warmups
and five paired orders AB, BA, AB, BA, AB. Both roles use Rust 1.74 on the M1 Ultra.
These measurements compare the candidate with accepted Rust, not DwarfStar.

## Engine results

| Metric | Baseline median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Prefill | 241.289 s | 214.658 s | -11.054% |
| Steady decode | 19.665 tokens/s | 19.967 tokens/s | +1.362% |
| First decode step | 62.948 ms | 64.254 ms | +0.124% |
| Peak RSS | 83.851 GiB | 85.377 GiB | +1.819% |
| Peak footprint | 11.551 GiB | 11.965 GiB | +3.728% |

Every timed pair, candidate/baseline ratios:

| Pair | Order | Prefill | Steady decode | RSS | Footprint |
|---|---|---:|---:|---:|---:|
| 1 | AB | 0.852223 | 1.001023 | 1.033871 | 1.073915 |
| 2 | BA | 0.883958 | 1.021528 | 1.018004 | 1.037283 |
| 3 | AB | 0.889464 | 1.013618 | 1.018210 | 1.032735 |
| 4 | BA | 0.896635 | 1.016212 | 1.018192 | 1.003139 |
| 5 | AB | 0.896143 | 1.003925 | 1.018189 | 1.071896 |

Frozen gate verdicts:

- `all_exact_transcripts`: PASS.
- `all_five_prefill_faster`: PASS.
- `both_order_prefill_medians`: PASS.
- `steady_decode_median`: PASS.
- `steady_decode_both_orders`: PASS.
- `peak_rss_median`: PASS.
- `peak_memory_footprint_median`: PASS.

RSS and footprint are separate gates: median ratios <=1.02 and <=1.15.
All five pairs must improve prefill, with both order medians <0.99. Steady decode
requires median >=0.97 and both order medians >=0.95. No threshold, exclusion
or repetition changed after results. Full distributions, MAD/min/max, order
subsets, warmup observations and passive phase timing are retained in the
engine audit and `passive-phase-audit.json`. Passive phase differences alone do
not prove a particular allocation or synchronization cause. No statistical
significance claim is made.

## Implementation and verification

The candidate reinstates one bounded 105-buffer scratch set per four native
1K bootstrap tiles. Relative to the previous rejected pool, it changes only
initialization of q_cur_buffer, attention_back_buffer, next_q_cur_buffer and
next_attention_back_buffer: omit CPU memset for NULL source and exact length
134,217,728, after checked prior GPU completion. Every other initializer and
all source memcpy operations remain unchanged. Nonpool constructors retain
existing behavior. No shader arithmetic or persistent-state policy changes.
Full source snapshots, a patch against accepted source, and the two-step
comment/predicate inverse against the prior pool make the scope inspectable.

Each selected destination is fully overwritten by an ordered blit before its
first content read. Source and destination are distinct, full ranges match,
and producers/copies/consumers preserve encoder order. Read `scope-proof.md`
and `source-proof.json` for the reference inventory and proof boundaries.

- New guarded copy-then-in-place-XOR proof: 24 cases, 2,147,549,248 payload bytes,
  24,576 guard bytes, 12 detected shortened copies and 12 detected omitted
  mutations. Completed before runtime edits. The old diagnostic verdict is
  preserved as incomplete; this new proof closes the gap for this milestone.
- Exact production helper: 1,073,741,824 selected-policy bytes; unchanged
  small-size suite checks 62,043,240 bytes, 192,360 guards, 2,205 identity reuses,
  84 invalid checkouts and final/error cleanup. Explicit source-copy inputs
  override clear omission; selected labels at other lengths keep clearing.
- 320 Rust tests, optimized production and observer builds, formatting,
  1,567 ring-index cases, router 7,840-word/96-guard checks and weak-owner proof.
- Fresh 12K internal and normal/profiled/observed 32K/128 full C0 all pass:
  50,031,360 exact full-logit FP32 comparisons; all 4,466 profile intervals pass.
- Real pool audit: 28 tiles, 119 snapshots, 2,940 checkouts across seven chunks;
  112 selected clear omissions, exactly 536,870,912 bytes per tile. Stable
  distinct identities, exact capacities, checked prior status, owner cleanup
  and at most one unfinished command all pass. Weak liveness is recorded
  separately from ownership and OS memory residency.

One source-preparation assertion was corrected before runtime edits/builds:
the observer anchor occurs in several functions, so the generator needed its
established boundary-function fallback. Original preparation source and failure
record are preserved. No inference or measurement retry resulted from this fix.
The prior unsigned nil-command representation is expected prospectively in this
milestone; there is no post-run sentinel amendment here.

Complete model/oracle/prompt/7,669-fixture hashes and all source/executable pins
are checked before and after execution. This validates this run's inputs and
captured evidence; it does not establish fresh larger-frontier correctness.

## Final state and reproduction

Candidate retained: `true`. Final default executable SHA-256:
`7590af7ade11d5b07d65f0fefc9e01c02668318896fc4e955307888d85cdaa35`.
The reused Cargo build directory contains the observer executable and is not the
default. `final-runtime-state.json` records the final 42-source state. Frozen
executable sizes and SHA-256 identities are packaged instead of executable files.
Publication, remote workflows and the paused 256K campaign remain deferred.

Run `python3 verify-report.py` in this report directory using Python 3.9+.
It verifies stored/decompressed hashes, frozen build and proof identities,
source/observer inverses, full-output records, source/ring/profile/pool/engine
re-audits and the final decision. It rejects altered prior completion and clear
omission records. No GPU, model, compiler or original workspace is required.
This re-audits captured evidence; it does not rerun hardware measurements.
`manifest.json` maps compressed files to original names, lengths and hashes.

## Next milestone — not started

Refresh the direct 32K comparison between the newly accepted Rust runtime and pinned DwarfStar on the same model, prompt and 128-token decode workload. Use fresh paired processes and the existing strict measurement adapters to quantify the remaining performance gap, then choose the next bottleneck from the current profile. Keep larger-frontier work and publication deferred. This proposal has not started.
