# Bounded 1K bootstrap experiment

Decision: **reject the candidate and restore accepted source**. Failed predeclared gates: peak_memory_footprint_median. Correctness passes, but the candidate does not qualify for retention.

The candidate uses four 1K bootstrap tiles per native 4K continuation chunk at overall frontiers through 32K. It preserves the initial prefix, 32-row diagnostic and 64-row legacy callers, decode, larger-frontier schedules and all Metal shader arithmetic. Checked host/Rust guards require every tile to belong to the current chunk. The raw 128-row ring keeps each tile's final 128 rows.

The preceding 4K bootstrap rejection remains unchanged. This separate hypothesis aimed to retain dispatch savings with a smaller scratch working set. The candidate and baseline use Rust 1.74; neither role is DwarfStar.

## Correctness and provenance

All 320 existing Rust tests, optimized native build and formatting pass. The complete three-file inverse and Rust 1.74 formatting proof preserve the other 39 pinned sources. Actual unchanged Metal router kernels match all 7,840 output words and 96 guards at 32/64/1024 rows. Independent raw-ring indexing checks pass 1,567 cases. Source, binary, model, oracle and 7,669 fixture identities are verified before and after the model study.

Baseline source is d72b7528 (documentation-only successor to e73a4b79). The input model and oracle-v5 source remain immutable. Frozen plans, commands, code, outputs, hashes and failures are included; private paths are redacted with original/stored hashes. Executables, model and large fixture contents are omitted.

The 12K internal prefix/chunk checks pass completely. Normal and profiled 32K/128 runs each check 16,677,120 FP32 logit values exactly: 33,354,240 total. The profiled run passes the unchanged 4,466-interval stage/component audit. These establish C0 at 32K, not higher frontiers.

## Diagnostic selection

Recorded bootstrap GPU span is 32.940s baseline and 6.074s candidate. The raw ratio is 0.184402; the control-normalized ratio is 0.195533. Both must be below 0.90. Diagnostic selection passes.

Instrumentation perturbs execution. Bootstrap timing spans the first through last queued command and can include gaps; it is not a sum of individual kernel durations. Diagnostic savings select an engine experiment but cannot establish whole-engine improvement.

## Complete engine decision

Two declared warmups and all five alternating unprofiled pairs are preserved. There are no additional runs or exclusions. Every process reproduces the exact greedy oracle transcript; full-logit correctness was checked separately above.

| Pair | Order | Baseline prefill s | Candidate prefill s | Latency ratio | Steady decode throughput ratio | RSS ratio | Footprint ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | AB | 226.109 | 200.086 | 0.884908 | 0.996671 | 0.999413 | 1.446976 |
| 2 | BA | 225.899 | 200.808 | 0.888928 | 0.996826 | 0.999408 | 1.447216 |
| 3 | AB | 226.250 | 200.843 | 0.887705 | 1.005275 | 0.999409 | 1.436919 |
| 4 | BA | 225.987 | 200.376 | 0.886671 | 0.993052 | 0.999385 | 1.467897 |
| 5 | AB | 226.298 | 200.875 | 0.887659 | 1.010157 | 0.999413 | 1.477703 |

| Gate | Result |
|---|---|
| all_exact_transcripts | PASS |
| all_five_prefill_faster | PASS |
| both_order_prefill_medians | PASS |
| steady_decode_median | PASS |
| steady_decode_both_orders | PASS |
| peak_rss_median | PASS |
| peak_memory_footprint_median | FAIL |

Median paired prefill latency changes -11.234%; steady decode throughput changes -0.317%; peak RSS changes -0.059%; separate peak memory footprint changes +44.722%.

Criteria were fixed before runtime edits: all five pairs prefill-faster; each AB/BA prefill median ratio below 0.99; median steady decode throughput ratio at least 0.97 and each order at least 0.95; median RSS ratio at most 1.02; median separate footprint ratio at most 1.15. The new footprint limit bounds this working-set hypothesis and is not changed after measurement. All medians, MADs, ranges, order subsets and raw values remain in engine/audit.json.

Passive phase timings already emitted by every run are summarized in passive-phase-audit.json. They add no instrumentation or exclusions. Command-wall/GPU differences do not prove a specific allocation, clock or stall cause.

## Final state and verification

All 42 accepted runtime source hashes are restored. The established default performance executable remains unchanged; experimental binaries remain preserved separately. Final state and decision are recorded in final-runtime-state.json and decision.json.

Run `python3 rust-star/reports/bootstrap-1k-d72b7528/verify-report.py`. It requires Python and the installed Rust 1.74 rustfmt for the exact source-formatting proof; it does not use the GPU/model. Keep the adjacent msrv-c0-5834fe56 and wide-bootstrap-c7adfb22 reports for immutable baseline/prior-source provenance. The verifier checks every artifact, complete source inverse, build/router/ring evidence, model/fixture pins, C0 outputs, profile coverage and all applicable diagnostic/engine gates.

This closes one milestone. No publication, DwarfStar parity claim, automatic 256K work or next-milestone execution occurred.
