# Conditional 1K bootstrap experiment design

This document is planning only. No runtime edits or execution until this full-C0
milestone passes, its verified report is committed, and a new plan is frozen.

The rejected 4K experiment reduced bootstrap work but increased separate macOS
peak memory footprint by 52.484% (~6 GiB). It failed whole-engine selection.
Four 1K tiles per native 4K chunk are a distinct hypothesis: amortize dispatches
while limiting row-dependent scratch to one quarter of that rejected batch.
This does not prove the prior regression's cause or guarantee a saving.

Eligibility: refreshed current Rust 1.74 profile must attribute at least 10% of
recorded GPU stage time to bootstrap ingress. Compare same-compiler executables.
Keep initial prefix, diagnostic 32-row callers, legacy 64-row paths, decode and
overall frontiers above 32K unchanged. Only the native continuation caller opts
into 1024 rows. The raw 128-row ring retains the final 128 rows of each tile.
Positions and compressed positions have capacities 1024 and 256 respectively.
Cap actual router row groups at MIN(rows,256), as proved in the rejected study.
No Metal shader arithmetic edits or changes to the Rust 1.74 compatibility fix.

The old single-tile guards cannot simply be copied: all four tiles share a
chunk endpoint. Require a nonzero active 4K chunk, endpoint <=32768, tile start
within that chunk, offset divisible by1024 and tile end <= the chunk endpoint.
The host also checks the context chunk start equals endpoint minus4096; preserve
existing transient-buffer bounds. Retain checked integer arithmetic in Rust.

Proposed prospective gates for the separate plan:
- Actual unchanged Metal router kernels: CPU equivalence and guards at32/64/1024.
- Complete three-file inverse and scope proof; all other source hashes unchanged.
- Rust1.74 formatting, existing320 host tests and forced native optimized build.
- 12K internal prefix/chunk checks; normal and profiled32K/128 full FP32 C0.
- Diagnostic bootstrap aggregate and control-normalized ratios both below0.90;
  unchanged strict profile packet/counter audit. This only selects an engine test.
- Two fixed warmups followed by five complete AB/BA/AB/BA/AB unprofiled pairs.
  Every pair prefill-faster; separate AB/BA median ratios below0.99; steady decode
  median throughput ratio >=0.97 and each order >=0.95; median RSS ratio <=1.02.
- New working-set gate: median separate peak-memory-footprint ratio <=1.15.
  Quarter of the rejected ~6GiB increment is ~13% of its11.5GiB baseline;
  15% is a prospectively bounded allowance, not a promise of exact scaling.
  Keep both RSS and footprint, all passive phase costs and every sample.

Stop on failure; preserve all runs and close the hypothesis without exclusions,
reruns or threshold adjustments. Restore accepted source if rejected. No256K,
publication or DwarfStar performance claim follows from this experiment.
