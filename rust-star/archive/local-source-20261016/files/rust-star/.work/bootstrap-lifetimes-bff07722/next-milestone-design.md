# Proposed next milestone: bound native 1K bootstrap to one unfinished tile

This is a design, not an implemented or accepted optimization. The rejected
1K result stays rejected. Finish the lifetime report and stop before this work.

## Evidence and hypothesis

The trace observes four distinct 1K scratch sets alive before the final chunk
wait. Earlier sets disappear after command completion; the last set disappears
when its function autorelease pool drains. Completed command objects can remain
in the array without retaining those earlier scratch buffers. A separate cache
or an assumption about command-array ownership is unnecessary for the first
experiment.

Add a checked completion wait after each nonfinal eligible 1K tile commits.
The final tile already has a checked completion wait. This should ensure that
the previous tile has completed and its local pool has drained before the next
boundary call allocates or initializes scratch. It bounds unfinished native
bootstrap work to one tile. It may give the allocator an opportunity to reuse
released storage; actual reuse and physical footprint reduction are unproved.

## Concrete scope

- Begin from the exact preserved rejected 1K source, on the accepted Rust 1.74
  baseline. Preserve all four router caps, raw-ring indexing fixes, native1K
  eligibility and all existing kernel arithmetic.
- Change only scheduling in the native eligible 1K branch of
  rust_star_metal_run_prefill_layer0_boundary. For a nonfinal tile, use the
  existing command_succeeded helper after commit and before returning. Preserve
  the existing final wait and per-command error checks. No unretained command
  buffers, callbacks, separate queues, pointer aliasing or explicit pooling.
- Keep 32-row collected diagnostics, 64-row legacy/accepted callers, prefix,
  decode, persistent KV/compressor state and frontiers above 32K unchanged.
- Retain the existing first/last command timing calculation. Its GPU span may
  include host-induced gaps; it cannot be redefined as summed kernel time.
- On any command failure, stop and preserve the run. Do not advance to another
  tile or convert a failed command into success.

This avoids sharing mutable CPU-written token, position, mask or sentinel
storage with unfinished GPU work. It introduces no new buffer aliases or
read-before-write assumptions. Explicit scratch pooling remains a later option
only if this narrow policy is insufficient. Pooling would additionally require
per-buffer full-write-before-read proofs and lifecycle cleanup; the lexical
inventory alone does not establish those proofs.

## Prospective acceptance to freeze before implementation/inference

Use a new work directory and plan. Freeze source inverse, same-compiler baseline
and candidate binaries, model/oracle/fixture inputs and scripts. Preserve every
attempt. One GPU workload at a time and no timed-run builds/model hashing.

1. Source/build and existing relevant host tests; exact protected-path and raw
   ring/router scope checks. No shader edits.
2. Fresh 12K internal and normal/profiled 32K/128 full C0 against oracle-v5.
   Greedy tokens alone are insufficient. Preserve strict profile coverage.
3. A separate weak-reference diagnostic should show no preceding-tile scratch
   at entry/registration, at most one unfinished native bootstrap command, and
   zero tracked scratch after each tile's pool drains. Instrumentation is
   excluded from engine performance runs.
4. Uninstrumented whole-engine study: two warmups followed by fixed
   AB/BA/AB/BA/AB against the accepted current-compiler baseline. Preserve the
   established prefill/order/decode/RSS gates and separate paired median peak
   memory footprint ratio<=1.15. Copy exact gate definitions from the prior
   immutable engine plan, rather than paraphrasing them into new thresholds.
5. Require all gates. Restore accepted source on failure; preserve every raw
   result. If all pass, retain and document the change, then report the next
   milestone. Do not claim DwarfStar parity or resume256K.

The diagnostic's lower live-byte expectation is a mechanism check. It does not
replace the whole-engine footprint gate or imply that the earlier 44.722%
footprint regression has been explained.
