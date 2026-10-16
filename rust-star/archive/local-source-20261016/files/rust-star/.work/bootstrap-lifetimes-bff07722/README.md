# Bootstrap lifetime diagnostic: four 1K scratch sets overlap

**Milestone complete; no runtime optimization retained.** Both diagnostic
executables preserve full 32K/128 C0: 33,354,240 FP32 words match oracle-v5 exactly.
The 1K path has four distinct scratch sets alive before its final chunk wait.
Earlier sets disappear after completion, and the remaining set disappears after
the function's autorelease pool drains. This supports a narrow next experiment:
one unfinished 1K tile at a time, using the existing checked completion helper.

It does **not** explain the earlier 44.722% whole-engine footprint regression.
The 1K trace actually has a lower sampled peak of logical live scratch lengths
than the accepted path. No performance acceptance or DwarfStar claim follows
from these two instrumented processes. The prior 1K rejection is unchanged.

## Observed results

GB below means decimal 10^9 bytes. These are different accounting measures;
do not add them together or interpret logical lengths as physical RAM usage.

| Measure | Accepted 64-row path | Rejected1K path |
| --- | ---: | ---: |
| Native tiles across seven 4K chunks | 448 | 28 |
| Complete trace snapshots | 1,799 | 119 |
| Largest sampled logical live scratch total | 11.181 GB | 9.973 GB |
| One tile's logical scratch at 32K | 0.219 GB | 2.493 GB |
| Most unfinished commands observed | 51 | 4 |
| Most simultaneous instances of a scratch label | 52 | 4 |
| Scratch after every chunk's final pool drain | 0 | 0 |
| Largest sampled Metal currentAllocatedSize | 106.386 GB | 105.178 GB |
| Largest sampled bootstrap OS phys_footprint | 11.339 GB | 11.590 GB |
| Whole-process time-l peak memory footprint | 12.346 GB | 15.423 GB |

The 1K 32K snapshot contains four 2,493,173,233-byte scratch sets. Eight Q/attention
buffers account for 4,294,967,296 bytes across those four sets. The accepted
path's largest snapshot instead includes 51 instances of each full-history
F16 KV buffer: 1,711,276,032 bytes per layer. Widening reduces those repeated
history conversions/allocations while increasing the size of each row scratch
set. Logical allocation totals alone do not determine OS footprint.

At every final wait, all chunk commands report completed and tracked scratch
from earlier tiles is gone, although the command array still exists. Once the
function pool drains, tracked scratch is zero for both paths. Thus retaining
completed command objects in that array is insufficient to explain the earlier
scratch survival. A completion boundary is the most direct next experiment.

## What was measured

Frozen accepted Rust 1.74 source and the preserved rejected1K source received
identical, reversible observation patches. GPU kernels, dispatch arithmetic,
inputs and schedules were unchanged. The patch enumerates 121 local buffer names:
105 scratch names and 16 persistent/diagnostic-output aliases. Model wrappers and
temporary growth allocations are outside this ledger. Persistent KV/compressor
state is never treated as disposable scratch.

Weak references and cached logical lengths avoid owning buffers between
snapshots. Each helper has its own autorelease pool; an optimized synthetic
Metal-allocation test verifies release, persistent ownership and deduplication.
Snapshots occur at entry, registration before encoding, submission, final wait,
and after the function's pool drains. They record command status, logical live
lengths, Metal device allocation size and task_vm_info counters. There are no
added GPU waits, completion handlers or buffer aliases.

Enumeration, status reads and OS counters are not atomic and can perturb CPU
scheduling, allocation and GPU overlap. Object disappearance is bracketed by
snapshots, not timed precisely. Model views, allocator reservations, physical
page state and other runtime resources are outside logical scratch accounting.
Whole-process footprint may peak between snapshots; this trace does not locate
that peak or reproduce the prior uninstrumented engine study. No timing ratio
from these diagnostic runs is an optimization result.

## Validation and preserved correction

- Both optimized Rust 1.74 diagnostic builds and the observer ownership proof
  pass. Both fresh 32K/128 output-only runs match the immutable oracle JSON and
  all 128 full-logit decode comparisons: 16,677,120 FP32 words each.
- Full model, prompt, oracle executable/source and 7,669 fixture hashes pass
  before and after inference. Oracle-v5 remains
  ccbd16d36895ef9fe8edf7f793014988ff1ab7f4.
- The initial plan/auditor mistakenly expected 32-row production tiles. The
  pinned source explicitly selects 64 rows for collect_outputs=false. This was
  documented during the accepted run, before the wide run. The original
  frozen auditor then failed coverage as expected: 1,799 versus 3,591 snapshots.
  Its failure is preserved. audit-v2.py changes only that expected width and
  passes on the same raw data. No inference rerun, exclusion or C0 gate change.
- The portable verifier reconstructs both patches, recomputes the complete
  trace audit and accounting, reproduces the original audit failure, and rejects
  missing-snapshot, wrong-byte-total and command-error mutations in temporary
  copies. It requires Python 3.9+ and no model, GPU or Rust toolchain.
- All 42 accepted source hashes are restored. The established default executable
  remains 2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4.
  Diagnostic binaries remain isolated in the ignored study work directory.

Run from the repository:

```sh
python3 rust-star/reports/bootstrap-lifetimes-bff07722/verify-report.py
```

## Next milestone

[The concrete design](next-milestone-design.md) adds a checked completion wait
after each nonfinal eligible 1K tile. The existing final wait stays in place.
This bounds unfinished work without sharing mutable buffers or introducing
pooling assumptions. A separate trace must establish the bound, and a fresh
uninstrumented fixed engine study must pass all existing timing/decode/RSS gates
and the unchanged peak-footprint ratio <=1.15. Memory reduction is a hypothesis,
not an accepted result. Stop here before implementation; keep 256K paused.

## Evidence layout

manifest.json pins every archived artifact and its original bytes. Large raw
logs, traces and source snapshots are losslessly compressed. The original
plan, scripts, source transforms, weak-ownership proof, build logs, input pins,
full-C0 outputs, raw traces, host observations, failed initial audit, corrected
audit and restoration record are preserved. The verifier uses temporary copies
and never edits the archived evidence.
