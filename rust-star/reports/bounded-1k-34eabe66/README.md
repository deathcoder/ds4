# Bounded 1K bootstrap: rejected at the scratch-lifetime gate

**Milestone complete; accepted runtime restored.** The candidate passes the
12K internal check and three fresh 32K/128 full-C0 runs: **50,031,360 FP32 words
match exactly**. Its trace never observes more than one unfinished bootstrap
command. It nevertheless fails the predeclared scratch-lifetime prerequisite:
some completed-tile buffers survive into the next tile's entry and registration.
**Engine timing did not start.** No acceptance threshold was changed and no
inference run was repeated.

## Exact failure

The complete trace contains 28 native 1K tiles and 119 snapshots across seven
4K chunks. At the 20K frontier, the first tile starts at position 16384:

| Snapshot | Tile position | Phase | Tracked scratch bytes | Previous-tile scratch bytes | Command status |
| --- | ---: | --- | ---: | ---: | --- |
| 54 | 16384 | after_pool | 1,977,044,465 | 0 | One completed |
| 55 | 17408 | entry | 1,420,731,848 | 1,420,731,848 | One completed |
| 56 | 17408 | registered | 2,448,059,889 | 5,242,880 | One completed |
| 57 | 17408 | submitted | 2,448,059,889 | 5,242,880 | One completed, one unfinished |
| 58 | 17408 | after_pool | 0 | 0 | Two completed |

At registration/submission, the remaining older storage comprises one prior
instance each of layer2_indexer_packed_score_buffer (1 MiB),
layer2_kv_norm_snapshot_buffer (2 MiB), and layer2_kv_rope_buffer (2 MiB).
Those labels each have two live instances in these snapshots. The older
5,242,880 bytes also make the registered total exceed the declared one-set
length for that frontier. The frozen auditor stops at its first post-pool
zero-scratch assertion. The supplemental failure analysis accounts for all four
violating snapshots without replacing or weakening that auditor.

The peak sampled logical scratch total is **2,493,173,233 bytes**, one quarter
of the prior four-set trace's 9,972,692,932 bytes. That lower global peak does
not erase the failed per-tile ownership conditions. Completed command status
and a drained local pool did not coincide with disappearance of all observed
buffer objects in this run. The snapshots do not identify the remaining owner,
exact release times, allocator behavior or a physical-memory leak. Their weak
reference enumeration and command-status reads are not atomic.

## Candidate and scope

The candidate starts from the exact preserved rejected 1K source. Its two Rust
files are byte-identical to that candidate. The only new source insertion is a
checked command_succeeded wait after each nonfinal eligible native 1K command is
committed and added to the batch. The final tile retains its existing wait and
command error audit. Removing the insertion recovers the previous shim exactly.

No shader arithmetic, buffer alias, explicit pool, callback, queue, oracle or
fixture changes. All other 39 source hashes match the accepted runtime. The
32-row collected diagnostics, 64-row callers, initial prefix, decode and
higher-frontier schedules retain their prior paths. The source argument
establishes command completion before another tile is invoked; the observed
failure shows why that is insufficient to assert immediate object release.

## Validation and decision

- All 320 Rust tests, optimized production/diagnostic builds and formatting pass.
- Actual unchanged Metal router kernels pass 7,840 output words and 96 guards at
  widths 32, 64, 1024. The independent raw-ring proof passes 1,567 cases.
- The 12K internal result matches its complete pinned fixture. Normal/profiled
  32K/128 runs pass 33,354,240 FP32 comparisons, and the separate lifetime run
  passes another 16,677,120. Every full logit is compared, not only greedy tokens.
- All 4,466 attention/tail profile intervals pass the unchanged strict audit.
- The unchanged weak observer passes its optimized ownership/deduplication proof.
  All 119 trace snapshots pass structural/accounting checks. The acceptance
  audit fails real lifetime observations; this is not an expected-width error.
- Full model, prompt, oracle source/executables and all 7,669 fixture hashes
  pass before and after validation. Oracle-v5 remains
  ccbd16d36895ef9fe8edf7f793014988ff1ab7f4.
- The candidate is rejected before the engine-study prerequisite. There are
  zero warmups and zero timing pairs. Prepared engine plans/scripts remain
  archived but unused. No speed or whole-engine memory claim is made.
- All 42 accepted source hashes are restored. The default executable remains
  2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4.
  Candidate binaries and a copy of the prior default remain isolated.

Logical buffer lengths, device allocation size, RSS and OS footprint are
separate accounting measures. These diagnostic runs do not determine whether
an uninstrumented candidate would pass the original <=15% footprint gate.
That study remains unperformed because its prerequisite failed.

## Next milestone

[The follow-up design](next-milestone-design.md) investigates explicit bounded
scratch reuse, with a checked completion boundary and per-buffer ownership/
write-coverage proofs. It should establish a storage bound independently of
prompt destruction of completed command resources. This is a distinct candidate
and needs a new prospective lifetime contract, full C0 and unchanged whole-engine
performance/footprint gates. It has not started. Stop here; keep 256K paused and
publication deferred.

## Portable evidence

```sh
python3 rust-star/reports/bounded-1k-34eabe66/verify-report.py
```

Python 3.9+ is sufficient; no GPU, model or Rust toolchain is needed. The verifier
checks archive hashes, the exact scheduling inverse and inherited source pins,
reconstructs the observation patch, recomputes profile/ring/failure audits,
reproduces the original failing lifetime assertion, and verifies source/default
restoration and zero engine runs. Raw logs, traces, plans, builds, proofs,
input pins, failure records and unused engine preparation are preserved.
Large artifacts are losslessly compressed. Binary identities are archived;
the binaries remain in the ignored work directory.
