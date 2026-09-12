# Rust Star Work Journal

This is the operational source of truth for current state, evidence, blockers,
and next actions. Stable scope and contracts live in `RUST_STAR_PROJECT.md`.
User/hardware/account-dependent work lives in `RUST_STAR_MANUAL_TASKS.md`.

## Resume Protocol

Before changing code:

1. Read `RUST_STAR_PROJECT.md` and this file completely.
2. Run `git status --short --branch`, inspect `git log -1`, and verify remotes.
3. Confirm that the current-state summary still matches the repository.
4. Check whether the active oracle manifest and benchmark inputs are pinned.
5. Update this journal before handoff, context compaction, or the end of a
   meaningful session.

Journal entries are reverse chronological. Do not edit old entries to change
history; add a correction and update the current-state summary.

## 2026-09-12 — Complete explicit scratch reuse: rejected

- All local source/GPU/build proofs pass: 320 Rust tests, optimized production/
  observed builds, formatting, 1,567 ring cases, router 7,840 words/96 guards,
  weak observer proof and pool helper 62,043,240 bytes/192,360 guards. Synthetic
  checks include 2,205 identity reuses and 84 rejected invalid checkouts.
- 12K internal and three fresh 32K/128 full-C0 runs pass 50,031,360 exact FP32
  comparisons; all 4,466 profile intervals pass. Pool audit covers 28 tiles,
  119 snapshots and 2,940 checkouts: 105 stable distinct buffers per chunk,
  exact capacities, completed prior work, at most one unfinished command,
  no prior-chunk scratch at new entry and successful final owner cleanup.
  Sampled logical peak is 2,493,173,233 bytes; all final weak samples are zero.
- Preserve the original unsigned/signed nil-command marker audit failure and
  failed driver statuses. A CPU-only probe established NSUIntegerMax before
  results were examined; the corrected auditor changes only that sentinel,
  still requiring status 4 for every reuse. No runtime/observer, capacity or
  performance-gate changes and no repeated inference. Three preparation-script
  assertion failures and corrections before builds are preserved too.
- All 12 fixed engine runs finish with exact transcripts: two excluded warmups
  and five AB/BA/AB/BA/AB pairs. Paired medians: prefill -10.243%, steady
  decode -0.005%, RSS +2.415%, peak memory footprint +11.663%.
  Decision: REJECTED; all accepted source/default restored. Peak RSS is the
  only failed gate: +2.415% exceeds the unchanged +2% limit.
  This compares Rust 1.74 with Rust 1.74, not DwarfStar or statistical significance.
- Full input-after model/oracle/prompt/7,669-fixture hashes and all 42 final
  source hashes pass. Prior default and frozen binaries are preserved; reused
  build target contains the observed executable and is not the default.
- Portable report: rust-star/reports/pool-1k-fcdfef5f/README.md. Verification
  passes 221 artifacts,
  source/observer inverses, C0/profile/ring/pool/engine re-audits, the original
  sentinel failure and rejection of a corrupted completion status. GPU/runner/
  hash sessions are complete. No publication, remote dispatch or 256K work.
- Staged whitespace review preserves exact frozen evidence: host-tests.log has
  a blank EOF, implementation.patch has six original context-space lines and
  a blank EOF, and pool-helper.inc has its exact final blank line. Every other
  staged path passes with only those three report files excluded. Their bytes
  remain verified and unchanged.
- Next milestone, NOT STARTED: diagnose scratch RSS residency and prove which
  large buffers can safely avoid unconditional clearing before another 32K
  candidate.
  Finish the local commit and user update, then STOP.

## 2026-09-12 — Scratch pool RSS gate already rules out retention

- First three complete pairs show peak-RSS ratios1.024184/1.024157/1.024152,
  all above the unchanged median limit1.02. With three of five ratios already
  failing, the final median cannot pass. This corrects an intermediate update
  that considered prefill/decode/footprint without checking RSS separately.
- The fixed block continues through pairs4/5; do not exclude or repeat results.
  After final audits/hashes, reject and restore accepted source/default. Early
  prefill improvements and footprint ratios below1.15 do not override RSS.
- Next proposal: bounded scratch-residency diagnosis and full-first-write proofs
  for possible selective initialization. Unconditional clearing is a hypothesis
  for the RSS cost, not an established cause. No next-milestone work has started.

## 2026-09-12 — Scratch pool passes full C0 and ownership; engine block active

- Both frozen Rust1.74 builds pass, as do320 host tests, formatting,7,840-word/
  96-guard router checks,1,567 ring cases and the weak-observer proof. The exact
  pool helper passes62,043,240 byte comparisons,192,360 guards,2,205 identity
  reuses and84 invalid checkouts across seven synthetic four-tile GPU chunks.
-12K internal plus normal/profiled/observed32K/128 checks all pass. Three full
  runs compare50,031,360 FP32 words exactly; all4,466 profile intervals pass.
- The frozen observer encodes no-prior-command as unsigned NSUIntegerMax; the
  original auditor expected signed-1. A CPU-only Foundation probe confirmed
  the representation before pool results were examined. Preserve the original
  auditor failure and failed driver statuses. audit-pool-v2.py changes only the
  first-checkout sentinel; reuse still requires completed status4. Runtime,
  observer, capacities and performance gates are unchanged; no inference retry.
- Corrected audit passes28 tiles/119 snapshots/2,940 checkouts across7 chunks:
  105 distinct stable buffers per chunk, exact capacities, at most1 unfinished
  command, no prior-chunk scratch at new entry and all final owner releases.
  Maximum sampled logical scratch2,493,173,233 bytes; every final weak sample
  is zero too. This is not an OS-footprint or speed acceptance claim.
- Fixed engine block is active via launch-engine.py in
  rust-star/.work/pool-1k-fcdfef5f. Two excluded warmups then AB/BA/AB/BA/AB;
  no build/full-model hash overlaps timing. Source/default retention remains
  undecided until all unchanged gates pass. Complete input-after, decision,
  portable report, docs and local commit, then report the next milestone/STOP.

## 2026-09-12 — Start explicit bounded scratch reuse

- D approved one milestone. Prospective plan and unchanged engine gates are
  frozen in rust-star/.work/pool-1k-fcdfef5f/plan.json and engine-plan.json.
- Use one distinct scratch buffer per label within each eligible four-tile 1K
  chunk, capped at 105 buffers and 2,493,173,233 logical bytes. Checked prior
  completion precedes all initialization; clear/copy the entire requested range
  to preserve fresh Metal buffer initialization, then keep original CPU writes.
- Persistent state is excluded. Prove allocation routing, no escape, identity,
  capacity, initialization and final/error owner cleanup before fresh full C0 and
  pool-aware diagnostics. Only all passing prerequisites allow the fixed engine
  block. Restore accepted runtime on failure; retain all evidence. Finish this
  milestone, report the next one and STOP. No publication or 256K work.

## 2026-09-12 — Close bounded 1K: exact outputs, failed scratch-lifetime gate

- All 320 Rust tests, production/diagnostic optimized builds, formatting, router
  GPU checks (7,840 words/96 guards), 1,567 ring cases and observer ownership
  proof pass. 12K internal plus normal/profiled/lifetime 32K/128 checks pass;
  three full runs compare 50,031,360 FP32 words exactly. All 4,466 profile
  intervals pass. No correctness failure or repeated inference.
- The frozen lifetime auditor fails a real post-pool condition. At frontier
  20480, tile16384 reports completed but its after_pool snapshot still observes
  1,977,044,465 scratch bytes. At tile17408 entry, 1,420,731,848 previous bytes
  remain; at registration/submission, three older buffers total 5,242,880 bytes.
  The following pool snapshot is zero. Supplemental analysis preserves all four
  violating snapshots and confirms complete 28-tile/119-snapshot accounting.
- Maximum unfinished command count is one and sampled global scratch peak is
  2,493,173,233 bytes, 75% below the previous four-set trace. Neither result
  overrides the failed per-tile ownership prerequisite. Completed status and a
  local pool drain did not coincide with immediate disappearance of every
  observed buffer; the remaining owner/exact release timing is unproved.
- Reject before engine timing: zero warmups/pairs, no threshold changes or
  repeats. Full input-after model/oracle/prompt/7,669-fixture hashes pass before
  restoration. All 42 accepted source hashes and the default 2bf77edb executable
  are restored/unchanged. A separate prior-default copy and both candidates are
  preserved. GPU/validation/hash sessions are complete and drained.
- Verified report: rust-star/reports/bounded-1k-34eabe66/README.md. First portable
  verification passes 116 artifacts, source reconstruction, complete C0/profile/
  ring accounting, reproduction of the original failed assertion, all failure
  observations and final restoration. Output is in
  rust-star/.work/bounded-1k-34eabe66/verify-execution.log. No verifier retry.
- Whitespace review preserves exact raw artifacts: the full staged check flags
  only host-tests.log (blank EOF) and three original patch-context space lines
  in implementation.patch. All other staged files pass with those two exact
  report paths excluded. Their verified bytes remain unchanged.
- Next milestone, NOT STARTED: explicit bounded scratch reuse with completion
  and per-buffer ownership/full-write proofs, a prospectively defined pool
  identity/capacity contract, fresh C0 and unchanged engine/memory gates. Do not
  relabel this candidate as passing or assume an extra wait ensures destruction.
  Commit/report this milestone, then STOP. No publication or 256K work.

## 2026-09-12 — Start bounded 1K completion experiment

- D approved the next milestone. New prospective plan and unchanged engine
  retention criteria: rust-star/.work/bounded-1k-34eabe66/plan.json and
  engine-plan.json. Finish this milestone and stop with the next milestone.
- Candidate is the preserved rejected 1K source plus one native-1K-guarded
  checked completion wait after nonfinal tiles. Final wait/error audit stays
  unchanged. No shader, arithmetic, buffer alias, fixture or oracle change.
- Freeze both uninstrumented and observed builds after host/router/ring and
  ownership proofs. Require 12K internal, normal/profiled 32K/128 full C0,
  strict profile coverage and a separate full-C0 lifetime bound. Only then run
  two warmups and five AB/BA/AB/BA/AB engine pairs with unchanged gates,
  including paired median footprint <=1.15. Preserve every attempt and restore
  accepted runtime on failure. No publication or 256K work.

## 2026-09-12 — Complete bootstrap lifetime diagnostic; retain accepted runtime

- Both optimized instrumented Rust 1.74 builds and the synthetic weak-ownership
  proof pass. Accepted 64-row and rejected 1K paths each pass fresh 32K/128 full
  C0: 33,354,240 exact FP32 words total. Full model/oracle/prompt/7,669-fixture
  hashes pass before and after. GPU and runner sessions are complete and drained.
- Complete coverage: 448/28 native tiles, 1,799/119 snapshots, seven chunks each.
  The 1K path overlaps four scratch sets; at 32K each is 2,493,173,233 bytes,
  totaling 9,972,692,932 logical bytes. Accepted path peak is 11,180,845,923 bytes.
  At every final wait, earlier scratch is gone despite retained completed command
  objects; every final pool drain leaves zero tracked scratch in both paths.
- Logical live lengths are not OS footprint. Instrumented whole-process peak
  footprint is 12,346,315,792/15,423,263,784 bytes accepted/1K. Neither these
  perturbed processes nor their sampled counters explain the earlier 44.722%
  engine footprint regression. No new timing, retention or DwarfStar claim.
- Original frozen audit fails its documented 32-row coverage assumption after
  both C0 runs and final hashes pass. Preserve that failure. The correction was
  frozen during the accepted run; audit-v2 changes only the expected width to
  the source-pinned 64 rows and passes without repeating inference.
- Verified portable report: rust-star/reports/bootstrap-lifetimes-bff07722/README.md.
  First verifier passes 60 artifacts, exact source reconstruction, raw-trace/C0
  accounting and three malformed-trace rejection cases. Verification output:
  rust-star/.work/bootstrap-lifetimes-bff07722/verify-execution.log. No verifier
  failure or retry. All 42 accepted source hashes and default 2bf77edb executable
  remain unchanged. Diagnostic binaries are isolated; no production edit retained.
- Next milestone, NOT STARTED: test a checked completion wait after each nonfinal
  eligible 1K tile, bounding unfinished bootstrap work to one tile without new
  buffer aliases. Concrete design is in the report. Require fresh full C0,
  separate lifetime-bound proof and fixed uninstrumented engine/memory gates,
  including unchanged footprint ratio <=1.15. Commit/report this milestone and
  STOP. Keep 256K paused and publication deferred.

## 2026-09-12 — Correct lifetime-plan baseline width, preserve original audit

- During the first diagnostic run, raw rows and pinned accepted-metal.rs show
  production tiles are64 rows;32 rows applies to collect_outputs diagnostics.
  The plan/auditor description was wrong. The frozen accepted executable,
  source, requested output-only path and model inputs are correct and unchanged.
- Preserve plan.json/audit.py and their expected coverage failure. New
  width-correction.json freezes audit-v2.py with the single32-to64 expected-width
  change before the wide run. Reuse the same two raw runs, with no new inference,
  exclusions, changed C0 criteria or performance claims. Expected accepted
  coverage is448 native tiles/1799 snapshots across seven chunks.

## 2026-09-12 — Start bounded bootstrap lifetime diagnostic

- D approved the next milestone. Keep one-milestone mode: measure and report,
  then stop before implementation of any resulting production optimization.
- New prospective plan: rust-star/.work/bootstrap-lifetimes-bff07722/plan.json.
  One accepted32-row and one rejected1K fresh32K/128 full-C0 diagnostic run.
  Identical weak-reference instrumentation enumerates121 local buffer names,
  separates16 persistent aliases, and samples command status, logical live
  lengths, device allocations and OS footprint. Prove observer ownership with
  synthetic allocations before model runs; preserve all attempts and failures.
- These runs are perturbed lifetime observations, not a new performance study.
  No scratch reuse is presumed safe. Derive a concrete bounded policy from the
  result, archive source inverse and raw evidence, restore all42 accepted source
  hashes, and stop. No default change, publication or256K work is authorized here.

## 2026-09-12 — Close 1K bootstrap: faster prefill, rejected memory footprint

- All 12 engine processes (two excluded warmups and five measured pairs)
  complete with exact oracle transcripts. Every pair is prefill-faster; paired
  median latency is -11.234%, with AB/BA ratios 0.887659/0.887799. Aggregate role
  medians are 226.109s baseline and 200.808s candidate. Steady decode -0.317%
  and peak RSS -0.059% pass their gates.
- Separate macOS peak memory footprint rises 44.722% at the paired median,
  exceeding the unchanged <=15% allowance. Role medians are 12,385,601,512 and
  17,927,292,616 bytes. Every footprint pair is above 1.43. This is the sole
  failed retention gate; reject the candidate without relaxing it or adding runs.
- Passive paired bootstrap wall time falls 25,305.902ms; transformer wall time
  falls 415.014ms; host overhead rises 307.742ms. These timings do not establish
  a particular allocator, command-lifetime or stall cause for the footprint.
- Full input-after hashes pass for model, prompt, oracle, all 7,669 fixtures,
  source and executables. Engine session 18051 and final-hash session 21502 are
  complete and drained. All 42 accepted source hashes are restored; the default
  performance executable remains 2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4.
  Experimental binaries and all raw evidence remain preserved separately.
- Verified portable report: rust-star/reports/bootstrap-1k-d72b7528/README.md.
  First verification .work/bootstrap-1k-verify-1.log passes 178 artifacts,
  complete source inverse, all 33,354,240 full-C0 FP32 comparisons, profile
  coverage and recomputation of every engine/memory gate. No verifier failure.
- Whitespace review preserves byte-exact raw artifacts. The full staged check
  flags only host-tests.log (blank EOF) and implementation.patch (three original
  patch-context space lines) in this report. Code/docs and all remaining staged
  files pass with only those two exact artifact paths excluded; hashes remain
  unchanged. No runtime or report-data repair is needed.
- Next milestone: measure bootstrap scratch allocation and command lifetimes,
  then consider a narrowly scoped reuse or bounded-in-flight policy under a
  new plan. Source review supplies a hypothesis, not a proved cause or fix.
  No follow-up implementation has started. Commit this milestone, report it,
  then STOP as requested. No publication, new DwarfStar claim or 256K work.

## 2026-09-12 — Four 1K engine pairs complete; final pair active

- Both warmups and four complete measured pairs finish successfully with exact
  oracle transcripts. Pair5 baseline started12:39:54 UTC in engine session18051.
  Finish its candidate role before calculating the complete gate decision.
- Prepared closure/report/verifier helpers are copied into the study directory.
  They do not change runtime, frozen study scripts, criteria or raw data. The
  portable verifier accounts explicitly for redacted plan and raw-file hashes.
- After the entire block, audit passive phases and complete final input hashes
  with GPU idle, retain/restore according to every predeclared gate, verify the
  portable report and commit locally. Then stop with outcome and next milestone.

## 2026-09-12 — 1K full C0 passes; fixed engine study begins

- The 12K internal check and normal/profiled 32K/128 full-output checks all
  pass. Two full runs check 33,354,240 FP32 words exactly. Every one of the
  4,466 profile component intervals passes the unchanged audit. Session72205
  completed successfully; no failed run or discretionary retry occurred.
- Diagnostic bootstrap time: 32,939.910ms baseline to6,074.175ms candidate;
  raw ratio0.184402 and control-normalized ratio0.195533 both pass <0.90.
  This is perturbed component evidence only, not whole-engine acceptance.
- Start engine-runner.py: two warmups, then AB/BA/AB/BA/AB, with current
  Rust1.74 baseline and frozen candidate31898eb7. Keep every raw run and all
  predeclared prefill/order/decode/RSS/footprint gates. No concurrent GPU work,
  model hashing, heavy builds, source edits or frozen-script changes.
- The memory auditor independently reproduces the old4K metrics and rejects
  its known1.524844 footprint ratio. This is an auditor regression check only;
  it does not alter the prior rejected study or contribute new timing samples.

## 2026-09-12 — 1K candidate build and local proofs pass

- All 320 Rust 1.74 tests, optimized build and formatting pass. Complete source
  inverse/formatter audit covers the three edited files; 39 others are unchanged.
  The raw-ring proof covers 1,567 cases. Actual unchanged Metal router kernels
  pass all 7,840 output words and 96 guards at widths 32, 64 and 1024.
- Candidate-build.json freezes source, binary and run scripts. No source or
  frozen-script edits during inference. Build session 66370 is complete.
- Full input verification is active in session 64776 (input-before.log), with
  GPU idle. Require its success before run-validation.py: 12K internal, normal
  32K/128 and profiled 32K/128 full-logit gates, stopping on any failure.
- Peak-memory-footprint median ratio <=1.15 remains a required engine gate,
  alongside every original prefill/order/decode/RSS criterion. No threshold
  changes, discretionary reruns or publication. Stop after this milestone.

## 2026-09-12 — Resume the bounded 1K bootstrap milestone

- User authorized the next milestone with “lets do it”; finish this milestone
  and report its outcome and next action, then stop. Continuous execution
  remains off. No publication or automatic 256K work.
- Verified the prepared experiment against the accepted 42 source hashes.
  Reconciled its baseline commit from e73a4b79 to documentation-only d72b7528;
  preserved both original plans and every unchanged acceptance threshold.
- Starting .work/bootstrap-1k-current/build-and-freeze.py. It freezes scripts,
  applies the scoped three-file candidate, verifies the complete inverse and
  Rust 1.74 formatting, then host/build/router/ring gates. Build failures stop
  and remain preserved. Full-model and engine runs remain gated on those checks.

## 2026-09-12 — Return to milestone-by-milestone updates

- User ended token-burn mode and requested finishing the current milestone,
  then reporting its outcome and the next milestone. The current Rust 1.74
  full-C0/profile milestone had just closed at e73a4b79535d11306e9be7ada4f00d8e72da5950.
- Stop at that completed checkpoint. No GPU work remains. All 42 accepted
  runtime source hashes still match its completion record.
- The separate 1K bootstrap experiment is prepared at
  .work/bootstrap-1k-current/, but prepare-source/build have NOT run: no
  pre-edit-freeze.json, source-edits.json or candidate-build.json exists there.
  Its plan currently pins e73a4b79; before a future start, reconcile that pin
  with any subsequent documentation-only commits without changing the accepted
  source, executable or prospective criteria. Freeze the final scripts first.
- Next milestone to report: bounded 1K bootstrap batches with source/router/ring,
  internal/full-C0, diagnostic and whole-engine/RSS/footprint gates. Do not start
  it automatically from the prior continuous-execution instruction.

## 2026-09-12 — Current Rust 1.74 full C0 and profile complete

- Both fresh 32K/128 processes pass all 33,354,240 FP32 comparisons against
  immutable oracle-v5 outputs. The strict profile audit covers all 4,466
  intervals. Full model/prompt/oracle, executable, 42 source and 7,669 fixture
  pins match before and after. Session 4092 is complete and drained.
- Verified portable report: rust-star/reports/msrv-c0-5834fe56/README.md.
  Python-only verification .work/msrv-c0-verify-1.log passes; complete source
  snapshots and compiler-build/profile-audit provenance are included. This
  closes the Rust 1.74 repair's fresh 32K full-model gap; no speed comparison.
- Bootstrap ingress totals 32,939.910 GPU ms, 16.233% of recorded stage time.
  It exceeds the prospective 10% eligibility threshold for a distinct 1K
  working-set experiment. Four tiles per native 4K continuation chunk preserve
  legacy/prefix/decode and >32K behavior. Keep previous rejected studies closed.
- Next: freeze .work/bootstrap-1k-current/ plan with same-compiler baseline,
  exact source/router/ring/full-C0 gates, diagnostic selection, five fixed
  engine pairs and separate median peak-memory-footprint ratio <=1.15.
  Only after this milestone's local commit may preparation mutate runtime.

## 2026-09-12 — Normal current-compiler 32K full C0 passes

- Session 4092 completes current-32k-normal with exact prefill and all 128
  decode steps: 16,677,120 FP32 words. Entire expected output JSON matches.
- The same frozen runner starts current-32k-profiled at 07:18:55 UTC. Finish
  its full C0, unchanged component audit and final complete input/source hashes
  before packaging. No throughput or compiler performance claim follows.

## 2026-09-12 — Current Rust 1.74 full-model C0 validation active

- Routed-half median report closes at 5834fe56; its work-directory milestone
  completion records the verified unselected decision and unchanged runtime.
- .work/msrv-c0-5834fe56/plan.json prospectively fixes two fresh 32K/128
  full-output processes: normal, then profiled. Any failure stops; no retries.
- Full model, prompt, oracle source/binaries, current executable, all 42 runtime
  sources and all 7,669 tracked fixture files are hashed and frozen. GPU idle
  during hashing. Current executable d8c3919e was built with Rust 1.74; accepted
  historical performance executable stays unchanged and separate.
- Session 4092 runs run.py; read execution.log and active-run.json. No runtime
  edits, heavy compilation or other GPU work until both runs and final pins close.
  The unchanged profile coverage audit is adapted only for two runs and their
  names. This is exactness and diagnostic profiling, not a timing comparison.
- Prepared package/verify tools are result-independent; run only on completion.

## 2026-09-12 — Close longer median protocol; routed-half remains unselected

- All five frozen runs completed, 23,015,768,064 exact FP16 comparisons,
  including 4,896,374,784 direct candidate/original comparisons. All six initial
  A/A cells pass, but mixed session 0 case 2 fails BA neutrality (0.989395)
  and session 1 case 2 fails arena neutrality (0.989549). No samples excluded.
- Candidate also misses every per-cell speed gate and the overall <0.95 gate
  (descriptive paired median 0.984955). Failed mixed controls prevent a reliable
  gain claim. No runtime integration, threshold change or rescue rerun.
- Portable report rust-star/reports/routed-half-median-abe52d80/README.md
  verifies 83 artifacts, full kernel/protocol inverse, raw medians and all gates.
  Verification: .work/routed-half-median-verify-1.log. All 42 runtime sources
  remain unchanged. GPU session 52217 completed successfully and is drained.
- Next: fresh normal/profiled 32K/128 full C0 for the current Rust 1.74 build,
  under frozen inputs and source pins, then a distinct profile-led hypothesis.
  The previous 4K bootstrap and both routed-half studies stay closed.

## 2026-09-12 — Longer-window neutral controls pass; candidate comparison active

- Fresh edges and both predeclared A/A sessions pass. All six neutral cells
  satisfy every unchanged median/order/arena/dispersion threshold; all 72
  pairs and output checks are retained. This establishes the new protocol's
  initial neutral screen, not a candidate or engine improvement.
- Mixed session 0 began 06:58:49 UTC under the same runner, session 52217.
  Two full sessions interleave contemporary A/A and candidate comparisons;
  finish both before evaluating candidate selection. Source and scripts stay
  frozen; no concurrent GPU or heavy compilation work.
- Result-independent package/report/verifier tools are prepared in the work
  directory. They also invert the entire protocol change to the closed prior
  screen and prove identical kernels and thresholds. Execute them only after
  the running block and final source pins complete.

## 2026-09-12 — Start a fresh longer median-based timing protocol

- The failed mean-of-three screen is closed at
  abe52d805638380df2184321e68044d0ef9ae924. Its verified report and every raw
  sample remain unchanged; milestone-completion.json records closure.
- New attempt: .work/routed-half-median-abe52d80/. Kernel and combined Metal
  sources are byte-identical to the first screen. Exactness cases, pair orders,
  arena schedule and all neutral/candidate thresholds are unchanged.
- Fixed change before compilation: eight conditioning and twelve measured
  complete chains per role; median of the twelve GPU command intervals is the
  primary slot estimate. Keep the arithmetic mean as descriptive data and
  preserve every raw interval. Explicit script-edit ledgers invert the entire
  harness/auditor/runner changes; new source/plan/build hashes are frozen.
- Clang harness build passes. Fresh edge session 0 began 06:53:09 UTC, exec
  session 52217. Then both predetermined A/A sessions must pass before any
  candidate timing. Read active-run.json, execution.log and phase status files;
  no concurrent GPU/heavy compile and no changes to frozen runtime/scripts.
- This is a new measurement protocol, not a rerun or reclassification of the
  prior failed study. Runtime remains unchanged. Close the full decision and
  continue toward C0/whole-engine evidence only if local selection succeeds.

## 2026-09-12 — Close routed-input screen after failed A/A neutrality

- Edge checks pass all 16 cases and 64,536,576 FP16 words. Both predetermined
  A/A sessions complete with all 72 pairs and 7,851,737,088 further words.
  Total comparisons are 7,916,273,664; 668,516,352 directly compare candidate
  and original outputs in edges/full-case warm-ups, with the rest A/A checks.
- Five of six neutral cells fail the fixed variability/order/arena criteria.
  The runner stops before candidate timing; no kernel speed conclusion or
  runtime integration is allowed. This is a failed measurement protocol, not
  evidence that the kernel itself is slower. All attempts/samples are retained.
- Raw three-sample slots contain occasional duration ratios up to 1.457587.
  This motivates a separate prospective test of eight conditioning chains and
  twelve measured chains per role, using their median as the slot estimate.
  Keep the same neutral/candidate thresholds and require fresh A/A success;
  do not relabel the existing failed data with a different statistic.
- Portable report rust-star/reports/routed-half-neutral-24c5d03e/README.md
  verifies all 67 artifacts, the complete kernel/assembly/ABI/host proofs and
  all raw edge/A/A audits. Every one of the 42 runtime source hashes still
  matches the accepted compatibility source at 24c5d03e. No GPU work is active.
- Commit the closed screen locally, then freeze and execute the new protocol
  as a separate attempt. No candidate speed, full-model, DwarfStar or frontier
  claim follows; the interrupted 256K campaign remains paused. No push.

## 2026-09-12 — Start the routed-input half local screen

- Compatibility closure is committed at
  24c5d03e1a5e4d6c732a7380f5bbd6acaa83f96f; its completion record pins the
  report, verifier, source and retained performance executable. No push.
- Started .work/routed-half-prefill-c7adfb22/run.py, exec session 24784.
  Clang host harness build passes; edge session 0 began 06:42:51 UTC. The
  actual original and candidate Metal kernels compile inside this process.
  Read active-run.json, execution.log, phase statuses and failure.json before
  any additional GPU or heavy compilation work.
- This standalone experiment leaves the runtime unchanged at 24c5d03e. Source
  parts, full inverse, exact production arguments/encoding and frozen scripts
  are pinned. Edge checks precede both predeclared A/A sessions; mixed timing
  only follows passing neutral controls. Preserve any execution/exactness or
  statistical failure without retuning the gates or adding discretionary runs.
- The next decision is local selection, not an engine gain. If selected, freeze
  a separate accepted-baseline integration/full-C0/whole-engine plan before
  changing runtime code. If rejected, close the hypothesis and continue.

## 2026-09-12 — Complete Rust 1.74 compatibility and return to performance work

- Corrected source passes pinned Rust 1.74 formatting, all 320 host tests,
  optimized native build, all 94 Python artifact tests and the five-word
  synthetic writer/reader fixture. Both original compiler failures are retained.
- Verified report: rust-star/reports/msrv-174-179e0c24/README.md. Its 50-artifact
  verifier uses Python and Rust 1.74 rustfmt to independently reproduce the
  complete source inverse, verify every prior/current source and log identity,
  and recheck the artifact contract without GPU or model execution.
- Only metal.rs changes: equivalent replacements of two Option calls, and a
  private safe array-borrow helper replacing 239 each_ref/39 each_mut calls.
  All other 41 pinned sources, including every GPU source, remain unchanged.
  The declared minimum stays Rust 1.74; no unsafe code or heap allocation is
  introduced by the helper. Actual source reversal passes in the verifier.
- The accepted performance executable stays exactly 4ced6ee8. The separately
  compiled 1.74 binary identity is in the report; no new full-model C0, engine
  result, Linux execution or GitHub CI success is inferred from these checks.
- Changed source/docs pass whitespace checks. Preserve original whitespace in
  the archived compiler output, test log, patch and exact helper template; the
  staged check passes with only those four byte-pinned evidence paths excluded.
- No GPU work is active. Commit this local checkpoint, record compatibility
  closure, and directly start .work/routed-half-prefill-c7adfb22/run.py. Keep
  all its predetermined edge/neutral/candidate gates and preserve every failure.

## 2026-09-12 — Extend the compatibility repair from compiler evidence

- Rust 1.74 installation succeeded. The accepted-source check reports two
  missing is_none_or calls and 157 unstable array_methods errors. The initial
  two-call repair passes formatting, but host tests fail on 278 array methods
  (239 each_ref and 39 each_mut call sites, including test-only code).
  Preserve .work/msrv-174-c7adfb22/failure.json and every compiler log; its
  runner stopped before optimized build or subsequent contract checks.
- Corrected attempt: .work/msrv-174-array-borrows/. A private safe ArrayBorrows
  helper uses std::array::from_fn and iter_mut, preserving element order and
  references without cloning or allocating the elements. Mechanically replace
  the 239/39 method calls, retain the two Option fixes, and leave the other 41
  pinned sources (including GPU code) untouched. Full formatted inverse proof
  recovers the preceding source exactly.
- Frozen runner started 2026-09-12T06:36:01 UTC, exec session 23805. Formatting
  passes; pinned host tests are compiling. Read execution.log and completion.json
  or failure.json. It reuses the prior isolated Cargo target and same native
  build marker; no GPU work is active. Do not modify the frozen source/scripts.
- After this corrected run, package both attempts and verify the source reversal
  using Rust 1.74 rustfmt plus model-free artifact checks; then commit locally
  and continue the prepared routed-input experiment. No remote CI, new model
  correctness or performance claim follows from these compatibility checks.

## 2026-09-12 — Start the pinned Rust 1.74 compatibility milestone

- Wide-bootstrap rejection/restoration is committed locally at
  179e0c2408d3738bbb8d749775610741c51b9c19; its milestone-completion.json
  records the final manifest/verifier/restoration hashes. No push occurred.
- Started .work/msrv-174-c7adfb22/run.py at 2026-09-12T06:30:29 UTC. It first
  installs Rust 1.74.0 with rustfmt, then records the expected accepted-source
  compiler failure before changing the two report-validation Option calls.
  Read execution.log, per-command status/logs and completion.json or failure.json
  before taking the next action. The active exec session is 57137.
- Isolated CARGO_TARGET_DIR keeps the accepted performance executable intact.
  Every other pinned source must stay byte-identical; no GPU/model timing is
  active. Do not run the queued routed-half probe during compilation.

## 2026-09-12 — Close wide bootstrap as a rejected whole-engine candidate

- Completed every planned run: two warm-ups and five AB/BA/AB/BA/AB pairs,
  all 12 exact transcripts with profiling/readback disabled. Candidate c7adfb22
  versus frozen accepted 4ced6ee8 has median paired prefill latency -1.127%,
  steady decode +0.556%, and peak RSS -0.997%. Pairs 1 and 5 are slower;
  the all-five-faster and separate-order prefill gates both fail. No pair was
  excluded or rerun, and no retention threshold was changed.
- Separate macOS peak memory footprint increases +52.484%: median baseline
  11.539 GiB, candidate 17.593 GiB. Passive timings show median paired tile
  wall saving 16.477s, transformer command wall increasing 6.710s and host
  overhead increasing 4.960s. These aggregates do not identify a specific
  allocation or stall cause. GPU tile time spans first-to-last commands.
- Final full model/prompt/oracle/executable/source hashes pass. The portable
  report, rust-star/reports/wide-bootstrap-c7adfb22/README.md, verifies 245
  artifacts, complete source proofs, the initial compile and 12K failures,
  the 29,344-word router dispatch proof, all fresh full C0 records and every
  engine pair. The three full-output runs compare 50,031,360 FP32 words.
- First packaged verifier failed because ranked Python tuples become JSON
  arrays. Preserved its failure and original packaged scripts; both complete
  profile results match exactly after JSON canonicalization. Original frozen
  audits, raw data and gates are unchanged. Final verifier passes including
  restoration evidence; its log is .work/wide-bootstrap-c7adfb22-verify-3.log.
- Restored the three runtime files; all 42 source hashes and the default work
  executable match accepted 4ced6ee8 (SHA 2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4).
  Candidate commits, executables and failed attempts remain preserved.
- Source/document whitespace checks pass. The full staged check reports only
  original whitespace in three archived patch files and two compiler test logs;
  keep those byte-exact evidence files unchanged and check the remaining paths.
- Continue immediately: pinned Rust 1.74 compatibility repair, then the prepared
  routed-input half experiment. No GPU work remains active. No DwarfStar or
  higher-frontier claim; interrupted 256K work remains paused. No push.

## 2026-09-12 — Queue a pinned-toolchain compatibility check

- Source-only review finds two `.is_none_or` calls in engine-report validation
  in metal.rs. They exist in accepted 4ced6ee8 as well as c7adfb22; the manifest
  declares Rust 1.74 and both remote CI jobs install 1.74.0. Official Rust API
  documentation marks this method since 1.82.0:
  https://doc.rust-lang.org/std/option/enum.Option.html#method.is_none_or.
- No Rust 1.74 compiler test has run yet. Record the compiler failure on the
  restored accepted source, then use the equivalent `map_or(true, predicate)`
  calls and validate formatting, host tests/build, Python contracts and fixture
  interoperability with the pinned toolchain. Do not claim remote CI success.
- Observation and prospective validation are in
  rust-star/.work/msrv-174-c7adfb22/observation.json. No runtime source mutation,
  toolchain install or compilation may overlap the active engine block. This
  bounded compatibility repair can follow full report/pin/restore closure,
  before the prepared routed-input GPU experiment.

## 2026-09-12 — First whole-engine pair rejects 4K-bootstrap retention

- Pair 1 is valid and preserved: baseline 234.538128s prefill, candidate
  250.460674s, ratio 1.067888943 (candidate slower). This fails the prospectively
  required all-five-faster rule; the 4K candidate cannot be retained as an
  engine improvement under this plan. Do not discard/exclude/rerun this pair.
- Complete the remaining scheduled pairs to characterize the result, then
  final full input/source/binary pins, portable report and verification,
  restore the accepted three runtime files and executable from 4ced6ee8,
  journal/local commit, and continue. Keep c7adfb22 sources frozen while the
  existing runner is active, because its preflight requires those exact hashes.
- Passive pair-1 timings show bootstrap command wall improving 37.350s ->
  24.550s, while transformer command wall grows 191.002s -> 208.953s and
  aggregate host overhead grows 6.115s -> 16.773s. This offsets the subpath gain.
  Larger transient allocation/working-set effects are a hypothesis, not a
  demonstrated cause. A supplemental complete-block passive phase audit is
  prepared without new instrumentation, exclusions or acceptance thresholds.
- Pair-2 candidate also completes validly (239.419839s prefill); its baseline
  began 2026-09-12T05:53:27 UTC. Read engine/active-run.json for current status.
  The prepared routed-half experiment remains gated on full milestone closure.

## 2026-09-12 — Refresh remote CI evidence without a workflow dispatch

- Read-only connected GitHub queries still list 134 runs for the research
  branch. Latest is 33253238025, created 2026-08-29T12:43:51Z; both failed jobs
  report no steps. No newer remote success is established.
- The remote manual-only workflow matches the local file exactly, Git blob
  40fed1e1e2f08bf5169c3053630bbc37f0d47c34. No account/billing status or secrets
  were accessed; no remote mutation, workflow dispatch or push was performed.
- Corrected the manual ledger to distinguish the historical billing diagnosis
  from current account state, which remains unverified. M-001 still needs a
  successful manual workflow run. Filtered evidence is stored in
  rust-star/reports/ci-status-2026-09-12.json.

## 2026-09-12 — Prepare the next routed-input experiment without GPU overlap

- While the c7adfb22 whole-engine block runs, prepared source-only follow-on
  work at rust-star/.work/routed-half-prefill-c7adfb22/. No runtime files changed
  and no compilation or GPU execution was performed for this next experiment.
- The passing fresh candidate profile identifies routed gate/up as the largest
  individual component: 30,345.000ms, 17.20% of recorded stage GPU time.
  Hypothesis: convert normalized FFN inputs to half once, replacing repeated
  float-load/half-cast B staging while preserving all IQ2 dequantization, MMA,
  SwiGLU, clamp and output operations. Future integration can investigate
  reusing the already-dead attention-input half scratch instead of new buffers.
- Prospective plan, whole-kernel inverse proof, actual production arguments and
  encoding, standalone source/harness, edge checks and strict same-arena
  A/A/candidate audits are prepared. Two sessions reverse three routing/input
  cases; every neutral gate is inherited from the accepted same-arena protocol.
  New candidate criteria require every case/session/order median <0.98 and
  overall median <0.95, including conversion and map cost. No data yet.
- run.py is deliberately gated on prior milestone-completion.json, completed
  whole-engine audit and final input hashes. Finish all current engine pairs,
  verify/package the result, retain or restore and commit before activating it.
  Compile/source or GPU failures must be preserved; no silent retries.
- Both current engine warm-ups pass and are excluded. Pair 1 began with baseline
  at 2026-09-12T05:39:51 UTC. Read the active-run file for current progress.
  Supplemental parsing of the separately captured macOS peak memory footprint
  is prepared, alongside RSS, without adding or changing retention thresholds.

## 2026-09-12 — Repaired wide bootstrap passes C0 and diagnostic selection

- Frozen candidate c7adfb22706d5a5883d8a251a3a9046a2c8e587b, executable
  SHA-256 1a055037fbf65772258959503c97aeea184dbaa4913d3db1a523f7fae03b9794,
  passes 320 release tests, forced native build and exact source restoration.
- 12K internal checks pass all 43 layers of the initial prefix and both
  continuation chunks, including the original 7b4b76ac failure site. Separate
  normal and profiled full 32K/128 runs each pass all 16,677,120 FP32 words.
  No oracle values, arithmetic, tolerances or thresholds were changed.
- Strict profile audit passes all parent packets and attention/tail intervals.
  Bootstrap ingress drops from 31,943.208ms to 6,700.693ms. Raw ratio 0.209769
  and control-normalized ratio 0.202812 both pass the preset <0.90 gate.
  This diagnostic result is not a whole-engine or DwarfStar speedup claim.
- The complete unprofiled engine block is now active under
  .work/autonomous-prefill-a46dfe8b/wide-bootstrap-router-bound/engine/.
  Two warm-ups, then five pairs AB/BA/AB/BA/AB against frozen accepted4ced6ee8;
  all exactness, prefill, decode and RSS gates remain as prospectively fixed.
  Read engine/active-run.json and engine-execution.log before further GPU work.
  Do not mutate frozen runtime sources/scripts or launch concurrent GPU work.
- Post-run complete input hashing, portable package, independent verifier and
  decision-report generators are prepared. Execute only after all engine runs
  finish; preserve every attempt. The research mandate continues afterward.

## 2026-09-12 — Preserve wide-bootstrap failure and bound router dispatches

- Candidate 7b4b76ac passed 320 host tests and forced native build, then failed
  its first 12K internal model gate. All initial prefix layers 0–42 were exact;
  the first wide chunk at 4096 diverged at layer 0 final HC. Actual checksum
  18a5296a9c248c1c versus expected 8bda50b3f5ec5e19. The queue stopped before
  either 32K run or performance. Preserve wide-bootstrap/ unchanged.
- Source inspection found four hash-row/clamp-sum launches requesting 4096
  threads per group. Actual compiled Metal pipelines permit 1024 each. A
  standalone test of the unchanged shader kernels passes every selected ID
  and FP32 clamp word at 32, 64 and 4096 rows (29,344 output words plus 96
  guards) with MIN(rows,256) threads per group. This proves an invalid launch
  and its local repair, not yet the full-model cause or C0 acceptance.
- New attempt wide-bootstrap-router-bound/ has a separate prospective plan
  retaining all original exactness and performance thresholds. Repair only
  these four host dispatch dimensions; short schedules remain identical and
  all three edited source files restore completely to accepted 4ced6ee8.
  Build and the full frozen validation queue must pass before any timing.

## 2026-09-12 — Select native wide bootstrap after current-profile acceptance

- The unchanged 4ced6ee8 runtime passes the new normal-output profiled 32K/128
  run and all 16,677,120 FP32 words. Strict audit accepts every parent packet
  and tail/attention interval. Recorded stage GPU total is 196.360s;
  bootstrap ingress is 31.943s (16.27%), routed gate/up 29.092s, attention
  output 21.095s and routed down 16.134s. These are diagnostic costs only.
- Selected substantial scheduling work: one 4K native layers-0/1/2 bootstrap
  batch per continuation chunk instead of sixty-four 64-row calls. Initial
  prefix, 2K and legacy callers keep 32/64-row schedules; overall frontiers
  above 32K keep 64-row bootstrap. No imported shader arithmetic changes.
- Prospective complete-milestone criteria were written before candidate code
  or execution in .work/autonomous-prefill-a46dfe8b/wide-bootstrap/plan.json:
  source/restoration/build, 12K internal checks, full normal and profiled
  32K/128 C0, >=10% target diagnostic reduction raw and control-normalized,
  then two warm-ups plus five alternating unprofiled pairs with preset
  prefill/decode/RSS retention gates. No favorable early stopping.
- Narrow implementation touches metal.rs, long_prefill.rs and metal_shim.m.
  Explicit native-only guards admit 4096 rows. Position arrays cover 4096
  rows and 1024 compressor positions. The raw-cache conversion writes only
  the final aligned 128 rows for a wide batch, retaining old short-row writes.
  Existing four-row compressor refresh and all math kernels remain unchanged.
- First build failed before model execution because raw_cache_target_row was
  removed despite result metadata consumers. Restored the constant; retained
  failed-build log/source hashes/patch and complete inverse-edit provenance.
  Corrected release tests and forced native rebuild are active. No candidate
  correctness or performance result exists yet. Continue through the frozen
  milestones and then directly to the next project milestone.

## 2026-09-12 — Autonomous milestone sequence authorized

- User explicitly requested continuing milestone after milestone without
  stopping at individual milestones, until the project is complete. This
  supersedes the previous per-milestone handoff/stop convention. Preserve all
  C0, oracle, privacy, publication and evidence contracts.
- Resume verified at a46dfe8b with unchanged retained runtime 4ced6ee8 and only
  the three pre-existing untracked paths. Correct checkout remains
  /Users/deathcodevision/dev/ds4-rust-star-bootstrap; app cwd ds4 is different.
- First next milestone refreshes the current 32K bottleneck profile, then
  carries the largest feasible C0-preserving opportunity through implementation,
  exact validation and prospective whole-engine acceptance. A rejected
  hypothesis leads to the next justified milestone rather than ending the task.
- Prospective single-run diagnostic plan, frozen script, source/executable/input
  verification and privacy-filtered host metadata are under
  rust-star/.work/autonomous-prefill-a46dfe8b/. Full model, prompt, oracle and
  all 42 source pins pass. Current full-output profiled 32K/128 run started
  2026-09-12T04:54:59 UTC. Use active-run.json and execution.log to resume;
  never launch a duplicate GPU job. Do not edit frozen sources/scripts during it.
- No arbitrary performance percentage defines project completion. The charter
  calls the broader research ongoing; keep remaining initial-runtime contracts
  explicit and do not declare the entire project complete after one gain.
  The performance-first priority still applies: do not blindly resume the
  interrupted 256K campaign. Larger-frontier work must follow a justified,
  recorded milestone plan. No push/publication is added by this instruction.

## 2026-09-11 — Complete attention-output whole-engine milestone

- Runtime candidate 4ced6ee8 is retained. Verified portable report:
  rust-star/reports/attention-output-engine-4ced6ee8/README.md.
- Same-arena A/A and fresh candidate/current-control screens pass all gates
  and 53,552,873,472 FP32 comparisons. Every candidate pair improves the full
  isolated chain; separate AB/BA medians are 33.919–34.902% lower. The older
  failed two-arena study remains immutable and is not retroactively accepted.
- Narrow integration, 320 release tests, formatting/build, actual production
  Metal assembly/pipelines and complete-host inverse-edit proof pass. Full
  32K/128 C0 matches all 16,677,120 FP32 words; prefix/decode/legacy remain intact.
- Two warm-ups and five alternating unprofiled engine pairs all execute and
  preserve the exact oracle transcript. Median paired prefill latency changes
  -4.329%, steady decode -0.190%, peak RSS +0.001% and
  first-step latency -1.966%. Role-median prefill is
  262.632s baseline / 251.174s candidate.
  Every raw pair and preset criterion is retained in the report; no discarded
  pair, discretionary repeat, threshold revision or statistical-significance claim.
- This is a measured Rust-versus-Rust 32K decision against frozen 9d35953f;
  it does not update the DwarfStar comparison or establish higher-frontier C0.
  Extra FP16 scratch is 320 MiB with existing transient ownership.
- Full model/prompt/oracle/both-executable and all 42 tested-source pins pass
  after execution. The portable verifier recomputes source assemblies/scope,
  isolated intervals/gates, full C0 receipts and all whole-engine statistics.
  All GPU work is finished; 256K remains paused and nothing is pushed.

## 2026-09-11 — Attention-output timing interim checkpoint

- Five of the twelve planned normal-engine processes have completed: two
  excluded warm-ups, pair 1 A/B, and pair 2 B. Pair 2 A is active. The frozen
  runner queues the remaining pairs automatically; do not launch a duplicate.
- Pair 1 prefill is 262.632459s baseline / 251.745201s candidate, a 4.145435%
  paired latency decrease. Steady decode is 1.246800% lower in this pair.
  Pair 2 candidate prefill is 250.759300s; its baseline is still pending.
- Completed records pass the existing measurement adapter. No process failure
  or retry has occurred. Observed swap remains 33.25 MiB, with no recorded
  thermal/performance warning. These are interim observations, not retention.
- Authoritative live evidence and active process identity remain under
  `rust-star/.work/attention-output-milestone-08ebd1aa/engine/` in
  `progress.json`, `active-run.json`, and per-run directories. Frozen plans,
  engine-runner.py and engine-audit.py must remain unchanged during execution.
- Finish all five pairs, final full-input/source/binary hashes, portable report
  verification, and the preset keep-or-restore decision before closing this
  milestone. The 256K campaign stays paused; no publication is authorized.

## 2026-09-11 — Full 32K C0 passes; begin whole-engine milestone

- Candidate 4ced6ee8 passes full 32K prefill and 128 decode steps: all
  16,677,120 FP32 logit values and generated tokens are exact. The fresh
  correctness process completes in 292.181s; this is not a speed measurement.
- Continue to the prospectively declared engine-measure block: baseline and
  candidate warm-ups, then five alternating fresh-process pairs at 32K/128.
  Both correctness collection paths and all profiling are disabled. The strict
  existing adapter checks normal timing, residency and transcript contracts.
- Keep all 42 runtime inputs, both executables, plans and runner/auditor files
  frozen. Report every run and decide from whole-prefill, decode and RSS gates.
  No additional profile campaign, source tuning, push or 256K resumption.

## 2026-09-11 — Freeze 4ced6ee8 and start full 32K C0

- All 320 release host tests, formatting, release build and production Metal
  pipeline compilation pass. Final assembly comparison matches the actual
  build-generated Objective-C source string exactly. Complete host restoration
  and all unchanged-source checks pass; all 42 source hashes match the commit.
- Candidate executable SHA-256 is
  2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4.
  Baseline remains frozen Rust 9d35953f. Full model/prompt/oracle/baseline pins
  pass before inference. Candidate full 32K/128 output-only C0 starts at
  2026-09-11T19:14:35Z; source files remain frozen throughout all live runs.
- Continue to the already declared unprofiled engine block only after the
  complete output gate passes. C0 harness elapsed time is not engine speed.

## 2026-09-11 — Candidate passes neutral screen; integrate for full 32K decision

- Both candidate-plus-A/A processes pass all predeclared criteria. Every one
  of 72 candidate pairs improves; separate-order medians are 33.92–34.90%
  lower. All six contemporary A/A cases pass every null gate. The screen adds
  35,232,153,600 exact FP32 comparisons to 18,320,719,872 A/A-only comparisons.
- Integrate only native continuation chunks in the shared tail host function:
  preconvert heads and grouped-low outputs using the existing conversion
  pipeline; select existing grouped FP16 RHS and a new instantiation of the
  unchanged dense template. Preserve evaluation order, original map and HC.
  Two transient buffers reuse existing ownership, adding 320 MiB at 4096 rows.
- Complete-host inverse-edit audit restores the frozen original exactly, and
  the new shader is the tested instantiation with symbol renaming only.
  The exact production-assembled MoE library and both pipelines compile.
  Release tests/build and fresh full-input preflight are running.
- Prospective engine-plan.json requires full 32K/128 C0, then two warm-ups and
  five alternating unprofiled fresh-process pairs against frozen Rust 9d35953f.
  Preset retention requires all five prefill pairs faster, both order medians
  below .99, bounded decode regression and memory cost. No 256K or push.

## 2026-09-11 — Same-arena control milestone passes

- Both predeclared fresh A/A-only processes pass all six case/session null
  screens: temporal, separate AB/BA, each active-arena subset, MAD and pair
  dispersion. All 18,320,719,872 FP32 comparisons are exact against protected
  references and match prior frozen checksums. No failed run or retuning.
- Continue directly to the two prospective candidate-plus-contemporary-A/A
  processes using the identical frozen harness/library. Runtime sources remain
  unchanged until all candidate and current-control criteria pass.

## 2026-09-11 — Start attention-output decision through meaningful milestones

- User expands scope to resolve the measurement control, conditionally validate
  the candidate, integrate it, pass full 32K C0, and reach a bounded unprofiled
  whole-prefill decision in this task. 256K remains paused; no push.
- Prospective umbrella plan: rust-star/.work/attention-output-milestone-08ebd1aa/plan.json.
  Two A/A-only fresh processes come first. Both roles now use identical active
  buffers with complete per-role comparison to a protected original reference.
  Both physical arenas and both role orders are balanced. Existing 1% role,
  temporal and MAD gates remain; replace the inapplicable within-pair physical
  comparison with separate role-neutrality gates in each active arena subset.
- No retrospective correction of the previous failed study. No repeat or
  retuning after a failed stage; dependent milestones require all earlier gates.

## 2026-09-11 — Half-input attention output is exact; A/A blocks selection

- Portable report rust-star/reports/attention-output-half-input-038dded4/README.md
  verifies 50 artifacts, source assembly/template instantiations/ABI, frozen
  plans, edge receipts and all 2,016 timing intervals. Runtime 9d35953f and
  all 41 source inputs remain unchanged; no model was loaded or code integrated.
- All 16 edge cases pass 172,605,440 FP32 words, CPU half conversions (including
  subnormals/ties), map entries, guards and input hashes. The main processes
  add 19,025,362,944 exact FP32 comparisons across low/dense/HC outputs: total
  19,197,968,384. Checksums and inputs repeat across both fresh processes.
- All 72 candidate pairs improve. All six case/session combinations satisfy
  the preset separate AB/BA candidate criteria, with 34.29–35.09% lower median
  complete attention-output chain times. Both added conversion passes are
  timed. The candidate uses 320 MiB additional FP16 scratch at 4096 rows;
  buffer allocation is outside repeated timing. No new quantization arithmetic
  is introduced: current projection kernels already round RHS loads to FP16.
- Every A/A case fails physical-buffer and dispersion-MAD gates: physical A/B
  medians show 1.99–2.59% differences, and first/second MAD is 2.07–2.56%, above
  1% limits. Session 1/case 2 also fails AB/BA and within-3% count criteria.
  Overall candidate selection is FALSE despite the large consistent signal.
  Do not retroactively relax/normalize the failed controls or infer engine gain.
- The observed effect tracks buffer-set identity, not a proven cache/clock/page
  cause. Keep the promising exact candidate for a separate bounded A/A-only
  same-buffer crossover with a protected exact reference. Establish neutrality
  before repeating the candidate under a fresh plan or integrating runtime.
- Main processes finish in 142.612s/143.597s; no failed execution, retry or
  discarded pair. Swap stays 33.25MB and no thermal warning is recorded. All
  frozen files/executable and runtime hashes pass final checks; all GPU work
  is finished. No push, release, full-model C0 or 256K resumption in this study.

## 2026-09-11 — Half-input attention-output edge gates pass

- All 16 edge cases pass 172,605,440 exact FP32 comparisons across grouped
  projection, dense projection and HC output. Both orders agree, CPU FP16
  conversion checks (including subnormal and tie cases) and every group-map
  entry pass. All guards and immutable-input hashes hold. Edge process ends
  normally in 9.457s; no failed attempt or retry.
- Freeze the unchanged tested harness/library and prospective plan for the
  two main processes. Each uses three 4096-row seeded cases, full-chain timing
  with both conversion costs, balanced AB/BA and physical arena assignments,
  and contemporaneous A/A. Runtime sources remain unchanged; no model load.

## 2026-09-11 — Start bounded attention-output half-input study

- User authorizes the next profile-led target after 038dded4. Current native
  run-skip runtime 9d35953f and all 41 source inputs stay unchanged. Work
  rust-star/.work/attention-output-half-input-038dded4; no model load or full
  engine benchmark in this standalone study.
- Source inspection finds two Q8 matrix projections repeatedly casting FP32
  RHS tiles to FP16. Test preconversion once per RHS, using the unchanged
  grouped half-input kernel and a half-input instantiation of the unchanged
  dense template. Keep the group map, arithmetic, accumulator types and HC
  expansion intact. Full timed chain includes both added conversion passes.
- At 4096 rows, extra FP16 scratch is 320 MiB. Logical tiled RHS-plus-conversion
  traffic changes from 16 GiB to 8.9375 GiB; this is not measured DRAM traffic
  and ignores caches, weights and outputs. Do not assume bandwidth savings
  translate to latency. Baseline real attention-output attribution was 35.076s.
- Host build/source/ABI checks pass. Freeze before 16 edge cases (eight row
  counts, two data patterns including FP16 rounding boundaries), with complete
  low/dense/HC output comparisons, CPU conversion and map checks, guards and
  immutable inputs. A failure stops the screen and is preserved.
- If edges pass, run exactly two fresh equal-warm processes with three 4096-row
  seeded cases, 12 candidate and 12 A/A pairs each. Inherit neutral-control and
  separate AB/BA candidate criteria; include one excluded per-stage diagnostic
  chain per variant/case. New-chain A/A neutrality must be demonstrated anew.
  No retuning, extra repeats, runtime integration, push or 256K resumption.

## 2026-09-11 — Native run-skip passes full 32K C0 and real-component selection

- Runtime 9d35953f is retained as a correctness-validated candidate. Normal
  candidate, baseline profile and candidate profile all pass full 32K/128 C0:
  50,031,360 FP32 words in total. Fresh process elapsed times are 312.064s,
  296.182s and 276.177s; these are harness observations, not engine speed data.
- Both profiles pass all 287 attention packets (1,883 component intervals)
  and 287 tail packets (2,583 intervals), complete prefix coverage, strict
  counter ordering and exact parent attribution. All 320 release host tests,
  format/build checks and both production Metal specializations pass.
- Compressed-attention time is 29.100284s baseline / 27.131982s candidate:
  raw ratio 0.932361 (6.76% lower), normalized aggregate 0.950530 (4.95% lower),
  and median matched normalized ratio 0.947557. All three prospective criteria
  pass; 126/140 normalized packets improve, MAD 0.023656. The unchanged
  control sum is itself 1.91% lower, and normalized chunk aggregates at starts
  4096/12288 slightly regress by 0.89%/0.24%; do not hide this variation.
- The 1.968302s raw component saving is only 0.828% of baseline recorded GPU
  stage time (237.619650s). The pair is profiled/perturbed, fixed baseline-first
  order, and cannot establish whole-prefill gain or DwarfStar parity. Keep the
  candidate for a later bounded engine decision; do not launch a broad campaign.
- Portable report rust-star/reports/32k-compressed-skip-9d35953f/README.md includes
  raw outputs/profiles, frozen plans and build/input pins, full host restoration,
  exact tested-kernel renaming, source assembly, constants and a strict verifier.
  Full model/prompt/oracle/baseline/candidate and all 41 source pins pass final
  checks. All GPU work is finished; swap stayed 33.25MB with no recorded thermal
  warning. No push or 256K resumption. Only 32K is validated for this new kernel.

## 2026-09-11 — Native run-skip passes normal full 32K C0

- Candidate 9d35953f completes normal output-only 32K/128 C0 in 312.064s.
  All 16,677,120 FP32 words and generated tokens match the immutable targets.
  This validates the actual integrated native dispatch with both padded and
  unpadded continuation shapes; elapsed time is harness evidence, not speed.
- The runner starts the fresh baseline component-profiled C0 next, then the
  candidate profile. All sources remain frozen. Final selection awaits both
  complete C0 passes and strict profile auditing against the declared criteria.

## 2026-09-11 — Freeze native run-skip executable and start normal 32K C0

- Runtime commit 9d35953f54d4c1c5c8c85ceebf696d1432025c92 is frozen with 41 runtime/imported
  source hashes. Candidate executable SHA-256:
  631b4844599d814ebfcae6706d26c0854169d4573feed86391c41880f71cfd99.
- All 320 release Rust tests, cargo format check, release build, source
  restoration audit and production Metal compilation for both padding modes
  pass. Fresh full model, prompt, oracle source/executables and baseline pins
  pass before inference. The comparator remains runtime 1711856e.
- Start the declared candidate-32k-output run. Runner proceeds to baseline and
  candidate profiles only after full output C0; failures are retained and stop
  the sequence. Preserve sources throughout the three checks. No engine claim.

## 2026-09-11 — Integrate native compressed run-skip for bounded real-model validation

- User authorizes the next step after isolated candidate 2f06a2e8. Work
  rust-star/.work/32k-compressed-skip-2f06a2e8; comparator is frozen runtime
  1711856e. Copy the tested shader with symbol renaming only into a Rust-owned
  source, appended solely to the attention-ingress library.
- Lazy padded/unpadded NSG8 pipelines select the candidate only for native
  compressed 4096-row continuation chunks. Source restoration proves that
  prefix, legacy, decode, imported kernels and all other host code are intact.
  Production-assembled Metal library and both pipelines compile successfully.
- Predeclare candidate normal 32K/128 C0, then fresh baseline and candidate
  component-profiled C0. Require all 50,031,360 FP32 words across three runs,
  complete strict counter coverage and frozen input/source pins. Diagnostic
  selection requires raw target aggregate, control-normalized aggregate and
  140 matched normalized median ratios all below one. Controls are attention
  output, routed down and shared expert on the same compressed-layer packets.
- This selects only for later engine assessment; a perturbed single profile
  pair cannot establish engine speed. No extra timing, push or 256K work.
  Release build/tests and full model/prompt/oracle pins are underway.

## 2026-09-11 — Empty-block run-skip candidate passes the isolated screen

- Report rust-star/reports/compressed-skip-runs-fbbea533/README.md verifies
  56 artifacts, source/kernel/library/ABI/host-code restoration, frozen inputs,
  edge receipts, traversal traces and all 2,016 recorded timing intervals.
- All 119,150 integer trace cases, 120 GPU edge cases (3,932,160 FP32 words),
  and seven production-shape checks in both processes pass. Timed/initial
  pairs add 23,085,449,216 exact FP32 comparisons; total 23,089,381,376.
  Full mask/pad preparation, guards, immutable checksums and inputs pass.
- Every contemporary A/A gate passes at all six shape/session cases. Maximum
  first/second median deviation is 0.322%, AB/BA 0.454%, physical buffer 0.301%,
  and MAD 0.460%; all 72 null pairs remain within 3% of parity.
- Every candidate shape/session passes the preset AB/BA median <0.99 and 5/6
  faster-per-order criteria. All 72 candidate pairs improve; order-specific
  median kernel-time reductions range 1.14–6.08%. Both fresh processes complete
  normally in 192.964s/191.966s, without retiming or threshold changes.
- Select the frozen run-skip kernel for later narrow native compressed-prefill
  integration and full 32K/128 C0, followed by bounded component validation.
  Runtime 1711856e remains unchanged; no new model/logit C0 or engine gain is
  established. The component was only about 12.04% of diagnostic GPU stage time;
  do not equate this kernel percentage with whole-prefill improvement or launch
  a broad engine campaign before assessing the real-component effect.
- All 40 runtime/imported source pins and frozen executables/inputs pass final
  checks. Host swap is unchanged at 49.25MB with no recorded thermal/performance
  warning. All study GPU work is finished. No push, release or 256K resumption.
  Report and updated project/journal are ready for a local completion commit.

## 2026-09-11 — First run-skip process is exact with faster candidate medians

- Session 0 completes in 192.964s with all seven production shapes exact.
  Candidate/original AB/BA medians at 4096/16384/28672 are
  0.960509/0.970953, 0.947580/0.942933 and 0.988563/0.986260.
  All 36 candidate pairs improve; each order/shape clears the preset 1% margin.
- Preserve these preliminary results. The second process is active in reverse
  shape order; complete its declared run and all contemporary A/A gates before
  a selection decision. No runtime source has changed or model been loaded.

## 2026-09-11 — Run-skip trace and GPU boundary gates pass; start timed screen

- All 119,150 exhaustive/seeded integer cases preserve active-block ordering,
  mask-pointer state and padded K/V addressing, with bounded metadata reads.
  Coverage includes all 0/1/2 marker sequences through length nine, masked and
  unmasked paths, padded/unpadded tails, and seeded cases through 4,096 blocks.
- All 120 GPU edge cases pass 3,932,160 FP32 comparisons in 1.911s, including
  all 15 all-masked positive-zero outputs, long zero runs and both sides of
  four-block/padded boundaries. CPU/GPU preparation, guards and inputs pass.
- The run-skip candidate may increase metadata reads near active blocks; its
  hypothesized gain is fewer loop/padding checks and pointer increments.
  Source restoration and all 40 runtime pins pass. Freeze the validated kernel,
  edge receipts, host/audit/runner and predeclared plan before the two timed
  processes. No speed evidence exists yet and runtime remains untouched.

## 2026-09-11 — Start bounded fully-masked-block run-skip study

- User authorizes the next target. Start fbbea533, runtime 1711856e unchanged;
  work rust-star/.work/compressed-skip-runs-fbbea533/. Clone the original
  compressed FlashAttention template and insert four-zero-block run scanning
  plus scalar leftovers, consolidating mask-pointer advances. Active-block
  arithmetic and the partial padded tail stay byte-for-byte on the old path.
- Before timing, compare exhaustive/seeded integer traversal traces and run
  120 small GPU mask/key-boundary cases (16 query rows, two heads), including
  all-masked output zeros, every-fourth/seven-zero runs and padded tails.
  Preserve every failure. Host builds and source restoration proof pass.
- Then two prospective equal-warm processes, seven correctness shapes and
  three timed shapes, 12 candidate plus 12 A/A pairs per shape. Inherit all
  neutral controls and require candidate AB/BA medians below 0.99 and at least
  5/6 faster pairs per order in every shape/session. No threshold change or
  extra run. No runtime integration, engine claim, push or 256K in this screen.
- Plan and edge/build/source inputs are frozen before executing the new GPU
  kernel. Full output, guard, mask/padding and input checks remain mandatory.

## 2026-09-11 — Fresh controlled head-first screen is exact and rejected

- Report rust-star/reports/compressed-grid-warm-c21ebfda/README.md verifies
  41 artifacts, plan/source/executable bindings, full schedule/exactness
  receipts and all 2,016 recorded GPU intervals. Both fresh processes finish
  normally in 196.957s and 193.930s, with no retries or changed criteria.
- All 23,085,449,216 final FP32 comparisons pass over seven production shapes,
  144 measured pairs and 28 initial warm-up pairs. CPU/GPU preparation, output
  guards, immutable checksums and full input hashes pass. All 40 runtime sources
  remain pinned to 1711856e before/after; no runtime change or new model run.
- All six contemporary A/A cases pass every neutral-protocol gate. Maximum
  first/second median deviation is 0.170%, AB/BA deviation 0.398%, physical
  buffer deviation 0.561%; 71/72 ratios are within 3%, with every case passing.
- Candidate AB/BA medians are 1.36–2.83% slower; only 2/72 pairs improve.
  Every shape/session fails all candidate speed criteria. Reject selection
  for runtime integration or a full-model candidate gate. This scoped warm
  kernel regression is not an engine regression; runtime was never changed.
- Retain both negative reports unchanged and keep the validated method with
  contemporaneous A/A for future work. A next unmeasured lead is eliminating
  traversal/pointer advancement across fully masked cache blocks while keeping
  active-block order and padded-tail behavior exact. No new kernel was written.
- All GPU work is finished. Host swap usage stayed at 57.25MB, with no recorded
  thermal/performance warning; GPU clocks remain unmeasured. No push/release,
  256K resumption or engine performance claim. Evidence and docs are ready for
  a local completion commit.

## 2026-09-11 — First fresh head-first process is exact but fails speed selection

- Session 0 completes in 196.957s with all seven shape checks and 72 measured
  pairs exact. Candidate/original AB/BA medians at chunk starts 4096, 16384,
  28672 are 1.025025/1.025329, 1.016950/1.013566 and 1.022825/1.025588.
  The candidate is slower in every order/shape median and fails selection.
- Contemporary A/A AB/BA medians remain within 0.4% of parity in this process.
  The second predeclared process is active in reverse shape order; preserve
  both complete runs. No candidate integration, extra run or changed criterion.

## 2026-09-11 — Freeze fresh head-first screen with contemporary A/A controls

- User authorizes the next candidate screen. Starting commit c21ebfda;
  work rust-star/.work/compressed-grid-warm-c21ebfda/. Existing head-first
  wrapper is tested prospectively with the validated equal-warm method.
  Preserve the prior failed screen and its older timing evidence unchanged.
- Two fresh processes cover seven production shapes for correctness and three
  for timing. Twelve A/A and twelve candidate pairs per timed shape, balanced
  AB/BA, physical output buffers and interleaved protocol order. Each role gets
  four conditioning plus three queued measured dispatches. No runtime edits.
- All inherited A/A null gates must pass at all six shape/session cases. For
  candidate selection, every AB and BA median must be below 0.99 with at least
  5/6 faster pairs per order at every shape/session. No adjustment, discarded
  pair, favorable repeat or normalization. This selects only a later C0 study.
- Objective-C build and source equivalence audit pass. Original arithmetic,
  masks, ABI and CPU/GPU/output guards are unchanged; all 40 runtime sources
  remain pinned. Target recheck: M1 Ultra 48 GPU cores, 128GB memory.
  Plan/scripts/binary are frozen before GPU execution. No model, engine pairs,
  256K, push or packaged release in this standalone screening task.

## 2026-09-11 — Equal-warm A/A timing protocol passes both fresh processes

- Report rust-star/reports/compressed-aa-protocol-a6bf9e24/README.md verifies
  all 39 artifacts, frozen source/plan/run bindings and independent statistics.
  All 20,937,965,568 final-output FP32 comparisons pass across 144 measured
  pairs and 12 initial warm-up pairs; full preparation/guard/input checks pass.
- All six equal-warm shape/session cases pass every predeclared null gate.
  First/second medians range 0.998509..1.004863; AB/BA medians are within
  0.487% of parity, physical-buffer medians within 0.324%, and all 72 warmed
  pair ratios within 3%. The contemporary balanced legacy-style control has
  first/second penalties 6.60–15.28%, failing every shape/session null screen.
- Both original processes finish normally in 117.988s/116.902s. No repeat,
  threshold adjustment or discarded pair. Source checks pass before/after;
  runtime 1711856e and all 40 runtime/imported sources remain unchanged.
- Retain four conditioning dispatches plus three queued measured samples per
  role, with balanced execution/physical-buffer order, as a scoped isolated
  protocol. Future candidate screens must include contemporaneous A/A and
  succeed separately in both orders. Warmth, submission and averaging changed
  together; cache versus GPU clock causality remains unmeasured. This is not
  normal-engine timing or a retroactive correction of the failed candidate.
- All GPU work is finished. No runtime integration, model reload, new Rust
  build/test campaign, engine gain, push/release or 256K resumption. Evidence
  and journal/project updates are ready for a local completion commit.

## 2026-09-11 — Predeclare bounded compressed-attention A/A mitigation

- User authorizes addressing the standalone first/second timing effect. Start
  from a6bf9e24; keep runtime 1711856e and the failed candidate report frozen.
  Work: rust-star/.work/compressed-aa-protocol-a6bf9e24/plan.json.
- Two fresh processes, three production shapes in reversed session order,
  twelve balanced pairs per shape per protocol. Compare balanced legacy timing
  with four identical conditioning dispatches plus three measured samples per
  role, all queued before a host wait; balance role and physical buffer order.
- Predeclare per-shape/per-session neutrality: first/second, AB/BA and physical
  buffer median ratios within 1%, ratio MAD within 1%, and at least 10/12 pairs
  within 3%. Every warmed case must pass. No retuning or favorable retry.
- Source audit and Objective-C build pass. GPU runs have not started. Preserve
  full output, checksum, guard, CPU/GPU preparation and input integrity checks.
  This compound mitigation cannot identify cache versus GPU clock causality.
  No runtime change, model load, candidate retest, engine claim, push or 256K.

## 2026-09-11 — Attention breakdown and bounded screen complete

- Report rust-star/reports/32k-attention-1711856e/README.md verifies all 74
  artifacts, full-model output receipts, both component packet families,
  source/build/input bindings, standalone exactness and the separate A/A
  control. Hardware recheck confirms Apple M1 Ultra with 48 GPU cores.
- Retain runtime 1711856e's diagnostic-only attention profiler. Full 32K/128 C0,
  all 320 Rust tests, release build, source audits and final pins pass. Largest
  diagnostic components are compressed attention 26.158s, indexed attention
  18.197s and indexer scoring 11.389s; acknowledge encoder perturbation/gaps.
- Do not promote the head-first wrapper. It is exact in isolation but fails
  all three predeclared speed screens. The A/A control establishes first/second
  order sensitivity in this specific standalone harness; cache versus GPU
  clock effects remain unresolved. No correction is applied retroactively.
- All GPU work is finished. Sources and frozen executable remain unchanged.
  The earlier routed grid candidate stays correctness-only; 256K stays paused.
  Commit evidence and updated project state locally; no push or release.

## 2026-09-11 — Compressed-attention grid screen fails; A/A explains order bias

- The standalone head-first grid wrapper matches all 4,697,620,480 FP32 words
  across seven production chunk shapes, including padded/unpadded KV, CPU/GPU
  mask-block/padding checks, output guards and input immutability. All 21 timed
  pairs and every warm-up are retained. Process time 26.151s; no model loaded.
- Candidate paired medians at chunk starts 4096/16384/28672 are
  0.939445/1.008735/0.988854, with 4/7, 3/7 and 4/7 faster pairs. Every shape
  fails the predeclared speed screen. No kernel integration or engine timing.
- Every BA pair has the candidate slower, while 11/12 AB pairs improve.
  Predeclared a separate A/A null control after observing that concern, using
  the identical original pipeline object and original grid in both roles.
  It also passes 4,697,620,480 FP32 comparisons, in 25.154s. First/second
  median ratios are 1.069804/1.032649/1.036785. This confirms order sensitivity
  in this standalone timing setup; it does not establish the cache/clock cause
  or retroactively normalize/accept the failed candidate. All evidence stays.
- Retain the diagnostic profiler, not the head-first optimization. Runtime
  1711856e and all 40 sources remain frozen; full model/prompt/oracle pins pass
  after C0, and both standalone freezes/source/binary checks pass afterward.
  No GPU job remains active. Portable report and final docs are next.

## 2026-09-11 — Full 32K attention profile passes; select compressed attention

- Diagnostic 1711856e passes full 32K/128 C0: 16,677,120 FP32 words, 287
  attention packets / 1,883 intervals and 287 tail packets / 2,583 intervals.
  Component ranking: compressed attention 26.157594s, indexed attention
  18.197261s, indexer scores 11.388780s, top-k selection 1.600754s. All other
  attention components are below one second each.
- Profiling remains perturbing: compressed attention parent rises from the
  previous unsplit-attention profile's 23.122070s to 26.510443s. Counter-span
  coverage ranges 82.10–99.94%, median 99.84%; preserve every raw packet.
  These are diagnostic attributions, not production engine measurements.
- Bind one standalone compressed FlashAttention grid-order experiment before
  GPU execution: transpose head/query dispatch traversal with a wrapper into
  the unchanged original arithmetic. Check all seven native 32K chunk shapes,
  padded/unpadded KV, full finite bitwise outputs and guards. Time three shapes
  with two warmups and seven alternating pairs each. Runtime stays frozen.
- Work/plan: 32k-attention-breakdown-ce94f3fb/compressed-grid-probe/. No model
  reload, new frontier, broad engine campaign or push. Final input pins are
  being rechecked before the isolated probe starts.

## 2026-09-11 — Freeze attention diagnostic 1711856e and start 32K C0

- Release build passes. Frozen source 1711856ea309f70877102d15facb4e1243a9fdcc;
  executable SHA-256 a174c1e06ff399056b5946bbac1a34583e43ec7aaac0b078c280e4f04a81f3c8.
  All 40 runtime/imported-source hashes match the committed tree. Plan, tests,
  build, preflight, source audit and toolchain identities are bound in the
  candidate manifest.
- Start only attention-32k-profiled, requiring full 32K/128 FP32 C0 and strict
  attention/tail packet audits. Keep all sources frozen while it runs. No
  optimization kernel has been selected or measured yet; 256K stays paused.

## 2026-09-11 — Attention instrumentation passes host/source checks

- Added separate 18-sample indexed and 8-sample compressed attention packets.
  Sampling is enabled only by the profiled long-prefill path after prefix
  rebuild; native attention uses fences only when its sample buffer exists.
  A packet must be consumed before another is recorded, and invalid sample
  counts/counters/durations are rejected with the full raw packet retained.
- All 320 release Rust tests pass, including two new attribution/invalid-packet
  tests. Cargo formatting and explicit formatting of included long_prefill.rs
  pass. Source audit restores both original attention executor bodies after
  removing guarded instrumentation; every Metal source hash remains unchanged.
- Fresh full model/prompt/oracle/baseline verification passes. Release binary
  build is in progress. No model run has started and no optimization is chosen
  before the new real-input attribution is available.

## 2026-09-11 — Start bounded 32K attention breakdown

- User authorizes the next profile-led step. Starting commit ce94f3fb, runtime
  candidate 45520435; retain its correctness-only label. Work directory:
  rust-star/.work/32k-attention-breakdown-ce94f3fb/.
- Add diagnostic-only timestamp components to the native indexed/compressed
  continuation attention paths. Preserve normal dispatch, arithmetic and
  fixtures. Existing tail diagnostics remain intact; all intervals must stay
  strictly ordered with diagnostic-only fences.
- Predeclare one full 32K/128 output C0 profile: 147 indexed packets with nine
  components and 140 compressed packets with four (1,883 intervals), plus the
  existing 287 tail packets / 2,583 intervals. Build/test, pin and freeze first.
- Rank the resulting component costs and bind one exact standalone experiment
  before timing it. No broad engine campaign, push or 256K resumption.

## 2026-09-11 — Bounded 32K grid-order validation complete

- Report: rust-star/reports/32k-route-order-45520435/README.md. The portable
  verifier passes all 48 artifact hashes, source/build/input bindings, rejected
  compilation attempt, 50,031,360 exact FP32 comparison receipts and 574
  ordered profile packets. Final full model/prompt/oracle/baseline/candidate
  hashes and all 40 runtime/imported-source hashes match the frozen inputs.
- Retain 45520435 as a correctness-validated candidate. Real 32K routed gate/up
  attribution improves 1.40% raw / 1.42% versus unchanged controls, with matched
  median ratio 0.985016 and 209/287 matched tails improving. This meets the
  predeclared selection rule only; no engine speedup is accepted. Its roughly
  0.43s component saving is too small to materially close the reference gap.
- All three corrected live runs pass. The initial QK_NL compilation failure,
  corrective constant binding and explicit reuse of the 318 host-test results
  remain in the report. Report formatting initially divided by zero for two
  host-only stages; their GPU ratio now reads n/a. No measurement changed.
- No further GPU work is active. Larger frontiers remain unvalidated for this
  candidate; the prior 256K campaign stays paused. Evidence and docs are ready
  for a local commit; no push or packaged product release is requested.

## 2026-09-11 — Three full 32K gates and diagnostic selection pass

- Candidate 45520435 finishes its profile at 08:41:41 UTC. All three declared
  runs pass, totaling 50,031,360 exact FP32 comparisons. Independent audit
  validates both sets of 287 tails / 2,583 ordered component intervals.
- Routed gate/up attribution falls 30.856877s to 30.425503s: ratio 0.986020
  (1.40% lower, 0.431374s saved). Against unchanged controls, aggregate ratio
  is 0.985846 and matched median 0.985016; 209/287 matched tails improve.
  Both predeclared selection criteria pass. Retain as a correctness-validated
  candidate eligible for later engine timing, not an accepted speedup.
- The small gate/up delta is only about 0.20% of baseline summed stage GPU
  duration, so it cannot materially close the reference gap alone. The two
  diagnostic stage totals are 212.158813s and 211.657294s; neither is a valid
  engine throughput measurement. No broader run starts.
- Final full-input/source pin verification and portable evidence packaging
  are pending. Preserve the first candidate's compilation failure and keep
  256K paused.

## 2026-09-11 — Baseline diagnostic passes full 32K C0

- Frozen d6c14fa0 completes the matched profile at 08:37:20 UTC. All
  16,677,120 FP32 output words and all 128 decode steps match oracle-v5.
  Process elapsed time is 273.302s, ineligible as headline engine timing.
- Candidate 45520435 profile starts at 08:37:21 UTC as the final declared
  live run. Strict profile audit and normalized component comparison remain
  pending; no benefit or retain/revert decision is accepted yet.

## 2026-09-11 — Corrected grid candidate passes full normal 32K C0

- Frozen 45520435 completes normal 32K/128 at 08:32:46 UTC: all 16,677,120
  FP32 output words match oracle-v5, including complete prefill and every
  decode row. Process elapsed time is 289.398s; diagnostic RSS is not an
  engine memory or throughput measurement. Executable SHA-256:
  7fe2eb851abe9b5f2b8688bc95a8c998e7c0405fa5814ae34d40366cfa62abdc.
- The matched baseline profile starts at 08:32:46 UTC, followed by the candidate
  profile. Both require the same full-output C0 and strict profile validity.
  No real-input component benefit or engine speedup is accepted yet. Keep the
  corrected source frozen; retain the earlier shader-compilation failure.

## 2026-09-11 — Freeze corrected grid candidate 45520435

- Release build and assembled Metal pipeline compilation pass. Frozen source:
  4552043579bb86b6f94e682e3a154937322169ee; executable SHA-256
  7fe2eb851abe9b5f2b8688bc95a8c998e7c0405fa5814ae34d40366cfa62abdc. All 40 source hashes
  match the committed tree. Unchanged Rust/host tests are explicitly reused
  from 9171518a (318 passed); shader compilation and live runs use this candidate.
- The three full 32K/128 output checks now start sequentially in
  rust-star/.work/prefill-route-order-32k-iq2-bound/. Keep sources frozen,
  preserve the first shader-compilation failure and keep 256K paused.

## 2026-09-11 — Corrected native-tail Metal library compiles

- Standalone compilation of the assembled native library and new pipeline
  passes on Apple M1 Ultra: execution width 32, maximum threadgroup size 512.
  New source audit confirms both IQ2 template arguments are the imported
  original value 16, and the function body still differs only in grid axes.
- Rebuilding the release executable; host test evidence is reused explicitly
  for this Metal-only correction. The failed 9171518a attempt remains unchanged
  and no corrected full-model gate has started yet.

## 2026-09-11 — Correct shader macro scope before full-model validation

- Candidate 9171518a stops during first library compilation: appended shader
  references QK_NL after imported moe.metal undefines it. Exit 1, 22.085s process
  elapsed and 23,887,872-byte peak RSS. No model inference or output comparison
  completes; neither profile starts. Preserve all logs, build/source/input
  identities and failed-candidate-shader.metal in the original work directory.
- Replace only the two IQ2 template arguments with the original fixed value 16.
  Kernel body, arithmetic, host ABI and dispatch remain unchanged. This is a
  source-scope correction, not a numerical relaxation or blind retry.
- Corrected work: rust-star/.work/prefill-route-order-32k-iq2-bound/. Require
  standalone compilation of the assembled Metal library/new pipeline before
  the rebuilt executable starts the same three full 32K C0/diagnostic runs.
  Reuse the unchanged Rust/host test evidence explicitly; all live gates and
  the Metal compile check apply to the corrected frozen candidate.

## 2026-09-11 — Freeze grid candidate 9171518a; three 32K checks start

- All 318 release Rust tests, formatting, source-equivalence audit and release
  build pass. Fresh full-model, prompt, oracle source/executables and baseline
  executable pins pass. Candidate 9171518a8f84b8df30c5e700dfab58cb3c8b6b05 is frozen;
  executable SHA-256 a907449347b9489845f034d479fcf5e46e9772d047dfa47500da5e67a5e89965.
  All 40 runtime/imported-source hashes match its committed tree.
- Run only the declared normal candidate C0 and matched baseline/candidate
  profiled 32K/128 checks, sequentially with host/memory observations. All are
  full-output gates. Keep the candidate frozen until all three complete and
  the strict packet/comparison audit runs. No larger frontier or engine pairs.

## 2026-09-11 — Native-tail grid candidate implemented; freeze pending

- Added rust-star/runtime/src/routed_prefill.metal. Source audit restores the
  accepted fused IQ2 template byte for byte after reversing the two coordinate
  substitutions and function rename. build.rs concatenates it after the imported
  MoE definitions. The shared native-tail executor lazily creates its pipeline
  and transposes the grid only when native_chunk is true. Existing prefix and
  legacy-tile dispatches, root Metal and decode sources remain unchanged.
- Formatting, all 318 release Rust tests and fresh model/prompt/oracle/baseline
  hashes pass. Release executable build is pending; no model gate has started.
- Before any live run, froze the diagnostic decision rule: compare gate/up
  against unchanged attention-output, routed-down and shared-expert work.
  Require both aggregate and median matched-tail normalized ratios below 1.0
  to justify a later engine timing study. Preserve raw absolute times too.
  This is selection for further testing, not acceptance of an engine speedup.

## 2026-09-11 — Start bounded 32K grid-order candidate validation

- User authorizes the next step after the synthetic screen. Starting source is
  653f791c; frozen d6c14fa0 remains the baseline executable. Work directory:
  rust-star/.work/prefill-route-order-32k-653f791c/.
- Isolate the candidate in a Rust-owned shader used only by native 4K
  continuation tails. Keep initial prefill, 32-row diagnostics, decode, root
  Metal sources and every immutable fixture unchanged. No selector flag.
- Predeclare three sequential real-model runs: normal candidate full 32K/128
  output C0, baseline profiled full 32K/128 C0, candidate profiled full 32K/128
  C0. Rehash model/inputs, build/test and freeze before running. Strict ordered
  tail-profile audit remains required; no timestamp or numerical tolerance.
- This is one bounded diagnostic comparison, not headline engine timing. Use
  component attribution and unchanged stages to decide whether a later timing
  campaign is worthwhile. Preserve every failure; no blind retry. The 256K
  campaign remains paused and no push is authorized.

## 2026-09-11 — Bounded routed-prefill grid-order screen passes

- Completed the declared standalone experiment on the M1 Ultra in 10.138s.
  Report: rust-star/reports/prefill-route-order-d6c14fa0/README.md. All 20
  packaged artifact hashes and the independent arithmetic/coverage/statistics
  audit pass. The first preparation assertion failure is retained; no frozen
  GPU attempt failed or was retried, and no timed sample was discarded.
- Transposing only the work/output grid axes preserves every compared FP16
  word: 2,270,429,184 comparisons across 64 executions / eight cases. Actual
  GPU routing maps match independent CPU enumeration; finite/nondegenerate
  baseline outputs and destination guard bytes pass. Synthetic inputs/weights
  use all 256 experts and six distinct routes, with empty/partial/full tiles.
- Seven alternating pairs per timed case, after two warmups per variant:
  2K balanced median paired GPU ratio 0.992044 (5/7 faster), 4K balanced
  0.912801 (7/7), 4K skewed 0.980903 (7/7). Both 4K cases meet the prebound
  screening criteria. Batch sizes are kernel input rows, not full-model context
  measurements. Peak process RSS is 1.555GiB. Host snapshots are retained.
- This is a modest candidate, not closure of the reference gap. Routed gate/up
  contributes 14.52% of recorded stage GPU time in the accepted 32K profile.
  Applying the synthetic 4K savings unchanged would save roughly 0.28–1.27%
  of summed stage GPU time; this is extrapolation, not engine measurement.
- No production optimization is accepted. All 39 frozen source hashes and the
  frozen d6c14fa0 executable still match; production runtime/imported shaders
  are unchanged. No real model was loaded, no new full-output C0 or engine
  throughput result is claimed, and no 256K/retained campaign was restarted.
- Next bounded step: isolate this mapping in a Rust-owned candidate shader,
  run the existing full 32K output C0 gate, and collect a matched lower-context
  diagnostic before deciding on a larger engine timing campaign. Preserve
  accepted production and the paused 256K evidence; no push.

## 2026-09-11 — Begin bounded lower-context routed-prefill experiment

- User authorizes the next step after recovery. Keep the 256K campaign paused.
  Work stays at d6c14fa0; runtime and imported production kernels are unchanged.
- Rehashed the latest accepted 32K component profile and audit from
  128k-score64-b6b0f125. Tail GPU time is 86.851s, with 32.201s attributed to
  attention output and 30.598s to routed gate/up. These perturbed estimates
  partition the tail; they are not separate engine throughput measurements.
- Freeze one hypothesis: transpose only the fused IQ2_XXS gate/up work/output
  grid axes. Preserve the 64x32 MMA tile, arithmetic, activation and stores.
  Baseline shader matches accepted native b6b0f125. Initial preparation rejected
  an incorrect assumption of textual identity with the newer oracle-v5 template;
  no build/GPU run occurred, and setup-failure.json preserves this correction.
- Evidence: rust-star/.work/prefill-route-order-d6c14fa0/. Synthetic controls
  cover partial 32-row work tiles; fixed timing cases use 2K/4K batches,
  256 experts, six routes and balanced/skewed routing. Require exact FP16 output
  words after every execution, finite values and guard bytes. Use two warmups
  per variant then seven alternating pairs per timed case.
- Advance only if both 4K cases improve on the paired median with at least five
  faster pairs each and every exact check passes. This is a kernel hypothesis
  screen, not full-model C0 or an engine speed claim. No broad model sweep or push.

## 2026-09-11 — Recover context in replacement session

- User reports that the previous conversation stopped persisting new messages;
  resume from the project charter and this journal. The application persistence
  issue is user-reported and was not investigated in this repository.
- Confirmed the correct worktree is /Users/deathcodevision/dev/ds4-rust-star-bootstrap,
  branch agent/rust-star-bootstrap at d6c14fa0. The parent ds4 checkout remains
  on codex/dspark-observability-0. Existing uncommitted project/journal edits
  preserve the user's decision to stop the long-context campaign.
- Rehashed all 39 frozen runtime/imported-source files and all four artifacts
  bound by interrupted-attempt.json: every hash matches. Oracle-v5 worktree
  HEAD remains ccbd16d36895ef9fe8edf7f793014988ff1ab7f4. No matching model runner
  or observer process was found in the current process inventory. No executable
  or full-model rehash, live C0 test, or new benchmark was performed this session.
- Read the manual ledger and existing 32K reference-comparison / 128K score-tile
  reports. Their historical timings do not establish current-candidate parity.
  Keep 256K paused and preserve both rejected attempts and interrupted evidence.
- Next: use the latest accepted lower-context profile to select one bounded
  prefill bottleneck experiment, preserving oracle-v5 C0. No broad validation
  campaign, performance acceptance, publication, or push is implied by recovery.

## 2026-09-10 — Stop 256K validation after user challenges performance priority

- The user questioned spending the day on higher-context checks while prefill
  remains slower at lower contexts. Stop the expensive validation sequence;
  do not resume 256K or retained cycles automatically. The next useful work is
  focused prefill performance investigation at a lower context, using existing
  profiles and a short experiment before another broad validation campaign.
- Validation group 6856 was terminated with SIGTERM; runner exited 143. Frozen
  d6c14fa0 passed eight gates, through full 128K/128 C0 and 192K prefill. Its
  fresh 256K run was interrupted after 151552 prefill tokens,
  before any decode comparison. Neither retained gate started on this candidate.
  This is a user-driven interruption, not a correctness failure or acceptance.
- Evidence: rust-star/.work/frontier256-handoff-bounded/interrupted-attempt.json
  binds preserved logs/progress/build metadata. Candidate executable and runtime
  source hashes still match the freeze. Both earlier rejected attempts remain
  preserved. Final full-model rehash, acceptance audit, packaged report and
  completion commit were not performed. No 256K performance result exists.
- Latest direct Rust/oracle paired prefill latency ratios remain 1.72589 at
  32K (32k-tail-89e32745) and 1.342774 at 128K (128k-measure-ef37e566).
  The subsequent 8.28% prefill improvement (128k-score64-b6b0f125) is against
  the previous Rust implementation and does not establish reference parity.
- Validation and both observers have exited and their sessions are drained.
  Preserve completed host/memory observations. No model workload remains
  running, no new benchmark is started, and no push.

## 2026-09-10 — Freeze bounded handoff d6c14fa0 and start full revalidation

- Candidate d6c14fa0918a5130da72aa7be6e56b82696fb8b0 is frozen. Executable
  SHA-256: 8bdcf022811eca2c2cf0b5c7aca10bc51705a1a18d988129cffbdfd43bb58d11.
  All 318 Rust/94 Python tests, release build, formatting, CLI and current
  input/host pins pass. Production handoff/reset regression is independently
  verified. Runtime and imported shaders remain frozen throughout validation.
- Host and memory observers start at 17:07:52 UTC before the first gate. The
  baseline is after prior rejection, regression and compilation, with existing
  swap; it is not a fresh boot. Preserve complete whole-system observations.
- An initial runner setup check catches missing copied canonical reference
  folders before any model gate starts. Preserve its error, restore/re-hash
  unchanged references, and start the same candidate's first full gate run.
  The 2K control passes and 8K is active at 17:13 UTC. All eleven gates must
  pass, including complete retained 128K and 256K cycles; no acceptance yet.
- Active evidence is rust-star/.work/frontier256-handoff-bounded/. Both prior
  candidates and corrections remain preserved. Independent full audit, final
  input verification, observer completion, report/docs and local completion
  commit remain pending. No push, performance pairs or new milestone.

## 2026-09-10 — Bound long-context handoff resources; full-size copy regression passes

- Above 128K, handoff now submits one layer per command, waits for completion,
  and removes that layer's history and executor-slot aliases. A per-command
  autorelease pool retires Metal's strong resource references. All 229 copy
  ranges, shape checks and destination keys match the rejected predecessor.
  The <=128K one-command schedule remains; model mapping, view cache and
  attached residency are preserved. Rust validates 43 versus one commands.
- The production handoff and request reset pass two full-size synthetic cycles
  at each of 128K and 256K: 2,728,804,352 transferred FP32 words and all
  untouched destination tails match deterministic GPU-filled values. Weak
  ownership checks prove completed layer sources retire before the next
  command, including slot aliases. Final cleanup and model/residency identity
  through reset pass. The synthetic residency allocation is small; full-model
  retained C0 is still required and is not claimed by this regression.
- Evidence is in rust-star/.work/frontier256-handoff-bounded/. Both prior
  rejected attempts are preserved with source/build hashes and raw logs.
  Observers and model runner from the prior attempt are stopped and drained.
  All 318 Rust and 94 Python tests and the input/host preflight pass. The
  release executable is building, followed by freezing and all eleven gates.
  Final acceptance, report,
  documentation and local completion commit remain pending. No push or pairs.

## 2026-09-10 — Reject retained 256K second-cycle handoff after Metal memory failure

- Frozen 0d5f50ff passes ten gates, including fresh 256K full C0 and both retained
  128K cycles. The first retained 256K cycle also passes all 16,677,120 words.
  The second cycle passes its 129,280-word prefill comparison, then Metal rejects
  the prefill-to-decode blit with Insufficient Memory
  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory). No second-cycle decode
  step runs, and the retained 256K gate is not accepted.
- Failure finishes at 16:49:50.847831 UTC, after 7973.811s for the retained
  process, with 109,890,371,584 bytes peak RSS. Complete logs, ten successful
  outputs, the first retained-256K trace, source/build hashes and all system
  observations remain in rust-star/.work/frontier256-decode-nsg/. Its
  failed-attempt.json verifies that the frozen executable and every source
  remain unchanged. System compression and multi-GiB swap occur; these are
  whole-system counters and do not prove per-process attribution.
- Static handoff currently encodes all 43 layers' FullKv/compressed source and
  destination buffers in one command, and releases prefill state only after
  it completes. On the second cycle, the model residency set remains attached
  from the first decode. Investigate a bounded per-layer GPU ownership transfer
  at long contexts, preserving model/context/residency reuse and exact copies.
  Do not blindly retry the unchanged candidate or weaken the retained gate.
- The model runner has exited and the observers are being stopped. Preserve
  this second rejected attempt alongside c1047756. No final acceptance report
  or completion documentation was generated. Fix and regress the memory
  boundary, freeze a new candidate and rerun required gates before acceptance.

## 2026-09-10 — Retained 128K reset and reuse pass; final 256K gate active

- Retained 128K completes at 2026-09-10T14:36:57.035948+00:00: both complete traces match
  all 16,677,120 FP32 words each. Same process, model mapping, Metal context,
  request reset, model-warm reuse and retained residency checks pass. The first
  residency-reused flag is false and the second is true. Diagnostic elapsed
  time is 2636.090s, with 94.710GiB peak RSS.
- Ten of eleven corrected gates pass. The final retained 256K process starts
  immediately afterward and remains active, requiring two complete traces.
  At 15:23 UTC its first prefill has reached 221,184 tokens; no final output
  acceptance is claimed yet. System compression fluctuates substantially and
  swap remains in use; preserve the complete observer time series.
- Candidate 0d5f50ff remains frozen. Independent final audit, final input hashes,
  report packaging, documentation completion and a local evidence commit remain
  pending. No push or performance pairing is authorized.

## 2026-09-10 — Corrected fresh 256K full-output pass; retained gates active

- Frozen 0d5f50ff completes fresh 256K at 13:53:00 UTC: all 129 output rows,
  16,677,120 FP32 words and 128 decode steps match the accepted oracle-v5
  reference. Diagnostic process time is 4019.475s and peak RSS 23.327GiB.
  This validates the dispatch correction at the previous first-decode failure.
- All nine gates through fresh 256K pass. The explicitly warmed retained 128K
  gate is running, followed by retained 256K. Each requires two complete traces
  with request reset and model/context/residency reuse; neither is accepted yet.
- System swap reaches approximately 2368MiB during corrected fresh 256K and
  compression also occurs. These are whole-system observations, not attributable
  solely to the candidate. Fresh process RSS is not total memory use and does
  not prove an optimization or uncompressed residency. Full time series remain
  in rust-star/.work/frontier256-decode-nsg/.
- Keep the candidate frozen. Finish both retained gates, independent audit,
  final input verification and observation coverage, then package evidence,
  update documentation, commit locally and stop. No push or performance pairs.

## 2026-09-10 — Correct dense decode SIMD-group selection; revalidation pending

- Native gathered decode now follows oracle-v5: one SIMD group through 2048
  keys, two through 4096, four above; pipeline constants, shared memory and
  dispatch dimensions agree. The two additional pipelines are cached lazily.
  Root Metal kernels, prefill arithmetic and RoPE remain unchanged.
- Synthetic GPU regression compiles the actual shim factory and separately
  compiles unmodified oracle-v5 FlashAttention/reduction. All 294,912 FP32 words
  match across nine shapes (128 through 4097 keys). The previous fixed dispatch
  matches all short controls and differs above 2048; at 2176 keys, 26,153 words
  differ. An initial shared-library control and the corrected standalone test
  wrapper are preserved with build/source/result hashes in
  rust-star/.work/frontier256-decode-nsg/.
- 318 release Rust tests, 94 Python tests, formatting and input rehashes pass.
  Release executable build/freeze and all eleven live native gates are next.
  Accepted oracle captures are reused after reference rehashing. The rejected
  c1047756 attempt remains immutable. No new full-model gate has started.
- Candidate 0d5f50ff is frozen; executable SHA-256
  1ace3c5031d832d34eb0064fdfd71b3feeb0a0c3000d568496b4828eb82548c5.
  Release build and CLI checks pass. All eleven native gates are now queued
  sequentially; both observers captured a pre-gate baseline.
- Observe system-wide compression from before the first corrected native gate;
  this is a post-rejection baseline, not a fresh boot. Do not infer total memory
  or uncompressed residency from process RSS. No performance pairing or push.

## 2026-09-10 — Reject first 256K decode attempt; investigate SIMD-group boundary

- Frozen c1047756 passes eight shorter gates and the complete 256K prefill
  comparison, then fails at decode position 262144, logit 0: actual c16ca0ca,
  expected c16ea2a8. No retained gate starts. Preserve the full first attempt in
  rust-star/.work/frontier256-a3010176/, including failed-attempt.json, all raw
  outputs, sources/build hashes, host and memory observations. No blind retry.
- Both observers and the queued validation runner have exited. All four oracle
  captures remain accepted and immutable. No model workload is active.
- The oracle's dense decode FlashAttention selects two SIMD groups when the key
  count exceeds 2048. Rust fixes one SIMD group and 3328 bytes of shared memory.
  At 256K the odd layers have 128 + 2048 = 2176 keys. Verify this policy mismatch
  with a small synthetic GPU regression, then correct the native selection and
  dispatch, freeze a new candidate, and rerun every required live gate.
- The capacity evidence includes system-wide memory compression and small swap
  growth; it does not prove uncompressed residency. Preserve this rejected
  attempt in the final report and do not describe validation as retry-free.

## 2026-09-10 — 256K capacity and correctness milestone started

- User authorizes the bounded next milestone: memory audit, repeatable oracle-v5
  256K outputs, exact native prefill plus 128 decode steps, and retained-session
  reset/reuse validation. Performance pairing remains a later milestone; no push.
- Start from accepted score-tile b6b0f125, completion a3010176. Work directory:
  rust-star/.work/frontier256-a3010176/. Existing evidence and oracle-v5 remain
  immutable. No model process was active at startup; swap is zero, disk 76GiB free.
- The current Q/HC/norm/split transients are already chunk-bounded. Audit the
  context-sized FullKv/compressed histories, sparse selector scratch, device
  limits and decoder adoption before changing the 128K execution guards.
- Capacity probe passes with 27.267GiB of simultaneous modeled histories and
  selected transients; planning estimate is 108.443GiB including history/selector
  growth from the measured 128K peak. Neither is a complete residency bound.
  Input hashes pass. Oracle captures show zero recorded swap and no thermal
  warning.
- Candidate extends external-reference guards to 256K and adds two full-output
  retained diagnostic cycles with actual residency metadata checks. All kernel
  arithmetic, RoPE constants and existing timing-command limits are unchanged.
  All 318 release Rust tests, 94 Python tests, release build and formatting
  checks pass. Candidate c1047756 is frozen with executable SHA-256
  f427ceee303921482d3b4550195ee346cb39367525f98916ae4e4a356481d280;
  all runtime and imported Metal sources are hash-bound. CLI help and mutually
  exclusive diagnostic options pass. First 192K capture passes at 06:31 UTC:
  129,280 finite logits, selected token 5164, 1882.664s, 80.894GiB peak RSS.
  The 192K repeat passes at 07:02 UTC (1837.463s, 80.894GiB peak RSS); both
  full vectors agree byte for byte. First 256K capture passes at 07:51 UTC:
  all 129 finite rows (16,677,120 words), prefill token 7221, last token 100464,
  2987.565s and 80.896GiB peak RSS. The 256K repeat passes at 08:43 UTC:
  3116.494s, 80.896GiB peak RSS, all 16,677,120 output words and greedy tokens
  agree byte for byte. All four captures and the capture runner pass.
  The first six native gates pass from the frozen binary: 2K, 8K (including
  score-tile metadata), 12K internal state, full 32K output, 32K internal and
  32K profiled output. The strict 287-packet/2,583-interval audit passes.
  Fresh full-output 128K control also passes: all 16,677,120 words and 128
  decode steps match. New 192K prefill-only control also passes all 129,280
  logits (token 5164). Eight gates pass; full 256K prefill/decode is active,
  followed by retained 128K/256K. Internal HC remains capped at 24K.
  First observed host swap is 6.12MiB during native controls; no thermal warning
  is recorded. Late fresh-256K RSS variation prompts system-memory inspection:
  an initial vm_stat snapshot reports 20.419GiB compressor storage representing
  25.338GiB of uncompressed pages. The timed series begins at 10:44:55 UTC and
  continues through remaining runs. These are system-wide counters with no
  earlier baseline, not per-process attribution or proof of an uncompressed
  resident working set. Preserve raw counters and the initial snapshot timing
  bounds alongside all existing observations and attempts.

## 2026-09-09 — 128K score-tile optimization accepted; milestone complete

- Retain frozen candidate b6b0f125 against accepted Rust baseline ef37e566.
  Five alternating pairs and both warmups pass on their first attempts; no
  retries, exclusions or concurrent model jobs. The campaign exits 0 at 15:06
  UTC. Independent reconstruction of every native metric, artifact hashes and
  all distributions passes; the unchanged prebound criteria all pass.
- Median paired prefill latency ratio is 0.917193 (8.28% lower), with all five
  pairs faster; candidate/baseline median prefill is 1216.402/1358.855s.
  Median paired steady throughput ratio is 1.010612 (+1.06%); medians are
  15.145/14.860 tokens/s. Median peak RSS is 94.513/94.511GiB, paired ratio
  1.000015. These are Rust-to-Rust development measurements, not an update to
  the Rust-versus-oracle headline. Ratios of paired observations differ from
  ratios of separate medians; all dispersion and every first-step latency are
  preserved. Pair 5 first-step latency is 10.85% slower; median paired latency
  is 3.97% lower. No outlier was removed.
- All 317 Rust tests, 94 Python tests, release build and ten live gates pass.
  Fresh normal and profiled 128K C0 each match all 16,677,120 FP32 words and
  128 decode steps. Retained 32K/128K each verify two exact transcripts,
  request reset and same model/context/residency. Internal HC remains accepted
  through 24K. Strict 32K/128K profiles and seven exact synthetic score shapes
  pass; the 14 isolated timing pairs remain supporting diagnostic evidence.
- Full 128K profiling required diagnostic-only fences after independent Metal
  passes produced overlapping timestamps. The rejected profile, probe limits,
  first freeze-helper failure, six-gate metadata attempt and red writer
  regression remain preserved. The corrected b6b0f125 binary reran all ten
  gates before timing. No timestamp or numerical guard was relaxed.
- Final model/prompt/oracle-source/oracle-executable/CLI hashes match at
  15:08 UTC. The observer exits cleanly; no model process remains. The portable
  report rust-star/reports/128k-score64-b6b0f125/README.md verifies 322 artifacts
  without the model, including frozen sources, native results, oracle row
  receipts, profiles, all paired statistics and the retention verdict.
- Documentation now records the accepted result. Stop at this milestone;
  no push, 256K run, coding-agent corpus or further optimization is started.

## 2026-09-09 — Correct score-kernel report labels before acceptance

- Stopped candidate 7b1a5b58 validation after six successful numerical gates
  (2K, 8K, 12K, 32K output, internal and profiled). The retained-32K gate was
  incomplete and is excluded. No candidate 128K gate or engine timing ran.
- The 8K writer still named the previous score kernel in two metadata fields.
  The strengthened writer regression fails with zero new names versus two
  expected. Correct both labels to rust_star_indexer_scores_tiled64; arithmetic
  and dispatch remain unchanged. Preserve the stopped attempt and regression.
- Rebuild and freeze a new candidate in rust-star/.work/score64-metadata-fix/,
  then rerun all ten gates with explicit kernel metadata assertions. All 317
  release tests now pass, including the previously failing writer regression;
  fresh input verification passes at 08:44 UTC. Release build passes. Candidate
  b6b0f125 is frozen with executable SHA-256
  99f4971e684ed455841065ca597e9ee10055e3a937bbaeb416a9bb9b70800c4c;
  all tracked runtime and 11 imported Metal sources are hash-bound. The ten
  live gates run sequentially from the new executable: seven gates pass through
  retained-32K, including both corrected 8K labels and the strict 32K profile
  audit. Both retained transcripts match with request reset/residency reuse.
  Fresh full-output 128K C0 now passes: all 16,677,120 FP32 words and the
  128-step transcript match oracle-v5. Profiled 128K also passes full-output
  C0 and the strict component audit. All ten gates now pass, including both
  exact retained-128K transcripts and reset/residency checks. The independent
  gate/lifecycle audit and fresh campaign input hashes also pass. Both warmups
  pass on their first attempts. Pairs 1–4 pass both exact transcripts; final pair 5
  baseline passes and its candidate is active (A=accepted Rust, B=candidate) in
  paired-128k-score64-b6b0f125/. Keep the five alternating pairs and prebound
  acceptance plan unchanged. No final performance conclusion exists yet. Acceptance criteria
  and the frozen ef37e566 timed baseline remain unchanged.

## 2026-09-09 — 128K profiling and focused optimization milestone started

- Candidate 7b1a5b58 is frozen (binary SHA-256
  e576a2abd3d1ccf57fd5428f167b896bf6c3dfa92c500511d8f36a30bfece8c4),
  including all tracked runtime and 11 imported Metal source hashes. The ten
  live gates run sequentially in score64-ec5db25e/; the first five pass (2K,
  8K, 12K, complete 32K output and 32K internal). The 32K profile is active. No engine performance result exists yet. An initial freeze-helper
  comparison rejected empty lines left by different comments; the preserved
  pre-model failure was corrected by comparing nonempty code lines. Kernel code
  and C0/timestamp guards were not changed by that correction.
- Fenced diagnostic ec5db25e completed at 08:02 UTC: all 129,280 prefill
  logits and all 128 full decode rows match oracle-v5 exactly. The independent
  audit passes all 1,271 tails / 11,439 intervals. Counter-span coverage is
  99.793–99.943% (median 99.896%), with no rejected packet. This establishes
  full-output 128K prefill profiling, not 128K decoder-stage profiling.
- Observed prefill GPU stage total is 1186.370s: attention 581.775s (49.04%),
  tail 376.485s (31.73%), bootstrap ingress 148.347s. Even-layer attention is
  383.761s; its per-chunk total grows from 3.255s to 21.496s. These are separate
  perturbed diagnostic measurements, not eligible engine throughput results.
- Select the 64-column indexer-score tile hypothesis from this attention
  attribution. Seven isolated shapes match all 167,815,044 score words and
  causal masks. Seven alternating pairs at each large shape give median GPU
  latency ratios 0.660566 (32K shape) and 0.619380 (128K shape), all 14 improved.
  Source/word-count/statistics audit passes; raw pairs and the prebound hypothesis
  are preserved in profile128-9b4d7d0d/score-tile-*. No engine speedup is claimed.
- Integrate only that score tiling change: new Rust-owned indexer_scores.metal,
  64 columns / 256 threads / 20KiB shared staging, same 8x8 dot and head order.
  One host geometry helper covers all three prefix/continuation call sites.
  All 317 release tests and 94 Python tests pass; fresh input hashes pass at
  08:09 UTC. Release executable build also passes. Freeze this candidate and
  run ten live gates before the five engine pairs. The prebound acceptance plan
  requires lower median paired prefill latency with at least four faster pairs,
  steady throughput ratio >=0.97 and peak-RSS ratio <=1.03; report every first
  step and all dispersion. No favorable retries. See optimization-acceptance.json.
  New optimization work directory:
  rust-star/.work/score64-ec5db25e/. Frozen timed baseline remains ef37e566.
- Baseline engine-profile rejected at chunk 57,344/layer 41 after 562.51s:
  shared-expert start precedes routed-sum end. Raw 18-word packet and exit 1 are
  preserved in baseline-profile.log/status.json; no valid profile was emitted.
  A standalone nine-pass Metal probe reproduces non-increasing packets in
  100/100 independent-pass runs and 0/100 fenced runs. This supports legitimate
  independent-pass overlap, not permission to weaken timestamp validation.
- Add diagnostic-only fences between the nine tail profile encoders. Preserve
  strict monotonicity and mark serialization as MTLFence in the profile JSON.
  Normal inference still uses its original single encoder. The external-profile
  CLI extension passed 317 host tests; 317 release tests with fences also pass.
  Full model/prompt/oracle/reference hashes pass fresh preflight at 07:35 UTC.
  Release binary build and five CLI help/path/flag rejection checks pass.
  Freeze the diagnostic source/binary, then run live 128K profiled full-C0.
  Frozen ef37e566 stays the
  normal timed baseline; use the corrected diagnostic binary for attribution.
- Diagnostic ec5db25e is frozen (binary SHA-256
  f6b0c5424792cb5008e34d6588c7ca27da9b3b1031930ad45557d0e1aa7cdc01).
  Its full 128K profiled external-reference run and strict component audit now
  pass, as recorded above. Both raw failed and successful attempts are retained.
- While the baseline profile runs, extend the external-reference diagnostic with
  optional `--profile-json PATH` so the instrumented 128K prefill can be checked
  against every immutable output word. This only enables the existing diagnostic
  counters; no inference optimization has been selected. Build and live checks
  are pending. Preserve both the normal full-C0 gate and this profiled gate.
- User authorized the proposed next milestone from 9b4d7d0d. Work remains on
  agent/rust-star-bootstrap. Work directory: rust-star/.work/profile128-9b4d7d0d/.
- Start with the frozen accepted ef37e566 executable and pinned 128K transcript.
  Capture the existing perturbing engine-profile path, audit all 1,271 native
  tails / 11,439 component intervals and generation counters before using them.
  Keep oracle-v5 and all old reports immutable. No optimization selected yet.
- Use measured attribution to select a focused change. Require host tests/build,
  fresh full 128K C0, shorter output/lifecycle/profile controls, then five pinned
  alternating Rust-versus-Rust pairs with warm-ups and independent audits.
  Preserve failures/rejected hypotheses; retain only supported improvements.
- Keep journal current and complete portable evidence/docs before stopping at
  this milestone. No quota burning, extra model concurrency or push requested.

## 2026-09-09 — 128K measurement and retained-engine milestone COMPLETE

- Candidate ef37e566 passes 317 Rust tests, 94 Python tests, release build,
  six live gates and the 287-tail/2,583-interval 32K profile audit. Fresh full
  128K C0 matches every prefill word and all 128 decode rows. Retained 128K
  completes both exact transcripts with request reset and residency reuse.
- Five alternating fresh-process pairs and two warm-ups pass on first attempts.
  Rust/oracle median prefill 1494.551/1079.569s;
  steady decode 14.578/15.618 tokens/s.
  Median paired prefill latency ratio 1.342774; steady
  throughput ratio 0.971815. This is characterization,
  not a speedup claim. All five samples remain in the report.
- Fresh median peak RSS 92.345/80.895GiB
  Rust/oracle; retained two-cycle peak 94.571GiB is a separate diagnostic.
  These warmed measurements are distinct from prior unwarmed diagnostic RSS
  and are not universal Metal/model memory bounds.
- Independent statistics/artifact audit, portable verifier and final model,
  prompt, oracle source/binary hashes pass. Report:
  rust-star/reports/128k-measure-ef37e566/README.md (167 verified artifacts).
  The earlier 5f3b9672 report-writer failure and red regression are preserved
  under rejected/; the corrected candidate has no failed timed attempts.
- The paired campaign finished September 8 at 22:02 UTC. A later interruption
  stopped the read-only final input hash check before it produced a result;
  the check was resumed September 9 and passed. No paired sample was rerun.
- Fresh and retained measurements now admit the pinned 2K/32K/128K frontiers,
  each with 128 decode steps. Full-logit checks remain independent from timing.
  Retained support covers the fixed transcript/lifecycle, not arbitrary serving;
  live profile acceptance remains 32K and internal HC acceptance remains 24K.
- STOP at this completed milestone. Proposed next: profile the dominant 128K
  transformer-chain prefill cost, validate counters and C0, then select and
  measure a focused optimization. It is not started. 256K and the coding-agent
  corpus remain future work. No quota burning or push performed.

## 2026-09-08 — Four timed 128K pairs pass; final pair active

- Pair 4 BA passes on its first attempt: Rust/oracle prefill
  1,380.256/1,030.604s and steady decode 15.170/15.618 tokens/s. Every native
  transcript remains exact. Four pairs and both warm-ups pass without retries.
- Final pair 5 AB is active. Closeout is queued after its completion: independent
  statistics/artifact and source/tool hash audit, final model/oracle verification,
  observer shutdown and report draft. Portable packaging/verifier, final prose
  review, journal/charter/contracts and final commit remain required.

## 2026-09-08 — Three timed 128K pairs pass; final two remain

- Pair 3 AB passes on its first attempt: Rust/oracle prefill
  1,524.108/1,135.044s and steady decode 14.187/14.590 tokens/s. All three
  native timed transcripts match exactly; no failed or retried measurement.
- Pair 4 BA is active. The first three pairs consistently show higher native
  prefill latency and lower steady decode, but final five-pair distributions,
  artifact/input verification and report/docs are still required. Preserve the
  negative characterization without selecting favorable samples.

## 2026-09-08 — Two timed 128K pairs pass; pair 3 starts

- AB and BA pairs both pass on first attempts. Pair 2 Rust/oracle prefill
  1,536.733/1,133.219s; steady decode 14.190/14.602 tokens/s. All native
  timed transcripts remain exact. These are interim results; three pairs remain.
- Pair 3 AB is active. Current host observations report no thermal/performance
  warnings and stable swap 42.19MiB. No retries, source or plan changes.
- Passive timing in pair 1 assigns about 86% of native prefill wall time to the
  remaining transformer chain. A future 128K profile should identify its dominant
  stages before selecting an optimization; no next milestone is started.

## 2026-09-08 — First timed 128K pair passes; reverse order begins

- AB pair 1 passes on its first attempt, with exact native transcript and
  eligible measurement metadata. Rust/oracle prefill 1,494.551/1,079.569s;
  steady decode 14.578/15.639 tokens/s. These are preliminary single-pair
  observations, not the accepted five-pair summary or an optimization claim.
- Pair 2 BA started at 19:10:50 UTC. Four pairs remain; preserve every result,
  keep source/input pins frozen and finish independent audits/report/docs.

## 2026-09-08 — Both 128K warm-ups pass; first timed pair running

- Oracle and ef37e566 fresh-process warm-ups pass on their first attempts.
  Native exact transcript and collection-disabled eligibility pass. Warm-ups
  remain excluded from all five-pair distributions. Process wall times are
  1,109.361s oracle and 1,512.092s Rust; these are not paired results.
- Timed AB pair 1 started at 18:27 UTC. The fixed five-pair campaign continues
  serially with no retries; all results will be audited before interpretation.
  Host observations still record no thermal/performance warnings, swap 42.19MiB.
- close-evidence.py waits for campaign completion, then independently audits
  pairs, rehashes final model/oracle inputs, stops the observer and renders the
  report draft. Packaging, final review, documentation and commit remain required.

## 2026-09-08 — All six gates pass; 128K paired campaign frozen

- Corrected candidate ef37e566 passes all six live gates and strict lifecycle
  audit: full 128K C0, retained 128K, 32K output, 2K output, retained 32K,
  and profiled 32K. Component audit accepts 287 tails / 2,583 intervals.
- Reverified model, prompt, oracle binary/source and frozen candidate sources.
  Five alternating fresh-process pairs plus one warm-up per engine started at
  17:43 UTC. Plan SHA-256:
  a9b8630fc3be1f1c6dde16cef79b6bc54598337fd231e36b0be5863b4dcfc1e8.
  Work: rust-star/.work/paired-128k-ef37e566/.
- Runtime and measurement tools remain frozen. No paired result yet; preserve
  every attempt and the earlier rejected writer run. Final evidence audit,
  portable report and documentation remain required before completion.

## 2026-09-08 — Corrected retained 128K lifecycle passes

- ef37e566 completes both 128K/128 transcripts and serializes the lifecycle
  report successfully. Both cycles select prefill token 1110, final token 2945,
  transcript checksum 1079188149413743454. Request state resets; model mapping,
  context, views and residency are reused as required. Peak process RSS 94.571GiB.
- Diagnostic first-load prompt including warming: 1,386.939s; retained prompt:
  1,377.238s. Page warming 11.627s and residency work are avoided on reuse, but
  prefill dominates. These are single same-process diagnostics, not paired rates.
- Full 128K C0 and retained 128K now pass. Four remaining 32K/2K/lifecycle/
  profile controls are running, then the lifecycle audit and five paired runs.
  The earlier writer failure remains preserved; no paired measurements yet.

## 2026-09-08 — Corrected writer candidate passes fresh full 128K C0

- Corrected candidate ef37e566, executable SHA-256
  50f59a30144d106564d776a8b7a966d06a3d3b3dbe6c6bc3bf30ad69c81fa752,
  passes 317 Rust tests/build and the unchanged 94 Python tests. Full 128K C0
  repeats exactly: every prefill word and all 128 decode rows match.
- New retained 128K execution is running, followed by four shorter/profile/
  lifecycle controls. Work: rust-star/.work/measure128-writer-fix/.
  The initial writer-admission failure and its red regression test remain in
  rejected/. No timed pair has started. Runtime source remains frozen.

## 2026-09-08 — Retained 128K report rejected by old writer guards

- Preserve failed 5f3b9672 attempt in measure128-43bf0dfa/: full 128K C0 passed;
  retained run exited 1 after 2,763s with "inconsistent measurement metadata".
  No paired processes were started. Validation and campaign stopped immediately.
- Cause found in write_engine_run_json: context allowlist remains 2K/32K and
  prefill-token validation assumes those frontiers. The retained writer invokes
  that validator after both execution cycles return. This is report admission,
  not a recorded full-logit/transcript mismatch; timings were not serialized.
- The new fresh 128K writer test reproduced exactly the same metadata rejection
  without the model. Extended context, prefill-token and full-transcript checks
  in serialization, with fresh/retained regression tests. Freeze a corrected
  candidate and rerun all live acceptance.
  Keep this failed attempt distinct from the new candidate; no favorable retries.

## 2026-09-08 — Measurement candidate passes complete fresh 128K C0

- Frozen 5f3b9672 passes every 128K prefill word and all 128 full-logit decode
  rows against the unchanged repeated oracle-v5 reference. JSON/checksums match
  the prior accepted native result exactly. No failures or retries.
- Timed retained 128K two-cycle execution is active. Remaining 32K output,
  2K, retained 32K and profiled 32K controls follow, then a strict lifecycle
  audit and the predeclared five-pair fresh-process comparison with two warm-ups.
- Work/queue/evidence: rust-star/.work/measure128-43bf0dfa/. Keep runtime source
  frozen through the campaign. No paired result or retained 128K claim yet.

## 2026-09-08 — Timed 128K path implemented; frozen build next

- Imported verified prompt/selection-only fixture measurement-131072-v5 from
  the prior repeated C0 report, with hashes/provenance. Timed native prefill now
  accepts exactly 128K in addition to 2K/32K; diagnostic limits stay explicit.
  Fresh/retained engine cycles use the pinned 129-selection transcript.
- No Metal arithmetic changes. Sampling, timers, residency charge/reset and
  collection-disabled eligibility remain unchanged. Added portable fixture
  coverage and a 128K adapter acceptance/rejection test; all 13 adapter tests pass.
- Format pass also normalizes existing formatting in recently touched Rust
  regions. Freeze and build before native gates. No new live/model timing yet.

## 2026-09-08 — 128K measurement and retained-engine milestone started

- User authorized the next milestone at 43bf0dfa on agent/rust-star-bootstrap.
  Work: rust-star/.work/measure128-43bf0dfa/. Preserve prior correctness evidence.
- Import the verified 128K prompt/129-selection transcript with explicit
  provenance; extend fresh and retained timed execution without output capture.
  Preserve diagnostic guards, timing/sampling/residency contracts and oracle-v5.
- Acceptance: host tests/build, fresh 128K full-logit C0, timed fresh/retained
  128K transcript and reset/residency checks, shorter 2K/32K regressions, then
  five alternating fresh-process 128K pairs with one warm-up per engine.
  Record warmed memory, first-step/prefill/decode distributions and host state.
- No required speedup; negative results remain valid characterization. Freeze
  candidate/inputs before timing, stop on failures and preserve every attempt.
  Finish portable evidence/docs and stop at this milestone; no quota burning.

## 2026-09-08 — 128K correctness milestone COMPLETE and retained

- Candidate 0eb484ea: 314 tests and release build pass. All six fresh unchanged
  oracle-v5 captures pass: paired full prefill at 64K/96K and paired full
  128K prefill plus 128 decode traces. Both fresh native 128K runs match all
  129,280 prefill words and 16,547,840 decode words with exact selections.
- Eleven native/regression checks pass without failures/retries: 64K/96K
  prefill, two full 128K runs and all seven established shorter/32K/lifecycle/
  profile controls. Profile audit: 287 native tails, 2,583 valid intervals.
  Final model/prompt/oracle source/binary hashes match their original pins.
- Portable report: rust-star/reports/128k-c0-0eb484ea/README.md, 117 verified
  artifacts plus hash manifest. Raw exact reference vectors, capture/check logs,
  source snapshot, tests/build, component audit and 222 host samples are retained.
  No recorded thermal/performance warnings; observed swap 27.06–35.94MiB.
- The separate external-reference checker extends diagnostic execution through
  128K; existing built-in/measurement contracts remain 32K. No arithmetic or
  RoPE training constant changed. Internal HC acceptance still ends at 24K.
- Native 128K diagnostic RSS 13.880/13.878GiB and macOS peak footprint
  21.910/21.891GiB exclude warmed engine comparability and are not total model/
  Metal memory bounds. The retained 32K control reaches 83.919GiB warmed RSS.
  No new throughput or memory-reduction claim; prior performance reports stand.
- Stop at this completed milestone. Proposed next milestone: admit 128K into
  measurement and retained-engine protocol, then run pinned paired performance
  and memory validation. It has not been started. 256K and coding-agent corpus
  remain future work; do not resume quota-burning mode.

## 2026-09-08 — Nine native checks pass; lifecycle and profile remain

- Both 128K/128 native repetitions, 64K/96K prefill, complete 32K output,
  12K, legacy 8K, 2K and internal 32K controls pass. Accepted internal targets
  remain through 24K; later HC diagnostics do not gain a new correctness claim.
- Retained-engine two-cycle validation is running, followed by the final
  profiled 32K control. Then component audit, final input hashes and portable
  report/docs close the milestone. No capture/native failures or retries.

## 2026-09-08 — Both fresh native 128K/128 runs pass exactly

- Both independent native processes match the repeated oracle-v5 reference:
  129,280 prefill words, 16,547,840 decode words and all greedy selections.
  New 64K/96K prefill and the old complete 32K/128 output control also pass.
- Remaining established controls are now running, beginning with 12K; shorter
  8K/2K, internal 32K, retained reuse and profiled 32K follow. Then audit the
  profile, reverify inputs, package portable evidence and update final docs.
  Milestone remains active through these checks. No failures/retries so far.

## 2026-09-08 — First complete native 128K/128 C0 run passes

- Frozen candidate 0eb484ea matches all 129,280 prefill words and all
  16,547,840 decode words over 128 steps at 131,072 prompt tokens, with exact
  greedy selections throughout. This includes state adoption and sparse-history
  growth beyond the prompt frontier. Native 64K/96K and old 32K/128 also pass.
- A second fresh full native 128K run is underway, then six remaining old
  shorter/lifecycle/profile controls. The milestone remains active until repeat,
  regressions, portable evidence, final input verification and docs pass.
- No failed captures/native checks or retries. Diagnostic process RSS remains
  distinct from warmed benchmark model residency; do not claim a memory gain.

## 2026-09-08 — Native 96K full-output check passes; 128K starts

- Native 96K prefill matches all 129,280 oracle-v5 FP32 words exactly,
  selected token 918. Established 32K/128 and native 64K also pass.
- The first native 128K prefill plus 128 full-logit decode steps is running;
  a second fresh identical native run and remaining old controls follow.
  Candidate/source/input pins remain unchanged, with no failed attempts.

## 2026-09-08 — Native 64K full-output check passes

- Frozen candidate 0eb484ea preserves complete 32K/128 output C0 and matches
  every one of the 129,280 new 64K prefill words (selected token 39).
  The sequential queue has advanced to native 96K, followed by two 128K/128
  checks and remaining established controls. No failures or retries so far.
- Diagnostic RSS differs from the prior warmed engine benchmark: these probes
  do not call warm_tensor_pages. Preserve macOS RSS and peak-footprint values
  separately, but do not interpret them as total Metal/model residency or a
  new memory reduction. The final report must retain this comparability caveat.

## 2026-09-08 — Repeated full 128K oracle traces accepted; native gates next

- All six unchanged oracle-v5 captures pass: paired full prefill at 64K/96K
  and paired 128K prefill plus 128 full-logit decode rows. The two 128K traces
  are byte-identical (16,677,120 FP32 words each), selected prefill token 1110,
  final selected token 2945; every row is finite and no row selects EOS.
- 128K full-trace SHA-256:
  794631eda78dc137db2bb5eb540308c7d51b1efffb86fd2988a508a40e5443b1.
  Capture acceptance and raw evidence: rust-star/.work/frontier128-35f69741/.
- Frozen candidate 0eb484ea (314 tests/build) now enters the eleven-check
  sequential native validation queue: established 32K output, 64K/96K output,
  two full 128K runs, then remaining shorter/lifecycle/profile controls.
  No native larger-frontier claim until these checks pass. No capture retries.

## 2026-09-08 — 64K and 96K oracle prefill references repeat exactly

- Two fresh unchanged oracle-v5 processes match every prefill word at 64K
  (selected 39) and 96K (selected 918). Both references extend the pinned
  32K corpus tokens. Capture logs/raw vectors remain in frontier128-35f69741/.
- Frozen candidate 0eb484ea passes 314 tests/build. Executable SHA-256:
  85081aa01de6d305db92557fb94335694e3bdbdf2dbbe4e8e1ca5e9f038205fc.
- Next in the active sequential capture plan: two 128K prefill + 128-decode
  full-logit traces. Native checks wait for complete oracle repeatability.
  No larger-frontier native C0 or throughput claim yet. No failed captures.

## 2026-09-08 — External frontier checker implemented; oracle capture active

- Added validated external prompt/selection/logit vectors and a separate
  long-prefill-reference-probe command. It checks every FP32 word and labels
  oracle provenance as externally verified, not implicitly trusted by runtime.
  The 128K reference requires all 129 rows; malformed/non-finite/incorrect
  greedy vectors are rejected before model execution. Initial 314 tests pass.
- Generalized native boundary/output/complete-state guards through 128K and
  parameterized decode adoption/positions. Existing benchmark/built-in C0
  entry points remain 32K. External prompts must extend the pinned 32K prefix.
- Audited dynamic sparse merge depth, score allocation and 4K transient offsets;
  retained histories remain absolute. Corrected an additional Rust boundary
  guard found during review. Freeze/rebuild this final candidate before gates.
- Unchanged oracle-v5 capture sequence is active, starting with two 64K
  prefill captures. Host thermal/swap observations are recorded. No native
  larger-context correctness claim until repeated oracle and runtime checks pass.

## 2026-09-08 — 128K correctness milestone authorized and started

- User authorized the next milestone. Start at 35f69741 on the existing
  agent/rust-star-bootstrap worktree. Work: rust-star/.work/frontier128-35f69741/.
- Pin the unchanged oracle-v5/model/prompt, capture two fresh identical prefill
  references at 64K and 96K and two full 128K/128-step traces. Preserve attempts.
- Audit chunk schedules, sparse sorting and decoder adoption before extending
  native execution. Validate each frontier against full logits, then full 128K
  decode and all established shorter/32K/lifecycle controls. Monitor memory.
- A diagnostic external-reference entry point may support new evidence without
  silently broadening the existing 32K benchmark claim. No performance claim
  or oracle changes are part of this milestone. Finish portable evidence/docs,
  keep the journal current, and stop after completion. One GPU workload at a time.

## 2026-09-08 — Memory-efficient prefill milestone COMPLETE and retained

- Candidate de6d2c11: 312 tests/build; seven live controls including three full
  32K/128-step full-logit gates; retained two-cycle transcripts; two passing profiles and
  strict component audits. Preliminary 8b511b5f evidence remains separately.
- Five valid AB/BA pairs plus two warmups, no invalid attempts/retries. Median
  paired RSS ratio .91546827 (8.453% lower), median saving 7.742GiB; raw median
  candidate/baseline peaks 83.850/91.593GiB. All five pairs use less memory.
- Median paired prefill ratio .84245722 (15.754% lower), all five faster;
  raw medians 239.875/285.279s, MAD .686/4.438s. Steady decode paired ratio
  .99694819 (−.305%), raw medians 20.278/20.285 tok/s. First-step paired ratio
  1.02429546 (+2.43%), raw medians 56.582/55.126ms; retain this small tradeoff.
- 128K modeled allocation/fill probe passes 14.837GiB, excluding model/other
  scratch. Live recommended working set 107.520GiB; largest-buffer limit
  80.640GiB. History-growth delta 9.322GiB; measured 32K peak plus that delta
  gives 93.173GiB planning estimate, explicitly not a bound or 128K C0 proof.
- Portable report: rust-star/reports/32k-memory-de6d2c11/README.md, 181 artifacts
  independently verified without model/Metal. Includes frozen source/inputs,
  both-role raw measurements, oracle word/HC targets, phase-1 evidence, host
  observations, allocation probe/provenance, audits and retention decision.
- Host: 79 periodic thermal samples, all reads/no-warning checks pass; swap
  34.81–35.06MiB. 86 load samples include interactive CPU bursts, so no isolated
  laboratory claim. Model/prompt/oracle-v5/historical reports unchanged.
- Runtime README, charter and current checkpoint updated. Capacity probe kept
  at rust-star/diagnostics/prefill_capacity_probe.m. No model/GPU work remains.
  Stop here per milestone mode; next 128K correctness milestone is not started.

## 2026-09-08 — Memory comparison checkpoint: three valid pairs

- Both warmups and the first three pairs passed transcript checks; pair 4 is
  active. No retries or invalid attempts. Preliminary RSS savings are 7.742,
  7.750 and 7.685GiB; paired prefill ratios .82750 / .84325 / .82028, steady
  decode ratios .99695 / .98824 / 1.03353. These are partial results, not the
  final acceptance decision. Complete five pairs and the separate profile.
- Runtime and frozen input/acceptance manifests remain unchanged. Report tools
  are prepared in memory-prefill-final/. After measurement, independently audit,
  calculate the bounded-scope capacity estimate, record host conditions, render
  and verify portable evidence, update current documentation, commit and stop.

## 2026-09-08 — All memory gates and allocation preflight pass; paired study frozen

- All seven de6d2c11 gates pass, with 287 native tails / 2,583 valid component
  intervals audited. Standalone 128K modeled allocation/fill probe passes:
  15,931,240,960 requested bytes (14.837GiB), largest buffer 536,870,912 bytes,
  972,540 page/end-byte checks. This excludes model residency/other scratch and
  is not full 128K inference or an aggregate runtime memory bound.
- Live M1 Ultra reports maxBufferLength 86,586,540,032 and recommended working
  set 115,448,725,504 bytes. Full histories remain context-sized; 32K guards stay.
- Model and prompt rehashed successfully. Frozen comparison work:
  rust-star/.work/paired-rust-memory-de6d2c11/.
  Candidate exe 3206f15e2523f577b1f00fe62d0faa068c90152c6168d16882f9414d1053fe0d;
  accepted 912fae1e exe adea36df8200ae6c99f4b549faebcf90fd01f9da34207dc0add562180cf713c4.
  Plan SHA-256 7588f507bc3e39202c254c27050375bde5362974cb481885212d95e902e04fbb;
  acceptance SHA-256 5256849bda0e9bd14a65a6a2f700454c8930dddb0a98280502e9bb98ad860c82.
- Two warmups/five alternating pairs then separate profile are active, managed
  by memory-prefill-final/follow-through.py. Stop on failure and preserve every
  attempt. No measured improvement claim yet. Finish analysis/report and stop.

## 2026-09-08 — Memory candidate passes output, internal and lifecycle controls

- Frozen de6d2c11 passes 312 tests/build, 12K internal, legacy 8K, complete 2K,
  full 32K output and internal gates, and both retained lifecycle transcripts.
  Full-output controls match every prefill logit and all 128 decode-logit rows;
  accepted internal HC remains through 24K only. No failures or retries so far.
- Final profiled 32K gate is running. On success, the sequential workflow audits
  all component intervals, runs the standalone 128K modeled allocation/fill
  preflight, rehashes model/prompt, freezes five AB/BA Rust-to-Rust pairs plus
  two warmups, then runs an independent profile. One GPU workload at a time.
- Predeclared acceptance requires lower median RSS and four/five lower-RSS pairs,
  median paired prefill latency <=1.03 and steady decode throughput >=0.97.
  No performance or total-memory claim until the comparison completes. Current
  frontier guards remain 32K; capacity is distinct from 128K inference C0.

## 2026-09-08 — Allocation tranche passed; bootstrap transient scope completed

- Candidate 8b511b5f passed 312 release tests/build and the native 12K internal
  allocation/offset control (154.772s). Evidence remains in memory-prefill-291eb611/;
  this preliminary gate is not full 32K acceptance.
- Extended the same chunk-relative contract to bootstrap layer-1/2 HC, layer-2
  Q normalization, attention normalization and split buffers. Updated producer
  offsets, Q/sparse/tail consumers and layer HC checksums together. KV, compressed
  state and token histories retain absolute addressing; legacy controls retain
  their original layout. Native bootstrap rejects rows outside its active chunk.
- Freeze the combined candidate for all seven correctness/lifecycle/profile gates
  in memory-prefill-final/. Measured memory/performance and 128K capacity preflight
  are still outstanding; no memory or speed result is claimed yet.

## 2026-09-08 — Memory scope candidate implemented; validation next

- Partial bootstrap wraps only 16 execution weights plus two indexer weights,
  and skips allocations/initialization for discarded layers 3..42. Full/legacy
  callers keep all buffers and metadata. Rust validates the new 18-view contract.
- Native odd/even continuation slots allocate 4096-row Q, normalized inputs,
  HC and split activations. Producer blits, sparse/Flash consumers and tail
  offsets now use the matching chunk base; KV/compressed histories keep absolute
  addressing. Legacy 8K paths remain at base zero.
- Execution-only long prefill no longer allocates unused diagnostic host vectors.
  No kernel arithmetic, fixture, frontier guard or oracle changes. Candidate
  must pass tests/build and all seven live gates before any measurement claim.
- Build and sequential validation are prepared in memory-prefill-291eb611/.
  Source freezes during the gates. Capacity audit and measured acceptance remain.

## 2026-09-08 — Memory-efficient prefill milestone authorized and started

- User authorized the recommended milestone. Start from 291eb611 on
  agent/rust-star-bootstrap; freeze the accepted 912fae1e executable before
  building changes. Work: rust-star/.work/memory-prefill-291eb611/.
- Scope: eliminate unnecessary initial bootstrap allocations, bound temporary
  continuation storage to the active chunk with correct relative indexing,
  preserve shorter/full-32K/lifecycle C0, measure memory and paired performance,
  and produce an evidence-based 128K capacity preflight. Do not claim full 128K
  C0 from capacity calculations or relax current frontier guards without evidence.
- Require lower observed memory with no material paired performance regression;
  preserve all attempts and compare against frozen accepted Rust. Oracle-v5 and
  historical evidence stay unchanged. One model/GPU workload at a time.
- Current work is source/lifetime audit. No candidate code or new model run yet.
  Keep journal current and stop after this milestone's evidence/report.

## 2026-09-08 — Next-step recommendation; no new milestone started

- User asked for the next step. Recommend memory-efficient prefill and 128K
  readiness: remove unnecessary shared bootstrap allocations, audit lifetimes
  and make transient query storage depend on the active chunk where valid.
- The earlier source audit found a FullQ allocation of 4GiB at 32K and 16GiB
  at 128K; these are allocation calculations, not measured total memory. Address
  absolute indexing and device limits before shrinking buffers or raising guards.
- Proposed acceptance: preserve exact 32K/128 and shorter/lifecycle controls,
  demonstrate measured memory improvement without material paired latency/decode
  regression, and produce an evidence-based 128K capacity preflight. Repeated
  128K oracle capture/full C0 remains a separate acceptance step.
- Corrected stale charter Open Items that still described the completed scoped
  bootstrap and diagnostic work as pending. Runtime and benchmarks unchanged.
  Recommendation only: await user authorization before starting implementation.

## 2026-09-08 — Scoped bootstrap milestone complete and retained

- Retain candidate 912fae1e: initial long-prefill bootstrap now completes only
  layers 0..2, preserves layer-2 HC and then uses the unchanged exact rebuild
  for layers 3..42. Discarded later-layer execution/output-head work is omitted.
  Shared setup allocations and 1,218 wrapped views remain. No kernel arithmetic,
  fixture, or oracle-v5 change. Complete 2K/legacy callers preserve their contract.
- All 312 Rust tests, release build and seven live gates pass. Three 32K
  full-output gates check all 129,280 prefill and 16,547,840 decode words exactly;
  internal accepted checkpoints end at 24K. Reset/reuse checks both transcripts.
- All five alternating Rust-to-Rust pairs and both warmups pass on attempt 1.
  Median paired prefill latency is 0.918349x accepted Rust (8.165% lower), with
  improvement in all five pairs (6.0–8.7%). Candidate/baseline medians are
  273.573/298.948s, MAD 0.379/3.034s. Steady decode is 20.557/20.384 tokens/s,
  paired ratio 1.005162; four pairs improve and one is 2.0% slower.
- Accept the documented tradeoffs: first-step paired median is 1.021207x,
  raw medians 57.479/53.608ms; candidate observed peak 93.281GiB versus baseline
  maximum 92.892GiB. No memory reduction or new Rust-versus-DwarfStar claim.
  The generic oracle role is frozen accepted Rust 94565861, while oracle-v5
  remains the independent unchanged C0 reference.
- Separate post-paired profile passes: 1,483 parent rows, 2,583 valid component
  intervals, exact transcript and complete accounting. Initial bootstrap GPU
  span is 5.129s; native tails account for 87.132s. Largest components are
  attention output 32.957s, routed gate/up 30.155s and routed down 16.493s.
  These are perturbed diagnostics, not additional paired performance samples.
- Better counter-failure diagnostics are retained: chunk/layer, failing sample,
  component/boundary/reason and all 18 raw ticks, plus explicit sentinel rejection.
  Final-source probe passes 256 commands without wrong GPU outputs or timestamp
  anomalies. Current profile successes do not explain the earlier rare failure
  or guarantee unattended capture completion; validity was not relaxed.
- Portable report: rust-star/reports/32k-bootstrap-912fae1e/README.md. All 403
  artifacts verify without model/Metal/Git, including nested immutable reference
  and accepted-baseline evidence, frozen candidate sources, both-role measurement
  audits, full-word C0, profile/probe accounting and host-summary recomputation.
  Host logs retain transient desktop/update load; no thermal warnings were
  recorded and swap use stayed 13.94–15.12MiB during paired observations.
- Runtime implementation still matches the tested frozen source exactly; later
  edits are documentation/evidence only. About 94GiB disk space remains. No push
  or new publication. Stop at this completed milestone; await user direction.

## 2026-09-08 — Five bootstrap pairs complete; isolated profile running

- Both warmups and all five alternating pairs pass with no retries or invalid
  attempts. Candidate prefill improves in all five pairs. Median paired latency
  ratio is 0.918349 (8.165% lower); candidate/baseline median prefill is
  273.573/298.948s. Median absolute deviations are 0.379/3.034s.
- Steady decode medians are 20.557/20.384 tokens/s; paired median ratio 1.005162.
  Four pairs are slightly faster; the fifth is 2.0% slower. First-step paired
  median ratio is 1.021207, with raw medians 57.479/53.608ms. Review all of this
  in the retention decision; do not call every decode measure improved.
- Candidate observed peak is 93.281GiB; baseline maximum is 92.892GiB. Shared
  setup allocations remain, and this milestone makes no memory-reduction claim.
- The separate engine-profile run is active. After it ends, the guarded runner
  independently audits measurements and profile accounting, then stops for
  report/retention/docs. No runtime edit before that run finishes.

## 2026-09-08 — Two valid bootstrap pairs; decision still pending

- Both warmups and the first two alternating pairs pass on attempt 1. Pair 1
  candidate/baseline prefill is 273.194/294.871s; pair 2 is 271.753/295.914s.
  Steady decode is 20.632/20.482 and 20.557/20.342 tokens/s respectively.
  These are preliminary observations, not the final five-pair acceptance result.
- Pair 3 is running. No source/build changes or other model workload introduced.
  Thermal observations report no warnings; light desktop background activity
  is preserved in the host/load logs. Continue through all five pairs, the
  isolated profile and portable audit/report, then make the retention decision.

## 2026-09-08 — All bootstrap gates pass; frozen Rust-to-Rust pairs begin

- Candidate 912fae1e passes all seven live gates: 12K internal, detailed 8K,
  complete 2K, 32K normal output, 32K internal/full output, two retained cycles,
  and 32K profiled full output. All 129,280 prefill and 16,547,840 decode words
  match oracle-v5 on each full-output gate. Internal targets remain accepted
  only through 24K; retained cycles establish transcript/reset correctness.
- Profile audit passes: 1,483 parent rows, 287 native tails and 2,583 ordered,
  valid component intervals. Initial scope is 0..2 exactly once; prefix layers
  3..42 appear exactly once. The initial-bootstrap diagnostic GPU span is 5.151s;
  do not turn this instrumented phase observation into a throughput claim.
- Final-source independent counter probe: 256 commands, zero wrong GPU words,
  counter errors, zero/backward spans, or touching/overlapping boundaries.
  Source/build/output hashes are in counter-probe-final-provenance.json. The
  earlier failure remains unexplained; strict validity is unchanged.
- Model and prompt were rehashed; both frozen executable identities verified.
  The declared acceptance criteria and measurement-tool hashes are bound into
  the correctness manifest and plan. Work:
  rust-star/.work/paired-rust-bootstrap-912fae1e/. The generic oracle slot is
  accepted Rust 94565861, not the unchanged C0 oracle-v5. Both warmups and five
  AB/BA pairs run sequentially, then one isolated profile. No runtime edits,
  builds, or other model workloads during this block. Preserve every attempt.

## 2026-09-08 — Scoped bootstrap passes first complete 32K output gate

- 12K internal, detailed 8K and complete 2K controls pass. The first normal
  32K output-only gate passes all 129,280 prefill words and all 128 decode rows,
  preserving oracle-v5 checksums and the complete selected-token transcript.
- The internal-state 32K gate is running; retained lifecycle and profiled C0
  follow. Source and frozen executable remain unchanged during live validation.
- Prepared independent Rust-to-Rust measurement audit, strict component/coverage
  audits and a guarded sequential follow-through runner. It starts only after
  all seven gates pass, reruns the final-source standalone counter probe, then
  freezes a new plan and runs both warmups/five pairs plus a separate profile.
- Baseline Rust binary equals the accepted e7b81aa2 binary exactly. Keep the
  generic runner oracle role explicitly labelled as Rust; C0 oracle-v5 is
  unchanged and remains separate. No throughput or retention decision yet.

## 2026-09-08 — Bootstrap candidate built; direct Rust comparison planned

- Frozen candidate 912fae1ebbd641fd8edaf4e9f6b26249737775b0, tree
  8cb3b90da6fc288137efa13d1ef6eb4d84e671c2, executable SHA
  adea36df8200ae6c99f4b549faebcf90fd01f9da34207dc0add562180cf713c4.
  All 312 Rust tests and release build pass. Native 12K live control is running;
  exact prefix rebuild has passed through layer 37 at this checkpoint.
- Frozen/hash-checked the accepted restored Rust executable before candidate
  build replaced the target path. accepted-build.json records its provenance.
- For a more controlled optimization decision, run five alternating paired
  comparisons against that accepted Rust runtime in the same session, with two
  warmups and all attempts retained. The runner's generic oracle slot will hold
  the accepted Rust comparator, explicitly labelled; oracle-v5 remains the
  separate unchanged full-output correctness reference. Do not present this
  study as a new Rust-versus-DwarfStar measurement.
- Predeclare lower median paired prefill latency and improvement in at least
  four of five pairs; review decode distributions for a material regression.
  Full C0/lifecycle/profile gates precede the paired block. No runtime edits or
  builds during measurements. Stop after this milestone's report and decision.

## 2026-09-08 — Scoped bootstrap and rejected-counter diagnostics implemented

- Independent Metal probe: 256 commands, nine dependent encoders each, four
  workloads and a reused 18-sample buffer. All GPU outputs match; no counter
  errors, zero/backward spans, touching/overlapping boundaries occurred. This
  does not reproduce or explain the earlier large-model capture failure.
- Preserved strict validity. Rejected native samples now log all 18 raw ticks,
  failing index/component/boundary and reason; Rust adds chunk/layer context.
  Rust validation also rejects the Metal error sentinel, including a final
  sentinel that would otherwise pass a monotonicity-only check. Failure tests
  verify rejection before attribution and diagnostic values/context.
- Applied/reviewed layers-0..2 bootstrap scope: early return retains layer 2 HC,
  skips discarded layers 3..42 and output head, and reports actual scope/selector
  dispatch counts. Full 2K/legacy callers retain their complete contract. The
  accepted later-layer prefix rebuild stays in place; shared setup allocations
  and all 1218 wrapped views still occur and are reported honestly.
- Initial-bootstrap profile span is now 0..2; dedicated audit requires prefix
  layers 3..42 exactly once. Build and all seven live gates are next. Candidate
  is not yet accepted as correct or faster. No oracle or kernel arithmetic change.

## 2026-09-08 — New authorized milestone: diagnosable profiling and scoped bootstrap

- User authorized a meaningful next milestone, allowing a larger scope. Bundle
  rejected-counter diagnostics with the layers-0..2 initial bootstrap experiment;
  stop after this milestone's evidence, retention decision and report are complete.
- First capture all rejected raw counter values and chunk/layer context without
  relaxing validity. Use a small independent dependent-dispatch Metal probe to
  investigate timestamp ordering separately from the large model. Preserve failures.
- Then review/apply the prepared bootstrap scope while retaining the exact later
  prefix rebuild. Require controls, full 32K prefill/128-decode C0, lifecycle,
  profiled C0 and frozen paired measurement before retaining a performance change.
- Start from restored 94565861 on agent/rust-star-bootstrap. Oracle-v5 and all
  historical reports stay pinned. Work: rust-star/.work/counter-bootstrap-94565861/.
  One model process at a time; no agent delegation or new external publication.

## 2026-09-08 — Dense-output experiment closed and reverted; stop for recap

- Completed the declared five valid 32K/128 pairs, retaining the documented
  interrupted pair 4 attempt and its complete-pair recovery. Independent audit
  rehashes measurements, warmups, interrupted artifacts and reproduces statistics.
- Rust/reference median prefill: 351.974/176.897 s. Median paired latency ratio:
  2.01485 versus the accepted baseline 1.72589. Absolute Rust time fell, but the
  reference improved more; the measured benefit criterion failed. Median steady
  decode: 20.170/19.821 tokens/s, paired ratio 1.01664. Host variation and the
  restart prevent isolated causal attribution; no new speed baseline is accepted.
- The separate post-paired profile exited 1 with invalid tail timestamps and
  produced no artifact. It was not retried. The getter rejects error-valued or
  non-increasing samples, but rejected values were not logged; exact cause is
  unresolved. This optional-profiler limitation is documented, not hidden or
  addressed by relaxing validity. Earlier profiled full-output C0 remains separate.
- Dense-output execution changes are reverted. metal_shim.m and long_prefill.rs
  exactly match 9c1d5009; matching profiling C header declarations remain. Fresh
  312 Rust tests and release build pass. Source hashes, restoration patch, rebuilt
  binary hash and logs are preserved. No additional live gate is claimed for the
  reversion; it restores the previously full-C0-gated implementation exactly.
- Negative report: rust-star/reports/32k-dense-output-92ca3ef3/README.md.
  All 232 artifacts verify standalone, including failed-profile evidence and the
  earlier successful 2,583-row profiled C0 gate. Historical reports remain unchanged.
- Burn recap: completed exact-gated nine-component profiling (largest component:
  attention output, 36.75%); tested and rejected dense output; prepared but did
  not apply a layers-0..2 bootstrap draft and guarded validation helpers; identified
  absolute-position transient allocations that block naive broader-context growth.
  The earlier native-tail scratch milestone predates the burn and is not counted.
- User restored milestone-by-milestone work. STOP here after the final commit and
  recap. Proposed next work is better invalid-counter diagnostics, then the
  reviewed bootstrap experiment. No follow-up model run or implementation starts.

## 2026-09-08 — Recovered pair passes; dense-output retention criterion cannot pass

- Pair 4 attempt 2 passes; pair 5's oracle half passes and Rust half is running.
  No additional retry. Original interrupted pair remains intact.
- Completed valid Rust/reference prefill ratios: 1.69773, 1.88010, 2.38594,
  2.14752. Three already exceed the accepted baseline's 1.72589 median ratio,
  so the final five-pair median cannot meet the normalized improvement criterion.
- Finish the declared final pair, separate profile and audited negative report,
  then revert only the dense-output implementation. Keep the exact-gated
  component profiler and matching header declarations. This is failure to
  demonstrate benefit under the measured conditions, not isolated causal proof
  that the kernel change caused all observed variation.
- Final checks/docs/commit and the burn recap remain pending. Do not start the
  prepared bootstrap milestone; user requested stopping after this milestone.

## 2026-09-08 — Host restart interrupted pair 4; explicit recovery underway

- Correction to the preceding status: saved running state survived, but no
  benchmark/helper processes survived the host restart. Current boot time is
  2026-09-08 05:23:23 UTC; final pre-interruption host observation is
  2026-09-07 23:34:38 UTC and records shutdown-related thermal-query errors.
- Preserved inputs/restart-interruption.json, state-before-restart-recovery.json
  and original logs in the frozen paired directory. Both executables rehashed.
- Used the documented paired-runner retry operation for pair 4 only, with the
  explicit host-shutdown/restart reason. It reruns the entire BA pair as attempt 2.
  Original candidate half (538.625 s prefill) and missing oracle half remain
  preserved; this is not a selective rerun of an unfavorable completed pair.
- Retry command is currently running. recovery-run-milestone.py waits for its
  paused/passed state, then runs pair 5, the separate profile, audits and report.
  Work log: dense-output-milestone-9c1d5009/recovery-run.log. Original finisher
  processes are dead and must not be restarted concurrently. Load observation resumed.
- Updated audits to require exactly this one invalid attempt and the expected
  [1,1,1,2,1] final attempt sequence, verifying preserved interrupted artifacts.
  Report now discloses cross-restart measurements; no first-attempt-only claim.
  Fixed a report-export substitution that otherwise retained the old 25-dispatch
  check instead of the candidate's 31-dispatch contract. Runtime is unchanged.
- Stop after this milestone and provide the requested burn-period recap.

## 2026-09-08 — User ends continuous mode; finish this milestone and stop

- User reports the usage limit was reached and has reset. Return to stopping
  after each milestone: finish the current dense-output campaign, audited report,
  retention decision and documentation, then stop and recap the burn period.
- Do not start the bootstrap implementation or broader-context milestone now.
  Prepared follow-up files remain unapplied in the dense-output work directory
  and rust-star/.work/bootstrap-scope-milestone-prepared/. The latter preserves
  all seven live gates and adds explicit initial/prefix profile coverage, guarded
  on a verified dense-output report and recorded retention decision.
- Burn began after native-tail scratch/output milestone fd72c897: its completion
  predates the continuous request. Burn-period delivered work is the exact-gated
  component profiler, the current dense-output experiment and follow-up preflight.
  Distinguish accepted changes, negative findings and draft-only work in the recap.
- At this checkpoint three full pairs pass and pair 4's oracle half is running.
  Runtime remains frozen; final pair/profile/audit/report/decision are pending.

## 2026-09-08 — Three dense-Q8 pairs pass; variance remains material

- Three complete pairs pass on attempt 1; pair 4 is running. Rust prefill times
  are 351.974, 332.585 and 424.092 s, versus reference 207.320, 176.897 and
  177.746 s. These are partial observations, not a final median or retention
  decision. Preserve the slower third pair; do not rerun it for a favorable result.
- Prepared render-non-improvement.py in the active work directory. Use only if
  the positive renderer's existing two-part improvement predicate fails. It
  asserts that failure, preserves all metrics and labels the report explicitly
  as criterion-not-met. It does not change runtime or weaken the predicate.
  Hashing, verification, runtime disposition and current docs still follow.
- The bootstrap draft's 36 non-selector dispatches were independently counted
  across both Metal dispatch APIs, the 4096-row branch and expanded shared macro.
  Actual selector helper dispatches remain additional. Proof: bootstrap-dispatch-review.json.
- git apply --check passed for the unapplied draft. Temporary Rust copies passed
  rustfmt parsing/formatting (about six seconds of host CPU work during pair 3);
  no build or live validation was run, and frozen runtime source is unchanged.
  Keep this incidental host activity with the other background-load qualifications.

## 2026-09-08 — First dense-Q8 timed pair passes; report clarity improved

- Pair 1 passed on attempt 1. Pair 2 (reverse order) has passed its Rust half;
  the oracle half is running. Do not infer retention from these partial results.
- Report export now writes an explicit assessment-pending README immediately
  after cloning baseline evidence. A failed renderer can no longer leave the
  old baseline narrative appearing to describe the new candidate.
- Report rendering now includes median/MAD/min/max for prefill and steady decode
  directly in the README. Review both metrics and host conditions before deciding
  retention, even if the prefill-only improvement predicate passes.
- All modifications during the timed block are journal/report/preflight helpers;
  the runtime and frozen binaries remain unchanged.

## 2026-09-08 — Read-only follow-up audit during paired run

- Oracle half of timed pair 1 passed; Rust half is running. Runtime remains frozen.
- Report generator now discloses changed profile instrumentation: current native
  tails use nine counter-sampled encoders, while the older aggregate profile did
  not. Their GPU-time difference cannot isolate the dense-output code change.
- Bootstrap draft review adds explicit profile coverage requirements: initial
  layers 0..2 followed by prefix layers 3..42. Existing continuation checks alone
  do not verify that prefix coverage. Draft remains unapplied.
- Broader-context preflight found end-position-sized transient allocations in
  bind_long_prefill_layer. One FullQ buffer is 4 GiB at 32K, 16 GiB at 128K,
  32 GiB at 256K and 128 GiB at 1M. These are source allocation calculations,
  not observed residency or total capacity. Audit device limits and absolute
  offsets before extending guards; chunk-bounded transient storage is a likely
  prerequisite. Details remain in broader-context-preflight.md in the active
  dense-output evidence directory. No broader-context model process was run.

## 2026-09-08 — Dense-Q8 warmups pass; timed pairs running

- Both fresh-process warmups passed. The frozen five-pair campaign has entered
  its timed phase; no source edits or builds are running alongside measurements.
- The measurement helper will finish the separate profile and audits. A single
  finish-report.py watcher then exports, renders, hashes and verifies the report.
  Both helpers live in rust-star/.work/dense-output-milestone-9c1d5009/.
  Inspect report-status.json and finish-report.log before manually running any
  export step, to avoid racing the watcher or overwriting a partial report.
- Automatic report generation deliberately stops if its improvement predicate
  fails. Preserve all evidence and assess/reject the candidate honestly; do not
  relax criteria or rerun measurements to seek a favorable result. Final review,
  retention decision, current docs and commit remain manual even after verification.
- The unapplied bootstrap draft remains the next hypothesis. Review confirms
  prefix setup consumes completed layers 0..2 HC and resets later-layer prefix
  state; exact live gates still must prove that the early return is sufficient.
- The project remains active. Continue until completion or actual usage exhaustion;
  no usage-reset credit has been authorized. Check running helpers before resuming.

## 2026-09-08 — Direct dense-Q8 seven gates pass; paired plan frozen

- All seven checks pass on attempt 1, including profiled full 32K prefill/128-decode
  C0 and independent accounting of all 287 tails / 2,583 component intervals.
- Frozen paired directory: rust-star/.work/paired-32k-128-92ca3ef3-dense-output/.
  Plan SHA c046b3e84019aca1e25e8254930743c04f1d250111b17ef779ff6929baf15590.
  Model/prompt freshly rehashed; oracle executable and copied Metal sources match
  the pinned manifest. Two warmups, then five AB/BA/AB/BA/AB pairs; no retries.
- Warmups running. A separate engine profile and independent audit follow pairs.
  No runtime edits or builds during the block. Retention/report/docs remain pending.
- Diagnostic component totals: attention output 35.248 s versus 36.173 s in the prior
  component gate; other major components are similar. This small diagnostic
  change is not a throughput result. The paired campaign determines the decision.
- Prepared but DID NOT APPLY proposed-bootstrap-scope.patch and its generator
  draft-bootstrap-scope.py in the active work directory. Review/apply only after
  this candidate's measurements, report and retention decision are complete.
  It scopes the temporary bootstrap to layers 0..2, retains truthful partial-state
  metadata/optional selection, and preserves the accepted later prefix rebuild.

## 2026-09-08 — All six standard direct dense-Q8 gates pass

- Native 12K, legacy 8K, 2K, both full 32K modes and retained two-cycle reset/reuse
  pass on attempt 1. The retained test verifies both 128-token transcripts; its
  same-process timing is deliberately ineligible for paired performance claims.
- The seventh check, profiled 32K full-output C0 plus component attribution, is
  running. Only after it passes will the helper freeze and start five paired runs.
- Paired preparation now rehashes the copied oracle Metal sources against the
  pinned build manifest as well as checking its executable, model and prompt hashes.
- No source change or failed live run. Candidate 92ca3ef3 is still unmeasured.

## 2026-09-08 — Direct dense-Q8 internal 32K gate passes; lifecycle running

- Both complete 32K modes now pass every prefill word and all 128 decode tensors.
  Accepted internal HC targets through 24K match; later HC values remain diagnostic.
- Five gates pass on attempt 1. Retained two-cycle reset/reuse is running; the
  profiled full-output gate follows. Runtime 92ca3ef3 remains frozen.
- Read-only next-step audit suggests the initial monolithic 4K execution performs
  layers 3..42 and an output head before the correct prefix rebuild recomputes those
  layers. The rebuild entry requires only completed layers 0..2. Before optimizing,
  audit ownership/state dependencies and preserve the correct prefix rebuild.
  Notes: prefix-bootstrap-preflight.md in the active evidence directory.
- A future 128K oracle dry-run plan was also validated, without starting a model
  process. It requires 2K/32K controls and four fresh captures per frontier; it
  covers post-prefill logits only, so separate full-decode trace evidence is needed.
  Neither future item changes the currently running milestone or source.

## 2026-09-08 — Direct dense-Q8 complete 32K output gate passes

- Candidate 92ca3ef3 matches all 129,280 prefill words and all 16,547,840 FP32
  decode words over 128 steps against oracle-v5. Prefill checksum 2996662110144943749;
  decode checksum 1716816788919264169; selected-token checksum 11977321378026793466.
- All three shorter controls pass too. Internal-mode 32K is running, followed by
  retained two-cycle reset/reuse and profiled full-output C0. No failed attempt.
- Performance still unmeasured; keep the frozen source and pending seven-gate
  runner/measurement plan intact. Work remains active under continuous authorization.

## 2026-09-08 — Direct dense-Q8 shorter controls all pass

- Native 12K internal states, detailed legacy 8K and complete 2K prefill controls
  all pass on attempt 1. The 2K selected token remains 15342.
- Full 32K output-only gate is running; complete internal-mode 32K, retained
  lifecycle and profiled full-output checks remain queued. No failed live attempt.
- Tests/build: 312 passing tests, frozen runtime 92ca3ef3. No performance result yet.

## 2026-09-08 — Direct dense-Q8 legacy 8K control passes

- Native 12K and detailed legacy 8K gates both pass on attempt 1. The 8K control
  matches all 129,280 FP32 logits and its pinned selector inventory (token 77179).
- 2K running; both complete 32K modes, retained lifecycle and profiled full-output
  gate remain. No source/oracle change, no live failure and no speed claim.

## 2026-09-08 — Direct dense-Q8 native 12K gate passes

- Candidate 92ca3ef3 matches all 43 accepted internal HC states through the 8K
  and 12K continuation chunks. This 12K gate does not compare a full output tensor.
- Legacy8K is running; 2K, both full32K modes, retained lifecycle and profiled
  full-output validation remain queued. 312 tests/build passed; no failed live
  attempt. Frozen candidate and all oracle targets remain unchanged.

## 2026-09-07 — Direct dense-Q8 candidate built; seven gates running

- Frozen candidate 92ca3ef3ee21ea076852d7ecbda415a15acb270a; executable SHA
  f2a0679725b9c21e0946561d42920b1afa82f05208aa66426268b7ecfef13ea9. All 312 Rust tests and release build pass.
- Native 12K gate running. Sequential queue then checks legacy 8K, 2K, complete
  32K output-only/internal modes, two retained reset/reuse cycles, and a complete
  profiled 32K output gate with component accounting audit (seven checks total).
- On complete success only, finish-measurements.py rehashes model/prompt, freezes
  five alternating pairs plus two warmups, runs a separate profile and independent
  audit. It stops on any failure. Report/retention decision and final docs remain
  manual follow-through after measurement; no speed claim yet.
- Background sessions: build-and-validate 70107; finish-measurements 18973.
  Runtime stays frozen. Update only journal/helper evidence during live execution.

## 2026-09-07 — Direct dense-Q8 output projection candidate implemented

- Native output-low projection now uses eight aligned dense-Q8 dispatches with
  physical output row pitch 8192 and 1024 dispatched columns per group. Input,
  weight and output group offsets preserve token-major layout and MMA ordering.
  Actual bit equality remains subject to live gates.
- Omits the native grouped map, host ID fill and two scratch buffers. Native
  tails now use 20 scratch buffers and 31 dispatches; odd schedule53+padding.
  Legacy paths retain original map/scatter kernels and 26-dispatch tails.
- No Metal kernel source, oracle, quantization or production backend change.
  Added matching C header declarations for the previously validated profile API.
- Tests/build and all six exact/lifecycle gates pending. Evidence directory:
  rust-star/.work/dense-output-milestone-9c1d5009/.
  Candidate is not accepted as a speed improvement until measured.

## 2026-09-07 — Tail component profiling milestone complete

- Candidate e7b81aa2 passes 312 Rust tests and both profiled/normal complete
  32K output-only C0 gates: every prefill word and all 128 full decode rows.
- Independent audit verifies 287 tails / 2,583 component intervals, monotonic
  raw timestamps, complete coverage and per-call attribution conservation.
  Portable report: rust-star/reports/32k-tail-components-e7b81aa2/README.md;
  all 16 artifacts verify. No throughput claim; previous paired report unchanged.
- Attributed native-tail GPU totals: attention output 36.173 s (36.75%), routed
  gate/up 34.285 s (34.83%), routed down 19.267 s (19.57%). Routed map is only
  0.332 s (0.34%), so it is not a meaningful optimization target now.
- Next candidate: use the existing dense Q8 kernel for eight aligned output-low
  groups with direct strided stores, avoiding grouped-map/scatter overhead while
  preserving MMA order. Exactness and speed remain hypotheses until gated/measured.
- User's continuous-work authorization remains active; continue into that milestone.

## 2026-09-07 — Tail component profile candidate built; live C0 running

- Frozen implementation e7b81aa2e468f8fa8fd7b173d7c22fb3a51be46d; executable SHA
  a0ef134f35bc5b32df616951edba1b69ac4025c2fafe6642840b9699a1b2e75e.
- 312 Rust tests and release build pass. Profiled 32K output-only C0 gate is
  running, followed automatically by the normal 32K output-only gate.
- No throughput claim is made for this instrumentation milestone. Component
  coverage, raw timestamp ordering, per-call attribution and both full-output
  checks remain pending. Source stays unchanged during live runs.

## 2026-09-07 — Native tail component profiling in progress

- User authorized continuous next-milestone work until completion or usage exhaustion.
- Adding nine profile-only compute encoder intervals to the native tail: attention
  output, FFN ingress, router, routed map, routed gate/up, routed down, routed sum,
  shared expert, final HC. Normal inference retains its single encoder/25 dispatches.
- Per-call timestamp shares are attributed to the completed tail command GPU
  duration; component rows are separate from parent totals to prevent double counting.
  Raw ticks and explicit perturbation/scaling metadata remain in the evidence.
- Added an output-only C0 probe profile option so instrumented execution must
  match all 32K prefill and 128 decode tensors, not only the token transcript.
- Initial development build saw a mixed source snapshot while the new CLI/API
  was being added and failed with an unresolved export; log preserved as tests.log.
  A fresh build against the complete source is running (tests-final.log). No live
  run or frozen candidate used that failed development build.
- Build/tests and live validation pending. Baseline fd72c897/89e32745 remains
  the accepted measured implementation. Work: rust-star/.work/tail-components-fd72c897/.

## 2026-09-07 — Native tail scratch/output milestone complete

- Native tails reuse 22 command-local buffers, released before decode/reset.
  Retained HC output is written directly, avoiding a 256 MiB copy per tail.
  The unused 192 MiB FP32 routed-mid destination/conversion is removed.
  Kernel arithmetic, v5 targets and legacy diagnostic paths are unchanged.
- Candidate `89e3274568745fe5311d16478ccf35f016958a08` passes 310 Rust tests, all six live
  C0/regression/lifecycle gates, five valid pairs and the separate profile.
  Standalone audited report: `rust-star/reports/32k-tail-89e32745/README.md`.
- Rust/reference median prefill 362.284/212.376 s; steady decode 17.064/17.039 tokens/s.
  Observed Rust peak process memory 91.613 GiB. Absolute prefill change
  versus the prior ingress campaign: -2.51%; host-load and swap
  observations qualify cross-session speed and memory attribution.
- Even-tail host elapsed time falls 66.497 → 59.400 s; odd-tail host time
  falls 63.637 → 56.772 s. GPU tail time does not improve in this profile.
  The candidate is retained with improved paired prefill and exact validation;
  precise isolated speedup attribution remains limited by host conditions.
- All 210 report artifacts pass standalone verification. Current largest GPU
  stage group is `even/tail`; inspect its components before optimizing again.
  Broader context work remains future.

## 2026-09-07 — Native tail paired results and host conditions recorded

- Five-pair Rust/reference prefill medians: 362.284/212.376 seconds; steady
  decode 17.064/17.039 tokens/s. Independent audit still pending.
- Prior ingress campaign: 371.601/206.136 seconds. Rust absolute prefill falls
  about 2.51% while reference rises about 3.03%; cross-session attribution is
  limited by background load. All five pairs remain retained.
- Observed Rust peak process memory is 91.613 GiB (prior 92.250 GiB), with
  2794.69 MiB host swap used after pairs. This is not proof of reduced total
  allocation requirements. No thermal/performance warning was reported.
- Separate profile is running; final audited report and docs remain pending.

## 2026-09-07 — All five native tail pairs pass; profile running

- All five alternating pairs and both warmups pass on attempt 1. No rejected
  measurement or retry. Frozen timed block completed successfully.
- Separate 32K profile is running. Independent audit, portable report and
  final documentation remain pending; candidate remains unchanged.

## 2026-09-07 — Four native tail pairs pass

- Pairs 1–4 pass on attempt 1; final pair 5 (AB) is running. Both warmups pass.
  No invalid measurement/retry. Profile, independent audit and final report pending.

## 2026-09-07 — Three native tail pairs pass

- Pairs 1–3 pass on attempt 1; pair 4 (BA) is running. Both warmups passed.
  Full statistics, separate profile and independent audit remain pending.

## 2026-09-07 — Two native tail pairs pass

- Pairs 1 (AB) and 2 (BA) pass on attempt 1; pair 3 (AB) running.
  No invalid attempt/retry. Final statistics/profile/audit remain pending.

## 2026-09-07 — First native tail pair passes

- Pair 1 (AB) passes on attempt 1. Pair 2 (BA) is running; both warmups passed.
  No rejected attempt. Full distribution, profile and audit remain pending.

## 2026-09-07 — Native tail warmups pass; pair 1 running

- Both warmups pass with the expected transcript. Pair 1 (AB) is running
  under frozen plan d27a9f5e. No failed attempt or retry.
- Five-pair statistics, isolated profile and final audit/report pending.

## 2026-09-07 — Native tail measurement plan frozen

- Five alternating AB/BA/AB/BA/AB pairs, one warmup per engine, then a separate
  profile. Plan SHA `d27a9f5e56d6c804f62c10173a9a4e324498eba52d40227709b7d23cf15efe2f`.
- Campaign: `rust-star/.work/paired-32k-128-89e32745-tail/`.
  Model/prompt freshly rehashed; candidate/oracle binaries frozen. No builds
  or runtime edits during measurement. Host load/thermal/memory recorded.
- All correctness gates passed. Warmups running; pairs/profile/audit pending.

## 2026-09-07 — All native tail correctness/lifecycle gates pass

- Candidate 89e32745 passes all six gates on attempt 1: native 12K, legacy 8K,
  2K, both full 32K modes and retained two-cycle reset/reuse. 310 tests pass.
- Model/prompt rehash and frozen paired-plan preparation follow. Timed pairs,
  separate profile and independent report audit remain pending.

## 2026-09-07 — Both native tail 32K modes pass

- Both output-only and diagnostic 32K runs match every prefill logit and all
  128 complete decode outputs. Accepted internal HC targets through 24K match.
- Retained two-cycle reset/reuse is running. All first five gates pass on
  attempt 1; fresh paired/profile measurement remains pending.

## 2026-09-07 — Native tail full 32K output gate passes

- Candidate 89e32745 matches every 32K prefill logit and all 128 full decode
  tensors (selected prefill 85166, final decode token 14).
- Diagnostic 32K is running; retained lifecycle and fresh paired/profile
  evidence remain pending. All completed gates pass on first attempt.

## 2026-09-07 — Native tail shorter controls pass

- Native 12K, detailed legacy 8K and 2K controls pass for 89e32745.
  The legacy 8K run matches all 129,280 logits and its original schedule.
- Full 32K output-only gate is running; diagnostic 32K, retained lifecycle,
  paired campaign and isolated profile remain pending. No failed attempt.

## 2026-09-07 — Native tail passes exact 12K states

- Candidate 89e32745 passes all accepted native internal states through 12K,
  across all 43 layers. Legacy detailed 8K control is running.
- Full 32K outputs, retained reset/reuse and fresh timings remain pending.
  No candidate failure, fixture update or tolerance has been introduced.

## 2026-09-07 — Native tail candidate build and tests pass

- Candidate `89e3274568745fe5311d16478ccf35f016958a08` passes 310 Rust tests and release build.
  Executable SHA `7f14080aea91a3631bb85b8c55dd8f02031e983ab2b74cc1618235a0262a74a4`.
- Native 12K gate is running; remaining live gates and paired/profile evidence
  pending. No arithmetic, oracle or fixture changes.

## 2026-09-07 — Native tail scratch and output milestone started

- Baseline: ingress candidate ecf529a9/report 2d8b8405. Largest profile group
  even/tail is 49.919 GPU / 66.497 host seconds; odd tails share the executor.
- Native 4K tails now reuse command-local scratch via the existing transient
  buffer cache, released at prefill-to-decode handoff/reset. Retained HC buffers
  remain separate. Native final HC writes directly to its retained destination.
- The unused native routed-mid FP32 diagnostic conversion/allocation is removed;
  32-row and legacy prefix controls retain their old outputs and schedule.
  Native tail dispatch count changes 26 → 25; no model arithmetic changes.
- Build, six live gates and fresh paired/profile audit are pending. Evidence:
  `rust-star/.work/tail-milestone-2d8b8405/`. Oracle-v5 and fixtures unchanged.

## 2026-09-07 — Odd-layer ingress batching milestone complete

- Native odd-layer ingress now processes 4096 query rows per call with
  unchanged kernel reduction geometry and direct predecessor input.
  Ingress API calls fall from 17,920 to 140. No oracle/fixture/arithmetic change.
- Candidate `ecf529a9743baac5e7c182a8c14dfb34d3276984` passes 310 Rust tests,
  all six live C0/regression/lifecycle gates, five valid alternating pairs and
  the isolated complete profile. Reference remains pinned oracle-v5 ccbd16d.
- Rust/reference median prefill 371.601/206.136 s; steady decode 17.264/17.075 tokens/s.
  Peak Rust process memory: 92.250 GiB. Full distributions, old-baseline
  comparison, host observations and standalone verification: `rust-star/reports/32k-ingress-ecf529a9/`.
- Odd-layer ingress profile: 22.709 GPU seconds,
  29.797 host seconds, 140 stage calls. Largest current
  GPU stage group: `even/tail`. Target that measured cost next.
- Ingress GPU/host time falls from 54.263/81.355 to 22.709/29.797 seconds.
  The optimization is retained with exact outputs and improved paired latency
  ratio. Absolute paired prefill improves only 0.60%; OBS/video load and about
  1.8 GiB host swap limit isolated speedup and memory attribution.
- All historical reports and fixtures remain immutable. This is a 32K/128
  development result; broader contexts and 256K headline work remain future.

## 2026-09-07 — Ingress paired measurements and host conditions recorded

- Five-pair Rust/reference median prefill: 371.601/206.136 seconds; steady
  decode 17.264/17.075 tokens/s. Independent audit is still pending.
- Previous attention baseline was 373.843/177.723 seconds. Rust absolute
  latency is only slightly lower while reference latency rises substantially;
  cross-session ratios are descriptive, not isolated causal speedup claims.
- OBS/camera/video activity appears throughout this campaign. Host swap starts
  at 1823.19 MiB used and ends at 1791.19 MiB; no thermal/performance warnings
  reported. Observed Rust peak resident memory 92.250 GiB does not establish
  reduced allocation requirements under these different host conditions.
- Separate profile is running; final audit/report/docs pending.

## 2026-09-07 — All five ingress pairs pass; profile running

- All five alternating pairs and both warmups pass on attempt 1. No rejected
  measurement or retry. Frozen campaign completed successfully.
- Separate 32K profile is running. Independent audit, portable report and
  final documentation remain pending; do not modify the frozen candidate.

## 2026-09-07 — Four ingress benchmark pairs pass

- Pairs 1–4 pass on attempt 1. Final pair 5 (AB) is running; both warmups passed.
  No invalid attempt/retry. Isolated profile, audit and final report pending.

## 2026-09-07 — Three ingress benchmark pairs pass

- Pairs 1–3 pass on attempt 1; pair 4 (BA) is running. Both warmups passed.
  No rejected attempt. Final distribution, profile and report audit pending.

## 2026-09-07 — Two ingress benchmark pairs pass

- Pairs 1 (AB) and 2 (BA) pass on attempt 1. Pair 3 (AB) is running.
  No invalid pair/retry; final statistics, profile and audit remain pending.

## 2026-09-07 — First ingress benchmark pair passes

- Pair 1 (AB) passes on attempt 1. Both warmups passed; pair 2 (BA) running.
  No rejected pair or retry. Full statistics/profile/audit remain pending.

## 2026-09-07 — Both ingress campaign warmups pass

- Oracle and candidate warmups both pass with the expected transcript.
  Pair 1 (AB) is running under frozen plan ec570b76. No invalid attempt.
- Candidate, oracle and plan remain unchanged; five pairs, isolated profile,
  audit and final docs/report remain pending.

## 2026-09-07 — Ingress paired plan frozen; warmups started

- Five alternating AB/BA/AB/BA/AB pairs, one warmup per engine, then a separate
  profile. Plan SHA `ec570b76290ba2b994c25bf62956844230d6fa37ce0e482ed1c09e58c7db2f64`.
- Work: `rust-star/.work/paired-32k-128-ecf529a9-ingress/`.
  Model and prompt freshly rehashed; candidate and oracle binaries frozen.
- All six gates and 310 Rust tests pass. Runtime edits/builds stop for the
  paired block. Host load, thermal state and memory are captured throughout.

## 2026-09-07 — All ingress correctness and lifecycle gates pass

- Candidate ecf529a9 passes all six gates, including both full 32K modes and
  two retained reset/reuse cycles. 310 Rust tests pass; v5 targets unchanged.
- Model/prompt hashes are being revalidated before the fresh five-pair plan.
  Timings and isolated profile remain pending. Initial metadata failure is
  retained separately and will be included in the audited report.

## 2026-09-07 — Both complete 32K ingress gates pass

- Output-only and diagnostic 32K runs both match all prefill logits and all
  128 full decode tensors. All accepted internal HC targets through 24K match.
- The sixth gate, two full retained-engine reset/reuse cycles, is running.
  Fresh paired/profile measurements remain pending; candidate ecf529a9 frozen.

## 2026-09-07 — Ingress complete 32K output gate passes

- ecf529a9 matches every 129,280-word prefill output and all 128 full-logit
  decode steps at native 32K (selected prefill 85166, final decode token 14).
- Diagnostic 32K is running; retained two-cycle lifecycle and fresh paired
  campaign/profile are pending. No oracle or fixture changes.

## 2026-09-07 — Ingress shorter controls pass; full 32K started

- Native 12K, detailed 8K (all 129,280 logits, selected token 77179) and 2K
  controls pass for ecf529a9. No changed tensor targets or tolerance.
- Full 32K output-only validation is running, then diagnostic 32K and two
  retained-engine reset/reuse cycles. Timed campaign remains gated on all six.

## 2026-09-07 — Whole-chunk ingress passes native 12K

- Corrected candidate ecf529a9 passes 310 Rust tests, release build and every
  accepted internal state across all 43 layers through native 12K.
- The 8K legacy control is running. 2K, both full 32K output modes, retained
  lifecycle and fresh paired/profile measurements remain pending.
- Initial metadata failure remains preserved; no numerical mismatch or target
  change has occurred. Frozen build identity is in ingress work/batch-build.json.

## 2026-09-07 — Ingress metadata assertion corrected; attempt retained

- First native 12K attempt stopped at layer 3's stale schedule assertion:
  actual 48 dispatches / 28 model views matched one ingress batch, while the
  assertion still expected 1318 / 1171 for 128 batches. HC comparison had not
  yet run for that layer; no numerical correctness claim is made for this attempt.
- Corrected metadata by subtracting 127 × 10 dispatches and 127 × 9 bindings.
  Original executable, build, logs and failed validation record remain in
  `rust-star/.work/ingress-milestone-12fb9314/metadata-attempt-05447277/`.
- Rebuild and all six gates must pass before any paired plan is created.

## 2026-09-07 — Ingress candidate build and tests pass

- Candidate `054472775f705e8744fbed16e2bdce4ed5fc9a8c` passes all 310 Rust tests and release build.
  Frozen executable SHA `f803f68cf78bfde5115c13b1ec72b4d33831a7ab8130ec0d3773aec91e56aafb`.
- Also fixed the pre-existing formatter-only wrap of the v5 selector label.
  No oracle, kernel arithmetic or fixture changes. Native 12K validation is
  running; remaining controls, full-output/lifecycle gates and timings pending.

## 2026-09-07 — Odd-layer ingress batching milestone started

- Baseline is the completed attention candidate 1eb20ddb/report 12fb9314.
  Its separate profile identifies odd/ingress at 54.263 GPU / 81.355 host seconds.
- Native ingress now stages the immutable predecessor HC by reference and
  batches 4096 independent rows per call. Projection threadgroups retain their
  original 32-row arithmetic; legacy diagnostic calls remain 32 rows.
- Oracle-v5 and fixtures are unchanged. Build, six live correctness/lifecycle
  gates, five fresh paired runs, separate profile and audited report are pending.
  Evidence: `rust-star/.work/ingress-milestone-12fb9314/`.

## 2026-09-07 — Even-layer attention batching milestone complete

- Native continuation attention now processes 4096 query rows per call with
  unchanged kernel reduction geometry and direct output to the tail buffer.
  Attention calls fall from 18,816 to 147. No oracle/fixture/arithmetic change.
- Candidate `1eb20ddbfa29b893dca8865494e0b442e39e92e4` passes 310 Rust tests,
  all six live C0/regression/lifecycle gates, five valid alternating pairs and
  the isolated complete profile. Reference remains pinned oracle-v5 ccbd16d.
- Rust/reference median prefill 373.843/177.723 s; steady decode 19.933/19.386 tokens/s.
  Peak Rust process memory: 93.229 GiB. Full distributions, old-baseline
  comparison, host observations and standalone verification: `rust-star/reports/32k-attention-1eb20ddb/`.
- Even-layer attention profile: 41.339 GPU seconds,
  45.620 host seconds, 147 calls. Largest current
  GPU stage group: `odd/ingress`. Target that measured cost next.
- Later OBS/video activity changed background load. All pairs are retained;
  exact isolated speedup attribution is not claimed. Memory remains a tradeoff.
- All historical reports and fixtures remain immutable. This is a 32K/128
  development result; broader contexts and 256K headline work remain future.

## 2026-09-07 — Five attention pairs pass; isolated profile running

- All five alternating pairs and both warmups pass on attempt 1. No rejected
  samples or retries. The separate 32K profile is running; audit remains pending.
- Complete runner summary: Rust/reference median prefill 373.843/177.723 s;
  steady decode 19.933/19.386 tokens/s. Previous Rust median prefill was
  423.152 s. Background-load variation, including later OBS/video activity,
  limits attribution of an exact isolated speedup; all samples are retained.
- Candidate, oracle and targets remain unchanged. Final report/docs/commit
  follow the completed profile and independent audit.

## 2026-09-07 — Four attention pairs pass; background-load change recorded

- Both warmups and pairs 1–4 pass on attempt 1. Final pair 5 is running.
- Later host observations include OBS, camera/video and desktop activity;
  CPU load increased without thermal warnings. Retain every sample and disclose
  this environment variation with per-pair results. It limits attribution of
  an exact isolated speedup; no selective retry or sample removal is permitted.
- Isolated profile, independent audit and final report still follow the block.

## 2026-09-07 — Attention candidate gates complete; paired campaign starts

- Frozen 1eb20ddb passes all six live gates including both retained reset/reuse
  cycles. Both complete 32K output modes and all accepted internal states pass.
- Model and prompt rehashed. Frozen plan SHA `2ca644b554465dc38d16655872f3f28de184c60003c89e1044e60c0f90df6ed7`.
  Campaign: `rust-star/.work/paired-32k-128-1eb20ddb-attention/`.
- Warmups started; five alternating pairs and one separate profile follow.
  No builds or heavy tests during the timed block. Audit/report/docs remain.

## 2026-09-07 — Both attention candidate 32K C0 modes pass

- Frozen 1eb20ddb passes output-only and diagnostic 32K prefill plus all 128
  complete decode outputs against v5. All accepted internal HC targets through
  24K match. Shorter controls and 310 Rust tests already pass.
- Retained two-cycle reset/reuse is running. No candidate or fixture changes
  since the freeze. Paired plan preparation remains gated on lifecycle success.

## 2026-09-07 — Batched attention passes complete 32K output C0

- Frozen 1eb20ddb matches all 129,280 prefill logits and all 128 complete decode
  outputs against unchanged oracle-v5. Selected prefill token 85166 and complete
  transcript remain exact; no fixture or arithmetic change was needed.
- Diagnostic 32K is running next, followed by retained two-cycle reset/reuse.
  Shorter controls and 310 tests already pass. No paired timing has started.

## 2026-09-07 — Attention batching shorter controls pass

- Frozen 1eb20ddb passes 12K internal states, detailed 8K including all 129,280
  output logits, and the 2K regression control. No oracle or fixture change.
- Full 32K output-only validation is running, followed by diagnostic 32K and
  retained two-cycle reset/reuse. Paired measurement remains pending.

## 2026-09-07 — Batched attention passes 12K internal C0

- Frozen attention candidate 1eb20ddb passes the 12K native internal-state
  gate through all 43 layers. This checks the independently accepted internal
  targets, not a complete 12K full-output claim. No fixture changes required.
- Detailed 8K regression is running, then 2K, both full 32K modes and retained
  reset/reuse. Timing remains gated on the remaining live checks.

## 2026-09-07 — Attention candidate built; native C0 validation running

- Candidate `1eb20ddbfa29b893dca8865494e0b442e39e92e4` passes 310 Rust tests and release build.
  Executable SHA `a13b8d518c3715ed35f54deecb7845d0dfd3f075545527970c1b69bed1eeb908`.
- Work/evidence: `rust-star/.work/attention-milestone-502bb6b6/`.
  The 12K native internal-state gate is running first, followed by detailed
  8K, 2K, both full 32K modes and retained two-cycle validation. Five paired
  measurements and separate profile remain gated on all six live checks.

## 2026-09-07 — Even-layer attention batching milestone starts

- Authorized follow-up: optimize the v5 profile's largest GPU stage group while
  preserving exact outputs and lifecycle. Baseline is cc032078 / report
  `rust-star/reports/32k-upstream-v5-cc032078/`: 423.152 s prefill,
  20.484 tokens/s steady decode, 92.300 GiB peak process memory.
- Native continuation attention now dispatches all 4096 query rows together,
  retaining the original projection/score/attention reduction geometry. This
  amortizes scratch allocation, raw-ring reconstruction and F16 KV conversion;
  heads write directly into the existing 512 MiB tail-input buffer.
- The separate 32-row diagnostic controls remain available. No oracle, shader,
  arithmetic, decoder, SSD or distributed path changes. Candidate validation
  and performance remain pending; no speedup claim yet.
- Required closure: full C0 and shorter controls, retained reset/reuse, five
  fresh paired runs, isolated profile, audited report and journal update.

## 2026-09-07 — Final packaging and regression checks pass

- Default oracle-v5 builder now includes pinned third-party image-decoder
  headers and license. Fresh build passes; all 72 exported files match the
  pinned source tree, including all 69 files inventoried for the measured
  reference. Measured executables are unchanged. Failed and corrected build
  logs, provenance and exporter are included in the report.
- Final full Python suite passes 93 tests. Report verification additionally
  checks builder provenance, dependencies and these final regression results.
- Upstream-refresh milestone is complete. Final standalone verification passes
  all 193 artifacts; current docs and capture commands use oracle-v5.

## 2026-09-07 — Final default-builder packaging check

- All measurements, profile and the 187-artifact standalone report audit pass.
  A final default-v5 builder smoke check found its source exporter omitted
  upstream's new `third_party/iris` image-decoder headers. Measured reference
  came from the complete pinned tree, so numerical and timing evidence is
  unaffected. Failed log retained under refresh W `default-v5-builder.log`.
- Export pinned `third_party/` dependencies with the existing native sources;
  validate a fresh default build before closing the milestone. No runtime or
  oracle source changes; no new model run is required for this packaging fix.

## 2026-09-07 — Upstream refresh and oracle-v5 baseline milestone complete

- Completed pinned upstream c0a6119f refresh, oracle-v5 ccbd16d acceptance,
  applicable resident M1/Q2 deterministic-selector port, complete correctness
  and retained lifecycle gates, five fresh paired measurements and separate
  profile. Durable report and standalone tensor/evidence verifier: `rust-star/reports/32k-upstream-v5-cc032078/`.
- Median Rust/reference prefill 423.152/172.875 s; steady decode
  20.484/19.507 tokens/s. Full distributions and paired ratios are in the report. These
  remain 32K development results, not the 256K primary workload.
- Four repeated 2K/32K reference frontiers, final 8K control, exact tokenizer,
  two full 129-row traces and two full HC inventories at each 12K–24K pass.
  New 32K prefill target is exactly reproduced by the selector-only v4 control;
  all 129 selected IDs remain unchanged. All historical targets are preserved.
- The initial Rust diagnostic stopped on historical 16K layer 34 HC. Direct v5
  captures reproduced Rust, establishing separately versioned internal targets.
  A separate 8K selector fixture also versions four equal-score index swaps
  proven by two direct v5 captures; score inputs and selected sets are unchanged.
  All stopped attempts and successful full-output runs remain in evidence.
- Final Rust candidate `cc032078748ad4c450717b1cfd4e734fd57cf9e6` passes 310 Rust and 93 Python
  tests, both fresh full 32K modes, 12K internal/2K/8K controls and two retained
  cycles. Both full-output gates match v5 exactly. Internal HC
  acceptance stays through 24K; later internals are not claimed as oracle-exact.
- Peak Rust memory is 92.300 GiB versus 87.585 GiB in the previous campaign;
  the reference is 80.893 GiB. Track this tradeoff in the next optimization.
- Candidate SHA `e87354b9f0ed3e2f30ac1b3584d7f03a255bbb715664dc452e29953b205bd706`;
  reference SHA `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`;
  original plan SHA `915d68eda1ebf4f619f3767a2003af7baa01a21d616612dc283e221ddf94e8cc`.
- Profile's largest GPU stage group: `even/attention`. Next local milestone should
  reduce that measured cost while preserving full v5 outputs and lifecycle.
  No additional optimization or broader-context performance claim is included.

## 2026-09-07 — Five refreshed baseline pairs pass; profile running

- Both warmups and all five alternating pairs passed on attempt 1. No timed
  sample was rejected or retried. Isolated 32K profile is running.
- Preliminary complete summary: Rust/reference median prefill 423.152/172.875 s;
  steady decode 20.484/19.507 tokens/s. Independent audit remains pending.
- Final report rendering uses raw paired latency ratios and the audit's memory
  distributions; these metrics are not fields of the standard summary schema.

## 2026-09-07 — All final gates pass; frozen paired campaign starts

- Candidate `cc032078` passes all six live gates, including both full 32K C0
  modes and both retained reset/reuse cycles. Both retained transcripts match.
- Model and prompt hashes rechecked. Final plan SHA:
  `915d68eda1ebf4f619f3767a2003af7baa01a21d616612dc283e221ddf94e8cc`.
  Campaign: `rust-star/.work/paired-32k-128-cc032078-v5/`.
- Warmups started; five alternating pairs and the isolated profile are queued.
  No builds or heavy checks during measurements. Final audit/report/docs remain.

## 2026-09-07 — Frozen candidate completes both 32K C0 modes

- Final candidate `cc032078` passes output-only and diagnostic 32K gates:
  all prefill logits and all 128 full-logit decode steps match oracle-v5.
  Accepted internal HC targets through 24K also match. Shorter controls pass.
- The retained two-cycle lifecycle gate is running. Five paired measurements
  and the isolated profile remain pending; no timed pair has started.
- Paired-plan preparation now refreshes the benchmark-driver commit alongside
  oracle source/executable identity, avoiding inherited v4 freeze metadata.

## 2026-09-07 — Frozen cc032078 full output-only gate passes

- All prefill and128-step decode logits match v5 exactly on the final frozen
  candidate. Detailed8K,12K and2K already pass. Diagnostic32K is running;
  retained two-cycle gate follows, then frozen paired-plan preparation.
- No source changes since the frozen candidate; all earlier diagnostic fixture
  and metadata failures remain retained. No paired timing has started.

## 2026-09-07 — Final candidate shorter controls all pass

- Frozencc032078 passes detailed8K through all43 layers/full output,12K
  internal control and2K complete prefill/output control. Full32K output-only
  then diagnostic mode and retained two-cycle validation are running next.
- No runtime changes after this frozen build. Pending timing uses the same
  candidate once the remaining gates pass; reference remains pinned v5.

## 2026-09-07 — Complete detailed8K gate passes after fixture/accounting refresh

- Candidatecc032078 detailed8K probe exits0: all43 layers and all129,280
  output logits match, selected77179. Versioned second-tile and full-layer4
  selector checks pass; remaining even layers report4852 dispatches correctly.
- Queue continues12K,2K,both full32K modes and retained two-cycle validation.
  Reference v5 producer/fixtures remain unchanged; no paired timing yet.

## 2026-09-07 — Complete v5 diagnostic candidate built

- Candidate `cc032078748ad4c450717b1cfd4e734fd57cf9e6` passes310 Rust tests and release build.
  Executable SHA `e87354b9f0ed3e2f30ac1b3584d7f03a255bbb715664dc452e29953b205bd706`.
- Full v5 output, HC and selector targets are independently validated. Remaining
  legacy8K merge counts are corrected; detailed8K-first six-gate queue is active.
  All prior stopped attempts remain preserved. No paired timing yet.

## 2026-09-07 — Remaining-layer diagnostic merge accounting corrected

- Detailed8K candidate passes the new second-tile/full-layer4 selectors and
  reaches layer16, then metadata rejects37 actual dispatches against36 expected.
  The separate `remaining_prefill.rs` legacy8K driver retained stale counts.
- Corrected all such remaining-layer counters:37 per tile,4736 per continuation
  and4852 aggregate per even layer. Prefix36 stays unchanged. Existing report
  contract and its synthetic fixture now agree with the actual merge dispatch.
  No arithmetic, shader or golden tensor changes in this step.
- Original6aaa1192 attempt retained under `old-dispatch-attempt-6aaa1192/`.
  Rebuild and detailed8K-first full validation queue resumes before timing.

## 2026-09-07 — Full continuation selector checksum versioned

- Detailed8K passed the new second-tile selector then stopped on layer4's
  historical full-selector checksum. The other15 checksums, including full
  layer HC, match. Two fresh all-even-layer v5 captures reproduce Rust's
  layer4 selector checksum `c579e6261fe3045a` with identical full inventories.
- Separate `prefill-indexer-selectors-8192-v5` records all21 even-layer score
  and selector inventories and retains the full layer4 selector tensor.
  Only the stale layer4 selector expected checksum changes; JSON exposes its
  versioned inventory and actual checksum. All other tensor checks remain.
- Original053c5593 stopped attempt archived under
  `old-full-selector-attempt-053c5593/`. Rebuild and detailed8K-first validation
  queue will run before full32K, shorter and retained controls and timing.

## 2026-09-07 — Versioned v5 second-tile selector ordering

- Candidate8aa4d13d passed both complete32K C0 modes and2K, then detailed8K
  probe stopped on historical second-tile indexer top-k slot1930 (643 vs161).
  Two direct v5 captures reproduce Rust. All input score words are unchanged;
  only four index slots swap equally scored keys, and every row selects the
  identical key set. This is selector tie ordering, not an output mismatch.
- Separate `prefill-indexer-selector-4128-v5` target replaces only that expected
  index list. Other tile fixtures retain their original producer labels; JSON
  now explicitly identifies the separate selector fixture. Scores preserve
  causal negative-infinity masks as i32 bit patterns, as in the old schema.
- Raw capture succeeded; collector initially expected.bin instead of upstream
  .i32 and was corrected without rerunning that completed process. Logs kept.
- Original8aa4d13d attempt archived under `old-selector-attempt-8aa4d13d/`.
  Rebuild; run detailed8K first, then all remaining gates before measurements.

## 2026-09-07 — Both final-candidate32K full-output gates pass

- Candidate8aa4d13d passes fresh output-only and diagnostic32K/128 gates:
  all129,280 prefill and16,547,840 decode FP32 words match v5 exactly.
  Diagnostic mode also matches every accepted internal target through24K.
-2K then8K controls and retained two-cycle32K validation remain in the queue.
  Paired baseline has not started; all measurements still gated on lifecycle.

## 2026-09-07 — New v5 internal checkpoints all match Rust

- Fresh diagnostic candidate8aa4d13d passes every accepted internal checkpoint
  through24K, including the previously stopped16K layer34 and all changed
  later layers. This validates the new independently repeated targets in Rust.
- Diagnostic32K final chunk/output/128-step comparison is running.28K/32K
  internal hashes remain diagnostic-only; no unsupported internal C0 claim.
  Remaining shorter and retained gates still precede the paired baseline.

## 2026-09-07 — Final candidate output-only full C0 passes

- Candidate8aa4d13d fresh32K output-only run passes every prefill and128-step
  decode FP32 comparison. Diagnostic32K is now running against independently
  repeated v5 HC targets, followed by2K/8K and retained lifecycle controls.
- Rust/cargo versions and empty Rustflags rechecked against the frozen build
  configuration; `toolchain-verification.json` records exact matches.
- Measurements remain gated on the remaining live checks.

## 2026-09-07 — Final v5 checkpoint candidate frozen and validating

- Candidate `8aa4d13db803f96494d2c66988f12e81887228ca` passes310 Rust tests and release build.
  Executable SHA `3c86bce02479c9ae11d2284c3c0470976468609247dda8928fa63caece5609d0`.
  Inference implementation is unchanged from the previously exact v5 candidate;
  the only runtime source delta is independently captured diagnostic targets.
- New full six-gate queue is running. Wrapper `refresh-hc-and-resume.py`
  proceeds to frozen-plan preparation and five paired runs/profile only after
  all gates pass. Portable export/verifier then runs; final docs/commit remain.
- Original stopped attempt remains under `old-hc-attempt-9a519a36/`.

## 2026-09-07 — Repeated v5 internal targets accepted through24K

- Two fresh pinned-reference captures at each12K/16K/20K/24K agree for all43
  complete4096-row HC tensors and full frontier logits.16K layer34 exactly
  reproduces Rust;12K remains identical to historical targets. Separate new
  `long-prefill-chunks-v5` inventories and provenance preserve old v3 files.
- Runtime later-chunk expected checksums now come from those independently
  captured v5 tensors. No arithmetic or selection code changes in this step.
- Original9a519a36 successful full-output run and stopped historical-HC run
  archived under `old-hc-attempt-9a519a36/` in refresh W. Rebuild and complete
  six-gate queue will run again against the new frozen candidate before timing.

## 2026-09-07 — Direct v5 capture reproduces differing16K checkpoint

- First pinned-v5 full HC capture at16K reproduces Rust layer34 exactly:
  `3e1a2b79f656939a`. This establishes that the old internal target is stale
  for v5 at that layer, not a Rust mismatch against the new reference.
- Independent second16K capture is running, then paired20K/24K/12K captures.
  No checkpoint replacement until all full inventories repeat. The earlier
  failed diagnostic run and full-output pass remain preserved.

## 2026-09-07 — Historical16K internal target mismatch; timing blocked

- Fresh diagnostic32K candidate stops at chunk12288 (16K frontier), layer34:
  actual HC `3e1a2b79f656939a`, old v3 target `c0d6d32c463abad0`.
  All earlier checks pass. Full v5 output-only32K/128 gate already passes;
  do not waive this internal mismatch or infer a new target from Rust alone.
- Validation and downstream measurement queue stopped. Original failure/logs
  remain in W. Running `capture-hc-v5.py`: two fresh pinned-v5 captures each
  at16K,20K,24K,12K, full43-layer4096-row HC tensors plus output logits.
  First16K capture must reproduce Rust's differing layer34 checksum before
  target replacement; all paired inventories must match exactly. Large success
  payloads compact only after full SHA/FNV inventories; failed payloads retained.
- Next: if repeatable, version HC inventories separately and update only the
  explicitly versioned runtime table; preserve v3 fixtures and failure. Rebuild
  and repeat the complete six-gate candidate queue before any timing.

## 2026-09-07 — Fresh Rust v5 full-output C0 gate passes

- Frozen9a519a36 output-only32K run exits0: all129,280 prefill logits and
  all16,547,840 FP32 words over128 decode steps match the new v5 fixtures
  exactly. Selected85166 and the entire historical token transcript persist.
- This validates the coordinated selector port end to end against v5, without
  intermediate correctness collection in the prefill path. Fresh diagnostic
  32K mode is now running, followed by2K/8K and retained lifecycle controls.
- Paired baseline remains pending all gates; diagnostic timings are ineligible.

## 2026-09-07 — Rust12K internal control passes on v5 selector

- Frozen candidate9a519a36 passes the12K control: all43 layers, including
  historical prefix/8K/12K HC checks. Process exits0 (223.276s diagnostic).
- Full32K output-only C0 gate is running, then diagnostic32K,2K/8K and retained
  two-cycle checks. No paired run has started and no new speed claim is made.
- Prepared fail-closed paired-plan/export/standalone-verifier scripts in W.
  Full new v5 tensor fixtures will be included in the portable report so every
  FP32 row and both repeated reference inventories can be audited offline.

## 2026-09-07 — Rust v5 candidate built; live gates started

- Candidate `9a519a36a4e7670c6d67ea3b283df876b6999558` builds and passes310 Rust tests.
  Executable SHA `e512f5e4711622a5fc9b08ddbe2a4580624bf5bbfb4593d646a77fad5962be12`.
  Python suite93/93; both new differential fixture validators pass.
- V5 source ancestry retained via explicit ours merge, preserving the research
  runtime and historical root sources. Reference benchmark remains ccbd16d.
- Sequential live queue starts12K internal, then32K output-only and diagnostic,
  2K/8K controls, then two32K retained reset/reuse cycles. Stop on first failure;
  no paired timing before all gates pass. Logs and build hashes in refresh W.

## 2026-09-07 — V5 full decode repeatability and versioned fixtures pass

- Two fresh128-step runs match all129 full FP32 output rows exactly.
  All rows are finite; global argmax equals EOS-excluding selection throughout.
  The complete129-token transcript is unchanged from v4. Decode logits differ
  in16,547,704 of16,547,840 FP32 words, with the selector-only prefill change
  already isolated. This is an explicit target revision, not a tolerance change.
- Final-source8K capture matches historical fixture exactly. Tokenizer proof
  matches all32,768 input IDs; EOS ID1 validated from pinned tokenizer.
- New immutable fixtures `prefill-frontier-32768-v5` and
  `prefill-decode-32768-v5` preserve all historical files. Rust embeds/fixture
  identity now select v5; fresh Rust build, C0 and lifecycle checks follow.
- Evidence: `conformance-v5.json`, both full row inventories/binary traces,
  `v4-v5-trace-comparison.json`, `tokenization-v5.json` in refresh workdir.

## 2026-09-07 — Final oracle-v5 repeated frontier acceptance passes

- Final pinned source ccbd16d passes official correctness and four identical
  fresh captures at both2K/32K. Portable oracle bundle verifier passes16
  artifacts. Archive SHA77243250cedc5296747528b6f8c81b4e98bd796294dac38965f6c4cd1817039e.
- Benchmark executable for full trace/new pairs SHA
  a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0.
  Build provenance records pinned source hashes and no inference overlay.
- V5 raw tokenizer matches all32,768 historical prompt IDs.8K final-source
  control and then two129-row traces are running sequentially. Milestone
  remains incomplete until versioned fixtures, Rust gates, baseline/report.

## 2026-09-07 — Final-source capture and v5 acceptance guard tests

- Final-source oracle-v5 capture is running (four repetitions per2K/32K).
  Queue `finish-oracle.py` verifies the completed bundle, then validates raw
  tokenization and two full129-row traces before importing new v5 fixtures.
  It stops on any failed check and retains raw JSON and binary trace evidence.
- Existing Python suite passes91 tests. Two new artifact tests verify v5
  acceptance and rejection of only three repetitions; artifact suite18/18.
- Scoped source audit records that Rust already uses layer-local inverse-RoPE
  fusion state and separate KV-store dispatches, avoiding upstream's two
  affected mechanisms. M3/M5-only fusions stay outside M1 port scope.
  Evidence: `upstream-port-scope.json`. No new performance claim yet.

## 2026-09-07 — New32K logits attributed exactly to selector change

- V4 control with ONLY streaming-selector dispatch removed produces
  `fb827b1c...d4753b`, matching all129,280 FP32 words from pristine upstream.
  This isolates the new32K tensor to deterministic argsort/merge selection,
  not unexplained arithmetic changes across the upstream history. Old repaired
  streaming selector and new argsort have different selection policies; do not
  retroactively call v4 fixtures v5 or relax bit-exact comparisons.
- `selector-attribution.json` binds control/new tensors; all source/build logs
  are retained. Control8K is running. Next final-source v5 acceptance captures
  and full128-step traces, versioned fixtures, and live Rust full-output gates.

## 2026-09-07 — V5 candidate pinned; Rust selector tests pass

- Oracle candidate source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, tree
  `a0a33f4a00c3c78cd40350785485203e7c6b0b1d`, based on pinned c0a6119f.
  Changes are sampling-inclusive driver/full-step dumps and hardware-aware
  official tests with unchanged production admission. No old host-wait overlay.
- Registered candidate v5 capture/build identities; registration is not oracle
  acceptance. Final repeated capture/full-trace validation remains required.
- Rust now uses exact upstream argsort kernel source privately, avoiding any
  mutation to old DwarfStar shared kernels. Both streaming selector call sites
  use upstream block-sort/merge scheduling with accurate merge dispatch counts.
  Batched even-layer tails remain intact. Corrected Rust suite passes310/310;
  final release build is finishing. Live output acceptance still pending.
- Selector-only V4 control32K is still running. Complete attribution before
  replacing target versions; preserve old fixtures and reports indefinitely.

## 2026-09-07 — Upstream repeatability and repaired official gates pass

- Four fresh pristine upstream captures at each 2K/32K match within context.
  2K preserves v4 SHA 7b5e8518...bfaf.32K SHA is
  `fb827b1c1ebc44933c35cd7d6e724c279dfb90035c9dbd209c49e893b3d4753b`,
  differing from v4 in 129,276/129,280 words despite the same selected85166.
  Not yet an accepted replacement: explain the semantic delta first.
- Corrected M1-aware kernel suite and official logprob-vectors both PASS.
  Optional M3/M5 kernels remain disabled on M1; fallbacks retain full checks.
  Deterministic sampling/feedback timing harness also passes zero/three steps.
- V4 control built with ONLY streaming-selector dispatch removed. Running
  32K/8K captures to isolate whether argsort/merge explains new logits.
- Rust deterministic-selector port builds. Initial unit run309/310: one stale
  report-count string, corrected with merge dispatch accounting. Rebuild/tests
  running. No live Rust acceptance or new paired timing yet.

## 2026-09-07 — Upstream M1 test admission mismatch localized

- Pristine upstream builds. Official --metal-kernels exits 1 with 29 assertions:
  tests require compressor-store and gathered-KV fusions whose production
  guards deliberately admit only M3/M5. No model capture had run at failure.
  Original log/build/hash records retained; no acceptance claimed.
- Oracle worktree test repair shares the existing M3/M5 hardware predicate,
  asserts optional fusion refusal on M1, then exercises production fallback
  and retains complete projection/state/KV output and guard-byte checks.
  Runtime admission is unchanged; rebuilding test binary only while the
  original frozen benchmark captures run. Corrected suite still pending.
- First pristine upstream 2K capture matches all v4 FP32 logits exactly.
  First32K then four alternating repeats/context are running. No host-wait
  patch has been carried forward. Full model hash freshly verified.
- Compressor APE table batching already exists in Rust as one score+APE fused
  kernel. Do not duplicate upstream's weaker copy-then-add sequence. M3/M5-only
  fusion guards are not an M1 speed claim despite broader historical titles.

## 2026-09-07 — Upstream refresh milestone authorized and started

- User authorized end-to-end completion. Milestone: pinned upstream oracle-v5
  acceptance, applicable resident M1/Q2 Rust ports, full C0/lifecycle gates,
  fresh five-pair baseline and separate profile, durable report and journal.
- Work: `rust-star/.work/upstream-refresh-1190237b/`. Isolated oracle worktree
  `rust-star/.work/oracle-v5-source/`, branch `codex/oracle-v5`, starts from
  upstream c0a6119f. Build pristine upstream first, then check repeatable full
  outputs before deciding whether v4's host-wait repair remains necessary.
- Preserve all old targets and failed attempts. No automatic target replacement
  or unmeasured speed claim. Audit sampling-inclusive benchmark timing, which
  upstream's current driver does not yet satisfy for our paired contract.
- Rust port scope is the resident M1/Q2 path and reference compatibility;
  unrelated GLM/MXFP4/M5/TP/ROCm/SSD features remain outside this milestone.
  One large model process at a time; no heavy builds during timed measurements.

## 2026-09-07 — Upstream refresh assessment supersedes attention tuning

- User questioned optimizing before integrating upstream correctness/performance
  improvements. Fetched origin/main; reviewed pinned tip
  `c0a6119f363ef82125877142f13fb3fe491cba14` (241 upstream-side commits absent
  from oracle-v4 ancestry). No implementation merge, build or new model run.
- Relevant changes: 0658dac8 fixes inverse-RoPE fusion state and device-memory
  fencing; 023614ea removes non-reproducible streaming top512; e6a6ab51 ports
  exact decode compounds to M1–M4 (reported measurements are M3, not our M1);
  8cbdd762 batches compressor position-table expansion; 85a4f0e strengthens
  complete score/frontier validation. Many other gains target GLM, MXFP4,
  M5, ROCm, TP or SSD and are not direct resident M1/Q2 gains.
- Rust currently requires kernel_topk_stream512 during pipeline setup, so a
  wholesale shared-kernel replacement is incompatible without a coordinated
  wrapper port. Benchmark driver also differs; preserve sampling/timing fixes.
- Recommended next milestone: pin upstream in an isolated oracle worktree,
  audit which fork-only repairs remain necessary, establish repeatable oracle-v5
  at 2K/32K plus full128-step logits, port relevant changes with Rust C0 and
  lifecycle controls, then freeze a fresh paired baseline. Explain any changed
  logits before accepting a new version; never overwrite historical fixtures.
- Preserve v4 and completed reports. Attention optimization is deferred until
  the refreshed baseline/profile establishes that it is still the priority.
  This records an assessed recommendation, not a completed oracle upgrade.

## 2026-09-07 — Prefill optimization milestone complete

- Profiled all native32K continuation layers/stages, identified the dominant
  even-layer tail, and replaced 18,816 32-row tail calls with 147 4K calls.
  Arithmetic kernels and accepted oracle-v4 are unchanged. Source candidate
  `dd0d26c5424acd3e8b27606771c46ed3c71e715a`; executable SHA-256
  `5d05eb14b3288f2233095e869a8f422dc9630f71484f7580ea482f3772e74b56`.
- All live gates pass: internal12K, two complete32K full-logit/GPU-selection
  modes, exact 2K/8K, two retained32K reset/reuse cycles, 310 Rust tests and
  91 Python tests. Internal HC acceptance still ends at 24K.
- Completed five valid AB/BA/AB/BA/AB pairs and both warmups. Confirmed host
  restart interrupted pair 5 attempt 1; retained it and reran both engines
  under the external-interruption protocol. No performance-based exclusion.
- Independently audited median prefill: Rust 521.130 ± 21.751 MAD s versus
  reference 184.333 ± 5.872 s. Prior Rust median 813.107 s: 35.91% latency
  reduction / 1.560x throughput across campaigns; reference-normalized gain
  1.521x. Remaining latency gap 2.827x. First four paired throughput ratios
  median 0.34065; post-restart pair 0.35696; all-five median 0.35133.
- Steady decode: 19.752 ± 0.201 versus 19.595 ± 0.363 tokens/s, paired median
  1.00368 (range 0.96658–1.05343): near parity, no robust decode gain.
  Median peak process memory 87.585 versus 80.893 GiB; prior Rust 91.645 GiB.
- Separate before/after profiles reconcile 1,483 rows each. Even-tail GPU
  time falls 294.625 to 43.453 s. Largest remaining stage: even attention,
  77.645 s GPU / 124.838 s host over 18,816 calls. Profiles are excluded
  from speed samples; grouped bootstrap coverage is explicit.
- Durable report: `rust-star/reports/32k-prefill-dd0d26c5/README.md`.
  Portable verifier passes all 106 artifacts, 12 accepted measurement bundles,
  the interrupted pair's completed reference bundle, exact transcripts, both
  profile reconciliations and independent statistics. 116 host observations
  contain zero recorded thermal/performance warnings; restart caveat retained.
- Next milestone: investigate even-layer attention batching and command/setup
  costs, preserve C0 and lifecycle gates, then repeat the paired protocol.
  This completes the authorized milestone, not the 256K project headline.
  Journal, charter and runtime docs updated. No push or packaged release.

## 2026-09-07 — Five optimized pairs complete; after profile running

- All five complete pairs accepted. Pair 5 attempt 1 remains invalid due to
  confirmed host restart; both engines passed the complete replacement pair.
- Preliminary medians before independent audit: Rust prefill 521.130 s versus
  oracle 184.333 s, down from previous Rust 813.107 s. Steady decode 19.752
  versus 19.595 tokens/s; paired ratio 1.00368 remains near parity. Peak
  process medians 87.585 / 80.893 GiB. Candidate prefill MAD 21.751 s.
- Isolated after profile running. Reconcile stages, audit all evidence including
  interrupted pair, export portable report and finalize current-state docs.

## 2026-09-07 — Host restart confirmed; final candidate rerun active

- Resumed host observation shows uptime 8 minutes versus over 16 hours before
  interruption, with swap reset to zero. Host restart explains the lost
  sessions; restart trigger is unavailable. Evidence added to interruption
  record and preserved host-resumed-observation.json.
- Rerun reference passes. Final candidate is running, followed by isolated
  profile. Report must disclose the cross-restart campaign and host drift.

## 2026-09-07 — Final pair externally interrupted; complete pair rerun

- On continuation, execution session 56086 no longer exists and no runner or
  model process remains. Pairs 1–4 passed; pair 5 candidate is left running
  in state with no completed measurement. Termination cause is unavailable.
- Preserved `interrupted-state.json` and `interruption.json` in campaign work.
  Per BENCHMARK_PROTOCOL.md external-interruption rule, invalidate attempt 1
  of pair 5 and rerun BOTH engines with the same frozen plan. No result was
  excluded for speed. This is an explicit correction to the initial no-retry
  expectation; report the interruption and both attempts in final evidence.
- Final pair rerun, isolated profile, audit and export remain to complete.

## 2026-09-06 — Four optimized timed pairs pass

- Pairs 1–4 pass on first attempts in AB/BA/AB/BA order. Final pair 5
  is running reference-first. Zero invalid attempts. Separate after profile
  starts automatically after the timed block; audit and export remain pending.

## 2026-09-06 — Three optimized timed pairs pass

- Pairs 1–3 pass on first attempts in AB/BA/AB order. Pair 4 is running
  candidate-first. No invalid samples. Remaining: pairs 4–5, after profile,
  independent audit, portable report and final journal checkpoint.

## 2026-09-06 — Two optimized timed pairs pass

- Pairs 1–2 pass on first attempts in AB/BA order. Pair 3 is running
  reference-first. No invalid attempts. Finish remaining pairs and profile.

## 2026-09-06 — First optimized timed pair passes

- Both warmups and pair 1 (AB) pass on first attempts. Pair 2 (BA) is
  running candidate-first. No invalid samples. Complete all five pairs and
  isolated after profile before auditing comparative statistics.

## 2026-09-06 — Optimized five-pair campaign started

- All 310 Rust and 91 Python tests pass. Full model/prompt hashes freshly
  verified against baseline pins. Candidate dd0d26c5 and oracle-v4 unchanged.
- Frozen plan SHA-256:
  `a446a6be765efa8198aec8b6111803b2b998dd3f98926b3ffd6341dae69859db`.
- Work: `rust-star/.work/paired-32k-128-dd0d26c5-batch/`. Runner and host/load
  observer active; one model process at a time, no builds during measurements.
  Finish both warmups, all five AB/BA/AB/BA/AB pairs, isolated after profile,
  independent audit and durable export. Stop on invalid attempts; no retries.

## 2026-09-06 — Batched-tail live correctness and lifecycle complete

- All six frozen-candidate gates pass: internal12K, two full32K C0 modes,
  exact 2K/8K, and two retained32K/128 requests with exact transcripts.
- Final harness tests are running. Next freshly hash inputs, freeze the new
  five-pair plan against unchanged oracle-v4, finish all pairs and a separate
  after profile, then audit/export the report. No paired speed claim yet.

## 2026-09-06 — Batched-tail shorter controls pass; retained run active

- 2K and complete legacy8K controls pass, including every full output logit.
- Retained32K two-cycle reset/reuse is running as the only model process.
  Two fresh full32K C0 modes and internal12K already passed. Candidate/source
  remain frozen; paired measurements have not started.

## 2026-09-06 — Both batched-tail full32K C0 modes pass

- Fresh output-only and internal32K runs both pass all 129 full-logit rows,
  GPU selections and the immutable 128-step transcript. Internal checkpoints
  through 24K pass; later hashes remain diagnostic only.
- 2K/8K regressions and retained32K two-cycle reset/reuse are now queued/running.
  After these pass, run Python checks, rehash model/prompt, freeze the new
  five-pair plan, and collect the post-optimization profile after timed pairs.

## 2026-09-06 — First batched-tail full32K C0 gate passes

- `batch-32k-output` passes all 129 full-logit rows and production GPU
  selections, final token 14. Process duration 493.406 s is diagnostic only,
  not a paired speed claim. Earlier internal12K passed in 214.636 s.
- Fresh `batch-32k-internal` is running; 2K/8K and retained32K two-cycle
  validation follow. Candidate remains frozen as `dd0d26c5`.

## 2026-09-06 — Batched-tail 12K internal gate passes

- Frozen candidate `dd0d26c5424acd3e8b27606771c46ed3c71e715a`, executable
  SHA-256 `5d05eb14b3288f2233095e869a8f422dc9630f71484f7580ea482f3772e74b56`.
- All 43 layers match established internal targets through the 12K frontier.
  This early localization gate does not claim a full-output target at 12K.
- `batch-32k-output` is running; full internal32K, 2K/8K and retained32K
  follow sequentially. Queue stops on failure. No paired timings admitted.

## 2026-09-06 — Batched-tail candidate built; correctness queue ready

- The new candidate builds and all 310 Rust tests pass. An initial compile
  caught a guard parameter-name typo; corrected, with the failed log retained.
- Predeclared live gates: 12K internal checkpoints for early localization,
  fresh 32K output-only and internal/full-output runs, 2K/8K controls, then
  retained32K two-cycle reset/reuse. Stop on any failure and preserve evidence.
- Freeze the candidate and start `validate-batch.py`; no paired measurement
  or performance claim until these gates pass.

## 2026-09-06 — Profile verified; batching the dominant even-layer tail

- Before-profile passes its exact transcript, all continuation-layer coverage,
  nonnegative stage counters and aggregate GPU/wall reconciliation: 1,483 rows.
- Even tails: 294.625 s GPU / 345.976 s host across 18,816 calls. Odd batched
  tails: 40.337 s GPU across 140 calls. Even attention: 73.932 s GPU; odd
  ingress: 61.540 s. Binding is only 3.875 s host, rejecting the hypothesis
  that binding itself explains the aggregate 94 s setup cost.
- Selected optimization: retain even-layer attention heads inside each existing
  tile command, then run the unchanged tail kernels once per 4K chunk (147
  calls). No shared Metal kernel or oracle change. Guards require consecutive
  tiles and all 4K rows before tail execution. The reusable buffer participates
  in existing prefill ownership/reset cleanup.
- Implementation is unvalidated. Build/tests, early internal C0 localization,
  repeated complete 32K outputs, 2K/8K and retained controls follow. Keep any
  failed attempt; do not claim a gain before the new paired schedule.

## 2026-09-06 — Unoptimized 32K stage profile running

- Frozen profile commit `3c77afe8325b1a28fbb0fb4fe05e2ec96c81d78f`, binary
  SHA-256 `27b0d00db42967bc8417595ae0529d2e52f712d15e737d24eb809df8d8da4365`.
- `profile-before-rust-star engine-profile ... --context 32768 --gen-tokens 128`
  is running as the only model process. Outputs: `profile-before.json/.log`
  and `profile-before-process.json` in the milestone work directory.
- `analyze-profile.py` verifies exact transcript, all-layer coverage and stage
  sum reconciliation before any optimization decision. Do not start another
  model process or treat the instrumented timings as paired samples.

## 2026-09-06 — Native32K stage profiler ready

- Added profile-only chunk/layer stage rows using existing command GPU/wall
  counters and host timers. Continuations separate binding, ingress/compression,
  attention and tail; shared bootstrap ranges and prefix layers are explicit.
  No extra GPU waits or arithmetic changes. Profile presence alone blocks
  paired eligibility; the new regression test passes (310 Rust tests total).
- Release build passes. Freeze this profiling executable and run the unchanged
  32K/128 workload before selecting an optimization. Live profile pending.
- Removed 10.55 GiB of duplicate products generated in an older cache path;
  older entries and frozen evidence were preserved. Builds now use the existing
  `rust-star/.work/runtime-target` cache explicitly.

## 2026-09-06 — Prefill optimization milestone started

- User authorized completion end to end. Starting from `fe73defe` on the
  existing Rust Star worktree, with accepted oracle-v4 and immutable targets.
- First add profile-only chunk/layer stage attribution for the native32K
  scheduler, keeping production timers and arithmetic unchanged. Investigate
  even-layer 32-token attention/FFN tails versus odd-layer 4K batched tails.
- Work: `rust-star/.work/prefill-milestone-fe73defe/`. Profile before selecting
  an optimization; validate full logits, shorter controls and retained lifecycle,
  then freeze and complete a fresh five-pair comparison and report.

## 2026-09-06 — 32K paired baseline milestone complete

- Completed both warmups, all five first-attempt AB/BA/AB/BA/AB pairs, and
  the separate 32K/128 profile. Zero invalid samples in the repaired plan.
  Independent audit rehashes all twelve measurement bundles, verifies frozen
  binaries/transcripts, recomputes all distributions/ratios, regenerates the
  official summary byte-for-byte, and validates profile stage shares.
- Durable report: `rust-star/reports/32k-baseline-cbb078bb/README.md`.
  Portable verifier passes all 149 evidence artifacts (excluding its manifest).
  Exported oracle-v4 acceptance independently verifies all 16 artifacts.
  Original private plan and one test log have explicitly recorded portable
  renderings; raw measurements and oracle artifacts retain original bytes.
- Steady decode medians: Rust 19.574 ± 0.144 MAD versus reference
  19.454 ± 0.076 tokens/s. Paired median ratio 1.00616, range
  0.98766–1.02341: near parity, no robust speed advantage established.
- Prefill medians: 813.107 versus 187.890 s, a 4.33x Rust latency gap.
  Peak process medians: 91.645 versus 80.893 GiB. Transformer setup median
  94.357 s and transformer GPU median 568.313 s warrant finer prefill
  attribution; overlapping interval medians must not be summed.
- Separate decode profile: routed experts 21.79%, attention cache/routing
  18.64%, QKV ingress 16.21% of transformer GPU time. Profile is perturbed
  and excluded from speed samples. At 32K it does not profile prefill stages.
- All 140 timed-block host observations show no recorded thermal/performance
  warnings. Desktop background applications remained open and are recorded.
- Next milestone: instrument 32K prefill per chunk/stage across all 43 layers,
  separate setup/encoding/waits/GPU work, implement the largest measured
  C0-safe improvement, then repeat this protocol. Resource reuse is a
  hypothesis for setup savings, not an established optimization result.
- Native layer work and this development baseline are complete; 256K primary
  headline, broader context support and agent co-design remain future work.

## 2026-09-06 — Five repaired pairs complete; profile running

- Both warmups and all five AB/BA/AB/BA/AB pairs pass on first attempts;
  zero invalid samples. Runner exited paired measurement with status 0 and
  started the separate 32K/128 candidate profile.
- Initial completed summary (independent audit still pending): steady decode
  median Rust 19.5739 versus reference 19.4540 tokens/s; median paired ratio
  1.00616, range 0.98766–1.02341. This is near parity, not a robust speed win.
- Median prefill Rust 813.107 s versus reference 187.890 s: a substantial
  4.33x latency gap. Candidate transformer setup median is 94.357 s;
  transformer GPU median 568.313 s. These aggregate timings do not identify
  an individual prefill kernel. Peak process medians: 91.645 / 80.893 GiB.
- Finish the profile, independently audit all artifacts/statistics, and export
  the durable report before marking the milestone complete.

## 2026-09-06 — Four repaired timed pairs pass

- Pairs 1–4 pass on first attempts in AB/BA/AB/BA order. Final pair 5 (AB)
  is running reference-first. No invalid attempts; isolated profile follows.

## 2026-09-06 — Three repaired timed pairs pass

- Pairs 1–3 pass on first attempts in AB/BA/AB order. Pair 4 (BA) is running
  candidate-first. No invalid attempts so far; pair 5 and profile follow.

## 2026-09-06 — Two repaired timed pairs pass

- Pairs 1 (AB) and 2 (BA) pass on first attempts. Pair 3 (AB) is running.
  No invalid attempt so far; finish all five pairs and separate profile.

## 2026-09-06 — First repaired timed pair passes

- Pair 1 (AB) passes on both engines, first attempt. Pair 2 (BA) is running
  with candidate first. Warmups remain passed; no invalid attempt so far.
- Complete the predeclared five pairs and isolated profile before statistics
  or comparative claims. Evidence remains under `paired-32k-128-cbb078bb-topk`.

## 2026-09-06 — Both repaired benchmark warmups pass

- Reference and candidate fresh-process warmups pass under frozen plan
  `c251b3c8...ab5f80`. Timed pair 1 (AB) has started; no completed pair yet.
- Work remains `rust-star/.work/paired-32k-128-cbb078bb-topk/`. Continue all
  five pairs and separate profile without rebuilding or changing the plan.

## 2026-09-06 — Correctness accepted; repaired five-pair schedule started

- All four fresh candidate 32K/128 full-output runs and 2K/8K controls pass.
  Two independent reference 129-row traces pass, in addition to oracle-v4's
  four fresh captures at each 2K/32K frontier and official correctness tests.
- Candidate `cbb078bb79871169f6a1dd367f96c141969e917e`, executable SHA-256
  `0baadfd5c4e24633817a1d2f9073e4cff0bb0fe91905e77298f7e9a0b34ee663`.
  Oracle-v4 `df99eba7f722f8170dc2960b8b4666f73818a5ba`, executable SHA-256
  `e9d9595c37aded826edd3a4c16b49ff854a4da285996dab4486063a1ca57abda`.
- New frozen plan SHA-256:
  `c251b3c894ce9efeed12cdc9f083bd0329d195fdee6d46dc5a7bfcbb59ab5f80`.
  Work: `rust-star/.work/paired-32k-128-cbb078bb-topk/`.
- Runner started: one warmup per engine, five AB/BA/AB/BA/AB pairs, then a
  separate candidate profile. No builds or heavy tests during timed work.
  Host power/thermal/swap/load observations are retained. One model at a time.
- Sampling metadata accurately distinguishes Rust's global lowest-ID argmax
  from the reference's EOS exclusion: complete immutable logit traces prove
  the same non-EOS selections for this fixed 128-step workload only.
- Old rejected schedules remain untouched. No completed valid pair or speed
  claim exists yet. Finish the schedule, independently audit, export durable
  evidence and report, then identify the next optimization from the profile.

## 2026-09-06 — All four repaired 32K candidate repetitions pass

- Output-only a/b and internal c/d all pass every one of the 129 full-logit
  rows and production GPU selections. Both internal runs pass established
  checkpoints through 24K; later diagnostic hashes are not promoted targets.
- `topk-d` completed in 852.631 seconds. The sequential queue is checking
  2K/8K regressions before preparing the new five-pair plan.

## 2026-09-06 — First repaired internal-checkpoint repetition passes

- `topk-c` passes established internal checkpoints through 24K and all 129
  complete output rows with production GPU selection in 850.371 seconds.
- Later internal hashes remain diagnostic, not promoted targets. Fresh d is
  running; shorter controls and the frozen paired schedule remain pending.

## 2026-09-06 — Both repaired output-only repetitions pass

- Fresh `topk-a` and `topk-b` both pass every prefill/decode FP32 word and
  production GPU selection for the complete 32K/128 workload.
- Internal-checkpoint c/d and 2K/8K controls remain. The queue has started c.
  No paired timings have been admitted yet.

## 2026-09-06 — Repaired Rust candidate first full gate passes

- `topk-a` passes all 129 full-logit rows and production GPU selections in
  820.534 seconds. Candidate remains the frozen `cbb078bb` executable.
- Fresh output-only b is running; internal c/d and 2K/8K controls follow.
  Evidence: `topk-a.json`, `topk-a.log`, and `topk-validation.json` in the
  current work directory. Timed paired measurements have not started.

## 2026-09-06 — Frozen oracle-v4 complete decode conformance passes

- The separately frozen benchmark executable passes two fresh 32K/128 runs:
  every FP32 word in all 129 rows and every greedy selection matches the
  unchanged immutable fixtures. Evidence: `conformance-v4.json` and the two
  per-run verification files in `paired-32k-128-7406473-sampling/`.
- Candidate validation is running sequentially: output-only a/b, internal
  c/d, then 2K/8K controls. The queue stops on any failure. No timed baseline
  starts until all six pass; old rejected schedules remain preserved.

## 2026-09-06 — Oracle-v4 repeatability and unchanged targets accepted

- Official Metal kernels and logprob vectors pass. Four fresh conformance
  captures at each of 2K/32K pass; independent bundle validation verifies
  all 16 artifacts. Every FP32 word equals the existing immutable target.
- Oracle-v4 is explicitly pinned to `df99eba7` / tree `14ffb83c`. The shared
  streaming top-k race is repaired; no target tensor was replaced or relaxed.
- Evidence: `oracle-v4-acceptance/`, `oracle-v4-verification.json`, and
  `oracle-v4-target-equivalence.json` under the current work directory.
- The separately frozen benchmark executable is now checking two complete
  129-row 32K/128 traces. Four fresh Rust gates and 2K/8K controls follow.

## 2026-09-06 — Streaming top-k count race reproduced and repaired

- Projection-wait experiment failed at 12K layer 14 in synchronized run c:
  HC actual `0x9b11372e4070eb8f`, expected `0x08b9db35dec40f27`.
  Queue stopped; d and shorter controls did not run. The insufficient
  projection-wait changes have been removed; all evidence is preserved.
- Found an intra-kernel race in `kernel_topk_stream512`: after insertion's
  barrier, faster SIMD groups can increment the next batch's count before
  slower groups snapshot the current count and decide whether to compact.
- Standalone unmodified kernel passes 2,000 normal repetitions, but a legal
  test-only SIMD scheduling delay reproduces wrong top-k indices immediately:
  repetition 0, row 0, rank 9, actual 1132 versus expected 1191.
- Added a barrier after every thread snapshots count. Same adversarial test
  passes all 2,000 repetitions / 64,000 rows / 32,768,000 selected indices.
  Regression tests independently CPU-sort scores across 1025, 3072, 8192,
  and 8224 columns with normal and uneven progress. All pass.
- The kernel is shared by the reference. Pinned explicit candidate oracle-v4
  source `df99eba7f722f8170dc2960b8b4666f73818a5ba`, tree
  `14ffb83c44382662fb02a4fe50d713dd55827726`, on local `codex/oracle-v4`.
  It differs from v3 only in the streaming barrier and sampling-inclusive
  benchmark driver. V4 is not accepted until its full correctness gates pass.
- Immutable target tensors remain unchanged. Rebuilds and fresh candidate /
  reference conformance follow; no paired comparison is currently valid.

- Repair/tooling checkpoint `45ed501a`, followed by merge `cbb078bb` to
  retain the pinned oracle-v4 producer in branch ancestry. Rust candidate
  SHA-256 `0baadfd5c4e24633817a1d2f9073e4cff0bb0fe91905e77298f7e9a0b34ee663`.
  Full source hashes are in `topk-barrier-build.json`.
- All 91 Python tests and 309 Rust tests pass; both release builds pass.
  Oracle-v4 official Metal/logprob correctness tests pass. Four 2K/32K
  conformance repetitions are running. `run-topk-validation.py` will verify
  the bundle and unchanged targets, then run two complete reference decode
  traces, four fresh Rust 32K gates, and 2K/8K controls, stopping on failure.

## 2026-09-06 — Testing oracle-v3 compressor synchronization in Rust

- Source audit found that Rust's shared odd/even long-prefill compressor
  helpers lacked oracle-v3's completed projection-to-consumer boundary.
  Reference commit `d35fb12d` documents an M1 intermittent hazard that survived
  encoder-only and asynchronous command splits. This is a repair hypothesis,
  not yet proof of the Rust failure's exact origin.
- Added a host-waited projection boundary to those shared helpers, preserving
  arithmetic and counting both command buffers in wall/GPU timing totals.
  Building the synchronized candidate while the preserved unsynchronized
  internal-checkpoint diagnostic continues.
- Failure-capture build and all 309 Rust tests pass. The first enhanced gate
  already reproduced a full-prefill logit mismatch before decode; original
  paired schedule remains blocked and must not be resumed.

- Synchronized release build and all 309 Rust tests pass. Frozen diagnostic
  binary: `compressor-wait-rust-star`, with exact source/binary hashes in
  `compressor-wait-build.json`. Four fresh 32K gates (output-only a/b, internal
  c/d), then 2K and 8K controls, are queued after the current unsynchronized
  localization process. Queue stops on the first failure.

- Unsynchronized internal-checkpoint run passes all full outputs and GPU
  selection. Its established checks through 24K pass; later diagnostic hashes
  agree with the prior passing capture. This does not clear the observed
  intermittent failure. Synchronized output-only run a is now running.

- Synchronized output-only run a PASSES all 129 full-logit rows and GPU
  selections. Fresh run b is active; c/d and 2K/8K remain. One success does
  not establish restored stability after the intermittent negative evidence.

- Synchronized output-only run b also PASSES all full outputs and GPU
  selections. Internal-checkpoint run c is active; d and 2K/8K remain.

## 2026-09-06 — Intermittent prefill logit divergence reproduced

- Enhanced checked-top1 gate a fails at the completed 32K prefill, before
  decode: logit 0 actual `0xc11edc6b`, expected `0xc11e1889`. GPU argmax
  agreed with the same actual logits. This reproduces upstream numerical
  divergence; the original token failure cannot be dismissed as selection only.
- Starting an internal-checkpoint diagnostic to localize the earliest divergent
  chunk/layer. Existing passing captures remain historical evidence, not proof
  of stable current correctness. Benchmark remains blocked.
- Added failure-only completed-logit inspection to the engine so another
  token failure reports CPU argmax and reference-word differences without
  recomputing the failing row. Passing runs retain the small GPU readback.

## 2026-09-06 — Corrected paired baseline blocked by candidate C0 failure

- Corrected plan `871c8762...f82ab` stopped at timed pair 1: candidate
  transcript mismatch at generation step 9, actual 1337 versus expected 7487.
  Exit 1, no timeout; process wall 815.748 seconds, peak 97,420,083,200 bytes.
- Both warm-ups passed and the timed oracle half passed. Those successes
  do not override this failure. Zero valid pairs; no comparative baseline
  or speedup claim can be made. Failed artifacts remain intact.
- No thermal/performance warning was recorded; system swap stayed near 2 GiB.
  Treat this as an engine correctness defect, not an external-event retry.
- Started a separate full-output diagnostic with the unchanged candidate to
  localize prefill versus decode divergence. Do not resume the failed plan.
  Correctness reliability now gates completion of the baseline milestone.

- Independent GPU argmax fixture stress passes all 12,800 selections
  (100 repetitions of 128 immutable full-logit rows). This does not test
  live producer dependencies or clear the failed engine run.
- Existing full-output gate used CPU selection. Added diagnostic checked-top1
  mode to execute production GPU selection and compare it against the same
  completed logits before applying immutable full-output C0 checks. Timed
  engine inference remains unchanged; enhanced diagnostic build is underway.

- The unchanged candidate passes the separate full-output diagnostic: all
  129 rows exact. This does not clear intermittent/timing-path failure.
- Enhanced checked-top1 release build and all 309 Rust tests pass. First live
  enhanced 32K gate is running (`checked-top1-a.log`); no inference fix or
  restored benchmark eligibility is claimed.

## 2026-09-06 — Corrected paired plan frozen

- Timing repair committed as `53dc0dba`. Replacement plan SHA-256:
  `871c87620e28eecfcc7badc9ce905d911819c49adf19f0d2ef7ef99fbc3f82ab`.
- Candidate remains the accepted `7406473f` binary. Reference uses exact
  oracle-v3 inference plus the validated, pinned benchmark-driver repair.
- Starting both warm-ups and five fresh alternating 32K/128 pairs, followed
  by a separate profile. Evidence is in
  `rust-star/.work/paired-32k-128-7406473-sampling/`.

- Both corrected full-length warm-ups pass, including the exact Rust Star
  transcript. Timed pair 1/5 is running; no invalid attempts or retries.

## 2026-09-06 — Sampling-inclusive oracle validated twice

- Repaired oracle passes independent runs a and b: all 129 full FP32 rows
  and selected tokens match the unchanged accepted fixtures exactly.
- Both benchmark builds and all 88 Python tests pass. Committing the
  benchmark-only correction before freezing the replacement paired plan.
- Next: complete both warm-ups, all five alternating pairs, independent
  audit, and separate profile; publish the measured development baseline.

## 2026-09-06 — Paired32K timer-contract defect found and repair underway

- Source audit found that accepted `ds4-bench` places per-token timers after
  argmax. Its `gen_steady_tps` excludes sampling while Rust Star includes it,
  violating the written paired protocol. Earlier steady-decode comparisons
  using this driver are historical legacy-timer evidence, not evidence of the
  required sampling-inclusive primary metric; preserve their numbers unchanged.
- Stopped plan `65015ab5...047dd3` during the candidate half of pair 1.
  Both warm-ups and the completed oracle half remain intact; the partial state
  remains running and will not be resumed or retried. `rejected-plan.json`
  records the protocol defect and deliberate termination. This was not an
  engine/C0 failure or a result-dependent exclusion.
- Repair aligns each generation step as eval + next-token argmax/commit;
  initial selection is charged to prefill. CSV declares sampling inclusion
  and uses higher precision. The adapter rejects legacy timing CSVs.
- Added a deterministic C mock-clock test exercising the actual benchmark
  loop, including zero-generation behavior and closed-loop token feedback.
  It passes. One adapter test exposed error-validation ordering; retained
  the original missing-field check before the new sampling marker guard.
- Building a separate benchmark executable from exact oracle-v3 inference
  sources, overlaying only the benchmark driver. Original accepted executable
  and all inference sources remain unchanged. New full-output conformance,
  all Python tests, and a fresh predeclared plan follow before any timed run.
  Work: `rust-star/.work/paired-32k-128-7406473-sampling/`.
- Corrected oracle release build passes. All 88 Python tests pass, including
  deterministic timing/feedback and legacy-marker rejection. The first of two
  fresh complete 32K/128 full-output conformance runs is underway.
- Repaired oracle conformance a PASSES all 129 full-logit rows and every
  selection. Independent repeat b is running. All 54 non-driver source files
  match the pinned oracle-v3 tree exactly.

## 2026-09-06 — Controlled paired32K baseline milestone started

- User authorized choosing a meaningful next milestone and completing it.
  Scope: freeze the accepted `7406473f` candidate and oracle-v3, complete
  five predeclared alternating 32K/128 pairs plus both warm-ups, independently
  validate/aggregate all evidence, then profile and identify the next measured
  optimization. No speedup threshold and no early stop for favorable results.
- Preserve all failures; engine/correctness/capacity failures block the schedule
  rather than being retried to fill the table. One large model process at a time.
- Preparing private plan, correctness binding, executable/kernel snapshots,
  fresh model/prompt hashes, and privacy-filtered before/after host records.
  Evidence: `rust-star/.work/paired-32k-128-7406473/`.
- Model and prompt freshly rehashed and exact; candidate and accepted oracle
  executables frozen with pinned oracle Metal sources. Plan SHA-256:
  `65015ab5fb2ad9b03895b55de82bd999267e542fa9955a16325287ecb4047dd3`.
  Source `7406473f`; five 32K/128 pairs in AB/BA/AB/BA/AB order, one warm-up
  per engine. Oracle warm-up is running. Scoped runtime overrides are removed;
  thermal/swap/load observations are retained every 45 seconds. A separate
  32K profile starts only after successful paired-block completion.
- Oracle warm-up passes; candidate warm-up is running. No timed pair has
  completed yet, and no thermal/performance warning has been recorded.
- Both full-length warm-ups pass, including the exact candidate transcript.
  Timed pair 1/5 is running in AB order. No retries or invalid attempts.

## 2026-09-06 — Native 32K extension complete through the output head and decode

- Completed the requested end-to-end stage on `agent/rust-star-bootstrap`,
  extending `8e63d80e`. Native prefill executes all 43 layers through 32,768
  tokens, then adopts all layer histories on the GPU and greedily decodes
  128 positions (32768–32895).
- Two complete C0 runs pass: the diagnostic and production output-only paths
  each match all 129,280 immutable prefill logits and all 16,547,840 subsequent
  FP32 decode words. Every selected token matches; prefill 85166, final 14.
  Independent fresh oracle-v3 captures repeat the entire new decode fixture.
- Repeated internal HC checkpoints pass through 24K. The 28K/32K internal
  snapshots remain explicitly unverified (`hc_checked_layers:0`, null match),
  despite matching their preserved singleton captures. Full-output C0 remains
  mandatory and unchanged; no tolerance or oracle target was relaxed.
- Fresh32K normalized engine validation passes with correctness collection
  disabled and an exact transcript. Single-run prefill: 41.232422 tok/s;
  complete generation: 17.832623 tok/s; steady decode: 17.851319 tok/s.
  This is bring-up evidence, not a paired or headline speedup claim.
- Retained32K passes both exact 128-token cycles after request reset in one
  Metal context. Model mapping/residency are reused, request KV state is rebuilt,
  and context creation is correctly charged only once. Retained prefill:
  43.712198 tok/s; generation: 19.175227 tok/s. This lifecycle report remains
  explicitly paired-ineligible.
- Final live 2K and legacy8K regressions pass all established boundaries and
  all 129,280 output words; selected tokens 15342 and 77179 respectively.
- Final release build 14, all 309 Rust tests, all 86 Python tests, canonical
  decode-fixture verification, formatting, whitespace, four CLI help checks,
  and the cross-artifact audit pass. Build 14 differs from the live-tested
  build 13 only in CLI help and a source comment; inference is unchanged.
- Scoped autorelease pools prevent replaced long-prefill histories from
  accumulating. System swap stayed about 2.5–3.3 GiB through fixed output-only,
  fresh, and retained runs, versus about 89 GiB at the old diagnostic's exit.
  The normalized fresh process peak was 97,419,493,376 bytes. Reclaimed another
  7,534,338,048 allocated bytes (7.02 GiB) by removing ignored debug/test
  artifacts after the final tests. About 149 GiB disk space is now available;
  model, release executable, accepted fixtures and failure evidence are retained.
- Evidence: `rust-star/.work/native-32k-v31/final-verification.json` binds the
  source files, reports, tests, and executable hashes. Main reports are
  `native32k-a.json`, `native32k-output-only.json`,
  `engine32k/measurement.json`, `engine32k/engine-run.json`, `retained32k.json`,
  `2k-final.json`, and `8k-final.json` in that directory.
  Live-tested build-13 SHA-256:
  `a95d9effeba6914748a28ea82fb73c3e0187ad2f78fac2be7fc5f19341173228`.
  Final build-14 SHA-256:
  `8d9b607331fa4dce37caeff0a8ef7d2b9c3cc8f9285eb0590f347cca4a6aa84d`.
- Reproduce full-output correctness with `long-prefill-chunk-probe MODEL`
  and `long-prefill-chunk-probe MODEL --output-only`; both default to 32K.
  The engine and retained commands accept `--context 32768 --gen-tokens 128`.
  The accepted model/oracle hashes remain pinned in the fixture manifests.
- This implementation, imported fixtures, and synchronized journal/charter/docs
  form one local checkpoint. No push was performed. Next: an immutable paired
  32K development comparison and profiling, followed by broader contexts under
  the same C0 contract. There is no 128K/256K/1M correctness or speedup claim.

## 2026-09-06 — End-to-end native 32K extension requested

- User explicitly requested the next stage end to end through the last layer.
  Starting from `8e63d80` on `agent/rust-star-bootstrap`; the complete retained
  8K chain/output and 2K regression are established controls.
- Objective: native 32K prefill through all 43 layers and the output head,
  exact full-logit and decode-transcript validation against oracle-v3, and
  integration of the accepted frontier into the eligible engine.
- Inspection found the current continuation helpers fixed to positions
  4096--8191. Generalize chunk geometry and retained per-layer state, preserving
  the validated 8K path as a regression. Reuse the exact completed 8K context
  to advance later 4096-token chunks; do not import oracle activation state.
- Implemented dynamic chunk extents in bootstrap, paired ratio-4 compression,
  indexed attention, odd-layer ingress/compression/Flash attention, and HC tail
  scheduling. Added per-layer history ownership across reused executor slots,
  a generic Rust chunk driver, and a strict report for repeated layer outputs.
  The first build reached Rust and found four FFI return aliases missing from
  scope; changed them to the existing i32 convention. Rebuilding next.
- Independent captures now repeat all 43 layer HC outputs at 12K, 16K, and
  20K; 24K captures are running. Capture scripts keep compact hashes and logs
  while removing new temporary raw payloads after hashing/comparison.
- Corrected Rust boundary admission for later 4096-token frontiers. Release
  build 2 passes. Paired oracle layer-output checkpoints now pass through
  24K; 28K is running. Prepared an isolated source-identical oracle-v3 helper
  that will capture every full FP32 decode-logit row after 32K, alongside the
  token transcript. Its first build lacked a generated include from the
  source extraction; restored the pinned include and the build/link pass.
- Oracle repetition failed at 28K: layers 0--14 match, layers 15--42
  and frontier logits differ between independent runs. Preserved the second
  raw capture and both hash inventories; no 28K target accepted. All earlier
  12K--24K frontier logits also repeat exactly. The first native 12K gate is
  running while the oracle divergence is investigated.
- Native 12K gate passes: retained full 8K control plus all 43 new chunk
  HC outputs match exactly (`native12k-a.json`), selected token 8131. Build 3
  passes with guarded 32K decoder-history adoption and report rejection tests.
  Focused independent 28K layer-15 stage captures are now running to localize
  the oracle divergence. Imported compact accepted 12K--24K inventories.
- All 308 host tests pass. Build 4 passes with uniform continuation from
  the complete 4K prefix, preserving all repeated 8K HC targets and full
  logits as an internal control. This removes the redundant legacy diagnostic
  bootstrap from the new path; its live validation is next in the model queue.
  Focused 28K capture c matches original a at layer 15, but neither original
  full-logit result. Initial a/b layer-15 drift affects all 4096 rows
  (65,417,635 FP32 words, maximum absolute difference 0.29110569).
- Uniform 4K-to-8K attempt failed at layer 5 (`native24k-a.log`): the
  older monolithic 4K path was explicitly not a complete C0 prefix control.
  Added a checked per-layer prefix rebuild using the established ratio-4
  sparse and odd Flash/production-FFN schedules; build 5 passes, live test
  pending. Original retained-8K-to-12K pass remains valid.
- Both uninstrumented 32K oracle transcript captures repeat exactly: pinned
  prefill logits plus all 16,547,840 FP32 decode words across 128 steps, and
  every greedy selection (final token 14). Imported the full fixture. Added
  native full-row decode validation and guarded per-layer history adoption;
  pending live verification.
- Existing graph dumps synchronize/restart command batches. Prepared isolated
  diagnostic-only GPU snapshot hooks that retain HC via blits and read it
  after prefill finishes, preserving inference batch submission. Deferred
  32K/28K paired captures are running; 32K must match the immutable logits.
- Deferred run a preserves immutable 32K logits; deferred run b first differs
  at layer 19 and fails that terminal anchor. Both remain recorded; no
  unanchored long target accepted. Further snapshots must pass the immutable
  full 32K terminal logits and agree across two independent captures.
- Prefix rebuild test passes layers 0--3, then rejects the layer-4 raw-KV
  capacity: its sparse kernel requires 4,352 padded rows, not 4,096. Corrected
  the allocation. Added timing-only execution with optional diagnostic
  readbacks, complete command GPU/wall accumulation, and 32K engine/retained
  dispatch using the pinned token transcript. Build 7/live checks pending.
- Compacted redundant failed new captures, reclaiming 19.30 GiB while
  retaining SHA inventories, first divergent HC pair, and final HC evidence.
- Stopped instrumented retries after further terminal failures. The actual C0
  contract remains unchanged: exact immutable 32K full logits and every word
  of the independently repeated 128-step decode trace. Internal HC checks
  remain mandatory where repeated targets exist (prefix, 8K, 12K--24K);
  28K/32K snapshots are diagnostic only and explicitly report zero checked
  HC layers/null HC match. They cannot substitute for the full-output gate.
  No target or tolerance was changed. The corrected uniform 24K gate is
  running (`native24k-c.log`). This avoids blocking end-to-end validation
  on the separate instability introduced by oracle instrumentation.
- Uniform prefix all 43 layers, full 8K logits, and all 43 layers of the
  12K chunk pass in the corrected `native24k-c` run. Stopped the redundant
  partial-frontier run after that control (it had begun 16K) and started
  the full 32K gate (`native32k-a`, release build 8). All 86 Python tests
  pass; the new 66,191,876-byte decode fixture verifies its canonical envelope.
- Added `--output-only` for a second end-to-end C0 run using the exact timing
  path without internal tensor readbacks. Its report explicitly disables
  internal-HC claims while still requiring all immutable prefill and decode
  output words. The timed engine itself collects neither diagnostic trace.
  Build 9 is running; full native 32K, production repeat, and engine validation
  are still pending.
- Full native run now passes all 43 HC targets through 20K and is checking
  24K. Builds 9 and 10 pass. Added autorelease pools to new C ABI entry points
  so replaced history dictionaries release promptly, and kept retained
  context setup at zero because no context is created on that cycle. These
  lifetime/accounting changes will be covered by the output-only repeat and
  retained engine run.
- Full native run passes every repeated internal checkpoint through 24K:
  all 43 layers in the prefix and each validated chunk. It is entering the
  final two chunks, whose internal HC values are diagnostic; immutable
  prefill/decode output checks follow. Build 11 includes zero retained-context
  setup accounting and avoids output-fixture decoding in timed prefill.
- The old-build full diagnostic continues through 28K; swap reached about
  40 GiB despite earlier free-memory summaries. Recorded the actual swap/disk
  observation. The repeat must use the autorelease-pool fix and verify bounded
  lifetime behavior; do not use this old-build diagnostic for performance.
- First full native 32K C0 gate PASSES (`native32k-a.json`): all 129,280
  immutable prefill words (selected token 85166), all 16,547,840 decoder
  words over positions 32768--32895, and every greedy selection (final token
  14). All repeated HC checkpoints through 24K passed en route. Both later
  diagnostic HC arrays also agree with their singleton oracle captures,
  but those internal captures remain explicitly unaccepted as repeated targets.
- The old build accumulated severe swap by exit (about 89 GiB used at the
  immediate observation). Do not benchmark that build. Build 11 passes and
  contains scoped autorelease pools plus output-only production C0 validation;
  run that repeat before fresh/retained engine measurements.
- Build 12 passes with explicit native first-token feedback into the full
  decode gate. The output-only repeat has passed prefill through 24K; swap
  remains about 3.1 GiB with scoped autorelease pools (versus the old build
  accumulating tens of GiB). All 309 final Rust tests and 86 Python tests pass.
- Output-only production C0 repeat PASSES (`native32k-output-only.json`):
  all immutable prefill and 128 decode rows match the first native report,
  internal collection disabled, final token 14. Swap stayed approximately
  3.1 GiB through completion. Build 13 passes (32K CLI help included); all
  309 build-12 host tests pass. Starting sequential fresh32K, retained32K,
  2K and legacy8K validation with this final executable.
- Synchronized the charter's supported correctness frontier and runtime docs.
  Corrected stale manual-ledger M-005 to the documented August 27 completion;
  future extended captures now name oracle-v3 rather than the quarantined v1.
- Fresh final-build 32K/128 engine and normalized adapter PASS
  (`engine32k/engine-run.json`, `measurement.json`). Diagnostic collections
  are disabled; `paired_protocol_eligible:true`; all selected tokens match,
  final 14, checksum 11977321378026793466. This single bring-up measurement
  records 794714.414 ms prefill (41.232422 tok/s), 17.832623 complete-generation
  tok/s, 17.851319 steady tok/s, and 63.535709 ms first decode. Model warm-up
  was 13509.666 ms; process peak was 97,419,493,376 bytes. No comparative or
  headline speedup claim. Retained two-cycle validation is now running.
- Final review added the new command's standalone `--help` and clarified
  its context placeholder plus one source comment. This CLI/comment-only
  change awaits build 14 after the measurement queue; inference is unchanged
  from the successful build-13 fresh run. Retained validation is still active.
- Retained32K two-cycle lifecycle PASSES (`retained32k.json`): both
  128-token oracle transcripts exact, final 14, model/Metal context/residency
  reused after request reset. First-load prompt including warming: 815544.401
  ms; retained prompt: 749630.569 ms (1.087928x). Retained prefill 43.712 tok/s,
  generation 19.175 tok/s. This is explicitly paired-ineligible lifecycle
  evidence. System swap stayed about 2.5 GiB throughout. 2K/8K controls are
  running; building CLI-only revision 14 and preserving build-13 executable SHA.
- Final standalone2K regression PASSES (`2k-final.json`): all retained
  transformer boundaries and 129,280 full logits exact, selected 15342.
  Legacy8K is the last live model validation running in the queue. The
  validated build-13 executable SHA-256 is
  `a95d9effeba6914748a28ea82fb73c3e0187ad2f78fac2be7fc5f19341173228`.
- Work/evidence: `rust-star/.work/native-32k-v31/`. Final live engine
  validation is pending. Keep only one large model process active, bound
  temporary capture disk use, and continue through all layers without a
  per-layer approval request.

## 2026-09-06 — Complete remaining layers requested

- User explicitly asked to continue through every remaining layer. Active goal:
  layers 15--42 plus exact full 8K output logits, preserving the 2K gate.
- Starting checkpoint `d8ebff8`, branch `agent/rust-star-bootstrap`; about
  171 GiB available. Layer-15 prefix and continuation oracle captures are
  running as four sequential fresh processes with one dump position per run.
- Preparing shared odd/even remaining-layer drivers, retaining the existing
  validated arithmetic rather than duplicating a full per-layer scheduler.
- Layer 15's 20 independent payload pairs match. Its compact provenance and
  checksums are installed; layer-16 captures are running. Added shared odd/even
  Rust executors and dynamic layer binding helpers, plus per-layer report
  contracts. Initial build found host serializer placement and constant-pattern
  naming errors; corrected them, and the second release build passes.
  The live layer-15 gate passes (`gate15.json`): 1367 dispatches and 1199/1199
  mappings. Layer 16's 26 payload pairs also match; its target set is installed
  and the release build is running.
  Added a bounded prefix-HC handoff slot so later layers do not accumulate
  another full prefix-HC history.
- Layer-16 release build and live gate pass (`gate16.json`): 4724 dispatches,
  2227/2227 mappings. All 307 host tests pass with the new report/output rules.
  Layer 17's 20 and layer 18's 26 capture pairs match; targets 15--18 are
  installed and the layer-18 release build is running. A whole-chunk prefix
  capture pilot is running with all-layer dumps and independent repetitions.
  It must match all existing isolated 15--18 controls before adoption. This
  pilot keeps SHA-256/FNV/size metadata for both runs, removes first-run
  temporary payloads after hashing, and removes second-run payloads only after
  exact repetition and isolated-control agreement to keep disk use bounded.
  Added guarded 8K
  output-head support (only after completed layer 42) and full 129280-word
  comparison to the pinned existing 8K fixture. These final-head additions
  await host tests and a complete live chain; no full 8K claim yet.
- Whole-chunk prefix pilot passes all 487 repeated payloads and all isolated
  layer-15--18 controls. Temporary raw payloads were compacted only after their
  hashes were persisted (second repetition only after exact verification).
  The layer-18 build passed. Running gate18, then the 2K control, then the
  whole-chunk continuation pilot sequentially; stop the sequence on failure.
- Layer-18 live gate and the complete 2K regression pass. Whole-chunk
  continuation capture also passes all 487 repeated payloads and isolated
  controls. All four broad capture runs match all 129,280 accepted output
  logits. Installed compact targets through layer 42; release build passes.
  Added consumed activation-buffer release while preserving KV/compressor
  state. The first complete layer-42/output-head live gate is running.
- First complete gate passes layers 15--25, then rejects layer-26 prefix final
  HC (actual `14066919688155987687`, expected `14659157045767518558`). KQV and after-attention HC match. A fresh unchanged
  repeat is running to distinguish deterministic arithmetic from state drift.
  Final Rust tests: all 307 pass; Python tests: all 86 pass.
- Unchanged repeat fails at the identical layer-26 prefix hash. Isolated FFN
  oracle capture reproduces the accepted final HC. Oracle scores at row 547
  tie experts 60/218 for sixth place: stable insertion chooses 60; the pinned
  descending bitonic network chooses 218. Rust intermediate capture is running.
  Replaced batched router insertion sorting with the same bitonic permutation
  and changed its 41 call sites to one 256-thread group per row. Rebuilding;
  the 2K control must pass again because it shares this router.
- Diagnostic comparison confirms the routing cause: all FFN input/norm,
  logits/probabilities, and shared-expert tensors match exactly. Only row 547
  differs: six normalized routing weights, 4096 routed output words, and
  16384 final HC words. Evidence: `layer26-ffn-comparison.json` and
  `layer26-routing-ties.json`. Temporary dump code was removed.
  Fixed-router release build passes. Running `gate42-fixed`, then `2k-final`,
  then a fresh full repeat sequentially, stopping on any failure. Re-running
  host tests after the router change. No oracle targets changed.
- The repaired live gate now passes layers 26 and 27; continuing through 42.
- `gate42-fixed.json` passes the complete retained 8K chain and output head:
  all 129,280 FP32 words match the unchanged oracle; selected token 77179.
  All 28 new layers match their pinned targets and schedule/mapping contracts.
  All 307 Rust tests and 86 Python tests pass after the router repair.
  Final 2K control also passes every boundary and full logits (token 15342).
  The fresh complete repeat is running.
- Fresh `gate42-fixed-repeat.json` also passes every layer and all 129,280
  logits. Reports are identical after excluding wall/GPU timing fields.
  Final release build, 307 Rust tests, 86 Python tests, formatting, and diff
  whitespace checks pass. New report guards reject altered layer hashes,
  missing layer coverage, and invalid output claims.
- Final report SHA-256:
  - `gate42-fixed.json`: `0cdf187f5fe25e49df6fee8b5790ca66abc67ce3b24bd4c296738052f9623f7b`.
  - `gate42-fixed-repeat.json`: `ffecafc456f3b74d35371e0e73a480075db21a83c62e5c63a3744ab252253fda`.
  - `2k-final.json`: `8ed58b6164a3a6e2f8b376669858b20d7cb64a24e224770f9d9b92befcd91885`.
- Post-validation cleanup reverified and removed only exact duplicate raw
  layer-15/16 b captures (6.54 GiB) and ignored extra test-profile artifacts
  (10.47 GiB). Original a captures, all logs/hash metadata, pinned oracle,
  model, source, and production executable remain. Available disk: about
  154 GiB. Inventories: `duplicate-capture-cleanup.json` and
  `test-cache-cleanup.json`; broad capture compaction is recorded separately.
- All requested remaining layers and the output head are complete. Next:
  extend native prefill and the exact transcript/eligible engine gate to the
  accepted 32K frontier. No throughput or speedup claim was made.
- Evidence/work scripts: `rust-star/.work/remaining-layers-v30/`.
  All layers and the output head pass, including the fresh repeat and 2K
  regression. Included with the related implementation commit; consult
  `git log -1` for that checkpoint.
- Dion Labs packaged-release policy is acknowledged. Current remotes remain
  `deathcoder/ds4` and `antirez/ds4`; this is the Rust Star research branch.

## 2026-09-05 — Exact complete layer-14 retained continuation

- Extending `2e0ab3b` through complete layer 14 using the validated layer-12
  ratio-4 executor bound to retained layer-13 HC. New report fields, scheduling
  checks, and rejection tests are being added for v29.
- Two accepted-oracle-v3 fresh captures dumping both chunks match all 13
  prefix tensors but differ on all 13 continuation tensors, including ingress.
  These continuation captures are rejected as C0 targets. A second experiment
  restricts dumping to position 4096 to isolate first-chunk instrumentation.
- Local evidence: `rust-star/.work/layer14-continuation-v29/`; capture producer,
  model, prompt, and 4096-row geometry retain the existing pinned contract.
  Continuation-only captures c/d now match all 13 tensors exactly. Together
  with the 13 matching a/b prefix tensors, all 26 targets are independently
  repeatable and installed in the v29 constants. Compact provenance is tracked
  at `rust-star/fixtures/prefill-layer14-continuation-8192-v3/checksums.json`.
  Release build passed. Integrated `layer14.json` passes schema v29: all 26
  hashes match, with 40+40 boundary dispatches, 36 prefix-tail dispatches, and
  128 continuation tiles / 4608 tail dispatches. All model mappings match.
  The 2K full-logit regression also passes. All 307 Rust tests (release profile),
  86 Python tests, formatting, and diff whitespace checks pass. New negative
  report checks reject altered hashes, partial tiles, and lost pointer matches.
- Live probes ran sequential fresh processes on the pinned Q2 model and target
  Metal device, clearing inherited `DS4_`/`RUST_STAR_` overrides. Commands:
  `long-prefill-continuation-bootstrap-probe` and
  `prefill-layers012-attention-loop-probe`, each with `--json` in the evidence
  directory. No kernels or arithmetic variants were added.
- Report SHA-256: v29 `be4e4cc9bb4c42fd5888c52933d7efc36a9b155c93f6e89a488d46c6452ef72f`;
  2K `91c587e1837ea46333fb1f59a4e447b3d1d4c0f530bc51ff8efce05016ccce9a`.
- Instrumented runs a/c match the accepted 8K logits; b/d differ in all 129,280
  logits. The repeatable layer-14 targets remain bounded evidence; downstream
  instrumented logits are not replacement targets or a Rust 8K C0 claim.
- Removed only ignored extra test-profile artifacts after validation, reclaiming
  10.46 GiB; 41.38 GiB was available afterward. Production executable, normal
  build cache, raw captures (including rejected a/b continuation), and all logs
  remain available. Inventory: `test-cache-cleanup.json` in the evidence folder.
- Complete retained layer 14 is now exact. No complete 8K logits or benchmark
  claim. Next: proceed to layer 15.

## 2026-09-05 — Restored exact standalone 2K regression control

- Traced the layer-4 failure to the shared ratio-4 helper's final-tile guard.
  Commit `94485d4` added `position_start != 0`, excluding full single-batch
  prefixes from the small-batch tail projection previously used at 2K.
- The pinned oracle explicitly reprojects the final four normalized rows with
  the small-batch kernel to preserve recurrent-state accumulation order.
- Removed the start-position exclusion while retaining the end-of-buffer
  boundary test. Intermediate rolling tiles and explicit production-state
  rebuilds retain their existing scheduling.
- First release build passed. The 2K live run now passes layer 4 and stops at
  layer-5 KQV row 0 (`0x3e5e3921` versus `0xbc4ceaff`); retained in `2k.log`.
  This exposed a second shared-path inconsistency: `29d9521` switched the
  ratio-128 mask/strides to real key rows but updated only layer 3's padding
  and kernel. Layers 5--41 still used the non-padding kernel.
- Applied the validated layer-3 partial-block padding and eight-SIMD-group
  kernel setup to every subsequent ratio-128 layer. Each layer refreshes the
  padding after staging its own KV. Added 19 padding dispatches to the report.
- Second release build passed, but its run stopped at the host schedule guard:
  the added 19 padding dispatches also require updating the Rust report
  expectations. Updated the matching schedule contracts and report tests.
  Tensor validation did not execute in that run (`2k-fixed.log`).
- Third release build passed. `2k-final.json` / `2k-final.log` now pass the
  complete native 2K model, every retained transformer C0 boundary, full logits,
  and greedy selection (token 15342). All 1,216 transformer mappings remain
  no-copy. The integrated `layer13.json` / `layer13.log` also passes v28,
  preserving the complete retained chain through layer 13.
- Validation passed: release build; `cargo fmt --check`; 307 Rust tests via
  `cargo test --release` with `CARGO_INCREMENTAL=0`; 86 Python tests; and
  `git diff --check`. Both model probes used the pinned Q2 model on Metal with
  inherited `DS4_` / `RUST_STAR_` overrides removed, sequential fresh processes.
  Commands: `prefill-layers012-attention-loop-probe` and
  `long-prefill-continuation-bootstrap-probe` with `--json` report paths.
- Report SHA-256: 2K `101249213ddb5ca987d47c543b934277e00769465474446b0acae8102ef7d117`;
  layer 13 `56beeae726e28414cf1720bf39caf2ba277281d426b48cbe6aff8b9406f7f98d`.
- Removed only the ignored extra test-profile library, test executable, and
  their Cargo metadata after tests passed. Retained the production executable,
  normal release build cache, all fixtures and logs. Reclaimed another 10.45 GiB
  of test output; available space was 76.49 GiB afterward. Exact inventory:
  `rust-star/.work/layer4-state-repair/test-cache-cleanup.json`.
- Next: implement complete layer 14 from the validated v28 checkpoint.
  Evidence is being recorded in `rust-star/.work/layer4-state-repair/`.
  No fixture or comparator was changed; no benchmark was run.

## 2026-09-04 — Reclaimed rebuildable Cargo output

Objective:

- Respond to the user's low-disk-space cleanup request while preserving model,
  source, correctness evidence, and the executables needed for the layer-4 repair.

Actions and verification:

- Confirmed no Cargo, rustc, Rust Star, or DwarfStar model process was active.
- Removed the ignored `rust-star/runtime/target` tree and
  `rust-star/.work/runtime-target/debug` tree.
- Removed compiler intermediates, dependency archives, and Cargo metadata from
  `rust-star/.work/runtime-target/release` and
  `rust-star/.work/layer13-continuation-v28/baseline-target/release`, retaining
  each directory's tested `rust-star` executable at its original path.
- All removed paths were checked against Git ignore rules before deletion.
  No model, tracked source, fixture, raw capture, log, oracle executable,
  benchmark result, or worktree was removed. This also preserves both current
  and unchanged-baseline evidence for the pending 2K layer-4 investigation.
- Observed available space increased from 32,594,882,560 to 117,438,021,632
  bytes: about 79.0 GiB reclaimed, with about 109.4 GiB available afterward.
  `df -h` reports usage reduced from 99% to 94%.
- Both retained release executables successfully run `--help` after cleanup.
  No model inference or benchmark was needed for this cache-only operation.
- The exact removed-path inventory and space readings are retained locally in
  `rust-star/.work/disk-cleanup-20260904.json`.

Next:

- Resume the pre-existing layer-4 recurrent-state mismatch investigation. The
  existing candidate and baseline executables remain directly runnable; Cargo
  will regenerate compiler intermediates on the next build.

## 2026-09-04 — Exact complete layer-13 retained continuation

Objective:

- Extend the layer-12 checkpoint at `008e6f8` on
  `agent/rust-star-bootstrap` through complete layer 13 for both the native
  first 4K prefix and the retained positions-4,096--8,191 continuation.

Changes and evidence:

- Bound layer 12's live prefix and continuation HC allocations to the existing
  odd-layer executor using layer-13 model views. Preserved native batch
  arithmetic, GPU-owned compressor/cache history, and a strongly retained
  layer-13 prefix HC for the layer-14 handoff. No oracle activation is uploaded
  into execution state and no production arithmetic kernel was changed.
- Captured layer 13 twice with the accepted oracle-v3 executable. All 20
  payload pairs (3,507,617,792 bytes per capture) are byte-identical. The
  committed `rust-star/fixtures/prefill-layer13-continuation-8192-v3/checksums.json`
  binds their hashes, model, prompt, executable, and capture configuration.
- The optimized M1 Ultra integration matches all 16 checked boundary/tail
  hashes. Prefix ingress/QKV/compression uses 17 dispatches and 13/13 mappings;
  complete prefix attention/FFN totals 49 dispatches and 28/28 mappings. The
  continuation runs 128 ingress/QKV tiles followed by production compression,
  Flash attention, and FFN: 1,318 dispatches and 1,171/1,171 mappings total.
  All earlier retained layer gates remain exact in the same run.
- Report schema is now `rust-star-long-prefill-continuation-bootstrap-probe-v28`.
  Complete layer-13 prefix/continuation claims are true; complete 8K transformer,
  output-logit C0, throughput, and speedup claims remain false.
- The dump-enabled oracle runs drift in their downstream final logits despite
  identical layer-13 tensors. Those logits are rejected as C0 targets. A
  separate uninstrumented process exactly reproduces the accepted 8K JSON
  SHA-256 `791ee1ea8129889e3adaf4ce6e042156b85323e46df32581e4df275055848f94`.
  The four recurrent-state capture payloads are repeatability evidence; they
  are not counted among the 16 Rust boundary checks.

Validation:

- `cargo fmt --manifest-path rust-star/runtime/Cargo.toml --check`, all-target
  checking, the optimized build, all 307 Rust tests, and all 86 Python tests pass.
  The report test rejects seven independent alterations to layer-13 hashes,
  tile completion, or model-pointer accounting before writing any C0 result.
- A fresh full-model SHA-256 matches the accepted imatrix-Q2 target.
- Live command, with `MODEL` set to the target GGUF outside the repository:
  `rust-star/.work/runtime-target/release/rust-star long-prefill-continuation-bootstrap-probe "$MODEL" --json rust-star/.work/layer13-continuation-v28/rust-layer13-release.json`.
  The report SHA-256 is
  `91d7615a992e2a9ef6a1655869516aa36d413331beafeb8681714ff69369f9ef`.
- The independent complete-model 2K regression failed before layer 13, at
  layer-4 attention recurrent KV element 0: actual `0x3c87158e`, expected
  `0x3c8751ce`. Its source path was not changed by this implementation.
  A second fresh process fails with the identical element and bit patterns.
  An isolated release build exported from base commit `008e6f8` passes build
  validation; all 40 exported source files match that commit exactly. Its 2K
  run fails at the same element with identical actual/expected bits, confirming
  that this failure predates the layer-13 change. Preserve
  `rust-star/.work/layer13-continuation-v28/{rust-2k-regression,rust-2k-repeat,baseline-2k}.log`.
  The broader 2K live gate is not green. No eligible benchmark was run.

Decision and next step:

- Accept the complete retained layer-13 boundary as the validated bounded
  checkpoint. Repair the pre-existing layer-4 recurrent-state mismatch in the
  standalone 2K control before extending further; do not weaken its comparator
  or replace the expected fixture to make it pass.
- The next new transformer boundary is layer 14's paired ratio-4
  prefix/continuation, sparse attention, and FFN from live layer-13 HCs. Require
  complete 8K logits only after layers 14--42 and the output head are complete.
- Updated the current checkpoint, immediate actions, runtime documentation,
  project Open Items, and the standing journal instruction together. Keep this
  journal in the same local commit as the implementation; no remote push was
  performed.

## 2026-09-04 — Layer-13 retained continuation implementation and capture in progress

Objective:

- Extend the v27 layer-12 checkpoint through layer 13's exact ratio-128
  prefix and 4K-to-8K continuation using live retained HC and compressor state.

Progress and pending validation:

- Added layer-13 bindings around the existing odd-layer executor, plus explicit
  report fields and schedule checks. No new arithmetic kernel is introduced.
  Compilation and fresh oracle capture are in progress; layer 13 is not yet
  accepted as C0. Expected checksums must come from repeated oracle captures.
- The ignored oracle worktree's `ds4-bench` has later diagnostic changes and
  fails the accepted executable hash check. Use its preserved
  `ds4-bench.accepted-oracle-v3`, SHA-256
  `8e37f40cef769e34ef82a202d202a42b267322437ab8100c1303cc6aa8583bf3`.
  The diagnostic worktree and executable were left unchanged.
- Capture attempt `rust-star/.work/layer13-continuation-v28/a` produced tensors
  but failed to write final logits because the output directory was absent.
  It remains preserved as failed setup evidence. Corrected captures `a2` and
  `b2` run sequentially with the pinned executable, canonical prompt, 8,192
  tokens, 4,096-token prefill chunks, and zero generated tokens.
- Both corrected captures completed. All 20 layer-13 tensor pairs are
  byte-identical; the runtime gates use 16 ingress/compressor/attention/FFN
  checksums. Provenance is retained in
  `rust-star/fixtures/prefill-layer13-continuation-8192-v3/checksums.json`.
- Dump-enabled final 8K logits differ between those captures and from the
  accepted fixture. They are explicitly not accepted as new C0 targets.
  A separate uninstrumented fresh process reproduces the accepted JSON SHA-256
  `791ee1ea8129889e3adaf4ce6e042156b85323e46df32581e4df275055848f94`.
  Thus the bounded repeated layer-13 evidence remains usable, but downstream
  dump-induced drift must not be mistaken for complete-8K correctness.
- Fixed an initial model-weight binding placement error caught by compilation;
  the corrected all-target check passes. The optimized build, all 307 Rust
  tests (including rejection of seven altered/partial layer-13 reports), and
  all 86 Python tests pass. The integrated target-Mac Rust gate is running;
  layer-13 acceptance is still pending. A fresh full read of the model matches
  the pinned SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
  No benchmark or speedup claim is attached.

## 2026-09-04 — Reconciled journal with the committed layer-12 checkpoint

Objective:

- Restore an accurate resume point and preserve the user's instruction to keep
  this journal current as implementation and validation advance.

Repository and evidence inspected:

- Worktree: `/Users/deathcodevision/dev/ds4-rust-star-bootstrap`.
- Branch: `agent/rust-star-bootstrap`; HEAD: `008e6f83c471fbced8d9f761df8a4227a3d12cef`
  (2026-09-02, `Complete exact layer-12 continuation`). Local HEAD matches the
  locally stored `fork/agent/rust-star-bootstrap` tracking ref; no fetch or
  remote publication was performed during this review.
- The September 1 journal entry stopped at the layer-4 compressor boundary.
  Subsequent commits completed layer 4 (`2dbb627`), layer 5 (`09c4460`),
  layer 6 (`de8d67b`), layer 7 (`05a79c7`), layer 8 (`5029512`), layer 9
  (`ca20acc`), layer 10 (`3eb82c0`), layer 11 (`aaee512`), and layer 12
  (`008e6f8`). Prefix repairs and intermediate compressor gates are retained
  in Git history; the old journal entries remain unchanged.
- Cross-checked `rust-star/README.md`, `rust-star/runtime/README.md`,
  `rust-star/runtime/METAL.md`, and the runtime's v27 report implementation.
  They agree that the retained 4K-to-8K chain now reaches complete layer 12.
- The committed layer-12 evidence records 20 exact ingress/QKV/paired-compressor
  hashes across its prefix and continuation boundaries (40 dispatches and
  17/17 no-copy mappings each). Its prefix tail uses 36 dispatches; 128
  continuation sparse-attention/FFN tiles use 4,608 dispatches and 2,176/2,176
  mappings, with all six tail hashes matching the repeated oracle.
- These are previously recorded correctness results, not fresh measurements
  from this session. No model execution, Rust test suite, or benchmark was run.
  Documentation edits were checked with `git diff --check` and assertions
  covering the resume point, next actions, and ongoing journal rule.
- Existing untracked capture/check logs and `rust-star/runtime/.work/` were
  left intact. The separate DSpark checkout is not this worktree.

Decision and next step:

- Use schema `rust-star-long-prefill-continuation-bootstrap-probe-v27` and
  complete layer 12 as the committed starting point. Complete 8K transformer,
  output-logit C0, throughput, and speedup claims remain unestablished.
- Next implement layer 13's ratio-128 prefix and second-chunk continuation from
  layer 12's live GPU HC buffers, then verify compressor state, attention,
  FFN, and final HC against independently repeated oracle-v3 captures. Preserve
  native batch arithmetic and do not seed execution state from oracle tensors.
- Continue through layers 14--42 and the output head only as each boundary
  passes. Require all 129,280 logits to match the accepted supplemental 8K
  fixture before broadening to 32K or attaching a long-context performance claim.
- Keep the journal's current checkpoint and immediate actions synchronized
  after every meaningful implementation or validation step, including failures,
  and before handoff or compaction. Include the journal with related commits.

## 2026-09-01 — Exact layer-4 4K-to-8K compressor boundary

Objective:

- Repair the retained first-chunk dependency and make the production-size
  layer-4 continuation boundary bit-exact.

Changes and evidence:

- Found a transient-buffer ownership alias: with output collection disabled,
  layers 3--42 reused same-role buffers and overwrote the layer-3 prefix before
  continuation. The runtime now retains only the five layer-3 prefix buffers
  required by the handoff, adding about 1.15 GiB rather than retaining every
  intermediate tensor.
- Matched DwarfStar's 4K FlashAttention geometry. Layer 3 now keeps the real
  4,128-key extent, uses the 64-column KV-pad path and `has_kvpad` nsg8 kernel,
  and dispatches all 512 query blocks. Forty generated layer-3--42 launches no
  longer hard-code the 2K-only 256-block grid. The full layer-3 attention,
  inverse RoPE, output projection, attention HC, and FFN HC boundaries are
  exact.
- Reordered the retained workflow so the exact 4,096-row layer-3 continuation
  is completed before rebuilding layer 4's first 4K prefix. The prefix rebuild
  then remains live while positions 4,096--8,191 execute.
- Reproduced DwarfStar's ratio-4 post-prefill refresh at the production layer-4
  boundary: project the final four normalized tokens with the small-batch
  kernel, clear the eight-row recurrent state, and set rows 0--3 directly with
  score-plus-APE without a decode-style shift. Existing 32-row rolling
  diagnostics retain their previously validated update path.
- The M1 Ultra live gate now matches all ten layer-4 continuation checksums:
  HC ingress, norm, Q, KV, both complete 1,024-row compressed caches, and all
  four recurrent-state tensors. Both prefix and continuation schedules use 40
  dispatches and 17/17 no-copy model mappings.
- Evolved the artifact to
  `rust-star-long-prefill-continuation-bootstrap-probe-v14`. It records the
  compact eight-checksum layer-3 prefix gate, the layer-4 prefix rebuild, the
  10/10 continuation boundary, and a true complete-boundary C0 claim. It still
  keeps complete layer-4, complete-8K, output-logit, throughput, and speedup
  claims false.

Decision and next step:

- Treat the layer-4 ingress/QKV/paired-compressor continuation boundary as
  complete. Extend the exact second-chunk schedule through layer-4 mixed
  attention, attention HC, FFN, and final HC before advancing to layer 5.

## 2026-09-01 — Layer-4 continuation isolates stale first-chunk state

Objective:

- Feed the exact production-batch layer-3 second chunk into layer 4 and locate
  the first remaining 8K mismatch.

Changes and evidence:

- Added a production-size 4,096-row layer-4 ingress/QKV/paired-compressor
  boundary with 40 Metal dispatches and 17/17 no-copy model mappings.
- Captured the complete layer-4 continuation boundary twice from DwarfStar;
  both runs were byte-identical. Rust matches HC ingress, learned norm, Q,
  finalized KV, and all four final recurrent-state tensors: 8/10 aggregate
  checksums are exact.
- Both compressed-cache tensors differ only in row zero. Rows 1--1,023 are
  bit-for-bit exact for both the 512-wide attention cache and 128-wide indexer
  cache. Row zero is the only replay row that consumes pre-4,096 compressor
  state.
- A fresh DwarfStar prefix-state capture and Rust readback confirmed that the
  first halves of all four retained layer-4 ratio-4 state tensors differ,
  while their unused second halves agree. Rebuilding the state with the exact
  small-batch tail projection did not repair it.
- A final-32-row comparison then located the upstream boundary: Rust's
  pre-4,096 layer-4 normalized activation differs in every value from the
  production oracle. The exact second-chunk layer-3 output and layer-4 ingress
  are not at fault; the remaining dependency is the first-chunk production
  layer-3-to-layer-4 handoff.
- Evolved the report to
  `rust-star-long-prefill-continuation-bootstrap-probe-v13`. It records the ten
  layer-4 checksums, the stable 8/10 match, 1,023 state-independent exact rows
  per cache, and explicit false complete-layer-4/complete-8K claims.
- Debug and optimized M1 Ultra runs reproduced the same layer-4 correctness
  fields. The release artifact SHA-256 is
  `374111be71efabdbb8a8cfa6d69443c43be6d30d41957041795b65d9ac1c713b`.
- Validation passed 307 Rust tests, 86 Python tests, formatting, the optimized
  build, the Metal ownership/dispatch probe, and the Rust-writer/Python-reader
  C0 artifact contract. The repository-wide fixture loop remains blocked by
  the pre-existing `prefill-layer0-ffn-transition-pos4099-v1` manifest, whose
  `operations` array has been absent since commit `94485d4`.

Decision and next step:

- Keep this as a diagnostic checkpoint. Do not seed prefix tensors from the
  oracle and do not claim a complete layer-4 continuation boundary.
- Replay the first 4K layer-3 production batch from the retained exact layer-2
  output, retain its exact layer-4 prefix norm/Q/KV/compressor state, then rerun
  this boundary. After row zero matches, continue through layer-4 attention
  and FFN.

## 2026-08-31 — Tiled layer-3 output rejected as production-batch boundary

Objective:

- Retain the complete 4,096-row layer-3 second-chunk output and determine
  whether the proven 32-row continuation loop can feed layer 4 exactly.

Changes and evidence:

- Added one 256 MiB shared Metal buffer that receives every layer-3 final-HC
  tile by GPU blit. The checksum ABI reads that retained allocation only after
  all 128 synchronized continuation tiles have completed and returns one full
  FNV plus 128 independent tile FNVs.
- Captured the full production-geometry layer-3 output twice in fresh DwarfStar
  processes. Both 268,435,456-byte tensors have SHA-256
  `09b31ad2a23ef6805ab62c9cbd353caf280546770e432760f14252e400c42cca`
  and FNV `17010162403439886297`.
- The optimized M1 Ultra Rust run retained all 4,096 by 16,384 FP32 values and
  reproducibly produced FNV `12296721839747081491`. Exactly 114/128 tile
  checksums match the production tensor. Mismatches start at positions 4096,
  7296, 7360, 7392, 7424, 7552, 7616, 7648, 7712, 7776, 7840, 8000, 8064,
  and 8096.
- The terminal tile and compressor states retain their previous exact
  checksums. The first tile still matches its accepted 32-row-geometry fixture
  but differs from the first tile of the production 4,096-row tensor. This
  proves that the discrepancy is batch geometry, not a failed retention copy.
- A separate repeated layer-4 capture exposed the same rule earlier in its
  ratio-4 compressor: the first eight production-batch compressed rows differ
  from a 32-row continuation-state capture. The incomplete layer-4 fixture
  generated before that distinction was discovered was moved to Trash and is
  not part of the repository.
- Evolved the report to
  `rust-star-long-prefill-continuation-bootstrap-probe-v11`. It records the
  tiled and production checksums, matching-tile count, all mismatch starts,
  and an explicit false
  `complete_layer3_second_chunk_output_checksum_claim`.

Validation:

- Debug and optimized M1 Ultra live probes both completed all 14,560 layer-2/
  layer-3 dispatches with 5,760/5,760 no-copy mappings and reported the same
  checksum comparison.
- Optimized macOS build, `cargo fmt --check`, and `cargo check --all-targets`.
- All 307 Rust tests and all 86 Python artifact/adapter/runner tests.
- The optimized JSON report SHA-256 is
  `4785db7b77b1ec58cdbcc58b31c72f960ad620134751ac2724c4aec4edfe446a`
  under `rust-star/.work/layer3-tiled-vs-production-8192-release.json`.

Decision and next step:

- Keep the tiled loop as retained-state and structural evidence, but reject it
  as the exact input boundary for layer 4. Implement layer 3 over the complete
  production-size 4,096-row second chunk, require the full oracle checksum,
  then feed that live Metal allocation into a production-size layer-4 batch.

## 2026-08-31 — Complete retained layer-2/layer-3 second-chunk loop

Objective:

- Carry the proven positions-4,096--4,159 boundary through every remaining
  32-row layer-2/layer-3 tile in the second 4K chunk without overstating C0.

Changes and evidence:

- Generalized the layer-3 continuation attention boundary through tile start
  8,160. The retained full-KV allocation now holds 8,192 rows, and the
  ratio-128 compressed allocation holds all 64 first-8K rows while preserving
  its first-chunk prefix.
- Each tile advances the recurrent compressor state row by row and emits on
  the production 128-token cadence. Expected-history inputs are now an
  optional paired diagnostic: the first two exact anchors still validate their
  captured raw/compressed histories, while later tiles consume only native
  retained state.
- Two fresh optimized M1 Ultra integrations completed positions 4,096--8,191
  through layers 2 and 3 in 128 retained tiles. Both observed all 32
  second-chunk ratio-128 emissions, 14,560 dispatches, and 5,760/5,760 no-copy
  mappings. Their final layer-3 HC, compressor-KV state, and
  compressor-score-state checksums were identical at respectively
  `12338792942122668793`,
  `11261122877073481448`, and `8521713610302668492`.
- Evolved the report to
  `rust-star-long-prefill-continuation-bootstrap-probe-v10`. It records two C0
  anchor tiles and 126 structurally executed unverified tiles, with explicit
  false claims for complete layer-2/layer-3 C0, the complete 8K transformer,
  output-logit C0, and throughput.
- The live probe retained its existing exact first-two-tile checks. All 307
  Rust tests, all 86 Python tests, the optimized build, formatting, JSON
  validation, and diff checks passed.

Decision:

- Claim complete native retained-state execution of the second 4K chunk for
  layers 2 and 3. Do not claim C0 beyond the first two tiles, complete 8K
  transformer execution, output-logit equality, throughput, or a speedup.

Next:

- Generalize the same retained batch boundary through layers 4--42, initially
  using the existing per-layer exact fixtures as anchors, then connect the
  output head and compare all 129,280 8K logits.

## 2026-08-31 — Second consecutive unseeded layer-3 continuation tile

Objective:

- Continue the exact positions-4,096--4,127 layer-2/layer-3 path through the
  next 32 rows in the same live Metal context, with no first-tile replay or
  oracle activation upload.

Changes and evidence:

- Captured full production-geometry layer-2 and layer-3 outputs twice in fresh
  accepted-oracle-v3 processes. All repeated payloads were byte-identical.
  Imported only positions 4,128--4,159, their exact prior layer-3 history, and
  the ratio-128 state after 64 continuation rows into
  `dwarfstar-oracle-v3-prefill-layer23-continuation-second-tile-4128`.
- Generalized the layer-2 sparse tile, downstream tail, layer-3 ingress, and
  layer-3 attention ABIs around an aligned tile start. Layer 2 retains its 4K Q
  batch; layer 3 grows its retained KV capacity once and appends finalized live
  tiles on the GPU for the next invocation.
- The consecutive gate exposed two first-tile-only assumptions. Layer-2 sparse
  attention now advances its logical raw span without shifting the 4,352-row
  ring origin. Layer-3 attention now reproduces DwarfStar's 4,288-key padded
  4K-batch Flash geometry while all unavailable future rows remain masked.
- The optimized M1 Ultra run passes bit-for-bit through both complete tiles.
  The second layer-2 tile matches 16 outputs over 36 dispatches and 17/17
  no-copy mappings. The connected second layer-3 tile matches 26 ingress,
  recurrent-state, attention, FFN, and final-HC outputs over 76 dispatches and
  28/28 mappings. Combined second-tile evidence is 42 exact outputs, 112
  dispatches, and 45/45 mappings.
- Evolved the report to
  `rust-star-long-prefill-continuation-bootstrap-probe-v9`, including explicit
  second-consecutive-tile and two-unseeded-layer-3-tile claims. The clean probe
  recorded 4,542.477/4,527.168 ms summed continuation wall/GPU time; this is
  diagnostic evidence, not a paired performance result.
- The compact 44-tensor fixture verifies at 38,538,240 bytes. All 307 Rust
  tests, all 86 Python tests, importer compilation, formatting, and the live
  optimized integration pass succeeded.
- Deleted only the reproducible raw capture directory and its accidental empty
  nested counterpart after validation, reclaiming 9.1 GiB. The 37 MiB compact
  fixture and v9 JSON evidence remain.

Decision:

- Claim two consecutive complete unseeded native layer-2/layer-3 continuation
  tiles. Do not claim the complete 8K transformer, output-logit equality,
  throughput, or a speedup.

Next:

- Generalize the retained schedule across the remaining second-chunk tiles,
  then extend the exact live chain through layers 4--42 and the output head.

## 2026-08-31 — Unseeded complete layer-3 continuation tile

Objective:

- Eliminate the two pre-4,096 layer-3 attention-history seeds and rerun the
  existing complete positions-4,096--4,127 layer-3 gate.

Changes and evidence:

- Traced the first native raw/compressed history mismatch back through the 4K
  first-chunk transformer. Layer 2 had 1,024 ratio-4 compressed rows at that
  extent, but still ran dense mixed attention; it now executes DwarfStar's
  indexed top-512 schedule while the exact 2K/512-row control remains dense.
- Added the two layer-2 indexer model ranges and nine sparse dispatches needed
  for the 4K schedule. The 4K transformer now reports 2,377 dispatches and
  1,218/1,218 no-copy mappings.
- Found and removed a separate 2,048-row ceiling in every decomposed batch
  router stage. The long-prefill router now covers all 4,096 rows instead of
  silently leaving the second half untouched.
- Removed the history copies from the continuation ABI. The pinned raw128 and
  compressed32 arrays remain read-only expected-value checks against the
  native retained half bits; they are not execution inputs.
- The M1 Ultra probe passed with 4,160/4,160 bootstrap mappings and 7,556
  continuation dispatches. Positions 4,096--4,127 are exact through the
  complete layer-2 and layer-3 tiles; layer 3 matches 17 retained and
  downstream outputs over 76 dispatches with 28/28 no-copy mappings. The clean
  optimized rerun completed in 4,717.845/4,692.537 ms summed wall/GPU time.
- Evolved the report to
  `rust-star-long-prefill-continuation-bootstrap-probe-v8`, with explicit
  native-retained-history and unseeded-complete-layer-3 claims.
- The clean release build, all 306 Rust tests, all 85 Python tests, formatting,
  JSON scope inspection, and diff checks passed. No paired performance
  benchmark was run, so this evidence carries no throughput or speedup claim.
- Deleted only the reproducible layer-3 boundary diagnostic captures after the
  clean pass, reclaiming 3.0 GiB. The accepted oracle/model workspace and v8
  report remain intact.

Decision:

- Claim an unseeded complete native layer-3 continuation tile. Do not claim the
  complete 8K transformer, output-logit equality, throughput, or a speedup.

Next:

- Extend the same native batched second-chunk schedule across the remaining
  tiles, then layers 4--42 and the output head, preserving C0 at each boundary.

## 2026-08-31 — Cache-seeded complete layer-3 continuation tile

Objective:

- Complete positions 4,096--4,127 through native layer-3 ratio-128 attention
  and FFN while preserving the live layer-2-to-layer-3 handoff.

Changes and evidence:

- Captured the complete layer-3 ratio-128 state, two exact attention-history
  extents, and 15 downstream tensors twice in fresh accepted-oracle-v3
  processes. All 19 retained tensors are bit-identical across repetitions.
- Kept the live, exact layer-2 final HC to layer-3 ingress/QKV path. Only the
  pre-4,096 layer-3 attention history is fixture-seeded: 128 raw KV rows and 32
  ratio-128 compressed KV rows.
- Added native ratio-128 state updates, dense mixed FlashAttention, inverse
  RoPE, attention projection, biased top-k routing, routed/shared experts, and
  final HC. The connected layer-3 tile matches all 17 state and downstream
  outputs bit-for-bit over 76 dispatches with 28/28 no-copy model mappings.
- Direct comparison showed that the native first-4K retained layer-3 histories
  are not yet exact. The first raw-KV mismatch was `0x3e6ffffe` versus
  `0x3d700000`; the first compressed-KV mismatch was `0x3cfffffe` versus
  `0x3c800000`.
- After the fixture and repeated-capture identities were pinned and validated,
  removed only the reproducible raw capture directories, reclaiming about
  4.25 GiB. The accepted oracle/model workspace remains intact.

Decision:

- Claim a complete layer-3 continuation tile only from the pinned pre-4,096
  history boundary. Do not claim an unseeded layer-3 tile, complete 8K
  transformer, output-logit equality, or throughput.

Next:

- Diagnose and repair the first-4K layer-3 raw and compressed attention
  histories, eliminate both fixture seeds, and rerun this exact tile gate.

## 2026-08-31 — First complete native second-chunk layer-2 tile proven

Objective:

- Remove the oracle KQV-back execution boundary from positions 4,096--4,127
  and connect the production multirow sparse scheduler directly to the already
  exact attention-output/FFN tail.

Changes and evidence:

- Captured full 4,096-row layer-2 indexer-score and streaming top-512 tensors
  twice from the pinned accepted oracle-v3 executable. Both repetitions were
  bit-identical. The fixture retains only the first 32 rows (320 KiB) while
  pinning the full 32 MiB score and 8 MiB selection identities by SHA-256.
- Added the ten-dispatch M1 production sparse tile: aligned F16 indexer Q and
  weight projections, batched RoPE/QAT, tiled scores, streaming top-512,
  chronological ordering, F16 compressed-cache staging, eight-head indexed
  attention, and inverse RoPE. The first 32 score rows, selected rows, and
  KQV-back rows all match DwarfStar bit-for-bit.
- Retained the native KQV-back Metal buffer and passed it directly into the
  existing 26-dispatch attention-output/FFN tail. All 13 downstream tensors
  remain exact through the final layer-2 HC state. The C ABI no longer accepts
  an oracle KQV-back input for this path.
- Corrected the initial TensorOps assumption after inspecting DwarfStar's
  device gate: pre-M5 machines deliberately use aligned F16, tiled score, and
  eight-head attention kernels. A transient NAX experiment was rejected after
  its first score differed from the M1 oracle.
- Evolved the report to
  `rust-star-long-prefill-continuation-bootstrap-probe-v4`. It records 53
  combined diagnostic dispatches, the native live-buffer boundary, and a true
  complete-layer-tile claim while preserving false complete-8K, output-logit,
  and throughput claims.

Validation:

- The M1 Ultra continuation probe completed 7,556 bootstrap dispatches with
  4,160/4,160 mappings in 4,684.330/4,670.532 ms summed wall/GPU time. The
  connected layer-2 diagnostic used 53 dispatches and 18/18 mappings.
- Ignored JSON evidence SHA-256 is
  `c1270a5e228fd8d028dc84bfe3e22c5faab780de123c5230cc81da3edac3579e`.
- The optimized macOS build, all 306 Rust tests, all 83 Python tests, clean
  fixture regeneration, formatting, and `git diff --check` passed.

Decision:

- Accept positions 4,096--4,127 as the first complete native layer-2 tile of
  the second chunk. This closes the multirow sparse-attention correctness gap
  for the representative tile without expanding the claim to all 8K tokens.

Next:

- Retain the final layer-2 HC tile as live layer-3 input and prove the same
  positions through layer 3, then generalize the connected continuation across
  the remaining second-chunk tiles and later layers.

## 2026-08-31 — First native-batched continuation tail proven

Objective:

- Move beyond the one-row layer-2 sparse boundary and validate the production
  batch output/FFN/final-HC schedule over a real tile from the second 4K chunk.

Changes and evidence:

- Captured every layer-2 tensor downstream of KQV-back for positions
  4,096--8,191 twice in fresh accepted oracle-v3 processes. All full 4,096-row
  tensors repeated bit-for-bit. The importer pins their complete identities
  while retaining only the first 32 rows, producing a 13 MiB fixture with one
  input and 13 exact outputs.
- Added a 26-dispatch native Metal tail over positions 4,096--4,127. It wraps
  14 mmap-backed model ranges and executes the grouped attention output,
  attention HC update, FFN HC ingress/norm, token-hash router, routed experts,
  shared expert, and final HC update. All 13 FP32/I32 outputs match the repeated
  DwarfStar capture by bit pattern.
- Kept the proof boundary explicit: the 32-row tail consumes captured oracle
  KQV-back, so it proves the downstream batch schedule but not complete layer-2
  execution. The independent native position-4,099 sparse-attention replay
  remains exact. Schema
  `rust-star-long-prefill-continuation-bootstrap-probe-v3` reports the two
  claims separately and preserves false complete-layer, complete-8K,
  output-logit, and throughput claims.
- The runtime loads this large fixture at execution time. Embedding it in the
  Rust object exceeded Mach-O's 4 GiB string-table limit once combined with the
  existing fixture corpus, so no new binary-size dependency was accepted.

Validation:

- The M1 Ultra continuation probe completed all 7,556 bootstrap dispatches
  with 4,160/4,160 mappings in 4,677.923/4,664.606 ms wall/GPU. The combined
  layer-2 diagnostic used 43 dispatches and 18/18 mappings: 17/4 for the true
  sparse transition and 26/14 for the oracle-seeded downstream tail.
- Ignored JSON evidence SHA-256 is
  `cf57e2dd7ad374d0d4dd0fb6e1bc81f741b24cc783fcb1f770b6968384057ea2`.
- The optimized macOS build, all 306 Rust tests, all 83 Python tests, fixture
  payload validation, clean importer regeneration, JSON parsing, formatting,
  and `git diff --check` passed.

Decision:

- Accept the 32-row downstream batch tail as C0 while refusing to call it a
  complete layer. This removes output projection, router, expert, and HC
  numerics as unknowns at the second-chunk boundary.

Next:

- Generate the first 32 rows' KQV-back natively with the real multirow sparse
  attention scheduler, feed it directly into the proven tail, and retain the
  resulting final HC as live layer-3 input. Then generalize that connected
  schedule through the remaining layer-2 tiles and layers 3--42.

## 2026-08-31 — First sparse layer-2 boundary proven through FFN

Objective:

- Close layer 2 at the first production sparse boundary through its
  token-dependent FFN and final HC state before extending the native-batched
  second chunk across later layers.

Changes and evidence:

- Reused the two retained position-4,099 oracle captures already stored in the
  workspace. Every downstream payload is independently bitwise repeated,
  including attention output/HC, FFN HC ingress, router logits/probabilities,
  selected experts, routed/shared expert outputs, and final HC.
- Extended `retained-sparse-boundary-probe` from 16 attention-side checks to 30
  exact tensors through final layer HC. The six captured experts uniquely map
  to input token 7,129 in `blk.2.ffn_gate_tid2eid.weight`; the previous
  placeholder token 0 had hidden the token-hash FFN boundary.
- Corrected stale dispatch metadata after the already-landed fused execution
  path: the retained layer uses 51 base dispatches plus one per top-k merge.
  Position 4,099 therefore uses 52 dispatches, not 54. Position 8,195 uses 53
  layer-2 dispatches; its layer-0/layer-1 predecessors use 27/25, for 105 total
  rather than the obsolete 113.
- Promoted the position-4,099 report to `complete_layer_claim=true` while
  preserving false claims for preceding layers, complete decoder, logits, and
  throughput. Updated fixture provenance, CLI help, JSON invariants, and the
  runtime documentation.

Validation:

- The real M1 Ultra `retained-sparse-boundary-probe` passed all 30 tensor checks
  with 35/35 no-copy model mappings and 52 dispatches.
- The adjacent `retained-sparse-multimerge-probe` passed all 44 tensor checks
  with 85/85 mappings and 105 total dispatches, confirming the accounting
  correction did not regress the repeated-merge control.
- The optimized macOS build, all 306 Rust tests, all 82 Python tests, both JSON
  evidence parses, formatting, and `git diff --check` passed.

Decision:

- Accept the complete retained layer-2 sparse boundary as C0. The result proves
  the downstream output/FFN seam but remains a one-token retained control; it
  does not substitute for native-batched second-chunk execution.

Next:

- Batch the sparse attention/output/FFN schedule over the second 4K chunk,
  retain layer-2 final HC for layer 3, and then generalize the exact schedule
  through layers 3--42 before checking all 129,280 terminal 8K logits.

## 2026-08-31 — First native-batched 8K sparse transition reached C0

Objective:

- Locate the first exact divergence in the retained 4K-to-8K bootstrap and
  prove layer 2's position-4,099 sparse attention with the same geometry as the
  oracle's second 4,096-token prefill chunk.

Changes and evidence:

- Added independently repeated oracle-v3 transition fixtures for layer-0
  attention, KV, FFN/router, and HC boundaries; layer-1/layer-2 attention
  ingress; and layer-2 attention-compressed row 1,024. Corrected the upstream
  `hc_attn_pre` hook interpretation: it captures the collapsed 4,096-wide
  attention input, not the 16,384-wide HC state.
- Generalized the layer-0/layer-1 FlashAttention block grid from the old fixed
  2K extent to the active prefill rows and query rows. Every retained
  layer-0 boundary, the direct/retained layer-1 HC handoff, layer-1/layer-2
  attention ingress, and layer-2 Q setup then matched bit-for-bit.
- Found a real ratio-4 compressor boundary bug. The batch encoder treated
  absolute position 2,048 as the final prefill tile even inside a grown 4K/8K
  retained buffer, replaying the four-row tail projection early and corrupting
  compressed row 512 onward. Final-tile detection now derives from retained
  output capacity. The first 1,024 attention/indexer rows and newly emitted
  attention row 1,024 match their independent oracle captures exactly after
  the required attention-cache FP16 storage roundtrip.
- Rejected an FP16 roundtrip on raw KV as a no-op and preserved the existing
  one-token sparse decode path. Its position-4,099 12-split control remains C0
  exact with 11 dispatches.
- Identified the remaining four-ULP KQV mismatch as a geometry error in the
  diagnostic, not a tolerance problem. DwarfStar uses a 4,096-token prefill
  kernel with a 4,352-row physical raw ring, a 4,224-row logical batch span,
  raw start 3,968, 2,048 compressed capacity, chronological top-k sorting, and
  the dual-head prefill attention pipeline. The old diagnostic instead used a
  one-token 12-split decode reduction over 128 raw and 1,025 compressed rows.
- Added an oracle-matched row-4,099 replay that reconstructs the final raw ring,
  preserves the batch's `n_tokens`/`pos0` metadata, sorts the selected rows by
  cache index, and dispatches
  `kernel_dsv4_indexed_mixed_attention_heads16_dual`. KQV output and inverse
  RoPE now match the second-chunk oracle capture bit-for-bit without weakening
  C0. The report records the physical ring/span/capacity and single-dispatch
  prefill schedule separately from the decode control.

Validation:

- The full target-Mac `long-prefill-continuation-bootstrap-probe` passed the
  new sparse-transition C0 gate, completed all 64 continuation tiles and 7,556
  bootstrap dispatches, and preserved 4,160/4,160 no-copy model ranges.
  Continuation wall/GPU time was 4,697.736/4,685.322 ms. Ignored JSON evidence
  SHA-256 is
  `218a9ca7a2c60b3b7422815bc8450bb73f65a8d2863f79ab6c014452a2c2e63a`.
- The standalone one-token sparse control passed after the raw-KV experiment
  was reverted.
- Formatting, the optimized macOS build, all 306 Rust tests, all 82 Python
  tests, JSON parsing, and `git diff --check` passed.

Decision:

- Accept position 4,099 as the first exact native-batched sparse checkpoint.
  Keep the decoder and prefill attention schedules distinct in diagnostics and
  metadata; equal inputs do not make their online-softmax reductions
  interchangeable under C0.

Next:

- Extend the retained second chunk through layer 2's attention output and FFN,
  then generalize the exact batched sparse schedule across layers 3--42 and
  require all 129,280 terminal 8K logits to match oracle-v3 bit-for-bit.

## 2026-08-30 — Sequential 4K-to-8K continuation shortcut rejected

Objective:

- Determine whether the already exact retained decoder schedule could execute
  the second 4K prompt segment while preserving the pinned batched 8K result.

Changes and evidence:

- Generalized the prefill-to-decoder handoff from a fixed 2K source to completed
  2K or 4K prefill state. The 4K handoff copied the last 128 raw rows plus
  1,024 ratio-4 rows, 32 ratio-128 rows, and every recurrent compressor state
  for all 43 layers using the existing 229-blit ownership contract.
- Added `long-prefill-sequential-continuation-probe` and schema
  `rust-star-long-prefill-sequential-continuation-probe-v1`. It completes the
  first 4K native transformer, adopts that state, then supplies the accepted
  prompt tokens at positions 4,096--8,191 through global RoPE, rolling raw
  caches, recurrent compressors, and production sparse ratio-4 attention.
- The full target-Mac diagnostic completed without a Metal or ownership error
  and crossed the first sparse boundary at position 4,099. Its terminal
  selected token was 77179, equal to the oracle, but all 129,280 logits differed
  bit-for-bit. The first mismatch was index 0 and maximum absolute error was
  10.662007332. Continuation wall/GPU time was
  196,711.436/185,012.119 ms. Ignored JSON evidence SHA-256 is
  `25b0b4d98785a79d5050019019e575dd9cccfe5c74fc496892f1a3108d69365c`.
- The independent complete 2K control remains C0 exact through all logits and
  selects token 15342. Its ignored JSON evidence SHA-256 for this run is
  `6b6508e1f39beb3a59ba8f06b4b5dba8b429fcb796079343a48aca321fcd2fee`.
  Formatting, the all-target build, optimized macOS build, 306 Rust tests, and
  82 Python tests passed.

Decision:

- Reject sequential decoder execution as the second prompt chunk. Equal argmax
  is not C0, every logit differs, and the 196.7-second wall time is not a viable
  prefill path. Preserve this as negative architectural evidence rather than
  weakening the 8K oracle gate.

Next:

- Build a native-batched positions-4,096--8,191 transformer path that appends
  per-layer raw/compressed state, uses global positions and the production
  sparse ratio-4 branch, and reuses batch projection/FFN numerics. Require all
  129,280 terminal logits to match the pinned 8K oracle bit-for-bit.

## 2026-08-30 — Retained 4K-to-8K bootstrap transition executed

Objective:

- Execute positions 4,096--8,191 as a true continuation of the completed first
  4K transformer instead of resetting the Metal context or compressor state.

Changes and evidence:

- Generalized the native bootstrap boundary to an 8,192-row working extent and
  added prefix-preserving growth for retained layer-0/1/2 raw KV, layer-2
  inputs, Q norm, attention split, tokens, and paired compressed histories.
  Fixed-size recurrent compressor state is preserved in place. A new batch
  command lifetime starts at the second-chunk boundary while global positions
  continue at 4,096.
- Generalized the Rust tile loop to an aligned nonzero start and added
  `long-prefill-continuation-bootstrap-probe` with schema
  `rust-star-long-prefill-continuation-bootstrap-probe-v1`. The command first
  completes the 4K transformer, then runs 64 64-row continuation tiles through
  complete layers 0/1 plus layer-2 raw KV and paired ratio-4 compressors in the
  same context.
- The target-Mac run completed 7,556 continuation dispatches with
  4,160/4,160 no-copy mappings. Retained state reached 8,192 raw rows and 2,048
  layer-2 attention/indexer compressed rows. Continuation wall/GPU time was
  5,135.963/5,124.933 ms. Ignored JSON evidence SHA-256 is
  `3bb768bfd4ce6cb5d2c67fd0e218100788fdd335f9764b9265809f7cbc7bfbaa`.
- The independent complete 2K control still matched every retained transformer
  boundary and all 129,280 logits bit-for-bit, selecting token 15342. Its
  ignored JSON evidence SHA-256 for this run is
  `74b01678afda096a0c05b7ae16053416fd3020bef6751f01eca937c79cc47b78`.
  Formatting, the all-target build, the optimized macOS build, 305 Rust tests,
  and 82 Python tests passed.

Decision:

- Accept the retained second-chunk bootstrap as the ownership and capacity
  checkpoint. It deliberately makes no complete-8K-transformer, output-logit
  C0, or throughput claim.

Next:

- Extend positions 4,096--8,191 through layers 2--42, including the
  production sparse ratio-4 branch after 1,024 compressed rows, then require
  all 129,280 output logits to match
  `dwarfstar-oracle-v3-prefill-frontier-8192` bit-for-bit.

## 2026-08-30 — Supplemental oracle-v3 8K frontier accepted

Objective:

- Establish an exact intermediate target for the first retained long-prefill
  continuation instead of debugging only against the terminal 32K tensor.

Changes and evidence:

- Rebuilt `ds4-bench` from accepted host-synchronized producer commit
  `d35fb12d01d500b9cefcef24092c295687ceaf7e` / tree
  `617415ee9f8ea7dc176d63dada1d5a7582063824` in an ignored detached worktree.
  The capture executable SHA-256 is `8e37f40c...83bf3`; the model and prompt
  retain their accepted SHA-256 identities.
- Ran four sequential fresh-process Metal captures at exactly 8,192 tokens.
  Every complete JSON tensor is byte-identical with SHA-256
  `791ee1ea8129889e3adaf4ce6e042156b85323e46df32581e4df275055848f94`.
  The packed FP32 logits SHA-256 is
  `626454dd1d12717abe29c9fe7d4140bcb265046bf8253bc46ea02575a9c53a1a`,
  and lowest-ID argmax selects token 77179.
- Imported `dwarfstar-oracle-v3-prefill-frontier-8192` with the exact first
  8,192-token prefix of the accepted 32K stream. The importer rehashes the
  producer, executable, full model, prompt, token prefix, four JSON tensors,
  and four CSV records before writing the fixture. Rust and Python loaders
  enforce its shape, token endpoints, and selection.
- The four observed prefill rates were 180.60, 181.06, 177.54, and 176.44
  tok/s. They are labeled conformance observations only: this capture was not a
  paired benchmark and makes no Rust Star performance claim.

Decision:

- Accept the 8K tensor as supplemental oracle-v3 C0 evidence from the same
  immutable producer, without changing the canonical 2K/32K acceptance set.
  Use it as the mandatory output gate for positions 4,096--8,191.

Next:

- Implement an explicit retained second-chunk boundary with global RoPE,
  raw-window rollover, compressor append/state rebuild, and production sparse
  ratio-4 attention, then require all 129,280 output logits to match this 8K
  fixture bit-for-bit.

## 2026-08-30 — First native 4K transformer chunk completed through logits

Objective:

- Extend the accepted 4K bootstrap chunk through every remaining transformer
  layer and the output head without weakening the exact 2K regression.

Changes and evidence:

- Parameterized the layer-2-through-42 native batch schedule and output head
  with an explicit 2,048/4,096-row working extent. The first 4,096 tokens of
  the accepted oracle-v3 32K stream now execute layers 0--42 in order and
  retain every layer's raw, compressed, indexer, and recurrent compressor
  state for continuation.
- Added `long-prefill-transformer-probe` and schema
  `rust-star-long-prefill-transformer-probe-v1`. The target-Mac run completed
  7,556 bootstrap dispatches with 4,160/4,160 no-copy mappings, then 2,372
  transformer dispatches with 1,216/1,216 no-copy mappings. It retained 4,096
  raw rows, 1,024 ratio-4 rows, and 32 ratio-128 rows per applicable layer.
- The 4K output head selected token 565. No intermediate DwarfStar frontier
  was available, so the report explicitly makes no C0 or oracle-match claim.
  Bootstrap wall/GPU time was 5,615.892/5,184.460 ms, transformer wall/GPU
  time was 14,227.714/14,080.315 ms, and output-head wall/GPU time was
  207.622/2.454 ms. Ignored JSON evidence SHA-256 is
  `e39b67e0198f852b5ba8ab2bdc77a407342b80e5b69dfebb3792346cddb9f315`.
- The independent 2K control still matched all retained layer boundaries and
  129,280 logits bit-for-bit and selected token 15342. Its ignored JSON
  evidence SHA-256 is
  `6fc5b3096e9ff399690faf08f96f3db7d75d93dfcea33048ce7ce32098d0dd17`.
- The complete target-Mac gate passed formatting, 303 Rust tests, the release
  macOS build, all 81 Python tests, Metal smoke, every differential fixture,
  and the cross-language artifact roundtrip. Published commit `44fdafa`.

Decision:

- Keep the complete first-chunk schedule as the long-context baseline. Do not
  treat a second isolated 4K batch as continuation: chunk two must append to
  retained compressed state, use global RoPE positions, roll the raw window,
  and cross the production sparse ratio-4 threshold.

Next:

- Pin an exact 8K frontier from the accepted synchronized oracle-v3 producer,
  then implement positions 4,096--8,191 as an explicit retained continuation
  and compare its complete logits to that frontier.

## 2026-08-30 — First native 4K long-prefill bootstrap chunk executed

Objective:

- Remove the fixed-2K working-extent assumption from the earliest native
  prefill boundary and test the first real chunk of the accepted 32K stream.

Changes and evidence:

- Added an explicit `prefill_rows` field to the Rust/Objective-C layer-0
  boundary ABI. Full-KV, attention-mask, compressor-cache, command-batch, and
  terminal-tile geometry now derive from that value; all existing callers
  continue to pass 2,048.
- Added `long-prefill-bootstrap-probe` and schema
  `rust-star-long-prefill-bootstrap-probe-v1`. The probe consumes the first
  4,096 tokens of `dwarfstar-oracle-v3-prefill-frontier-32768` in 64 native
  64-row tiles and truthfully labels the missing intermediate oracle: it makes
  no 4K C0, complete-32K, or throughput claim.
- The target-Mac run completed all 7,556 dispatches in 4,988.073 ms summed wall
  and 4,935.518 ms summed Metal time. All 4,160 model ranges preserved their
  mmap pointer. The context retained 4,096 layer-0/layer-1/layer-2 raw rows,
  1,024 layer-2 attention-compressed rows, 1,024 indexer-compressed rows, and
  the recurrent compressor tails. The ignored JSON evidence has SHA-256
  `1afdcafa371a41af01460b836fbb48dc30ebf632960c7d48da3cfe556b19a80d`.
- The independent complete 2K native model-prefill control then passed every
  retained C0 boundary and all 129,280 logits, selecting token 15342. Its
  terminal transformer schedule used 2,372 dispatches with 1,216/1,216 no-copy
  views; the output head used 5/5.
- `cargo check --all-targets`, all 302 Rust library tests, and all 81 Python
  contract tests passed. No eligible benchmark was run.

Decision:

- Keep the explicit working extent and 4K bootstrap probe. This establishes
  that the earliest raw/compressor allocations and command lifetime are no
  longer bounded to 2K without weakening the exact 2K path.

Next:

- Parameterize the layer-2-through-42 transformer schedule for a 4K working
  chunk, preserve the resulting per-layer raw/compressed/recurrent state, and
  only then begin the second 4K chunk. The final eight-chunk output must match
  the accepted oracle-v3 32K logits bit-for-bit before any long-context speed
  claim.

## 2026-08-30 — Accepted oracle-v3 and imported the 32K Rust frontier

Objective:

- Pin the host-synchronized producer, require four exact fresh processes at
  both frontiers, and admit the resulting 32K tensor to Rust only after the
  independent archive and C0 gates passed.

Evidence:

- Published producer commit `d35fb12d01d500b9cefcef24092c295687ceaf7e` /
  tree `617415ee9f8ea7dc176d63dada1d5a7582063824` and capture-kit commit
  `3f1ab49b16cefaebf5d4c55e5aa59b0c5a0c149e`.
- Captured `rust-star/results/oracle-v3-20260830T111217Z.tar.gz`; archive
  SHA-256 is `be77e20c42875f370169c777e2cb26d090fbb516826dc215aa33488d4b4e37bc`.
  Independent verification passed 16 artifacts and 12,651,264 bytes.
- All four 2K processes produced JSON SHA-256 `2b3cdf9b...6ddb0`; all four
  32K processes produced `269f9529...58e12`. Both full 129,280-logit tensors
  are C0-identical to the accepted v2 values. The 32K prefill observations were
  178.78, 178.52, 178.19, and 177.88 tok/s; performance mode was intentionally
  disabled, so these are conformance observations rather than a benchmark.
- Imported the audited 32,768-token stream (SHA-256 `874c479e...02a0`) and
  packed FP32 logits (SHA-256 `603430b4...c547`) as
  `dwarfstar-oracle-v3-prefill-frontier-32768`. The differential verifier
  passed 648,192 bytes, and the Rust loader enforces the token endpoints,
  tensor shape, and selected token 85166.

Decision:

- Advance `current-oracle` from v2 to v3 for the unchanged 2K/32K C0 tensor
  values. V2 remains immutable historical evidence, but its producer is not a
  valid source for new long-context captures.
- Allow differential fixtures to declare any explicitly pinned supported
  oracle version; reject unknown IDs or mismatched commit/tree pairs.

Next:

- Generalize the native Rust prefill schedule from its fixed 2K lifecycle to
  eight 4K chunks while retaining per-layer raw, compressed, indexer, and
  recurrent compressor state across chunks, then compare its 32K logits to the
  newly embedded v3 fixture.

## 2026-08-30 — Residual 32K producer race isolated and synchronized

Objective:

- Reject or accept the stronger long-context producer under the new four-run
  `oracle-v3` contract, then isolate any remaining fresh-process drift.

Evidence:

- The first `oracle-v3` attempt correctly stopped at 32K repetition 2. Run 1
  had JSON SHA-256 `11db1c711ccd235127115eb3a211d6c5c0a91ee3be092e7f718d30c4048fbcfa`;
  run 2 had `269f9529f0e649d6e4f5ecd125ef7c4808f825e65bf9d503c512ef611c958e12`.
  Their argmax and top-50 set matched, but 129,278 of 129,280 FP32 logits
  differed (`max_abs=0.07219076`, `rmse=0.01476714`). The failed bundle is
  preserved under ignored target-Mac results as negative evidence.
- Zero-initializing every new Metal tensor and reusable scratch allocation did
  not stabilize two fresh 32K processes. One output matched the accepted
  numerical path and one differed much more (`rmse=0.22811150`), eliminating
  unspecified initial allocation contents as a sufficient explanation.
- Waiting for all submitted Metal work at every long-prefill compressor
  projection-to-read boundary produced byte-identical full-vocabulary logits
  in two fresh 32K processes. Both direct-prompt files had SHA-256
  `0a75e5998b3b1b41e6fb3a5d86516fda1bace2fe7d02d0b6efd04c74b84e93c1`
  and exactly matched the previously accepted tensor bytes.
- The synchronized diagnostic measured 180.99 and 179.09 prefill tok/s versus
  180.29 for the failed v3 run and 182.43--183.36 for the zero-initialization
  diagnostic. This is a small cost, not a separate performance claim.

Decision:

- Promote the host-waited compressor dependency boundary to the normal M1
  long-prefill path. An encoder boundary and an asynchronous command-buffer
  split both reduced the race frequency but were insufficient.
- Remove the broad allocation-zeroing diagnostic from the source. It changed
  scheduling and outputs without resolving the race.
- The failed v3 attempt is rejected. The synchronized source must be pinned as
  a new immutable producer and pass all four fresh-process repetitions before
  `oracle-v3` or any imported 32K Rust fixture is accepted.

Next:

- Validate and publish the synchronized source checkpoint, repin the v3
  capture contract, and run its complete four-repeat 2K/32K acceptance set.

## Current State


### Current resume checkpoint (updated 2026-09-12)

- COMPLETED SINGLE MILESTONE: explicit scratch reuse rejected.
  Full C0/pool proofs and all 12 engine runs complete; all accepted source/default restored.
  Report: rust-star/reports/pool-1k-fcdfef5f/README.md. 221 artifacts verify.
  Report this milestone and the next proposal, then STOP.
  Wide bootstrap c7adfb22 is rejected
  after all 12 engine runs and restored at 179e0c24. Verified report:
  rust-star/reports/wide-bootstrap-c7adfb22/README.md. Median prefill -1.127%,
  but two slower pairs and a failed order gate; memory footprint +52.484%.
- COMPLETE: declared Rust 1.74 compatibility. One Rust file changes with safe
  array borrows and equivalent Option checks; all 41 other pinned sources and
  the accepted performance executable remain unchanged. All 320 Rust/94 Python
  tests, optimized build, formatting and synthetic artifact checks pass. Report:
  rust-star/reports/msrv-174-179e0c24/README.md (50 artifacts, full source inverse).
- COMPLETE: routed-half local screen stops after five of six A/A cells fail.
  Runtime unchanged at 24c5d03e; all 16 edge cases and 72 A/A pairs complete.
  Report rust-star/reports/routed-half-neutral-24c5d03e/README.md verifies 67
  artifacts and 7,916,273,664 FP16 comparisons (includes A/A). Candidate timing
  never ran. Do not make a kernel speed claim from this failed control screen.
- COMPLETE: longer median protocol, 83 verified artifacts and 23,015,768,064
  exact FP16 comparisons. Initial controls pass; two mixed neutral cells and
  candidate speed gates fail. No reliable gain or integration. Report:
  rust-star/reports/routed-half-median-abe52d80/README.md. GPU validation is active below.
- COMPLETE: current Rust 1.74 normal/profiled 32K/128 full C0; 33,354,240
  exact FP32 comparisons and all 4,466 profile intervals pass. All final input,
  fixture, source and executable hashes match. Report:
  rust-star/reports/msrv-c0-5834fe56/README.md. Baseline evidence is complete.
- COMPLETE: 1K bootstrap study. All build/source/router/ring/internal/full-C0
  gates pass, including 33,354,240 exact FP32 words. All five engine pairs are
  faster; median paired prefill -11.234%, steady decode -0.317%, RSS -0.059%.
  Separate peak-memory-footprint +44.722% fails the <=15% gate. Candidate is
  rejected; all 42 accepted sources and the default executable are restored/
  unchanged. Verified 178-artifact report:
  rust-star/reports/bootstrap-1k-d72b7528/README.md. No GPU workload active.
- COMPLETE: bootstrap lifetime diagnostic. 60-artifact verified report:
  rust-star/reports/bootstrap-lifetimes-bff07722/README.md. Both fresh C0 runs
  pass 33,354,240 exact FP32 comparisons. Four 1K scratch sets overlap; earlier
  sets vanish after completion, then the final set after pool drain. Logical
  peak is 9.973 GB wide versus 11.181 GB accepted; this does not explain higher
  OS footprint. Preserve the original 32/64 expected-width audit failure and
  its single-field correction on unchanged runs. Runtime/source/default unchanged.
- COMPLETE, REJECTED: bounded 1K waits pass 50,031,360 exact FP32 comparisons,
  12K internal checks and all profile/build/local proofs. Four snapshots at the
  20K frontier violate scratch-release conditions despite completed prior GPU
  work; 5 MiB of older buffers remains at the next registration/submission.
  One unfinished command and 75% lower global logical peak do not override the
  failed prerequisite. No engine runs; all accepted source/default restored.
  Verified 116-artifact report: rust-star/reports/bounded-1k-34eabe66/README.md.
- COMPLETE: explicit scratch reuse. paired median prefill -10.243%, steady decode -0.005%, RSS +2.415%, peak memory footprint +11.663%.
  REJECTED; all accepted source/default restored. Full C0/pool/engine evidence and
  portable report are verified. Next: Diagnose scratch RSS residency and prove which large buffers can safely avoid unconditional clearing before another 32K candidate.
  No next-milestone implementation/inference started; keep256K paused.


- COMPLETE: attention-output whole-engine milestone. Candidate 4ced6ee8 is
  retained. Report rust-star/reports/attention-output-engine-4ced6ee8/README.md.
  All neutral-control and candidate gates pass (53.553 billion isolated FP32
  comparisons); integrated full 32K/128 C0 passes all 16,677,120 FP32 words.
  All 320 host tests, production Metal and source-scope checks pass.
- Five complete alternating unprofiled engine pairs against frozen Rust 9d35953f
  show median paired prefill latency -4.329%, steady decode -0.190% and
  peak RSS +0.001%. First-step latency changes -1.966%.
  Two warm-ups are excluded and preserved; all 12 transcripts are exact.
  See every raw pair and preset retention gate in the verified report.
- All final input/source/binary pins pass. GPU work is complete. The new path
  has full-model evidence at 32K only; 256K stays paused. No DwarfStar speedup
  or new higher-frontier claim follows from this Rust-versus-Rust comparison.


- COMPLETE: native compressed run-skip 9d35953f passes all three full 32K/128
  C0 checks and prospective component selection. Report
  rust-star/reports/32k-compressed-skip-9d35953f/README.md. Retain the candidate;
  raw component time is 6.76% lower / 4.95% lower after control normalization,
  with heterogeneous chunks. Only 0.828% of recorded GPU stage time is saved;
  no accepted engine gain, higher-frontier gate or DwarfStar parity. All GPU
  work is finished; no additional benchmark is queued.


- COMPLETE: isolated empty-block run-skip candidate passes. Report
  rust-star/reports/compressed-skip-runs-fbbea533/README.md verifies 56 artifacts.
  All exactness and contemporary A/A gates pass; every candidate pair improves,
  with 1.14–6.08% lower order-specific median kernel time. Select for later
  narrow real-model validation; no runtime change or engine gain is accepted.
  All GPU work is finished.


- COMPLETE: fresh controlled head-first screen. Report
  rust-star/reports/compressed-grid-warm-c21ebfda/README.md verifies 41 artifacts.
  Full isolated exactness and every A/A gate pass; candidate medians are
  1.36–2.83% slower in both orders and fail selection. No runtime integration
  or full-model follow-up for this candidate. All GPU work is finished.


- COMPLETE: bounded A/A timing-protocol mitigation. Report
  rust-star/reports/compressed-aa-protocol-a6bf9e24/README.md verifies 39
  artifacts. Both fresh processes pass all six preset null screens and
  20.938 billion FP32 comparisons. Equal-warm order medians stay within 0.487%
  of parity. Retain this scoped method with A/A controls for later candidates;
  no optimization or engine gain is accepted. All GPU jobs have finished.


- COMPLETE: 32K attention breakdown and one bounded compressed-grid screen.
  Report rust-star/reports/32k-attention-1711856e/README.md verifies 74 artifacts.
  Diagnostic runtime 1711856e passes full C0 and strict component coverage;
  retain instrumentation only. The head-first wrapper passes isolated exactness
  but fails speed selection, and A/A confirms this harness's order sensitivity.
  No engine gain or larger frontier is accepted; no GPU run is active.

- COMPLETE: bounded 32K routed-prefill grid-order validation at 45520435.
  Report rust-star/reports/32k-route-order-45520435/README.md verifies 48
  artifacts, all three full-output C0 runs and both strict profiles. Final
  input/source pins pass. Normalized aggregate/median gate ratios are
  0.985846/0.985016; retain as a correctness-validated candidate only.
  No engine speedup is accepted. Preserve first candidate 9171518a's shader
  compilation failure. No GPU run remains active.

- COMPLETE: bounded lower-context routed-prefill grid-order screen. Report
  rust-star/reports/prefill-route-order-d6c14fa0/README.md. Synthetic exactness
  passes 2.270 billion FP16 comparisons; 4K balanced/skewed paired kernel
  latency falls 8.72%/1.91%, all 14 pairs faster. That standalone screen left
  runtime d6c14fa0 unchanged; the subsequent real-model result is above.
  Synthetic kernel improvement does not establish engine improvement.

- PAUSED: 256K capacity/correctness milestone, after the user challenged its
  priority while lower-context prefill remains slower than the reference.
  Frozen d6c14fa0 passes eight gates through 128K full C0 and 192K prefill;
  its fresh 256K run is interrupted, and its retained checks were not started.
  Preserve both earlier rejected candidates and all current evidence under
  rust-star/.work/frontier256-handoff-bounded/. See interrupted-attempt.json.
  Candidate binary/runtime hashes are unchanged. No final acceptance, completed
  report, performance claim or completion commit. Do not resume automatically.


- COMPLETE: authorized 128K profiling and focused optimization milestone.
  Retain candidate b6b0f125: all ten live gates, 317 Rust/94 Python tests,
  release build, five first-attempt pairs and two warmups pass. Median paired
  prefill latency falls 8.28% (5/5 faster); steady throughput rises 1.06%;
  peak-RSS ratio is 1.000015. All first-step results and dispersion remain
  reported, including the slower fifth pair. Strict profiles, isolated score
  controls, independent metrics/artifact audits and final input hashes pass.
  Report rust-star/reports/128k-score64-b6b0f125/README.md verifies 322 artifacts.
  Runtime candidate is frozen at b6b0f125; the completion commit adds evidence
  and documentation. Earlier rejected attempts are preserved. All jobs have
  ended. Stop here; no next milestone or push is authorized.

- COMPLETE: 128K measurement/retained-engine milestone, candidate ef37e566.
  317 Rust / 94 Python tests, six live gates, five valid pairs and two warm-ups;
  portable report has 167 verified artifacts. See the newest completion entry
  and rust-star/reports/128k-measure-ef37e566/README.md for measured latency,
  throughput and warmed memory. Independent audits and final pins pass.
  Earlier writer rejection is preserved. Its follow-up milestone is active above.

- 128K correctness milestone COMPLETE and retained at candidate 0eb484ea.
  314 tests/build, six repeated oracle captures and all eleven native checks
  pass, including two full 128K/128 C0 runs. Profile audit 287 tails/2,583 rows;
  final input pins match. Report rust-star/reports/128k-c0-0eb484ea/README.md
  contains 117 verified artifacts. All requested work is complete; stopped.
- Separate external-reference diagnostic execution now reaches 128K. Built-in
  diagnostic fixtures retain 32K; timed fresh/retained support now includes the
  pinned 128K/128 transcript. Internal HC targets retain 24K.
  That correctness report made no throughput or total-memory claim. Diagnostic
  RSS omits benchmark page warming; the new report supplies warmed measurements.
- The subsequent measurement/retained-engine milestone is complete as recorded
  above. Keep the prior correctness and performance reports immutable.

- Memory-prefill milestone COMPLETE and retained at de6d2c11. 312 tests/build,
  seven live gates, five valid pairs/two warmups and both profile audits pass.
  Report rust-star/reports/32k-memory-de6d2c11/README.md (181 verified artifacts).
  Median paired peak RSS −8.45% (7.74GiB saved); prefill latency −15.75%; steady
  decode −0.31%, first-step latency +2.43%. Partial bootstrap wraps 18 views;
  native activation scratch uses 4K rows. KV/compressed history remains absolute.
- 128K modeled allocation/fill preflight passes 14.837GiB. Planning estimate
  93.173GiB remains a historical planning estimate, not a bound. The subsequent
  completed milestone above independently establishes scoped full 128K C0.
- Earlier bootstrap/counter milestone 912fae1e remains accepted history:
  rust-star/reports/32k-bootstrap-912fae1e/README.md (403 artifacts), 8.165% paired
  prefill improvement then. Oracle-v5 remains the correctness target; the latest
  Rust-versus-DwarfStar throughput result is unchanged by Rust-to-Rust studies.
- Dense-Q8 candidate 92ca3ef3 is rejected and reverted after seven passing live
  gates and five valid pairs. Paired prefill ratio 2.01485 versus baseline 1.72589;
  Rust/reference medians 351.974/176.897 s. One host-interrupted pair is preserved
  and was rerun whole under the documented recovery rule. Negative report:
  rust-star/reports/32k-dense-output-92ca3ef3/README.md.
- The earlier dense-output post-paired profile failed with invalid timestamps.
  Its raw values were not captured, so the cause remains unknown. Diagnostics
  now preserve all values/context and reject sentinels; two current profiles
  pass, with strict validity and all historical failures preserved.
- Tail component profiler e7b81aa2 remains: 312 tests, normal/profiled full-output
  gates and 2,583 component rows audited. Report:
  rust-star/reports/32k-tail-components-e7b81aa2/README.md (16 artifacts).
  Attention output was the largest attributed component (36.75%). No throughput
  claim for profiling itself. The prior restored execution matched 9c1d5009;
  all 232 negative-report artifacts still verify. Scoped bootstrap 912fae1e is
  now the accepted execution path. The component-profiler milestone is closed.
- User restored milestone stopping on 2026-09-08. The burn recap and the
  subsequent bootstrap/counter milestone are complete. Wait for the next user
  instruction before starting another milestone.
### Latest accepted Rust-versus-DwarfStar comparison (historical native tail study)

- Authorized native tail scratch/output milestone is complete. Whole-chunk
  tails reuse 22 temporary buffers, write directly to retained HC output and
  omit unused FP32 routed-mid conversion (25 dispatches instead of 26).
  Retained HC remains separately owned; cache clears at decode handoff/reset.
- Candidate `89e3274568745fe5311d16478ccf35f016958a08`;
  executable SHA `7f14080aea91a3631bb85b8c55dd8f02031e983ab2b74cc1618235a0262a74a4`.
  Oracle-v5 source ccbd16d remains unchanged; executable SHA
  `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`.
- 310 Rust tests and all six live gates pass: native 12K, detailed 8K, 2K,
  complete 32K output-only/diagnostic and two retained reset/reuse cycles.
  Unchanged Python toolkit retains its 93-test reference acceptance evidence.
- Five alternating pairs and both warmups pass on attempt 1; separate profile
  and independent audit pass. Report: `rust-star/reports/32k-tail-89e32745/README.md`.
  Rust/reference median prefill 362.284/212.376 s; steady decode 17.064/17.039 tokens/s. Observed Rust peak memory 91.613 GiB.
- Absolute paired prefill is 2.51% lower; the reference is 3.03% slower.
  OBS/video activity and 3098.88 → 2794.69 MiB host swap limit cross-session
  speed and memory attribution. Even-tail host time falls 66.497 → 59.400 s;
  GPU time rises slightly. All 210 report artifacts verify independently.
  No fixture, arithmetic or oracle change.
- The historical largest GPU group was `even/tail`. Its component costs are
  now measured by the component-profiler and scoped-bootstrap reports above.
- Frozen plan SHA `d27a9f5e56d6c804f62c10173a9a4e324498eba52d40227709b7d23cf15efe2f`.
  Evidence: `rust-star/.work/tail-milestone-2d8b8405/`. All prior reports/failures stay immutable.
- Internal HC targets remain accepted through 24K; full 32K output gates are
  mandatory. Broader contexts, 256K headline and agent co-design remain future.

### Accumulated implementation and measurement evidence

- Phase: target-Mac bootstrap, quick `oracle-v1`, no-copy model mapping, the
  canonical differential-fixture envelope, complete layer 0, steady-state
  layer-0 execution, the first four-layer scheduler boundary, the minimum
  three-step position-advancing four-layer slice, and the generalized six-layer
  compression-schedule boundary are complete. A bounded position-127 replay
  first crossed the ratio-128 emissions for layers 3 and 5 with external
  activations; the integrated decoder now crosses the same boundary while
  committing the complete 128-token oracle transcript. The
  ordered position-advancing executor now extends through all 43 transformer
  layers and the exact post-transformer output boundary is complete through
  full-vocabulary logits and deterministic greedy selection. A separate
  four-position command remains an independent control. A position-127 command
  closes the feedback edge at every step, matches the final full-vocabulary
  logits and live layer-3/layer-5 ratio-128 rows, and has a diagnostic timed
  path with correctness readback excluded. A separate cold-prefill command now
  constructs position-0 raw/compressor state from the live prompt token, matches
  the complete prefill logits, and preserves the same position-127 exactness.
  Layers 0 through 42 execute in order under one Rust-owned Metal context with
  exact GPU-resident HC-state handoffs and one retained KV allocation per
  layer. Every even layer from 2 through 42 crosses its first ratio-4
  compressed-attention emission boundary at position 3, and the emitted rows
  remain visible at position 4; odd compressed layers retain ratio-128 state
  without emitting at this frontier.
  Layer 3 crosses from token-hash routing to biased top-k routing.
- Working branch: `agent/rust-star-bootstrap`.
- Branch base: upstream `antirez/ds4` commit
  `b0309611041655f4e45671cfd9c9886aff161406`.
- Local `main`: fast-forwarded to the same upstream commit.
- Fork `origin/main`: synchronized to the same upstream commit through the
  GitHub app.
- Oracle: `oracle-v3` is the active 2K/32K C0 contract. Its host-synchronized
  producer passed four exact fresh processes at both frontiers and preserves
  the accepted v2 tensors bit-for-bit. V2 remains immutable evidence, but its
  producer is revoked for new long-context work after rare drift. The
  historical `oracle-v1` 32K tensor remains quarantined and immutable.
- Capture kit: `rust-star/capture_oracle_v1.py` preserves the historical
  capture path. `rust-star/capture_oracle_v2.py` prepares the deterministic
  privacy-filtered bundle and requires repeated fresh-process C0 evidence; the
  verifier accepts both immutable versions and rejects v2 repetition drift.
  `rust-star/capture_oracle_v3.py` raises the producer gate to four exact fresh
  processes at both frontiers, correctly rejected its first candidate, and
  accepted the host-synchronized `d35fb12` producer.
- Differential tooling: the initial bundle/full-logit format is stable and has
  cross-platform verification, exact C0 comparison, and drift diagnostics. The
  `rust-star-differential-fixture-v1` envelope now covers kernel,
  layer-segment, and decode-step scopes with strict tensor shape, path,
  finiteness, size, and hash verification.
- Benchmarking: `rust-star/BENCHMARK_PROTOCOL.md` fixes the v1 paired workload,
  eligibility, run ordering, aggregation, and capacity semantics. The paired
  raw/summary JSON contract and offline validator/aggregator are implemented
  and synthetically tested. A fresh-process DwarfStar adapter now normalizes
  one frontier while externally recording wall time and peak RSS. The
  checkpointed paired runner
  enforces warm-up, A/B and context ordering, explicit retries, and final
  validated export through an engine-neutral contract. Rust Star now has a
  fresh-process adapter and a raw `rust-star-engine-run-v1` producer for the
  exact 2K/128 path. Generation uses timing-only submission and validates the
  selected-token transcript afterward. The adapter independently rejects any
  raw run whose prefill/generation collection, transcript, or scheduler
  metadata is inconsistent. Native prefill now has a separate execution-only
  mode that retains its GPU state while omitting diagnostic fixture decoding,
  host boundary allocation, and boundary copies. The first five-pair target-Mac
  2K/128 development comparison completed without retries or invalid attempts.
  Rust Star's paired median was 0.7534x DwarfStar for steady decode, 0.3083x for
  complete generation, and 0.4133x for prefill. The first-token wall interval
  was 225.84x the oracle median pairwise ratio. Per-layer timing then traced the
  transition cliff to full-2K transient scratch retained across all 43 prefill
  layers, which exhausted unified-memory headroom and evicted early mapped
  weights before decode. Production prefill now pools same-role dead scratch,
  ping-pongs the cross-layer HC state, and releases prefill storage after the
  GPU state handoff while leaving diagnostic collection and persistent decoder
  state distinct. Two exact fresh-process 2K/128 development runs improved the
  first generated step to 833.639 and 732.984 ms, steady decode to 20.555 and
  20.789 tok/s, complete generation to 18.254 and 18.708 tok/s, and prefill to
  125.378 and 129.047 tok/s. Against the earlier five-pair medians this is about
  a 14.9x first-token improvement, 18.8% higher steady decode, 2.66x higher
  complete-generation throughput, and 73.3% higher prefill. The immutable
  `35ee7bb` candidate then completed a new five-pair exact comparison without
  retries or invalid attempts. Rust Star measured 20.031224379 tok/s steady,
  18.162539840 tok/s complete generation, 124.303074737 tok/s prefill, and
  708.321500 ms first-token latency. DwarfStar measured 23.56, 23.16, 181.83,
  and 47.741 ms respectively. The median within-pair ratios were 0.850831888x
  steady, 0.783881737x complete generation, 0.683350269x prefill, and
  14.531941455x first-token latency. Relative to the prior Rust medians, the
  validated candidate improved steady decode 14.5%, complete generation 2.58x,
  prefill 66.9%, and first-token latency 93.5%. The remaining paired steady gap
  was approximately 14.9%. A new explicitly ineligible per-layer profile found
  that steady transformer cost is distributed across the repeated layer
  schedule rather than concentrated in one pathological layer. Removing three
  unused diagnostic snapshots per layer, performing both RoPE transforms in
  place, and retaining a compute encoder across adjacent FFN stages reduced
  profiled steady transformer GPU time by 9.16% and raised profiled steady
  decode from 20.171 to 21.938 tok/s. Two fresh eligible runs reproduced 21.864
  and 21.769 tok/s with the exact 128-token transcript. Immutable commit
  `771b7c4` then completed five new exact pairs without a failure, retry, or
  invalid attempt. Rust Star measured 21.875065281 tok/s steady versus
  DwarfStar's 22.66, for a median within-pair ratio of 0.964571386x. The
  validated remaining steady gap was approximately 3.54%. Production-only GPU
  top-1 now avoids transferring and scanning the 129,280-logit row during timed
  greedy sampling. Immutable commit `4fc61f7` completed five new exact pairs
  without a failure, retry, or invalid attempt. Rust Star measured
  21.940665084 tok/s steady versus DwarfStar's 22.61, for a median within-pair
  ratio of 0.969791317x. The validated remaining steady gap is now approximately
  3.02%. A new M1-native diagnostic profiler now attributes the transformer
  command chain across eight dispatch families using one shared Metal timestamp
  counter buffer. It is explicitly ineligible and splits compute encoders only
  when `engine-profile` enables it. Routed experts account for 25.31% of the
  profiled steady transformer GPU interval, attention QKV ingress 22.82%, and
  attention output/HC 14.72%; together they account for 62.85%.
  A single-context alternating routed-expert launch benchmark now compiles the
  NSG=2 and NSG=4 pipelines together and checks every output against the pinned
  layer-0 fixture. On 100 measured rounds, NSG=4 was 1.83% slower in median GPU
  time, so production remains at DwarfStar's NSG=2 geometry.
  Attention QKV ingress now matches pinned DwarfStar's paired Q-A/KV Q8_0
  projection, removing one decode dispatch per layer. A 100-round alternating
  single-context diagnostic measured 0.213 ms paired versus 0.216 ms separate
  median GPU time for the exact attention-setup chain, a 1.44% reduction.
  Attention output/HC launch tuning is now bounded by an exact same-context
  harness. All four independent NSG=2/8 alternatives changed the Q8 reduction
  result, so production remains at pinned DwarfStar's NSG=4/4 geometry.
  Immutable commit `7de9511` then completed five exact 2K/128 pairs without a
  failure, retry, or invalid attempt. Rust Star measured 23.495447999 tok/s
  steady versus DwarfStar's 22.91, and all five pairs favored Rust Star. The
  median within-pair steady ratio was 1.025554256x: Rust Star is now 2.56%
  faster at this development frontier. Complete generation remains 3.60%
  behind at 0.964026748x, prefill remains 31.54% behind at 0.684638182x, and
  first-token latency remains 12.192259243x higher. This is an exact 2K/128
  development result, not the protocol's 256K headline claim.
  Immutable commit `2c6f80e` then completed five exact 2K/128 pairs without a
  timed failure, retry, or invalid attempt. It registers all 1,136 existing
  no-copy model views in a residency set attached to the inference queue and
  performs a coarse GPU touch inside `prefill_ms`. Rust Star measured
  24.230812222 tok/s steady versus DwarfStar's 23.39, and all five pairs favored
  Rust Star. The median within-pair steady ratio was 1.037257019x, a validated
  3.73% lead. Complete generation measured 24.198539179 versus 22.81 tok/s; all
  five pairs favored Rust and the median ratio was 1.061823177x, a 6.18% lead.
  Median Rust first-token latency fell from `7de9511`'s 546.059208 ms to
  48.720167 ms and is effectively tied with DwarfStar at a 0.995386079x paired
  ratio. Prefill remains 34.07% behind at 0.659348296x. This is an exact 2K/128
  development result, not the protocol's 256K headline claim.
  Passive prefill attribution now separates the bootstrap, remaining
  transformer, output head, handoff, residency, and residual host cost. The
  timing-only bootstrap queues Metal command buffers and synchronizes only at
  the tail; the exact 128-token transcript remains eligible. Immutable commit
  `eae11df` retained 64 exact 32-row tiles and completed five
  exact pairs without a failure, retry, or invalid attempt. Rust Star's paired
  prefill ratio improved from `2c6f80e`'s 0.659348296x to 0.673232187x, reducing
  the validated 2K gap from 34.07% to 32.68%. Steady decode remains 3.71%
  faster than DwarfStar and complete generation remains 6.45% faster. The
  initial 64-row rejection was subsequently traced to fixed one-tile matrix
  grids, a fixed four-group FlashAttention grid, and a 32-row grouped-output
  routing allocation. Immutable checkpoint `cdb6377` makes those geometries
  row-derived and uses 32 exact 64-row production tiles while retaining all
  32-row diagnostics. Immutable `cdb6377` completed five exact pairs without a
  failure, retry, or invalid attempt. Rust Star measured 133.070209951 prefill
  tok/s versus DwarfStar's 186.35; the median pairwise ratio is 0.713851042x
  and the remaining gap is 28.61%. Rust retains a 3.87% steady-decode lead and
  6.47% complete-generation lead. A full-2K bootstrap remains rejected because
  the 128-row raw-cache append requires wrapping. An explicitly ineligible
  synchronized per-layer prefill profile now attributes layers 2 through 42.
  Rust's 21 even layers average 326.577 ms GPU and its 20 odd layers average
  251.904 ms (1.296x); DwarfStar's split profile averages 283.103 and 223.180
  ms respectively (1.268x). The remaining transformer deficit is distributed
  across the alternating attention families rather than one anomalous layer.
  Representative layers 4 and 5 are now further split into QKV/RoPE,
  compressor/staging, Flash block preparation, FlashAttention, and output/HC.
  Three exact Rust profiles and three Dwarf profiles per layer put the median
  main FlashAttention dispatch about 30% behind in both families while every
  surrounding Rust stage is at or ahead of Dwarf. Kernel source, constants,
  launch grid, threadgroup memory, fast-math defaults, and M1 shared-storage
  policy match, so the next bounded experiment targets NSG scheduling rather
  than buffer ownership or arithmetic changes.
  That experiment is now accepted: the shared non-vector FlashAttention
  specialization and all 43 matching launches use four rather than eight SIMD
  groups. Three exact profiles cut the representative kernel medians by 63.4%
  and 65.4%, and three eligible controls raised median prefill from 138.083 to
  171.682 tok/s without changing the 128-token transcript or decode schedule.
  Immutable `0cbffdf` completed five exact pairs with no retries or invalid
  attempts. Rust measured 163.873 prefill tok/s versus DwarfStar's 182.370;
  the 0.897865x median pairwise ratio leaves 10.21%, down from `cdb6377`'s
  28.61% validated gap. Steady decode is effectively tied/slightly ahead at
  1.001367x and complete generation is 1.014559x ahead. With the profiled Rust
  transformer now faster than Dwarf's earlier split total, the next target is
  the 64-row bootstrap/residency interval before the transformer. A bounded
  128-row bootstrap attempt matched the first five generated selections but
  diverged at step 6, coincident with the first new ratio-4 compressor row, and
  was fully reverted. Eligible runs now passively report context, bootstrap,
  transformer, and output-head setup time inside the existing host-overhead
  interval. Three clean exact controls measured a 173.252 tok/s median and only
  95.057 ms median across those four setup phases. Additional passive fields
  then localized 751.101 of 884.340 ms host overhead to model-residency
  preparation, with only 15.646 ms around the decoder-state handoff and 0.176
  ms unattributed. Bulk residency insertion, overlapping residency preparation
  with the transformer, and omitting the eager residency request were each
  exact but did not reduce complete prefill; all three candidates were reverted.
  A separate retained-lifecycle diagnostic now answers the amortization
  question without weakening the fresh-process contract. It runs two exact
  2K/128 cycles in one process and one Metal context, fully charges first-load
  page warming and residency, resets request-scoped activation/KV state, and
  requires the second cycle to reuse the existing view cache and residency set.
  Two corrected exact runs measured 172.518--174.154 tok/s first-load prefill
  and 185.554--186.311 tok/s retained prefill while avoiding 11.426--13.428
  seconds of host page touching and 567.920--640.124 ms of residency
  request/touch work. All four transcripts matched. These same-process results
  are explicitly paired-ineligible and establish the lifecycle mechanism, not
  a replacement throughput claim.
- Implementation: dependency-free Rust host scaffold under `rust-star/runtime/`
  strictly parses GGUF v3 directories, validates the Flash resident-Q2
  shape/recipe, and writes candidate full-logit artifacts. A macOS-only
  Rust/Objective-C/Metal boundary now owns a device, queue, pipeline, and
  correctness-checked shared-buffer dispatch probe. The complete Objective-C
  shim, optimized Rust build, shared-buffer probe, and real-model validator now
  pass on the M1 Ultra. The real GGUF exposed and fixed an incorrect Q8_0
  expectation for the F16 indexer Q projection. Rust now owns a read-only
  shared GGUF mmap; Metal wraps the page-aligned embedding tensor without a
  copy and runs the imported DwarfStar F16 embedding gather behind a bitwise
  CPU reference. The imported decode-time Q8_0 matvec now consumes a real
  layer-0 activation and the no-copy `blk.0.attn_q_a.weight` span, reproducing
  all 1,024 FP32 DwarfStar fixture outputs by bit pattern. The first connected
  chain now runs six imported operations in one command buffer from token 201's
  embedding through Q-Lora, with six mmap-backed model spans and bitwise checks
  at mixer, HC split, collapsed HC, learned attention norm, and Q-Lora.
  The connected path now covers all 29 layer-0 dispatches through Flash
  attention, output projection, hash routing, routed/shared experts, and final
  HC update. A narrow Rust `LayerExecutor` owns the Metal context, permanently
  binds it to one model mmap, and caches exact model views and activation
  buffers. It executes layers 0 through 3 in order; later layers omit
  embedding/repeat and consume the preceding layer's retained final HC Metal
  buffer directly. KV storage is keyed by layer identity rather than reusing
  one scratch allocation. All four layers use independently captured cache-row
  and differential fixtures and pass every retained boundary by FP32 bit
  pattern. Layer 2 derives its compressed RoPE base, scale, original context,
  and YaRN parameters from the model's compression schedule. A separate exact
  scheduler path commits the declared layer command buffers without inter-layer
  waits, retains layer-scoped activation lifetimes for deferred comparison, and
  performs one tail wait after the declared chain tail. The legacy three-layer
  chain declares layer 2, while the four-layer chain declares layer 3, so both
  remain independently validated scheduler controls. Layer 3 binds the model's
  256-value `exp_probs_b.bias` and uses biased top-k selection instead of the
  early layers' token-indexed hash tables. The four-layer steady-state harness
  now resolves all 100 model spans, decodes the four fixtures, and allocates
  host outputs once. Warmup and measured chains reuse those bindings; measured
  iterations collect only Metal timing metadata, followed by one exhaustive C0
  readback after the final sample. The same prepared executor now advances from
  token 201/position 1 through token 361/position 2 to token 1915/position 3
  without recreating its six layer-scoped KV caches. It verifies preserved
  FP16-rounded raw-cache history, grows each cache from two to four visible
  rows, derives RoPE/attention/router state from the explicit token and
  position, and hands off layer 5's final 16,384-element HC state. The original
  four-layer path remains an independently executed regression control. The
  decoder now owns recurrent attention-compressor state for layers 2 through 5
  plus indexer-compressor state for the even compressed layers 2 and 4. On the
  M1 Ultra it follows DwarfStar's
  separate paired F16 projection and one-row state-store path, retains the
  legacy concat/softmax/multiply/sum reduction for the first ratio-4 emission,
  applies learned norm, compressed RoPE, state shift, and FP8/indexer QAT, then
  appends the emitted layer-2 and layer-4 rows to attention. All retained
  boundaries match pinned position-1 and independently repeated position-2/3
  DwarfStar fixtures bit-for-bit. The generalized schedule applies ratio 4 plus
  indexer compression to even compressed layers and ratio 128 without indexer
  compression to odd compressed layers; at this frontier layers 2 and 4 emit
  while layers 3 and 5 only accumulate state. The new bounded ratio-128 replay
  independently owns 128 activation rows, projected KV/score state, and four
  no-copy model-weight views for each of layers 3 and 5. It submits all 128
  steps in one command buffer, performs the legacy single-row reduction only at
  position 127, and matches both 512-value DwarfStar emissions by FP32 bit
  pattern. Its inputs are externally captured `attn_norm` activations: it does
  not sample tokens and is not a complete decoder execution. The prepared
  executor's checked contiguous-chain contract now accepts eight layers. Layers
  6 and 7 retain independent raw caches and compressor states; layer 6 matches
  its first ratio-4 attention emission and indexer update at position 3, while
  layer 7 advances ratio-128 state without emitting. The four-layer and
  six-layer commands remain independently executed regression controls.
  The remaining layer-8–42 boundaries are represented by one checked fixture
  registry rather than pair-specific branches. The 43-layer command preserves
  all earlier controls, retains 43 raw caches and all layer-scoped compressor
  state, advances three exact decode positions, and hands off layer 42's final
  16,384-element HC state. The separate `decoder-output-probe` consumes that
  retained state without a host handoff and reproduces the four-value HC
  projection, HC weights, collapsed HC row, learned output norm, and all
  129,280 vocabulary logits at positions 1–3 by FP32 bit pattern. Lowest-ID
  argmax selects 361, 1915, and 262. The original command keeps those inputs
  external as a control. `closed-loop-decoder-probe` instead starts from token
  201 and feeds each selected token into the next position, then repeats the
  exact sequence without chained tensor collection. The timed intervals retain
  the required full-logit transfer and CPU argmax but exclude fixture readback
  and comparison. `position127-decoder-probe` extends the same ownership
  boundary through positions 1–127. With initial committed token 201 it
  reproduces all 128 oracle tokens, then compares the final 129,280 logits and
  persistent layer-3/layer-5 ratio-128 compressed rows bit-for-bit.
  `cold-prefill-decoder-probe` instead starts with empty Rust-owned cache and
  compressor storage, evaluates raw prompt token 36662 at position 0, matches
  all 129,280 prefill logits and selected token 201, and continues through the
  same 128-token transcript without loading captured initial state. A separate
  2K sequential initializer now owns a true 128-row raw-KV ring and
  context-sized compressed state. It matches two fresh DwarfStar one-token
  decode replays bit-for-bit at the final logits, but all 129,280 logits differ
  from two independently repeatable native batched-prefill captures while both
  select token 15342. The first native-batch arithmetic boundary is now isolated:
  the layer-0 hidden-combination and learned-normalization inputs are identical
  between schedules, while `blk.0.attn_q_a.weight` uses the M1 legacy
  `kernel_mul_mm_q8_0_f32` batch kernel and first diverges from the one-row
  matvec. A compact final-128-row fixture and live probe reproduce both kernels
  bit-for-bit and retain their expected 1,024-value final-row disagreement. A
  second compact final-32-row boundary now continues through the layer-0 KV
  projection, fused Q/KV learned RMSNorm, Q-B projection, and Q head
  RMSNorm/RoPE. Five no-copy model views and all 2,195,456 produced FP32 values
  match repeated native DwarfStar captures by bit pattern. The same final tile
  now starts from its 32 token IDs: embedding gather, four-stream HC repeat,
  plain RMSNorm, the legacy F16 batch mixer, and fused HC collapse/learned norm
  feed that Q/KV schedule in one ten-dispatch command buffer. Ten no-copy model
  views and all 2,457,600 produced FP32 values match repeated native DwarfStar
  captures by bit pattern. That boundary now continues through grouped
  attention output, FFN HC ingress and learned norm, the decomposed M1 router,
  fused IQ2_XXS routed gate/up plus weighted SwiGLU, Q2_K routed down and sum,
  the Q8_0 shared expert, and the additive FFN HC tail. The complete final-tile
  layer uses 43 dispatches and 25 no-copy model views; all 6,979,776 retained
  produced FP32 values and 192 selected expert IDs match repeated native
  DwarfStar captures exactly. A separate preserved boundary continues the same
  live command buffer into layer 1's HC ingress, learned attention norm, and
  Q-A projection. It uses 47 dispatches and 30 no-copy model views; the three
  new final tiles add 294,912 exact FP32 values without a host activation
  handoff. The 84-dispatch complete boundary now finishes layer 1's attention,
  routed/shared experts, and additive FFN HC tail with 49 no-copy views. The
  generalized executor is exact for two adjacent tiles spanning positions
  1984--2047. Its independent control loads a captured prefix for each tile;
  its persistent-context control instead retains both layers' live first-tile
  KV buffers and makes the final tile append to and consume that state. The
  final execution prefix is not assembled from the capture, while an explicit
  oracle comparison still guards both retained buffers before continuation.
  The same ownership contract now runs all 64 native tiles from position 0
  through 2047. It starts with empty KV buffers, consumes all canonical prompt
  tokens, and validates both accumulated layer prefixes before every append.
  This establishes exact complete-2K layer-0/layer-1 KV state. Every live
  layer-1 output tile now also feeds native layer-2 HC ingress, Q-A/KV
  projection, and fused learned norm. All 1,048,576 values in the repeated
  full-2K layer-2 `KVnorm` oracle match bit-for-bit, closing downstream
  validation of the non-final layer-1 outputs. The same 64-tile schedule now
  applies layer 2's compressed-attention RoPE and E4M3FN finalization, retains
  one live full-2K layer-2 raw-KV allocation, and matches every `KVrope` and
  `KVcur` value plus each accumulated prefix. The same empty-seed loop now owns
  both layer-2 ratio-4 compressors. It matches all 512 attention and indexer
  compressed rows and the four final recurrent state tensors bit-for-bit while
  preserving 65 no-copy model mappings per tile. The same persistent context
  now retains every normalized layer-2 query row and executes exact dense mixed
  attention over 2,048 raw plus 512 compressed rows. Q-B, compressed YaRN,
  FlashAttention, inverse RoPE, and both attention-output projections reproduce
  all 8,388,608 attention-output values bit-for-bit with four additional
  no-copy model mappings. The terminal command now continues through attention
  HC post-processing, the full token-hash routed/shared FFN, and the additive
  FFN HC update. Full 2K attention-HC and final-HC identities plus exact
  final-tile intermediates match the repeated oracle, completing native layers
  0, 1, and 2 at the 2K prompt boundary. The terminal command now carries that
  full final HC state directly through layer 3's HC attention ingress, learned
  norm, Q-Lora projection, Q/KV learned normalization, compressed RoPE, and FP8
  KV finalization. It retains all 2,048 layer-3 raw-KV rows plus the exact Q,
  HC, and attention-split state for continuation. All ten layer-3 boundaries
  match repeated DwarfStar captures exactly. The same terminal command now owns
  layer 3's full ratio-128 attention compressor. All 16 emitted FP8 rows and
  both 128x512 final recurrent-state tensors match two fresh DwarfStar processes
  bit-for-bit. It now also owns layer 3's dense mixed attention, inverse RoPE,
  Q8 output projections, and additive HC post. The full 2048x4096 attention
  output and final HC tile match repeated captures exactly, with the full HC
  identity checksum-pinned. The same command now owns layer 3's complete FFN:
  HC ingress, learned norm, biased top-6 selection, unbiased router-weight
  normalization, routed/shared experts, and the additive final HC update. All
  final-tile intermediates and the full final HC identity match two fresh
  DwarfStar processes exactly. The expanded command preserves 44/44 terminal
  no-copy mappings across 79 dispatches. That retained final HC now feeds layer
  4's HC attention ingress, Q/KV learned normalization, compressed RoPE, and
  FP8 KV finalization without a host activation upload. All ten final-tile
  boundaries match four fresh DwarfStar captures; the complete tensor
  identities are SHA-256 pinned. The expanded terminal command now also owns
  both layer-4 ratio-4 compressors. All 512 attention rows, all 512 indexer
  rows, and all four final recurrent-state tensors match two fresh DwarfStar
  processes bit-for-bit. Three additional no-copy views now drive layer 4's
  dense mixed attention over 2,048 raw plus 512 compressed rows, inverse RoPE,
  both Q8 output projections, and the additive HC post. The full 2048x4096
  attention output and final HC tile match two fresh DwarfStar processes
  exactly. Twelve final no-copy views now drive layer 4 through FFN HC ingress,
  learned norm, biased top-6 routing, routed/shared experts, and the additive
  final HC update. Every retained FFN boundary and the complete final HC
  identity match two fresh DwarfStar processes exactly. That final HC now feeds
  layer 5's HC attention ingress, learned norm, Q-A/KV projections, fused Q/KV
  normalization, Q-B, compressed RoPE, and FP8 KV finalization without a host
  activation upload. All ten layer-5 final-tile boundaries match four fresh
  DwarfStar processes exactly. That retained state now feeds layer 5's paired
  F16 compressor projections and exact full-batch ratio-128 schedule. All 16
  compressed rows and both final 128x512 recurrent-state tensors match two
  fresh DwarfStar processes bit-for-bit. Three additional no-copy views now
  drive layer 5's dense mixed attention over 2,048 raw plus 16 compressed rows,
  inverse RoPE, both Q8 output projections, and the additive attention HC post.
  The complete 2048x4096 attention output, diagnostic rows, final HC tile, and
  full HC identity match two fresh DwarfStar processes exactly. Twelve final
  no-copy views now carry that state through layer 5's FFN HC ingress, learned
  norm, biased top-6 routing, routed/shared experts, and additive final HC
  update. Every retained FFN boundary and the complete final HC identity match
  two fresh DwarfStar processes exactly. That retained final state now feeds
  layer 6's HC attention ingress, learned norm, Q-A/KV projections, fused Q/KV
  normalization, Q-B, compressed RoPE, and FP8 KV finalization. All ten new
  boundaries match four fresh DwarfStar captures exactly. The command preserves
  121/121 no-copy mappings across 236 dispatches after continuing through layer
  6's paired ratio-4 attention/indexer compressors. All 512 compressed rows and
  four final recurrent-state tensors match two fresh DwarfStar captures
  bit-for-bit. Three additional no-copy views now drive layer 6's dense mixed
  attention over 2,048 raw plus 512 compressed rows, inverse RoPE, both Q8
  output projections, and the additive attention HC post. The complete
  2048x4096 attention output, diagnostic rows, final HC tile, and full HC
  identity match two fresh DwarfStar processes exactly, establishing complete
  native layers 0–5 plus exact full-2K layer-6 attention HC post. Twelve final
  no-copy views now carry that state through layer 6's FFN HC ingress, learned
  norm, biased top-6 routing, routed/shared experts, and additive final HC
  update. Every retained FFN boundary and the complete final HC identity match
  two fresh DwarfStar processes exactly, establishing complete native layers
  0–6 at the 2K prompt boundary. The retained final state now continues through
  layer 7's complete Q/KV state, ratio-128 compressor, dense mixed attention,
  biased top-6 routed/shared FFN, and additive final HC update, then directly
  through layer 8's complete Q/KV state, paired ratio-4 attention/indexer
  compressors, dense mixed attention, routed/shared FFN, and final HC update,
  then through layer 9's complete ratio-128 path and layer 10's complete paired
  ratio-4 path, through layer 11's complete ratio-128 path, and then through
  layer 12's complete paired ratio-4 path, through layer 13's complete
  ratio-128 path, then through layer 14's complete paired ratio-4 path, and
  through layer 15's complete ratio-128 path, and then through layer 16's
  complete paired ratio-4 attention/indexer path, and then through layer 17's
  complete ratio-128 path, and then through layer 18's complete paired ratio-4
  attention/indexer path, and then through layer 19's complete ratio-128 path,
  then through layer 20's complete paired ratio-4 attention/indexer path, and
  then through layer 21's complete ratio-128 path, and then through layer 22's
  complete paired ratio-4 attention/indexer path, and then through layer 23's
  complete ratio-128 path, and then through layer 24's complete paired ratio-4
  attention/indexer path, then through layer 25's complete ratio-128 path,
  through layer 26's complete paired ratio-4 attention/indexer path, and then
  through layer 27's complete ratio-128 path, and then through layer 28's
  complete paired ratio-4 attention/indexer path, and then through layer 29's
  complete ratio-128 path, and then through layer 30's complete paired ratio-4
  attention/indexer path, then through layer 31's complete ratio-128 path, and
  then through layer 32's complete paired ratio-4 attention/indexer path, and
  then through layer 33's complete ratio-128 path, and then through layer 34's
  complete paired ratio-4 attention/indexer path, and then through layer 35's
  complete ratio-128 path, and then through layer 36's complete paired ratio-4
  attention/indexer path, then through layer 37's complete ratio-128 path, and
  then through layer 38's complete paired ratio-4 attention/indexer path, and
  then through layer 39's complete ratio-128 path, and then through layer 40's
  complete paired ratio-4 attention/indexer path, and then through layer 41's
  complete ratio-128 path, and then through layer 42's complete paired ratio-4
  attention/indexer path.
  All retained layer-7 through layer-42 boundaries, full
  attention outputs, compressor states, and full HC
  identities match fresh DwarfStar processes exactly, establishing complete
  native transformer layers 0–42 at the same prompt boundary with 1,216/1,216
  no-copy mappings across 2,372 terminal dispatches. The same persistent
  context now consumes the retained final layer-42 HC row through the exact
  five-dispatch output head. Its five additional model mappings preserve
  pointer identity, all 129,280 logits match the independently repeated
  DwarfStar batched-prefill frontier bit-for-bit, and lowest-ID argmax selects
  token 15342. This establishes complete native batched model prefill through
  logits with 2,377 dispatches and 1,221 no-copy mappings. A separate
  layer-2 position-2051 diagnostic now covers the complete ratio-4 sparse
  mechanism: F16 indexer projections, compressed RoPE, indexer QAT, direct
  scores, exact descending top-512 selection, the 12-way indexed mixed
  attention/reduction, and inverse RoPE. Twelve retained tensors from two fresh
  DwarfStar processes match bit-for-bit, and all three model spans preserve
  mmap pointer identity. The capture explicitly overrides the sparse threshold
  to 512; source inspection corrected the prior assumption that the default
  switched immediately after 512 rows. Pinned DwarfStar remains dense through
  1,024 compressed rows and first switches at 1,025. Two additional fresh
  production-default captures at position 4099 now match on every sparse
  boundary, including the two-block argsort merge required by 1,025 rows. The
  same schedule is wired into retained even-layer state, including same-step
  attention/indexer row commit and 35 no-copy mappings. A retained-state C0
  control now seeds the exact layer-2 state immediately before position 4099,
  executes the general retained layer path through its 1,025th compressed row,
  and matches 16 sparse-boundary tensors by bit pattern across 54 dispatches.
  It intentionally stops its correctness claim before the token-dependent FFN
  and does not claim preceding-layer execution. The retained scheduler now
  generalizes the same exact top-k algorithm beyond that first boundary. Its
  scratch allocation is fixed from context capacity, while each step derives
  the initial sort blocks, active work width, ping-pong offsets, merge count,
  and final top-512 dispatch from visible rows. A second seeded layer-2 control
  at position 8195 commits compressed row 2,049, runs three initial sort blocks
  and two merge passes over a 1,025-index work width, and matches the same 16
  boundaries across 55 dispatches with 35/35 mappings. A complete retained
  position-8195 decoder step now executes all 43 layers through the repeated
  sparse-merge branch, matches all 129,280 logits exactly, and selects token
  35597 across 1,813 transformer dispatches with 1,370/1,370 pointer matches.
  Full native model prefill and the eligible engine-measurement producer remain
  pending.
- Measurements: The exact complete native layers-0–35 full-2K command reported
  31809.591 ms wall / 31050.835 ms GPU in its focused correctness run across
  1,951 dispatches with 1,004/1,004 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 25011.323 ms wall / 24213.220 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–34 full-2K command reported
  14317.612 ms wall / 13546.455 ms GPU in its focused correctness run across
  1,904 dispatches with 976/976 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 19871.581 ms wall / 19730.470 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–33 full-2K command reported
  13126.804 ms wall / 12632.873 ms GPU in its focused correctness run across
  1,834 dispatches with 944/944 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 15919.880 ms wall / 15796.966 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–32 full-2K command reported
  14678.233 ms wall / 14571.577 ms GPU in its focused correctness run across
  1,787 dispatches with 916/916 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 13283.525 ms wall / 13179.070 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–31 full-2K command reported
  16492.572 ms wall / 15726.058 ms GPU in its focused correctness run across
  1,717 dispatches with 884/884 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim.
  The complete target-Mac gate independently repeated the same exact boundary
  at 9087.549 ms wall / 8984.633 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–30 full-2K command reported
  12541.042 ms wall / 12450.093 ms GPU in its focused correctness run across
  1,670 dispatches with 856/856 no-copy model mappings. This includes
  exhaustive correctness readback and is not a throughput claim. The complete
  target-Mac gate independently repeated the same exact boundary at
  8843.842 ms wall / 8740.028 ms GPU with identical dispatch and mapping
  counts.
  The prior exact complete native layers-0–29 full-2K command reported
  13595.215 ms wall / 13486.902 ms GPU focused and 8502.094 ms wall /
  8380.564 ms GPU in the complete target-Mac gate across 1,600 dispatches with
  824/824 no-copy model mappings.
  The isolated sparse indexed-attention diagnostic reported 18.864 ms wall /
  0.497625 ms GPU across 10 dispatches with 3/3 no-copy model mappings. Its wall
  interval includes command setup, synchronization, and exhaustive readback;
  neither value is a throughput claim.
  The production-default 1,025-row layer segment reported 25.162 ms wall /
  0.550000 ms GPU across 11 dispatches with 3/3 no-copy model mappings. The
  command also reran the 513-row diagnostic first. This is correctness evidence,
  not throughput. The complete target-Mac gate repeated the default boundary at
  22.303 ms wall / 0.547833 ms GPU with the same schedule and mapping counts.
  The focused retained-state boundary control reported 53.413 ms wall /
  29.894125 ms GPU across 54 dispatches with 35/35 pointer matches. It includes
  seed upload, synchronization, and exhaustive correctness readback and is not
  a throughput claim. The complete target-Mac gate repeated it at 58.511 ms
  wall / 30.369250 ms GPU with the same exact tensor, dispatch, and mapping
  counts.
  The focused retained multimerge control reported 61.399 ms wall /
  32.045250 ms GPU across 55 dispatches with 35/35 pointer matches. It includes
  seed upload, synchronization, and exhaustive correctness readback and is not
  a throughput claim. The complete target-Mac gate repeated it at 49.455 ms
  wall / 28.882250 ms GPU with identical tensor, schedule, and mapping counts.
  The complete target-Mac gate reported 2289.152 ms wall / 2161.306708 ms GPU
  with the same schedule and mapping counts. These intervals include exhaustive
  correctness readback and are not throughput claims. The prior exact complete
  native layers-0/1/2/3/4/5/6/7 full-2K command reported 1950.645 ms wall /
  1864.381 ms GPU in its focused
  correctness run, across 313 dispatches with 164/164 no-copy model mappings.
  Its complete target-Mac gate reported 1915.398 ms wall / 1819.682 ms GPU with
  the same schedule and mapping counts. These intervals include exhaustive
  correctness readback and are not throughput claims. The prior exact complete
  native layers-0/1/2/3/4/5/6 full-2K command reported 1709.387 ms wall /
  1620.434 ms GPU in its focused correctness run, across 266 dispatches with
  136/136 no-copy model mappings; its complete gate reported 1636.955 ms wall /
  1548.723 ms GPU.
  The prior exact complete native layers-0/1/2/3/4/5 plus layer-6
  dense-attention full-2K command reported 1545.043 ms wall / 1464.156 ms GPU
  in its focused correctness run, across 245 dispatches with 124/124 no-copy
  model mappings. The complete target-Mac gate reported 1635.054 ms wall /
  1454.353 ms GPU with the same schedule and mapping counts.
  The prior exact complete native layers-0/1/2/3/4/5 full-2K command
  reported 1358.984 ms wall / 1270.350 ms GPU in its focused correctness run
  and 1363.509 ms wall / 1272.876 ms GPU in the complete target-Mac gate,
  across 196 dispatches with 104/104 no-copy model mappings. The prior exact
  complete native layers-0/1/2/3/4 plus layer-5 dense
  mixed-attention full-2K command reported 1242.696 ms wall / 1152.157 ms GPU
  in its focused correctness run and 1253.568 ms wall / 1159.442 ms GPU in the
  complete target-Mac gate, across 175 dispatches with 92/92 no-copy model
  mappings. The prior layer-5 compressor checkpoint reported 1076.840 ms
  wall / 991.235 ms GPU focused and 1072.662 ms wall / 972.161 ms GPU in its
  complete gate, across 166 dispatches with 89/89 mappings. The prior layer-5
  Q/KV checkpoint reported 1085.161 ms wall / 994.834 ms GPU across 159
  dispatches and 85/85 mappings. The prior exact
  complete native layers-0/1/2/3/4 full-2K command
  reported 1079.281 ms wall / 991.837 ms GPU in its focused run and 1601.151 ms
  wall / 985.798 ms GPU in the complete target-Mac gate, across 149 dispatches
  with 76/76 no-copy model mappings. The prior attention-only checkpoint
  reported 934.193 ms wall / 833.479 ms GPU focused and 952.654 ms wall /
  851.603 ms GPU in the complete gate, across 128 dispatches with 64/64
  mappings. The prior paired-compressor checkpoint reported 740.517 ms wall /
  656.454 ms GPU focused and 750.548 ms wall / 655.250 ms GPU in the complete
  gate, across 119 dispatches with 61/61 mappings.
  The prior Q/KV-only checkpoint reported 1169.303 ms wall / 647.777 ms GPU in
  its focused run and 947.998 ms wall / 651.830 ms GPU in the complete
  target-Mac gate, across 89 dispatches with 53/53 mappings. The prior complete-layer-3
  checkpoint reported 675.511 ms wall / 592.734 ms GPU focused and 759.645 ms
  wall / 671.882 ms GPU in its complete gate. The layer-3 attention-only checkpoint
  reported 583.424 ms wall / 501.227 ms GPU focused and 569.588 ms wall /
  490.821 ms GPU in its complete gate. The prior ratio-128
  compressor checkpoint reported 494.638 ms wall / 396.686 ms GPU focused and
  483.530 ms wall / 399.101 ms in the complete target-Mac gate. The prior Q/KV
  command reported 482.230 ms wall / 402.931 ms GPU in its focused run and
  447.543 ms wall / 366.771 ms GPU in the complete target-Mac gate, across 42
  dispatches with 25/25 no-copy model mappings. These include full correctness
  readback and are not throughput claims. The prior layer-3-ingress control was 441.440 ms wall
  / 371.934 ms GPU, the layer-2-only control was 408.397 ms wall / 338.484 ms
  GPU, and the attention-only control was 196.069 ms wall / 169.211 ms GPU.
  Metal batching was 42.861x faster than synchronized submission
  in the retained M-002 probe. DwarfStar medians are 164.86 prefill / 19.90
  generation tok/s at 2K and 161.05 prefill / 17.36 generation tok/s at 32K.
  The first real-model kernel matched all 20,480 checked FP32 values; its final
  validation dispatch reported 0.020 ms GPU time and is not an inference-speed
  claim. The Q8_0 projection matched its 1,024-value decode fixture and reported
  0.031 ms GPU time in its completion gate, also not a throughput claim. The
  first native M1 batch boundary matched 131,072 Q-A outputs plus its 1,024-value
  sequential control; the complete gate reported 2.490 ms wall / 0.410 ms GPU
  for the isolated 128-row batch dispatch. All 1,024 final-row values differ
  between the native batch and sequential schedules with maximum absolute error
  `3.62396240234375e-05`; this is localization evidence, not prefill throughput.
  The extended final-32-row Q/KV setup matched all 2,195,456 produced FP32
  values; the complete gate reported 12.023 ms wall / 2.817 ms GPU for its
  five-dispatch isolated layer segment. This is correctness evidence, not
  prefill throughput. The first focused continuous token-to-Qcur run matched
  all 2,457,600 produced FP32 values, retained 10/10 mmap pointer identities,
  and reported 56.433 ms wall / 2.544 ms GPU. Its cold standalone wall time
  includes pipeline compilation and readback and is not a throughput claim. The
  complete gate repeated C0 exact at 55.079 ms wall / 2.561 ms GPU.
  The complete layer-0 final-tile boundary was C0 exact in its focused v5 run
  at 135.893 ms wall / 73.494 ms GPU with 25/25 no-copy model views. The full
  gate repeated it at 138.441 ms wall / 81.329 ms GPU. Both intervals include
  setup, synchronization, exhaustive readback, and comparison and are not
  prefill-throughput claims.
  The first exact layers-0/1 handoff run reported 125.125 ms wall / 69.515 ms
  GPU with 30/30 no-copy views; the full gate repeated it at 129.903 ms wall /
  78.876 ms GPU. These intervals have the same correctness-oriented scope and
  are not throughput measurements.
  The connected six-dispatch layer segment matched 9,264 retained FP32 values and
  reported 0.102 ms GPU time in the final gate; this remains correctness
  evidence rather than decoder throughput. The nine-dispatch Q/K projection
  setup matched another 34,816 FP32 values and reported 0.473 ms GPU time in
  its final gate, also not a throughput claim. The twelve-dispatch RoPE/cache
  gate matched 34,816 direct outputs plus the 512-value stored row, preserved
  two 512-value guard rows, and reported 0.949 ms GPU time; this is likewise
  correctness evidence rather than decoder throughput. Persistent layer-0
  execution measured 1.503 ms wall median (0.094 ms MAD) and 1.138 ms GPU
  median (0.033 ms MAD) across 30 bit-identical samples. The first continuous
  layers-0/1 correctness run reported 75.956/31.655 ms wall/GPU for cold layer
  0 and 27.276/26.313 ms for layer 1; those two-layer timings include setup and
  correctness readback and are not decoder-throughput claims. The canonical
  layers-0/1/2 gate reported 47.226/20.865 ms, 16.310/15.398 ms, and
  16.401/15.662 ms wall/GPU respectively; these intervals likewise include
  synchronization and correctness readback. Five alternating synchronized and
  chained samples measured 82.330 ms versus 80.175 ms median wall time, a
  1.0269x ratio or 2.62% reduction. Median summed command GPU time was 53.488 ms
  versus 52.458 ms, so the scheduler result remains a narrow diagnostic rather
  than a decoder-throughput claim. Over the four-layer slice, five alternating
  pairs measured 98.371 ms synchronized versus 97.216 ms chained median wall
  time, a 1.0119x ratio or 1.17% reduction. Median summed GPU intervals were
  68.128 and 67.986 ms; this is again a scheduler diagnostic, not a
  decoder-throughput claim. The prepared fixed-position four-layer replay
  measured 4.318 ms wall median (0.180 ms MAD) and 3.767 ms summed-GPU median
  (0.131 ms MAD) across 20 samples after five warmups, then passed one complete
  four-layer C0 collection. It is a steady-state execution microbenchmark, not
  token throughput. The first position-advancing run reported 103.106 ms wall
  and 70.862 ms summed GPU for its cold position-1 step, then 7.333 ms wall and
  6.347 ms summed GPU for position 2. Both steps were C0 exact across all four
  layers and cache histories; these remain correctness diagnostics rather than
  model-throughput measurements. The first exact position-3 compressor run
  reported 110.896/77.239 ms at the cold position-1 step, 8.863/8.123 ms at
  position 2, and 10.455/9.618 ms at position 3 (wall/summed GPU). It matched
  the emitted 512-value FP8 compressed KV row and every downstream layer
  boundary; this is also correctness evidence, not token throughput.
  The integrated six-layer gate reported 143.285/108.618 ms at the cold
  position-1 step, 14.098/13.178 ms at position 2, and 11.552/10.111 ms at
  position 3 (wall/summed GPU). It matched both ratio-4 emissions and every
  retained layer 0–5 boundary. These are correctness diagnostics, not token
  throughput measurements. In the final integrated gate, the bounded
  position-127 ratio-128 replay reported 32.461/8.236 ms wall/GPU for layer 3
  and 6.881/6.273 ms for layer 5. Both first emissions were C0 exact; setup,
  synchronization, and exhaustive comparison remain in scope, so these values
  are correctness diagnostics rather than inference-speed measurements. The
  integrated eight-layer gate reported 223.448/190.007 ms at its cold
  position-1 step, 16.989/16.442 ms at position 2, and 17.782/17.160 ms at
  position 3 (wall/summed GPU). It matched every retained layer 0–7 boundary
  and all three ratio-4 emissions. These values are also correctness
  diagnostics, not token-throughput measurements. The complete 43-layer gate
  reported 837.045/800.215 ms at position 1, 62.799/61.229 ms at position 2,
  and 53.374/49.691 ms at position 3 (wall/summed GPU). It matched every
  retained layer 0–42 boundary and all even-layer ratio-4 emissions through
  layer 42. These intervals also include correctness-oriented work and are not
  token-throughput measurements. The final output-head correctness gate
  reported 15.025/2.946 ms wall/GPU at cold position 1, 1.679/1.355 ms at
  position 2, and 1.691/1.359 ms at position 3. All 129,280 logits and four
  preceding output tensors were C0 exact at every step; these timings include
  synchronization/readback and are likewise not throughput claims. The first
  full-gate closed-loop diagnostic measured 54.610, 51.727, and 53.989 ms for
  its three committed steps after an exhaustive C0 preparation pass: 18.712
  tok/s complete and 18.919 tok/s over the final two steps. It includes command
  encoding, synchronized Metal execution, full logits transfer, and CPU argmax,
  but remains ineligible for a paired claim because it starts from captured
  state and generates only three tokens.
  The integrated position-127 diagnostic evaluates 127 closed-loop positions,
  reproduces the 128-token transcript, and reported 16.493 evaluated
  positions/s in the complete gate. Correctness readback occurs after timing.
  This is not paired throughput because the initial cache and compressor state
  are captured rather than produced by cold prefill.
  The first exact cold-state run paid model-residency costs and reported
  3379.018 ms for one-token prefill/first selection plus 0.729 evaluated decode
  positions/s.
  An immediate fresh-process repeat with resident weights reported 1130.755 ms
  prefill and 18.246 evaluated positions/s; the complete gate later reported
  1078.665 ms and 18.311 positions/s. All were bit-identical. None is a paired
  claim because one-token prefill is not a protocol frontier and the residency
  conditions differ. The first 2K sequential state run reported 18.603
  tokens/s over 110090.070 ms and was exact against the repeated DwarfStar
  decode replay. It is not prefill throughput: DwarfStar's native batched
  prefill is the required oracle and produced a different full-logit tensor.
  The first exact paired-compressor full-2K run reported 3624.704 ms summed
  wall and 3514.757 ms summed GPU time across its 64 correctness-oriented
  tiles. It retained 65/65 model mappings, used 118 dispatches on regular tiles
  and 122 on the final state refresh, and is not a prefill-throughput claim.
  The first complete-model native 2K prefill run reported 21478.015/21389.015
  ms wall/GPU for the terminal 43-layer transformer schedule and
  270.931/2.697 ms for its synchronized output-head correctness pass. Every
  one of the 129,280 batched-prefill logits was C0 exact and token 15342 was
  selected. These remain correctness timings, not an eligible engine
  measurement.
  The first exact native-prefill handoff used one 229-copy GPU blit command
  across all 43 layers, then matched all 2,052 post-prompt greedy selections
  through position 4099. Its first and final full-logit tensors were C0 exact,
  including all 21 even layers' first production sparse step at compressed row
  1,025. The correctness-oriented run reported 78.904 ms handoff wall time and
  16.563 synchronized positions/s; neither is a throughput claim.
- Manual handoff: `RUST_STAR_MANUAL_TASKS.md` records Actions approval, target
  compilation/model inspection, quick/extended oracle capture, and the deferred
  secure-access decision with exact evidence requirements.
- Parallel research: DSpark remains separate on
  `origin/codex/dspark-observability-0` and is not on this phase's critical path.
- Publication: the branch has published history, but this checkout already
  had five local commits at task start. This task adds a local implementation
  checkpoint; no push was performed. Prefer the connected GitHub app for
  future authorized remote writes from this environment.

## Immediate Next Actions

1. The explicit scratch-reuse milestone is complete. Preserve the verified
   report, original sentinel assertion and all raw evidence. All accepted source/default restored.
2. Report rejected, full C0/pool coverage and engine/memory results.
3. Next proposal, not started: Diagnose scratch RSS residency and prove which large buffers can safely avoid unconditional clearing before another 32K candidate.
4. STOP after the update. Keep oracle-v5 immutable,256K paused and publication
   deferred; do not begin the next milestone automatically.

## Entries

### 2026-08-30 — Strengthened the residual 32K compressor dependency

- Began the planned Rust Star 32K engine extension by capturing the missing
  32K-to-32,896 closed-loop oracle. All eight audited helper captures agree
  byte-for-byte on all 129 greedy input tokens with SHA-256
  `a9422bb7...0ba9`, prefill token 85166, and final token 716.
- The longer audit invalidated the stronger claim that producer `b81c099` was
  fully repeatable. Of three new fresh processes, two reproduced accepted v2
  post-prefill SHA-256 `603430b4...09c547`, first-step SHA-256
  `63478a55...2b5bb`, and final-step SHA-256 `36d2c349...177b0`; one produced
  different full tensors while preserving every selected token. At prefill it
  differed in 129,219/129,280 logits, with maximum absolute error
  `0.00280476` and RMSE `0.000501144`.
- A layer-17-only command-buffer boundary was rejected: it produced another
  distinct FP32 result even though the earlier forensic run first observed the
  corrupted retained row at layer 17. This proves that the safe fix cannot be
  scoped to the first visible divergence without additional dependency
  evidence.
- Strengthened every compressor projection-to-read edge from an encoder-only
  boundary to an ordered command-buffer boundary. This adds no host wait and
  changes no arithmetic. Two fresh 32K-to-32,896 runs from executable SHA-256
  `6b99a58c...bee0b` are byte-identical for the full transcript and all three
  captured logit tensors, and exactly reproduce the accepted v2 hashes above.
  The same executable preserves the existing 2K batched-prefill and first-step
  fixtures bit-for-bit.
- The full model-backed `ds4_test` suite passed, including the 30,474-token
  long-context path, official logprob vectors, local golden vectors, Metal
  kernel exactness, tensor equivalence, and the SSD-streaming cache-pressure
  repro. Optional streaming decode-prefill, MTP, and DSpark suites remained
  explicitly skipped by their existing opt-in gates.

Decision:

- Keep the accepted v2 tensor values as the current C0 target, but revoke the
  assumption that its pinned producer is reliably repeatable. Do not import the
  drifting continuation logits or promote a Rust 32K result against an
  unreproducible source. Preserve the bit-stable 129-token transcript as local
  evidence until the stronger producer is versioned and accepted.

Next:

- Freeze the command-buffer repair, capture and independently verify a new
  versioned 2K/32K oracle from that immutable checkpoint, then import the exact
  32K continuation fixture and begin Rust Star's chunked-prefill boundary.

### 2026-08-30 — Accepted repeatable oracle-v2 at 2K and 32K

- Added the `oracle-v2` capture profile around immutable repaired producer
  `b81c099b1f7888358fcdc820e7e70566c04aafae` / tree
  `4b913890...07400`. The format retains the v1 manifest schema but v2 requires
  correctness evidence, coverage at 2K and 32K, and at least two fresh-process
  conformance repetitions per frontier. The capture fails immediately on a
  repeated JSON tensor hash mismatch; the independent verifier compares every
  FP32 bit and required C0 metadata field again.
- Froze capture kit `855b2b8c4ef8dd9400f4783392fce1c7323bc01d` /
  tree `1b65617c...3b716`. All 78 Python artifact/benchmark tests passed,
  including deliberate v2 drift and missing-repetition rejection. The existing
  `oracle-v1` archive independently continued to verify unchanged.
- Captured privacy-filtered bundle
  `rust-star/results/oracle-v2-20260830T091839Z.tar.gz` on the target M1 Ultra.
  Archive SHA-256 is
  `538332a60b82b264cbdde3c4ba07d8d307ae974627a0ab2550e84c1c52a4f2de`.
  Independent archive verification passed with eight hashed artifacts and
  6,325,632 verified bytes. Performance was intentionally disabled because the
  host was noisy; correctness acceptance is independent of benchmark
  eligibility.
- Both 2K captures are byte-identical with JSON SHA-256
  `2b3cdf9b...6ddb0`, argmax 15342, and the historical v1 2K tensor. Both 32K
  captures are byte-identical with JSON SHA-256 `269f9529...58e12` and argmax
  85166. Metal-kernel and official-logprob-vector correctness suites passed
  inside the isolated source build.
- Compared old and proposed oracle tensors explicitly. At 2K, all 129,280
  logits and required metadata remain C0. At 32K, all 129,280 historical v1
  logits differ from deterministic v2, with maximum absolute error
  `1.00521278`, RMSE `0.16934920`, cosine similarity `0.99955505`, and
  `KL(P||Q)=0.00359783`; argmax remains 85166, top-10 overlap is 9/10, and
  top-50 overlap is 46/50. This is a numerical version boundary caused by the
  repaired dependency hazard, not a behavioral token change.
- The prior ABBA characterization measured a `+0.084%` repaired/original
  median wall delta under the same noisy-host label, ruling out an obvious
  speed regression without promoting it to benchmark evidence. Historical v1
  measurements remain preserved under v1.

Decision:

- Advance `current-oracle` to `oracle-v2` for 2K and 32K. Preserve v1's exact
  2K evidence and all historical performance results; keep only v1's unstable
  32K tensor quarantined. Rust Star may now use the v2 32K tensor for C0 work.

Next:

- Extend the eligible Rust Star engine and closed-loop exactness gate to 32K
  against `oracle-v2`, then measure that frontier only after it passes C0.

### 2026-08-30 — Quarantined the unstable 32K oracle and repaired its producer

- Generalized `capture_prefill_decode_frontier.c` to derive the prompt length
  from an exact little-endian U32 token file, optionally capture the direct
  post-prefill logits, and omit the multi-hundred-megabyte layer payload when
  it is not needed. Its original 2K invocation remains compatible.
- Reconstructed the first 32,768 token IDs from the existing 419,509-token
  capture. The first 2,048-token prefix is byte-identical to the pinned 2K
  fixture. Two fresh 32K-to-32,896 runs had an identical 129-token greedy input
  transcript but different first and final full-vocabulary logits. Two further
  exact-capacity 32K runs captured the direct post-prefill row and differed in
  129,278 of 129,280 FP32 logits with maximum absolute difference `0.383039`;
  both still selected token 85166. The current OS, Xcode, compiler, model, and
  pinned source identity match the original manifest, whose capture contained
  only one 32K conformance repetition.
- Localized the first serialized-state difference to 15 FP32 values in layer
  17's ratio-128 compressed row 191, emitted at the end of the
  positions-20,480--24,575 chunk. Independent repeated top-512 and indexer
  score-builder probes were bit-identical. Diagnostic dumps immediately around
  the layer-17 projection/compressor boundary made the raw KV projection, raw
  score projection, compressed output, post-prefill logits, and next-step
  logits bit-identical across fresh processes.
- Added a minimal in-command-buffer dependency boundary after the compressor
  KV/score projections. It ends only the retained compute encoder; it does not
  submit, wait on the host, or change arithmetic. Four ordinary fresh-process
  32K runs then matched exactly at post-prefill and next-step logits, selecting
  token 85166. The repaired post-prefill SHA-256 is
  `603430b4...09c547`, and the next-step SHA-256 is
  `63478a55...2b5bb`. Their deterministic output differs from the historical
  August 32K tensor, so the repair does not silently replace or retroactively
  validate `oracle-v1`.
- Froze the repaired producer at immutable commit `b81c099`. The generalized
  original executable used for the ABBA control has SHA-256
  `b817ed0a...f7df7`; the repaired executable has SHA-256
  `8e5f7e29...fcfcf1`.
- Re-ran the exact 2K boundary with the repair. Direct post-prefill logits and
  first decode logits are byte-identical to the pinned fixtures, and the first
  generated token remains 15342. The 2K C0 contract is unchanged.
- Ran a host-contaminated ABBA wall-time characterization using two immutable
  executables and identical 32K input/output work. The original measured
  `185.05` and `185.23` seconds; the repair measured `187.69` and `182.90`
  seconds. Their medians were `185.140` and `185.295` seconds respectively, a
  `+0.084%` repaired/original delta. This is not headline performance evidence,
  but it rules out an obvious regression at this gate. Both original outputs
  differed from one another while both repaired outputs remained byte-exact.

Decision:

- Preserve the historical 32K artifact but quarantine it as a C0 target.
  Retain the dependency repair as the deterministic candidate and require an
  explicit oracle version before Rust Star uses the repaired 32K tensor for C0
  claims. The bounded characterization found no material performance cost;
  normal paired protocol evidence remains separate.

Next:

- Construct and verify the versioned replacement oracle metadata/artifacts from
  immutable checkpoint `b81c099`, then extend Rust Star's exact context gate
  against it.

### 2026-08-30 — Accepted retained lifecycle after three exact repetitions

- Froze commit `cb3555e`, executable SHA-256
  `930677c0c079b80648cd72e72ae5fa621359c579acf197ce78487b0bbc327371`,
  context 2,048, generation 128, and three repetitions before execution. Each
  repetition ran the fully charged first cycle and reset retained cycle in one
  process/context and wrote independent
  `rust-star-retained-engine-run-v1` evidence under
  `rust-star/.work/retained-engine-control-cb3555e/`.
- The intended quiet-host gate was not met: macOS `mediaanalysisd` remained at
  approximately 100--160% CPU before, during, and after the set. The process
  was sampled but not suspended. All repetitions are therefore labeled
  host-contaminated lifecycle characterization, never paired benchmark evidence.
- All six complete 128-token transcripts matched checksum
  `17615242442502606640`. Every record required request-state reset, fresh
  residency on cycle one, reused residency on cycle two, and
  `paired_protocol_eligible: false`.
- First-cycle prefill measured 174.114, 175.188, and 175.632 tok/s. Retained
  prefill measured 186.695, 187.463, and 187.541 tok/s, for a 187.463 median,
  0.45% full range, and 1.070068x median within-run ratio. Median complete
  first-cycle-to-retained prompt ratio was 2.195006x. Median avoided host page
  warm was 12,289.730 ms and median avoided residency request/touch work was
  566.659 ms. Retained generation measured 23.376--23.616 tok/s.
- Accepted the lifecycle architecture for a future long-lived engine based on
  exact reset/reuse correctness and repeated low-variance behavior. Immutable
  pipelines, no-copy model views, and the attached residency set may survive a
  request; activation, KV/compressor, command-chain, and profiler state may not.
  First-load costs remain fully charged and separately reported. This decision
  does not promote the same-process timings or change the paired protocol.
- Evidence SHA-256 values: repetition 1
  `290fd5802d7e6d0fb7c0d18d7f3f979ef6e8621ca902d5a95b5efb2f80f41b3e`,
  repetition 2
  `4f135fdd9ba125de6be0078b7e7d500dc2cd1d599f3d77f0682fac16c4652452`,
  and repetition 3
  `432e2a523f155bdd105ce17d955de00e84468b7cfee1bd6d4442858495319060`.

Next:

- Return to the next exact context frontier. A later quiet-host retained set is
  useful for cleaner characterization but is no longer a correctness or
  architectural blocker.

### 2026-08-30 — Proved an exact retained Metal engine lifecycle

- Added `engine-retained-measure` and stable
  `rust-star-retained-engine-run-v1` evidence. The command runs two exact
  context-2048/generation-128 cycles in one process and one Metal context. The
  first cycle fully charges model-page warming, context creation, residency-set
  creation/request, and the coarse GPU view touch. The second retains immutable
  pipelines, no-copy model views, and the attached residency set while
  rebuilding the prompt and decoder state.
- Made native residency preparation idempotent and explicit: a reused call
  reports the same 1,136 views/allocations and queue attachment with zero
  residency/touch GPU interval. Normal `engine-measure` rejects such reuse;
  the retained schema requires it only on cycle two and is always marked
  paired-ineligible.
- The first live two-cycle attempt correctly failed closed on cycle two:
  prefill selected token 8954 instead of oracle token 15342. The value was the
  previous request's final token, exposing retained mutable activation state.
  Added a narrow request reset that drops activation/KV/compressor buffers,
  prefill properties, command chains, and profiler state while preserving the
  context's immutable pipeline/model/residency ownership. The failed attempt
  remains under `rust-star/.work/retained-engine-lifecycle-01/`.
- Two corrected runs matched both complete 128-token oracle transcripts and
  checksum `17615242442502606640`. The final current-source run under
  `rust-star/.work/retained-engine-lifecycle-04/` measured a 23,296.775 ms
  first-load prompt: 11,425.564 ms model-page warming plus 11,871.211 ms
  prefill (172.518 tok/s). Retained prompt was 11,037.192 ms (185.554 tok/s),
  a 1.075564x prefill ratio and 2.110752x prompt ratio. It avoided 640.124 ms
  of residency host/touch work; generation remained effectively stable at
  23.740 versus 24.046 tok/s. The preceding corrected run measured 186.311
  retained prefill tok/s and the same exact transcript.
- An adjacent final normal `engine-measure` control remained exact and
  paired-eligible at 173.271 prefill tok/s and 23.734 steady decode tok/s. This
  proves the new
  diagnostic did not relax or redirect the fresh-process measurement path.
- Spotlight's `mds_stores` was near 96% CPU during the first corrected evidence
  run and subsided before the final repetition. These same-process timings are
  not promoted as paired benchmark evidence; they are sufficient to establish
  exact state reset, retained residency ownership, reproducibility, and honest
  first-load/retained accounting.
- Validation: 300 optimized Rust tests, all 75 Python tests, release build,
  Rust formatting, JSON parsing, and `git diff --check` passed. The current
  executable SHA-256 is
  `930677c0c079b80648cd72e72ae5fa621359c579acf197ce78487b0bbc327371`.

Next:

- Repeat a small retained control set once Spotlight/indexing is quiet, then
  decide whether the server lifecycle should adopt this reset boundary. The
  next performance frontier remains longer-context eligibility; do not alter
  the paired fresh-process contract.

### 2026-08-29 — Localized prefill host time and rejected residency shortcuts

- Collected three fresh exact eligible 2K/128 controls after the competing
  Chromium build ended. Median Rust Star prefill was 173.252 tok/s, with a
  172.402--174.332 range; median steady decode was 23.263 tok/s. Median setup
  attribution was 48.246 ms context, 10.559 ms bootstrap, 36.228 ms
  transformer, and 0.025 ms output head, totaling 95.057 ms or 0.80% of median
  prefill. These phases are too small to justify pipeline or resource caching
  as the next optimization.
- The first attempted control used a stale release executable that omitted the
  newly required setup fields. The engine preserved the exact transcript, but
  the adapter failed closed. Force-rebuilding produced executable SHA-256
  `4d23df4aa0c9adc4e9aa19ea00c4bfe1fc45e909bcad35c2599d65bc24e8eaf6`;
  the rejected evidence remains under
  `rust-star/.work/prefill-host-attribution-03/`.
- Added passive `prefill_handoff_host_ms`, `model_residency_host_ms`, and
  `prefill_unattributed_host_ms` fields. The Rust writer and Python adapter now
  require every host subfield to be finite, nonnegative, and to sum exactly to
  `prefill_host_overhead_ms` within one microsecond.
- The first exact attribution run from executable SHA-256
  `d3d196cb119c15e8d46a81e088231a891b8d63b42b6667a6a4b0a12f8631d971`
  measured 751.101 ms of 884.340 ms host overhead in residency preparation,
  15.646 ms around the decoder-state handoff, and only 0.176 ms unattributed.
  Its transcript checksum was `17615242442502606640`; the adapter accepted all
  1,136 model views, all 1,136 residency allocations, and queue attachment.
- Rejected three exact residency experiments. Bulk `addAllocations:count:`
  measured 794.046 ms residency host time versus 751.101 ms in the adjacent
  scalar control. Preparing/requesting residency while the transformer ran
  moved the cost into an 830.286 ms output-head stall. Omitting
  `requestResidency` moved it into an 807.156 ms coarse-touch wall interval.
  None improved complete prefill, and all Objective-C candidates were reverted.
- The latter experiments ran while OBS/camera processes raised host load, so
  their throughput values are not performance claims. Their stage-local stalls
  are sufficient negative evidence because each moved the same approximately
  0.8-second residency cost to the next dependent operation.
- Validation: the complete 128-token oracle transcript passed in every live
  candidate; 297 optimized Rust tests and all 75 Python tests passed; Rust
  formatting, Python compilation, the new Rust under-attribution regression,
  and `git diff --check` passed. The restored production source was rebuilt as
  executable SHA-256
  `6d88cdd3b73440cd63858d65aab51f12dedb35960628c076573eb8e080836b0f`.

Next:

- Treat residency as a lifecycle question rather than another launch tweak:
  determine whether a retained engine can pay the cost once while still
  reporting first-load TTFT honestly, then return to paired evidence under a
  quiet host.

### 2026-08-29 — Made the blocked GitHub Actions workflow manual-only

- Reviewed all 134 published `Rust Star host contracts` runs through GitHub's
  API. Both jobs in every run failed before executing a step; their check-run
  annotations report that the owning account is locked over a billing issue.
  This is an account-level launch failure, not evidence of a Rust, Python, or
  Metal regression.
- Confirmed `agent/rust-star-bootstrap` is unprotected and has no required
  status checks.
- Removed the automatic `push` and `pull_request` triggers while preserving
  `workflow_dispatch` and both clean-host validation jobs. Future pushes will
  no longer produce meaningless red runs; the recipe remains available for a
  deliberate run after the account lock is resolved.
- Updated manual task M-001 to record the blocker and recovery procedure.
- No runtime build, model correctness test, or performance benchmark was run;
  the change affects CI orchestration and project records only.
- Ruby's YAML parser accepted the workflow and `git diff --check` passed.

### 2026-08-29 — Rejected 128-row bootstrap and added host setup attribution

- Tried the smallest 128-row production bootstrap: widened only the accepted
  tile validator/schedule and the two row-position stack fixtures. Aligned
  128-row tiles fill the physical raw-KV ring contiguously from offset zero, so
  no wrapped append was required for this bounded schedule.
- The first exact 2K/128 process rejected the candidate at the transcript gate:
  generated steps 1--5 matched, then step 6 selected token `716` instead of
  `77448`. This aligns with the first newly emitted ratio-4 compressor row
  after the prompt, so the experiment was not benchmarked or accepted. All
  128-row source changes were reverted; production remains 32 command buffers
  of 64 rows and diagnostics remain 64 synchronized 32-row tiles.
- A same-timestamp apply/revert left Cargo's cached Objective-C archive stale
  even though source had returned to 64 rows. The frozen `0cbffdf` executable
  passed its exact 128-token control, proving the model and host were healthy.
  Added `RUST_STAR_FORCE_NATIVE_REBUILD` as an explicit build-script
  cachebuster, force-recompiled the shim, and restored exact production
  execution. The rebuilt control selected prefill token `15342`, final token
  `8954`, and checksum `17615242442502606640`.
- Added passive eligible timing for Metal context creation, bootstrap
  setup/encoding, transformer setup/encoding, and output-head preparation. The
  adapter now requires every setup field to be nonnegative and finite and
  rejects their sum if it exceeds the existing residual host-overhead interval.
  Documentation now consistently describes the accepted 64-row schedule.
- The first force-rebuilt attribution run was correctness-valid but not usable
  as performance evidence: a separate Chromium `ninja` build drove load average
  to approximately 50. It measured 106.959 prefill tok/s, including 7,200.575
  ms bootstrap GPU time versus the established 2.29--2.48 seconds. Its passive
  setup split was 170.003 ms context, 97.365 ms bootstrap, 59.670 ms
  transformer, and 0.055 ms output head. Private output remains under
  `rust-star/.work/prefill-host-attribution-02/`.
- Validation: exact transcript control passed; 297 optimized Rust tests and 74
  Python tests passed; Rust formatting, Python compilation, and
  `git diff --check` passed.

Next:

- When the Chromium build is idle, collect three fresh exact attribution runs.
  Use their median setup split to choose between context/pipeline caching,
  bootstrap resource reuse, or transformer view/allocation reuse; do not infer
  a performance change from the contaminated control.

### 2026-08-29 — NSG=4 closes 18.40 points of the validated prefill gap

- Froze executable SHA-256
  `d453ba11c37fa0f9738b32491848a09953bbd8af51b3d413076c876db5c59a24`
  from immutable commit `0cbffdf59dfb84127ffc49ebf14a250d6253f7bd` and
  tree `8e32fa77c3d3f94ce91af155c411200d32be3c4f`.
- Ran predeclared plan SHA-256
  `6f3ed971728d155a27fa426c9152c7a74267d21c5710b808a8e01468181737cc`
  against pinned DwarfStar. Both warmups and all five C0 pairs passed; there
  were no failures, retries, blocked pairs, or retained invalid attempts.
- Rust Star median prefill was 163.873178865 tok/s versus DwarfStar's 182.37.
  The median within-pair ratio is 0.897865063x, leaving a 10.21% gap. Compared
  with `cdb6377`'s 0.713851042x ratio and 28.61% gap, this is a 25.78% relative
  ratio improvement and closes 18.40 percentage points of the gap.
- Rust Star steady decode measured 23.462039051 versus 23.43 tok/s, a
  1.001367437x paired ratio. Complete generation measured 23.405874030 versus
  23.07 tok/s, a 1.014558909x ratio. First-token latency was 55.307 versus
  48.724 ms, a 1.126438498x latency ratio.
- Final raw and summary SHA-256 values are
  `d5a851705a07518b5757b2176be569c807f65aaef90fc531cb488e37a03269fc`
  and `701b2b1f0c28b6842ec8ea4cf4966e081520420d8cfa7dd2354f568f75249b57`.
  An independent validator reproduced the summary byte-for-byte.
- Decision: the remaining 2K prefill deficit is no longer transformer
  FlashAttention. NSG=4's synchronized transformer profile is about 8.46 s GPU
  versus DwarfStar's earlier 10.41 s layer total, while Rust still spends about
  2.36 s GPU in its 64-row bootstrap plus residency/host overhead. Preserve the
  accepted transformer and next localize that pre-transformer interval.

Validation:

- Five exact paired measurements plus both fresh-process warmups.
- Zero retries, invalid attempts, or blocked pairs.
- Independent paired-result validation and byte-identical summary.

### 2026-08-29 — Four-SIMD-group FlashAttention cuts prefill by 24.3%

- Changed the shared 512-wide non-vector FlashAttention specialization from
  eight to four SIMD groups and bound all 43 matching dispatches to the same
  named `kRustStarFlashNonvecNsg` constant. Unrelated compressor and
  attention-output launch geometries are unchanged.
- Three exact paired-ineligible profiles reproduced the complete oracle
  transcript. The layer-4 Flash median fell from 140.823 to 51.532 ms, a 63.4%
  reduction; layer 5 fell from 71.147 to 24.633 ms, a 65.4% reduction. The new
  Rust medians are respectively 52.5% and 54.9% faster than DwarfStar's
  independent 108.419 and 54.666 ms medians for the same representative
  stages.
- Profile SHA-256 values are
  `5e290bb4609f2f2d8cacb95726f6c2aa638a49563786d7c1e1cc28b4241b81e7`,
  `2b40959eae2cad5c872ff72e512feb65f23f1a348a6e36ecee4be8feadbca4e2`,
  and `8e824adc6f71cda9bee04e94df38cb2dbaec9e78287c156677ae0f17e944e096`.
- Three eligible exact candidate runs measured 172.234, 170.329, and 171.682
  prefill tok/s. Their 171.682 median is 24.3% above the immediately preceding
  NSG=8 control's 138.083 tok/s. Steady decode remained 23.790--23.942 tok/s.
  Raw SHA-256 values are
  `2732109c4c167e1b6df8adeb1f5355dad7cfbfab312f2df51d33d1decdbab6c4`,
  `0e39c4ee1a97791865aeff5a140223f43bb11748fc6c64431f65a71f9521f9a3`,
  and `9f4ac289b1367048affd58b5771e203811bbdf4a45e409edd80b84df941bf993`.
- The finalized named-constant source compiled to executable SHA-256
  `d453ba11c37fa0f9738b32491848a09953bbd8af51b3d413076c876db5c59a24`,
  identical to the tested candidate binary. Its final exact eligible control
  measured 165.365 prefill and 23.505 steady tok/s; normal profile arrays were
  all `null`. Raw SHA-256:
  `38332d61da94495825497310da5ec5f6caa6dc7c1b9fc1b5155fc8a7376c7379`.
- Decision: accept NSG=4 for the M1 Ultra target. The improvement is large,
  repeated, isolated to the established bottleneck, bit-exact end to end, and
  reproduced in unprofiled eligible execution. Freeze an immutable commit and
  run the standard five-pair comparison before updating the validated DwarfStar
  gap.

Validation:

- Optimized macOS release build.
- Three exact profiles and four exact eligible controls.
- 297 Rust tests and 73 Python tests pass.
- `cargo fmt -- --check` and `git diff --check` pass.

### 2026-08-29 — Attention internals isolate the 512-wide Flash dispatch

- Extended the paired-ineligible prefill profiler with five synchronized
  attention intervals in representative even layer 4 and odd layer 5:
  QKV/RoPE, compressor/staging, Flash block preparation, the main
  FlashAttention dispatch, and output/HC. The report now validates ten positive
  intervals, requires each five-stage sum to equal its attention interval, and
  labels all 52 diagnostic command buffers and host waits.
- Three exact Rust profiles selected prefill token 15342, final token 8954, and
  the pinned 128-token transcript checksum `17615242442502606640`. Median layer
  4 intervals were 16.657, 3.891, 0.031, 140.823, and 48.905 ms. Median layer 5
  intervals were 16.729, 1.564, 0.027, 71.147, and 51.007 ms.
- Repeated DwarfStar's independent layer-stage capture three times per layer.
  Its median grouped layer-4 intervals were 18.781 ms QKV/RoPE, 18.040 ms
  compressor/indexer, 108.419 ms attention, and 53.668 ms output/HC. Layer 5
  measured 18.285, 3.770, 54.666, and 52.736 ms. Rust is therefore already
  faster in every surrounding group, while its main FlashAttention dispatch is
  29.89% slower in layer 4 and 30.15% slower in layer 5.
- Source and runtime inspection ruled out the first buffer-mode hypothesis.
  Both paths compile the same `kernel_flash_attn_ext_f16_dk512_dv512` source
  with the same function constants, 256-by-head grid, 32-by-8 threadgroup,
  28,672-byte threadgroup allocation, and default fast math. Both use shared
  KV scratch on the M1 Ultra; DwarfStar enables private GPU-only scratch only
  on M5.
- Accepted Rust profile SHA-256 values are
  `655036c88580cf34268db7058301131e26b241df1b1360fd8b6526f5f10c4fcd`,
  `abdf50876a893a3518a9ed2c53af58d27f23283a1e704dc4ac6bbdc2ac6f95dd`,
  and `42635cf17d4df9424af052ab85605ac09c92469edcb388965cf1e860c48fa42a`.
  The final executable SHA-256 is
  `d53bd748bd3904d5292d7c7bf5a5a33da2f254fa82d877844ace36828becfe2b`.
  Dwarf measurement SHA-256 values are
  `9d456de1872b812388c85df3a4d02dab200231e6df734ee716c227b5bcbeb86c`,
  `474604408a5722aeb69f2e4cec135c9d02815f51c5d5c96ff304837ba6cbb0bd`,
  and `5e3442e326e985970e2f6dfc7b5b511e6a3fef2e9c5f56612d7cd73419ef2bcf`
  for layer 4, plus
  `36bcf4cfd6ab55ff24f855dfdbdb2ea2abc385f893c80f7dabb4b25701196c8d`,
  `a1b844197fc27e95fd7ab3f78bbbe8bc50d08c33f4bd6676d4a6ad4fca2150ac`,
  and `f1363e5336eac84fee5ebb84a22820cbe6d73d97c7ba19ebd133f6e9f21ce71a`
  for layer 5.
- A final unprofiled control from the same executable remained exact and paired
  eligible, emitted all three prefill profile arrays as `null`, and measured
  138.083 prefill tok/s plus 24.029 steady decode tok/s. Raw SHA-256:
  `af76275bd9fe8beb613e1412196525e347178bdcce84cf856248e5f6e14dbf04`.
- Decision: do not change buffer storage or kernel math from this diagnostic.
  Build a same-context NSG scheduling benchmark for both representative shapes;
  promote a candidate only after exact output comparison and repeatable median
  improvement.

Validation:

- Optimized macOS release build and exact live profile/control runs.
- 297 Rust tests and 73 Python tests pass.
- `cargo fmt -- --check` and `git diff --check` pass.

### 2026-08-29 — Representative layers localize the prefill deficit to attention

- Extended the paired-ineligible prefill profiler with one additional
  synchronized boundary in representative even layer 4 and odd layer 5. The
  report preserves all 41 layer totals and additionally requires each
  attention-plus-FFN pair to sum exactly to its layer total.
- The exact 128-token transcript remained unchanged across two clean profiles.
  Layer 4 measured 220.694--228.611 ms attention and 100.945--101.421 ms FFN;
  DwarfStar's independent split profile measured 182.709 and 97.515 ms.
  Rust's layer-4 attention is 20.79--25.12% slower while its FFN is only
  3.52--4.11% slower. Attention accounts for 90.68--93.05% of the
  representative total-layer deficit.
- Layer 5 measured 145.157--150.229 ms attention and 103.672--104.385 ms FFN;
  DwarfStar measured 119.490 and 100.706 ms. Rust's layer-5 attention is
  21.48--25.73% slower while its FFN is 2.95--3.65% slower. Attention accounts
  for 89.31--89.64% of this deficit.
- Final accepted evidence:
  `rust-star/.work/prefill-stage-profile-02/engine-run.json`, SHA-256
  `1f2a6c3b1241ad85170ed831b96dacf508ad451248be9dea2a421f4b0ff955a1`.
  The final profiled executable SHA-256 is
  `10ea66c49de49c8712e144d1851a7ffe7f57ebdcb2c2822343fd7e9dd0c0503b`.
  The first clean repeat remains at SHA-256
  `9c21df58e352b2816980cf50f2393bbdf8621b393a6da1705ae46db3037d4013`.
- Tested a bounded production-only snapshot-elision hypothesis on layers 4 and
  5. It aliased Q/KV state in place and removed diagnostic blit/encoder breaks.
  A live exact profile moved layer 4 from 329.556 to 332.064 ms and layer 5
  from 254.614 to 257.081 ms, with both attention intervals also slower. The
  candidate was rejected and fully reverted; its raw SHA-256 is
  `0cb08cad76a0e1cae398ceee09b3c16d35e8bacc8e3218c90f9d11bf02e51ce4`.
- Decision: do not generalize snapshot elision from whole-engine noise. Build
  an alternating same-context attention harness that separately measures
  QKV/RoPE, compressor, FlashAttention, and output/HC candidates. The shared
  attention path, not FFN or one anomalous layer, is the established target.
- A final normal-path control from the same executable selected token 15342,
  matched the full transcript, emitted both prefill profile fields as `null`,
  remained paired eligible, and measured 132.706 prefill tok/s plus 23.211
  steady decode tok/s. Raw SHA-256:
  `11abfb087e86a06842b6f14e4e7e759c6865c43b0d0d54bb3b2e9b12e91e972b`.

Validation:

- Optimized macOS release build and exact live `engine-profile`.
- Full Rust and Python suites pass at the final checkpoint.
- `cargo fmt` and `git diff --check` pass.

### 2026-08-29 — Remaining prefill gap follows the alternating layer families

- Extended `engine-profile` with a prefill-only diagnostic that splits the
  native layers-2-through-42 transformer command into 41 synchronized layer
  command buffers and records each completed Metal GPU interval. Normal
  `engine-measure` does not allocate profiling state, split the command, or add
  waits. The raw report carries all 41 values, their shares, the perturbation
  method, and a fail-closed paired-protocol blocker.
- Rejected two timestamp-counter designs before accepting evidence. The M1
  Ultra does not expose dispatch-boundary sampling. Stage-boundary marker
  passes resolve, but their timestamps are not global workload fences: four
  markers absorbed multi-layer scheduler intervals and produced false
  2.6--2.9 second spikes. Those runs are not optimization evidence.
- The accepted synchronized profile preserved the exact 128-token oracle
  transcript and reported 11,896.204 ms summed GPU time across layers 2--42.
  Even layers totaled 6,858.124 ms (326.577 ms mean); odd layers totaled
  5,038.080 ms (251.904 ms mean), a 1.296436x even/odd ratio.
- DwarfStar's independent split profile totals 10,408.751 ms over the same
  layers. Its even layers average 283.103 ms and odd layers 223.180 ms, a
  1.268498x ratio. Rust is therefore broadly 15.35% slower on the even family
  and 12.87% slower on the odd family. No single Rust layer is pathological;
  the largest accepted interval is layer 36 at 333.502 ms.
- Accepted profile evidence:
  `rust-star/.work/prefill-layer-profile-05/engine-run.json`, SHA-256
  `4ab018bdac5c47559e976f3bde660aa84f72cee36b2f60d877ce588860738b0d`.
  The release executable SHA-256 is
  `71627f7b9cf9ee6263b7e2dd2bcd21369bf9ce4c7444b713fb26ed1d86b3dabd`.
- A fresh unprofiled control from the same executable selected prefill token
  15342, matched the exact transcript, emitted no prefill profile, remained
  paired eligible, and measured 132.637 prefill tok/s plus 23.664 steady
  decode tok/s. Its raw SHA-256 is
  `d7dc37853aeefad0482f37c5c7f831c2ec82ed1f6d5b957e015cfffc78dd4ac1`.
- Decision: next split a representative even and odd layer into attention and
  FFN intervals. The structural match already prioritizes the shared prefill
  attention path, then the even-only ratio-4 compressor/indexer work; it does
  not justify tuning an isolated layer.

Validation:

- Optimized macOS release build.
- 297 Rust tests pass.
- `cargo fmt` and `git diff --check` pass.
- Exact live `engine-profile` and independent normal-path `engine-measure`
  controls both pass the full 128-token transcript.

### 2026-08-29 — Wide bootstrap raises the validated prefill ratio to 0.7139x

- Froze the `cdb6377` executable and ran the predeclared five-pair 2K/128 plan
  with SHA-256
  `0e8776c817da116d33601b12dcee27598135ea2b7c9e0bf06b26138725412212`.
  Both fresh-process warmups and all five timed pairs passed. There were no
  failures, retries, blocked final pairs, or retained invalid attempts.
- Rust Star median prefill was 133.070209951 tok/s versus DwarfStar's 186.35.
  The median within-pair ratio was 0.713851042x, leaving a 28.61% gap. Compared
  with `eae11df`'s 0.673232187x ratio, the wide bootstrap improves the ratio
  6.03% relatively and narrows the gap by 4.06 percentage points.
- Rust Star retained both generation leads: 23.543315664 versus 22.72 tok/s
  steady (1.038685191x, +3.87%) and 23.502293406 versus 22.13 tok/s complete
  generation (1.064717406x, +6.47%). First-token wall time measured a
  1.129910481x pairwise ratio, so Rust was 13.0% slower on that interval in
  this comparison rather than tied.
- Independent validation and aggregation reproduced the byte-identical
  summary. Raw result SHA-256 is
  `e596d1aae9e49ed329fb88c7df17c8be7312ff058098ebac0a2a55550ed818c4`;
  summary SHA-256 is
  `dd5620d419d609878c05d027b5327ae18dc35ed68ac650dec42c6b42cb1de6ce`.
  Private evidence is under `rust-star/.work/paired-2k-128-cdb6377/` and remains
  intentionally untracked.
- Decision: accept 0.713851042x as the new validated 2K prefill ratio. The next
  optimization target is the approximately 11.94-second Rust remaining-
  transformer GPU interval versus DwarfStar's 10.92-second full transformer;
  first add per-layer attribution so the comparison is structurally valid.

### 2026-08-29 — Exact 64-row bootstrap cuts its GPU interval by 29.7%

- A DwarfStar split-profile control measured 10,921.115 ms of transformer
  execution and only 99.724 ms of graph encoding. Its even ratio-4 attention
  layers were approximately 181--187 ms, odd ratio-128 attention layers were
  approximately 118--125 ms, and FFNs were approximately 96--104 ms across
  both families. The remaining Rust prefill deficit is therefore GPU schedule
  and kernel work, not graph-construction overhead.
- Reopened the rejected 64-row Rust bootstrap as a correctness investigation.
  The prior wide launch supplied 64 rows to its arguments but launched only one
  32-row matrix tile, four 8-query FlashAttention groups, and a grouped-output
  routing buffer sized for 32 rows. The full-2K experiment also overflowed the
  128-row raw-cache append instead of wrapping it.
- Made the F16/Q8 matrix X grids, FlashAttention query grids, grouped-output
  routing offsets, allocation, and work capacity derive from `rows`. Production
  now runs 32 64-row command buffers with one tail wait. Diagnostic commands
  still run the original 64 exact 32-row tiles.
- Temporary final-tail checksum tracing localized and then verified each seam.
  After the fixes, token IDs, HC ingress, Q/KV, RoPE/cache, FlashAttention,
  grouped output, FFN, layer-0/1 HC, layer-2 KV, and both layer-2 compressor
  tails all matched the 32-row C0 artifacts by FP32 bit pattern. All tracing
  code was removed before publication.
- Published immutable code checkpoint
  `cdb63779c45e3f4b0cbe45d29f29938bd53985b6` with tree
  `1124229819dacb0061998194eb8abdfa7b62245c` and executable SHA-256
  `14eed212b82a358aff88b726f96ba9a09798870edb1d31ddb8cea93e7332d28e`.
- The clean eligible exact run matched the complete 128-token transcript and
  measured 132.726509564 prefill tok/s. Bootstrap GPU time fell from the prior
  normalized 3,457.269167 ms to 2,429.833625 ms, a 29.7% reduction; total
  prefill throughput is 4.52% above the prior normalized 126.983490539 tok/s.
  Steady decode remained 23.661023050 tok/s. Evidence SHA-256 is
  `23a87c1f8ffdff3dc427f0b80a2f337236fbd10a523e80c5bd02736bce7f7a96`.
  This is a single development run, not a paired claim.
- The independent synchronized 64-by-32-row layer-2 compressor regression
  passed every retained exact boundary. Report SHA-256 is
  `42fd4f7d2733d738d98a3ad8a41d0dc74e03e83f49570dea57d56d845996dbf2`.
- Validation: optimized macOS build, exact production 2K/128 engine run, exact
  legacy 32-row compressor loop, all 296 Rust tests, all 73 Python tests,
  `cargo fmt --check`, and `git diff --check`.
- Next: freeze the implementation and run a new immutable five-pair comparison
  before replacing the published `eae11df` ratio.

### 2026-08-29 — Queued prefill improves the paired ratio, not enough to close the gap

- Published immutable candidate `eae11df60c1441445faef8170335fd1846b3ffaa`
  with source tree `01ecc7181638f08b3ade0d881e2cca945af4a617` and
  executable SHA-256
  `3cb264ad88f0afc2ebc2d579b9484809d805560b9608cd3d6f03722b7789eb1f`.
- Ran the exact five-pair 2K/128 schedule from plan SHA-256
  `ada7c64751b40b1aebf4d0e59702142c310d0ba6d3843fc539c50b11a3b188fd`.
  Both warmups and all five timed pairs passed; there were no failures, retries,
  blocked pairs, or retained invalid attempts.
- Rust Star median prefill was 126.958125743 tok/s versus DwarfStar's 188.58.
  The median within-pair ratio was 0.673232187x, leaving a 32.68% gap. Compared
  with `2c6f80e`'s 0.659348296x ratio, queueing improves the paired ratio 2.11%
  relatively and narrows the gap by 1.39 percentage points. Absolute throughput
  was lower for both engines in this run, so the paired ratio is the valid
  cross-session comparison.
- Rust Star retained its decode lead: 23.655165050 versus 22.81 tok/s steady
  (1.037052391x, +3.71%) and 23.611303513 versus 22.18 tok/s complete generation
  (1.064531267x, +6.45%). First-token wall time was effectively tied at a
  1.013342227x paired ratio.
- Private finalized summary SHA-256:
  `2baf206b59714f66a42743667d74ce22e977ad9647813db7f9ca4f7d252e016f`.
  Raw result SHA-256:
  `05de7b8b18e8c749a230044afa54e4c1249ca36644f2328853f5da6288bdba8b`.
- Decision: retain the one-wait 32-row queue because it is exact and modestly
  improves paired prefill. The next prefill work must reduce GPU kernel/schedule
  cost while preserving 32-row arithmetic; wider-row batching is closed by the
  oracle failures.

### 2026-08-29 — Exact prefill attribution and queued 32-row bootstrap

- Added passive eligible-run attribution for bootstrap tiles, remaining
  transformer, output head, prefill-to-decode handoff, model residency, and
  residual host overhead. No diagnostic tensors, extra Metal encoders, or
  synchronization points are added.
- Baseline attribution evidence
  `rust-star/.work/prefill-attribution-01/engine-run.json` (SHA-256
  `49425b0607c97e204c5222938dd84b45eff668e2c72f30d64489b7178e22205a`)
  measured 126.291576234 prefill tok/s. Of 16,216.441833 ms total, the 64
  bootstrap tiles used 3,473.159292 ms GPU, the remaining transformer used
  11,334.500417 ms GPU, and residual host overhead was 1,154.636625 ms. The
  deficit is therefore dominated by GPU kernel/schedule work, not Rust host
  bookkeeping.
- Generalized the production-only bootstrap experimentally. Both 64-row tiles
  and one full 2,048-row tile failed the first selected-token oracle; the latter
  selected token 16 instead of 15,342. Both were rejected, and all diagnostic
  and production arithmetic remains at the proven 32-row width.
- Kept the 64 exact 32-row command buffers but removed 63 inter-tile host waits.
  The serial Metal queue carries live KV/compressor dependencies and the final
  tile waits once. Diagnostic collection remains synchronized per tile. A raw
  eligible run measured 133.684055612 prefill tok/s and matched the exact
  128-token transcript; its JSON SHA-256 is
  `dfbb3399f1b4334c90d25073c32b49dc72faf6296beb3da05bafce6da0499e68`.
  This single run is not a paired performance claim.
- Fresh-process normalized evidence at
  `rust-star/.work/prefill-queued32-adapter-01/measurement.json` passed at
  126.983490539 prefill tok/s. Measurement SHA-256 is
  `c5c61201978b1fadc09c130da68639770b617b851aa207f64879b0b6a9af6771`;
  its raw engine-run SHA-256 is
  `b692b82cd59b3791a6ef2b09d00a4134bd020c73efc0e704a11993c94a31f0ae`.
  The normalized raw attribution reports 3,457.269167 ms bootstrap GPU,
  11,772.479042 ms remaining-transformer GPU, and 745.343792 ms residual host
  overhead.
- Hardened `measure_ruststar.py` parsing: all attribution fields are mandatory,
  GPU time may not exceed stage wall time, host overhead must be nonnegative,
  and wall components must sum to `prefill_ms` within 0.001 ms.
- Validation: optimized macOS build passed; all 296 Rust tests and 73 Python
  tests pass. The synchronized 64-tile layer-2 compressor C0 control also
  passed; its private report SHA-256 is
  `dc34b754f0823f02fd821533cbc4f1d5e87091ef7389bd929c4b3936d085e414`.
  The next gate is an immutable five-pair exact 2K/128 comparison before
  attributing any prefill gain to queueing rather than run variance.

### 2026-08-29 — Metal residency closes the first-token transition cliff

Objective:

- Isolate the approximately 500 ms of first-token wall time that was absent
  from Metal GPU timestamps, without moving work outside measured intervals.

Evidence:

- Pooling decode activation scratch was rejected. Its exact position-8195
  control passed, but an exact 2K/128 run regressed to 638.007 ms first-token
  latency and 22.4535 tok/s steady. Report SHA-256 values are
  `f2224827e267e35f16c0ac7d1734c3873b416f38f92e3ac6654c27876e2ea4be`
  for the retained control and
  `e946c3a650fd04e9bb49b7bb0358e8a143c40791512a1e33d643304710113b15`
  for the engine run. The implementation was reverted.
- Source comparison found that pinned DwarfStar attaches stable mapped model
  views to an `MTLResidencySet` and performs a coarse GPU weight touch before
  inference. Rust Star previously used only CPU `POSIX_MADV_WILLNEED` plus one
  volatile byte touch per tensor-data page.
- Rust Star now registers its 1,136 existing exact no-copy model views in one
  residency set, attaches it to the command queue, and synchronously touches
  83,334 one-mebibyte strides. The aggregate aligned view spans account for
  86,370,050,944 bytes. Setup is explicitly inside `prefill_ms` and reports
  independent wall/GPU timing metadata.
- The complete retained position-8195 control remained C0 exact across all 43
  layers, 1,705 dispatches, 1,370/1,370 identity-preserving mmap views, all 43
  HC boundaries, all 129,280 logits, and token 35597. Report SHA-256 is
  `5bf001e893e87a448bcb803db752afda1c629bb5748c8bd9221e3474440e8b1a`.
- Two exact 2K/128 candidate runs measured 53.510833 and 49.586167 ms for the
  first token. The latter measured 24.072612466 tok/s steady and 24.036244767
  tok/s complete generation; its residency setup took 7.768 ms wall and
  7.12425 ms GPU. Report SHA-256 values are
  `61470efb31cd5f150190ac47c0a54d419e5319e2c21351556b3b9a470494134f`
  and `20547f7bdded40792d9f12be481172cd0dab61394c202cc1bbbd512000b9b028`.
- An adjacent run of the immutable pre-change `7de9511` executable still took
  615.953042 ms for the first token, including 612.763333 ms transformer wall
  versus only 62.572500 ms GPU. Its report SHA-256 is
  `b93e32fa1f465d7aee2bea11aa42812de1e86cd8fe01e882376d7eb4f51a9d47`.
  The candidate therefore reduced adjacent-control first-token latency 91.95%
  while retaining the exact transcript.
- Published source commit `2c6f80e93340938e14d369a90c0e15d535e371ab`
  with tree `bc664db70bf9a03ea82b44cd70b007a84a2fff2e`. The immutable
  optimized executable SHA-256 is
  `bc28df410a504cf8197f21782eed11669da355ac5d973bff493f22b137d7e747`.
- The first attempted plan correctly blocked during candidate warmup because it
  was mistakenly bound to a stale executable that lacked the new residency
  record. No timed pair ran. That checkpoint remains preserved under
  `paired-2k-128-2c6f80e`; the corrected plan uses a new output directory and
  SHA-256 `edd2341a277d361ac16ac9a7ee82f7295e2f2ffcd305cda4afd16a0a564a47c1`.
- Both corrected warmups and all five `AB`, `BA`, `AB`, `BA`, `AB` pairs passed
  with no timed failure, retry, invalid attempt, capacity failure, swap use, or
  recorded thermal/performance warning. All exact candidate transcripts passed
  the adapter's independent residency, scheduler, and transcript gates.
- Rust Star steady decode measured median 24.230812222 tok/s (MAD 0.030629462,
  range 24.197976798--24.285034979) versus DwarfStar 23.39 tok/s (MAD 0.04,
  range 23.33--23.44). All five pairs favored Rust. The median within-pair ratio
  was 1.037257019x (MAD 0.001239146, range 1.032336894--1.038496166), a
  validated 3.73% lead.
- Complete generation measured Rust 24.198539179 versus DwarfStar 22.81 tok/s;
  all five pairs favored Rust and the median ratio was 1.061823177x, a 6.18%
  lead. Median first-token latency was 48.720167 versus 48.661 ms, with a
  0.995386079x median paired ratio. Prefill measured 131.607846785 versus
  199.62 tok/s, a 0.659348296x ratio and 34.07% remaining deficit.
- Relative to `7de9511`, the new Rust median improved 3.13% steady, 12.49%
  complete generation, and 1.79% prefill while first-token latency fell 91.08%.
  The validated summary re-aggregates to SHA-256
  `5308900982d45da6fe0a3f13cc5ea46b11bf38a5bdac3d061a1bec636f5e062f`.
  The local result-manifest SHA-256 is
  `503d5e82793f9bedfdbdf86d4f700a93495545d5d90038c36acc4d3a99e2eaf1`.

Decision:

- Accept and publish the residency policy. The exact 2K/128 development
  headline is now a 3.73% steady-decode lead, 6.18% complete-generation lead,
  and effectively tied first-token latency. Keep prefill as the next
  performance bottleneck and do not extrapolate this narrow result to 256K.

Next:

- Profile and reduce the remaining prefill deficit, then extend the eligible
  exact engine to the next context frontier.

### 2026-08-29 — Rust Star crosses DwarfStar in validated steady decode

Objective:

- Re-measure the accepted attention optimizations as one immutable engine and
  determine whether their focused gains survive the exact paired workload.

Evidence:

- Published source commit `7de9511ff858333cfaa4a6ae0ac9814a24488d7a`
  with tree `7d1f61971742c6b840ef9e51ecff384b6fc38c85`, then copied its
  optimized executable to an immutable local path. Executable SHA-256 is
  `4c4a65bd2bec62448f9bedd57b7c17efc0524c60bf7ed2618b4bbce303857983`.
- An initial healthy exact engine control matched the complete 128-token oracle
  transcript and reported 23.702300769 steady tok/s. Its report SHA-256 is
  `a880c4ebcff89bf4fff737a60f20f0878104a235451fbce45f9303bc4f8db2ed`.
- Froze paired plan SHA-256
  `ee690ee8f7ec7978b804451ac3675582e9a9a938bd28a0b4917540a57956479d`.
  Both warmups and all five `AB`, `BA`, `AB`, `BA`, `AB` pairs passed with no
  adapter failure, retry, invalid attempt, capacity failure, swap use, or
  observed thermal event.
- Rust Star steady decode measured median 23.495447999 tok/s (MAD
  0.104610085, range 23.220698759--24.110855674) versus DwarfStar 22.91 tok/s
  (MAD 0.26, range 22.56--23.65). All five pairs favored Rust Star. The median
  within-pair ratio was 1.025554256x (MAD 0.006067758, range
  1.002188121--1.045779226), a validated 2.56% Rust Star lead.
- Complete generation measured Rust 21.512204760 versus DwarfStar 22.31 tok/s,
  for a 0.964026748x median pairwise ratio. Prefill measured 129.293920693
  versus 188.85 tok/s, for 0.684638182x. First-token latency measured
  546.059208 versus 45.98 ms, for a 12.192259243x median pairwise ratio.
- Relative to the prior `4fc61f7` Rust medians, this checkpoint improved steady
  decode 7.09%, complete generation 11.94%, prefill 4.22%, and first-token
  latency 37.38%. The paired steady ratio improved 5.75%.
- Raw SHA-256 is
  `434aa14133dbbf7a3cb261d65c669801420669834c646abf8310de687a31e545`;
  summary SHA-256 is
  `d35d9a778e00613af3ac66c56be58b82ee1803bc581616691765f04c8cbc405c`;
  runner-state SHA-256 is
  `d9a0774482273520992e9862ced67f435733c615e88e24f8a7029b45455c72df`;
  and result-manifest SHA-256 is
  `49fe8da8bc80fe6a1c9fe20bf1a8b93b17a1f047ad8fb08809895e1a90b88e02`.
  Full private-path evidence is under
  `rust-star/.work/paired-2k-128-7de9511/` and intentionally untracked.

Decision and next step:

- Accept the steady 2K/128 lead as the new development frontier. It is not the
  protocol's 256K headline claim. Optimize first-token residency/scheduling
  next because it now explains the remaining complete-generation deficit;
  keep prefill as the second independent performance gap.

### 2026-08-28 — Q-head/KV RoPE dispatch fusion accepted

Objective:

- Reduce the decode Q/KV RoPE boundary without changing the Q-head RMSNorm
  reduction tree or either RoPE arithmetic path.

Implementation:

- Tested an exact 128-thread Q-head launch against the original 256-thread
  geometry. It preserves the four active SIMD-group contributions and all
  32,768 outputs, but was rejected after measuring slower.
- Added a combined 65-workgroup kernel. Groups 0--63 reproduce the accepted
  per-head Q RMSNorm/RoPE kernel exactly; group 64 reproduces the standalone
  in-place KV tail RoPE equations with the same dense or compressed YaRN
  parameters.
- Production decode now uses the combined dispatch while the split path remains
  selectable and directly benchmarked. Prefill launch geometry is unchanged.

Evidence:

- The 10-warmup/100-measured thread control recorded 0.015 ms GPU median for
  256 threads and 0.016 ms for 128 threads, a 7.34% regression. It remains
  rejection evidence; SHA-256
  `66a8d86769183cfae7776fd9c7c88e936fafdc101c84b0ba3768466b1b92a56e`.
- The finalized optimized-runtime 10-warmup/500-measured fusion run recorded
  0.018 ms separate versus 0.010 ms fused GPU median, a 46.71% reduction, while matching all
  32,768 Q and 512 KV values after every execution. Evidence SHA-256
  `29b8150145aec0f60d47a156fc37eded06c0ab8a1303a99a9759daf2d1c29a28`.
- The complete 43-layer position-advancing decoder remained bit-exact at
  positions 1, 2, and 3, covering both dense and compressed YaRN parameters.
  Evidence SHA-256
  `44869d802080ecfe9435e4aa2655d029398fcccffeea2b99e75665f24de79df1`.

Decision:

- Keep the original 256-thread Q reduction geometry and enable Q-head/KV RoPE
  dispatch fusion in production decode.
- Treat the focused result as diagnostic evidence until a healthy immutable
  full-engine comparison is available.

Validation:

- The optimized macOS runtime built successfully; all 296 Rust tests and all 68
  Python artifact, adapter, and paired-runner tests passed.

Next:

- Continue measured attention-setup work or retry the immutable full-engine
  control under healthy residency.

### 2026-08-28 — Four-row Q-B projection accepted

Objective:

- Reduce Q-B projection overhead without changing any output row's Q8_0
  accumulation or SIMD reduction order.

Implementation:

- Added exact four- and eight-output-row specializations of the existing Q8_0
  matrix-vector kernel. Both retain NSG=4 and only share the normalized Q input
  across more independently accumulated output rows.
- Added `qb-rows-bench`, which alternates the original two-row geometry against
  one candidate in a persistent Metal context and validates Q-Lora norm, KV
  raw/norm, and all 32,768 raw Q values after every execution.
- Selected four rows for production: it halves Q-B workgroups from 16,384 to
  8,192, matched eight rows' GPU median, and used less shared/register state.

Evidence:

- The finalized four-row 10-warmup/100-measured run recorded 0.215 ms baseline
  versus 0.175 ms candidate median GPU time for the full attention-setup chain,
  an 18.89% reduction. Evidence SHA-256
  `a8d7d4437a62949b622c6226d48edd0e8d0c0e6fdb251fadb0b27130cba74920`.
- The eight-row control was also exact and recorded 0.174 ms candidate median,
  but its 0.429 ms wall median was worse than the first four-row run's 0.418 ms
  and it doubles per-workgroup row state. Evidence SHA-256
  `01a06211b984815a48d163c7878d0e35706e35bb12d44769d8baead4c650e40a`.
- The complete 43-layer position-advancing decoder remained bit-exact at
  positions 1, 2, and 3. Evidence SHA-256
  `5ef626cc909a74ac55f197ccf892bc847d9118089818b5b22f0f438afd4ea0d4`.

Full-engine control:

- Immutable executable SHA-256
  `6966067a1cb42e0a32fca91a20820ecaa92dc45f4480e0fe8dfa26f3ce4c0216`
  reproduced the exact 128-token transcript and remained paired eligible, but
  again entered the slow residency mode at 0.547 steady tok/s despite 94% free
  memory and no thermal or performance warning. It is failure evidence only;
  report SHA-256
  `bf5ae65149b035b663bb3d35d8ba1d1296d1045d55341e5c0217038f8614b87c`.

Validation:

- Both row-group candidates matched every pinned setup value for all 100
  measured executions.
- Layers 0--42 matched every retained C0 boundary at three consecutive decode
  positions with production four-row Q-B enabled.
- The optimized macOS runtime built successfully; all 294 Rust tests and all 68
  Python artifact, adapter, and paired-runner tests passed.

Decision:

- Keep four-row Q-B enabled in production. Do not choose eight rows on a
  statistically indistinguishable GPU median when four rows have lower state
  pressure and better observed host interval.
- Do not use the slow immutable run for a throughput claim or updated project
  headline.

Next:

- Continue with the Q-head RMSNorm/RoPE boundary while waiting for a healthy
  full-engine residency state.

### 2026-08-28 — Dense Flash reduction/inverse-RoPE fusion accepted

Objective:

- Remove the standalone inverse-RoPE dispatch and its in-place 32,768-value
  attention pass without changing any observable FP32 boundary.

Implementation:

- Added `kernel_flash_attn_ext_vec_reduce_inverse_rope`, retaining the exact
  split-K reduction statements, publishing the pre-RoPE diagnostic output,
  synchronizing device memory, and then executing the reference 256-lane
  inverse-RoPE loop inside each head workgroup.
- Production aliases the raw/post-RoPE buffers and selects the fused reducer for
  dense attention. Exact probes keep distinct buffers; sparse indexed attention
  retains its standalone inverse-RoPE kernel.
- Added `attention-rope-fusion-bench`, which alternates the 18-dispatch control
  and 17-dispatch candidate in one Metal context and validates every post-RoPE,
  attention-low, attention-out, and HC-post value after each execution.
- Updated dense decode from 1,287 to 1,244 transformer dispatches. The first
  production sparse schedule drops only its 22 dense-attention launches, from
  1,749 to 1,727; its 21 sparse indexed-attention layers are unchanged.

Evidence:

- The finalized 10-warmup/100-measured alternating run recorded 0.576 ms
  separate versus 0.535 ms fused median GPU time, a 7.12% reduction for the
  full layer-0 attention-output chain. Evidence SHA-256
  `be119bad8056fa1c137796051decb764853f1b42c5f2e129d855aa33b00537ec`.
- A first layer-0-only prototype passed but a 43-layer run correctly rejected a
  one-ULP YaRN mismatch at layer 2. Restoring the reference kernel's exact
  256-lane loop eliminated the drift; the final layers-0--2 evidence SHA-256 is
  `a0b32c3a85cdfa1cc94acbf01c04b6b7c0984458d142cda4d38cf81df9b96074`.
- The complete 43-layer position-advancing decoder is bit-exact at positions
  1, 2, and 3, including every compressed YaRN layer. Evidence SHA-256
  `749efad3ddaeb23419796e277c9424e527cfc97f3ddf6fd349801dd80d4655be`.

Validation:

- All 293 Rust tests passed.
- The fused and separate paths preserved the pre-RoPE boundary and reproduced
  every pinned downstream boundary on all 100 measured executions.
- The complete 43-layer decoder reproduced all retained C0 fixtures at three
  consecutive positions.

Decision:

- Keep dense Flash reduction/inverse-RoPE fusion enabled in production. It
  removes 43 launches from dense decode and shows a repeatable focused-chain
  improvement while preserving the strict C0 contract.
- Do not apply it to sparse indexed attention until that reducer receives its
  own exact fusion and same-context validation.

Next:

- Obtain a healthy immutable engine measurement, then use the refreshed profile
  to choose between Q-B launch geometry and another structural attention/setup
  fusion.

### 2026-08-28 — Attention-output launch alternatives rejected by C0

Objective:

- Refresh the immutable `298e4ac` stage profile and test whether the
  attention-output/HC family can improve on pinned DwarfStar's NSG=4/4 launch
  geometry without changing arithmetic.

Implementation:

- Added `attention-output-nsg-bench`, which lazily compiles NSG=2/4/8 variants
  for the grouped attention-low and fused output-B/HC kernels, alternates one
  candidate against NSG=4/4 in a persistent Metal context, and checks inverse
  RoPE, attention low/out, and the four-stream HC post-state after every run.
- Kept the production pipeline and threadgroup geometry at NSG=4/4. Candidate
  pipeline selection is diagnostic-only and restores production before return.

Evidence:

- The fresh immutable full-engine profile launched with 94% system memory
  availability, zero throttled pages, and essentially unused swap, but still
  entered the known slow GPU residency state: 0.594 steady tok/s. The exact
  transcript passed, but the report is retained only as failure evidence;
  SHA-256
  `eff3250e27d7cde40eba778e833290fe16986acd7663041970ac961ebc6c7bd2`.
- Attention-low NSG=2 first differed at `attn_low[1]`, actual `0x3e4db9c2`
  versus expected `0x3e4db9c4`; evidence SHA-256
  `ebcce5347aa60b302b98addcc0045608ba018506efa7e2f53c6e55480327beda`.
- Attention-low NSG=8 first differed at `attn_low[1]`, actual `0x3e4db9c0`
  versus expected `0x3e4db9c4`; evidence SHA-256
  `4c2ffb08bea9406439105736ab955590c22568250737aff53709dc9d458ee75b`.
- Output-B/HC NSG=2 first differed at `attn_out[0]`, actual `0xbf36d5e4`
  versus expected `0xbf36d5e6`; evidence SHA-256
  `1eb5c2a3a9a96bd2fd393943b9084eb457e77f36231f2dcd3ca3ed5141906af2`.
- Output-B/HC NSG=8 first differed at `attn_out[1]`, actual `0x3fb7a5c1`
  versus expected `0x3fb7a5c4`; evidence SHA-256
  `e0ae7df582f6af37b2202e63a015029c347a8f9431b4a8e62508cd441695efa9`.

Validation:

- Optimized macOS build passed and all candidate Metal functions compiled on
  the M1 Ultra.
- All 292 Rust tests and 68 Python tests passed.
- Every invalid candidate was rejected before timing or production selection.

Decision:

- Retain NSG=4/4. Unlike routed-expert launch tuning, changing either
  attention-output reduction tree is not C0-compatible, so no performance
  comparison or eligible full-engine run is warranted.
- Retain the benchmark because it makes this exactness boundary executable and
  prevents future tuning from accepting a few-ULP drift as a speedup.

Next:

- Investigate fusing inverse RoPE into the attention boundary. It can remove a
  complete decode dispatch and an in-place 32,768-value pass per layer, so its
  potential is materially larger than suppressing the diagnostic-only 4,096-
  value `attn_out` store.

### 2026-08-28 — Paired Q-A/KV projection accepted

Objective:

- Reduce the second-largest profiled family, attention QKV ingress at 22.82%
  of steady transformer GPU time, without changing any decode arithmetic.

Implementation:

- Ported pinned DwarfStar's `kernel_mul_mv_q8_0_f32_pair` and its exact
  standalone reduction trees into a dedicated Metal source.
- Replaced the separate layer Q-A and KV Q8_0 dispatches with one paired
  dispatch while retaining both standalone pipelines as an explicit diagnostic
  control.
- Added `qkv-pair-bench`, which alternates separate and paired paths in one
  persistent Metal context, reverses order every round, and validates Q-Lora
  norm, KV raw/norm, and the complete 32,768-value Q raw output after every
  execution.
- Updated the decode schedule contract from 30 to 29 dispatches for layer 0,
  1,330 to 1,287 for the dense transformer, and 1,792 to 1,749 for the sparse
  transformer. Native batched-prefill schedules are unchanged.

Evidence:

- `rust-star/.work/qkv-pair-bench-100.json` records 20 warmup plus 100 measured
  alternating rounds per path; SHA-256
  `5138e0e3d11fcd70f69ada8f1bef2e7f934f63b433a655a6cae8681e671249bc`.
- Separate Q-A/KV: 0.216 ms median GPU time, 0.002 ms MAD.
- Paired Q-A/KV: 0.213 ms median GPU time, 0.001 ms MAD, 1.44% faster for the
  full eight/nine-dispatch attention-setup chain.
- The focused attention-setup fixture is exact; evidence SHA-256
  `083d2489ca5cb163ae9213a5dab010ccdeb94a3f6451358d49f41069b67aed4c`.
- The complete 29-dispatch layer-0 continuation is exact through attention,
  router, routed/shared experts, and final HC state; evidence SHA-256
  `a5a735cb2480dcf2bb2d63f6d75935b08234b4d610734e52367804f6c5bdcbde`.
- A separate full-engine baseline entered the known paging/throttled mode at
  0.598 steady tok/s and is retained only as failure evidence, not as a speed
  comparison.

Validation:

- Optimized macOS build passed and the paired kernel compiled on the M1 Ultra.
- All 291 Rust tests and 68 Python tests passed.
- Both A/B paths reproduced every pinned QKV boundary on all 100 measured
  executions, and the downstream complete layer remained bitwise exact.

Decision:

- Keep paired Q-A/KV enabled in production. The controlled 1.44% family-chain
  improvement is modest but repeatable, removes 43 transformer dispatches per
  decode token, and restores parity with pinned DwarfStar's structural path.
- Do not claim a full-engine throughput gain from the paging-distorted run.

Next:

- Re-profile the immutable fused build in a healthy memory/power state, then
  move to the remaining attention-output/HC family or Q-B geometry based on the
  refreshed stage distribution.

### 2026-08-28 — Routed-expert NSG=4 launch geometry rejected

Objective:

- Test whether the largest profiled dispatch family can improve on
  DwarfStar's two-SIMD-group IQ2/Q2 decode launch geometry on the M1 Ultra.

Implementation:

- Generalized the fused IQ2_XXS gate/up lookup-table loader so NSG=1/2/4 split
  all 256 grid and 128 sign values across the specialized threadgroup. NSG=2
  preserves the prior indexes and arithmetic exactly.
- Centralized production routed launch geometry behind one NSG value and kept
  that value at two.
- Compiled NSG=2 and NSG=4 routed pair/down pipelines in one Metal context and
  added `routed-nsg-bench`, which reverses A/B order each round, records each
  command buffer's GPU interval, and verifies routed mid/output tensors by
  exact FP32 bit pattern after every execution.

Evidence:

- `rust-star/.work/routed-nsg-tune/alternating-100.json` records 10 warmup plus
  100 measured alternating rounds per variant; SHA-256
  `007bea23bf5e1a2fe3e5b31e81a490df543dec14ebeb076a6f6ac1802cf83d1f`.
- NSG=2: 0.278 ms median GPU time, 0.004 ms MAD.
- NSG=4: 0.283 ms median GPU time, 0.004 ms MAD, 1.83% slower.
- Both variants reproduced the pinned routed-mid and routed-output fixtures on
  every execution.
- Separate-process measurements were explicitly rejected as decision evidence:
  GPU power/residency modes reversed the apparent winner, and one control run
  degraded to 0.606 steady tok/s despite exact output.

Validation:

- Optimized macOS build passed.
- All 290 Rust tests and 68 Python tests passed.
- The 100-round target-Mac diagnostic completed with exact output on every
  NSG=2 and NSG=4 execution.
- A final 30-round smoke test on the lazy diagnostic pipeline path reproduced
  the rejection (NSG=4 was 2.90% slower); evidence SHA-256
  `df61c5047b1881e2fcbf398d2088ed6c34bc1d2ed0e89c70564b6e7665ebf3fe`.

Decision:

- Keep production at NSG=2. The source now supports controlled specialization,
  but NSG=4 does not advance to an eligible full-engine comparison.
- Retain the alternating diagnostic because it prevents future launch tuning
  from being misled by the M1 Ultra's multimodal GPU timing state.

Next:

- Move to attention QKV ingress, the second-largest profiled family, and look
  for a structural optimization rather than another launch-only change.

### 2026-08-28 — Metal stage counters attribute the remaining transformer cost

Objective:

- Attribute Rust Star's steady transformer GPU interval to coarse production
  dispatch families without changing the paired-eligible scheduler.

Actions and evidence:

- Added an explicit `engine-profile` counter path with eight compute families:
  attention QKV ingress; RoPE/KV/compressors; attention cache/index routing;
  attention; attention output/HC; FFN ingress/router; routed experts; and the
  shared expert/final FFN HC update.
- The M1 Ultra reports dispatch-boundary counter sampling as unsupported. It
  supports compute-stage boundaries but rejected 43 separate counter buffers,
  so the final implementation uses one shared 688-sample timestamp buffer with
  disjoint 16-sample ranges for all 43 layer command buffers.
- Profiling is disabled by default. Enabling it splits compute encoders at the
  eight family boundaries, resolves counters only after the completed chain,
  scales counter shares to the completed per-layer command-buffer GPU
  intervals, marks the record perturbed, and makes paired eligibility false.
- The exact 2K/128 profile completed the full oracle transcript. Its steady
  transformer attribution was: routed experts 25.314% / 11.582 ms per token;
  QKV ingress 22.816% / 10.439 ms; attention output/HC 14.717% / 6.733 ms;
  attention 10.832% / 4.956 ms; RoPE/KV/compressors 10.414% / 4.765 ms; FFN
  ingress/router 7.763% / 3.552 ms; shared expert/final HC 6.662% / 3.048 ms;
  and cache/index routing 1.481% / 0.678 ms.
- Encoder splitting inflated steady transformer GPU time from 40.051 ms/token
  in the adjacent eligible control to 45.751 ms/token in the profile, a 14.23%
  perturbation. The profile's absolute milliseconds are diagnostic; its ranking
  and shares define the next search order.
- The adjacent normal `engine-measure` control retained exact selection,
  disabled correctness/layer/counter collection, remained paired eligible, and
  reported 22.522450779 steady tok/s. This is a single control, not a new paired
  claim. Its null stage report demonstrates that dormant profiling does not
  alter the production schedule.
- Private evidence is under
  `rust-star/.work/stage-profile-dispatch-family/`. Profile report SHA-256 is
  `f68064503b67b83a17e0b40e938cc79c73cdd05f15877e3f683f905870f86b8f`;
  eligible-control report SHA-256 is
  `387988ed33d20d602cbc6eda673e5f1bd736c627772829456e6e7c5a1ad45161`.

Validation:

- Optimized macOS build and all 289 Rust tests.
- All 68 Python artifact/adapter/paired-runner tests.
- Exact 128-token oracle transcript in both the perturbed profile and the
  unprofiled eligible control.
- `cargo fmt --check` and `git diff --check`.

Decision and next step:

- Start with the routed-expert family because it is the largest measured
  target. Compare the fused IQ2_XXS gate/up plus weighted SwiGLU and Q2_K
  down/sum paths against pinned DwarfStar, then isolate any candidate in a
  same-input replay before touching the eligible engine path. QKV ingress is
  the second target if routed-expert parity leaves no actionable difference.

### 2026-08-28 — Production sampling moves lowest-ID argmax onto Metal

Objective:

- Test whether Rust Star's full 129,280-logit host copy and checked CPU scan
  account for the residual 3.54% steady gap, after confirming that DwarfStar's
  greedy path performs top-1 selection on the GPU.

Actions and evidence:

- Added a finite-checked, lowest-token-ID Metal reduction to the existing
  output-head command. Production timing now reads an eight-byte token/validity
  result; full-logit and intermediate readback remain unchanged in every C0
  diagnostic.
- The complete 2K/128 engine path matched the exact 128-token oracle transcript
  in every run. All 288 optimized Rust tests passed, including the unchanged
  full-logit fixture and stable-report controls.
- Two alternating local development pairs compared the immutable `771b7c4`
  executable with the new path. Pair ratios were 1.001781148x and 1.007805722x
  steady throughput. Old/new medians were 21.793512982 and 21.897939975 tok/s,
  a provisional 1.004793435x ratio.
- Output-head wall time fell from 2.487554 to 2.262153 ms/token in the first
  pair and from 2.506579 to 2.332198 ms/token in the reversed-order pair. The
  single-threadgroup reduction added about 0.057 ms/token of GPU work, leaving
  a median wall saving of about 0.20 ms/token.
- One earlier run immediately after the large optimized link was invalid for
  performance interpretation: transformer wall time rose to roughly 1.50
  seconds/token under page pressure. Its transcript was exact, and an immediate
  clean rerun restored normal 21.861 tok/s behavior. It is retained only as
  failure evidence under `rust-star/.work/top1-output-head-01/`.
- Valid development evidence is retained under
  `rust-star/.work/top1-output-head-02/` and
  `rust-star/.work/top1-output-head-ab/`.
- Published source commit `4fc61f789e3e1df726833c32f7b1be0446c43b2a`
  with tree `53872576e334aa2f9a76c561142d5ec85320fa4f`, then copied its
  optimized executable to an immutable local candidate path. Executable
  SHA-256 is
  `22eb2bab4f6a52f34c73fad8b018695c451d3120e6900511304f4504808b3ea8`.
- Froze plan SHA-256
  `1511d56f177be34ec3aa6ee48f21b1e9fed3a04d9917c35ac5d9dec3f832c88a`.
  Both warmups and all five `AB`, `BA`, `AB`, `BA`, `AB` pairs passed with no
  adapter failures, retries, invalid attempts, capacity failures, or observed
  thermal events.
- Rust Star steady decode measured median 21.940665084 tok/s (MAD
  0.067226655, range 21.639385646--22.014262891) versus DwarfStar 22.61 tok/s
  (MAD 0.04, range 22.57--22.71). The median within-pair ratio was
  0.969791317x (MAD 0.003578508, range 0.958767640--0.973369825), leaving a
  validated 3.02% steady gap.
- Complete generation measured Rust 19.218353297 versus DwarfStar 22.06 tok/s,
  for a 0.872293133x median pairwise ratio. Prefill measured 124.056813477
  versus 181.19 tok/s, for 0.685070803x. First-token latency measured
  871.961417 versus 50.26 ms, for a 17.533320056x median pairwise ratio. This
  optimization addresses only steady sampling; it makes no prefill or
  first-token claim.
- Independent aggregation reproduced the byte-identical summary. Raw SHA-256
  is `aaac87bc1270a1be0a8767e6a1c8c8dacebf1393ae20da1a2a96ede11e5d6cf9`;
  summary SHA-256 is
  `ada1073a90e21168be375642bd0ca6943d76bcfac83fad75745db52a25b22626`.
  The local result-manifest SHA-256 is
  `e023ba3b82a8e6fdd98db40ad9ede7a9906f524bb2daaa1f8296ec142ac1f65c`.
  Full private-path evidence is under
  `rust-star/.work/paired-2k-128-4fc61f7/` and intentionally untracked.

Validation:

- Optimized macOS build and all 288 Rust tests.
- All 68 Python artifact/adapter/paired-runner tests.
- Exact retained position-8195 control across 43 layer HC states and all
  129,280 final logits; selected token 35597. Report SHA-256 is
  `165443d86ddf8187dc4bd172ca55fae17944ce397da6280eaeecb4d4258be1a9`.
- Candidate executable SHA-256 is
  `22eb2bab4f6a52f34c73fad8b018695c451d3120e6900511304f4504808b3ea8`.
- `cargo fmt --check` and `git diff --check`.

Decision and next step:

- Keep GPU top-1 as the new validated production path aligned with DwarfStar's
  greedy boundary. The five-pair steady median improved 0.30% over `771b7c4`,
  and the paired ratio improved 0.54%, but the gain is small and does not
  motivate broad unchecked Rust memory access.
- Continue attribution inside repeated transformer dispatch families; they now
  account for essentially all of the remaining 3.02% steady gap.

### 2026-08-28 — Cross-layer command-buffer grouping rejected

Objective:

- Determine whether the remaining 3.54% steady gap comes from Rust Star's 43
  transformer command buffers per generated token versus DwarfStar's larger
  batch submission.

Experiment and evidence:

- Implemented an internal production-only command-buffer grouping contract.
  Exact diagnostic paths retained one buffer per layer, while the seeded
  position-8195 complete-decoder control exercised each grouped production
  schedule. Every tested grouping remained C0 across all 43 final HC states,
  all 129,280 logits, and selected token 35597.
- One transformer command buffer was decisively slower: its eligible exact run
  measured 19.317714414 steady tok/s. Steady transformer GPU time remained
  about 40.384 ms/token, but transformer wall time rose to 49.261 ms/token
  because the GPU could not start until Rust finished encoding the complete
  1,813-dispatch transformer graph.
- Four-layer groups, eleven transformer buffers plus the output head, measured
  21.674 steady tok/s and also failed to beat the validated baseline.
- Two-layer groups, 22 transformer buffers plus the output head, measured
  22.166830346, 21.910702064, and 21.912 steady tok/s in three fresh eligible
  processes. The median is only about 0.17% above the validated 21.875065281
  tok/s baseline and lies inside observed run variation; it does not justify a
  new scheduler contract or paired run.
- Private evidence is retained under
  `rust-star/.work/single-transformer-command-*`,
  `rust-star/.work/four-layer-command-groups-*`, and
  `rust-star/.work/two-layer-command-groups-*`.

Decision and next step:

- Rejected all command-buffer grouping variants and restored the validated
  43-buffer production scheduler. Queue-ordered per-layer commits are useful:
  they overlap GPU execution of early layers with CPU encoding of later ones.
- The residual steady gap is not command-buffer commit overhead. Continue with
  kernel/dispatch-family attribution inside the repeated layer GPU work.

### 2026-08-28 — Production scheduling reaches 96.46% of DwarfStar in five exact pairs

Objective:

- Freeze the production no-snapshot schedule and determine whether its 8.76%
  profile gain survives the predeclared alternating-order paired protocol.

Actions and evidence:

- Published source commit `771b7c4ccb5beabbeb145e09a7f5fe73c582f20d`
  with tree `750c2f65da27de5afd98f3c8647f9ba9eba63dfb`, then copied its optimized
  executable to an immutable local candidate path. Executable SHA-256 is
  `56c252ab8def1b0ce71580757accb27ff3a21dd0c96cbff0fffbf4191f71e05b`.
- Froze plan SHA-256
  `e0ab758fdd751e7ef69fc9313f40745defade85585021008a7a68a89520ea51e`.
  Before the run the Mac was on AC, reported no recorded thermal/performance
  warning, had 94% free memory, zero throttled pages, and no model process.
- Both full-length warm-ups passed. Five pairs completed in the declared `AB`,
  `BA`, `AB`, `BA`, `AB` order with zero adapter failures, retries, invalid
  attempts, capacity failures, or observed thermal events. Afterward the Mac
  still reported 94% free memory, zero throttled pages, and no recorded thermal
  or performance warning.
- Rust Star steady decode measured median 21.875065281 tok/s (MAD
  0.012764947, range 21.789438485--21.887830228) versus DwarfStar 22.66 tok/s
  (MAD 0.05, range 22.59--22.74). The within-pair median was 0.964571386x
  (MAD 0.000363122, range 0.962476742--0.968059718). The remaining validated
  steady gap is 3.54%, down from 14.9% at `35ee7bb`.
- Complete generation measured Rust 19.343260789 tok/s versus DwarfStar 22.1,
  for a 0.874572570x median pairwise ratio. Prefill measured 123.187716755
  versus 180.01 tok/s, for 0.684338185x. First-token latency measured
  812.705917 versus 50.857 ms, for 15.505437619x.
- Relative to `35ee7bb`, the Rust median improved 9.20% in steady decode and
  6.50% in complete generation. Prefill changed -0.90%, while first-token
  latency worsened 14.74%; this optimization does not address either frontier.
- Independent aggregation reproduced the byte-identical summary. Raw SHA-256
  is `ac80b093057f15f0fa5ecf8a96c09fd417a0ee1050ca97176b083c00ed1af0a9`;
  summary SHA-256 is
  `1f4833ffa653921fafb639d45026efe7d218ab6277201f256ed7f55ba985c9d1`.
  The local result-manifest SHA-256 is
  `8ff7475a10ccbcee7f94372efa736304efd828905dfb1d46f34e3a310a10d70a`.
  Full private-path evidence is under
  `rust-star/.work/paired-2k-128-771b7c4/` and intentionally untracked.

Decision and next step:

- Accept 96.46% of DwarfStar as the new validated steady baseline. Rust Star
  has not yet beaten DwarfStar, but the residual is now small enough for
  dispatch-family/kernel attribution rather than broad host-side optimization.
- Profile the remaining repeated transformer GPU work without adding reads or
  waits to `engine-measure`; keep first-token residency and prefill as separate
  problems.

### 2026-08-28 — Production decode drops unused snapshots and redundant encoder boundaries

Objective:

- Attribute the remaining 14.9% paired steady-decode gap and keep the best
  production optimization only if the retained complete-decoder C0 control
  remains bit-identical.

Changes and evidence:

- Added `engine-profile`, an explicitly paired-ineligible form of the exact
  2K/128 engine workload. It reads the already completed 43 layer command
  buffers' GPU timestamps after each generated-token interval and emits the
  summed steady per-layer values. Normal `engine-measure` remains separate,
  emits `null` for the profile array, records collection as false, and remains
  paired eligible.
- Baseline profile evidence under
  `rust-star/.work/steady-layer-profile-01/` has manifest SHA-256
  `524aa791a3e853fa9049cf718207662fd3306a34a6862490d961be8ea28e0ecd`.
  It measured 20.171179318 steady tok/s, 46.960839 ms transformer wall and
  44.164538 ms transformer GPU per steady token. The hottest single layer was
  only 1.128632 ms/token or 2.56% of transformer GPU time; ratio-4 even layers
  accounted for 51.68% and ratio-128 odd layers for 44.19%. The remaining cost
  was therefore repeated schedule overhead/work, not one anomalous layer.
- Source comparison with pinned DwarfStar showed that its batch path reuses a
  compute encoder, while Rust Star's timing-only production layer still kept
  three probe-only snapshots and repeatedly ended/recreated encoders. Added an
  internal output-collection contract to the Rust/Objective-C boundary. Exact
  tensor probes preserve the old observable schedule. Production decode skips
  the Q/pre-RoPE-KV, post-RoPE-KV, and attention pre-inverse-RoPE snapshots,
  executes Q and inverse RoPE in place, and keeps adjacent FFN dispatches in
  one encoder. Required persistent compressed-cache commits still use blits and
  retain their dependency boundaries.
- Optimized profile evidence under
  `rust-star/.work/steady-layer-profile-02-inplace/` has manifest SHA-256
  `b62c868d6e46179259f619564d17084a711ec9f0c0103abfc7da613b35ec8b11`.
  It measured 21.937837562 steady tok/s, up 8.76%; transformer wall fell 8.29%
  to 43.066214 ms/token and transformer GPU fell 9.16% to 40.121048 ms/token.
  Prefill was effectively unchanged. The first-token sample worsened from
  741.225 to 795.800 ms, so this checkpoint makes no first-token improvement
  claim.
- The optimized production schedule passed the seeded position-8195 retained
  decoder control: every one of 43 16,384-float layer HC outputs and all
  129,280 logits matched DwarfStar by FP32 bit pattern, and the exact selected
  token remained 35597. The report SHA-256 is
  `2202486531c149d2666515334a357a317e03d7ebf1bbc7ee1823550ba20e4c02`
  under `rust-star/.work/production-schedule-c0-01/`.
- Two independent non-profiled `engine-measure` processes remained paired
  eligible, matched transcript checksum `17615242442502606640`, and measured
  21.863585930 and 21.769 steady tok/s. Their complete-generation rates were
  19.376402772 and 19.363778103 tok/s; prefill measured 123.294150291 and
  123.200729385 tok/s. Evidence is under
  `rust-star/.work/production-schedule-measure-01/` and intentionally untracked.

Validation:

- Optimized macOS release build.
- Retained position-8195 complete-decoder C0 control across all layer HC states
  and full logits.
- Two exact fresh-process eligible 2K/128 engine measurements.
- `cargo fmt --check` and all 288 Rust tests.
- All 68 Python artifact/adapter/paired-runner tests.
- `git diff --check` before publication.

Decision and next step:

- Keep the production-only schedule optimization. It applies the useful
  Polars-style lesson—remove redundant materialization and preserve a fixed
  execution plan—at the GPU scheduling boundary where profiling found the
  cost; no broad or unjustified `unsafe` Rust was added.
- Freeze the new commit and run a fresh five-pair alternating-order comparison.
  Until then the 21.8 tok/s development result is evidence, not a replacement
  for the last validated paired ratio.

### 2026-08-28 — Scratch-residency gain reproduces in five exact pairs

Objective:

- Freeze `35ee7bb` and determine whether the diagnostic scratch-residency gain
  survives the predeclared alternating-order paired protocol.

Changes and evidence:

- Rebuilt the optimized runtime from committed source. Candidate executable
  SHA-256 is
  `1f401310cdde2d15a127a4833269e09f4f67f1c4cb85933ebc7a2cce0201968d`;
  source tree is `59c18a01cfcb62a523aaa6baef470fb29e6ad114`.
- Captured a fresh privacy-filtered host manifest with AC power, no recorded
  thermal/performance warning, 95% free memory, zero throttled pages, and no
  model process. Froze plan SHA-256
  `ed69618ada4d5a830ed327de48f5229140f36a72843c0f3638bea79d627aef65`.
- Both full-length warm-ups passed. All five 2K/128 pairs completed in the
  declared `AB`, `BA`, `AB`, `BA`, `AB` order with zero failures, retries,
  invalid attempts, capacity failures, or observed thermal events. Post-run
  memory returned to 93% free with zero throttled pages.
- Rust Star steady decode measured median 20.031224379 tok/s (MAD
  0.018443385, range 19.987054309--20.050703952) versus DwarfStar 23.56 tok/s
  (MAD 0.06, range 22.54--23.62). The within-pair candidate/oracle median was
  0.850831888x (MAD 0.003550728, range 0.847281160--0.888696734).
- Complete generation measured Rust 18.162539840 tok/s versus DwarfStar 23.16
  tok/s. The within-pair median was 0.783881737x (range
  0.779761425--0.827919048).
- Prefill measured Rust 124.303074737 tok/s versus DwarfStar 181.83 tok/s. The
  within-pair median was 0.683350269x (range 0.681938272--0.685419985).
- First-token latency measured Rust 708.321500 ms versus DwarfStar 47.741 ms.
  The within-pair median ratio was 14.531941455x (range
  12.219232661--15.366224835).
- Relative to the prior five-pair Rust medians, scratch reuse improved steady
  decode 14.5%, complete generation 2.58x, prefill 66.9%, and first-token
  latency 93.5%. The paired steady ratio improved from 0.753377393x to
  0.850831888x. Rust Star has not yet beaten DwarfStar.
- Finalization rehashed every measurement. Independent aggregation reproduced
  the byte-identical summary. Raw SHA-256 is
  `139561e3dd048a0a5d504e841aa8cd09e254dc22b42e39a5c063b88f682fa970`;
  summary SHA-256 is
  `b0eb0563e5bf8ce0570650dc75c955c065b9259e3a1a41ad5cdf018670d329df`.
  Complete private-path evidence remains under
  `rust-star/.work/paired-2k-128-35ee7bb/` and is intentionally untracked.
- Internal timing medians place steady transformer execution at 47.247 ms wall
  and 44.693 ms summed GPU per token. The output head costs 2.666 ms wall and
  1.386 ms GPU per token. The first generated step is 705.921/688.398 ms in the
  transformer but only 2.875/2.190 ms in the output head, so both remaining
  gaps are transformer/device-residency problems rather than output sampling.

Performance-engineering note:

- The suggested animal-branded Rust project was likely Polars. Its transferable
  advantages are columnar/contiguous data, zero-copy Arrow views, fixed query
  plans, SIMD, parallelism, and eliminating allocations and redundant work;
  `unsafe` supports a few proven low-level primitives but is not itself the
  source of speed. Rust Star should continue applying the analogous zero-copy,
  fixed-shape, buffer-lifetime, and scheduled-work principles. Test
  `-C target-cpu=native` and unchecked host loops only as isolated profiled
  candidates. Keep unsafe surfaces narrow, document their caller invariants,
  and retain a safe/C0 differential control; current timing shows Metal GPU
  work dominates, so broad unsafe Rust changes would not close the main gap.

Decision and next step:

- Keep and publish the scratch-residency checkpoint and its paired evidence.
  Next add timing-only steady per-layer/kernel attribution, then optimize the
  largest Metal schedule or memory-traffic contributor. Separately trace the
  residual first-token weight-residency cliff without shifting warm-up work out
  of a declared inference interval.

### 2026-08-28 — Timing-only prefill scratch reuse removes decoder residency cliff

Objective:

- Explain and remove Rust Star's 10.95-second first generated step without
  changing arithmetic, weakening C0, or instrumenting the timed paired runs.

Changes and evidence:

- Added DwarfStar-equivalent mapped-weight warming before internal inference
  timers: `POSIX_MADV_WILLNEED` plus one volatile byte touch per tensor-data
  page. The raw engine report now records warm bytes, pages, checksum, and wall
  time so residency policy is explicit and independently auditable.
- Added untimed diagnostic breakdowns for decoder preparation, first and steady
  transformer wall/GPU time, output-head wall/GPU time, and the first token's
  43 completed per-layer command-buffer GPU intervals. Reading those timestamps
  submits no work and occurs after the first-token stopwatch.
- The diagnostic run showed early decoder layers taking roughly 676--701 ms
  each while late layers took roughly 23--29 ms. Native timing-only prefill had
  encoded the entire 43-layer schedule into one command buffer while retaining
  distinct multi-row scratch for every layer. That working set consumed the
  remaining 128 GB unified-memory headroom and evicted early mapped weights in
  layer order; the first decode then faulted them back in sequentially.
- Production native prefill now reuses transient buffers by semantic role for
  layers 3--42. The final HC buffer uses two parity slots because the next layer
  consumes it. Raw KV, compressed attention/indexer rows, and their score/KV
  states remain unique per layer. Diagnostic collection still allocates every
  layer boundary independently. After the decoder-state blit completes, the
  prefill properties and pooled activation-cache entries are released.
- Fresh-process run `rust-star/.work/rust-star-ctx-2048-warm-04` remained exact
  and reported 125.377945557 prefill tok/s, 833.638750 ms first token,
  20.554818230 steady tok/s, and 18.253799463 complete-generation tok/s. Its
  first-token per-layer median was 20.472 ms and maximum 73.675 ms.
- Fresh-process repeat `rust-star/.work/rust-star-ctx-2048-warm-05` reproduced
  the exact 128-token checksum `17615242442502606640` and reported
  129.046988121 prefill tok/s, 732.983916 ms first token, 20.788667769 steady
  tok/s, and 18.707757287 complete-generation tok/s. Its first-token per-layer
  median was 20.113 ms and maximum 28.771 ms.
- Relative to the earlier five-pair Rust medians, the repeat is about 14.9x
  faster on first-token latency, 18.8% faster on steady decode, 2.66x faster on
  complete generation, and 73.3% faster on prefill. This is diagnostic evidence,
  not a paired headline result. Rust Star is now approximately 9.4% behind the
  earlier DwarfStar steady median of 22.95 tok/s.
- The full Mac regression resumed after an interruption and passed every
  remaining exact model gate: 2K live-KV, layer-2 KV norm/state, compressor,
  exact native 2K prefill across all 1,216 no-copy mappings, full 43-layer
  decode/output head, four-token closed loop, position 127, native prefill
  handoff through position 4099, sparse/multimerge controls, and retained
  position 8195. The complete Rust suite passed 287 tests; the Python suite
  passed all 68 tests; formatting, optimized Mac compilation, and the runtime
  model gates are green.

Decision and next step:

- Keep the C0-safe scratch lifetime optimization. Commit and freeze this
  candidate, then rerun the unchanged five-pair 2K/128 protocol. If reproduced,
  optimize the remaining steady transformer GPU/command path; the output head
  is no longer the dominant gap.

### 2026-08-27 — First checkpointed C0 paired 2K/128 comparison complete

Objective:

- Replace single-run bring-up numbers with the predeclared five-pair,
  alternating-order development comparison against the pinned DwarfStar oracle.

Changes and evidence:

- Corrected the paired runner's warm-up to use the plan's declared generation
  length. The prior hardcoded eight-token warm-up was incompatible with the
  intentionally narrow exact 2K/128 candidate. The protocol requires an untimed
  smallest-context warm-up but does not require a shorter generation. The
  checkpoint/resume test now verifies both warm-up artifacts use 128 tokens.
- Committed that runner-only change as `983d5f3`, rebuilt the unchanged Rust
  runtime release binary, and passed all 68 Python tests. The candidate
  executable SHA-256 remained
  `0754a2e3edf8563002b4d83d8642f1aef85bc7426f1d2dd2109a95ecc1975822`.
- Froze an immutable local plan with SHA-256
  `4eeee3a90e830ccfc446e6762a8ad55a34c1792b3d484b2ef6f702f5665b81da`.
  It binds source commits/trees, exact executable/model/prompt hashes, explicit
  build/runtime configurations, the privacy-filtered host manifest, and the C0
  correctness manifest.
- Both full-length warm-ups passed. All five timed 2K/128 pairs completed in
  alternating `AB`, `BA`, `AB`, `BA`, `AB` order with zero failures, retries,
  invalid attempts, capacity events, or recorded macOS thermal/performance
  warnings.
- Rust Star steady decode had median 17.493423061 tok/s (MAD 0.425360410,
  range 17.068062651--19.317236451) versus DwarfStar 22.95 tok/s (MAD 0.27,
  range 22.64--24.03). The within-pair candidate/oracle median was 0.753377393x
  (MAD 0.005669741, range 0.747707652--0.803880002).
- Complete generation, which includes the first generated step, measured a
  paired median 0.308332588x (range 0.298489542--0.319406508). Rust Star's
  median was 7.041875502 tok/s versus DwarfStar's 22.60 tok/s.
- Prefill measured a paired median 0.413274711x (range
  0.375442069--0.493098704). Rust Star's raw median was 74.480368506 tok/s
  versus DwarfStar's 178.29 tok/s.
- Rust Star's first generated step took a 10,952.916750 ms median versus
  DwarfStar's 49.088 ms. The within-pair latency ratio was 225.843754304x
  (range 210.176285188--237.544243664), explaining why its complete-generation
  ratio is far below its steady-state ratio.
- Finalization rehashed every measurement. Independent aggregation reproduced
  the byte-identical summary. Raw SHA-256 is
  `b8daf73a1ef556d9181e0100fe9dadec0c6d65c14d798121440d1e5548571997`;
  summary SHA-256 is
  `7ebc757a361584de75ae452ce757a4d683a0b5343f709948ad4b40c6c7ed5c78`.
  Complete private-path evidence remains under
  `rust-star/.work/paired-2k-128-983d5f3/` and is intentionally untracked.

Decision:

- This is valid C0 development evidence at 2K, not the protocol's publishable
  256K primary claim. Rust Star is currently slower at every user-visible
  metric. Preserve the negative result and optimize from its measured shape:
  first remove the cold decoder-transition penalty, then improve native prefill
  and the remaining roughly 25% steady-decode gap.

Next:

- Add a correctness-neutral diagnostic around the first post-prefill decoder
  step to distinguish resource/pipeline warm-up and weight residency from real
  transformer/output-head execution. Keep the paired interval unchanged and
  rerun the five-pair plan only after a measured fix.

### 2026-08-27 — Timing-only native prefill reached an eligible 2K/128 measurement

Objective:

- Remove exhaustive prefill boundary collection from measured execution while
  preserving the exact complete-model C0 command unchanged.

Changes and evidence:

- Added an explicit `collect_outputs` boundary to both native prefill Metal
  ABIs. Diagnostic callers still validate every output pointer and copy every
  retained tensor; timing callers execute the same command schedules while
  retaining only GPU-owned decoder state.
- The Rust timing path does not decode diagnostic output fixtures or allocate
  their host tensors. It transfers only final output-head logits for mandatory
  lowest-ID argmax and validates the complete selected-token transcript after
  all generation intervals.
- The unchanged `prefill-layers012-attention-loop-probe` passed on the M1 Ultra:
  2,372 transformer dispatches, 1,216/1,216 no-copy model ranges, 17,772.379 ms
  wall time, exact full logits, and selected token 15,342. Its JSON SHA-256 is
  `df1c7eedb4e1a7079c470d1575618088239aa113b1ad83e9c1a02e413f4f7577`.
- The first timing attempt failed safely before Metal work on one remaining
  diagnostic raw-cache slice. The guarded retry passed `measure_ruststar.py`
  with both collection flags false, 44 command buffers and two waits per
  generated token, and the exact 128-token oracle transcript.
- The clean eligible single-run artifact reported 90.668305877 prefill tok/s,
  7.775007935 generation tok/s including the first step, and 18.406557599
  steady tok/s. Intervals were 22,587.826917 ms prefill, 9,563.290792 ms first
  generation step, and 6,899.714915 ms for the remaining 127 tokens. The normalized
  measurement SHA-256 is
  `6fb9f647ff6926b784f2dd377672f29577181afec8792c3621205bbe8714bca6`.
- The adapter recorded 73,873.229625 ms complete-process wall time and
  121,602,048 bytes child peak RSS. This is the adapter's host RSS field, not a
  claim about total Metal/unified-memory residency.
- The complete host-runtime gate passed: formatting, 287 Rust tests, optimized
  macOS build, the M1 Ultra dispatch probe, 68 Python tests, every pinned
  differential fixture, the Rust/Python artifact smoke test, all native prefill
  controls, the full 2K-to-position-4099 handoff, retained sparse controls, and
  the steady-state diagnostic benches.

Decision:

- The raw producer may now set `paired_protocol_eligible: true`; the adapter
  still fails closed if either collection flag, transcript, scheduler metadata,
  or blocker field is inconsistent. The exact C0 command remains an independent
  regression control and is not replaced by transcript validation.

Next:

- Run the checkpointed alternating-order DwarfStar/Rust Star A/B plan at
  2K/128. Treat this single Rust Star run as bring-up evidence, not a comparative
  performance conclusion.

### 2026-08-27 — Rust Star measurement boundary added; prefill collection isolated as blocker

Objective:

- Connect the exact native-prefill/decoder path to the engine-neutral paired
  process contract without reporting correctness-oriented intervals as speed.

Changes and evidence:

- Added `engine-measure` and stable `rust-star-engine-run-v1` output for the
  exact 2,048-token prefill plus 128 committed greedy tokens.
- The generation loop prepares model bindings outside timing, includes command
  encoding, 44 synchronized command buffers, two waits, full output-head
  sampling, and token commitment per step, and compares all selected tokens
  with the pinned oracle only after timing.
- Added `measure_ruststar.py` and its library. The fresh-process adapter records
  wall time and child peak RSS, sanitizes logs, checksums evidence, normalizes
  rates, and refuses inconsistent eligibility, collection, transcript, or
  scheduler metadata.
- Unit coverage exercises successful normalization, rate inconsistency,
  explicit ineligibility, and an eligibility flag contradicted by prefill
  collection metadata.
- The complete host-runtime gate passed 287 Rust tests, an optimized macOS
  build, the M1 Ultra Metal dispatch probe, 68 Python tests, every pinned
  differential fixture, and the Rust-writer/Python-reader C0 smoke test.
- The first optimized target-Mac 2K/128 development run reached Metal generation.
  A live process sample measured 22.3 GB current physical footprint and 66.9 GB
  peak footprint. The peak came from exhaustive native-prefill fixtures and
  host output materialization. The run ended before atomic JSON installation,
  so no throughput number is retained or claimed.

Decision:

- Keep the raw producer explicitly `paired_protocol_eligible: false` and make
  the adapter fail closed. The next change must split timing-only native prefill
  from the exact diagnostic collector rather than weakening C0 validation or
  publishing a memory-distorted rate.

Next:

- Give the prefill Metal ABI an execution-only path with no host boundary
  outputs, construct model descriptors without decoding fixtures, retain the
  existing correctness command unchanged, and repeat the exact 2K/128 run.

### 2026-08-27 — Exact native 2K handoff through production sparse position 4099

Objective:

- Transfer the retained exact batched-prefill state into the ordinary decoder
  representation without host readback, then prove the complete closed loop
  through the first default sparse ratio-4 step.

Evidence:

- Added a pinned-oracle capture that performs exact 2,048-token
  `ds4_session_sync`, then greedily evaluates positions 2048--4099. Two fresh
  processes produced the same 2,052 input-token transcript, immediate logits,
  final logits, and 51,659,152-byte retained prefill payload bit-for-bit.
- The compact fixture retains only the transcript and the two 129,280-logit
  frontiers. The full retained payload remains local diagnostic evidence and
  is not imported into the repository.
- Added one GPU-only adoption ABI. It copies all 43 raw-ring tails, layers
  2--42 attention compressed/recurrent state, and all even-layer indexer
  compressed/recurrent state in 229 blits under one command buffer and one
  host wait.
- The optimized M1 Ultra run matched the native prefill token 15342, every one
  of the 2,052 greedy selections, all position-2048 logits, and all
  position-4099 logits. At position 4099 all 21 even layers observed newly
  committed compressed row 1,025 and used the production two-block top-512
  sparse schedule. Input token 312 selected token 2538.
- The run reported 21,803.054/21,712.993 ms transformer wall/GPU time,
  262.624/2.705 ms output-head wall/GPU time, 78.904/0.207 ms handoff wall/GPU
  time, and 16.563 synchronized decode positions/s. These are correctness
  diagnostics with per-position waits and full frontier checks, not an engine
  throughput claim.

Decision:

- Native batched prefill, persistent-state ownership, exact post-prompt decode,
  and the first production sparse frontier are now one continuous C0 path.
  Captured initial state is no longer used by this control.

Next:

- Build a timing-specific 128-token engine-measurement producer from this
  exact state path, validate its artifact, and connect it to the paired runner.

### 2026-08-27 — Exact complete-model native 2K prefill through logits

Objective:

- Consume the retained final layer-42 HC row directly from the native batched
  prefill context and complete the exact output head without weakening any
  transformer-only regression boundary.

Evidence:

- Reused the independently repeated DwarfStar 2K batched-prefill frontier
  fixture. Its 129,280-logit payload has SHA-256
  `7b5e851884bbb0aa8c2a249c8497af0feccb267cbd0a40e0a4a5aee584ecbfaf`
  and selects token 15342 by lowest-ID argmax.
- Extended the proven decode output-head implementation to select either its
  retained one-row layer-42 buffer or row 2047 of the retained 2K prefill
  layer-42 HC buffer. Decode ownership and all prior transformer controls
  remain unchanged.
- The exact plain HC RMSNorm, F16 four-way HC projection, HC weighting, fused
  HC collapse/learned RMSNorm, and 129,280-row Q8_0 vocabulary projection add
  five dispatches and five pointer-identical model mappings. The complete
  native model schedule is now 2,377 dispatches and 1,221 no-copy mappings.
- The optimized focused M1 run matched every transformer boundary and every
  output logit bit-for-bit, selected token 15342, and reported 21478.015 ms
  wall / 21389.015 ms GPU for the terminal transformer schedule plus
  270.931 ms wall / 2.697 ms GPU for the correctness-readback output head.
  These intervals are correctness diagnostics, not throughput claims.
- The repeated target-model gate again matched all 129,280 logits and selected
  token 15342. It reported 32159.143 ms wall / 31397.883 ms GPU for the
  transformer schedule and 69.829 ms wall / 2.700 ms GPU for the output head.
  All 284 Rust tests, 62 Python tests, the complete pinned fixture corpus,
  strict target-model inspection, and every native runtime control passed.
- The retained sequential-replay diagnostic remained exact against its decode
  oracle at 21.708 tokens/s over 94341.601 ms. Its intentional divergence from
  batched-prefill arithmetic still makes it ineligible for paired throughput.

Decision:

- Native batched model prefill is now exact and complete through full logits
  and deterministic greedy selection at the canonical 2K prompt boundary.
  Sparse post-prompt integration and an eligible throughput path remain
  outside this claim.

Next:

- Connect the retained native 2K state to closed-loop decoding and integrate
  the production-default sparse ratio-4 path before emitting paired engine
  measurements.

### 2026-08-27 — Exact complete layer-42 full-2K transformer prefill

Objective:

- Carry layer 41's retained final HC through the final transformer layer's
  complete native prefill path and validate the even-layer paired ratio-4
  attention/indexer compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-42 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, both
  ratio-4 compressors and recurrent states, dense mixed attention, FFN, and
  both additive HC updates, chained from the layer-41 complete fixture.
  Together they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-42 mappings and
  70 dispatches, taking the terminal schedule to 2,372 dispatches and
  1,216/1,216 pointer matches.
- The optimized focused M1 correctness run matched every retained tensor and
  the full layer-42 attention/HC checksums bit-for-bit, reporting 20845.548 ms
  wall / 20741.050 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- The complete target-model regression repeated the 2K layers 0–42 schedule
  at 35813.130 ms wall / 35048.038 ms GPU with all 2,372 terminal dispatches
  and 1,216/1,216 no-copy mappings intact. The separate sequential frontier
  remained decode-replay C0 exact at 21.283 tokens/s over 96226.723 ms; its
  captured-state schedule is explicitly ineligible for a throughput claim.
- All 284 Rust tests and all 62 Python tests passed. The complete pinned
  differential corpus, all four new fixture bundles, strict target-model
  inspection, retained sparse controls, decoder/logit controls, and runtime
  benchmarks passed the one-command Mac Studio gate.

Decision:

- The exact native full-2K transformer prefill frontier is now complete
  through all layers 0–42. Output-head integration, complete-model native
  batched prefill, output logits, and a throughput-producing path remain
  outside this claim.

Next:

- Integrate the exact output head with the retained layer-42 final HC while
  preserving every transformer boundary as a regression control.

### 2026-08-27 — Exact complete layer-41 full-2K prefill

Objective:

- Carry layer 40's retained final HC through layer 41's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-41 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 compressor and recurrent state, dense mixed attention, FFN, and
  both additive HC updates, chained from the layer-40 complete fixture.
  Together they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-41 mappings and
  47 dispatches, taking the terminal schedule to 2,302 dispatches and
  1,184/1,184 pointer matches.
- The optimized focused M1 correctness run matched every retained tensor and
  the full layer-41 attention/HC checksums bit-for-bit, reporting 31283.057 ms
  wall / 30976.413 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- The complete target-model gate repeated the boundary at 32546.263 ms wall /
  31779.314 ms GPU with the same exact dispatch and mapping counts.
- All 279 Rust tests and all 62 Python tests passed. All four new fixture
  bundles and the complete pinned fixture corpus passed independent manifest,
  size, and SHA-256 verification.
- The unchanged 2K sequential diagnostic reproduced the decode-replay logits
  exactly over 94268.320 ms at 21.725 tokens/s. This remains correctness
  evidence rather than a throughput claim for the native batched path.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 41.
  Layer 42, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 42 while preserving all
  retained layers-0–41 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-40 full-2K prefill

Objective:

- Carry layer 39's retained final HC through layer 40's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-40 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, both
  ratio-4 compressors and their recurrent states, dense mixed attention, FFN,
  and both additive HC updates, chained from the layer-39 complete fixture.
  Together they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-40 mappings and
  70 dispatches, taking the terminal schedule to 2,255 dispatches and
  1,156/1,156 pointer matches.
- The optimized focused M1 correctness run matched every retained tensor and
  the full layer-40 attention/HC checksums bit-for-bit, reporting 17261.744 ms
  wall / 17169.696 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- All 275 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The complete target-Mac gate independently repeated the exact layer-40
  boundary at 25921.658 ms wall / 25162.075 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logit controls, and both benchmark smokes. Its 2K sequential
  diagnostic reproduced the final decode-replay logits exactly over
  96718.190 ms at 21.175 tokens/s. These are correctness diagnostics, not
  throughput claims.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 40.
  Layer 41, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 41 while preserving all
  retained layers-0–40 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-39 full-2K prefill

Objective:

- Carry layer 38's retained final HC through layer 39's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-39 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 attention compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-38 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-39 mappings and
  47 dispatches, taking the terminal schedule to 2,185 dispatches and
  1,124/1,124 pointer matches.
- The first focused M1 preflight caught two mechanically shifted quantized
  row-size literals before execution. After restoring the correct 1,056-byte
  row size, the optimized correctness run matched every retained tensor and
  the full layer-39 attention/HC checksums bit-for-bit, reporting 15557.383 ms
  wall / 15471.540 ms GPU. This includes exhaustive correctness readback and
  is not a throughput claim.
- All 270 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The complete target-Mac gate independently repeated the exact layer-39
  boundary at 24440.831 ms wall / 23671.163 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 96062.015 ms at 21.320 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 39.
  Layer 40, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 40 while preserving all
  retained layers-0–39 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-38 full-2K prefill

Objective:

- Carry layer 37's retained final HC through layer 38's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-38 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, both
  ratio-4 compressors and their recurrent states, dense mixed attention, FFN,
  and both additive HC updates, chained from the layer-37 complete fixture.
  Together they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-38 mappings and
  70 dispatches, taking the terminal schedule to 2,138 dispatches and
  1,096/1,096 pointer matches.
- The first focused M1 run localized four stale layer-36 compressor projection
  indices in the cloned layer-38 schedule. After correcting them, the
  optimized correctness run matched every retained tensor and the full
  layer-38 attention/HC checksums bit-for-bit, reporting 20538.412 ms wall /
  19774.631 ms GPU. This includes exhaustive correctness readback and is not a
  throughput claim.
- All 266 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The complete target-Mac gate independently repeated the exact layer-38
  boundary at 21674.890 ms wall / 21576.655 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 97252.434 ms at 21.059 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 38.
  Layer 39, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 39 while preserving all
  retained layers-0–38 boundaries as regression controls.

### 2026-08-27 — Exact complete layer-37 full-2K prefill

Objective:

- Carry layer 36's retained final HC through layer 37's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-37 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved oracle
  executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 attention compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-36 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-37 mappings and
  47 dispatches, taking the terminal schedule to 2,068 dispatches and
  1,064/1,064 pointer matches.
- All 261 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-37 attention/HC checksums bit-for-bit, reporting
  15705.615 ms wall / 14994.391 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-37
  boundary at 18044.576 ms wall / 17293.729 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 92544.759 ms at 22.130 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 37.
  Layer 38, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 38 while preserving all
  retained layers-0–37 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-36 full-2K prefill

Objective:

- Carry layer 35's retained final HC through layer 36's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-36 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved
  `b030961` oracle executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-35 complete fixture. Together
  they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-36 mappings and
  70 dispatches, taking the terminal schedule to 2,021 dispatches and
  1,036/1,036 pointer matches.
- All 257 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-36 attention/HC checksums bit-for-bit, reporting
  24203.696 ms wall / 24099.362 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-36
  boundary at 27551.388 ms wall / 26790.433 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 97424.253 ms at 21.021 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 36.
  Layer 37, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 37 while preserving all
  retained layers-0–36 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-35 full-2K prefill

Objective:

- Carry layer 34's retained final HC through layer 35's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-35 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch. The capture used the preserved
  `b030961` oracle executable with SHA-256 `55a39062aa8a88c7301f0992dc23a44157e5327137371204822ae5a48e213c51`,
  isolated from the separately active DSpark worktree.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, the
  ratio-128 attention-compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-34 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-35 mappings and
  47 dispatches, taking the terminal schedule to 1,951 dispatches and
  1,004/1,004 pointer matches.
- All 252 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-35 attention/HC checksums bit-for-bit, reporting
  31809.591 ms wall / 31050.835 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-35
  boundary at 25011.323 ms wall / 24213.220 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 94789.051 ms at 21.606 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 35.
  Layer 36, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 36 while preserving all
  retained layers-0–35 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-34 full-2K prefill

Objective:

- Carry layer 33's retained final HC through layer 34's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-34 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, all ten measurement CSVs were complete, and no capture
  log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-33 complete fixture. Together
  they retain 53,134,848 verified bytes.
- Extended the persistent Metal context with 32 no-copy layer-34 mappings and
  70 dispatches, taking the terminal schedule to 1,904 dispatches and 976/976
  pointer matches.
- All 248 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-34 attention/HC checksums bit-for-bit, reporting
  14317.612 ms wall / 13546.455 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-34
  boundary at 19871.581 ms wall / 19730.470 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 95319.280 ms at 21.486 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 34.
  Layer 35, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 35 while preserving all
  retained layers-0–34 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-33 full-2K prefill

Objective:

- Carry layer 32's retained final HC through layer 33's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-33 tensors from ten sequential fresh DwarfStar processes
  over the canonical 2,048-token prompt. Every first/second capture pair was
  bitwise identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 attention-compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-32 complete fixture. Together
  they retain 52,299,264 verified bytes.
- Extended the persistent Metal context with 28 no-copy layer-33 mappings and
  47 dispatches, taking the terminal schedule to 1,834 dispatches and 944/944
  pointer matches.
- All 243 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-33 attention/HC checksums bit-for-bit, reporting
  13126.804 ms wall / 12632.873 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-33
  boundary at 15919.880 ms wall / 15796.966 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes. The sequential diagnostic reproduced decode-replay logits exactly
  over 93650.728 ms at 21.868 tokens/s; that is correctness evidence, not a
  native batched-prefill throughput measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 33.
  Layer 34, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 34 while preserving all
  retained layers-0–33 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-32 full-2K prefill

Objective:

- Carry layer 31's retained final HC through layer 32's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-32 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-31 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-32 mappings and
  70 dispatches, taking the terminal schedule to 1,787 dispatches and 916/916
  pointer matches.
- All 239 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-32 attention/HC checksums bit-for-bit, reporting
  14678.233 ms wall / 14571.577 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-32
  boundary at 13283.525 ms wall / 13179.070 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 32.
  Layer 33, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 33 while preserving all
  retained layers-0–32 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-31 full-2K prefill

Objective:

- Carry layer 30's retained final HC through layer 31's complete native
  prefill path and validate its odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-31 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-30 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-31 mappings and
  47 dispatches, taking the terminal schedule to 1,717 dispatches and 884/884
  pointer matches.
- All 234 Rust tests and all 62 Python tests passed. All four new fixture
  bundles passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-31 attention/HC checksums bit-for-bit, reporting
  16492.572 ms wall / 15726.058 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-31
  boundary at 9087.549 ms wall / 8984.633 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 31.
  Layer 32, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 32 while preserving all
  retained layers-0–31 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-30 full-2K prefill

Objective:

- Carry layer 29's retained final HC through layer 30's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-30 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-29 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-30 mappings and
  70 dispatches, taking the terminal schedule to 1,670 dispatches and 856/856
  pointer matches.
- All 230 Rust tests passed. All four new fixture bundles passed independent
  manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-30 attention/HC checksums bit-for-bit, reporting
  12541.042 ms wall / 12450.093 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-30
  boundary at 8843.842 ms wall / 8740.028 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 30.
  Layer 31, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 31 while preserving all
  retained layers-0–30 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-29 full-2K prefill

Objective:

- Carry layer 28's retained final HC through layer 29's complete native
  prefill path and validate its odd-layer ratio-128 compressor against
  DwarfStar.

Evidence:

- Captured 28 layer-29 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture process failed.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-28 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-29 mappings and
  47 dispatches, taking the terminal schedule to 1,600 dispatches and 824/824
  pointer matches.
- All 225 Rust tests passed. All four new fixture bundles passed independent
  manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-29 attention/HC checksums bit-for-bit, reporting
  13595.215 ms wall / 13486.902 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-29
  boundary at 8502.094 ms wall / 8380.564 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 29.
  Layer 30, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 30 while preserving all
  retained layers-0–29 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-28 full-2K prefill

Objective:

- Carry layer 27's retained final HC through layer 28's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-28 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-27 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-28 mappings and
  70 dispatches, taking the terminal schedule to 1,553 dispatches and 796/796
  pointer matches.
- Kept the layer-28 oracle payloads runtime-loaded so the growing fixture
  corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- All 221 Rust tests and 62 Python tests passed. All four new fixture bundles
  passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-28 attention/HC checksums bit-for-bit, reporting
  23550.771 ms wall / 22839.249 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-28
  boundary at 14864.326 ms wall / 14745.832 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 28.
  Layer 29, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 29 while preserving all
  retained layers-0–28 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-27 full-2K prefill

Objective:

- Carry layer 26's retained final HC through layer 27's complete native
  prefill path and validate its odd-layer ratio-128 compressor against
  DwarfStar.

Evidence:

- Captured 28 layer-27 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-26 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-27 mappings and
  47 dispatches, taking the terminal schedule to 1,483 dispatches and 764/764
  pointer matches.
- Kept the layer-27 oracle payloads runtime-loaded so the growing fixture
  corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- All 216 Rust tests and 62 Python tests passed. All four new fixture bundles
  passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-27 attention/HC checksums bit-for-bit, reporting
  20828.778 ms wall / 20099.925 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-Mac gate independently repeated the exact layer-27
  boundary at 9444.319 ms wall / 9320.928 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, the exact 43-layer
  decoder/logits path, the 2K sequential diagnostic, and both benchmark
  smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 27.
  Layer 28, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 28 while preserving all
  retained layers-0–27 boundaries as regression controls.

### 2026-08-26 — Exact complete layer-26 full-2K prefill

Objective:

- Carry layer 25's retained final HC through layer 26's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-26 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates, chained from the layer-25 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-26 mappings and
  70 dispatches, taking the terminal schedule to 1,436 dispatches and 736/736
  pointer matches.
- Kept the layer-26 oracle payloads runtime-loaded so the growing fixture
  corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- All 212 Rust tests and 62 Python tests passed. All four new fixture bundles
  passed independent manifest, size, and SHA-256 verification.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full layer-26 attention/HC checksums bit-for-bit, reporting
  18893.506 ms wall / 18128.441 ms GPU. This includes exhaustive correctness
  readback and is not a throughput claim.
- The complete target-model gate independently repeated the exact layer-26
  boundary at 6421.745 ms wall / 5754.788 ms GPU, then passed the complete
  fixture corpus, all retained sparse controls, exact 43-layer decoder/logits,
  the 2K sequential diagnostic, and both benchmark smokes.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 26.
  Layer 27, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 27 while preserving all
  retained layers-0–26 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-25 full-2K prefill

Objective:

- Carry layer 24's retained final HC through layer 25's complete native
  prefill path and validate the odd-layer ratio-128 compressor against
  DwarfStar.

Evidence:

- Captured 28 layer-25 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four SHA-256-pinned differential fixtures covering Q/KV,
  ratio-128 compressor state, dense mixed attention, FFN, and both additive
  HC updates, chained from the layer-24 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-25 mappings and
  47 dispatches, taking the terminal schedule to 1,366 dispatches and 704/704
  pointer matches.
- Crossing layer 25 initially pushed embedded Rust fixture metadata beyond
  Mach-O's 4-GiB section limit. Layer-25 oracle payloads now load from their
  pinned fixture files at probe startup, preserving exact validation while
  removing that binary-growth failure.
- All 207 optimized Rust tests passed. The M1 Ultra focused correctness run
  matched every retained tensor and full attention/HC checksum bit-for-bit,
  reporting 14256.520 ms wall / 13414.574 ms GPU. This includes exhaustive
  correctness readback and is not a throughput claim.
- The complete target-model gate independently repeated the exact layer-25
  boundary at 12407.395 ms wall / 12278.251 ms GPU, then passed all 62 Python
  tests, the complete fixture corpus, the 2K sequential diagnostic, retained
  sparse controls, exact position-8195 decoder/logits, and both benchmark
  smokes. The 2K diagnostic was resumed after an operator interruption and
  completed in 126413.791 ms without rerunning the already-passed earlier
  stages.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 25.
  Layer 26, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the exact full-2K frontier through layer 26 while preserving all
  retained layers-0–25 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-24 full-2K prefill

Objective:

- Carry layer 23's retained final HC through layer 24's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-24 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned and chained
  from the layer-23 complete fixture.
- Extended the persistent Metal context with 32 no-copy layer-24 model
  mappings and 70 dispatches, taking the complete terminal schedule to 1,319
  dispatches and 676/676 pointer matches.
- Kept the pinned 32-MiB full attention oracle runtime-loaded so the growing
  fixture corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- The focused gate initially exposed a mechanical fixture-name swap: the
  correct retained layer-22 ingress was compared against layer 24's oracle.
  Correcting the two constant names restored the retained regression without
  changing GPU execution.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 8274.750 ms
  wall and 8160.218 ms GPU; this is a correctness timing with exhaustive
  readback, not a throughput claim.
- The complete target-model gate repeated the boundary at 7757.827 ms wall /
  7637.127 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 203 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  100.1-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 24.
  Layer 25, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 25 while preserving all
  retained layers-0–24 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-23 full-2K prefill

Objective:

- Carry layer 22's retained final HC through layer 23's complete native
  prefill path and validate its ratio-128 compressor against DwarfStar.

Evidence:

- Captured 28 layer-23 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical, and no capture log contained an error or mismatch.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention compressor state, dense mixed attention, FFN, and both additive
  HC updates. Their complete captures are SHA-256-pinned and chained from the
  layer-22 complete fixture.
- Extended the persistent Metal context with 28 no-copy layer-23 model
  mappings and 47 dispatches, taking the complete terminal schedule to 1,249
  dispatches and 644/644 pointer matches.
- Kept the pinned 32-MiB full attention oracle runtime-loaded so the growing
  fixture corpus remains below Mach-O's 4-GiB Rust metadata-section limit.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 11244.205 ms
  wall and 10918.043 ms GPU; this is a correctness timing with exhaustive
  readback, not a throughput claim.
- The complete target-model gate repeated the boundary at 7274.859 ms wall /
  7166.631 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 198 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  95.3-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 23.
  Layer 24, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 24 while preserving all
  retained layers-0–23 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-22 full-2K prefill

Objective:

- Carry layer 21's retained final HC through layer 22's complete native
  prefill path and validate both even-layer ratio-4 compressors independently
  against DwarfStar.

Evidence:

- Captured 31 layer-22 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-22 model
  mappings and 70 dispatches, taking the complete terminal schedule to 1,202
  dispatches and 616/616 pointer matches.
- The added full attention oracle crossed Mach-O's 4-GiB Rust metadata-section
  limit when embedded. Loading that pinned 32-MiB tensor from its versioned
  fixture at runtime keeps the exact bitwise comparison while restoring clean
  test builds.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 15146.208 ms
  wall and 15048.284 ms GPU; this is a correctness timing with exhaustive
  readback, not a throughput claim.
- The complete target-model gate repeated the boundary at 15966.815 ms wall /
  15175.116 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 194 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  98.7-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 22.
  Layer 23, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 23 while preserving all
  retained layers-0–22 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-21 full-2K prefill

Objective:

- Carry layer 20's retained final HC through layer 21's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-21 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-21 model mappings
  and 47 dispatches, taking the complete terminal schedule to 1,132 dispatches
  and 584/584 pointer matches.
- The first target-Mac run exposed four layer-21 weight references that had
  retained layer-19 mapping indices during mechanical rebasing. Correcting
  those indices made every retained tensor and full attention/HC checksum match
  bit-for-bit in two independent optimized M1 Ultra replays.
- The focused replay reported 6453.507 ms wall / 6312.943 ms GPU and the second
  replay reported 5785.641 ms wall / 5695.444 ms GPU. These are correctness
  timings with exhaustive readback, not throughput claims.
- The complete target-model gate repeated the boundary at 6325.002 ms wall /
  6208.810 ms GPU with the same exact dispatch and mapping counts.
- Complete target-Mac validation passed formatting, optimized
  Objective-C/Metal compilation, all 189 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse control, the
  normal 103.6-second 2K sequential frontier, and both benchmark smoke controls.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 21.
  Layer 22, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 22 while preserving all
  retained layers-0–21 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-20 full-2K prefill

Objective:

- Carry layer 19's retained final HC through layer 20's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-20 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-20 model mappings
  and 70 dispatches, taking the complete terminal schedule to 1,085 dispatches
  and 556/556 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 17933.832 ms wall
  and 17082.936 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- Target-Mac validation passed optimized Objective-C/Metal compilation, all
  185 Rust tests, 62 Python tests, the complete pinned fixture corpus, every
  retained decoder/sparse control, and both benchmark smoke controls. It
  repeated the layer-20 boundary at 6004.181 ms wall and 5883.395 ms GPU.
- The comprehensive wrapper's unchanged 2K sequential frontier was stopped
  after more than 40 minutes of sustained compute under concurrent OBS load;
  its normal prior dedicated run was about 102 seconds. No result or timing
  from that interrupted control is claimed, and the wrapper did not reach its
  final success footer. This does not affect the two independent exact
  layer-20 replays or the directly rerun retained controls above.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 20.
  Layer 21, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 21 while preserving all
  retained layers-0–20 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-19 full-2K prefill

Objective:

- Carry layer 18's retained final HC through layer 19's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-19 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-19 model mappings
  and 47 dispatches, taking the complete terminal schedule to 1,015 dispatches
  and 524/524 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 5640.974 ms wall
  and 5542.788 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 180 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-19 boundary at
  5632.596 ms wall and 5524.921 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 102152.030 ms at 20.049 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 19.
  Layer 20, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 20 while preserving all
  retained layers-0–19 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-18 full-2K prefill

Objective:

- Carry layer 17's retained final HC through layer 18's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-18 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-18 model mappings
  and 70 dispatches, taking the complete terminal schedule to 968 dispatches
  and 496/496 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 8907.699 ms wall
  and 8044.939 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 176 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-18 boundary at
  5410.038 ms wall and 5298.708 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 100706.304 ms at 20.336 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 18.
  Layer 19, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 19 while preserving all
  retained layers-0–18 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-17 full-2K prefill

Objective:

- Carry layer 16's retained final HC through layer 17's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-17 tensors in ten accepted fresh DwarfStar processes over
  the canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical. One incomplete FFN capture was rejected and replaced in full.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-17 model mappings
  and 47 dispatches, taking the complete terminal schedule to 898 dispatches
  and 464/464 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 13618.961 ms wall
  and 12893.935 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 171 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-17 boundary at
  5050.306 ms wall and 4938.790 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 95270.878 ms at 21.497 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 17.
  Layer 18, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 18 while preserving all
  retained layers-0–17 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-16 full-2K prefill

Objective:

- Carry layer 15's retained final HC through layer 16's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-16 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-16 model mappings
  and 70 dispatches, taking the complete terminal schedule to 851 dispatches
  and 436/436 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 3760.783 ms wall
  and 3646.932 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 167 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-16 boundary at
  4752.235 ms wall and 4643.339 ms GPU.
- Its 2K sequential diagnostic reproduced the final decode-replay logits
  exactly over 97842.871 ms at 20.932 tokens/s. This remains correctness
  evidence rather than a performance measurement; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 16.
  Layer 17, complete-model native batched prefill, output logits, and a
  throughput-producing path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 17 while preserving all
  retained layers-0–16 boundaries as regression controls.

### 2026-08-25 — Exact complete layer-15 full-2K prefill

Objective:

- Carry layer 14's retained final HC through layer 15's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-15 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-15 model mappings
  and 47 dispatches, taking the complete terminal schedule to 781 dispatches
  and 404/404 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 4477.729 ms wall
  and 4303.657 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac validation matrix passed optimized
  Objective-C/Metal compilation, all 162 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. It repeated the layer-15 boundary at
  4504.211 ms wall and 4307.917 ms GPU.
- The first sequential-control attempt was terminated after 40 minutes by an
  app-session interruption without producing an artifact. An isolated recovery
  then reproduced the decode-replay logits exactly over 529388.007 ms at a
  diagnostic 3.869 tokens/s. This remains correctness evidence rather than a
  performance measurement; the incomplete batched prefill still differs from
  the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 15.
  Layer 16, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 16 while preserving all
  retained layers-0–15 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-14 full-2K prefill

Objective:

- Carry layer 13's retained final HC through layer 14's complete native
  prefill path and validate the even-layer paired ratio-4 attention/indexer
  compressors independently against DwarfStar.

Evidence:

- Captured 31 layer-14 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-14 model mappings
  and 70 dispatches, taking the complete terminal schedule to 734 dispatches
  and 376/376 pointer matches.
- The optimized M1 Ultra focused correctness run matched every retained tensor
  and the full attention/HC checksums bit-for-bit. It reported 6931.003 ms wall
  and 6724.343 ms GPU; this is a correctness timing with exhaustive readback,
  not a throughput claim.
- The complete target-Mac gate passed optimized Objective-C/Metal compilation,
  all 158 Rust tests, 62 Python tests, the complete pinned fixture corpus,
  every retained decoder/sparse regression, and both benchmark smoke controls.
  It repeated the layer-14 boundary at 24431.809 ms wall and 22369.166 ms GPU
  under contention. Its 2K sequential control reproduced the decode-replay
  logits exactly over 5881468.976 ms at a diagnostic 0.348 tokens/s. These are
  correctness results, not performance measurements; the incomplete batched
  prefill still differs from the complete-model logits as explicitly reported.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 14.
  Layer 15, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 15 while preserving all
  retained layers-0–14 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-13 full-2K prefill

Objective:

- Carry layer 12's retained final HC through layer 13's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-13 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 28 no-copy layer-13 model mappings
  and 47 dispatches, taking the complete terminal schedule to 664 dispatches
  and 344/344 pointer matches.
- The optimized M1 Ultra correctness run matched all retained tensors and the
  full attention/HC checksums bit-for-bit. Its focused gate reported 19474.658
  ms wall and 17616.812 ms GPU under transient contention; the complete gate
  repeated the same boundary at 3869.312 ms wall and 3657.041 ms GPU. Both are
  correctness timings, not throughput claims.
- The complete target-Mac gate passed optimized Objective-C/Metal compilation,
  all 153 Rust tests, 62 Python tests, the complete pinned fixture corpus,
  every retained decoder/sparse regression, and both benchmark smoke controls.
  Its contended 2K sequential control reproduced the decode-replay logits
  exactly over 7146464.377 ms at a diagnostic 0.287 tokens/s; that run is
  correctness evidence, not a performance measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 13.
  Layer 14, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 14 while preserving all
  retained layers-0–13 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-12 full-2K prefill

Objective:

- Carry layer 11's retained final HC through layer 12's complete native
  prefill path and validate the even-layer paired ratio-4 compressors
  independently against DwarfStar.

Evidence:

- Captured 31 layer-12 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates. Their complete captures are SHA-256-pinned.
- Extended the persistent Metal context with 32 no-copy layer-12 model mappings
  and 70 dispatches, taking the complete terminal schedule to 617 dispatches
  and 316/316 pointer matches.
- The optimized M1 Ultra correctness run matched all retained tensors and the
  full attention/HC checksums bit-for-bit. Its focused gate reported 3534.237
  ms wall and 3438.780 ms GPU; this includes exhaustive correctness readback
  and is not a throughput claim.
- The complete target-Mac gate repeated the same boundary at 3511.858 ms wall
  and 3422.943 ms GPU, then passed all 149 Rust tests, 62 Python tests, the
  complete pinned fixture corpus, every retained decoder/sparse regression,
  and both benchmark smoke controls. Its contended 2K sequential control still
  reproduced the decode-replay logits exactly at a diagnostic 0.328 tokens/s;
  that run is correctness evidence, not a performance measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 12.
  Layer 13, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 13 while preserving all
  retained layers-0–12 boundaries as regression controls.

### 2026-08-24 — Exact complete layer-11 full-2K prefill

Objective:

- Carry layer 10's retained final HC through layer 11's complete native
  prefill path and validate the odd-layer ratio-128 compressor independently
  against DwarfStar.

Evidence:

- Captured 28 layer-11 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  attention-compressor state, dense mixed attention, FFN, and both additive HC
  updates.
- Extended the persistent Metal context with 28 no-copy layer-11 model mappings
  and 47 dispatches, taking the complete terminal schedule to 547 dispatches
  and 284/284 pointer matches.
- The optimized M1 Ultra correctness run matched all 28 retained tensors and
  the full attention/HC checksums bit-for-bit. Its focused gate reported
  3193.051 ms wall and 3098.131 ms GPU; this includes exhaustive correctness
  readback and is not a throughput claim.
- The final target-Mac gate passed optimized Objective-C/Metal compilation,
  144 Rust tests, 62 Python tests, the complete pinned fixture corpus, and every
  retained decoder/sparse regression. Its contended 2K sequential control still
  reproduced the decode-replay logits exactly; that run is correctness evidence,
  not a performance measurement.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 11.
  Layer 12, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 12 while preserving the
  complete layers-0–11 command as a regression control.

### 2026-08-24 — Exact complete layer-10 full-2K prefill

Objective:

- Carry layer 9's retained final HC through layer 10's complete native prefill
  path and validate the even-layer paired compressors independently against
  DwarfStar.

Evidence:

- Captured 31 layer-10 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, paired ratio-4
  attention/indexer compressor state, dense mixed attention, FFN, and both
  additive HC updates.
- Extended the persistent Metal context with 32 no-copy layer-10 model mappings
  and 70 dispatches, taking the complete terminal schedule to 500 dispatches
  and 256/256 pointer matches.
- The optimized M1 Ultra correctness run matched all 31 retained tensors and
  the full attention/HC checksums bit-for-bit. Its final gate reported 2871.412
  ms wall and 2792.469 ms GPU; this includes exhaustive correctness readback
  and is not a throughput claim.
- The complete host-runtime gate passed after the focused hardware run:
  formatting, Objective-C/Metal compilation, optimized build, all 140 Rust
  tests, 62 Python tests, every pinned differential fixture, and the
  cross-language C0 artifact contract.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 10.
  Layer 11, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 11, then reassess whether
  to continue layer-by-layer or connect the retained prefill state more directly
  to the existing decoder/output path.

### 2026-08-24 — Exact complete layer-9 full-2K prefill

Objective:

- Carry layer 8's retained final HC through layer 9's complete native prefill
  path and validate the new boundary independently against DwarfStar.

Evidence:

- Captured 28 layer-9 tensors from ten fresh DwarfStar processes over the
  canonical 2,048-token prompt. Every first/second capture pair was bitwise
  identical.
- Imported four versioned differential fixtures covering Q/KV, ratio-128
  compressor state, dense mixed attention, FFN, and both additive HC updates.
- Extended the persistent Metal context with 28 no-copy layer-9 model mappings
  and 47 dispatches, taking the complete terminal schedule to 430 dispatches
  and 224/224 pointer matches.
- The optimized M1 Ultra correctness run matched all 28 retained tensors and
  the full attention/HC checksums bit-for-bit. It reported 2535.036 ms wall and
  2462.104 ms GPU; this includes exhaustive correctness readback and is not a
  throughput claim.
- Rust formatting, Objective-C/Metal compilation, and 131 unit tests passed
  before the focused hardware run.

Decision:

- The exact native full-2K prefill frontier is now complete through layer 9.
  Layer 10, complete-model prefill, output logits, and a throughput-producing
  path remain outside this claim.

Next:

- Extend the same exact full-2K frontier through layer 10, then reassess whether
  to continue layer-by-layer or connect the retained prefill state more directly
  to the existing decoder/output path.

### 2026-08-23 — Retained sparse top-k generalized through repeated merges

Objective:

- Remove the first-boundary guard and prove the production retained scheduler
  at the first row count that requires more than one top-k merge pass.

Implementation:

- Traced DwarfStar's complete sort schedule: choose the largest power-of-two
  sort width supported by the pipeline, retain up to 512 indices per initial
  block, derive the compact work width from the final partial block, then
  ping-pong pairwise merge passes until one exact top-512 remains.
- Added `retained-sparse-layer2-pos8195-v1`, a strict 38-operation, 47-tensor,
  6,567,092-byte fixture from two fresh 8K DwarfStar processes. It seeds a
  wrapped 127-row raw-ring window, 2,048 rows in both compressed caches, and all
  four recurrent states immediately before layer 2 position 8195.
- Replaced the fixed 513-index scratch and single merge with capacity-stable
  ping-pong storage plus visible-row schedule derivation. At row 2,049 this
  produces three 1,024-thread initial blocks, a 1,025-index active work width,
  and two merge dispatches. The original 1,025-row schedule remains one merge.
- Added `retained-sparse-multimerge-probe`, its stable JSON contract, CLI and
  runtime-gate integration. The command continues to deny preceding-layer,
  token-dependent FFN, complete-decoder, logits, and throughput claims.
- Removed all temporary DwarfStar capture hooks immediately after capture;
  `ds4.c` is unchanged. Two diagnostic-only oversized clamped expert scratch
  dumps had non-semantic fresh-process differences and were excluded; every
  seed, sparse tensor, weighted expert result, routed output, and final HC was
  byte-identical.

Validation:

- Strict fixture verification accepted 38 operations, 47 tensors, and
  6,567,092 bytes. Rust host suite: 129 tests passed. Optimized
  Rust/Objective-C/Metal compilation passed on the M1 Ultra.
- Focused runs kept both retained regressions C0 exact. Row 1,025 still matched
  16 tensors over 54 dispatches with 35/35 mappings. Row 2,049 matched 16
  tensors over 55 dispatches with 35/35 mappings and reported 61.399 ms wall /
  32.045250 ms GPU with correctness setup/readback in scope.
- The complete target-Mac gate passed 129 Rust tests, 61 Python tests, strict
  fixture validation, optimized Rust/Objective-C/Metal compilation, and every
  model-backed control. It repeated row 1,025 C0 exact at 51.990 ms wall /
  28.167000 ms GPU and row 2,049 at 49.455 ms wall / 28.882250 ms GPU.

Next:

- Drive the generalized schedule from complete retained decoder progression,
  then use the eligible native prefill/decode loop for engine measurements.

### 2026-08-23 — Retained state crosses the first default sparse row

Objective:

- Prove that the production retained layer executor, rather than only the
  isolated sparse diagnostic, reaches DwarfStar's first default sparse row with
  the correct cache, recurrent-state, and same-step update ordering.

Implementation:

- Captured two fresh DwarfStar layer-2 position-4099 controls and required exact
  agreement for the incoming HC, four pre-update compressor states, and all
  sparse-boundary outputs. Temporary capture hooks were removed immediately;
  `ds4.c` remains unchanged.
- Added `retained-sparse-layer2-pos4099-v1`, a strict 38-operation, 47-tensor,
  3,941,556-byte layer-segment fixture, plus a deterministic importer. The two
  score-state payloads preserve their exact non-finite bit patterns as integer
  tensors and are reinterpreted only at the runtime boundary.
- Added a diagnostic seed API that populates the exact persistent Metal keys
  used by the general retained executor: the incoming layer-2 HC, 127 raw ring
  rows, 1,024 attention/indexer compressed rows, and both recurrent states.
  Queue-ordered empty predecessors preserve the scheduler's declared layer-2
  chain ownership without claiming that layers 0 and 1 were executed.
- Added `retained-sparse-boundary-probe`. It executes the normal retained
  layer-2 position-4099 schedule, commits compressed row 1,025, runs the
  two-block top-k merge and 12-split indexed attention, and verifies 16 tensors
  by FP32 or integer bit pattern. The JSON explicitly denies complete-layer,
  complete-decoder, output-logit, and throughput claims because the captured HC
  bypasses preceding layers and the placeholder token is not an FFN oracle.

Validation:

- Strict fixture verification accepted 38 operations, 47 tensors, and
  3,941,556 bytes. The two fresh source captures matched for every new seed
  tensor.
- Rust host suite: 126 tests passed. Optimized Rust/Objective-C/Metal
  compilation and the focused M1 Ultra model run passed.
- The focused retained-state run matched all 16 tensors, preserved 35/35 model
  mmap pointers, and submitted 54 dispatches. It reported 53.413 ms wall /
  29.894125 ms GPU with seed, synchronization, and exhaustive readback in
  scope; this is correctness evidence only.
- The complete target-Mac gate passed 126 Rust tests, 61 Python tests, strict
  fixture validation, optimized Rust/Objective-C/Metal compilation, and every
  model-backed control. Its retained-state replay was C0 exact at 58.511 ms
  wall / 30.369250 ms GPU with the same 54 dispatches and 35/35 mappings.

Next:

- Generalize the retained argsort/merge workspace and repeated merge schedule
  beyond 1,025 compressed rows, then use it in a complete decoder progression.

### 2026-08-23 — Production-default sparse switch captured and wired

Objective:

- Replace the diagnostic threshold override with evidence at DwarfStar's real
  first sparse row and connect that exact schedule to retained decoder state.

Implementation:

- Ran two fresh 4K DwarfStar processes without a sparse-threshold override and
  captured layer 2 at position 4099. Every sparse-relevant payload was
  byte-identical: 1,025 score values, exact top-512 indices, both compressed
  caches, the physical raw cache, indexed attention, and inverse RoPE output.
- Added `sparse-indexed-attention-pos4099-v1`, a 12-tensor, 3,339,012-byte
  differential fixture. The importer preserves the older position-2051
  override and selects the default-boundary profile explicitly.
- Imported DwarfStar's two-block argsort merge into the isolated schedule. The
  public probe now executes the 513-row control first and then the 1,025-row
  production-default fixture.
- Extended retained even-layer ABI ownership with the two sparse indexer model
  spans. At row 1,025 it commits the newly emitted attention/indexer rows before
  scoring, runs the exact indexer projection/RoPE/QAT/score/sort/merge and
  12-split indexed attention schedule, then continues through the established
  inverse-RoPE, output, and FFN path.
- Kept an explicit guard after row 1,025. Repeated merge passes and workspace
  sizing for larger contexts remain a separate correctness step.
- Temporary DwarfStar dump hooks were removed; `ds4.c` is clean. The two local
  manual evidence logs remain private and untracked.

Validation:

- Both new DwarfStar captures matched bit-for-bit at every retained sparse
  tensor. The strict fixture verifier accepted 10 operations, 12 tensors, and
  3,339,012 bytes.
- Rust host suite: 124 tests passed. Optimized Rust/Objective-C/Metal compilation
  passed on the M1 Ultra.
- Focused model run executed both sparse controls C0 exact. The reported default
  boundary used 11 dispatches, preserved 3/3 mmap pointers, and measured
  25.162 ms wall / 0.550000 ms GPU with correctness setup/readback in scope.
- The complete target-Mac gate passed 124 Rust tests, 61 Python tests, strict
  validation of both sparse fixtures and every retained model-backed control.
  Its default-boundary replay was C0 exact at 22.303 ms wall / 0.547833 ms GPU.
- No complete retained decoder, output-logit, or throughput claim was made for
  the newly wired branch in this checkpoint.

Next:

- Add a retained-state C0 control that actually reaches row 1,025, then
  generalize the merge schedule beyond the first boundary before producing an
  engine measurement.

### 2026-08-23 — Exact diagnostic top-512 sparse indexed attention

Objective:

- Validate DwarfStar's complete ratio-4 indexer selection and sparse indexed
  attention mechanism on the M1 Ultra before changing retained decoder state.

Correction:

- The earlier journal wording that sparse selection begins immediately after
  512 compressed rows was incomplete. Pinned source commit `b030961` uses a
  default sparse threshold of 1,024 and a fixed model top-k of 512; the default
  first sparse row count is 1,025. A diagnostic environment override may lower
  the threshold without changing the production default.

Implementation:

- Captured layer 2 at position 2051 twice in fresh DwarfStar processes with
  `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=512`. All 13 dump artifacts were
  byte-identical, including indexer Q/weights/scores, exact top-512 indices,
  indexed-attention output, and inverse-RoPE output.
- Added a strict importer and a 12-tensor, 2,026,244-byte differential fixture.
  Its manifest records the override, pinned default, physical-to-compact raw
  cache slice, and explicit denial of default-switch, complete-decode, logits,
  and throughput claims.
- Imported DwarfStar's direct score, exact argsort, 12-way indexed attention,
  and reduction kernels. Added one no-copy ABI/CLI command that also reuses the
  existing F16 matvec, RoPE, indexer-QAT, and F32-to-F16 conversion kernels.
- Aligned Rust Star's dense-path guard with the pinned 1,024-row default. The
  isolated sparse path is not yet connected to retained 43-layer decoder state.

Validation:

- Fixture integrity validation passed: 9 operations, 12 tensors, 2,026,244
  verified bytes.
- Focused optimized M1 Ultra run: all retained values C0 exact, 10 dispatches,
  3/3 mmap pointer identities, 18.864 ms wall / 0.497625 ms GPU. This includes
  correctness setup/readback and is not a throughput measurement.
- Full target-Mac gate passed: 124 Rust tests, 61 Python tests, all prior
  model-backed controls, cross-language artifact validation, and the new
  fixture/probe. The gate sparse run repeated C0 exact with 3/3 mmap pointers
  at 21.616 ms wall / 0.538250 ms GPU; this remains diagnostic only.

Next:

- Integrate the sparse schedule into retained decoder ownership at 1,025
  compressed rows, with the 513-row override remaining an independent kernel
  boundary; then continue native prefill through layer 9.

### 2026-08-22 — Exact complete layer-8 full-2K prefill

Objective:

- Carry layer 7's retained final HC directly through layer 8's complete native
  prefill schedule without a host activation handoff.

Implementation:

- Captured layer 8's ten Q/KV boundaries, paired ratio-4 attention/indexer
  compressor outputs and states, dense-attention diagnostics/output/HC post,
  and complete routed/shared FFN in independent fresh DwarfStar processes;
  every paired full capture was byte-identical.
- Added four differential fixtures covering 31 retained tensors and 53,134,848
  bytes, with full-capture SHA-256 identities retained in their manifests.
- Wrapped 32 additional GGUF spans without copies, extended the shared Metal
  schedule by 70 dispatches, retained all layer-8 state in the persistent
  context, and added exhaustive C0 comparisons plus stable JSON reporting.
- Implemented layer 8's paired ratio-4 compressor ownership for both attention
  and indexer state. At the exact 2K boundary all 512 compressed rows remain on
  the dense path; sparse indexer top-k is explicitly unclaimed.
- Updated the CLI and runtime documentation. The artifact closes at
  `layer8_ffn_hc_post` and explicitly denies sparse post-prompt attention,
  complete-model-prefill, output-logit, and throughput claims.

Validation:

- All four new differential fixtures validate independently.
- Rust unit suite: 122 passed. Python tooling suite: 61 passed.
- Focused optimized M1 Ultra run: every retained layer-8 boundary C0 exact,
  2241.471 ms wall / 2153.893625 ms GPU, 383 dispatches, and 196/196 no-copy
  model mappings.
- Full target-Mac gate: all host/runtime controls passed; the complete
  layers-0–8 command reported 2289.152 ms wall / 2161.306708 ms GPU with the
  same schedule and mapping counts. These intervals include exhaustive
  correctness readback and are not throughput claims.

Next:

- Implement the fixed 512-row ratio-4 sparse-indexed decode boundary so
  generation can advance beyond the native 2K prompt, then continue the exact
  batched-prefill frontier through layer 9.

### 2026-08-22 — Exact complete layer-7 full-2K prefill

Objective:

- Carry layer 6's retained final HC directly through layer 7's complete native
  prefill schedule without a host activation handoff.

Implementation:

- Captured layer 7's ten Q/KV boundaries, ratio-128 compressor output/state,
  dense-attention diagnostics/output/HC post, and complete routed/shared FFN in
  independent fresh DwarfStar processes; every paired full capture was
  byte-identical.
- Added four differential fixtures covering 28 retained tensors and
  52,299,264 bytes, with full-capture SHA-256 identities retained in their
  manifests.
- Wrapped 28 additional GGUF spans without copies, extended the shared Metal
  schedule by 47 dispatches, retained all layer-7 state in the persistent
  context, and added exhaustive C0 comparisons and stable JSON reporting.
- Initialized layer 7's compressed-position buffer with the canonical
  ratio-128 positions; the first device run exposed this otherwise-silent
  ownership requirement at the compressed-KV boundary.
- Restored the standalone Q8 projection encoder lifecycle after the initial
  schedule-block relocation removed its original `endEncoding`; the projection
  regression control is C0 exact again.
- Updated the CLI and runtime documentation. The artifact closes at
  `layer7_ffn_hc_post` and explicitly denies layer-8, sparse post-prompt,
  complete-model-prefill, output-logit, and throughput claims.

Validation:

- All four new differential fixtures validate independently.
- Rust unit suite: 117 passed. Python tooling suite: 61 passed.
- Focused optimized M1 Ultra run: every retained layer-7 boundary C0 exact,
  1950.645 ms wall / 1864.381 ms GPU, 313 dispatches, and 164/164 no-copy model
  mappings.
- Full target-Mac gate: all host/runtime controls passed; the complete
  layers-0–7 command reported 1915.398 ms wall / 1819.682 ms GPU with the same
  schedule and mapping counts. These intervals include exhaustive correctness
  readback and are not throughput claims.

Next:

- Extend the exact terminal prefill frontier through layer 8, or prioritize the
  fixed 512-row ratio-4 sparse-indexed decode boundary if decode progression is
  the nearer benchmark gate.

### 2026-08-22 — Exact complete layer-6 FFN and final HC

Objective:

- Continue the retained layer-6 attention HC post through its routed/shared FFN
  and additive final HC update without a host activation handoff.

Implementation:

- Captured `hc_ffn_pre`, `ffn_norm`, router logits/probabilities/top-k/scaled
  weights, weighted routed SwiGLU, routed output, shared-expert output, and
  `hc_ffn_post` in two fresh DwarfStar processes; all ten full repeated captures
  were byte-identical.
- Added `prefill-layer6-complete-2048-v1`, retaining exact final-tile payloads
  while SHA-256-pinning every full 2K tensor identity, including the complete
  128 MiB final HC state.
- Wrapped layer 6's twelve FFN model spans directly from the GGUF mmap and
  appended the proven 21-dispatch HC ingress, biased top-6 router,
  IQ2_XXS/Q2_K routed expert, Q8_0 shared expert, and additive HC schedule.
- Extended the C ABI, retained Metal ownership, exhaustive Rust comparisons,
  stable JSON, CLI, documentation, fixture verification, and target-Mac gate.
  The artifact now closes at `layer6_ffn_hc_post` and explicitly denies layer-7
  prefill, sparse post-prompt top-k, complete-model-prefill, and throughput
  claims.

Validation:

- Fixture verifier: valid ten-tensor, 5,834,240-byte differential fixture.
- Rust unit suite: 113 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1709.387 ms wall
  / 1620.434 ms GPU, 266 dispatches, and 136/136 no-copy model mappings.
- Full target-Mac gate: 113 Rust tests and 61 Python tests passed; every retained
  runtime control remained exact, and the complete layers-0–6 command reported
  1636.955 ms wall / 1548.723 ms GPU with 136/136 mappings. These intervals
  include exhaustive correctness readback and are not throughput claims.

Next:

- Carry the retained layer-6 final HC into layer 7 and extend the complete
  full-2K prefill frontier one layer deeper.

### 2026-08-22 — Exact layer-6 dense mixed attention HC post

Objective:

- Continue the retained layer-6 paired-compressor state through dense mixed
  attention and its additive HC post without a host activation handoff.

Implementation:

- Captured attn_out, hc_attn_post, kqv_out, kqv_back, and attn_low in two fresh
  DwarfStar processes; all five full repeated captures were byte-identical.
- Added prefill-layer6-attention-2048-v1, retaining the complete 32 MiB
  attention output, three row-0 diagnostics, and final 32-row HC tile while
  checksum-pinning the complete 128 MiB HC identity.
- Wrapped layer 6's attention sinks and two Q8 output projections directly from
  the GGUF mmap and appended the proven nine-dispatch dense mixed-attention
  schedule.
- Added explicit retained buffers for layer 5's KQV-back and attention-low
  diagnostics. Those values previously remained in shared scratch only; layer
  6 attention correctly reuses that scratch, so the snapshots preserve the
  already-published layer-5 evidence boundary without adding dispatches.
- Extended the C ABI, exhaustive Rust comparisons, stable JSON, CLI,
  documentation, fixture verification, and target-Mac gate. The artifact closes
  at layer6_attention_hc_post and explicitly denies layer-6 FFN,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid five-tensor, 35,946,496-byte differential fixture.
- Rust unit suite: 112 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1545.043 ms wall
  / 1464.156 ms GPU, 245 dispatches, and 124/124 no-copy model mappings.
- Full target-Mac gate: 112 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1635.054 ms wall / 1454.353 ms GPU with 124/124
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Continue through layer 6's biased-top-6 routed/shared FFN and additive final
  HC update.

### 2026-08-22 — Exact layer-6 paired ratio-4 compressors

Objective:

- Continue the retained layer-6 Q/KV state through both native ratio-4
  attention/indexer compressors without a host activation handoff.

Implementation:

- Captured KVcompress, attn_state_kv, attn_state_score,
  indexer_KVcompress, indexer_state_kv, and indexer_state_score in two fresh
  DwarfStar processes; all six repeated captures were byte-identical.
- Added prefill-layer6-compressors-2048-v1, containing all 512 attention and
  indexer compressed rows plus the four final recurrent-state tensors.
- Wrapped the eight layer-6 compressor tensors directly from the GGUF mmap and
  appended the proven 30-dispatch paired ratio-4 schedule to the retained
  terminal Metal command.
- Extended the C ABI, exact Rust comparisons, stable JSON, CLI, documentation,
  fixture verification, and target-Mac gate. The artifact closes at
  layer6_paired_compressors and explicitly denies layer-6 attention, FFN,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid six-tensor, 1,392,640-byte differential fixture.
- Rust unit suite: 110 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1358.621 ms wall
  / 1266.291 ms GPU, 236 dispatches, and 121/121 no-copy model mappings.
- Full target-Mac gate: 110 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1365.469 ms wall / 1265.535 ms GPU with 121/121
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Continue through layer 6's dense mixed attention and additive attention HC
  post.

### 2026-08-22 — Exact layer-6 full-2K Q/KV state

Objective:

- Continue the retained layer-5 final HC state through layer 6's native Q/KV
  path without a host activation handoff.

Implementation:

- Captured `hc_attn_pre`, `attn_norm`, `q_lora`, `q_lora_norm`, `KVraw`,
  `KVnorm`, `Qraw`, `Qcur`, `KVrope`, and `KVcur` in four fresh DwarfStar
  processes. Both primary captures and both isolated `Qraw` captures were
  byte-identical.
- Added `prefill-layer6-qkv-2048-v1`, retaining exact final 32-row tiles while
  SHA-256-pinning the complete 2K identities.
- Wrapped layer 6's nine HC/QKV tensors directly from the GGUF mmap and appended
  the proven ten-dispatch Q/KV schedule to the retained Metal command.
- Extended the C ABI, Rust exact comparisons, stable JSON, CLI, documentation,
  fixture verification, and target-Mac gate. The artifact now closes at
  `layer6_qkv_state` and explicitly denies layer-6 compressors, attention, FFN,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid ten-tensor, 9,961,472-byte differential fixture.
- Rust unit suite: 109 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1338.542 ms wall
  / 1247.536 ms GPU, 206 dispatches, and 113/113 no-copy model mappings.
- Full target-Mac gate: 109 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1328.350 ms wall / 1241.698 ms GPU with 113/113
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Continue through layer 6's paired ratio-4 attention/indexer compressors.

### 2026-08-22 — Exact complete layer-5 full-2K prefill

Objective:

- Continue the retained layer-5 post-attention HC state through its complete
  biased-top-6 routed/shared FFN and additive final HC update.

Implementation:

- Captured `hc_ffn_pre`, `ffn_norm`, router logits/probabilities/top-k/scaled
  weights, routed weighted-SwiGLU/output, shared-expert output, and
  `hc_ffn_post` twice in fresh DwarfStar processes. All ten complete tensors
  were byte-identical.
- Added `prefill-layer5-complete-2048-v1`, retaining the exact final 32-row
  comparison tiles while SHA-256-pinning all complete source tensors and the
  complete final HC identity.
- Wrapped layer 5's twelve FFN tensors directly from the GGUF mmap and appended
  the proven 21-dispatch biased-top-6 schedule to the retained Metal command.
- Extended the C ABI, exact Rust comparisons, stable JSON, CLI, documentation,
  fixture verification, and target-Mac gate. The artifact now closes at
  `layer5_ffn_hc_post` and explicitly denies later-layer, sparse-attention,
  complete-model-prefill, and throughput claims.

Validation:

- Fixture verifier: valid ten-tensor, 5,834,240-byte differential fixture.
- Rust unit suite: 108 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1358.984 ms wall
  / 1270.350 ms GPU, 196 dispatches, and 104/104 no-copy model mappings.
- Full target-Mac gate: 108 Rust tests and 61 Python tests passed; the extended
  command remained exact at 1363.509 ms wall / 1272.876 ms GPU with 104/104
  mappings. These intervals include exhaustive correctness readback and are not
  throughput claims.

Next:

- Begin layer 6's native Q/KV boundary from the retained layer-5 final HC state.

### 2026-08-22 — Exact layer-5 full-2K dense mixed attention

Objective:

- Continue the retained layer-5 Q/KV and ratio-128 compressor state through
  dense mixed attention and the additive attention HC post-state.

Implementation:

- Captured `kqv_out`, `kqv_back`, `attn_low`, `attn_out`, and `hc_attn_post`
  twice in fresh DwarfStar processes. All five complete tensors were byte-stable.
- Added `prefill-layer5-attention-2048-v1`, retaining the complete 32 MiB
  attention output, three row-zero diagnostics, and the final 32-row HC tile
  while SHA-256-pinning the complete 128 MiB HC identity.
- Wrapped layer 5's sinks and two Q8 output projections directly from the model
  mmap and extended the retained Metal command with the proven nine-dispatch
  ratio-128 dense-attention schedule.
- Snapshotted layer 4's shared attention diagnostics before layer 5 reuses the
  scratch buffers; this preserves prior evidence without adding a dispatch.
- Extended the Rust ABI, exhaustive C0 checks, stable JSON, CLI, documentation,
  and target-Mac gate label. The layer-5 FFN remains explicitly out of scope.

Validation:

- Fixture verifier: valid five-tensor, 35,946,496-byte differential fixture.
- Rust unit suite: 107 passed.
- Focused optimized M1 Ultra run: every new boundary C0 exact, 1242.696 ms wall
  / 1152.157 ms GPU, 175 dispatches, and 92/92 no-copy model mappings. This
  includes exhaustive correctness readback and is not a throughput claim.
- Full target-Mac gate: 107 Rust tests and 61 Python tests passed, every native
  regression control remained exact, and the extended command reported
  1253.568 ms wall / 1159.442 ms GPU with 92/92 no-copy mappings.

Next:

- Continue the retained post-attention HC state through the complete layer-5
  FFN and additive final HC update.

### 2026-08-22 — Exact layer-5 full-2K ratio-128 compressor

Objective:

- Continue the retained layer-5 Q/KV state through all 16 ratio-128 emissions
  and establish the exact final recurrent-state boundary before dense attention.

Implementation:

- Captured `KVcompress`, `attn_state_kv`, and `attn_state_score` twice in fresh
  DwarfStar processes. Both captures were byte-identical; the aligned 2K batch
  ends with canonical zero KV scratch and negative-infinity score scratch.
- Added `prefill-layer5-compressor-2048-v1`, pinning 557,056 bytes across the
  16x512 compressed output and two 128x512 final state tensors.
- Wrapped four additional layer-5 compressor tensors directly from the model
  mmap and extended the retained Metal command with two F16 projections plus
  the proven five-dispatch ratio-128 batch schedule.
- Extended the Rust ABI, exhaustive C0 comparisons, stable JSON boundary, CLI,
  documentation, and target-Mac gate label. Layer-5 attention and FFN remain
  explicitly out of scope.

Validation:

- Fixture verifier: valid three-tensor, 557,056-byte differential fixture.
- Rust unit suite: 105 passed.
- Focused optimized M1 Ultra run: all 16 emissions and both terminal states C0
  exact, 1076.840 ms wall / 991.235 ms GPU, 166 dispatches, and 89/89 no-copy
  model mappings. This includes exhaustive correctness readback and is not a
  throughput claim.
- Full target-Mac gate: 105 Rust tests and 61 Python tests passed, every native
  regression control remained exact, and the extended command reported
  1072.662 ms wall / 972.161 ms GPU with 89/89 no-copy mappings.

Next:

- Continue the retained layer-5 state through dense mixed attention, additive
  attention HC post-processing, and the complete FFN.

### 2026-08-22 — Exact layer-5 HC ingress and full-2K Q/KV state

Objective:

- Carry layer 4's retained final HC directly into layer 5 and establish an
  exact, GPU-resident Q/KV checkpoint for the next ratio-128 compressor step.

Implementation:

- Captured layer-5 HC attention ingress, learned norm, Q-Lora, Q-Lora norm,
  raw/current Q, and raw/normalized/RoPE/final KV across four fresh DwarfStar
  processes. Every repeated full tensor was byte-stable.
- Added `prefill-layer5-qkv-2048-v1`, retaining compact final 32-row tiles for
  ten boundaries while pinning each complete tensor identity by SHA-256.
- Wrapped nine additional layer-5 tensors directly from the model mmap and
  extended the terminal command with the proven ten-dispatch HC/QKV schedule.
  The full finalized Q and KV buffers remain in the persistent Metal context.
- Extended stable JSON to the `layer5_qkv_state` boundary with explicit layer-5
  correctness and scope claims.

Validation:

- Fixture verifier: valid ten-tensor, 9,961,472-byte differential fixture.
- Rust unit suite: 104 passed.
- Focused optimized M1 Ultra run: all ten layer-5 boundaries C0 exact,
  1085.161 ms wall / 994.834 ms GPU, 159 dispatches, and 85/85 no-copy model
  mappings.
- Full target-Mac gate: 104 Rust tests and 61 Python tests passed, every native
  control remained exact, and the extended command reported 1088.225 ms wall /
  991.314 ms GPU with 85/85 no-copy mappings.

Next:

- Continue the retained full layer-5 Q/KV state through all 16 ratio-128
  compressor emissions and its final recurrent state.

### 2026-08-22 — Complete exact layer-4 prefill

Objective:

- Continue the retained layer-4 post-attention HC state through the complete
  FFN and establish native full-2K ownership through layer 4.

Implementation:

- Captured ten FFN hooks twice in fresh DwarfStar processes: HC ingress,
  learned norm, router logits/probabilities, biased top-6 selections, scaled
  weights, fused routed activations/output, shared output, and final HC. All
  full-2K identities were byte-stable.
- Added `prefill-layer4-complete-2048-v1`, retaining exact final-tile payloads
  and pinning the complete 128 MiB final-HC identity by SHA-256 and checksum.
- Wrapped twelve additional layer-4 FFN tensors directly from the model mmap
  and added the proven 21-dispatch biased router, routed IQ2_XXS/Q2_K expert,
  shared Q8_0 expert, and additive HC schedule.
- Extended the stable JSON boundary to `layer4_ffn_hc_post`, with 149 terminal
  dispatches, 76/76 no-copy mappings, exact FFN flags, and a complete layer-4
  prefill claim.

Validation:

- Fixture verifier: valid ten-tensor, 5,834,240-byte differential fixture.
- Rust unit suite: 103 passed.
- Focused optimized M1 Ultra run: every retained FFN boundary and the complete
  final HC identity C0 exact, 1079.281 ms wall / 991.837 ms GPU.
- Full target-Mac gate: 103 Rust tests and 61 Python tests passed, every native
  control remained exact, and the complete layer-4 command reported 1601.151
  ms wall / 985.798 ms GPU with 76/76 no-copy mappings.

Next:

- Continue the retained final HC state into layer 5, beginning with its HC
  attention ingress and Q/KV state.

### 2026-08-22 — Exact layer-4 dense mixed attention and HC post

Objective:

- Continue the retained full-2K layer-4 Q/KV and paired-compressor state
  through dense mixed attention and the additive four-stream HC update without
  a host activation upload.

Implementation:

- Captured `kqv_out`, `kqv_back`, `attn_low`, `attn_out`, and `hc_attn_post`
  twice in fresh DwarfStar processes. Every capture was byte-stable.
- Added `prefill-layer4-attention-2048-v1`, retaining the complete 2048x4096
  attention output, final 32-row HC tile, and compact row-zero diagnostics.
- Wrapped layer 4's sinks and two attention-output weights directly from the
  model mmap. Added the F32/F16 staging, FlashAttention, inverse RoPE, grouped
  map, low/output projections, and HC expansion as nine terminal dispatches.
- Snapshotted layer 3's shared diagnostic buffers before their reuse by layer
  4. The first focused run exposed that lifetime overlap; the corrected run
  passed all retained layer-3 and layer-4 comparisons.
- Extended the stable JSON boundary to `layer4_attention_hc_post`, with 128
  terminal dispatches, 64/64 no-copy mappings, and pinned full-output checksums.

Validation:

- Fixture verifier: valid five-tensor, 35,946,496-byte differential fixture.
- Rust unit suite: 102 passed.
- Focused optimized M1 Ultra run: complete attention output and HC post C0
  exact, 934.193 ms wall / 833.479 ms GPU.
- Full target-Mac gate: 102 Rust tests and 61 Python tests passed, every native
  control remained exact, and the terminal command reported 952.654 ms wall /
  851.603 ms GPU with 64/64 no-copy mappings.

Next:

- Complete the layer-4 FFN from the retained post-attention HC state.

### 2026-08-22 — Exact layer-4 paired ratio-4 compressors

Objective:

- Continue the retained full-2K layer-4 normalized activation through both
  even-layer ratio-4 compressors without a host activation upload.

Implementation:

- Captured `KVcompress`, `attn_state_kv`, `attn_state_score`,
  `indexer_KVcompress`, `indexer_state_kv`, and `indexer_state_score` twice in
  fresh DwarfStar processes. All six identities were byte-stable.
- Added `prefill-layer4-compressors-2048-v1`, containing 512 attention rows,
  512 indexer rows, and all four final recurrent-state tensors.
- Wrapped eight additional layer-4 compressor tensors directly from the model
  mmap and added four full-batch F16 projections plus exact ratio-4 replay
  pooling, learned norm, compressed RoPE, E4M3FN/indexer QAT, and final state
  refresh.
- Corrected the fused ratio-4 shift kernel to restore the consumed upper state
  bank to DwarfStar's canonical zero/negative-infinity scratch state. The first
  focused comparison localized the issue at the exact four-row bank boundary;
  the corrected run passed every C0 comparison.
- Extended the stable JSON boundary to `layer4_paired_compressors`, with
  119 terminal dispatches, 61/61 no-copy mappings, and six pinned checksums.

Validation:

- Fixture verifier: valid six-tensor, 1,392,640-byte differential fixture.
- Rust unit suite: 100 passed.
- Focused optimized M1 Ultra run: all 1,024 emissions and four recurrent-state
  tensors C0 exact, 740.517 ms wall / 656.454 ms GPU.
- Full target-Mac gate: 100 Rust tests and 61 Python tests passed, every native
  control remained exact, and the new terminal command reported 750.548 ms
  wall / 655.250 ms GPU with 61/61 no-copy mappings.

Next:

- Complete layer-4 dense mixed attention and FFN from the retained state.

### 2026-08-22 — Complete layer 3 flows into exact layer-4 Q/KV state

Objective:

- Continue the exact native 2K chain from layer 3's retained final HC through
  layer 4's complete pre-compressor Q/KV boundary.

Changes:

- Captured layer 4's HC attention ingress, learned norm, Q-Lora, fused Q/KV
  norm, Q-B, Q/KV RoPE, and FP8 KV-finalization tensors in four fresh DwarfStar
  processes. The separately captured 256 MiB Qraw tensors and every other full
  tensor were byte-identical between repeats.
- Added `prefill-layer4-qkv-2048-v1`, retaining ten exact final tiles while
  pinning all complete 2K identities by SHA-256.
- Added nine no-copy layer-4 model spans and ten Metal dispatches to the
  terminal command. The runtime retains the full layer-4 Q, KV, input HC, and
  attention split buffers for its compressor/attention continuation.
- Extended the ABI, C0 checks, stable JSON report, documentation, and fixture
  shape coverage through the `layer4_KVcur` boundary.

Validation:

- Focused optimized M1 Ultra run: all ten boundaries C0 exact, 89 dispatches,
  53/53 no-copy model mappings, 1169.303 ms wall / 647.777 ms GPU.
- Complete target-Mac gate: 99 Rust tests and 61 Python tests passed, every
  differential fixture verified, all retained Metal controls stayed exact,
  and the new boundary repeated C0 exact at 947.998 ms wall / 651.830 ms GPU.
- These timings include correctness setup/readback and are not throughput
  claims.

Next:

- Drive layer 4's paired ratio-4 attention/indexer compressors from the
  retained normalized activations and validate all 512 prompt emissions.

### 2026-08-22 — Native 2K prefill is complete through layer 3

Objective:

- Continue the exact full-2K layer-3 attention state through its complete FFN
  and additive four-stream HC post-state without a host activation handoff.

Changes:

- Captured the ten layer-3 FFN oracle boundaries twice in fresh DwarfStar
  processes. Every full tensor was byte-identical; the compact checked fixture
  retains all final 32-row boundaries and pins every full-2K SHA-256 identity.
- Added a batch Metal top-k router that reproduces DwarfStar's sequential
  strict-`>` top-6 selection over `sqrt(softplus(logits)) + bias`. Expert
  weights are gathered from the unbiased probabilities before normalization
  and the 1.5 scale.
- Extended the terminal command through FFN HC ingress, router projection,
  routed IQ2_XXS gate/up plus Q2_K down, the Q8_0 shared expert, and the final
  additive HC update. Large routed/shared scratch allocations are reused only
  after layer 2 completes.
- Added twelve layer-3 FFN model spans, seven ABI outputs, exact final-tile C0
  checks, a full-final-HC checksum guard, stable JSON metadata, and a complete
  layer-3 prefill claim.

Validation:

- Two fresh oracle captures agreed bit-for-bit for all 2,048 rows, including
  all biased top-k selections and the complete 128 MiB final HC tensor.
- Focused optimized M1 Ultra run: C0 exact, 79 dispatches, 44/44 no-copy model
  mappings, 675.511 ms wall / 592.734 ms GPU.
- Complete target-Mac gate: 98 Rust tests and 61 Python tests passed, every
  differential fixture verified, all retained Metal controls remained exact,
  and the new boundary repeated C0 exact at 759.645 ms wall / 671.882 ms GPU.
- These timings include correctness-oriented execution/readback and are not a
  throughput claim.

Next:

- Continue the native 2K prefill chain through layer 4, then implement sparse
  indexed attention beyond the fixed 512-row dense ratio-4 prompt boundary.

### 2026-08-22 — Layer 3 owns dense mixed attention and additive HC post

Objective:

- Continue the exact native layer-3 batch from its ratio-128 compressor through
  dense mixed attention, inverse RoPE, both output projections, and additive HC
  post-state.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical layer-3 `attn_out` and
  `hc_attn_post` captures. Their full SHA-256 identities are
  `5bfadcbc1d2ee7b42753b506420045409ba277ae2ffc7dbd0e87187e51b74e13`
  and `47c26665144097e0912284961d95f9b3ae72c8ce40c271e028dfdf71f4ee453b`.
- Independent repeated `kqv_out`, `kqv_back`, and `attn_low` captures pin row-0
  diagnostics for the attention and output-projection stages.
- `prefill-layer3-attention-2048-v1` retains the complete 32 MiB attention
  output, the final 32-row HC tile, and the three row-0 diagnostic tensors. The
  full 128 MiB HC identity is pinned without retaining a duplicate large blob.

Implementation:

- Added three mmap-backed model views for layer-3 sinks and Q8 attention output
  weights, plus nine ordered Metal dispatches for raw/compressed F16 staging,
  mask block scan, FlashAttention, inverse RoPE, grouped low projection, dense
  output projection, and four-stream HC expansion.
- The 2,064 logical keys are physically padded to 2,112 rows with 48 fully
  masked entries. This satisfies the non-vector FlashAttention specialization's
  64-row block contract; the prior 2,560-key layer-2 path was already aligned
  and therefore did not expose the requirement.
- Extended the C ABI, Rust report/JSON contract, fixture importer, persistent
  layer-3 HC ownership, and diagnostic comparisons. The terminal checkpoint now
  uses 58 dispatches and 32/32 no-copy model mappings.

Validation:

- The optimized real-model probe reproduced all 8,388,608 layer-3 attention
  values and the final 524,288-value HC tile bit-for-bit while preserving every
  prior boundary. It reported 583.424 ms wall / 501.227 ms GPU with correctness
  readback in scope; this is not a throughput measurement.
- The complete target-Mac gate passed 97 Rust tests and 61 Python tests, every
  fixture verifier and retained Metal control, and a second exact layer-3
  attention run at 569.588 ms wall / 490.821 ms GPU. The 2K sequential control
  remained C0 exact at 21.958 tokens/s over 93270.010 ms.

Decision and next:

- Accept `layer3_attention_hc_post` as the next exact native-batch checkpoint.
  Continue through the layer-3 biased top-k FFN and additive FFN HC post-state.

### 2026-08-22 — Layer 3 owns all 16 ratio-128 prompt emissions

Objective:

- Continue the exact native layer-3 batch from retained full-2K Q/KV state
  through the complete ratio-128 attention compressor and persistent state.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical `KVcompress`,
  `attn_state_kv`, and `attn_state_score` captures at layer 3.
- The 16x512 compressed output has SHA-256
  `9e6d904dc6df0601d0b3c32f9baf58230c893346d050b9330d5c38cc89143481`.
  The two 128x512 states have SHA-256 identities
  `8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90`
  and `6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778`.
- `prefill-layer3-compressor-2048-v1` retains all three complete tensors; no
  output is truncated to a final tile.

Implementation:

- Added two full-batch F16 compressor projections, exact score/APE addition,
  DwarfStar-order scalar 128-row softmax pooling, weighted RMSNorm, compressed
  YaRN RoPE, and E4M3FN finalization to the existing terminal command.
- Added four strict no-copy model views and retained the 16 compressed rows and
  both recurrent-state buffers in the persistent Metal context.
- Extended the C ABI, Rust report/JSON contract, fixture importer, and
  Rust/Python fixture tests. The checkpoint now uses 49 dispatches and 29/29
  no-copy model mappings.

Validation:

- The optimized real-model probe matched all three new full tensors bit-for-bit
  while preserving every prior layer-2 and layer-3 boundary. It reported
  494.638 ms wall / 396.686 ms GPU with correctness readback in scope; this is
  not a throughput measurement.
- The complete target-Mac gate passed 96 Rust tests and 61 Python tests, every
  fixture verifier and retained Metal control, and a second exact compressor
  run at 483.530 ms wall / 399.101 ms GPU. The 2K sequential control remained
  C0 exact at 21.939 tokens/s over 93347.958 ms.

Decision and next:

- Accept `layer3_ratio128_compressor` as the next exact native-batch checkpoint.
  Continue through layer-3 dense mixed attention and its additive HC post-state
  before implementing the biased top-k FFN.

### 2026-08-22 — Layer 3 owns exact full-2K Q/KV state

Objective:

- Continue the exact native layer-3 batch from Q-Lora through Q/KV setup,
  compressed RoPE, FP8 finalization, and persistent raw-KV ownership.

Oracle evidence:

- Two fresh primary DwarfStar processes produced byte-identical full-2K
  `q_lora_norm`, `KVraw`, `KVnorm`, `Qcur`, `KVrope`, and `KVcur` captures.
  Two additional fresh processes produced identical 256 MiB `Qraw` captures.
- The repeated `q_lora` capture retained SHA-256
  `bafe82a1535457caec52278bdb3c95e317aacb128c94f704fc523ca9581d6265`,
  exactly matching the prior checkpoint.
- `prefill-layer3-kv-state-2048-v1` retains exact final tiles for all seven new
  boundaries and pins their full capture SHA-256 identities.

Implementation:

- Added six dispatches to the existing terminal command: Q8 KV projection,
  fused Q/KV learned norm, Q8 Q-B, Q head norm/RoPE, KV RoPE, and FP8 KV
  finalization.
- Added four no-copy model views and retained the complete layer-3 finalized
  KV, Q, input HC, and attention split buffers in the persistent Metal context.
- Extended the ABI, strict tensor checks, stable JSON, fixture importer, and
  Rust/Python fixture tests. Large Qraw/Qcur identities use exact final tiles;
  the manifest pins their complete 2K SHA-256 captures.

Validation:

- The optimized focused probe matched all seven new final-tile boundaries
  bit-for-bit and matched the full-2K checksums for Q-Lora norm, KVraw, KVnorm,
  KVrope, and KVcur.
- The terminal schedule used 42 dispatches and 25/25 no-copy mappings. It
  reported 482.230 ms wall / 402.931 ms GPU with correctness readback in scope;
  this is not a throughput measurement.
- The complete target-Mac gate passed 95 Rust tests and 60 Python tests, every
  fixture verifier and retained Metal control, and a second exact layer-3 run
  at 447.543 ms wall / 366.771 ms GPU. The 2K sequential control also remained
  C0 exact at 20.168 tokens/s over 101546.809 ms.

Decision and next:

- Accept `layer3_kv_cur` as the next exact native-batch checkpoint. Build the
  layer-3 ratio-128 compressor state and its 16 prompt emissions before dense
  mixed attention and the distinct biased top-k FFN.

### 2026-08-22 — Full-2K layer-2 HC flows directly into layer-3 Q-Lora

Objective:

- Cross the first native layer-3 batched-prefill boundary without uploading or
  reconstructing layer 2's completed 16,384-wide HC state on the host.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical full-2K captures for
  layer 3 `hc_attn_pre`, `attn_norm`, and `q_lora`.
- Their full capture SHA-256 values are respectively
  `615f67cb9738b583263e1a7abd3970ad4df818f0bc9220053be4cbc6ba7d6cab`,
  `1348e3368a4c6b7185e730f11545c74a7fe8fa1c9877fcc9200a5de405f2b818`,
  and `bafe82a1535457caec52278bdb3c95e317aacb128c94f704fc523ca9581d6265`.
- `prefill-layer3-ingress-2048-v1` retains the exact final 32 rows for all
  three boundaries and pins the full captures in its manifest.

Implementation:

- Extended the existing terminal layer-2 command with four dispatches: HC RMS
  norm, the F16 HC mixer, fused HC collapse plus learned attention norm, and
  the Q8 Q-A projection.
- Added five mmap-backed layer-3 model views and kept the complete layer-2 HC
  buffer GPU-resident across the boundary.
- Extended the Rust ABI, exact tensor-shape validation, JSON evidence, fixture
  importer, fixture verifier coverage, and focused unit tests.

Validation:

- The optimized real-model probe passed on the M1 Ultra with all three new
  final-tile tensors bitwise equal to DwarfStar and their full-2K FNV checksums
  equal to `0xcb93e0a251fd7280`, `0xe64fba2f04bfcb54`, and
  `0xd01875758ab4b722`.
- The terminal schedule used 36 dispatches and retained 21/21 no-copy model
  mappings. It reported 441.440 ms wall / 371.934 ms GPU with correctness
  readback in scope; this is not a throughput measurement.
- The complete target-Mac gate passed: 94 Rust tests, the optimized build,
  Metal ownership/dispatch validation, all differential
  fixtures, the Rust/Python artifact contract, every real-model Metal control,
  and the 2K sequential diagnostic. The gate repeated the new boundary at
  456.961 ms wall / 376.116 ms GPU.
- After adding the fixture-specific Python catalog assertion, the final
  platform-independent regression pass contained 59 Python tests; all passed.

Decision and next:

- Accept `layer3_q_lora` as the next exact native-batch checkpoint. Continue
  through layer 3's Q/KV setup and state ownership before implementing its
  distinct biased top-k FFN.

### 2026-08-16 — Complete layer 2 reaches the exact full-2K HC frontier

Objective:

- Continue the exact layer-2 mixed-attention boundary through both HC updates,
  token-hash routing, routed/shared experts, and the final 16,384-wide HC state.

Oracle evidence:

- Two fresh DwarfStar processes produced byte-identical full-2K captures for
  `hc_attn_post`, FFN ingress/norm/router/expert boundaries, and `hc_ffn_post`.
- The full attention-HC state has SHA-256
  `4f7c61ad617347f186cb959457d5c6f1c95692451bf186c751f863a6417baad2`;
  the full final-HC state has SHA-256
  `67dbac97346ee6bea6bb967eafa6c7841fb3477798f9cc154c9dfed24ff5564b`.
- `prefill-layer2-complete-2048-v1` retains exact final-tile payloads and pins
  every full capture identity without adding a single 128 MiB fixture blob.

Implementation:

- The 64-tile context now retains every layer-2 input HC row, attention split,
  and token ID alongside Q/KV and compressor state.
- The terminal Metal command adds the attention HC expand and a full 2,048-row
  FFN batch: HC ingress/learned norm, token-hash router, fused IQ2_XXS
  pair-SwiGLU, Q2_K routed down/sum, Q8_0 shared expert, and additive HC expand.
- The decomposed router kernels' explicit maximum grew from 32 to 2,048 rows;
  the existing 32-row paths preserve identical dispatch sizes and arithmetic.

Validation:

- The full attention output, attention-HC final tile, FFN ingress, router
  selection/weights, routed/shared outputs, and final-HC final tile all match
  DwarfStar by FP32 bit pattern.
- Full-state checksums match both 2K HC capture identities. The terminal command
  retains 16/16 no-copy model mappings and uses 32 dispatches.
- The focused M1 Ultra run reported 408.397 ms wall / 338.484 ms GPU, including
  synchronization and correctness readback; it is not a throughput claim.
- The complete target-model runtime gate, 58 Python tests, all Rust tests, every
  pinned fixture verifier, and all prior Metal decode/prefill controls pass.

Next:

- Carry the final layer-2 HC state into the first exact layer-3
  batched-prefill boundary while retaining this complete layer-2 control.

### 2026-08-16 — Exact layer-2 dense mixed attention reaches the full 2K frontier

Objective:

- Continue the exact empty-seed layers-0/1/2 prefill path through layer 2's
  first real attention consumption boundary.

Oracle evidence:

- Tracing the pinned DwarfStar path corrected an important boundary
  assumption: exactly 512 ratio-4 compressed rows still use dense mixed
  attention. Sparse indexer top-k begins only when the compressed cache grows
  beyond 512 rows, after the 2K prompt.
- Two fresh-process full-2K attention-output captures were byte-identical.
- The retained 2048 x 4096 FP32 payload is 33,554,432 bytes with SHA-256
  68c2110283b472105f00e192817dbb682ebe0815f2ba4a76cd73ed20f3d97508.
- prefill-layer2-attention-2048-v1 records the exact 2,048 raw plus 512
  compressed KV shape and the absence of sparse selection at this boundary.

Implementation:

- The persistent tile context now retains all 2,048 layer-2 normalized query
  rows alongside the already exact raw KV and paired-compressor state.
- rust_star_metal_run_prefill_layer2_attention maps Q-B, attention sinks, and
  both output tensors once without copying, then executes a ten-dispatch
  command: full Q8 Q-B, compressed YaRN head norm/RoPE, raw and compressed F16
  staging, dense mixed FlashAttention, inverse compressed RoPE, and
  grouped/dense Q8 output projection.
- The mixed mask preserves the 128-row raw causal window and exposes
  (query + 1) / 4 compressed rows.
- prefill-layers012-attention-loop-probe reuses the exact 64-tile compressor
  loop in the same Metal context and compares all 8,388,608 output values by
  FP32 bit pattern.

Validation:

- The first optimized M1 Ultra run matched the complete DwarfStar attention
  output bit-for-bit.
- The isolated attention command retained 4/4 no-copy model mappings, used ten
  dispatches, and reported 196.069 ms wall / 169.211 ms GPU. This interval
  includes correctness-oriented setup, synchronization, and full readback; it
  is not a prefill-throughput claim.
- The fixture verifier, optimized build, all 92 Rust tests, all 58 Python
  artifact tests, and the complete target-model Metal runtime gate pass.

Scope:

- Layer 2 now has exact full-2K raw KV, both ratio-4 compressors, dense mixed
  attention, inverse RoPE, and attention output projection.
- Layer-2 attention HC post-processing, FFN, complete layer-2 prefill, later
  layers, output logits, and sparse ratio-4 attention after the prompt remain
  pending.

Next:

- Continue the live full-2K boundary through layer 2's attention HC update and
  FFN, then decide whether to advance batched prefill layer-by-layer or isolate
  the first 513-to-512 sparse selection in post-prompt decode.

### 2026-08-16 — Layer-2 paired compressors reach the full 2K frontier

Objective:

- Extend the empty-seed native layers-0/1 and layer-2 raw-KV loop through both
  ratio-4 compressors without introducing a host activation handoff.

Oracle and implementation:

- Captured the complete layer-2 attention/indexer compressed caches and final
  recurrent KV/score states twice in fresh DwarfStar processes. All six payload
  pairs were byte-identical. Added the strict
  `prefill-layer2-compressors-2048-v1` fixture, its provenance-preserving
  importer, and fixture-shape tests.
- Expanded the persistent Metal context from 57 to 65 mmap-backed model ranges
  and added owned attention/indexer compressed caches plus recurrent states.
  Four aligned F16 projections feed DwarfStar-equivalent fused eight-row
  pooling, learned norm, compressed RoPE, and E4M3FN/indexer QAT.
- Preserving DwarfStar arithmetic required two distinct details. The one-shot
  2K capture pools the aligned compressor rows as one fused batch rather than
  repeating the legacy one-row reduction. Regular tiles keep batch-projected
  tail state so later rows remain identical to that one-shot capture; only the
  final tile reruns its four-row tail with the small-batch projection used by
  DwarfStar's published final recurrent state.
- Added `prefill-layers012-compressor-loop-probe`, a stable JSON schema, atomic
  CLI output, exact prefix/state validation, and the command to the target-Mac
  validation script. Regular tiles use 118 dispatches; the final tail refresh
  uses 122.

Target-Mac evidence:

- The focused full-2K command passed all 512 attention compressed rows, all 512
  indexer compressed rows, and the four final recurrent state tensors by FP32
  bit pattern. Every tile preserved 65/65 no-copy mappings. Summed diagnostic
  intervals were 3624.704 ms wall and 3514.757 ms GPU; setup, synchronization,
  and exhaustive correctness checks make these non-throughput measurements.
- The pinned fixture verifier accepted 8 operations, 6 tensors, and 1,392,640
  payload bytes. Formatting and 90 Rust tests passed after adding the command
  and report contracts.
- The complete target-Mac gate repeated the new command at 3548.253 ms summed
  wall and 3444.137 ms summed GPU, then passed all retained runtime controls.
  The gate included 90 Rust tests, 57 Python tests, all 242 pinned differential
  fixtures, an optimized macOS build, strict validation of all 1,288 required
  model tensors, and both steady-state benchmark controls.

Decision and next:

- Accept exact persistent ownership of both layer-2 ratio-4 compressors across
  native 2K prefill. Next consume the resulting 512-row indexer cache for exact
  selection and mixed raw/compressed attention, then cross the layer-2 FFN.
- This checkpoint makes no claim about layer-2 mixed attention, layer-2 FFN,
  complete-model prefill, or inference throughput.

### 2026-08-16 — Layer-2 raw KV ownership reaches the full 2K frontier

Objective:

- Continue the exact layer-2 normalized-KV seam through its compressed-attention
  RoPE, E4M3FN finalization, and retained raw-KV state.

Oracle and implementation:

- Captured full-2K layer-2 `KVrope` and `KVcur` twice in fresh DwarfStar
  processes. Each corresponding 4 MiB pair was byte-identical. Their SHA-256
  values are `d46da14951b304fb4a19be43b82d350b273337de042da2308530438e431e117d`
  and `07f19c5197442f3c85350b32d0661e81b3f105a0e8640d3b3bced6c333267135`.
- Added the strict `prefill-layer2-kv-state-2048-v1` fixture/importer and the
  `prefill-layers012-kv-state-loop-probe` command. Layer 2 uses the production
  YaRN constants and E4M3FN simulation, then appends each finalized tile into
  one persistent GPU allocation.
- Every continuation compares the retained layer-2 prefix against the oracle.
  The extended tile uses 92 dispatches and 57 mmap-backed model views.

Target-Mac evidence:

- The focused full-2K loop passed all 64 `KVnorm`, `KVrope`, and `KVcur` tile
  comparisons plus all accumulated layer-0/layer-1/layer-2 KV-prefix gates.
  Summed correctness-oriented intervals were 3605.869 ms wall and 3497.958 ms
  GPU.
- Formatting, compilation, 88 Rust tests, and 32 focused Python fixture tests
  passed before the full repository gate.
- The complete target-Mac gate then passed 88 Rust tests, all 57 Python tests,
  all 241 pinned differential fixtures, and strict validation of all 1,288
  required model tensors. Its independent layer-2 KV-state replay reported
  3600.907 ms wall and 3497.247 ms GPU, preserved 57/57 no-copy model mappings
  for every tile, and was followed by every retained decoder control and
  steady-state benchmark.

Decision and next:

- Accept the complete native layer-2 raw-KV state as C0 exact ownership
  evidence, not throughput. Next implement the paired ratio-4 attention and
  indexer compressors, then cross mixed raw/compressed attention and the FFN.

### 2026-08-16 — Every live layer-1 output reaches exact layer-2 KVnorm

Objective:

- Use a compact downstream layer-2 oracle to validate every post-FFN layer-1
  output produced by the full native 2K loop.

Oracle and implementation:

- Captured layer 2's full `KVnorm` tensor twice from fresh DwarfStar processes.
  Both 4 MiB payloads were byte-identical with SHA-256
  `089138d8fc82c1eb55754451707f59475f2afb2a356dcc505314ddf29814e7b6`.
- Added the strict `prefill-layer2-kvnorm-2048-v1` fixture/importer and the
  `prefill-layers012-kvnorm-loop-probe` command. Each of the 64 tiles now adds
  layer-2 HC RMSNorm, F16 HC projection, fused HC collapse/learned norm, Q8_0
  Q-A and KV projection, and fused Q/KV learned RMSNorm.
- The new schedule uses 90 dispatches and 57 mmap-backed no-copy model views per
  tile. Each `32x512` output slice is compared bit-for-bit before continuation.

Target-Mac evidence:

- The focused full-2K loop passed all 64 layer-2 slices and every retained
  layer-0/layer-1 KV-prefix check. Summed correctness-oriented intervals were
  3688.567 ms wall and 3593.970 ms GPU.
- The complete gate repeated the new loop at 3543.522/3450.060 ms wall/GPU,
  then passed formatting, all 86 Rust tests, 56 Python tests, all 240 pinned
  differential fixtures, optimized Objective-C/Metal compilation, all 1,288
  required model tensors, every established Metal/decoder control, and both
  steady-state benchmarks.

Decision and next:

- Accept every live layer-1 output row as downstream C0 validated through the
  native layer-2 normalized-KV seam. Do not treat these summed intervals as
  throughput. Next cross layer 2's compressed RoPE/cache/attention boundary and
  finish its FFN before generalizing the full native prefill layer loop.

### 2026-08-16 — Empty-seed layers-0/1 KV ownership reaches all 2K prompt rows

Objective:

- Turn the two-tile persistent checkpoint into a reusable loop from position 0
  through 2047 without introducing another captured execution prefix.

Oracle reuse and implementation:

- Reused the existing 2,048 canonical token IDs plus the captured layer-0 and
  layer-1 KV prefixes through row 2015. No new large fixture was required.
- Generalized native reset/retain/consume state from the tail pair to every
  aligned 32-row tile. Tile 0 starts with zero captured KV rows; each of the 63
  continuations requires contiguous retained state and compares both complete
  accumulated prefixes against the oracle before appending.
- Added `prefill-layers01-live-kv-loop-probe` with schema
  `rust-star-prefill-layers01-live-kv-loop-probe-v1`. Its stable artifact
  records all 64 tile schedules, retained row counts, mapping identity, timing,
  and the exact claim boundary.
- The final tile still performs exhaustive layer-0/layer-1 output comparison,
  and the independently retained positions 1984--2015 tile remains fully C0
  checked. Other non-final layer-1 post-FFN outputs execute but are not all
  retained, so the artifact explicitly denies a complete layers-0/1 or model
  prefill claim.

Target-Mac evidence:

- The focused full loop advanced empty KV buffers to 2,048 exact rows for both
  layers. All 64 tiles preserved 49/49 mmap-backed model views; summed
  correctness-oriented intervals were 3492.491 ms wall and 3393.021 ms GPU.
- The complete gate repeated the 64-tile result at 3489.173/3383.418 ms
  wall/GPU, then passed formatting, all 84 Rust tests, 55 Python tests, all 239
  differential fixtures, optimized Objective-C/Metal compilation, all 1,288
  required model tensors, every established Metal/decoder control, and both
  steady-state benchmarks.

Decision and next:

- Accept the complete 2K layer-0/layer-1 KV chain as exact state-ownership
  evidence, not as a throughput or complete-prefill result. Next extend native
  batched execution through layer 2 and use the layer-2 KV oracle as the
  downstream validation boundary for live layer-1 outputs.

### 2026-08-16 — Live layer-0/layer-1 KV state crosses the tile boundary

Objective:

- Replace the captured KV boundary between the two exact complete layers-0/1
  tiles with Rust-owned, device-resident state while preserving the independent
  captured-prefix replay as a regression control.

Implementation:

- Added explicit reset/retain/consume modes to the native prefill boundary and
  persistent layer-0/layer-1 full-KV buffers to one Metal context.
- The positions 1984--2015 tile initializes from its 1,984-row oracle prefix,
  appends both live 32-row KV tiles, and retains 2,016 rows per layer. The
  positions 2016--2047 tile requires that contiguous state, validates both
  retained prefixes against the C0 oracle, then appends and attends from the
  retained buffers rather than copying its captured prefix into execution
  storage.
- Added `prefill-layers01-live-kv-chain-probe` and stable schema
  `rust-star-prefill-layers01-live-kv-chain-probe-v1`. Its artifact explicitly
  records one persistent context, two command buffers, one inter-tile host
  wait, a captured first prefix, no captured final execution prefix, and no
  full-prefill or single-command-buffer claim.
- Kept `prefill-layers01-row-coverage-probe` unchanged as the independent
  captured-prefix-per-tile control and added both commands to the Mac gate.

Target-Mac evidence:

- The focused live chain passed both tiles bit-for-bit with 49/49 no-copy model
  views per tile. It retained 2,016 rows after the first tile and 2,048 after
  the second; the focused intervals were 129.589/90.751 and 52.422/51.523 ms
  wall/GPU.
- The independent captured-prefix control also remained exact at
  129.548/90.039 and 127.754/87.312 ms wall/GPU.
- The complete gate passed formatting, all 83 Rust tests, 55 Python tests, all
  239 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. The live first/final
  tiles repeated exact at 129.000/90.011 and 52.492/51.516 ms wall/GPU.

Decision and next:

- Accept persistent two-tile layer-0/layer-1 KV ownership as the next native
  prefill checkpoint. Next turn the pair into a reusable backward-extending
  tile loop, preserving per-tile C0 evidence while avoiding a premature full
  2K or throughput claim.

### 2026-08-16 — Exact layers-0/1 coverage expands to two prompt tiles

Objective:

- Remove the final-tile-only position and cache-row assumptions, then prove the
  complete layers-0/1 schedule at the immediately preceding M1 tile while
  preserving every shorter boundary as an independent control.

Oracle evidence and fixture:

- Reused full layer-0 and layer-1 DwarfStar captures from two fresh 2K
  processes. The layer-0/layer-1 `KVcur`, `hc_ffn_post`, and
  `ffn_moe_topk` full-tensor pairs were byte-identical and retained their
  previously pinned SHA-256 identities.
- Added `prefill-layers01-previous-tile-2048-v1`, a compact decisive fixture
  for positions 1984--2015 containing six tensors and 4,326,912 verified bytes.
  It retains both layers' KV tile, selected expert IDs, and final four-stream
  HC state without duplicating every large intermediate.

Implementation:

- Generalized the 84-dispatch complete layers-0/1 executor from fixed position
  2016 to any aligned 32-row tile within the 2K frontier. KV-prefix length,
  RoPE positions, attention visibility, raw-ring target row, and guard length
  now derive from `position_start`.
- Added `prefill-layers01-row-coverage-probe` with schema
  `rust-star-prefill-layers01-row-coverage-probe-v1`. It executes positions
  1984--2015 and the preserved 2016--2047 control as two complete native tiles,
  each with 49 direct mmap-backed model views.
- Stable JSON explicitly records that the two tiles use captured per-tile KV
  prefixes and therefore does not claim a live inter-tile KV chain or complete
  native prefill.

Target-Mac evidence:

- The warm focused probe passed both tiles C0 exact at 127.196/87.020 ms and
  124.717/87.421 ms wall/GPU, with 49/49 no-copy views for each tile.
- The complete gate passed formatting, all 82 Rust tests, 55 Python tests, all
  239 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. The earlier/final tiles
  remained C0 exact at 128.771/91.582 and 124.334/87.370 ms wall/GPU.

Decision and next:

- Accept arbitrary-position exact 32-row replay and 64 covered prompt rows as
  the next row-coverage checkpoint. Next replace the captured boundary between
  these two tiles with live per-layer KV ownership before scaling the tile loop.

### 2026-08-15 — Native M1 final tile completes layer 1

Objective:

- Extend the direct live layer-0-to-layer-1 HC handoff through the remainder of
  uncompressed layer 1 while preserving the 43-dispatch layer-0 and
  47-dispatch layer-1-Q-A controls as independent commands.

Oracle evidence and fixture:

- Captured twenty full 2K layer-1 tensors twice from fresh pinned DwarfStar
  processes. Every pair was byte-identical. The strict importer pins every
  full-capture SHA-256 before retaining the final 32 rows plus the 2,016-row KV
  prefix needed for rectangular attention.
- Added `prefill-layer1-complete-2048-v1`: nine operations, twenty-one tensors,
  and 26,543,616 verified bytes. The fixture covers Q/KV setup and RoPE, KV
  finalization, FlashAttention, grouped attention output, both HC updates, the
  decomposed token-hash router, routed experts, and the shared expert.

Implementation:

- Added `prefill-layers01-complete-boundary-probe` with schema
  `rust-star-prefill-layers01-complete-boundary-probe-v1`. The optional Metal
  continuation adds the remaining thirty-seven layer-1 dispatches after Q-A,
  for 84 total: 43 in layer 0 and 41 in layer 1.
- The same command buffer consumes the live layer-0 four-stream FFN HC state,
  reconstructs layer 1's 2,048-row contiguous KV input from the captured prefix
  and live final tile, and finishes at `layer1_hc_ffn_post` without a host
  activation seam. All 49 model ranges are direct mmap-backed views.
- The complete command retains and checks 12,878,208 produced FP32 values plus
  384 selected expert IDs across both layers. Stable JSON asserts the complete
  layer-1 final-tile claim while continuing to reject a full-prefill claim.

Target-Mac evidence:

- A focused optimized run preserved the 43-dispatch 25/25-view and 47-dispatch
  30/30-view controls C0 exact, then passed the new 84-dispatch boundary C0
  exact with 49/49 no-copy views at 243.211 ms wall / 192.411 ms GPU.
- The complete gate passed formatting, all 80 Rust tests, 54 Python tests, all
  238 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. The new boundary remained
  exact at 277.821 ms wall / 226.582 ms GPU; the preserved 43- and 47-dispatch
  controls remained exact at 131.947/76.576 and 129.940/78.788 ms wall/GPU.

Decision and next:

- Accept complete native layers 0 and 1 for the isolated final M1 tile. Next
  broaden native retained-row coverage toward the complete 2K prefill rather
  than adding another layer to the isolated tile.

### 2026-08-15 — Native final tile hands layer-0 HC directly into layer-1 Q-A

Objective:

- Preserve the complete exact layer-0 final-tile command as an independent
  control while proving its live four-stream FFN HC output can feed native
  layer-1 batch arithmetic without synchronization or a host activation seam.

Oracle evidence and schedule decision:

- Selected the smallest decisive layer-1 boundary: plain four-stream RMSNorm,
  legacy F16 HC mixer, fused HC split/collapse/learned attention norm, and the
  legacy Q8_0 Q-A batch projection.
- Captured full layer-1 `hc_attn_pre`, `attn_norm`, and `q_lora` tensors from
  two fresh pinned DwarfStar 2K processes. All three pairs were byte-identical.
  Their full SHA-256 values are
  `a5ad89aaa9a5c3537c22a26730b918bdde4e07177e72c643059306bbb28439bc`,
  `37b0f1a783c0968445dd77214b2f62b62a0f5dcac01778ca5e0627948c392571`,
  and `ff0a9a83d0f3077c83ad1ba223310a496d3d8b51709da986d97887f4b0836901`.

Implementation:

- Added `prefill-layer1-ingress-2048-v1` and a reproducible strict importer
  that binds both full capture sets before retaining positions 2016--2047.
  The fixture contains three tensors and 1,179,648 verified bytes.
- Added `prefill-layers01-boundary-probe` with schema
  `rust-star-prefill-layers01-boundary-probe-v1`. The narrow Metal ABI accepts
  an optional layer-1 continuation, so the existing v5 layer-0 command still
  stops at exactly 43 dispatches and 25 model views.
- When requested, the same encoder feeds `after_ffn_hc_buffer` directly into
  four layer-1 dispatches and five additional no-copy model views. The new
  boundary retains 7,274,688 produced FP32 values plus the layer-0 router's 192
  selected expert IDs, reports the direct handoff in stable JSON, and keeps
  complete-layer-1 and full-prefill claims false.

Target-Mac evidence:

- The focused optimized run first repeated the preserved v5 control C0 exact
  at 124.696 ms wall / 65.868 ms GPU, then passed the 47-dispatch layers-0/1
  boundary C0 exact with 30/30 no-copy views at 125.125 ms wall / 69.515 ms
  GPU. Both include setup, synchronization, exhaustive readback, and comparison.
- The complete gate passed formatting, all 78 Rust tests, 53 Python tests, all
  237 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. Its layers-0/1 boundary
  remained exact at 129.903 ms wall / 78.876 ms GPU. The 2K sequential
  diagnostic remained exact against its replay oracle and preserved the
  required native-batch mismatch and paired-run ineligibility.

Decision and next:

- Accept the direct live HC handoff and layer-1 Q-A output as the next exact
  native boundary. Extend it through the remaining uncompressed layer-1
  attention and FFN schedule before broadening retained row coverage.

### 2026-08-15 — Native M1 final tile completes layer 0 through the FFN HC tail

Objective:

- Extend the continuous final 32-row token-to-attention-HC boundary through
  layer 0's active M1 FFN router, routed/shared experts, and additive HC tail.

Oracle evidence and schedule decision:

- Traced the pinned M1 production path through legacy F16 FFN ingress/router,
  decomposed softplus/sqrt/token-hash routing, expert-major mapping, fused
  IQ2_XXS paired gate/up weighted SwiGLU with an F16 intermediate, Q2_K routed
  down projection, six-expert sum, three Q8_0 shared projections with flat
  SwiGLU, and additive `kernel_dsv4_hc_expand4`.
- Captured `hc_ffn_pre`, `ffn_norm`, router logits/probabilities/top-k/weights,
  routed weighted-SwiGLU/output, shared output, and `hc_ffn_post` from two fresh
  pinned 2K DwarfStar processes. All ten full payload pairs were byte-identical.
  The strict importer binds their exact sizes and full SHA-256 identities
  before retaining positions 2016--2047.
- DwarfStar graph capture disables the fused routed pair for observability, but
  its production source contract requires the retained F16 intermediate and
  routed output to be bit-identical. Rust Star executes the production fused
  kernel and compares those outputs to the retained diagnostic boundary.

Implementation:

- Added `prefill-ffn-output-2048-v1`, its reproducible strict importer, Rust
  shape checks, and Python manifest/payload validation.
- Evolved `prefill-layer0-boundary-probe` to schema v5. One Rust-owned command
  buffer now runs 43 compute dispatches continuously from the final 32 token
  IDs through `hc_ffn_post`. The boundary wraps all 25 GGUF weight ranges as
  read-only no-copy Metal views and compares 6,979,776 produced FP32 values by
  bit pattern plus 192 selected expert IDs exactly.
- Added the seven decomposed M1 batch-router kernels and wired the native
  expert map, fused IQ2_XXS pair, Q2_K F16 down, sum-six, shared SwiGLU, and HC
  tail kernels behind the narrow Rust/C ABI. Stable JSON now reports both the
  attention and FFN oracle fixtures, the full schedule, retained checksums, and
  an explicit complete-layer-0-final-tile claim while preserving
  `full_prefill_claim: false`.

Target-Mac evidence:

- The focused optimized v5 run was C0 exact with 43 dispatches and 25/25
  no-copy views, reporting 135.893 ms wall and 73.494 ms GPU. Setup,
  synchronization, exhaustive readback, and comparison remain in scope; this
  is correctness evidence, not throughput.
- The complete gate passed formatting, all 76 Rust tests, 52 Python tests, all
  236 differential fixtures, optimized Objective-C/Metal compilation, strict
  validation of all 1,288 required model tensors, every established Metal and
  decoder control, and both steady-state benchmarks. Its v5 run remained C0
  exact at 138.441 ms wall / 81.329 ms GPU with 25/25 no-copy views. The 2K
  sequential diagnostic remained exact against its replay oracle while
  preserving the required 129,280-logit native-batch mismatch and paired-run
  ineligibility.

Decision and next:

- Accept the FFN/MoE/HC suffix as the exact completion of layer 0 for the
  isolated final native tile. Next hand its live HC state into layer 1 and
  broaden row coverage; do not relabel the final-tile gate as full 2K prefill.

### 2026-08-15 — Continuous native M1 final tile crosses attention output and HC post

Objective:

- Extend the exact final 32-row token-to-`kqv_back` boundary through layer 0's
  active grouped Q8 attention output and four-stream HC post-update.

Oracle evidence and schedule decision:

- Traced the M1 path rather than assuming the Metal 4 schedule. On this machine
  TensorOps is unavailable, `ds4_gpu_attention_output_q8_batch_f16_tensor`
  returns unsupported, and DwarfStar therefore uses
  `kernel_mul_mm_id_map0_ne20_8`, `kernel_mul_mm_id_q8_0_f32`,
  `kernel_mul_mm_q8_0_f32`, and `kernel_dsv4_hc_expand4` with F32 activations.
  The final 32 rows use the same 32-row legacy arithmetic tiles as the complete
  2,048-row projection, so no captured predecessor output tile is required.
- Captured full `attn_low`, `attn_out`, and `hc_attn_post` tensors from two
  fresh pinned DwarfStar processes over the canonical 2K prompt. Every pair
  was byte identical. Their full SHA-256 values are
  `edd7304f5f41313b19f432b4077c6bf08c97605c0bbaeb002e6afc749b768fa9`,
  `99d1251d729592a383a208258bf96579c5025fb35ed3215be26d0616f2094871`,
  and `19c0a248fce8b530bbc39f4c7e7ba0ff277b97466d8e2bc86ce199639b6739c1`.

Implementation:

- Added `prefill-attention-output-2048-v1` and a strict importer that requires
  both full repeated capture sets, their exact sizes, and their pinned hashes
  before retaining the final `32×8192`, `32×4096`, and `32×16384` tiles.
- Evolved `prefill-layer0-boundary-probe` to schema v4. Its one Rust-owned
  command buffer now executes 22 compute dispatches from token IDs through
  `hc_attn_post`. The two output weights are additional mmap-backed no-copy
  views, bringing the boundary to 13/13 pointer matches.
- Imported DwarfStar's exact expert-map/grouped-matmul schedule from the shared
  Metal source, retained the native dense Q8 projection, and applied the split
  post/combination weights directly to the live residual HC buffer. The five
  joined fixtures require 5,521,408 produced FP32 values to match by bit
  pattern, in addition to guarded cache checks and the reconstructed KV seam.

Target-Mac evidence:

- The focused optimized live run was C0 exact with 22 dispatches and 13/13
  no-copy model views, reporting 63.477 ms wall and 5.787 ms GPU. Setup,
  synchronization, exhaustive readback, and comparison remain in the wall
  interval; this is correctness evidence, not throughput.
- The complete target-Mac gate passed formatting, all 75 Rust tests, 51 Python
  tests, all 235 differential manifests, optimized Objective-C/Metal
  compilation, strict validation of all 1,288 target tensors, every established
  Metal/decoder control, and both steady-state benchmarks. Its fresh v4 run
  reported 65.043 ms wall and 7.116 ms GPU with 13/13 no-copy views. The long
  2K sequential diagnostic remained exact against its decode-replay oracle at
  19.138 tokens/s over 107012.259 ms and preserved the expected native-batch
  mismatch and paired-protocol ineligibility.

Decision and next:

- Accept the grouped Q8 attention output and HC post-update as the exact next
  segment of the native M1 final tile. Extend the same command buffer through
  the layer-0 FFN ingress and batched MoE/HC tail next; complete 2K native state
  construction and sparse ratio-4 indexed attention remain required before
  paired claims.

### 2026-08-15 — Continuous native M1 final tile crosses zero-prefix attention

Objective:

- Extend the exact final 32-row token-to-KV boundary through DwarfStar's active
  zero-prefix batched FlashAttention read and inverse-RoPE output.

Oracle evidence and schedule decision:

- Traced layer 0's `ratio == 0` batch path. DwarfStar attends over the complete
  contiguous 2,048-row `KVcur` tensor, uses a 128-token causal window, executes
  the non-vector block-map plus `kernel_flash_attn_ext_f16_dk512_dv512`, and
  then applies inverse tail RoPE to 64×512 output heads.
- Captured full `kqv_out` and `kqv_back` tensors from two fresh pinned
  DwarfStar processes over the canonical 2K prompt. Both pairs were byte
  identical. Their full SHA-256 values are
  `0678586b3fa811f40d053b4cef4f40c9172bb497d0ac1862dc43de7f5b04a1d9`
  and `79ec161cba188d6e9d1e0b3a84e3071e6c0cba53b5b7a56406d9ac6cf8ddabcc`.
- Kept continuity without replaying the first 2,016 rows: the runtime loads the
  repeated captured KV prefix as input, overwrites rows 2016--2047 with its
  live post-FP8 tile, and evaluates the rectangular 32-query by 2,048-key
  problem. The production range helper is intended for this geometry; the
  target-Mac C0 gate proves it matches the final rows of the square oracle.

Implementation:

- Added `prefill-attention-read-2048-v1` and its reproducible strict importer.
  The importer requires both complete capture pairs and their pinned hashes
  before retaining the 2,016×512 KV prefix and final 32×64×512 `kqv_out` and
  `kqv_back` tiles.
- Evolved `prefill-layer0-boundary-probe` to schema v3. One command buffer now
  executes 18 compute dispatches continuously from token IDs through the full
  final-tile attention result. It adds the F32-to-F16 full-KV stage, 8×64
  block-map specialization, eight-simdgroup non-vector attention kernel, and
  inverse RoPE. The sink weights are an eleventh mmap-backed no-copy model
  view.
- The four joined fixtures require 4,603,904 retained produced FP32 values to
  match by bit pattern. The probe also checks the complete reconstructed KV
  tensor, so captured prefix rows and the live replacement seam cannot drift
  silently.

Target-Mac evidence:

- The optimized live run was C0 exact across every prior and new boundary,
  preserved 11/11 mmap pointer identities, and reported 59.825 ms wall and
  5.249 ms GPU. Setup, synchronization, exhaustive readback, and comparison
  remain in the wall interval; this is correctness evidence, not throughput.
- The focused host suite passed 74 Rust tests, optimized Objective-C/Metal
  compilation, and strict validation of the new differential fixture.
- The complete target-Mac gate passed formatting, all 74 Rust tests, 50 Python
  tests, all 234 differential manifests, optimized Objective-C/Metal
  compilation, strict target-model inspection, every established
  Metal/decoder control, and both steady-state benchmarks. Its fresh v3 run
  reported 59.465 ms wall and 4.867 ms GPU with 11/11 no-copy views. The long
  2K sequential diagnostic remained exact against its decode-replay oracle at
  19.484 tokens/s over 105110.880 ms and preserved the expected native-batch
  mismatch and paired-protocol ineligibility.

Decision and next:

- Accept the rectangular zero-prefix attention read plus inverse RoPE as the
  exact continuation of the native M1 final tile. Next extend this same batch
  command buffer through the grouped Q8 attention output and HC post-update;
  complete 2K native state construction remains required before paired claims.

### 2026-08-15 — Continuous native M1 final tile reaches guarded KV storage

Objective:

- Extend the exact token-ID-to-Qcur final tile through the pinned M1 batch KV
  finalization path without claiming a complete 2K cache construction.

Oracle evidence and schedule decision:

- Traced the active batch path as standalone KV tail RoPE, in-place E4M3FN
  simulation for the 448 non-rotary channels, then F32-to-F16-to-F32 rounding
  and raw-ring scatter. It does not use the fused one-row decode KV-store
  kernel. Zero-prefix batch attention consumes the contiguous batch KV tensor;
  the raw ring preserves state for later chunks and decode.
- Captured full layer-0 `KVrope` and `KVcur` tensors from two fresh pinned
  DwarfStar processes over the canonical 2K prompt. Each corresponding pair
  was byte-identical. The full-capture SHA-256 values are
  `9a9642491e6ae5018a5dc5012eac67884458d4439f58a8bedb071c82dc3aaeb7`
  and `bef8d14d805a482960cbf7315ad0efccf211516a27bd767d0884b81f3ad33893`.
- Retained positions 2016--2047 and derived the exact raw-cache payload by
  applying IEEE-754 binary16 round-to-nearest-even and expansion back to
  binary32. Those rows map contiguously to physical ring rows 96--127. Earlier
  rows remain guards because the full 2K scatter aliases ring destinations and
  is outside this final-tile checkpoint.

Implementation:

- Added the strict `prefill-kv-state-2048-v1` fixture and reproducible importer.
  The manifest binds both full captures, records four production kernels, and
  distinguishes captured values from the derived cache-storage artifact.
- Evolved `prefill-layer0-boundary-probe` to schema v2. One Rust-owned Metal
  command buffer now executes 14 dispatches continuously from the final 32
  token IDs through KV RoPE, FP8 simulation, and F16-rounded cache storage.
  It snapshots pre-RoPE KV for the prior control, returns `KVrope`, `KVcur`, and
  the full 128-row raw ring, and requires sentinel rows 0--95 to remain intact.
- Added the F16-to-F32 contiguous conversion pipeline, extended the C/Rust ABI,
  stable JSON, host fixture/report checks, complete-gate integration text, and
  runtime/project documentation. Across the three joined fixtures, 2,506,752
  retained produced FP32 values must match by bit pattern.

Target-Mac evidence:

- The focused optimized live run was C0 exact across every old and new boundary,
  retained 10/10 mmap pointer identities, and proved all 49,152 guard-tile
  elements unchanged. The complete-gate rerun reported 57.703 ms wall and
  3.569 ms GPU for the cold 14-dispatch tile; setup, synchronization, exhaustive
  readback, and guard verification remain in scope.
- The complete gate passed formatting, 72 Rust tests, 49 Python tests, all 233
  differential manifests, optimized Objective-C/Metal compilation, strict
  target-model inspection, and every established Metal/decoder control. The
  long 2K sequential diagnostic remained exact against its decode-replay oracle
  at 19.521 tokens/s over 104914.162 ms and preserved the expected native-batch
  mismatch and paired-protocol ineligibility.
- A final fixture-shape regression test raised the Rust suite to 73 tests; that
  complete host suite and `git diff --check` passed after the documentation
  update.

Decision and next:

- Accept token IDs through final-tile guarded KV storage as one continuous
  native M1 boundary. Next capture and implement the zero-prefix batch
  FlashAttention read/output for this same tile before broadening retained rows.

### 2026-08-15 — Continuous native M1 final tile now starts at token IDs

Objective:

- Remove the captured normalized-activation seam from the native layer-0 batch
  boundary while preserving DwarfStar's actual M1 schedule and exactness.

Oracle evidence and schedule decision:

- Traced the pinned M1 path from token embedding through `kernel_get_rows_f16`,
  four-stream `kernel_repeat_f32`, plain `kernel_rms_norm_f32_4`, the legacy
  `kernel_mul_mm_f16_f32` batch mixer, and fused HC split/collapse/learned norm.
  The M1 does not select the newer fused TensorOps projection for this path.
- Captured full 2K `hc_attn_pre` and `attn_norm` artifacts in two fresh
  DwarfStar processes. Corresponding hashes were byte-identical, and the
  attention-normalized hash also exactly matched the independently captured
  Q/KV fixture input.
- Retained only the final 32 token IDs and complete final-tile HC/norm outputs,
  while recording the full-capture hashes in a strict differential fixture.

Implementation:

- Added `prefill-layer0-boundary-probe` and schema
  `rust-star-prefill-layer0-boundary-probe-v1`. One command buffer now executes
  ten dispatches continuously from token IDs through Q head RMSNorm/RoPE.
- Wrapped ten GGUF tensor spans as mmap-backed no-copy Metal views, retained a
  pre-RoPE Q snapshot with an in-command-buffer blit, and required the HC and
  Q/KV fixtures' 2,457,600 produced FP32 values to match by bit pattern.
- Added the reproducible fixture importer, differential manifest, stable JSON,
  CLI and complete-gate integration, host tests, and runtime documentation.
  The report explicitly rejects full-layer, full-prefill, and throughput claims.

Target-Mac evidence:

- The focused live run retained 10/10 mmap pointer identities and matched every
  boundary bit-for-bit. It reported 56.433 ms wall and 2.544 ms GPU for the
  cold ten-dispatch tile; setup, synchronization, and exhaustive readback remain
  in scope.
- The complete gate passed formatting, 72 Rust tests, 49 Python tests, all 232
  differential manifests, optimized Objective-C/Metal compilation, strict
  target-model inspection, and every established Metal/decoder control. The
  new boundary repeated C0 exact at 55.079 ms wall / 2.561 ms GPU. The long 2K
  sequential control remained exact against its decode-replay oracle at 19.579
  tokens/s over 104602.439 ms and preserved the expected native-batch mismatch.

Decision and next:

- Accept token IDs through Qcur as one continuous native M1 final-tile boundary.
  Extend it forward through KV RoPE/storage and batched attention before
  broadening row and layer coverage toward a complete eligible prefill path.

### 2026-08-15 — Native M1 batch boundary extended through layer-0 Q/KV setup

Objective:

- Extend the first exact batch projection through the actual layer-0 M1
  Q/KV schedule without substituting a debug-only or newer-GPU path.

Oracle evidence and schedule decision:

- Captured `q_lora`, `q_lora_norm`, `KVraw`, `KVnorm`, and `Qcur` across the
  canonical 2K prompt in two fresh DwarfStar processes. Every corresponding
  payload was byte-identical; the repeated `q_lora` hash also exactly matched
  the earlier boundary capture.
- Captured `Qraw` twice in two additional fresh processes. Both 256 MiB files
  were byte-identical. The hook selects the explicit Q-B path, but this does
  not replace active production arithmetic on the pinned M1 build: its fused
  Q-B/F16 entry point is unavailable and the normal schedule falls back to the
  same legacy Q8 batch projection followed by fused head RMSNorm/RoPE.
- Retained only positions 2016 through 2047, one complete legacy 32-row tile,
  while binding all seven full 2K capture hashes in a reproducible fixture.
  The seven payloads total 9,306,112 bytes.

Implementation:

- Added `prefill-qkv-boundary-probe` and schema
  `rust-star-prefill-qkv-boundary-probe-v1`. One Rust-owned Metal command wraps
  five GGUF ranges without copying and dispatches Q-A, KV-A, fused Q/KV learned
  RMSNorm, Q-B, and Q head RMSNorm/RoPE with DwarfStar's exact row/position
  geometry.
- Added a strict importer, differential manifest, stable privacy-limited JSON,
  CLI and complete-gate integration, host fixture/report tests, and runtime
  ownership documentation. The report explicitly rejects a full-prefill or
  throughput interpretation.

Target-Mac evidence:

- The first focused live run retained 5/5 mmap pointer identities and matched
  all 2,195,456 produced FP32 values bit-for-bit. It reported 11.200 ms wall and
  2.346 ms GPU for the isolated five-dispatch tile.
- The complete gate passed formatting, 70 Rust tests, 48 Python tests, all 231
  differential manifests, optimized Objective-C/Metal compilation, strict
  target-model inspection, and every established Metal/decoder control. The
  new boundary repeated C0 exact at 12.023 ms wall / 2.817 ms GPU. The long 2K
  sequential control remained exact against its decode-replay oracle at 19.336
  tokens/s over 105913.888 ms and preserved the expected native-batch mismatch.

Decision and next:

- Accept the complete layer-0 Q/KV setup and Q-head boundary as native M1
  arithmetic. Move the input boundary backward through batch HC mixing and
  learned attention norm toward token IDs; in parallel sequence, extend this
  output boundary through KV RoPE/storage and batched attention.

### 2026-08-15 — First native M1 batched-prefill arithmetic boundary isolated

Objective:

- Trace the actual DwarfStar 2K prefill schedule on the M1 Ultra and implement
  the first exact Rust-owned batch kernel boundary without claiming a complete
  prefill path.

Oracle evidence and corrected hypothesis:

- Two fresh 2K graph captures were byte-identical for both layer-0
  `attn_norm` and `q_lora`. The final `attn_norm` row exactly equals the same
  row produced by 2,048 sequential one-token evaluations, proving the schedule
  divergence begins after normalization.
- All 1,024 final-row `q_lora` values differ between schedules; maximum absolute
  error is `3.62396240234375e-05`.
- A live diagnostic run printed that Metal 4 tensor kernels are disabled on the
  M1 Ultra. The earlier source-only assumption that this machine selected the
  retained N128 TensorOps kernel was false. The actual path is the legacy
  `kernel_mul_mm_q8_0_f32` with 32-row tiles, 128 threads, and 6,144 bytes of
  threadgroup memory. A repeated live dump matched the original batch capture
  SHA-256 exactly.

Implementation:

- Added a reproducible importer and compact fixture containing the final 128
  normalized rows, their native batch outputs, and the sequential final-row
  control. The three payloads total 2,625,536 bytes and bind two independent
  full-batch captures.
- Added a narrow C ABI and Rust command that wraps
  `blk.0.attn_q_a.weight` from the shared model mmap, specializes DwarfStar's
  batch kernel with both bounds constants false, dispatches four by sixteen
  groups, then runs the existing four-simdgroup one-row kernel over the shared
  final input.
- The live M1 probe matched all 131,072 batch outputs and all 1,024 sequential
  control outputs bit-for-bit. It reported 3.145 ms wall / 1.050 ms GPU for the
  isolated 128-row batch dispatch and retained the expected 1,024/1,024
  cross-schedule mismatch. These are boundary timings, not prefill throughput.
- The initial standalone TensorOps attempt was preserved as a failed
  hypothesis during development: it first failed because the conditional
  kernel was absent, then produced different arithmetic when force-enabled.
  The live DwarfStar device log resolved the discrepancy and the implementation
  was corrected to the native M1 path.

Validation:

- The complete target-Mac gate passed: formatting, 68 Rust tests, optimized
  Objective-C/Rust compilation, 47 Python tests, all 230 differential
  manifests, strict model inspection, and every retained live Metal control.
- The new boundary repeated C0 exact at 2.490 ms wall / 0.410 ms GPU. The long
  2K sequential control again matched its decode-replay oracle at 18.478
  tokens/s over 110832.479 ms and retained the expected 129,280-logit native
  batch mismatch.

Next:

- Complete documentation and the full target-Mac runtime gate, publish this
  checkpoint, then extend the batch path through layer-0 Q/KV setup.

### 2026-08-15 — 2K state ownership reached the batched-prefill boundary

Objective:

- Generalize cold state past 128 positions, reach the first protocol context,
  and determine whether sequential exact decode can initialize eligible
  benchmark state without importing DwarfStar's batched prefill path.

Oracle evidence:

- Captured the canonical first 2,048 token IDs and two fresh native DwarfStar
  2K frontier-logit files. Both batch captures were byte-identical after FP32
  reconstruction, selected token 15342, and have payload SHA-256
  `7b5e851884bbb0aa8c2a249c8497af0feccb267cbd0a40e0a4a5aee584ecbfaf`.
- Added a narrow DwarfStar decode-replay capture helper: one-token cold prefill
  followed by 2,047 ordinary `ds4_session_eval` calls over the same canonical
  IDs. Two fresh replay logits were byte-identical, selected token 15342, and
  have SHA-256
  `aa657efb7a5cb7108ee639ea797eb4f1c223f36360d356f96674499935d1f405`.
- The batched and replay tensors differ at all 129,280 logits with maximum
  absolute error 2.325326. Preserve this as an arithmetic-boundary result, not
  a reason to weaken C0 or relabel sequential decode as prefill.

Implementation:

- Added an immutable context-capacity field to the Rust/Objective-C layer ABI.
  Raw KV now uses a 128-row physical ring with ordered wrapped-window staging;
  compressed capacity is derived independently from each layer's ratio.
- Corrected FlashAttention padding from context-scaled allocation to the two
  32-row tiles actually consumed by the vector kernel. Added an explicit guard
  after 512 ratio-4 compressed rows until sparse indexer selection exists.
- Kept synchronized multi-layer control scratch at the uniform ratio-4 maximum
  so those legacy controls can safely reuse unscoped buffers across the layer-2
  transition while compressed-cache ownership remains ratio-specific.
- Added `prefill-frontier-probe`, a three-payload differential fixture covering
  canonical token IDs, native batch logits, and decode-replay logits, stable
  diagnostic JSON, CLI/runtime-gate coverage, and fixture/report tests.

Target-Mac evidence and decision:

- The unchanged one-token/position-127 cold control remained C0 exact and
  reported 18.533 evaluated decode positions/s after the ring rewrite.
- The 2K sequential run reported 18.603 tokens/s over 110090.070 ms, matched
  every decode-replay logit bit, and selected token 15342. It explicitly
  reports batched-prefill C0 false and is not an engine measurement.
- The final complete gate passed 66 Rust tests, 46 Python tests, all 229
  differential manifests, optimized Objective-C/Metal compilation, strict
  target inspection, and every retained live Metal control. Its cold command
  reported 843.365 ms for one-token initialization and 17.670 evaluated decode
  positions/s. Its independent 2K sequential diagnostic again matched every
  replay-logit bit at 18.252 tokens/s over 112206.922 ms and preserved the
  expected 129,280-logit batched-prefill mismatch.
- Accept context-sized state ownership and the raw-ring transition as the next
  exact decode boundary. Reject the hypothesis that sequential decode can stand
  in for native batched prefill under C0. Import batched prefill next, then add
  sparse indexed decode and the measurement producer.

### 2026-08-15 — Captured initial state replaced by exact cold prefill

Objective:

- Construct position-0 raw KV and compressor state from the live prompt token
  rather than copied oracle rows, while retaining the complete position-127
  transcript and cache-emission gates.

Oracle evidence:

- Verified that the first 26-space prefix of the pinned benchmark input
  tokenizes to the single token 36662. Two fresh pinned DwarfStar processes
  wrote the full logits immediately after raw one-token prefill; both 517,120
  byte payloads were identical, selected token 201, and have SHA-256
  `a4973c1e1f53bf1659a9a15e66c3186d03432810e7606387c55f8ee083ffba35`.
- Added a strict prefill decode-step fixture and reproducible two-capture
  importer. The differential registry now contains 228 manifests.

Implementation:

- Added an explicit cold initial-state mode to the stable Rust/Objective-C
  layer boundary. Position 0 clears raw caches, recurrent compressor state, and
  persistent compressed rows, then the ordinary layer path writes raw KV row 0
  and seeds each compressor from its live attention-normalization activation.
- Position 1 in cold mode continues that state and cannot execute the captured
  cache-row/compressor-prime initialization branch. Existing commands retain
  the captured mode as independent regression controls.
- Added `cold-prefill-decoder-probe`, stable diagnostic JSON, CLI help, complete
  gate coverage, host fixture/report tests, and ownership documentation. It
  compares every position-0 logit before running the unchanged 127-position
  feedback loop and final ratio-128 checks.

Target-Mac evidence and decision:

- The first live cold-state process matched all prefill logits, all 128
  committed tokens, final token 33148, every final logit, and both first live
  ratio-128 rows. It reported 3379.018 ms prefill and 0.729 evaluated
  positions/s while paying cold model-residency costs.
- An immediate fresh-process repeat remained bit-identical and reported
  1130.755 ms prefill plus 18.246 evaluated positions/s with resident weights.
  Preserve both as diagnostics; their residency conditions differ and the
  one-token prompt is not a paired-protocol frontier.
- The complete gate passed 64 Rust tests, 45 Python tests, all 228 differential
  manifests, optimized Objective-C/Metal compilation, strict model inspection,
  every retained live Metal control, and both position-127 commands. Its cold
  command reported 1078.665 ms prefill and 18.311 evaluated positions/s.
- Accept removal of captured initial state at the one-token frontier. The next
  checkpoint is multi-token prefill with context-sized raw/compressed ownership
  and arbitrary-frontier initialization, followed by the measurement producer.

### 2026-08-15 — Integrated decoder crossed position 127 exactly

Objective:

- Extend the exact greedy decoder from its four-position control through the
  complete 128-token oracle transcript and verify the first ratio-128
  compressor emissions in live end-to-end execution.

Oracle evidence:

- Ran the pinned DwarfStar executable twice in fresh processes for 127
  evaluated positions, retaining initial committed token 201 plus all 127
  selected tokens and the final full-vocabulary logits. Both captures were
  byte-identical. The token payload SHA-256 is
  `95aedd05c1843ed9638bcd24e93b9ed4cde360a341f8c720c7efc908c3586697`;
  the position-127 logits SHA-256 is
  `1258d4c3ab662f72ac1f70be80ddbc8f9cf72db35b4450231369f3b6ae07c895`.
- Imported one strict decode-step fixture containing 128 token IDs and all
  129,280 final logits. The final selected token is 33148, and the differential
  registry now contains 227 manifests.

Implementation:

- Added `position127-decoder-probe`, its stable diagnostic JSON schema, CLI and
  complete-gate coverage. One prepared executor advances positions 1–127,
  feeds each lowest-ID argmax into the following step, and retains all
  layer-scoped raw and compressed state.
- Added a synchronization-only, layer-scoped compressed-cache row readback.
  The final C0 phase compares the complete token transcript, final logits, and
  live layer-3/layer-5 ratio-128 rows. Its timer ends before those readbacks and
  comparisons.
- Added a reproducible frontier-fixture importer plus Rust fixture/JSON and
  Python manifest tests.

Failed hypotheses and correctness fixes:

- The first integrated run diverged at committed token 16. The decoder used a
  fixed compressed-RoPE origin; the correct origin is the beginning of the
  completed compression window, `position + 1 - ratio`.
- After token exactness was restored, the live ratio-128 row was zero. A
  temporary synchronized work-buffer check localized the failure before the
  persistent-cache blit: packed KV, score, and softmax scratch allocations had
  room for eight rows instead of the active 128-row window. All compressor
  scratch is now sized from its ratio. The temporary diagnostic was removed.

Validation and decision:

- The live M1 Ultra command reproduced all 128 tokens, final token 33148, every
  final logit, and both first ratio-128 rows bit-for-bit. The complete gate
  reported 16.493 evaluated positions/s over 127 positions and passed 62 Rust
  tests, 44 Python tests, all 227 differential manifests, the optimized
  Objective-C/Metal build, strict model inspection, and every retained live
  Metal control.
- Accept the position-127 integrated frontier as C0 exact, but not as paired
  throughput. It begins from captured one-token cache/compressor state. The
  next checkpoint is cold prefill plus arbitrary-frontier initialization,
  followed by the engine-measurement producer.

### 2026-08-15 — Persistent compressed memory reused exactly at position 4

Objective:

- Remove the captured position-3 ceiling at the first semantically meaningful
  boundary: execute the next sampled token while retaining and consuming the
  ratio-4 compressed rows emitted by the preceding step.

Oracle evidence:

- Ran the preserved pinned DwarfStar executable twice in fresh processes with
  one prefill token and four generated tokens, batch-dumping every retained
  position-4 boundary for layers 0–42. All 1,376 payload pairs were
  byte-identical; their ordered aggregate SHA-256 is
  `73f0af1597da79b5e53c19716822c105df6256e353ee18d8c013d4809c74c3cb`.
- Captured the five terminal output-head tensors twice in two additional fresh
  processes. All pairs were byte-identical with aggregate SHA-256
  `8ca919fe0b7cc11ff861b9de6f576ba0f736b1fb4376878ff4d464cabc594054`.
  Position 4 consumes token 262 and lowest-ID argmax selects token 1554.
- Imported 43 complete layer fixtures plus one output-head fixture. The strict
  differential registry now contains 226 manifests.

Implementation:

- Replaced the four-row raw-cache allocation with a 128-row frontier and added
  32 persistent compressed attention/indexer rows per compressed layer. The
  bounded executor accepts positions 1–127, which covers the protocol's future
  128-token decode without yet claiming that full execution path.
- Preserved emission-step ordering: a new compressed row remains in its work
  buffer for the attention encoder that produced it, then a Metal blit commits
  it to persistent cache storage. Later positions stage all committed rows;
  emission positions stage prior committed rows followed by the new work row.
- Extended both explicit-input and feedback-loop C0 paths through position 4,
  updated their schemas to v2, and retained the earlier three-position
  transformer-only commands as independent controls.

Validation:

- The live M1 Ultra closed loop is C0 exact for all 172 layer/position
  boundaries and all 517,120 logits. It generated
  `361 -> 1915 -> 262 -> 1554`; the complete gate reported 18.798 tok/s
  overall and 18.916 tok/s steady over the diagnostic timed path.
- The timing interval still includes transformer submission/execution, output
  execution, logits transfer, and CPU argmax while excluding correctness
  collection. It remains paired-protocol ineligible because cold prefill and
  arbitrary-frontier initialization are not implemented.
- The complete gate passed 60 Rust tests, the optimized macOS
  Objective-C/Metal build, 43 Python tests, all 226 fixture manifests, strict
  real-model inspection, and every established Metal control.

Decision:

- Accept persistent compressed-cache lifetime and first reuse as exact. The
  next checkpoint is a 128-step generated-token run through the already-sized
  frontier, followed by cold prefill and engine-measurement output.

### 2026-08-15 — Three-position decoder loop closed and timed honestly

Objective:

- Feed exact greedy selections back into the following decoder positions and
  separate exhaustive C0 tensor collection from the measured execution path.

Implementation:

- Added `closed-loop-decoder-probe` and schema
  `rust-star-closed-loop-decoder-diagnostic-v1` without changing the existing
  explicit-input `decoder-output-probe` control.
- The exhaustive pass starts with token 201, derives 361, feeds 361 into
  position 2, derives and feeds 1915 into position 3, and finally selects 262.
  Every retained boundary across 43 layers plus all 129,280 logits per position
  remains C0 bit-identical.
- Added a prepared output-head mode that omits the four intermediate host
  copies while retaining the full logits copy required for CPU greedy argmax.
  The timed pass does not invoke any transformer collector or fixture
  comparison. Pipeline preparation and the exhaustive C0 allocation pass occur
  before its step intervals.
- Stable JSON explicitly classifies the result as diagnostic, records whether
  correctness/logits/argmax occur in the interval, and marks paired-protocol
  eligibility false with the remaining cold-prefill/arbitrary-frontier blocker.
- Added CLI help, runtime-gate coverage, host JSON tests, and runtime/benchmark
  documentation.

Validation and measurement:

- Formatting, 60 Rust tests, the optimized macOS Rust/Objective-C/Metal build,
  JSON parsing, and diff checks pass.
- The complete target-Mac gate passed formatting, 60 Rust tests, 42 Python
  tests, optimized compilation, all 182 fixture envelopes, every retained live
  Metal control, and the new closed-loop command.
- The live model run selected `361, 1915, 262` in both exhaustive and timed
  loops. The full-gate timed steps were 54.610, 51.727, and 53.989 ms, giving
  18.712 complete tok/s and 18.919 steady tok/s over the final two tokens.
- This is not a headline or paired result: it begins from the captured
  position-0 cache/compressor state, uses only three positions, and performs no
  cold prompt prefill.

Next:

- Generalize state allocation and execution to cold prefill and positions past
  three, then drive 128 closed-loop tokens into the engine-measurement contract.

### 2026-08-15 — Exact decoder output boundary completed

Objective:

- Extend the exact 43-layer transformer stack through DwarfStar's output HC
  collapse, learned output normalization, full vocabulary logits, and
  deterministic next-token selection for positions 1–3.

Oracle evidence:

- Captured `result_hc_pre`, `result_hc_weights`, `result_hc`, `result_norm`,
  and `result_output` for terminal decode positions 1, 2, and 3 in two fresh
  pinned DwarfStar processes. All 15 corresponding payload pairs were
  byte-identical; their ordered aggregate SHA-256 is
  `3cda1544443ba220b5019201c0232704cc0f177b8ab644689c09d24e70d40728`.
- Imported three strict output-head fixture envelopes containing 549,920 bytes
  each. Full-logit lowest-ID argmax selects token 361 at position 1, token 1915
  at position 2, and token 262 at position 3.

Implementation:

- Added `decoder-output-probe` and schema
  `rust-star-decoder-output-position-advancing-probe-v1`.
- Added a no-copy Metal output boundary that consumes the retained layer-42 HC
  buffer, wraps five output model ranges directly from the GGUF mmap, and
  dispatches the plain HC norm, F16 HC projection, four-value HC weighting,
  fused HC collapse/learned norm, and 129,280-row Q8_0 vocabulary projection.
- Preserved the existing `layers0-42-decode-probe` as an independent
  transformer-stack control. The new correctness schedule explicitly reports
  44 command buffers and two host waits per step.
- Added stable JSON writing, CLI/runtime-gate integration, output-fixture shape
  and argmax tests, documentation, and a repeated-capture fixture importer.
- The first live attempt differed by one bit at logit 0 while all preceding
  output tensors were exact. Tracing DwarfStar's dispatch selection showed that
  output dimensions above 65,536 force eight simdgroups; using the ordinary
  four-simdgroup projection changes the reduction order. A dedicated
  eight-simdgroup output pipeline restored C0 without changing earlier Q8
  projection controls.

Validation:

- The complete target-Mac gate passed formatting, 59 Rust tests, 42 Python
  tests, optimized macOS compilation, the cross-language artifact contract,
  strict validation of all 182 differential fixtures, every prior live Metal
  control, and the new three-position decoder-output probe.
- All five output tensors were bit-identical at all three positions, including
  every FP32 bit in each 129,280-value logit vector.
- In the final correctness gate the output-head command reported 15.025/2.946
  ms wall/GPU at cold position 1, 1.679/1.355 ms at position 2, and 1.691/1.359
  ms at position 3. These intervals include correctness-oriented scheduling
  and readback and are not token-throughput measurements.
- Generated JSON evidence remains under the ignored
  `rust-star/.work/runtime-target/` directory. The user's two untracked
  console logs remain untouched.

Decision:

- This is an exact explicit-input decoder output boundary, not yet a
  closed-loop generator. The selected token is computed and reported, but the
  next input remains supplied by the fixed regression sequence. No committed
  token-throughput claim is eligible yet.

Next:

- Feed selected tokens directly into subsequent steps, then separate exhaustive
  C0 collection from the timed path and emit the Rust Star
  engine-measurement contract for paired benchmarking.


### 2026-08-15 — Ordered exact execution extended through all 43 layers

Objective:

- Complete the transformer-stack boundary through layer 42 while retaining the
  published four-, six-, and eight-layer controls and every C0 contract.

Oracle evidence:

- Batch-captured positions 0–3 for layers 8–42 twice in fresh pinned
  DwarfStar processes. All 3,448 corresponding payload pairs, totaling
  78,356,784 bytes per run, were byte-identical. Their ordered aggregate
  SHA-256 is
  `0c63afebdf0b0e40ed48f2efb2c06b3b7fe193e420d7456de2a8fdbf84cd1f29`.
- All eight capture CSVs recorded one context/prefill token and three generated
  tokens. Position 3 contained `KVcompress` rows for exactly the 18 even layers
  from 8 through 42 and none of the odd layers.
- Imported 140 new strict fixture envelopes: one position-0 compressor prime
  and complete position-1/2/3 boundaries for each remaining layer. The
  repository now contains 179 validated differential-fixture manifests.

Implementation:

- Replaced layer-4–7 pair-specific complete-fixture, cache-row, compressor
  prime, and compressed-row branches with a checked registry covering layers
  4–42. The registry has explicit bounds and a separate even-layer compressed
  output table.
- Widened the checked contiguous scheduler and Objective-C layer/tail guards to
  43 layers without changing compressor arithmetic or the existing control
  commands.
- Added `layers0-42-decode-probe`, schema
  `rust-star-layers0-42-position-advancing-probe-v1`, atomic JSON output,
  runtime-gate coverage, registry/shape tests, and a reproducible repeated-run
  fixture importer.

Validation:

- The complete target-Mac gate passed: formatting, 57 Rust tests, optimized
  Objective-C/Metal build, 42 Python tests, all 179 fixture manifests, artifact
  interoperability, strict real-model inspection, every established no-copy,
  layer, scheduler, compressor, and benchmark control, plus the new 43-layer
  command.
- The 43-layer path was C0 exact for all 129 layer/position boundaries. The
  final gate reported 837.045/800.215 ms at position 1, 62.799/61.229 ms at
  position 2, and 53.374/49.691 ms at position 3 (wall/summed GPU).
- It retains 43 raw KV caches, validates every position-3 even-layer compressed
  KV row through layer 42, and exposes layer 42's exact 16,384-element HC state.
  Setup, synchronization, and exhaustive comparison remain in scope, so these
  timings are correctness diagnostics rather than token-throughput claims.

Decision:

- Accept the full transformer-stack boundary. The next exact gate is output
  normalization, logits, and token selection; do not call the current command
  a complete decoder until that boundary is implemented and verified.

### 2026-08-15 — Ordered exact execution extended through layer 7

Objective:

- Continue the live position-advancing HC handoff through the next ratio-4 and
  ratio-128 layer pair while retaining every established scheduler and
  compressor control.

Oracle evidence:

- Captured layer-6/7 position-0 primes and complete position-1/2/3 boundaries
  twice in fresh pinned DwarfStar processes. All 197 corresponding payload
  pairs were byte-identical.
- Layer 6's 512-value position-3 compressed KV row has SHA-256
  `63ec3c2a44f2eb9a5d73ba67d05d8699bde608cd43b98382465754bf78352a33`.
  Layer 7 correctly emitted no ratio-128 row at this frontier.
- Added eight strict fixture envelopes. The repository now contains 39
  validated differential-fixture manifests.

Implementation:

- Extended expected-tensor embedding, position-specific fixture selection,
  compressor priming, cache-row ownership, and compressed-emission comparison
  through layers 6 and 7.
- Widened the prepared contiguous-submission contract and Objective-C layer and
  command-tail guards from six to eight layers. The parity-derived compressor
  and biased-top-k paths required no arithmetic changes.
- Added `layers01234567-decode-probe`, stable JSON reporting, atomic output,
  CLI/runtime-gate coverage, host tests, fixture validation, and a reproducible
  importer backed by the existing repeated-capture construction.
- Preserved the four-layer and six-layer commands unchanged as independently
  executed controls.

Validation:

- The complete target-Mac gate passed: formatting, 55 Rust tests, optimized
  build, 42 Python tests, all 39 fixture manifests, artifact interoperability,
  strict real-model inspection, every no-copy and layer/scheduler probe, the
  four-, six-, and eight-layer position-advancing controls, the bounded
  position-127 replay, and both steady-state harnesses.
- The eight-layer path was C0 exact at every retained layer 0–7 boundary. The
  final gate reported 223.448/190.007 ms at position 1, 16.989/16.442 ms at
  position 2, and 17.782/17.160 ms at position 3 (wall/summed GPU).
- These intervals include correctness-oriented setup, synchronization, and
  exhaustive comparison and are not decoder-throughput claims.

Decision:

- Accept the eight-layer frontier. Before repeating pair-specific plumbing 17
  more times, batch-capture the remaining layers and introduce a checked layer
  fixture registry that can scale the same exact executor to layer 42.

### 2026-08-15 — First ratio-128 emissions matched at position 127

Objective:

- Cross the first layer-3 and layer-5 ratio-128 compressor emissions with a
  narrowly scoped, independently repeatable GPU replay, while making no token
  sampling or complete-decoder claim.

Oracle evidence:

- Captured positions 0 through 127 plus the position-127 `KVcompress` emission
  for layers 3 and 5 twice in fresh DwarfStar processes. All 258 corresponding
  activation and output payload pairs were byte-identical.
- The layer-3 emitted row has SHA-256
  `a0416831e464ab8652402b7fbcf854f1a1a401e878906379cc0382b1d1edcc30`;
  layer 5 has SHA-256
  `e63dc05a3efa88affe2e1483f88ea7a9212006730d368f193ff940abd66b230c`.
- Added two strict layer-segment fixtures, each containing 128 independently
  repeated 4,096-value `attn_norm` rows and one 512-value compressed output.
  The repository now has 31 validated differential-fixture manifests.

Implementation:

- Generalized the compressor pack/reduction kernels across ratio 4 and ratio
  128 while preserving the existing ratio-4 state-shift behavior.
- Added a standalone Rust-owned replay boundary that wraps the APE, KV, gate,
  and norm weights directly from the GGUF mmap, owns the 128-row activation and
  compressor state, encodes 263 dispatches in one command buffer per layer, and
  emits only at position 127.
- Added stable JSON reporting, a CLI command, complete-gate coverage, strict
  fixture tests, and a reproducible importer that rejects disagreement between
  repeated oracle captures.
- The report records the external activation boundary and explicitly sets
  sampling and full-decoder claims to false.

Validation:

- The complete target-Mac gate passed: formatting, 53 Rust tests, optimized
  build, 42 Python tests, all 31 fixture manifests, artifact interoperability,
  strict real-model inspection, every earlier no-copy/layer/scheduler probe,
  the new ratio-128 replay, and both steady-state harnesses.
- The existing four-layer and six-layer position-advancing controls remained C0
  exact through positions 1, 2, and 3 after the shared compressor path changed.
- Layer 3's 512-value position-127 output was C0 exact at 32.461 ms wall and
  8.236 ms GPU; layer 5 was C0 exact at 6.881 ms wall and 6.273 ms GPU. These
  include setup, synchronization, and comparison and are not throughput claims.

Decision:

- The first ratio-128 emissions are now correctness-gated. Preserve this replay
  as a bounded control and resume extending the ordered executor through the
  remaining decoder layers; do not describe the replay as token generation.

### 2026-08-15 — Compressor ownership generalized through layers 4–5

Objective:

- Replace the layer-2/3-specific compressor schedule with parity-derived
  ownership, extend the exact position-advancing executor through the next
  ratio-4/ratio-128 pair, and keep the published four-layer path as a control.

Oracle evidence:

- Captured all retained layer-4/5 boundaries at positions 0–3 twice in fresh
  DwarfStar processes. All 197 corresponding payloads were byte-identical.
- Layer 4 emitted its 512-value `KVcompress` row at position 3; both captures
  have SHA-256
  `30550322349818d5524c864ca624613c94f18feea0c2f9a57186afd9c8fff5a1`.
  Layer 5 correctly emitted no compressed row at this ratio-128 frontier.
- Added eight strict fixture envelopes: one position-0 compressor prime and
  complete position-1/2/3 boundaries for each layer. The repository now has 29
  validated differential-fixture manifests.

Implementation:

- Derived compressor ratio, indexer ownership, and emission cadence from layer
  parity for every compressed layer. Even layers use ratio 4 and own the
  indexer compressor; odd layers use ratio 128 and do not.
- Extended exact layer scheduling and the checked prepared-submission tail
  through layer 5. The common executor accepts a contiguous three-to-six-layer
  prefix, while the existing layers-0/3 entry point remains unchanged.
- Added the `layers012345-decode-probe` command, atomic JSON reporting, runtime
  gate coverage, and independently embedded layer-4/5 fixtures and primes.
- Added a reproducible importer that verifies both oracle captures before
  creating the canonical layer-4/5 fixture envelopes.

Validation:

- The complete target-Mac gate passed: formatting, 51 Rust tests, optimized
  build, 41 Python tests, all 29 differential fixtures, cross-language artifact
  checks, strict inspection of 1,288 required model spans, every incremental
  Metal probe, both position-advancing controls, and steady-state benches.
- The integrated four-layer control remained C0 exact at positions 1–3. The
  new six-layer path was also C0 exact at every retained boundary and reported
  143.285/108.618 ms, 14.098/13.178 ms, and 11.552/10.111 ms wall/summed GPU at
  positions 1, 2, and 3 respectively.
- These timings include correctness-oriented execution and are diagnostics,
  not decoder-throughput or end-to-end model-speed claims.

Decision and next gate:

- The next distinct compression boundary is position 127. Plan a bounded
  recurrent-state replay using the pinned oracle token sequence, beginning
  with layer 3 if necessary, and explicitly avoid presenting externally
  supplied tokens as sampling or a complete decoder.

### 2026-08-15 — First ratio-4 compressed KV emission matched at position 3

Objective:

- Cross the first compressed-attention boundary with real recurrent state,
  preserve the existing four-layer scheduler/cache controls, and require the
  emitted row plus all downstream boundaries to remain C0 exact.

Oracle evidence:

- Captured every retained layer 0–3 boundary at position 3 twice in independent
  fresh DwarfStar processes. All 131 corresponding payloads were byte-identical.
- Resolved the position-3 token as 1915 by uniquely matching the layer-0 hash
  router row. The standard selected routes were `[133,217,222,94,234,246]`,
  `[107,58,141,226,233,88]`, `[90,98,196,23,62,19]`, and
  `[64,87,198,214,128,1]` for layers 0 through 3.
- Captured layer 2's 512-value `KVcompress` row twice; both copies have SHA-256
  `73cbd1e2e062b9b52e67223b8fe024a7992eeaf9be49a461ebdf7d659ebd32c8`.
  Also captured independent layer-2/3 position-0 `attn_norm` inputs so Rust Star
  can prime the recurrent state without substituting a synthetic activation.
- Added four position-3 complete fixture envelopes and two position-0 prime
  envelopes. Layer 2's complete fixture contains 33 tensors and 741,808 bytes;
  the other complete fixtures contain 32 tensors and 739,760 bytes each.

Schedule correction:

- The earlier journal/README statement that layers 2 and 3 both used ratio 4
  was incorrect. DwarfStar's schedule alternates ratio 4 on even compressed
  layers and ratio 128 on odd compressed layers. Therefore layer 2 emits both
  attention and indexer compressed rows at position 3; layer 3 only updates its
  attention state and will first emit at position 127. Historical entries are
  retained unchanged; this entry and Current State are authoritative.

Implementation:

- Added persistent layer-scoped attention-compressor state for layers 2/3 and
  indexer-compressor state for layer 2, including exact position-0 priming.
- Mirrored the pinned M1 Ultra path: paired F16 matvec with two rows per
  threadgroup, separate one-row APE state store, and the legacy single-emission
  concat/softmax/multiply/sum reduction. The newer M3/M5 fused store and direct
  ratio-4 pool paths are not used.
- Added learned weighted RMS norm, compressed RoPE, ratio-4 state shift, FP8 KV
  quantization, and indexer Hadamard/FP4 QAT. The emitted layer-2 KV row is
  staged after four raw rows and consumed by FlashAttention at position 3.
- Grew each raw cache to four rows and the mixed attention staging capacity to
  five rows. Layer 2 now binds 33 no-copy model ranges, layer 3 binds 29, and
  layers 0/1 remain at 25.

Target-Mac evidence:

- `layers0123-decode-probe` passed at positions 1, 2, and 3. The emitted layer-2
  compressed row matched all 512 FP32 bit patterns, and every retained
  attention, router, expert, HC, and raw-cache boundary remained exact through
  layer 3.
- The first passing run reported 110.896/77.239 ms at position 1,
  8.863/8.123 ms at position 2, and 10.455/9.618 ms at position 3
  (wall/summed GPU). These intervals include correctness-probe behavior and are
  not decoder-throughput measurements.
- The pre-documentation gate passed 48 Rust tests and the optimized macOS run.
  The completed repository gate passed formatting, 49 Rust tests, 40 Python
  tests, all 21 differential fixture envelopes, the optimized macOS build, the
  strict 1,288-tensor target recipe, and every earlier incremental,
  synchronized, chained, and steady-state Metal control. Its integrated
  position-advancing run reported 110.111/74.626 ms at position 1,
  7.014/6.520 ms at position 2, and 9.448/7.164 ms at position 3
  (wall/summed GPU), all C0 exact.
- The first full-gate attempt exposed a lifetime bug outside the chained path:
  synchronized layers 2 and 3 shared compressor scratch keys despite their
  different state sizes. Compressor and indexer allocations are now always
  keyed by layer identity. The failed control passed before the complete gate
  was rerun from the beginning.

Next:

- Generalize the compressor schedule/state plumbing for the next layer pair,
  then design a bounded position-127 oracle-token gate for layer 3's first
  ratio-128 emission.

### 2026-08-15 — Four-layer state advanced across two exact decode positions

Objective:

- Replace the fixed token-201/position-1 assumption with the smallest real
  position-advancing loop while preserving the proven four-layer executor and
  keeping compressed-cache behavior outside the claimed boundary.

Oracle fixture:

- Confirmed the first two generated token IDs as 201 and 361 and captured every
  retained layer 0–3 boundary at position 2 twice in separate fresh DwarfStar
  processes.
- Each layer fixture contains 32 tensors and 739,760 verified bytes. All paired
  payloads were byte-identical; layer 0 declares 30 operations and layers 1–3
  declare 28 each.

Implementation:

- Added explicit token and position propagation through Rust, the stable C ABI,
  and the Objective-C shim. RoPE, inverse RoPE, raw-cache target rows, visible
  attention geometry, and hash-router tokens now advance with the step.
- Added `layers0123-decode-probe` and schema
  `rust-star-layers0123-position-advancing-probe-v1`. One prepared executor
  submits four command buffers with one tail wait at each position and retains
  four independent three-row cache allocations across both steps.
- Strengthened cache validation to cover every retained row and the newly
  written row after DwarfStar's exact FP16 cache-store rounding. The declared
  output handoff is layer 3's final 16,384-element HC state.
- The first position-2 run exposed a stale launch assumption: the FP32→FP16
  attention staging dispatch covered only the old 1,024-element/two-row shape.
  Scaling its threadgroups to the visible element count covered all 1,536
  elements and restored exact attention output.

Target-Mac evidence:

- Position 1/token 201 grew every cache to two rows and matched all four pinned
  layer fixtures. Position 2/token 361 preserved that history, grew every cache
  to three rows, and matched all four new fixtures at every retained boundary.
- The first successful run reported 103.106 ms wall / 70.862 ms summed GPU for
  the cold position-1 step and 7.333 ms wall / 6.347 ms summed GPU for position
  2. These values include probe correctness work and are not token throughput.
- The complete host/fixture portion passed 48 Rust tests, 38 Python tests, all
  15 differential fixtures, the optimized macOS build, and the strict
  1,288-tensor target recipe. After an output-poll cancellation, validation
  resumed from the first unexecuted Metal stage: every remaining incremental,
  full-layer, synchronized, chained, stateful, and steady-state control passed.
  The integrated stateful run reported 99.392/69.763 ms wall/summed-GPU at the
  cold first step and 8.943/6.455 ms at position 2. The 20-sample four-layer
  replay and 30-sample layer-0 control also completed with exact final readback.

Decision:

- Accept positions 1→2 as the minimum stateful decoder slice. Do not call it a
  complete decoder: 39 layers, the output head, logits, and sampling remain, and
  compressed cache state has not yet been emitted.

Next:

- Capture and implement the position-3 compressor/indexer boundary for layers
  2–3, where compression ratio four emits its first compressed-cache row.

### 2026-08-15 — Prepared four-layer replay isolated steady-state execution

Objective:

- Remove fixture decoding, model-directory lookup, host allocation, and tensor
  readback from the repeated four-layer timing interval while retaining one
  exhaustive post-measurement C0 gate.

Implementation:

- Added `rust-star-layers0123-steady-state-v1` and the
  `layers0123-bench` CLI with bounded warmup/iteration controls and atomic
  JSON output.
- Introduced prepared layer executions that resolve and validate 25 mmap-backed
  model spans per layer, decode each pinned fixture, and allocate every host
  result vector once. Existing single and chained correctness probes now use
  the same preparation object.
- Added a timing-only Metal command mode. It requires a completed declared
  chain, may be queried only through layer 0, and returns the first-submit to
  tail-completion wall interval plus the sum of all four command-buffer GPU
  intervals. It copies no activation, router, expert, cache, or HC data.
- Warmup and measured chains reuse one context, cached model views, activation
  allocations, and per-layer KV storage. After the last measured sample, the
  standard collector reads all four layers once and performs the unchanged
  bitwise comparisons.

Target-Mac evidence:

- A 2-warmup/5-sample smoke test reported 6.709 ms wall median and 5.524 ms
  summed-GPU median, then matched every final layer boundary.
- The canonical 5-warmup/20-sample run reported wall samples
  `[5.375, 4.769, 5.769, 4.265, 4.380, 4.062, 4.601, 4.197, 4.240,
  4.166, 5.656, 4.371, 4.525, 4.562, 4.231, 4.255, 4.008, 4.831,
  4.181, 4.212]` ms: 4.318 ms median, 0.180 ms MAD, 4.008 ms min,
  and 5.769 ms max.
- Summed-GPU timing was 3.767 ms median, 0.131 ms MAD, 3.602 ms min,
  and 4.851 ms max. One exhaustive collection after sample 20 matched all four
  pinned DwarfStar fixtures and expected expert routes bit-for-bit.
- The complete repository gate then passed 45 Rust tests, the optimized build,
  37 Python tests, all 11 differential fixtures, the strict 1,288-tensor model
  recipe, every incremental Metal probe, both four-layer scheduler controls,
  the new 5-warmup/20-sample replay, and the 30-sample layer-0 benchmark. Its
  integrated replay measured 4.241 ms wall median (0.139 ms MAD) and 3.781 ms
  summed-GPU median (0.110 ms MAD), followed by an exact final four-layer C0
  collection. The subsequent layer-0 control measured 1.900 ms wall median
  and 1.526 ms GPU median.

Decision:

- Accept prepared four-layer replay as the current steady-state
  execution/scheduler microbenchmark. Do not compare its fixed token-201,
  position-1 replay directly with DwarfStar token throughput.

Next:

- Keep this replay in the target-Mac gate, then either cross the next distinct
  layer boundary or introduce the minimum position-advancing decoder loop.

### 2026-08-15 — Layer 3 crossed the biased-top-k boundary and stayed exact

Objective:

- Extend both scheduler controls through layer 3, the first layer after the
  model's three token-hash router layers, and remeasure command chaining over
  the larger exact slice.

Oracle fixture:

- Captured all 32 retained layer-3 position-1 boundaries twice from separate
  fresh processes using the pinned DwarfStar executable; every corresponding
  artifact was byte-identical. Captured position-0 `KVcur` independently and
  derived the FP16-rounded cache row used by the two-position attention read.
- Added `rust-star/fixtures/layer3-complete-v1/`: 28 ordered operations and
  33 tensors totaling 741,808 verified bytes. The expected expert route is
  `[1, 58, 68, 240, 20, 24]`.

Implementation:

- Added the stable `rust-star-layers0123-continuous-probe-v1` and
  `rust-star-layers0123-chained-probe-v1` reports plus their CLI commands.
  The synchronized path owns four command buffers and four per-layer KV
  allocations; the chained path commits those buffers without inter-layer
  waits and waits once at the tail.
- Made the command-chain tail explicit in the Rust/Objective-C ABI. The
  existing layers-0/1/2 path declares layer 2, and the new path declares layer
  3. Objective-C validates the declaration, submission order, finalizer, and
  deferred collection against the same tail.
- The first model attempt correctly rejected a nonexistent
  `blk.3.ffn_gate_tid2eid.weight`. DwarfStar and GGUF inspection confirmed
  `deepseek4.hash_layer_count=3`: layers 0 through 2 use token-hash tables,
  while layer 3 uses `blk.3.exp_probs_b.bias` for biased top-k selection.
  The no-copy auxiliary router view and Metal router arguments now select the
  correct mode per layer.
- Updated the standard runtime gate, fixture validation, report writer tests,
  CLI help, and ownership documentation to cover layers 0 through 3.

Target-Mac evidence:

- The synchronized and chained paths both matched every retained FP32 boundary
  for all four layers. Layer 3 selected
  `[1, 58, 68, 240, 20, 24]`, used its own cache allocation, and consumed
  layer 2's live final HC buffer without a host upload.
- Five alternating synchronized/chained pairs remained C0 exact. Synchronized
  wall samples were `[98.490, 97.229, 97.661, 98.371, 100.046]` ms; chained
  samples were `[97.402, 95.409, 95.495, 97.216, 97.566]` ms. Medians were
  98.371 and 97.216 ms: 1.0119x, or 1.17% less wall time for chaining.
- Median summed command GPU intervals were 68.128 ms synchronized and 67.986
  ms chained. The near-equal GPU work and exact outputs support the intended
  interpretation: this measures removed host synchronization inside an
  exhaustive correctness probe, not decoder throughput.
- The complete target-Mac gate passed: 43 Rust tests, optimized macOS build,
  37 Python tests, all 11 differential fixtures, strict validation of 1,288
  required tensors, every standalone Metal probe, synchronized and chained
  four-layer C0 gates, and 30 bit-identical layer-0 steady-state samples. The
  gate's four-layer wall intervals were 99.980 ms synchronized and 96.654 ms
  chained; its layer-0 steady-state median remained 1.500 ms wall.

Decision:

- Accept layer 3, the explicit variable-length chain contract, and the
  hash-to-biased-top-k router transition. Keep the four-layer synchronized path
  as the correctness control and the one-wait path as the scheduler candidate.

Next:

- Move exhaustive boundary readback outside the production timing interval and
  add repeated four-layer execution before extending to another layer.

### 2026-08-15 — Three-layer command chaining remained C0 exact

Objective:

- Remove the two inter-layer host waits from the exact layers-0/1/2 scheduler,
  retain the synchronized path as a control, and measure the result.

Implementation:

- Added `rust-star-layers012-chained-probe-v1` and the
  `layers012-chained-probe` CLI. It commits the same three command buffers to one
  Metal queue, waits once after layer 2, then collects every retained boundary.
- Layer-scoped activation keys preserve layers 0 and 1 until tail comparison;
  per-layer KV allocations remain distinct, and later layers consume the
  preceding live HC Metal buffer through queue ordering. No host upload, event,
  or fixture seam was added.
- Kept `layers012-probe` unchanged as the synchronized C0 control and added the
  chained path to `check_runtime.sh`, the stable JSON writer tests, and runtime
  ownership documentation.

Target-Mac evidence:

- The first chained run matched all three fixtures bit-for-bit with one tail
  wait. All five subsequent alternating control/chained pairs also remained C0
  exact with the expected expert routes.
- Five-pair wall samples were synchronized
  `[82.834, 82.325, 84.160, 81.209, 82.330]` ms and chained
  `[81.481, 79.615, 79.922, 80.175, 81.410]` ms. Medians were 82.330 and
  80.175 ms: 1.0269x, or 2.62% less wall time for chaining.
- Median summed command GPU time was 53.488 ms synchronized and 52.458 ms
  chained. Because this small difference includes ordinary GPU variance and the
  path still performs exhaustive correctness readback, treat the wall result as
  a scheduler diagnostic rather than decoder throughput.
- The final target-Mac gate passed: 40 Rust tests, optimized macOS build, 36
  Python tests, all ten fixture verifiers, strict validation of 1,288 required
  tensors, every standalone Metal probe, synchronized and chained three-layer
  C0 gates, and 30 bit-identical layer-0 steady-state samples. Its synchronized
  wall sum was 83.246 ms and chained wall was 79.899 ms.

Decision:

- Accept the chained lifetime/scheduling boundary and its modest measured wall
  reduction. Retain both paths: synchronized execution is the control, while
  chaining is the exact scheduler candidate. Make no token-throughput claim.

Next:

- Capture and execute the next layer/state boundary through the same ordered
  API, then remeasure chaining over the larger exact slice.

### 2026-08-15 — Layers 0→1→2 matched with explicit per-layer KV ownership

Objective:

- Prove a continuous third-layer handoff, give each executed layer persistent
  cache ownership, and cross the first compressed-attention RoPE boundary.

Oracle fixture:

- Captured all 32 retained layer-2 position-1 boundaries twice from separate
  fresh processes using the pinned DwarfStar executable; all artifacts were
  byte-identical. Captured position-0 `KVcur` independently and derived the
  exact FP16-rounded cache row used by the two-position attention read.
- Added `rust-star/fixtures/layer2-complete-v1/`: 28 ordered operations and 33
  tensors totaling 741,808 verified bytes. The expected expert route is
  `[8, 188, 195, 75, 96, 176]`.

Implementation:

- Replaced the executor's reusable cache scratch object with persistent Metal
  buffers keyed by layer identity. The layers-0/1/2 sequence therefore retains
  three distinct KV allocations while carrying one live four-stream HC state
  across the two layer seams.
- Added the stable `rust-star-layers012-continuous-probe-v1` report and
  `layers012-probe` CLI, host tests, fixture validation, documentation, and
  automatic execution in `check_runtime.sh`.
- The first layer-2 run localized its earliest mismatch to `Qcur` after exact
  attention ingress, norm, Q-Lora, Q-Lora norm, and `Qraw` boundaries. DwarfStar
  source inspection showed that layer 2 starts the compressed-attention
  schedule. Parameterizing Q and KV RoPE with base 160,000, scale 1/16,
  original context 65,536, and the matching YaRN factors restored C0 without
  weakening any comparison.

Target-Mac evidence:

- The uninterrupted sequence matched every retained boundary: layer 0 selected
  `[25, 174, 215, 58, 48, 60]`, layer 1 selected
  `[228, 208, 35, 27, 113, 12]`, and layer 2 selected
  `[8, 188, 195, 75, 96, 176]`.
- The canonical run reported 47.226/20.865 ms wall/GPU for layer 0,
  16.310/15.398 ms for layer 1, and 16.401/15.662 ms for layer 2. It used three
  synchronized command buffers and three retained per-layer cache allocations;
  these are diagnostic correctness intervals, not throughput evidence.
- The complete target-Mac gate passed: 39 Rust tests, optimized macOS build, 36
  Python tests, all ten differential fixture verifiers, the cross-language C0
  smoke, strict validation of all 1,288 required tensors, every standalone
  Metal probe, the three-layer gate, and 30 bit-identical layer-0 steady-state
  samples.

Decision:

- Accept explicit per-layer cache ownership and the compressed-RoPE third-layer
  handoff. Keep one synchronized command buffer per layer as the correctness
  control for the next scheduler experiment.

Next:

- Implement and measure a three-layer command-buffer-chained variant, requiring
  the same bitwise checkpoints and ownership report before accepting it.

### 2026-08-15 — Persistent layer-0→layer-1 HC handoff matched DwarfStar

Objective:

- Move setup/execution ownership into the Rust scheduler and prove the first
  real cross-layer state handoff without a host upload or fixture seam.

Oracle fixture:

- Captured all 32 retained layer-1 position-1 boundaries twice from the pinned
  DwarfStar executable; every artifact was byte-identical across fresh
  processes. Captured position-0 `KVcur` separately and derived the exact
  FP16-rounded cache row used by the two-position attention read.
- Added `rust-star/fixtures/layer1-complete-v1/`: 28 ordered operations and 33
  tensors totaling 741,808 verified bytes. The fixture binds the pinned source,
  executable, model, prompt, machine, token, layer, and position identities.

Implementation:

- Added a narrow Rust `LayerExecutor` with explicit create/execute/destroy
  lifetime and monotonically ordered layer calls. The Objective-C context now
  rejects model-mapping rebinding and caches exact mmap-backed Metal views plus
  reusable activation buffers.
- Parameterized the proven full-layer chain by layer index. Layer 0 executes
  thirty dispatches; layer 1 executes twenty-eight by skipping embedding and
  repeat and reading layer 0's retained final-HC buffer directly.
- Added a stable `layers01-probe` CLI/JSON contract, fixture and report tests,
  documentation, and automatic target-model gate execution.

Target-Mac evidence:

- Layer 0 retained 25/25 no-copy model-range pointer matches, selected
  `[25, 174, 215, 58, 48, 60]`, and remained C0 exact.
- Layer 1 retained 25/25 pointer matches, selected
  `[228, 208, 35, 27, 113, 12]`, and matched every retained DwarfStar boundary
  bit-for-bit while consuming the live layer-0 HC state.
- The final full gate reported 70.143 ms wall / 30.658 ms GPU for layer 0 and
  26.414 ms wall / 25.457 ms GPU for layer 1. These intervals are diagnostic
  because setup, synchronization, and readback remain included.
- The complete target-Mac gate passed: 37 Rust tests, optimized macOS build, 35
  Python tests, all nine fixture verifiers, the cross-language C0 smoke, strict
  validation of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept the scheduler-owned Metal lifetime and direct HC handoff as the first
  reusable decoder boundary. Keep one synchronized command buffer per layer
  until per-layer KV ownership and a third sequential layer are exact.

Next:

- Add explicit per-layer KV-cache storage and capture/execute layer 2 through
  the same ordered executor API.

### 2026-08-14 — Layer-0 RoPE and first KV-cache write matched DwarfStar

Objective:

- Cross the first stateful decode boundary by finalizing Q/K position encoding
  and proving an exact, isolated cache-row mutation.

Oracle fixture:

- Captured layer 0, decode position 1 in four fresh pinned DwarfStar processes:
  two with standalone diagnostic Q normalization and two with the release
  fused Q-normalization/RoPE path.
- Every repeated artifact was byte-identical. The fused and diagnostic paths
  also produced byte-identical final `Qcur` tensors.
- Added `rust-star/fixtures/layer0-rope-kv-store-v1/` with raw Q/normalized KV
  inputs, diagnostic `Qnorm`, release `Qcur`, pre-store `KVrope`, post-E4M3FN
  `KVcur`, and the documented FP16-rounded cache-row derivation.

Implementation:

- Embedded the pinned DwarfStar `metal/dsv4_rope.metal` and
  `metal/dsv4_kv.metal` sources directly in the runtime Metal library.
- Extended the connected command buffer with fused per-head Q RMSNorm/RoPE,
  layer-0 KV RoPE, and the fused E4M3FN KV finalizer/raw-cache store.
- Added a three-row cache probe initialized with a finite sentinel. The kernel
  targets physical row 1; Rust checks every target-row bit and both neighboring
  guard rows.
- Added a stable CLI/JSON schema, fixture and unit coverage, documentation, and
  automatic execution in `check_runtime.sh`.

Target-Mac evidence:

- All ten model views retained exact mmap pointer identity.
- All 32,768 `Qcur`, 512 `KVrope`, 512 post-FP8 `KVcur`, and 512 cache-row
  values matched the pinned evidence by FP32 bit pattern. Both 512-value guard
  rows remained unchanged.
- The final gate reported 40.305 ms wall and 0.949 ms Metal GPU time. This is a
  cold standalone correctness boundary, not decoder throughput.
- The complete gate passed: 24 Rust tests, optimized macOS build, 30 Python
  tests, all four fixture verifiers, cross-language C0 smoke, strict validation
  of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept the first exact layer-0 cache mutation as the stateful attention
  checkpoint. Add the layer-0 attention scan next, without compression or
  output projection yet.

Next:

- Capture/import the uncompressed layer-0 FlashAttention read over positions
  0..1 and validate the pre-inverse-RoPE attention output.

### 2026-08-14 — Layer-0 Q/K projection setup matched DwarfStar

Objective:

- Continue the real token path beyond Q-Lora through the complete raw Q and
  learned-normalized KV projection setup.

Oracle fixture:

- Captured layer 0, decode position 1 `q_lora_norm`, `KVraw`, `KVnorm`, and
  `Qraw` from the pinned DwarfStar executable in two fresh processes; every
  payload was byte-identical across captures.
- Added `rust-star/fixtures/layer0-qkv-setup-v1/`, retaining the independent
  attention-norm/Q-Lora inputs and all four new boundaries. The fixture contains
  six tensors totaling 159,744 bytes and binds the same source, executable,
  model, prompt, machine, token, layer, and position identities.

Implementation:

- Imported DwarfStar's fused Q-Lora/KV learned RMSNorm, preserving the default
  norm-unification formula and dispatch geometry.
- Extended the connected command buffer with Q8_0 KV projection, fused Q/K
  norm, and Q8_0 Q-B projection. The path now has nine dispatches and ten
  independently page-aligned mmap-backed model views.
- Added a separate stable probe schema/CLI/JSON report, fixture verification,
  unit coverage, documentation, and automatic target-model gate execution.

Target-Mac evidence:

- All ten model views retained exact mmap pointer identity.
- All 1,024 Q-Lora norm values, 512 KV raw values, 512 KV norm values, and
  32,768 raw Q values matched DwarfStar by FP32 bit pattern.
- The final gate reported 38.399 ms wall and 0.473 ms Metal GPU time. This is a
  cold standalone correctness boundary, not decoder throughput.
- The complete gate passed: 22 Rust tests, optimized macOS build, 29 Python
  tests, all three fixture verifiers, cross-language C0 smoke, strict validation
  of all 1,288 required tensors, and every Metal probe.

Decision:

- Accept raw Q plus normalized KV as the next connected layer checkpoint. Keep
  head normalization/RoPE and cache mutation separate so the first stateful
  boundary has its own exact evidence.

Next:

- Capture/import Q head RMSNorm and RoPE plus KV RoPE, then validate the first
  layer-0 cache-row write without adding the attention scan yet.

### 2026-08-14 — Connected layer-0 attention ingress matched DwarfStar

Objective:

- Replace the isolated projection input with a real token embedding and make
  kernel/layer/decode-step evidence independently verifiable.

Artifact contract and oracle fixture:

- Added `rust-star-differential-fixture-v1` with kernel, layer-segment, and
  decode-step scopes, ordered operations, uniquely named tensor boundaries,
  safe relative paths, exact FP32 shapes/encodings, finite-value checks, byte
  counts, and SHA-256 verification.
- Migrated the Q8 projection fixture to the new schema and added a standalone
  verifier that the full runtime gate applies to every committed fixture.
- Captured layer 0, decode position 1 from the pinned DwarfStar executable in
  two fresh processes. All mixer, HC split, collapsed HC, attention norm, and
  Q-Lora payloads were byte-identical across captures.
- Retained seven tensor artifacts totaling 37,056 bytes under
  `rust-star/fixtures/layer0-attention-ingress-v1/`. The fixture records token
  201 and binds the pinned source/tree, executable, model, prompt, machine, and
  graph-dump selection.

Implementation:

- Imported DwarfStar's F16 gather, HC repeat, plain RMSNorm, vectorized F16
  matvec, fused HC split/collapse/weighted norm, and Q8_0 matvec.
- Generated the Objective-C NSString representation of the auditable Metal
  source at build time, keeping the source readable while retaining a
  dependency-free runtime binary.
- Encoded all six operations into one command buffer. Rust retains ownership
  of the read-only whole-model mmap; the shim independently wraps the six
  page-aligned weight ranges with `newBufferWithBytesNoCopy`.
- Added a stable JSON report, CLI command, tests, documentation, and automatic
  execution in the target-model runtime gate.

Target-Mac evidence:

- All six model views preserved exact mmap pointer identity.
- All 24 HC mixer values, 24 split coefficients, 4,096 collapsed HC values,
  4,096 learned-norm values, and 1,024 Q-Lora values matched DwarfStar by FP32
  bit pattern.
- The final gate reported 37.782 ms wall and 0.102 ms Metal GPU time. It
  includes a standalone correctness boundary and is not a decode-throughput
  claim.
- The complete gate passed: 21 Rust tests, optimized macOS build, 28 Python
  tests, both committed fixture verifiers, cross-language C0 artifact smoke,
  strict validation of all 1,288 required GGUF tensors, and every existing and
  new Metal probe.

Decision:

- Accept the connected attention-ingress segment as the first layer-level
  execution checkpoint. Continue immediately after Q-Lora while preserving
  narrow artifacts and avoiding a premature graph framework.

Next:

- Capture and import Q-Lora normalization/Q-B plus the paired KV setup needed
  for the layer-0 attention core.

### 2026-08-14 — First quantized decode projection matched DwarfStar

Objective:

- Validate quantized model bytes, a real runtime activation, DwarfStar's
  decode reduction order, and Rust/Metal ownership in one isolated operation.

Oracle fixture:

- Ran pinned DwarfStar source `b030961` on the M1 Ultra with one prefill token
  and one greedy decode token, filtering its existing graph dump hooks to layer
  0, position 1, `attn_norm` and `q_lora`.
- Retained the 4,096-value input and 1,024-value output under
  `rust-star/fixtures/q8-attn-q-a-v1/`. The manifest binds source tree, model
  SHA-256, machine, prompt, layer/step, tensor, dispatch geometry, and artifact
  SHA-256 values.
- Repeated the complete one-token DwarfStar capture in a fresh process; both
  activation and output files were byte-identical to the retained fixture.

Implementation:

- Imported DwarfStar's `kernel_mul_mv_q8_0_f32` arithmetic and its default four
  simdgroup/two-row dispatch behind the narrow Objective-C ABI.
- Reused the Rust-owned read-only `MAP_SHARED` model mapping and wrapped only
  the page-aligned `blk.0.attn_q_a.weight` span with
  `newBufferWithBytesNoCopy`; the runtime activation uses one small shared
  buffer.
- Added strict fixture/tensor/dispatch validation, bitwise output comparison,
  stable privacy-limited JSON, CLI support, fixture hash tests, documentation,
  and automatic target-model execution in `check_runtime.sh`.

Target-Mac evidence:

- The Q8_0 tensor begins at byte 79,060,632,640 and contains 4,456,448 bytes.
  Metal wrapped a 4,472,832-byte page-aligned view with a 1,088-byte inner
  offset and exact pointer identity.
- All 1,024 output FP32 bit patterns matched DwarfStar. Input/output checksums
  are `6001855774483604828` and `13770952831385691371`.
- The final standalone dispatch reported 3.990 ms wall and 0.031 ms GPU time;
  pipeline/boundary setup makes this correctness evidence, not decode
  throughput.
- The complete gate passed: 20 Rust tests, optimized macOS build, both no-copy
  model kernels, Metal dispatch/shared-buffer validation, 26 Python tests,
  cross-language C0 artifact smoke, and strict validation of all 1,288 required
  GGUF tensors.

Decision:

- Accept the no-copy Q8_0 matvec as the first runtime-activation arithmetic
  boundary. Build the next increment toward a minimal layer-0 chain and extend
  the artifact format at each new boundary.

Next:

- Specify the kernel/layer artifact envelope around this fixture, then derive
  layer-0 `attn_norm` from the token embedding and feed it into the validated
  projection.

### 2026-08-14 — No-copy model span and first DwarfStar kernel validated

Objective:

- Cross the first real-model execution boundary without adding a graph,
  allocator framework, or whole-model scheduler.

Changes:

- Added a dependency-free Rust `MappedModel` that parses the GGUF and owns one
  read-only `MAP_SHARED` mapping for its complete lifetime.
- Added a page-aligned Objective-C wrapper using
  `newBufferWithBytesNoCopy`; it requires the returned Metal contents pointer
  to equal the mmap pointer and releases the buffer before Rust can unmap.
- Imported DwarfStar's `kernel_get_rows_f16` argument layout, kernel body, and
  threadgroup geometry. Its pipeline is compiled lazily so the original
  dispatch probe's compile-time metric remains comparable.
- Added exact Rust F16-to-F32 reference conversion, bitwise comparison, a stable
  `rust-star-f16-embedding-probe-v1` JSON artifact, CLI support, documentation,
  and automatic target-model execution in `check_runtime.sh`.

Target-Mac evidence:

- Strict validation selected `token_embd.weight` at byte 77,928,033,088 with
  1,059,061,760 payload bytes. Metal wrapped a 1,059,078,144-byte page-aligned
  view with an 11,072-byte inner offset and exact pointer identity.
- Five rows spanning token IDs 0 through 129,279 produced 20,480 FP32 values;
  every bit matched the CPU reference. The retained checksum is
  `4028716204944876325`.
- The final validation dispatch reported 17.632 ms wall and 0.020 ms GPU time.
  This includes a cold standalone boundary and is correctness evidence, not a
  throughput claim.
- The complete runtime gate passed: 18 Rust tests, optimized macOS build, Metal
  dispatch/shared-buffer validation, 25 Python tests, cross-language C0 smoke,
  strict real-GGUF inspection, and the new no-copy gather.

Decision:

- Accept this mmap/Metal lifetime split as the first model ownership boundary.
  Keep the next increment to one decode projection/matvec before designing the
  graph scheduler.

Next:

- Capture or derive an exact DwarfStar fixture for one Q8_0 decode projection,
  then run the same weight span and activation through Rust Star.

### 2026-08-14 — Target-Mac bootstrap and oracle-v1 completed

Objective:

- Validate the Rust/Metal boundary and pin the DwarfStar oracle on the target
  M1 Ultra before implementing model execution.

Actions and evidence:

- Created a dedicated `agent/rust-star-bootstrap` worktree so the DSpark
  observability branch can continue in a separate Codex task.
- Installed the missing standard `rustfmt` component and ran the full M-002
  gate on the M1 Ultra.
- Fixed the target recipe for `indexer.attn_q_b.weight` from Q8_0 to F16 and
  added a regression test covering every ratio-4 indexer layer.
- Added `pipefail` to the documented evidence command after the first strict
  inspection failure was masked by `tee`.
- Passed 15 Rust tests, 25 Python tests, the optimized macOS build, the C0
  cross-language artifact check, shared-buffer validation, and strict real-GGUF
  inspection of all 1,288 required tensors.
- Captured and independently verified the complete 2K/32K `oracle-v1` bundle.
  Correctness, full-vocabulary frontier logits, and all six timed measurements
  passed. The archive SHA-256 is `5115f445...a09ee3c`.

Decision:

- Accept the quick capture as `oracle-v1`. Keep the extended context capture
  ready but move the critical implementation path to no-copy model mapping and
  the first isolated kernel.

Next:

- Implement and test the no-copy, page-aligned read-only GGUF span boundary,
  then import one decode-critical DwarfStar kernel behind C0 fixtures.

### 2026-08-09 — Initial Rust/Metal ownership and dispatch boundary added

Objective:

- Turn the host-only scaffold into the smallest testable Metal runtime boundary
  without claiming model inference or performance before the M1 Ultra run.

Changes:

- Added a macOS-only Objective-C ARC shim, compiled by Cargo through the Xcode
  toolchain, that owns the Metal device, queue, runtime-compiled probe pipeline,
  shared buffer, command encoders, synchronization, and GPU timestamps.
- Added a safe Rust lifecycle/configuration/reporting layer and `metal-probe`
  command. It compares synchronized per-dispatch command buffers with one
  batched command buffer, validates every resulting element against a CPU
  reference, and can atomically emit a privacy-limited stable JSON artifact.
- Kept non-macOS host contracts buildable through an explicit unsupported stub,
  added a macOS CI compilation job, and made the target-Mac check script run the
  probe before model inspection.
- Applied rustfmt to the complete pre-existing Rust crate when a temporary
  formatter became available; the GGUF/target diffs are formatting-only.
- Documented the ownership boundary in `rust-star/runtime/METAL.md` and extended
  manual task M-002 to request the resulting probe artifact.

Validation in this workspace:

- `check_runtime.sh` passed without a model using a temporary Rust 1.85
  toolchain: rustfmt, 15 Rust unit tests, optimized host build, all 25 Python
  contract tests, and the Rust-writer/Python-reader C0 smoke check succeeded.
- Python compilation, workflow YAML parsing, shell syntax, an independent Rust
  grammar parse, and `git diff --check` passed. The non-macOS `metal-probe`
  command failed explicitly as designed.
- One initial optimized link in the generated in-repository target directory
  reported unresolved thin-LTO symbols; an immediate rebuild and the complete
  check script then passed. Preserve macOS CI and clean-target evidence rather
  than treating this temporary-toolchain anomaly as target evidence.
- This environment still has no Xcode/Metal device or model. The Objective-C
  shim has not been compiled, and no Metal execution, model correctness run, or
  performance measurement has been performed here.

Next:

- Compile and run the complete check script on the M1 Ultra. If the boundary
  validates, add a no-copy read-only model span and one isolated DwarfStar
  kernel while preserving C0 differential hooks.

### 2026-08-09 — Checkpointed paired runner and Rust-side contract added

Objective:

- Complete the engine-neutral A/B orchestration before a Rust Star decoder is
  available, so target-Mac measurements cannot drift into an ad hoc procedure.

Changes:

- Added `rust-star/ENGINE_MEASUREMENT_FORMAT.md` as the shared fresh-process
  contract. DwarfStar implements it now; the Rust Star benchmark must emit the
  same metric semantics directly or through a thin adapter.
- Added `rust-star/paired_runner_lib.py` and `run_paired_benchmark.py` with an
  immutable SHA-bound plan, atomic `state.json` checkpointing after every engine
  process, untimed per-engine warm-ups, alternating A/B and context ordering,
  and optional one-pair-at-a-time execution.
- Timed failures block without automatic retries. Retrying requires an explicit
  reason, reruns the complete pair, retains the old attempt/evidence, and
  preserves already-finalized outputs under `superseded/`.
- Adapter invocations use argument arrays without a shell and fresh process
  groups. Timeouts and interrupts terminate the process group before control
  returns; a hard-interruption checkpoint remains blocked for deliberate review.
- Finalization occurs only after every predeclared pair has one final valid
  attempt. It re-hashes and revalidates every successful engine measurement,
  emits the strict raw artifact, revalidates that artifact, and writes the
  paired summary.
- Added `rust-star/PAIRED_RUNNER.md` and manual task M-005 for the eventual first
  target-Mac paired run. Local plans may contain private paths and are never
  copied into the shared result directory.

Validation:

- All 25 Python contract tests passed. Synthetic end-to-end tests cover pause
  and resume, engine/context ordering, blocked failures, explicit full-pair
  retry, plan immutability, measurement-tamper rejection, validated final
  export, and preservation of superseded results.
- Python compilation, both new CLI help paths, workflow/shell static checks,
  and `git diff --check` passed.
- No real engine, model, Metal, correctness, or performance run was possible in
  this environment.

Next:

- Implement the Rust Star measurement producer alongside the first runnable
  decoder, then instantiate a private paired plan from the completed oracle
  manifest rather than inventing identities in advance.

### 2026-08-09 — Isolated DwarfStar measurement boundary added

Objective:

- Make the pinned oracle measurable through the future paired-runner boundary
  without modifying DwarfStar or waiting for a Rust Star decoder.

Changes:

- Added `rust-star/measure_dwarfstar.py` and
  `dwarfstar_measurement_lib.py`. Each invocation runs one fresh `ds4-bench`
  process at one frontier, preserves its CSV and sanitized logs, and emits a
  checksummed `rust-star-engine-measurement-v1` record.
- Added external complete-process wall time and per-child peak RSS collection.
  Prefill, full-generation, and steady-generation durations are reconstructed
  from DwarfStar's reported counts/rates; the residual is explicitly named
  process overhead rather than inaccurately calling it model-load time.
- Documented a DwarfStar semantic trap: at a terminal single frontier,
  `kvcache_bytes=0` means no serialized session snapshot was created. The
  normalized contract now records `null`, not zero live KV usage.
- Hardened the adapter against missing/partial CSV output, partial generation,
  non-finite metrics, non-empty result directories, timeouts, and private path
  leakage in captured logs.
- Added `rust-star/MEASUREMENT_ADAPTER.md` and included the adapter in the host
  workflow path filters.

Validation:

- Python compilation, CLI help, shell syntax, workflow YAML parsing, and
  `git diff --check` passed.
- All 20 Python contract tests passed. Synthetic child-process tests cover CSV
  normalization, per-process measurements, path redaction, and persisted
  adapter-validation failures.
- No real DwarfStar/model/Metal measurement was run in this environment.

Next:

- Define the Rust Star adapter against the same single-process schema, then add
  a checkpointed paired orchestrator that enforces A/B and context order.

### 2026-08-09 — Manual ledger and paired benchmark v1 defined

Objective:

- Preserve every user/hardware/account-dependent handoff in one actionable
  ledger and remove benchmark-method ambiguity before measurements begin.

Changes:

- Added `RUST_STAR_MANUAL_TASKS.md` with status rules, security boundaries,
  exact commands, dependencies, evidence requirements, and success conditions
  for GitHub Actions approval, target-Mac runtime/model inspection, quick and
  extended oracle capture, and the deferred secure remote-access decision.
- Added `workflow_dispatch` to the host-contract workflow so it can be run from
  GitHub after fork Actions are enabled or approved.
- Added `rust-star/BENCHMARK_PROTOCOL.md` as
  `rust-star-paired-benchmark-v1`. It fixes C0 eligibility, batch-one closed-loop
  decode semantics, 256K primary context, development/full context sets,
  committed-token metrics, alternating paired order, repetition counts,
  thermal controls, raw-data retention, pairwise speedup aggregation, and
  capacity/failure treatment.
- Added `rust-star/PAIRED_RESULT_FORMAT.md`, a strict standard-library parser,
  `summarize_paired_benchmark.py`, and synthetic tests for machine-readable raw
  pairs, host/correctness manifests, exact build/runtime identities,
  predeclared coverage/order, invalid-pair retention, operational metrics, C0
  headline eligibility, and within-pair speedup aggregation.
- Updated the durable project open item to distinguish defining the protocol
  from executing it and collecting the first DwarfStar numbers.

Validation:

- Python compilation passed for the paired parser, CLI, and tests.
- All 16 Python contract tests passed, including cross-context schedule and
  invalid-retry cases.
- `git diff --check` and workflow/shell static checks passed.
- No model, Metal, correctness, or performance run was possible in this
  environment.

Next:

- Define the narrow benchmark-runner interface that will produce paired raw
  rows from DwarfStar and the future Rust executable without coupling either
  engine to the aggregator.

### 2026-08-09 — Strict Rust host-runtime scaffold added

Objective:

- Establish a narrow, executable Rust boundary that can be prepared without
  Metal or model access and prevents unsupported models from entering the
  future engine accidentally.

Changes:

- Added a dependency-free Rust 2021 crate at `rust-star/runtime/` with a
  committed lockfile and a minimum Rust version of 1.74.
- Added a bounded GGUF v3 parser that reads metadata/tensor directories without
  reading tensor payloads. It rejects duplicate names, unknown types, malformed
  booleans/UTF-8, excessive counts/strings, rank/size arithmetic overflow,
  invalid alignment, payload overlap, and out-of-file ranges.
- Derived and encoded the exact DwarfStar Flash shape: 43 layers, width 4096,
  vocabulary 129280, 64 attention heads, 256 experts with 6 active, and the
  model's compression schedule and semantic metadata.
- Added the initial resident imatrix-Q2 recipe validator: IQ2_XXS routed gate/up,
  Q2_K routed down, Q8_0 attention/shared/output, and F16 HC/compressor/indexer.
- Kept checkpoint identity honest: GGUF name metadata is informational; the
  completed oracle's whole-model SHA-256 remains necessary to prove `0731`.
- Added a full-FP32-logit writer with finite-value checks, nine-significant-digit
  output, signed-zero preservation, and lowest-token-ID argmax tie semantics.
- Added `rust-star/check_runtime.sh` to run formatting, Rust unit tests, release
  build, existing Python artifact tests, a Rust-writer/Python-reader C0 smoke
  check, and optional strict model inspection on the target Mac.
- Added a path-filtered GitHub Actions job pinned to Rust 1.74 so formatting,
  compilation, unit tests, release build, and the cross-language artifact smoke
  check can be validated before target-Mac access is available.

Validation in this workspace:

- All Rust files passed an independent Rust grammar parse.
- Shell syntax and `git diff --check` passed.
- All nine existing Python artifact/bundle tests passed.
- The check script correctly exits with a clear Rust-toolchain prerequisite in
  this container.

Not validated here:

- This environment has neither `rustc` nor Cargo, so type-checking, rustfmt, and
  the new Rust unit tests are pending the target-Mac run.
- No real GGUF was available. The first strict inspection may expose a justified
  difference between the actual oracle recipe and the currently derived
  DwarfStar template policy; any change must be evidence-driven and journaled.
- No Metal code, inference, correctness run, or performance measurement exists
  in the scaffold yet.

Next:

- Run `./rust-star/check_runtime.sh /absolute/path/to/model.gguf` on the M1
  Ultra, then run the quick `oracle-v1` capture.

### 2026-08-09 — Offline oracle validation and C0 comparator added

Objective:

- Progress the correctness infrastructure without access to the M1 Ultra or
  model, and give the future Rust runtime an exact output contract.

Changes:

- Added `rust-star/ARTIFACT_FORMAT.md` as the stable initial bundle and
  post-prefill full-logit contract.
- Added `rust-star/artifact_lib.py` with safe bundle extraction, artifact hash
  verification, strict finite-FP32 parsing, bit-pattern comparison, and drift
  metrics.
- Added `rust-star/verify_oracle_bundle.py` to validate a result directory or
  `.tar.gz`, including its sibling/explicit archive SHA-256.
- Added `rust-star/compare_logits.py`; it exits zero only for C0 by default and
  reports mismatch/ULP/error, cosine, KL, argmax, and top-k diagnostics for
  valid non-C0 artifacts.
- Added `rust-star/tests/test_artifact_tools.py` with synthetic fixtures for
  exact equality, signed zero, drift, metadata, tampering, partial manifests,
  and archive traversal.
- Updated the capture README and made tar extraction behavior explicit across
  supported Python versions.

Validation:

- Python compilation succeeded for the capture, library, both CLIs, and tests.
- Nine unit tests passed with deprecation warnings promoted to errors.
- CLI integration checks confirmed C0 exit 0, drift exit 1, and bundle/archive
  verification with the sibling checksum.
- `git diff --check` passed.
- No Metal, model, or performance run was attempted in this environment.

Next:

- Begin a strict, platform-independent Rust runtime scaffold while target-Mac
  capture remains pending.

### 2026-08-09 — Target-Mac oracle capture kit prepared

Objective:

- Prepare everything that can be cloned and executed on the M1 Ultra without
  granting this environment remote access to the machine.

Changes:

- Added `rust-star/capture_oracle_v1.py`, a one-command Python standard-library
  capture tool.
- Added `rust-star/README.md` with quick, extended, custom-context, reporting,
  and security instructions.
- Git-ignored local `rust-star/results/` and `rust-star/.work/` outputs.
- The tool hashes the entire GGUF; captures allowlisted hardware, macOS,
  compiler, Metal, power, and load metadata; exports pinned upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406`; and builds it in a temporary
  directory.
- It runs `--metal-kernels` and `--logprob-vectors`, captures full post-prefill
  FP32 frontier logits separately from timed runs, and aggregates repeated
  batch-one decode/prefill measurements with median, MAD, minimum, and maximum.
- The default quick set is 2K/32K with three repetitions. The opt-in full set is
  2K, 32K, 128K, 256K, 512K, and 1M.
- Result manifests are updated after each completed gate/run so partial evidence
  survives interruption or a long-context capacity failure.
- The shareable archive excludes the model, full environment, unrelated process
  listings, credentials, absolute model path, and Mac serial/UUID identifiers.

Validation in this Linux workspace:

- `python3 -m py_compile rust-star/capture_oracle_v1.py`
- CLI help and default/full dry-run plans.
- Invalid duplicate-context rejection.
- Median/MAD aggregation helper checks against synthetic rows.
- Pinned source commit/tree validation, offline `git archive` extraction,
  deterministic 1M prompt expansion, and command-path sanitization.
- `git diff --check` and ignore-rule verification.

Not validated here:

- No Metal build, GGUF load, correctness gate, logit dump, or performance run
  was possible because this workspace is not the target Mac and has no model.

Next:

- Run the documented default capture on the M1 Ultra and return its archive and
  SHA-256 for inspection before attempting the extended context set.

### 2026-08-09 — Fork synchronized and research branch published

Objective:

- Publish the bootstrap work after correcting the GitHub App installation
  scope.

Actions and evidence:

- Confirmed a new personal-account installation for `deathcoder` was visible
  alongside the existing `dion-labs` installation.
- Fast-forwarded `deathcoder/ds4:main` to upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406` without force.
- Created remote branch `agent/rust-star-bootstrap` from that commit.
- Recreated the three-file local documentation tree atomically through GitHub's
  blob, tree, commit, and ref APIs. The initial published commit was
  `6daf28c3faee652463fe0957397e023e4ae7fa98`; its tree
  `e9761869fc0a45fc3880ff42d133deb9c0da2371` exactly matched the local tree.
- No pull request was opened because branch publication, not upstream review,
  was requested.

Next:

- Capture the oracle manifest and baseline on the target M1 Ultra, then define
  the differential golden-artifact format.

### 2026-08-09 — Publication 403 root cause identified

Objective:

- Determine why other GitHub-connected sessions can publish while writes to
  this fork consistently fail.

Evidence:

- The authenticated GitHub user is `deathcoder`.
- The app installation/account inventory contains only organization
  `dion-labs` (installation `152207005`).
- There is no GitHub App installation for the personal `deathcoder` account.
- `deathcoder/ds4` repository metadata reports the user's collaborator
  permission (`push: true`), but Git database writes execute through an app
  installation token. No installation token is scoped to this personal fork.

Conclusion:

- The 403 is not caused by the branch name, upstream commit, or use of the
  low-level ref endpoint. The app is installed on the wrong owning account for
  this repository. Successful sessions likely target repositories covered by
  the `dion-labs` installation or use separately authenticated local git.

Next:

- Install the GitHub app on the `deathcoder` personal account and grant it
  access to `deathcoder/ds4`, then retry ref creation/update through the app.

### 2026-08-09 — GitHub app publication retry

Objective:

- Retry fork synchronization and branch publication through the connected
  GitHub app, and preserve that app as the preferred publication mechanism.

Evidence:

- The app successfully read `deathcoder/ds4` and reported `admin: true` and
  `push: true` repository permissions.
- It successfully resolved upstream commit
  `b0309611041655f4e45671cfd9c9886aff161406` through the fork.
- Updating `refs/heads/main` to that commit failed with GitHub HTTP 403:
  `Resource not accessible by integration`.
- Creating `agent/rust-star-bootstrap` from that commit failed with the same
  GitHub HTTP 403 from the create-reference endpoint.

Decision:

- Future sessions should try the GitHub app first, but must confirm an actual
  ref write succeeds. Reported repository permission metadata alone is not
  sufficient evidence that the installation token can publish.

Next:

- Refresh or reconnect the GitHub app installation with repository Contents
  write access, then retry the two ref operations before recreating the local
  documentation commit on the remote branch.

### 2026-08-09 — Bootstrap branch and durable project memory

Objective:

- Synchronize the fork work with upstream, create an isolated research branch,
  and preserve the brainstorming decisions across handoffs and compaction.

Repository actions:

- Cloned `deathcoder/ds4` as `origin`.
- Added `antirez/ds4` as `upstream`.
- Verified the fork's `main` was 137 commits behind and 0 ahead of upstream.
- Fast-forwarded local `main` from `80ebbc396aee40eedc1d829222f3362d10fa4c6c`
  to upstream `b0309611041655f4e45671cfd9c9886aff161406`.
- Created local branch `agent/rust-star-bootstrap` at that upstream commit.
- Added `RUST_STAR_PROJECT.md`, this journal, and an `AGENT.md` pointer that
  requires future Rust Star sessions to read and update them.

Decisions recorded:

- Target only the 48-GPU-core, 128 GB M1 Ultra Mac Studio and
  DeepSeek-V4-Flash-0731 initially.
- Start from the resident imatrix Q2 model and a minimal Rust decoder/benchmark.
- Require C0 full-FP32-logit bit equivalence in the default path.
- Version the upstream DwarfStar oracle rather than treating a moving branch as
  immutable truth.
- Make batch-one committed decode t/s primary; retain prefill, TTFT, memory,
  long-context, low-concurrency, and agent-loop metrics as research scope.
- Deprioritize speculative decoding while retaining the existing DSpark branch
  as evidence and a possible later source of proven mechanisms.
- Use 256K as an important operating point and test correct scaling toward 1M
  rather than creating an undefined separate capacity mode.

Validation:

- No build, model correctness test, or performance benchmark was run. This
  change contains project documentation only.

Blockers/notes:

- The connected GitHub integration could read the repositories but returned
  HTTP 403 for branch/ref writes. Local git work proceeded instead.
- A non-interactive `git push origin main:main` also failed because this
  workspace has no HTTPS GitHub username/token configured. Remote fork
  synchronization and branch publication remain pending write credentials.
- SSH was not a viable fallback: no authentication agent was available and the
  workspace's restricted network could not resolve GitHub on the SSH path.

Next:

- Publish the already committed local state from a GitHub-authenticated
  environment, then capture the oracle manifest and baseline on the target Mac.
