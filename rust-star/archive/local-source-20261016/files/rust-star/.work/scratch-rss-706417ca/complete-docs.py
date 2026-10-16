from pathlib import Path
r=Path(__file__).resolve().parents[3]
p=r/'RUST_STAR_JOURNAL.md';s=p.read_text();marker='## 2026-09-12 — Start bounded scratch RSS diagnosis'
entry='''## 2026-09-12 — Close scratch RSS diagnosis with a protocol gap

- All 25 frozen fresh-process cells complete without repeats or exclusions:
  100 tiles, 249,317,323,300 GPU-verified payload bytes, 550 residency snapshots
  and 100 identity sets. One 105-buffer logical set remains 2,493,173,233 bytes;
  pooled identities stay stable and all final weak samples are zero. Fresh
  modes have no previous-generation buffers at tile entry.
- CPU clearing adds a paired median 2,493,284,352 bytes of synthetic peak RSS.
  Omitting only four full-copy destination clears saves 536,805,376 bytes
  (511.94 MiB), positive in all five blocks and above every frozen measurement
  threshold. Median footprint saving is only 147,768 bytes and changes sign
  in one block. This is RSS residency evidence, not reduced allocation or a
  validated whole-engine prediction. Prior pool candidate remains rejected.
- Source proof covers q_cur_buffer, attention_back_buffer, next_q_cur_buffer
  and next_attention_back_buffer: each receives a complete 134,217,728-byte
  copy before content reads. Preserve source/other101 initialization, distinct
  identities, exact lengths and prior checked completion. Guarded copies pass
  24 cases, 2,147,549,248 payload bytes, 24,576 guard bytes and all12 deliberately
  shortened-copy controls. One pre-GPU host compile error and correction remain
  in the report; no measurement was repeated.
- PROTOCOL GAP: final source review found that the planned mutation after
  copying is missing from the guarded GPU proof. Re-poisoning before the next
  copy does not meet that requirement. The pre-run check and frozen auditor
  did not enforce it. Preserve their original optimistic output; decision.json
  supersedes selection with prospective_protocol_pass=false and
  select_for_integration=false. No threshold/source/result edit or GPU rerun
  follows discovery. The diagnostic closes with useful measurements and an
  incomplete prospective proof, not an accepted optimization.
- Portable report: rust-star/reports/scratch-rss-706417ca/README.md. Verification
  passes on its first run: 110 artifacts, source/measurement re-audits, protocol
  gap verdict and rejection of corrupted clear-byte and pooled-identity records.
  All 42 runtime sources/default executable stay unchanged; no model runs,
  publication, remote dispatch or256K. All GPU processes are complete.
- Next milestone, NOT STARTED: close the guarded copy-then-in-place-mutation
  test gap first; if passing, evaluate only the four destination clear omissions
  in the bounded pool with fresh full C0/ownership evidence and unchanged
  engine RSS <=1.02, footprint <=1.15 and all prefill/order/decode gates.
  Finish local commit and user update, then STOP.

'''
assert marker in s;s=s.replace(marker,entry+marker,1)
old='''- COMPLETED SINGLE MILESTONE: explicit scratch reuse rejected.
  Full C0/pool proofs and all 12 engine runs complete; all accepted source/default restored.
  Report: rust-star/reports/pool-1k-fcdfef5f/README.md. 221 artifacts verify.
  Report this milestone and the next proposal, then STOP.
'''
new='''- CLOSED SINGLE MILESTONE: scratch RSS diagnostic, with a protocol gap.
  All25 measurement cells pass; selected clears save a paired median511.94 MiB
  synthetic RSS. The planned after-copy mutation proof is missing, so full
  protocol/selection is false. Report and authoritative decision:
  rust-star/reports/scratch-rss-706417ca/README.md (110 artifacts verify).
  Runtime/default unchanged; no model runs. Close that proof gap before any
  next candidate; report this milestone and the next proposal, then STOP.
  Explicit scratch reuse remains rejected on RSS; full C0/pool proofs and all12
  engine runs are preserved in rust-star/reports/pool-1k-fcdfef5f/README.md.
'''
assert old in s;s=s.replace(old,new,1)
a=s.index('## Immediate Next Actions\n');b=s.index('\n## Entries',a)
s=s[:a]+'''## Immediate Next Actions

1. The scratch RSS diagnostic is closed with a protocol gap. Preserve all110
   verified artifacts and the original optimistic audit; decision.json is the
   authoritative non-selection verdict. Runtime/default remain unchanged.
2. Report synthetic RSS savings, source/copy coverage and the missing planned
   after-copy mutation test. No full-engine acceptance or footprint saving claim.
3. Next proposal, not started: close the guarded copy-then-in-place-mutation
   test gap, then conditionally evaluate only four destination clear omissions
   with fresh full C0/pool evidence and unchanged engine/performance/memory gates.
4. STOP after the update. Keep oracle-v5 immutable,256K paused and publication
   deferred; do not begin the next milestone automatically.
''' +s[b:];p.write_text(s)
p=r/'RUST_STAR_PROJECT.md';s=p.read_text();old='''Diagnose scratch RSS residency and prove which large buffers can safely avoid unconditional clearing before another 32K candidate. This is not a DwarfStar claim or fresh larger-frontier acceptance.
The current milestone is complete; stop for the user update.'''
new='''The subsequent scratch RSS diagnostic completes25 fresh processes/100 tiles
and249,317,323,300 GPU-verified bytes. CPU clearing costs a paired median
2,493,284,352 bytes RSS; omitting four full-copy destination clears saves
536,805,376 bytes (511.94 MiB), with negligible median footprint difference.
Source and guarded full-copy proofs pass, but final review finds a missing
planned mutation after the copy. The original auditor failed to enforce that
precondition: preserve its output; decision.json marks the full prospective
protocol incomplete and selects no integration. Runtime/default unchanged.
Verified110-artifact report: `rust-star/reports/scratch-rss-706417ca/README.md`.
Next close the guarded in-place mutation gap first; if it passes, evaluate only
those four destination clears with fresh full C0/pool evidence and unchanged
engine gates. Synthetic RSS savings do not predict whole-engine acceptance,
DwarfStar parity or larger-frontier correctness. The diagnostic milestone is
closed with the gap disclosed; stop for the user update.'''
assert old in s;p.write_text(s.replace(old,new,1))
p=r/'rust-star/INITIAL_SCOPE_STATUS.md';s=p.read_text();old='Diagnose scratch RSS residency and prove which large buffers can safely avoid unconditional clearing before another 32K candidate. Not started; stop for the milestone update.'
assert old in s;s=s.replace(old,'Remains rejected; bounded RSS diagnostic is closed below.',1)
marker='| Larger-frontier correctness on current source |'
row='''| Scratch RSS diagnostic | All25 cells/100 tiles pass249,317,323,300 synthetic byte checks; four complete-copy destination clears account for median511.94 MiB RSS, with negligible footprint difference. Source/copy proofs pass, but the planned after-copy mutation is absent: full protocol and integration selection are false. Verified110-artifact report; runtime unchanged. | First close the guarded mutation gap, then conditionally test only those four clear omissions with fresh full C0/pool and unchanged engine gates. Not started; stop for update. |
'''
assert marker in s;s=s.replace(marker,row+marker,1)
a=s.index('D authorized one explicit scratch-reuse milestone.')
s=s[:a]+'''D authorized one bounded scratch RSS diagnostic milestone. It is closed with
useful measurements and a disclosed missing after-copy mutation proof; no
candidate is selected for integration. All accepted source/default are unchanged.
Report the verified evidence and the next milestone, then stop. No follow-up
implementation or inference has started. Publication and256K remain outside
this milestone.
''';p.write_text(s)
