from pathlib import Path
import json
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());d=read(w/'decision.json');v=read(w/'report-verification.json');assert v['passed'];a=read(w/'engine/audit.json');state=read(w/'final-runtime-state.json');retained=d['retained'];ratio=lambda k:(a['summary'][k]['paired_ratio']['median']-1)*100
outcome='RETAINED' if retained else 'REJECTED';runtime='candidate source and production default installed' if retained else 'all accepted source/default restored';stats=f"paired median prefill {ratio('prefill_ms'):+.3f}%, steady decode {ratio('gen_steady_tps'):+.3f}%, RSS {ratio('peak_rss_bytes'):+.3f}%, peak memory footprint {ratio('peak_memory_footprint_bytes'):+.3f}%";nexttext='Refresh the direct 32K comparison against pinned oracle-v5 DwarfStar under a new prospective paired protocol.' if retained else 'Diagnose scratch RSS residency and prove which large buffers can safely avoid unconditional clearing before another 32K candidate.'
report='rust-star/reports/pool-1k-fcdfef5f/README.md';entry=f'''## 2026-09-12 — Complete explicit scratch reuse: {outcome.lower()}

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
  still requiring status4 for every reuse. No runtime/observer, capacity or
  performance-gate changes and no repeated inference. Three preparation-script
  assertion failures and corrections before builds are preserved too.
- All 12 fixed engine runs finish with exact transcripts: two excluded warmups
  and five AB/BA/AB/BA/AB pairs. Result: {stats}.
  Decision: {outcome}; {runtime}. Failed gates: {', '.join(d['failed_gates']) or 'none'}.
  This compares Rust1.74 with Rust1.74, not DwarfStar or statistical significance.
- Full input-after model/oracle/prompt/7,669-fixture hashes and all 42 final
  source hashes pass. Prior default and frozen binaries are preserved; reused
  build target contains the observed executable and is not the default.
- Portable report: {report}. Verification passes {v['artifacts']} artifacts,
  source/observer inverses, C0/profile/ring/pool/engine re-audits, the original
  sentinel failure and rejection of a corrupted completion status. GPU/runner/
  hash sessions are complete. No publication, remote dispatch or 256K work.
- Next milestone, NOT STARTED: {nexttext}
  Finish the local commit and user update, then STOP.

'''
p=r/'RUST_STAR_JOURNAL.md';s=p.read_text();i=s.index('## 2026-09-12');s=s[:i]+entry+s[i:];i=s.index('- ACTIVE SINGLE MILESTONE:');end=s.index('  Wide bootstrap c7adfb22',i);s=s[:i]+f'''- COMPLETED SINGLE MILESTONE: explicit scratch reuse {outcome.lower()}.
  Full C0/pool proofs and all 12 engine runs complete; {runtime}.
  Report: {report}. {v['artifacts']} artifacts verify.
  Report this milestone and the next proposal, then STOP.
'''+s[end:];i=s.index('- ACTIVE: explicit bounded scratch reuse');end=s.index('\n\n',i);s=s[:i]+f'''- COMPLETE: explicit scratch reuse. {stats}.
  {outcome}; {runtime}. Full C0/pool/engine evidence and
  portable report are verified. Next: {nexttext}
  No next-milestone implementation/inference started; keep256K paused.
'''+s[end:];i=s.index('## Immediate Next Actions');end=s.index('\n## Entries',i);s=s[:i]+f'''## Immediate Next Actions

1. The explicit scratch-reuse milestone is complete. Preserve the verified
   report, original sentinel assertion and all raw evidence. {runtime.capitalize()}.
2. Report {outcome.lower()}, full C0/pool coverage and engine/memory results.
3. Next proposal, not started: {nexttext}
4. STOP after the update. Keep oracle-v5 immutable,256K paused and publication
   deferred; do not begin the next milestone automatically.
'''+s[end:];p.write_text(s)
p=r/'RUST_STAR_PROJECT.md';s=p.read_text();i=s.index('Current implementation boundaries, evidence, and immediate actions');text=f'''The subsequent explicit scratch-reuse candidate is {outcome.lower()}. It owns one
105-buffer set per four-tile native bootstrap chunk, restores full zero/copy
initial state after checked completion, and releases the owner at final/error
exit. Persistent state and shader arithmetic remain unchanged. Three fresh
full-C0 runs pass 50,031,360 exact FP32 words; the complete pool audit passes
28 tiles/2,940 checkouts with a 2,493,173,233-byte logical peak. A documented
unsigned nil-command sentinel correction preserves the original audit failure
without changing runtime, ownership criteria or performance thresholds. All 12
fixed engine runs complete: {stats}. {runtime.capitalize()}.
{v['artifacts']}-artifact verified report: `{report}`.
{nexttext} This is not a DwarfStar claim or fresh larger-frontier acceptance.
The current milestone is complete; stop for the user update.

''';s=s[:i]+text+s[i:];p.write_text(s)
p=r/'rust-star/INITIAL_SCOPE_STATUS.md';s=p.read_text();s=s.replace('| Latest optimization |','| Earlier 1K experiment |');lines=s.splitlines()
for i,line in enumerate(lines):
 if line.startswith('| Explicit scratch reuse |'):lines[i]=f'| Explicit scratch reuse | {outcome}: full C0/pool proofs and all 12 engine runs complete; {stats}. Verified {v["artifacts"]}-artifact report. {runtime.capitalize()}. | {nexttext} Not started; stop for the milestone update. |'
 if line.startswith('| Bounded 1K completion experiment |'):lines[i]=line.split('| Next:')[0]+'| Historical rejected wait experiment; explicit scratch-reuse study is complete below. |'
 if retained and line.startswith('| Current lower-context performance |'):lines[i]=f'| Current lower-context performance | Explicit scratch reuse retained with exact full 32K/128 C0 and five passing paired engine decisions; prefill {ratio("prefill_ms"):+.3f}%, footprint {ratio("peak_memory_footprint_bytes"):+.3f}%. | Refresh direct pinned-DwarfStar comparison; no larger-frontier claim. |'
s='\n'.join(lines)+'\n';i=s.index('D authorized the explicit');s=s[:i]+f'''D authorized one explicit scratch-reuse milestone. It is complete and
{outcome.lower()}; {runtime}. Report the verified result and the next
milestone, then stop. No follow-up implementation or inference has started.
Publication and256K remain outside this milestone.
''';p.write_text(s)
print(outcome,stats)
