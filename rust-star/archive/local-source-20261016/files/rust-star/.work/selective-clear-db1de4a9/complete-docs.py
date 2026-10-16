from pathlib import Path
import json
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());d=read(w/'decision.json');state=read(w/'final-runtime-state.json');n=len(read(w/'report-manifest.json')['artifacts']);a=read(w/'engine/audit.json') if (w/'engine/audit.json').exists() else None
outcome='RETAINED' if d['retained'] else 'REJECTED';report='rust-star/reports/selective-clear-db1de4a9/README.md';nexttext=(w/'next-milestone-design.md').read_text().strip();summary=''
if a:
 summary=', '.join(label+' '+format((a['summary'][k]['paired_ratio']['median']-1)*100,'+.3f')+'%' for k,label in [('prefill_ms','prefill'),('gen_steady_tps','steady decode'),('peak_rss_bytes','RSS'),('peak_memory_footprint_bytes','footprint')])
else:summary='A prerequisite failed; engine acceptance was not reached.'
finalstate='Candidate production source and executable are retained locally.' if d['retained'] else 'All accepted source/default state is restored.'
p=r/'RUST_STAR_JOURNAL.md';s=p.read_text();marker=s[s.index('## 2026-09-12 —'):].splitlines()[0]
entry=f'''## 2026-09-12 — Complete selective scratch initialization: {outcome.lower()}

- New guarded copy-then-in-place-mutation proof passes: 24 cases,
  2,147,549,248 payload bytes, 24,576 guard bytes and all 24 shortened-copy or
  omitted-mutation negative controls. It ran before runtime edits; the prior
  incomplete diagnostic verdict remains unchanged.
- Four exact full-copy destinations omit CPU clearing; all other 101
  initializers, source memcpy and prior checked completion stay unchanged.
  Actual helper proof passes 1 GiB of selected-policy checks, 62,043,240 other
  bytes, 192,360 guards, 2,205 identity reuses and 84 invalid checkouts.
  Source/observer inverses, 320 Rust tests, optimized production/observer builds,
  formatting, ring/router/weak-owner checks pass. One preparation-anchor
  assertion and correction before builds are preserved; no inference repeat.
- {summary}.
  Decision: {outcome}. {d['reason']}
  {finalstate} Final 42-source and executable hashes pass.
- Verified portable report: {report}
  ({n} artifacts). Full audit details, every failed attempt and all raw results
  are preserved. No publication, remote workflow or 256K work.
- Next milestone, NOT STARTED: {nexttext}
  Finish local commit and user update, then STOP.

'''
if a:entry=entry.replace('- '+summary,'''- Fresh 12K internal and normal/profiled/observed 32K full C0 pass
  50,031,360 exact FP32 comparisons; all 4,466 profile intervals pass. Pool audit
  covers 28 tiles, 119 snapshots and 2,940 checkouts: 112 selected clear
  omissions, 536,870,912 omitted bytes per tile, stable distinct identities,
  exact capacities, checked completion and final owner cleanup. All 12 engine
  processes complete in the fixed order with no exclusions beyond two planned
  warmups. Paired medians: '''+summary)
assert marker in s;s=s.replace(marker,entry+marker,1)
a0=s.index('### Current resume checkpoint');b0=s.index('  Explicit scratch reuse remains rejected',a0)
s=s[:a0]+f'''### Current resume checkpoint (updated 2026-09-12)

- COMPLETED SINGLE MILESTONE: selective scratch initialization {outcome.lower()}.
  {summary}.
  {finalstate}
  Report: {report} ({n} artifacts verify).
  Current work is complete; report the result and next proposal, then STOP.
'''+s[b0:]
a0=s.index('## Immediate Next Actions');b0=s.index('\n## Entries',a0)
s=s[:a0]+f'''## Immediate Next Actions

1. Selective scratch initialization is complete: {outcome}. Preserve all
   verified evidence and final source/executable identities.
2. Report full correctness/ownership and engine/memory results with their
   32K scope. Rust-versus-Rust gains are not DwarfStar parity.
3. Next proposal, not started: {nexttext}
4. STOP after the update. Keep oracle-v5 immutable, 256K paused and publication
   deferred; do not begin the next milestone automatically.
'''+s[b0:];p.write_text(s)
p=r/'RUST_STAR_PROJECT.md';s=p.read_text();marker='\nCurrent implementation boundaries, evidence, and immediate actions are maintained'
entry=f'''
The subsequent selective scratch initialization milestone is {outcome.lower()}.
Its new guarded mutation proof closes the prior test gap without rewriting the
old diagnostic verdict. Four full-copy destinations omit clearing; source
initializers and checked completion are preserved. {summary}.
{finalstate} Verified {n}-artifact report:
`{report}`. Next milestone, not started: {nexttext}
Stop after reporting this completed milestone.
'''
assert marker in s;s=s.replace(marker,entry+marker,1);p.write_text(s)
p=r/'rust-star/INITIAL_SCOPE_STATUS.md';s=p.read_text()
if d['retained']:
 start=s.index('| Current lower-context performance |');end=s.index('\n',start)
 s=s[:start]+f'| Current lower-context performance | Selective scratch initialization passes fresh full C0 and every unchanged engine gate: {summary}. Production source/default retained locally. | Refresh direct comparison with pinned DwarfStar at32K; larger-frontier evidence remains source-dependent. |'+s[end:]
start=s.index('| Scratch RSS diagnostic');end=s.index('\n',start);line=s[start:end];cut=line.rfind(' | ',0,line.rfind('|'));line=line[:cut]+' | Closed with its original gap; the new selective-initialization study below supplies the missing proof. |';s=s[:start]+line+s[end:]
marker='| Larger-frontier correctness on current source |';row=f'| Selective scratch initialization | {outcome}: {summary}. New mutation proof and all local checks pass. {n}-artifact verified report. {finalstate} | {nexttext} Not started; stop for update. |\n';assert marker in s;s=s.replace(marker,row+marker,1)
start=s.index('D authorized one bounded scratch RSS diagnostic milestone.')
s=s[:start]+f'''D authorized one selective scratch initialization milestone. It is complete:
{outcome}. {finalstate}
Report the verified evidence and next milestone, then stop. Publication and
256K remain deferred; no follow-up milestone has started.
''';p.write_text(s)
print('Current journal, charter and initial-scope status updated:',outcome)
