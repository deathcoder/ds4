from pathlib import Path
import json
w=Path(__file__).resolve().parent;root=w.parents[2]
a=json.loads((w/'engine/audit.json').read_text());s=a['summary'];retained=a['candidate_retained_as_engine_improvement'];report='rust-star/reports/attention-output-engine-4ced6ee8/README.md'
assert json.loads((w/'completion-pins.json').read_text())['passed']
assert '"passed": true' in (w/'report-verification.log').read_text()
pref=s['prefill_ms']['paired_ratio']['median'];steady=s['gen_steady_tps']['paired_ratio']['median'];rss=s['peak_rss_bytes']['paired_ratio']['median'];first=s['gen_first_ms']['paired_ratio']['median']
decision='retained' if retained else 'rejected for engine retention; runtime restored to the baseline'
entry=f'''## 2026-09-11 — Complete attention-output whole-engine milestone

- Runtime candidate 4ced6ee8 is {decision}. Verified portable report:
  {report}.
- Same-arena A/A and fresh candidate/current-control screens pass all gates
  and 53,552,873,472 FP32 comparisons. Every candidate pair improves the full
  isolated chain; separate AB/BA medians are 33.919–34.902% lower. The older
  failed two-arena study remains immutable and is not retroactively accepted.
- Narrow integration, 320 release tests, formatting/build, actual production
  Metal assembly/pipelines and complete-host inverse-edit proof pass. Full
  32K/128 C0 matches all 16,677,120 FP32 words; prefix/decode/legacy remain intact.
- Two warm-ups and five alternating unprofiled engine pairs all execute and
  preserve the exact oracle transcript. Median paired prefill latency changes
  {(pref-1)*100:+.3f}%, steady decode {(steady-1)*100:+.3f}%, peak RSS {(rss-1)*100:+.3f}% and
  first-step latency {(first-1)*100:+.3f}%. Role-median prefill is
  {s['prefill_ms']['baseline']['median']/1000:.3f}s baseline / {s['prefill_ms']['candidate']['median']/1000:.3f}s candidate.
  Every raw pair and preset criterion is retained in the report; no discarded
  pair, discretionary repeat, threshold revision or statistical-significance claim.
- This is a measured Rust-versus-Rust 32K decision against frozen 9d35953f;
  it does not update the DwarfStar comparison or establish higher-frontier C0.
  Extra FP16 scratch is 320 MiB with existing transient ownership.
- Full model/prompt/oracle/both-executable and all 42 tested-source pins pass
  after execution. The portable verifier recomputes source assemblies/scope,
  isolated intervals/gates, full C0 receipts and all whole-engine statistics.
  All GPU work is finished; 256K remains paused and nothing is pushed.

'''
p=root/'RUST_STAR_JOURNAL.md';t=p.read_text();i=t.index('## 2026-09-11');t=t[:i]+entry+t[i:]
aidx=t.index('- IN PROGRESS: whole-engine attention-output milestone',t.index('## Current State'));b=t.index('- COMPLETE: native compressed run-skip',aidx)
summary=f'''- COMPLETE: attention-output whole-engine milestone. Candidate 4ced6ee8 is
  {decision}. Report {report}.
  All neutral-control and candidate gates pass (53.553 billion isolated FP32
  comparisons); integrated full 32K/128 C0 passes all 16,677,120 FP32 words.
  All 320 host tests, production Metal and source-scope checks pass.
- Five complete alternating unprofiled engine pairs against frozen Rust 9d35953f
  show median paired prefill latency {(pref-1)*100:+.3f}%, steady decode {(steady-1)*100:+.3f}% and
  peak RSS {(rss-1)*100:+.3f}%. First-step latency changes {(first-1)*100:+.3f}%.
  Two warm-ups are excluded and preserved; all 12 transcripts are exact.
  See every raw pair and preset retention gate in the verified report.
- All final input/source/binary pins pass. GPU work is complete. The new path
  has full-model evidence at 32K only; 256K stays paused. No DwarfStar speedup
  or new higher-frontier claim follows from this Rust-versus-Rust comparison.


'''
t=t[:aidx]+summary+t[b:];aidx=t.index('## Immediate Next Actions');b=t.index('## Entries',aidx)
t=t[:aidx]+f'''## Immediate Next Actions

1. Preserve the completed whole-engine decision and its verified report:
   {report}.
   Keep the exact tested source/binary/input identities and all earlier failed
   studies available. No additional measurement is queued.
2. Any next performance milestone should start from current bottleneck evidence
   and end at a bounded full-model correctness and whole-engine decision. Do
   not substitute a synthetic/component improvement for an engine result or
   infer DwarfStar parity from this Rust-versus-Rust comparison.
3. Keep 256K paused. New larger-frontier correctness, reference comparisons,
   publication or broader campaigns require their own scope and prospective
   plans. No automatic push or return to the interrupted long-context campaign.

'''+t[b:];p.write_text(t)
p=root/'RUST_STAR_PROJECT.md';t=p.read_text();needle='Current implementation boundaries, evidence, and immediate actions are maintained'
i=t.index(needle)
paragraph=f'''The complete attention-output milestone now extends that historical screen.
Fresh same-arena controls and the candidate/current-control screen pass every
gate and 53.553 billion isolated FP32 comparisons. Runtime `4ced6ee8` passes
320 release tests, production Metal/source-scope checks and full 32K/128 C0.
The candidate is {decision}. Five alternating unprofiled engine pairs
against frozen Rust `9d35953f` show median paired prefill latency {(pref-1)*100:+.3f}%,
steady decode {(steady-1)*100:+.3f}% and peak RSS {(rss-1)*100:+.3f}% (first-step latency
{(first-1)*100:+.3f}%). All 12 normal-engine runs including two excluded warm-ups
preserve the oracle transcript. Report `{report}`
contains the complete prospective criteria, every raw pair and a portable
source/correctness/statistics verifier. Extra transient scratch is 320 MiB.
All final pins pass and GPU work is finished. This is a 32K Rust-versus-Rust
result, not DwarfStar parity or a higher-frontier validation; 256K remains paused.

'''
t=t[:i]+paragraph+t[i:];p.write_text(t)
print('Updated journal history, current state, next actions and project charter')
