from pathlib import Path
import json,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];r=root/'rust-star/reports/128k-measure-ef37e566'
read=lambda p:json.loads(p.read_text())
assert read(r/'independent-audit.json')['passed'] and read(r/'final-model-verification.json')['passed']
s=read(r/'extended-summary.json')['contexts'][0];n=len(read(r/'artifact-hashes.json')['artifacts']);date=datetime.datetime.now().astimezone().date().isoformat()
c=s['candidate'];o=s['oracle'];ratio=s['candidate_over_oracle'];gib=2**30
result=f'''- Candidate ef37e566 passes 317 Rust tests, 94 Python tests, release build,
  six live gates and the 287-tail/2,583-interval 32K profile audit. Fresh full
  128K C0 matches every prefill word and all 128 decode rows. Retained 128K
  completes both exact transcripts with request reset and residency reuse.
- Five alternating fresh-process pairs and two warm-ups pass on first attempts.
  Rust/oracle median prefill {c['prefill_ms']['median']/1000:.3f}/{o['prefill_ms']['median']/1000:.3f}s;
  steady decode {c['gen_steady_tps']['median']:.3f}/{o['gen_steady_tps']['median']:.3f} tokens/s.
  Median paired prefill latency ratio {ratio['prefill_ms']['median']:.6f}; steady
  throughput ratio {ratio['gen_steady_tps']['median']:.6f}. This is characterization,
  not a speedup claim. All five samples remain in the report.
- Fresh median peak RSS {c['process_peak_bytes']['median']/gib:.3f}/{o['process_peak_bytes']['median']/gib:.3f}GiB
  Rust/oracle; retained two-cycle peak 94.571GiB is a separate diagnostic.
  These warmed measurements are distinct from prior unwarmed diagnostic RSS
  and are not universal Metal/model memory bounds.
- Independent statistics/artifact audit, portable verifier and final model,
  prompt, oracle source/binary hashes pass. Report:
  rust-star/reports/128k-measure-ef37e566/README.md ({n} verified artifacts).
  The earlier 5f3b9672 report-writer failure and red regression are preserved
  under rejected/; the corrected candidate has no failed timed attempts.
- Fresh and retained measurements now admit the pinned 2K/32K/128K frontiers,
  each with 128 decode steps. Full-logit checks remain independent from timing.
  Retained support covers the fixed transcript/lifecycle, not arbitrary serving;
  live profile acceptance remains 32K and internal HC acceptance remains 24K.
- STOP at this completed milestone. Proposed next: profile the dominant 128K
  transformer-chain prefill cost, validate counters and C0, then select and
  measure a focused optimization. It is not started. 256K and the coding-agent
  corpus remain future work. No quota burning or push performed.
'''
p=root/'RUST_STAR_JOURNAL.md';text=p.read_text();i=text.index('\n## 2026-',text.index('Journal entries are reverse chronological.'))+1;text=text[:i]+f'## {date} — 128K measurement and retained-engine milestone COMPLETE\n\n'+result+'\n'+text[i:]
text=text.replace('### Current resume checkpoint (updated 2026-09-08)',f'### Current resume checkpoint (updated {date})',1)
a=text.index('- ACTIVE: authorized 128K measurement/retained-engine milestone.');b=text.index('- 128K correctness milestone COMPLETE',a);text=text[:a]+f'- COMPLETE: 128K measurement/retained-engine milestone, candidate ef37e566.\n  317 Rust / 94 Python tests, six live gates, five valid pairs and two warm-ups;\n  portable report has {n} verified artifacts. See the newest completion entry\n  and rust-star/reports/128k-measure-ef37e566/README.md for measured latency,\n  throughput and warmed memory. Independent audits and final pins pass.\n  Earlier writer rejection is preserved. STOP; no next milestone is active.\n\n'+text[b:]
text=text.replace('fixtures/measurement commands retain 32K; internal HC targets retain 24K.', 'diagnostic fixtures retain 32K; timed fresh/retained support now includes the\n  pinned 128K/128 transcript. Internal HC targets retain 24K.',1)
a=text.index('## Immediate Next Actions');b=text.index('\n## Entries',a);text=text[:a]+'''## Immediate Next Actions

1. STOP at the completed 128K measurement/retained-engine milestone. All five
   pairs, correctness/lifecycle gates, final audits, portable evidence and
   documentation are complete. Do not resume quota-burning mode.
2. Proposed next milestone, awaiting user instruction: profile the dominant
   128K remaining-transformer prefill chain, validate component counters and
   C0, then choose a focused optimization and evaluate it with pinned pairs.
   Do not assume the dominant 32K kernel remains dominant at 128K.
3. Keep oracle-v5 and historical evidence immutable. Full 32K/128 C0 remains
   mandatory; accepted internal HC targets end at 24K. The new 128K comparison
   adds a frontier and does not rewrite historical 32K performance results.
4. 256K headline work and the coding-agent corpus remain future. Full KV history
   still scales with context; consider bounded raw history if capacity requires it.
   Do not revive rejected dense-Q8 execution without a new measured hypothesis.
5. Run one model workload at a time, bound disk use and keep this journal current.
   GitHub Actions billing and secure remote access remain in the manual ledger.
''' +text[b:];p.write_text(text)
p=root/'RUST_STAR_PROJECT.md';text=p.read_text();a=text.index('The subsequent 128K measurement/retained-engine milestone is now authorized');b=text.index('\nCurrent implementation boundaries',a)
text=text[:a]+f'''The subsequent 128K measurement/retained-engine milestone is complete at
ef37e566. Report: `rust-star/reports/128k-measure-ef37e566/README.md`
({n} verified artifacts). All 317 Rust tests, 94 Python tests, six live gates,
five alternating pairs and two warm-ups pass. The earlier stale writer-guard
failure is preserved; the corrected candidate has no failed timed attempts.
Rust/oracle median prefill is {c['prefill_ms']['median']/1000:.3f}/{o['prefill_ms']['median']/1000:.3f}s,
steady decode {c['gen_steady_tps']['median']:.3f}/{o['gen_steady_tps']['median']:.3f} tokens/s, and fresh peak RSS
{c['process_peak_bytes']['median']/gib:.3f}/{o['process_peak_bytes']['median']/gib:.3f}GiB. This is a measured 128K frontier, not a speedup claim
or a replacement for the historical 32K comparisons. Final input pins and the
independent evidence/statistics audit pass. Timed execution checks the exact
pinned transcript; fresh full-logit C0 is separate. Retained validation covers
two cycles with request-state reset and shared model/context/residency, not
arbitrary-prompt serving. The next proposed milestone is to profile the 128K
remaining-transformer chain and evaluate a focused optimization; it is not
started. Live profile acceptance remains 32K, with internal HC through 24K.
''' +text[b:]
text=text.replace('targets still end at 24K. Built-in measurement commands remain 32K; the separate\nexternal-reference diagnostic supports the new frontier.', 'targets still end at 24K. At that checkpoint measurement commands remained\n32K; the separate external-reference diagnostic supported the new frontier.',1);p.write_text(text)
print('Completion journal and charter updated; review before final commit')
