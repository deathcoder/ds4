from pathlib import Path
import json
w=Path(__file__).resolve().parent;root=w.parents[2];a=json.loads((w/'audit.json').read_text());pct=lambda k:100*(a['paired_ratios'][k]['median']-1)
result=f'prefill latency {pct("prefill_ms"):+.3f}%, steady decode throughput {pct("gen_steady_tps"):+.3f}%, peak RSS {pct("process_peak_bytes"):+.3f}% (paired Rust/DwarfStar medians)'
proposal='Audit whether wider routed gate/up token tiles can reuse dequantized expert weights while preserving each output accumulation order, then screen one bounded layout or scheduling hypothesis with exact intermediate comparisons and contemporary A/A controls. Preserve the failed routed-input-half screens. Freeze criteria before timing; stop on failed neutrality. Full C0 and whole-engine evidence are required before any later runtime acceptance.'
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();entry=f'''## 2026-09-12 — Complete direct 32K DwarfStar comparison

- All twelve fresh engine processes pass: two excluded warmups and five fixed
  AB/BA/AB/BA/AB pairs. No repeats or speed exclusions. Result: {result}.
- Existing strict adapters validate sampling-inclusive timing; all Rust timed
  transcripts match. Unchanged pinned DwarfStar uses its accepted producer
  evidence; its CSV does not expose a fresh transcript or full-logit claim.
- Complete before/after model, prompt, oracle, 7,669 fixture and 42 source pins
  pass. Accepted Rust executable remains 7590af7ade11d5b07d65f0fefc9e01c02668318896fc4e955307888d85cdaa35.
  Prior 223-artifact acceptance report re-verifies, including 50,031,360 exact
  FP32 comparisons. No runtime edits, rebuilds or model inference repeats.
- All 4,466 saved profile intervals regenerate byte-identically. Initial replay
  failed on Python tuple-versus-JSON-list equality in the existing-file branch;
  unchanged audit in a fresh temporary directory reproduces every saved byte.
  Preserve the failed replay. Largest component remains routed gate/up, 30.150s.
- Portable report: rust-star/reports/dwarfstar-32k-e0bec446/README.md.
  Verifier recomputes all pairs and rejects corrupted sampling/transcript
  metadata even after its artifact hashes are updated. Five pairs do not
  establish statistical significance; this comparison is limited to 32K/128.
- Next milestone, NOT STARTED: {proposal}
- Finish local commit and user update, then STOP. Oracle-v5 immutable;
  publication deferred; 256K remains paused.

''';s=s.replace('## 2026-09-12 — Start direct',entry+'## 2026-09-12 — Start direct',1)
s=s.replace('- ACTIVE SINGLE MILESTONE: direct 32K DwarfStar comparison in\n  rust-star/.work/dwarfstar-32k-e0bec446; measurement only, then STOP.','- COMPLETED SINGLE MILESTONE: direct 32K DwarfStar comparison.\n  '+result+'.\n  Report: rust-star/reports/dwarfstar-32k-e0bec446/README.md.\n  Runtime unchanged. Report next proposal, then STOP.',1)
aidx=s.index('## Immediate Next Actions\n');b=s.index('\n## Entries',aidx);s=s[:aidx]+f'''## Immediate Next Actions

1. Direct 32K DwarfStar measurement milestone is complete. Preserve all raw
   results, portable verification and unchanged runtime identities.
2. Report the measured reference gap and the 32K-only limits.
3. Next proposal, not started: {proposal}
4. STOP after local commit and user update. Keep oracle-v5 immutable,
   publication deferred and 256K paused.
'''+s[b:];p.write_text(s)
p=root/'RUST_STAR_PROJECT.md';s=p.read_text();old='Next milestone, not started: Refresh the direct 32K comparison between the newly accepted Rust runtime and pinned DwarfStar on the same model, prompt and 128-token decode workload. Use fresh paired processes and the existing strict measurement adapters to quantify the remaining performance gap, then choose the next bottleneck from the current profile. Keep larger-frontier work and publication deferred. This proposal has not started.';assert old in s;s=s.replace(old,f'''The subsequent direct 32K/128 comparison at e0bec446 is complete:
{result}. All two warmups and five alternating pairs pass, with complete
before/after input pins and unchanged accepted runtime. Verified report:
`rust-star/reports/dwarfstar-32k-e0bec446/README.md`. Full C0 evidence is reused
for unchanged binaries; timed Rust transcript checks are not new full C0, and
DwarfStar CSV has no per-run transcript. All 4,466 accepted profile intervals
replay exactly; routed gate/up remains largest at 30.150s. Next milestone,
not started: {proposal} Larger frontiers and publication stay deferred.''',1);p.write_text(s)
print(result)
