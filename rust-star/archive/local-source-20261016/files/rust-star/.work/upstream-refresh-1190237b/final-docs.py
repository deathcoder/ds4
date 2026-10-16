from pathlib import Path
import json
w=Path(__file__).resolve().parent;root=w.parents[2];assert json.loads((w/'measurement-report-complete.json').read_text())['passed'];report=Path(json.loads((w/'report-location.json').read_text())['path']);rel=str(report.relative_to(root));r=json.loads((report/'results/paired-summary.json').read_text())['contexts'][0];f=json.loads((report/'inputs/freeze.json').read_text());profile=json.loads((report/'profile32k-analysis.json').read_text());largest=max(profile['group_totals'],key=lambda n:profile['group_totals'][n]['gpu_ms']);candidate=r['candidate'];oracle=r['oracle'];metrics=f"median Rust/reference prefill {candidate['prefill_ms']['median']/1000:.3f}/{oracle['prefill_ms']['median']/1000:.3f}s; steady decode {candidate['gen_steady_tps']['median']:.3f}/{oracle['gen_steady_tps']['median']:.3f} tokens/s"
entry=f'''## 2026-09-07 — Upstream refresh and oracle-v5 baseline milestone complete

- Completed pinned upstream c0a6119f refresh, oracle-v5 ccbd16d acceptance,
  applicable resident M1/Q2 deterministic-selector port, complete correctness
  and retained lifecycle gates, five fresh paired measurements and separate
  profile. Durable report and standalone tensor/evidence verifier: `{rel}/`.
- {metrics}. Full distributions and paired ratios are in the report. These
  remain 32K development results, not the 256K primary workload.
- Four repeated 2K/32K reference frontiers, final 8K control, exact tokenizer,
  two full129-row traces and two full HC inventories at each 12K–24K pass.
  New32K prefill target is exactly reproduced by the selector-only v4 control;
  all 129 selected IDs remain unchanged. All historical targets are preserved.
- The initial Rust diagnostic stopped on historical16K layer34 HC. Direct v5
  captures reproduced Rust, establishing separately versioned internal targets.
  A separate 8K selector fixture also versions four equal-score index swaps
  proven by two direct v5 captures; score inputs and selected sets are unchanged.
  All stopped attempts and successful full-output runs remain in evidence.
- Final Rust candidate `{f['candidate_commit']}` passes 310 Rust and 93 Python
  tests, both fresh full 32K modes,12K internal/2K/8K controls and two retained
  cycles. Both full-output gates match v5 exactly. Internal HC
  acceptance stays through 24K; later internals are not claimed as oracle-exact.
- Candidate SHA `{f['candidate_sha256']}`;
  reference SHA `{f['oracle_sha256']}`;
  original plan SHA `{f['plan_sha256']}`.
- Profile's largest GPU stage group: `{largest}`. Next local milestone should
  reduce that measured cost while preserving full v5 outputs and lifecycle.
  No additional optimization or broader-context performance claim is included.

'''
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();i=s.index('## 2026-09-07');s=s[:i]+entry+s[i:];start=s.index('### Current resume checkpoint');end=s.index('### Accumulated implementation',start);s=s[:start]+f'''### Current resume checkpoint (updated 2026-09-07)

- Authorized upstream-refresh milestone is complete. Accepted oracle-v5 source
  ccbd16d/tree a0a33f4 is pinned to upstream c0a6119f. Rust candidate
  `{f['candidate_commit']}` uses deterministic argsort/merge and retains
  batched even-layer tails. No pending layer implementation or validation gate.
- Audited report: `{rel}/README.md`; run its `verify.py` without the model.
  Five valid alternating pairs, two warmups and one separate complete profile.
  {metrics}.
- Reference: four2K/32K repeats,8K control, tokenizer identity, two full129-row
  traces and two full HC inventories per 12K/16K/20K/24K. Rust:310/93 tests,
  both 32K full-logit modes,12K/2K/8K controls and retained two-cycle lifecycle.
- Explicit v5 fixtures replace current 32K and later internal targets; v3/v4
  fixtures and reports remain immutable. Initial stale16K checkpoint failure
  is retained and directly attributed to reference-version drift, not waived.
- Largest profiled GPU stage group: `{largest}`. Use this fresh baseline for
  the next optimization. Broader contexts and256K headline work remain future.
- Frozen candidate SHA `{f['candidate_sha256']}`;
  oracle SHA `{f['oracle_sha256']}`;
  plan SHA `{f['plan_sha256']}`. Work: `{str(w.relative_to(root))}/`.
- Prior optimized v4 report remains `rust-star/reports/32k-prefill-dd0d26c5/`;
  its host-restart disclosure and earlier baseline remain unchanged. Cross-
  campaign comparisons are descriptive, not isolated causal speedup claims.

'''+s[end:];start=s.index('## Immediate Next Actions');end=s.index('## Entries',start);s=s[:start]+f'''## Immediate Next Actions

1. The upstream-refresh milestone is complete. Review `{rel}/README.md`.
   Next meaningful local milestone: optimize the measured `{largest}` cost,
   preserving exact v5 outputs and the retained-engine lifecycle, then use a
   fresh paired campaign to measure the change. Do not repeat completed layers.
2. Use oracle-v5 for new captures. Preserve all historical fixture identities,
   old benchmark reports and failed attempts. Current internal HC acceptance
   ends at 24K; full 32K prefill and 128-step output checks remain mandatory.
3. Broader native contexts and the 256K primary headline remain future work.
   Define the coding-agent workload corpus before agent co-design claims.
4. Run one large model process at a time, keep disk use bounded, and update this
   journal at each meaningful step. Never resume a rejected benchmark plan.
5. GitHub Actions billing lock and deferred secure remote access remain in the
   manual task ledger.

'''+s[end:];p.write_text(s)
p=root/'RUST_STAR_PROJECT.md';s=p.read_text().replace('Oracle-v4 supersedes that producer for new work, as recorded above.', 'Oracle-v4 subsequently superseded that producer; oracle-v5 is now current, as recorded above.');s=s.replace('Use v4 for new captures; v3 and earlier producers remain\nhistorical. Existing fixture identities are not retroactively changed.', '''V4 remains historical evidence. Oracle-v5 supersedes it for new captures:
source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, tree
`a0a33f4a00c3c78cd40350785485203e7c6b0b1d`, based on upstream c0a6119f.
It passes official gates, four repeated 2K/32K captures, 8K equivalence and two
full129-row traces. Deterministic argsort/merge explains the changed32K prefill
target; generated token IDs stay identical. Independently repeated internal
HC targets through 24K are explicitly versioned too. Existing fixture producer
identities and historical results are never retroactively changed.''');marker='## Open Items\n';i=s.index(marker)+len(marker);s=s[:i]+f'''\nThe completed upstream-refresh baseline is `{rel}/README.md`: {metrics}.
Its largest profiled GPU stage group is `{largest}`; target that cost next.
Oracle-v5 is current; the previous v4 optimization remains historical evidence.
'''+s[i:];s=s.replace('Upstream review now prioritizes a controlled oracle-v5\n  refresh and applicable M1/Q2 ports before further attention tuning. Pin the\n  upstream snapshot, explain output changes, preserve historical targets and\n  establish a fresh baseline before selecting the next local optimization.', 'The subsequent oracle-v5 refresh is complete, as recorded above. Preserve\n  this historical optimization report and use the new baseline for further work.');p.write_text(s)
p=root/'rust-star/README.md';s=p.read_text();start=s.index('This directory contains');end=s.index('The capture does not need',start);s=s[:start]+f'''This directory contains versioned oracle capture tools. Use
`capture_oracle_v5.py` for new captures on the target M1 Ultra. V5 pins upstream
c0a6119f with sampling-inclusive benchmark timing and hardware-aware official
tests. Four 2K/32K repeats,8K equivalence and two complete 128-step traces pass.
Changed32K and internal targets have new versions; historical fixtures remain
immutable. Current status is maintained in `../RUST_STAR_JOURNAL.md`.

The [upstream-refresh baseline report](reports/{report.name}/README.md) includes
five pairs, the complete C0/lifecycle gates, an isolated profile and a standalone
verifier with the full v5 output tensors. It records {metrics}.

'''+s[end:];p.write_text(s)
p=root/'rust-star/runtime/README.md';s=p.read_text();start=s.index('Fresh32K normalized measurement');end=s.index('At 32K, `engine-profile`',start);s=s[:start]+f'''Oracle-v5 refresh is complete. Native 32K even-layer tails still run once per 4K
chunk after the 32-token attention tiles. Both selector sites now use upstream
argsort/merge; full 32K prefill and 128 decode steps match the new v5 targets
exactly. Independently repeated v5 internal checkpoints are accepted through
24K, with 2K/8K controls and both retained reset/reuse cycles passing.

The audited [v5 baseline](../reports/{report.name}/README.md) records five pairs:
{metrics}. Its separate profile identifies `{largest}` as the largest GPU
stage group. Use this baseline for the next optimization; historical v4 reports
and the stale diagnostic-target failures remain preserved.

'''+s[end:];p.write_text(s)
p=root/'rust-star/build_benchmark_oracle.py';s=p.read_text().replace('default="oracle-v4"','default="oracle-v5"');p.write_text(s)
print('Updated journal, charter, runtime/capture READMEs and current builder default')
