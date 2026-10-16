from pathlib import Path
import json
w=Path(__file__).resolve().parent;root=w.parents[2];assert json.loads((w/'measurement-report-complete.json').read_text())['passed'];report=Path(json.loads((w/'report-location.json').read_text())['path']);rel=str(report.relative_to(root));summary=json.loads((report/'results/paired-summary.json').read_text())['contexts'][0];audit=json.loads((report/'independent-audit.json').read_text());profile=json.loads((report/'profile32k-analysis.json').read_text());freeze=json.loads((report/'inputs/freeze.json').read_text());lead=max(profile['group_totals'],key=lambda n:profile['group_totals'][n]['gpu_ms']);c=summary['candidate'];o=summary['oracle'];memory=audit['memory_distributions']['candidate']['median'];metrics=f"Rust/reference median prefill {c['prefill_ms']['median']/1000:.3f}/{o['prefill_ms']['median']/1000:.3f} s; steady decode {c['gen_steady_tps']['median']:.3f}/{o['gen_steady_tps']['median']:.3f} tokens/s";attention=profile['group_totals']['even/attention']
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();i=s.index('## 2026-09-07');s=s[:i]+f'''## 2026-09-07 — Even-layer attention batching milestone complete

- Native continuation attention now processes 4096 query rows per call with
  unchanged kernel reduction geometry and direct output to the tail buffer.
  Attention calls fall from 18,816 to 147. No oracle/fixture/arithmetic change.
- Candidate `{freeze['candidate_commit']}` passes 310 Rust tests,
  all six live C0/regression/lifecycle gates, five valid alternating pairs and
  the isolated complete profile. Reference remains pinned oracle-v5 ccbd16d.
- {metrics}.
  Peak Rust process memory: {memory:.3f} GiB. Full distributions, old-baseline
  comparison, host observations and standalone verification: `{rel}/`.
- Even-layer attention profile: {attention['gpu_ms']/1000:.3f} GPU seconds,
  {attention['host_elapsed_ms']/1000:.3f} host seconds, 147 calls. Largest current
  GPU stage group: `{lead}`. Target that measured cost next.
- Later OBS/video activity changed background load. All pairs are retained;
  exact isolated speedup attribution is not claimed. Memory remains a tradeoff.
- All historical reports and fixtures remain immutable. This is a 32K/128
  development result; broader contexts and 256K headline work remain future.

'''+s[i:];a=s.index('### Current resume checkpoint');b=s.index('### Accumulated implementation',a);s=s[:a]+f'''### Current resume checkpoint (updated 2026-09-07)

- Authorized even-layer attention milestone is complete. Native continuation
  attention executes whole 4K chunks with unchanged kernel arithmetic and no
  per-tile heads copy. Both native selector sites share this implementation.
- Current candidate `{freeze['candidate_commit']}`;
  executable SHA `{freeze['candidate_sha256']}`.
  Oracle-v5 source ccbd16d/tree a0a33f4 remains unchanged; executable SHA
  `{freeze['oracle_sha256']}`.
- All six live gates pass: native 12K internal states, detailed 8K, 2K, both
  complete 32K prefill/128-decode modes and retained two-cycle reset/reuse.
  310 Rust tests pass; unchanged Python toolkit retains 93-test evidence.
- Audited report `{rel}/README.md`; standalone `verify.py` needs no model.
  Five valid alternating pairs, two warmups, isolated complete profile.
  {metrics}. Rust peak memory {memory:.3f} GiB.
- Later OBS/video activity varied host load. Per-pair results and observations
  are retained; comparisons are descriptive, with memory tracked explicitly.
- Attention calls: 18,816 → 147. Largest current GPU stage group: `{lead}`.
  Next local milestone should target that measured cost while preserving v5
  C0, retained lifecycle and tracking memory. No completed layers need repeating.
- Frozen plan SHA `{freeze['plan_sha256']}`.
  Work: `{str(w.relative_to(root))}/`. Prior v5 baseline and v4 reports remain
  immutable, including their original host observations and failed attempts.
- Internal HC acceptance ends at 24K; full 32K outputs remain mandatory.
  Broader contexts, 256K headline measurements and agent co-design are future.

'''+s[b:];a=s.index('## Immediate Next Actions');b=s.index('## Entries',a);s=s[:a]+f'''## Immediate Next Actions

1. Attention batching milestone is complete. Review `{rel}/README.md`.
   Next measured target: `{lead}`. Preserve exact v5 outputs, retained-engine
   lifecycle and track memory, then validate any change in a fresh paired run.
2. Use oracle-v5 for new captures. Preserve historical fixture identities,
   reports and failed attempts. Internal HC targets are accepted through 24K;
   complete 32K prefill and 128-step decode C0 checks remain mandatory.
3. Broader contexts, 256K headline work and the coding-agent corpus remain future.
4. Run one large model process at a time, bound disk use, keep this journal
   updated and never resume rejected benchmark plans.
5. GitHub Actions billing and secure remote access remain in the manual ledger.

'''+s[b:];p.write_text(s)
p=root/'RUST_STAR_PROJECT.md';s=p.read_text();a=s.index('The completed upstream-refresh baseline is');b=s.index('Current implementation boundaries',a);s=s[:a]+f'''The current attention-batching baseline is `{rel}/README.md`:
{metrics}. Rust peak process memory is {memory:.3f} GiB.
Native continuation attention runs once per 4K chunk with exact v5 outputs.
The largest profiled GPU stage group is `{lead}`; target that cost next.
Oracle-v5 remains current. The upstream-refresh and v4 reports are historical.

'''+s[b:];p.write_text(s)
p=root/'rust-star/README.md';s=p.read_text();a=s.index('The [upstream-refresh baseline report]');b=s.index('The capture does not need',a);s=s[:a]+f'''The [current attention-batching report](reports/{report.name}/README.md)
includes five pairs, full C0/lifecycle validation, a separate profile and a
standalone evidence verifier. {metrics}. The previous
[upstream-refresh baseline](reports/32k-upstream-v5-cc032078/README.md) remains
immutable evidence of oracle-v5 acceptance.

'''+s[b:];p.write_text(s)
p=root/'rust-star/runtime/README.md';s=p.read_text();a=s.index('Oracle-v5 refresh is complete.');b=s.index('At 32K, `engine-profile`',a);s=s[:a]+f'''Native even-layer continuation attention now executes all 4096 query rows in
one call, preserving the original kernel reduction geometry. Raw-ring setup
and compressed-KV conversion run once per chunk, and attention writes directly
to the existing tail-input buffer. The separate 32-row controls remain intact.
All v5 full-output, shorter-context and retained reset/reuse gates pass.

The audited [attention-batching report](../reports/{report.name}/README.md)
records five pairs: {metrics}. Peak Rust process memory is {memory:.3f} GiB.
Attention calls fall from 18,816 to 147; the largest current GPU stage group is
`{lead}`. Historical reports and fixtures remain unchanged.

'''+s[b:];p.write_text(s)
print('Updated journal, charter and runtime/capture READMEs')
