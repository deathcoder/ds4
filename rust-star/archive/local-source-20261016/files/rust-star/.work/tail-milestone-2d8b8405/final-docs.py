from pathlib import Path
import json
w=Path(__file__).resolve().parent; root=w.parents[2]
def read(p): return json.loads(p.read_text())
assert read(w/'measurement-report-complete.json')['passed']
r=Path(read(w/'report-location.json')['path']); rel=str(r.relative_to(root)); f=read(r/'inputs/freeze.json'); s=read(r/'results/paired-summary.json')['contexts'][0]; a=read(r/'independent-audit.json'); g=read(r/'profile32k-analysis.json')['group_totals']; lead=max(g,key=lambda n:g[n]['gpu_ms']); old=read(r/'baseline-ingress-summary.json')['contexts'][0]
c=s['candidate']; o=s['oracle']; memory=a['memory_distributions']['candidate']['median']; change=100*(c['prefill_ms']['median']/old['candidate']['prefill_ms']['median']-1)
metrics=f"Rust/reference median prefill {c['prefill_ms']['median']/1000:.3f}/{o['prefill_ms']['median']/1000:.3f} s; steady decode {c['gen_steady_tps']['median']:.3f}/{o['gen_steady_tps']['median']:.3f} tokens/s."
p=root/'RUST_STAR_JOURNAL.md';t=p.read_text();i=t.index('## 2026-09-07');t=t[:i]+f'''## 2026-09-07 — Native tail scratch/output milestone complete

- Native tails reuse 22 command-local buffers, released before decode/reset.
  Retained HC output is written directly, avoiding a 256 MiB copy per tail.
  The unused 192 MiB FP32 routed-mid destination/conversion is removed.
  Kernel arithmetic, v5 targets and legacy diagnostic paths are unchanged.
- Candidate `{f['candidate_commit']}` passes 310 Rust tests, all six live
  C0/regression/lifecycle gates, five valid pairs and the separate profile.
  Standalone audited report: `{rel}/README.md`.
- {metrics}
  Observed Rust peak process memory {memory:.3f} GiB. Absolute prefill change
  versus the prior ingress campaign: {change:+.2f}%; host-load and swap
  observations qualify cross-session speed and memory attribution.
- Current largest GPU stage group is `{lead}`. Inspect its component costs
  before choosing another optimization. Broader context work remains future.

'''+t[i:];i=t.index('### Current resume checkpoint');j=t.index('### Accumulated implementation',i);t=t[:i]+f'''### Current resume checkpoint (updated 2026-09-07)

- Authorized native tail scratch/output milestone is complete. Whole-chunk
  tails reuse 22 temporary buffers, write directly to retained HC output and
  omit unused FP32 routed-mid conversion (25 dispatches instead of 26).
  Retained HC remains separately owned; cache clears at decode handoff/reset.
- Candidate `{f['candidate_commit']}`;
  executable SHA `{f['candidate_sha256']}`.
  Oracle-v5 source ccbd16d remains unchanged; executable SHA
  `{f['oracle_sha256']}`.
- 310 Rust tests and all six live gates pass: native 12K, detailed 8K, 2K,
  complete 32K output-only/diagnostic and two retained reset/reuse cycles.
  Unchanged Python toolkit retains its 93-test reference acceptance evidence.
- Five alternating pairs and both warmups pass on attempt 1; separate profile
  and independent audit pass. Report: `{rel}/README.md`.
  {metrics} Observed Rust peak memory {memory:.3f} GiB.
- Cross-session comparisons remain descriptive; retain host-load/swap evidence
  alongside speed and memory. No fixture, arithmetic or oracle change.
- Largest GPU group: `{lead}`. Inspect its internal costs before the next
  optimization; do not repeat completed scheduling changes.
- Frozen plan SHA `{f['plan_sha256']}`.
  Evidence: `{w.relative_to(root)}/`. All prior reports/failures stay immutable.
- Internal HC targets remain accepted through 24K; full 32K output gates are
  mandatory. Broader contexts, 256K headline and agent co-design remain future.

'''+t[j:];i=t.index('## Immediate Next Actions');j=t.index('## Entries',i);t=t[:i]+f'''## Immediate Next Actions

1. Native tail milestone is complete. Review `{rel}/README.md`.
   Inspect component costs within `{lead}` before selecting the next exact
   optimization. Preserve v5 C0 and lifecycle, and measure memory alongside speed.
2. Keep oracle-v5 and historical fixtures/reports immutable. Internal targets
   are accepted through 24K; full 32K prefill/128-decode C0 remains mandatory.
3. Broader contexts, 256K headline work and the coding-agent corpus remain future.
4. Run one large model process at a time, bound disk use and keep this journal
   current. Never resume a rejected benchmark plan.
5. GitHub Actions billing and secure remote access remain in the manual ledger.

'''+t[j:];p.write_text(t)
p=root/'RUST_STAR_PROJECT.md';t=p.read_text();i=t.index('The current ingress-batching baseline is');j=t.index('Current implementation boundaries',i);t=t[:i]+f'''The current native-tail report is `{rel}/README.md`.
{metrics} Observed Rust peak memory is {memory:.3f} GiB.
Native tails reuse scratch, write retained HC directly and omit an unused
conversion with exact v5 outputs. Largest GPU stage group: `{lead}`;
inspect its component costs next. Host load and swap qualify cross-session
speed and memory comparisons. Oracle-v5 and historical evidence are unchanged.

'''+t[j:];p.write_text(t)
p=root/'rust-star/README.md';t=p.read_text();i=t.index('The [current ingress-batching report]');j=t.index('The capture does not need',i);t=t[:i]+f'''The [current native-tail report](reports/{r.name}/README.md) includes
five pairs, full C0/lifecycle validation, a separate profile and a standalone
verifier. {metrics} Read the report's host-condition limits when comparing
campaigns. Earlier reports remain immutable evidence.

'''+t[j:];p.write_text(t)
p=root/'rust-star/runtime/README.md';t=p.read_text();i=t.index('Native odd-layer continuation ingress now');j=t.index('At 32K, `engine-profile`',i);t=t[:i]+f'''Native tails now reuse 22 command-local scratch buffers across layers and
chunks, write retained HC directly and omit unused routed-mid FP32 conversion.
The transient cache clears at prefill-to-decode handoff and request reset.
Retained HC buffers remain separately owned. Existing whole-chunk ingress and
attention batching stay in place; legacy diagnostic paths remain unchanged.

The [native-tail report](../reports/{r.name}/README.md) records all six live
C0/lifecycle gates, five pairs and the independent evidence audit.
{metrics} Observed Rust peak memory is {memory:.3f} GiB.
Host-load/swap differences limit cross-session attribution. The largest GPU
stage group is `{lead}`; inspect its component costs before optimizing further.

'''+t[j:];p.write_text(t)
print('Updated current docs and journal')
