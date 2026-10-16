from pathlib import Path
import json,statistics
w=Path(__file__).resolve().parent;root=Path(json.loads((w/'report-location.json').read_text())['path'])
def read(name):return json.loads((root/name).read_text())
raw=read('results/paired-raw.json');pairs=[p for p in raw['pairs'] if p['valid']]
s=read('results/paired-summary.json')['contexts'][0];old=read('baseline-tail-summary.json')['contexts'][0];audit=read('independent-audit.json');freeze=read('inputs/freeze.json')
latency=statistics.median(p['candidate']['prefill_ms']/p['oracle']['prefill_ms'] for p in pairs);old_latency=1/old['candidate_over_oracle']['prefill_tps']['median']
assert len(pairs)==5 and len(raw['pairs'])==6
assert not (s['candidate']['prefill_ms']['median']<old['candidate']['prefill_ms']['median'] and latency<old_latency)
lines=['# Rust Star: dense Q8 output projection — not retained','',
'The experiment passed its exact-output gates but did not meet the measured improvement criterion. The dense-output implementation is reverted to the accepted grouped projection path. The component profiler and its matching C declarations remain. No new performance baseline is accepted.','',
'Native tails were changed to eight aligned dense-Q8 output-low dispatches with direct token-major stores, using a physical pitch of 8192. This removed the grouped map/scatter path and two scratch buffers (22 → 20), while increasing native dispatches from 25 to 31. Metal kernel arithmetic was unchanged. `implementation.patch` preserves the rejected change.','',
'## Five-pair evidence','',
'Five alternating AB/BA pairs completed, with one prior warmup per engine. Correctness collection and profiling were disabled for timed runs; initial selection belongs to prefill, and decode includes evaluation, selection and commitment. Each engine invocation was a fresh process. These are 32K/128 development results.','',
'| Distribution (5 valid runs each) | Median | MAD | Min–max |','|---|---:|---:|---:|']
for role,label in [('candidate','Rust'),('oracle','Oracle-v5')]:
 for metric,name,scale in [('prefill_ms','prefill seconds',.001),('gen_steady_tps','steady decode tokens/s',1),('gen_first_ms','first decode ms',1)]:
  d=s[role][metric];lines.append(f"| {label} {name} | {d['median']*scale:.3f} | {d['mad']*scale:.3f} | {d['min']*scale:.3f}–{d['max']*scale:.3f} |")
lines+=['','| Pair | Attempt | Order | Rust prefill s | Oracle prefill s | Rust/oracle latency |','|---|---:|---|---:|---:|---:|']
for p in pairs:lines.append(f"| {p['repetition']} | {p['attempt']} | {p['order']} | {p['candidate']['prefill_ms']/1000:.3f} | {p['oracle']['prefill_ms']/1000:.3f} | {p['candidate']['prefill_ms']/p['oracle']['prefill_ms']:.5f} |")
lines+=['',f"Median paired prefill latency ratio: **{latency:.5f}**, versus **{old_latency:.5f}** for the accepted native-tail baseline ({(latency/old_latency-1)*100:+.2f}% ratio change; lower is better). Paired steady-decode throughput ratio: {s['candidate_over_oracle']['gen_steady_tps']['median']:.5f}.",'']
for role in ['candidate','oracle']:
 before=old[role]['prefill_ms']['median']/1000;after=s[role]['prefill_ms']['median']/1000
 lines.append(f"- {role.capitalize()} median prefill: {before:.3f} → {after:.3f} s ({(after/before-1)*100:+.2f}%).")
lines+=['', 'The absolute Rust reduction does not establish a benefit from this optimization: the reference improved more across the campaigns. These runs do not isolate the code change from varying host conditions. The criterion requires both lower absolute median prefill and a lower paired latency ratio; it failed.','',
'## Interruption and host conditions','',
'Pair 4 attempt 1 was interrupted during its oracle half by a host shutdown/restart. Its completed Rust half (538.625 s prefill) and original state remain preserved. The documented external-event recovery procedure reran the entire BA pair as attempt 2. The other completed pairs, including the slower third pair, were not discarded or retried. Raw evidence contains all six attempts; statistics use the five valid complete pairs. Recovery boot/state evidence is under `inputs/`.','',
'The campaign spans the restart, with warmups preceding it. Background browser, video/camera, updater and other host activity varied. Source review, report-helper edits and a temporary-copy Rustfmt check of about six seconds during pair 3 also occurred. No runtime build or second model process ran alongside timed measurements. Full host/load observations are included.','',
f"The audit retains {len(audit['unusual_thermal_observations'])} unusual thermal observations, including shutdown-related query failures. Median peak process memory was {audit['memory_distributions']['candidate']['median']:.3f} GiB for Rust and {audit['memory_distributions']['oracle']['median']:.3f} GiB for the reference. These are observed residency values under varying conditions, not allocation-capacity claims.",'',
'## Correctness and validation','',
'- 312 Rust tests and the candidate release build passed. The unchanged Python toolkit retains its 93-test acceptance evidence.',
'- All seven live gates passed: native 12K internal-state control, full 8K legacy control, 2K control, complete 32K output-only and internal modes, retained two-cycle reset/reuse, and a separate profiled full-output 32K gate.',
'- Both complete 32K modes and the profiled full-output gate match all 129,280 prefill logits and all 16,547,840 FP32 words across 128 decode steps. Accepted internal HC targets through 24K match. The 12K gate does not claim full 12K output-tensor equivalence.',
'- Retained lifecycle verifies both oracle transcripts; its same-process timings are ineligible for the paired comparison.',
'- Oracle-v5, fixtures, tokenizer evidence and historical failures are preserved. No tolerance or oracle change was introduced.',
'- Independent auditing rehashes every valid measurement, both warmups, the interrupted candidate measurement and its artifacts, and reproduces all paired statistics.','',
'## Separate profile failure','',
'The post-paired `engine-profile` attempt exited 1 with `rust-star: invalid tail timestamps` and wrote no profile artifact. The original log and explicit failure record are preserved as `profile32k.log` and `profile-failure.json`. It was not retried. The audit verifies this failure; it does not claim a successful post-paired profile.','',
'The getter rejects an error-valued counter or any non-increasing adjacent timestamp. Rejected raw values were not logged, so the exact condition is unresolved. Keep this fail-closed behavior until better diagnostics establish the cause; no timestamp-validity condition was relaxed. This is a known limitation of the optional profiler. Normal timed inference completed all five valid pairs.','',
'## Earlier exact-gated component evidence','',
'The separate premeasurement profiled C0 gate succeeded. Its 287 tails and 2,583 component rows pass ordering, coverage and attribution conservation checks. This evidence is distinct from the later failed profile. Counter shares are scaled to parent command GPU intervals and are perturbed attribution, not standalone kernel latency or a throughput result.','',
'| Component | Attributed GPU seconds | Share |','|---|---:|---:|']
components=read('correctness/components.json')['tail_components']['rows'];totals={}
for c in components:totals[c['component']]=totals.get(c['component'],0)+c['attributed_gpu_ms']
total=sum(totals.values())
for name,value in sorted(totals.items(),key=lambda x:-x[1]):lines.append(f'| {name} | {value/1000:.3f} | {value/total*100:.2f}% |')
lines+=['',
'## Disposition and reproduction','',
'The dense-output code is reverted; the validated grouped path and optional component profiler remain. Improve invalid-counter diagnostics before relying on unattended profiling. The proposed layers-0..2 initial-bootstrap scope and broader-context allocation work remain unapplied follow-up work. Stop at this milestone as requested.','',
f"Rejected candidate source: `{freeze['candidate_commit']}`. Executable SHA: `{freeze['candidate_sha256']}`.",f"Oracle source: `{freeze['oracle_commit']}`. Executable SHA: `{freeze['oracle_sha256']}`.",f"Frozen plan SHA: `{freeze['plan_sha256']}`.",'',
'Run `python3 verify.py` without the model to verify artifact integrity, paired statistics, the explicit interrupted attempt, full-output evidence, premeasurement component attribution and the preserved post-paired failure. The accepted performance baseline remains [32k-tail-89e32745](../32k-tail-89e32745/README.md); the original component study remains [32k-tail-components-e7b81aa2](../32k-tail-components-e7b81aa2/README.md).','']
(root/'README.md').write_text('\n'.join(lines))
print('Rendered negative experiment with explicit failed profile and separate successful gate evidence.')
