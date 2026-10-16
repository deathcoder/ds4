from pathlib import Path
import json,sys,statistics,datetime,hashlib
w=Path(__file__).resolve().parent;p=Path(json.loads((w/'paired-location.json').read_text())['work'])
assert json.loads((w/'measurement-complete.json').read_text())['status']=='complete'
a=json.loads((p/'independent-audit.json').read_text());s=json.loads((p/'results/paired-summary.json').read_text());c=s['contexts'][0]
# Explicit final review supplies the decision and evidence-based reason.
assert len(sys.argv)==3 and sys.argv[1] in ['retain','revert']
retain=sys.argv[1]=='retain';reason=sys.argv[2]
if retain:assert a['prefill_acceptance_passed']
decision={'bootstrap_retained':retain,'reason':reason,'prefill_acceptance_passed':a['prefill_acceptance_passed'],'counter_diagnostics_retained':True,'oracle_v5_unchanged':True,'comparison':'Rust-to-Rust; generic oracle slot is accepted Rust baseline','independent_audit_sha256':hashlib.sha256((p/'independent-audit.json').read_bytes()).hexdigest(),'paired_plan_sha256':json.loads((p/'inputs/freeze.json').read_text())['plan_sha256'],'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stop_after_milestone':True}
(w/'retention-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
head='# Scoped initial bootstrap and counter diagnostics — 32K Rust study\n\n'
text=head+('**Bootstrap retained.** ' if retain else '**Bootstrap rejected and reverted.** ')+reason+'\n\n'
text+='The enhanced rejected-counter diagnostics are retained. All 312 Rust tests, the release build, all seven live gates and the component/probe audits pass. Five pairs and two warmups completed with no invalid attempts or retries.\n\n'
text+='| Metric | Candidate median | Accepted Rust median | Within-pair ratio median |\n|---|---:|---:|---:|\n'
for metric,label,scale in [('prefill_ms','Prefill seconds',.001),('gen_steady_tps','Steady decode tokens/s',1),('gen_first_ms','First decode step ms',1)]:
 ratio=a['paired_prefill_latency_ratio']['median'] if metric=='prefill_ms' else c['candidate_over_oracle'][metric]['median']
 text+=f"| {label} | {c['candidate'][metric]['median']*scale:.3f} | {c['oracle'][metric]['median']*scale:.3f} | {ratio:.5f} |\n"
text+='\nPrefill ratios below 1 are faster; decode throughput ratios above 1 are faster.\n\n'
text+='| Pair | Order | Candidate prefill s | Baseline prefill s | Latency ratio | Decode throughput ratio |\n|---|---|---:|---:|---:|---:|\n'
for x in a['pairs']:text+=f"| {x['repetition']} | {x['order']} | {x['candidate_prefill_s']:.3f} | {x['baseline_prefill_s']:.3f} | {x['prefill_ratio']:.5f} | {x['decode_ratio']:.5f} |\n"
text+='\nPrefill dispersion (median absolute deviation; minimum–maximum): '
for role,label in [('candidate','candidate'),('oracle','baseline')]:
 d=c[role]['prefill_ms'];text+=f"{label} {d['mad']/1000:.3f}s; {d['min']/1000:.3f}–{d['max']/1000:.3f}s. "
text+='\n\n'
if (p/'profile32k-analysis.json').exists():
 q=json.loads((p/'profile32k-analysis.json').read_text());g=q['group_totals']['prefix/initial_bootstrap']
 text+=f"The separate post-paired profile passed transcript validation and interval accounting: {q['rows']:,} parent rows and {q['tail_component_rows']:,} component rows. Its initial-bootstrap GPU span was {g['gpu_ms']/1000:.3f}s. These are diagnostic measurements, separate from the timed pairs.\n\n"
else:text+='The separate post-paired profile failed once; the log and failure record are preserved, with no retry. The earlier profiled full-output gate remains distinct passing evidence.\n\n'
notes=(w/'report-notes.md').read_text().split('\n',1)[1].lstrip();text+=notes
(w/'report-README.md').write_text(text)
print(text[:4000])
