from pathlib import Path
import json,hashlib,datetime,statistics
w=Path(__file__).resolve().parent;read=lambda x:json.loads(x.read_text());sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();p=Path(read(w/'paired-location.json')['work'])
assert read(w/'measurement-complete.json')['status']=='complete'
a=read(p/'independent-audit.json');assert a['milestone_acceptance_passed']
c=read(p/'results/paired-summary.json')['contexts'][0];raw=read(p/'results/paired-raw.json')['pairs']
def dist(xs):
 m=statistics.median(xs);return dict(median=m,mad=statistics.median(abs(x-m) for x in xs),min=min(xs),max=max(xs))
for role in ['candidate','oracle']:
 c[role]['process_peak_bytes']=dist([x[role]['process_peak_bytes'] for x in raw])
for metric in ['process_peak_bytes','prefill_ms']:
 c['candidate_over_oracle'][metric]=dist([x['candidate'][metric]/x['oracle'][metric] for x in raw])
cap=read(w/'capacity-analysis.json');probe=read(w/'capacity-probe.json');gib=2**30
reason='Peak process RSS is lower in the predeclared paired comparison, while prefill and steady decode meet their non-regression thresholds.'
decision={'memory_changes_retained':True,'reason':reason,'milestone_acceptance_passed':True,'oracle_v5_unchanged':True,'independent_audit_sha256':sha(p/'independent-audit.json'),'paired_plan_sha256':read(p/'inputs/freeze.json')['plan_sha256'],'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stop_after_milestone':True}
(w/'retention-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
s='# Memory-efficient prefill and 128K allocation readiness\n\n**Candidate retained.** '+reason+'\n\n'
s+='| Metric | Candidate median | Accepted Rust median | Median within-pair ratio |\n|---|---:|---:|---:|\n'
for metric,label,scale in [('process_peak_bytes','Peak process RSS (GiB)',1/gib),('prefill_ms','Prefill (seconds)',.001),('gen_steady_tps','Steady decode (tokens/s)',1),('gen_first_ms','First decode step (ms)',1)]:
 s+=f"| {label} | {c['candidate'][metric]['median']*scale:.3f} | {c['oracle'][metric]['median']*scale:.3f} | {c['candidate_over_oracle'][metric]['median']:.5f} |\n"
s+='\nLower ratios favor the candidate except throughput, where higher is faster. Peak RSS is measured by process resource usage; it is distinct from Metal allocation size.\n\n'
s+='| Pair | Order | Candidate RSS GiB | Baseline RSS GiB | Candidate prefill s | Baseline prefill s | Decode ratio |\n|---|---|---:|---:|---:|---:|---:|\n'
for x in read(p/'results/paired-raw.json')['pairs']:
 ca=x['candidate'];ba=x['oracle'];s+=f"| {x['repetition']} | {x['order']} | {ca['process_peak_bytes']/gib:.3f} | {ba['process_peak_bytes']/gib:.3f} | {ca['prefill_ms']/1000:.3f} | {ba['prefill_ms']/1000:.3f} | {ca['gen_steady_tps']/ba['gen_steady_tps']:.5f} |\n"
s+='\nPrefill dispersion (median absolute deviation; range): '
for role in ['candidate','oracle']:
 d=c[role]['prefill_ms'];s+=f"{role} {d['mad']/1000:.3f}s; {d['min']/1000:.3f}–{d['max']/1000:.3f}s. "
s+='\n\n'
s+=f"The allocation probe passed with {probe['total_requested_bytes']/gib:.3f}GiB allocated and GPU-filled simultaneously. The device reports a {probe['max_buffer_length']/gib:.3f}GiB maximum individual buffer and {probe['recommended_max_working_set_size']/gib:.3f}GiB recommended working set. Modeled history increases by {cap['incremental_history_bytes_32k_to_128k']/gib:.3f}GiB between 32K and 128K. Adding that increase to the measured candidate maximum gives a **planning estimate of {cap['planning_estimate_32k_peak_plus_history_delta_bytes']/gib:.3f}GiB**, not a bound or a verified 128K working set.\n\n"
if (p/'profile32k-analysis.json').exists():
 q=read(p/'profile32k-analysis.json');s+=f"The separate post-paired profile passed transcript and interval checks ({q['rows']} parent rows, {q['tail_component_rows']} component rows). It is excluded from the timed comparison.\n\n"
else:s+='The separate post-paired profile failed once; its raw log and failure record are retained without a retry. The earlier profiled full-output gate passed independently.\n\n'
s+=(w/'report-notes.md').read_text();(w/'report-README.md').write_text(s);print(s[:3500])
