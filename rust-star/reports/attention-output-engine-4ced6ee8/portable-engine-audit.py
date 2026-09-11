from pathlib import Path
import json,hashlib,statistics,math,sys
w=Path(__file__).resolve().parent;root=w.parents[2];d=w/'engine'
sys.path.insert(0,str(w/'support'))
from ruststar_measurement_lib import parse_engine_run
read=lambda p:json.loads(p.read_text())
manifest=read(w/'artifacts.json')['files']
def sha(p):
 key=str(p.relative_to(w));entry=manifest[key]
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['stored_sha256']
 return entry['original_sha256']
plan=read(w/'engine-plan.json');build=read(w/'integration/candidate-build.json');med=statistics.median
assert read(d/'execution-complete.json')['passed'] and read(d/'final-pins.json')['passed']
progress=read(d/'progress.json');assert len(progress)==12
names=['warmup-A','warmup-B']+[f'pair-{i}-{r}' for i,o in enumerate(plan['orders'],1) for r in o]
assert [r['name'] for r in progress]==names
raw={}
for i,name in enumerate(names):
 path=d/name;s=read(path/'status.json');t=read(path/'engine-run.json');parsed=parse_engine_run(path/'engine-run.json',context=32768,gen_tokens=128)
 assert s['exit_code']==0 and s['warmup']==(i<2) and s['peak_rss_bytes']>0
 assert s['executable_sha256']==(plan['baseline_executable_sha256'] if s['role']=='A' else build['executable_sha256'])
 assert t['selection']==dict(prefill_token=85166,final_token=14,selected_tokens_checksum=11977321378026793466,oracle_transcript_match=True)
 for n in ['prefill_layer_profile','long_prefill_profile','stage_profile']:assert t[n] is None,n
 assert parsed==read(path/'parsed-metrics.json') and progress[i]==dict(**s,metrics=parsed)
 assert s['process_wall_seconds']>(parsed['prefill_ms']+parsed['gen_ms'])/1000
 raw[name]=dict(**parsed,process_wall_seconds=s['process_wall_seconds'],peak_rss_bytes=s['peak_rss_bytes'])
metrics=['prefill_ms','prefill_tps','gen_steady_tps','gen_tps','gen_first_ms','process_wall_seconds','peak_rss_bytes']
def stats(v):
 m=med(v);return dict(count=len(v),median=m,mad=med(abs(x-m) for x in v),minimum=min(v),maximum=max(v),values=v)
pairs=[]
for i,order in enumerate(plan['orders'],1):
 a,b=raw[f'pair-{i}-A'],raw[f'pair-{i}-B'];ratios={m:b[m]/a[m] for m in metrics}
 pairs.append(dict(pair=i,order=order,baseline=a,candidate=b,candidate_over_baseline=ratios))
summary={m:dict(baseline=stats([r['baseline'][m] for r in pairs]),candidate=stats([r['candidate'][m] for r in pairs]),paired_ratio=stats([r['candidate_over_baseline'][m] for r in pairs]),orders={o:stats([r['candidate_over_baseline'][m] for r in pairs if r['order']==o]) for o in ['AB','BA']}) for m in metrics}
rules=plan['retention_criteria']
gates=dict(all_exact_transcripts=True,all_five_prefill_faster=all(r['candidate_over_baseline']['prefill_ms']<1 for r in pairs),both_order_prefill_medians=all(v['median']<rules['separate_ab_ba_median_prefill_ratio_below'] for v in summary['prefill_ms']['orders'].values()),steady_decode_median=summary['gen_steady_tps']['paired_ratio']['median']>=rules['median_steady_decode_tps_ratio_at_least'],steady_decode_both_orders=all(v['median']>=rules['separate_order_steady_decode_tps_ratio_at_least'] for v in summary['gen_steady_tps']['orders'].values()),peak_rss_median=summary['peak_rss_bytes']['paired_ratio']['median']<=rules['median_peak_rss_ratio_at_most'])
result=dict(schema='rust-star-attention-output-whole-path-audit-v1',execution_passed=True,raw_runs=12,timed_pairs=5,warmups=2,full_c0_required_and_passed=read(w/'integration/validation-acceptance.json')['passed'],gates=gates,candidate_retained_as_engine_improvement=all(gates.values()),summary=summary,pairs=pairs,warmup_observations={n:raw[n] for n in names[:2]},comparison='Rust candidate versus frozen Rust 9d35953f; not DwarfStar',scope='32K prefill plus 128-token decode on target M1 Ultra',plan_sha256=sha(w/'engine-plan.json'),raw_files={str(p.relative_to(w)):sha(p) for p in sorted(d.glob('*/*')) if p.is_file()})
p=d/'audit.json'
if p.exists():assert read(p)==result,'Audit changed'
else:p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(gates=gates,retained=result['candidate_retained_as_engine_improvement'],prefill=summary['prefill_ms'],steady=summary['gen_steady_tps'],peak_rss=summary['peak_rss_bytes']),indent=2))
