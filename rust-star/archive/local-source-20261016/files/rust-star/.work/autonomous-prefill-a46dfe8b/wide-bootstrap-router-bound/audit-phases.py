from pathlib import Path
import json,hashlib,statistics,math
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());med=statistics.median
assert read(w/'engine/execution-complete.json')['passed'];plan=read(w/'engine-plan.json');progress=read(w/'engine/progress.json');assert len(progress)==12
manifest=read(w/'artifacts.json')['files'] if (w/'artifacts.json').exists() else None
def sha(p):
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if manifest:
  e=manifest[str(p.relative_to(w))];assert h==e['stored_sha256'];return e['original_sha256']
 return h
def stats(v):
 m=med(v);return dict(count=len(v),median=m,mad=med(abs(x-m) for x in v),minimum=min(v),maximum=max(v),values=v)
keys=['prefill_bootstrap_setup_ms','prefill_tile_wall_ms','prefill_tile_gpu_ms','prefill_transformer_setup_ms','prefill_transformer_wall_ms','prefill_transformer_gpu_ms','prefill_host_overhead_ms','prefill_context_setup_ms','model_residency_host_ms','prefill_unattributed_host_ms'];raw={};files={}
for row in progress:
 p=w/'engine'/row['name']/'engine-run.json';d=read(p);assert all(d[k] is None for k in ['prefill_layer_profile','long_prefill_profile','stage_profile']);t=d['timing'];values={k:t[k] for k in keys};assert all(math.isfinite(v) and v>=0 for v in values.values());values['transformer_command_wall_minus_gpu_ms']=t['prefill_transformer_wall_ms']-t['prefill_transformer_gpu_ms'];assert values['transformer_command_wall_minus_gpu_ms']>=0;raw[row['name']]=values;files[str(p.relative_to(w))]=sha(p)
metrics=list(raw['warmup-A']);pairs=[dict(pair=i,order=o,baseline=raw[f'pair-{i}-A'],candidate=raw[f'pair-{i}-B'],candidate_minus_baseline_ms={k:raw[f'pair-{i}-B'][k]-raw[f'pair-{i}-A'][k] for k in metrics}) for i,o in enumerate(plan['orders'],1)]
r=dict(passed=True,interpretation='Supplemental passive timings already present in every engine run. No new instrumentation, exclusions or retention threshold. Transformer command wall minus GPU is an aggregate timing difference, not proof of a particular stall or allocation cause.',summary={k:dict(baseline=stats([p['baseline'][k] for p in pairs]),candidate=stats([p['candidate'][k] for p in pairs]),paired_delta_ms=stats([p['candidate_minus_baseline_ms'][k] for p in pairs])) for k in metrics},pairs=pairs,warmups={n:raw[n] for n in ['warmup-A','warmup-B']},raw_files=files)
p=w/'passive-phase-audit.json'
if p.exists():assert read(p)==r
else:p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(passed=True,runs=len(progress),paired_phase_deltas_ms={k:v['paired_delta_ms']['median'] for k,v in r['summary'].items()}),indent=2))
