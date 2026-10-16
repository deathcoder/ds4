from pathlib import Path
import json,hashlib,statistics,sys,math,datetime
w=Path(__file__).resolve().parent;sys.path.insert(0,str(w/'adapters'))
from ruststar_measurement_lib import parse_engine_run
from dwarfstar_measurement_lib import parse_dwarfstar_csv
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert 'Ran 18 tests' in (w/'adapter-tests.log').read_text() and (w/'adapter-tests.log').read_text().strip().endswith('OK')
p=read(w/'plan.json');freeze=read(w/'freeze.json');assert all(sha(w/n)==h for n,h in freeze['files'].items())
assert p['orders']==['AB','BA','AB','BA','AB'] and p['context']==32768 and p['gen_tokens']==128
assert p['executable_sha256']['B']==read(w/'final-runtime-state.json')['default_executable_sha256']
assert p['executable_sha256']['A']==read(w/'oracle-v5-benchmark-build.json')['executable_sha256']
for name in ['input-before.json','input-after.json']:
 x=read(w/name);assert x['passed'] and x['sources']==read(w/'final-runtime-state.json')['sources'] and x['executables']==p['executable_sha256'] and x['freeze_sha256']==sha(w/'freeze.json')
 old=read(w/'prior-plan.json')['source_reference'];assert x['inputs']==old['inputs'] and x['fixtures']==old['fixtures'] and len(x['fixtures'])==7669
queue=[('warmup-A','A',None),('warmup-B','B',None)]
for pair,order in enumerate(p['orders'],1):
 for role in order:queue.append((f'pair-{pair}-{role}',role,pair))
rows=[];previous=None;host_count=0;thermal_unusual=[]
for name,role,pair in queue:
 run=w/'engine'/name;mroot=run/'measurement';s=read(run/'status.json');m=read(mroot/'measurement.json')
 assert (s['name'],s['role'],s['pair'],s['warmup'])==(name,role,pair,pair is None)
 assert s['exit_code']==0 and s['executable_sha256']==p['executable_sha256'][role]
 assert previous is None or s['started_utc']>=previous
 previous=s['finished_utc'];assert previous>s['started_utc']
 pre=read(run/'preflight.json');assert pre['passed'] and pre['executables']==p['executable_sha256'] and pre['sources']==read(w/'final-runtime-state.json')['sources'] and pre['freeze_sha256']==sha(w/'freeze.json')
 assert m['status']=='passed' and m['returncode']==0 and not m['timed_out'] and m['context']==32768 and m['gen_tokens']==128
 assert m['engine']==('dwarfstar' if role=='A' else 'rust-star')
 for a in m['artifacts'].values():
  ap=Path(a['path']);assert not ap.is_absolute() and '..' not in ap.parts
  f=mroot/ap;assert f.stat().st_size==a['bytes'] and sha(f)==a['sha256']
 raw=mroot/('benchmark.csv' if role=='A' else 'engine-run.json')
 metrics=(parse_dwarfstar_csv if role=='A' else parse_engine_run)(raw,context=32768,gen_tokens=128)
 for k in ['process_peak_bytes','process_wall_ms']:
  assert math.isfinite(m[k]) and m[k]>0;metrics[k]=m[k]
 residual=m['process_wall_ms']-metrics['prefill_ms']-metrics['gen_ms'];assert residual>=-max(5,m['process_wall_ms']*.01)
 metrics['process_overhead_ms']=max(0.,residual);assert metrics==m['metrics']
 if role=='B':
  engine=read(raw);assert engine['selection']['selected_tokens_checksum']==11977321378026793466
  assert not engine['timing'].get('prefill_stage_profile_collection',False)
  assert all(engine[n] is None for n in ['prefill_layer_profile','long_prefill_profile','stage_profile'])
 hosts=[json.loads(l) for l in (run/'host.jsonl').read_text().splitlines()];assert hosts[0]['phase']=='before' and hosts[-1]['phase']=='after';host_count+=len(hosts)
 for h in hosts:
  assert all(h[k+'_exit']==0 for k in ['thermal','swap','vm_stat'])
  if 'No thermal warning level' not in h['thermal'] or 'No performance warning level' not in h['thermal']:thermal_unusual.append(dict(run=name,record=h))
 rows.append(dict(**s,metrics=metrics))
assert read(w/'engine/complete.json')['passed'] and read(w/'engine/complete.json')['runs']==12
assert rows==read(w/'engine/progress.json')
keys=['prefill_ms','prefill_tps','gen_steady_tps','gen_tps','gen_first_ms','process_wall_ms','process_peak_bytes']
def stats(v):
 med=statistics.median(v);return dict(median=med,mad=statistics.median(abs(x-med) for x in v),minimum=min(v),maximum=max(v),n=len(v))
timed=[r for r in rows if not r['warmup']];engines={role:{k:stats([r['metrics'][k] for r in timed if r['role']==role]) for k in keys} for role in ['A','B']}
pairs=[]
for i,order in enumerate(p['orders'],1):
 by={r['role']:r['metrics'] for r in timed if r['pair']==i};assert set(by)=={'A','B'}
 pairs.append(dict(pair=i,order=order,rust_over_dwarfstar={k:by['B'][k]/by['A'][k] for k in keys}))
ratios={k:stats([r['rust_over_dwarfstar'][k] for r in pairs]) for k in keys}
orders={order:{k:stats([r['rust_over_dwarfstar'][k] for r in pairs if r['order']==order]) for k in keys} for order in ['AB','BA']}
result=dict(passed=True,runs=12,timed_pairs=5,engines=engines,pairs=pairs,paired_ratios=ratios,order_subsets=orders,ratio_of_medians={k:engines['B'][k]['median']/engines['A'][k]['median'] for k in keys},host_observations=host_count,unusual_thermal_observations=thermal_unusual,correctness_basis=p['correctness'],runtime_changed=False,scope='32K/128 only; five pairs do not establish statistical significance')
(w/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,runs=12,timed_pairs=5,paired_ratio_medians={k:v['median'] for k,v in ratios.items()},host_observations=host_count,unusual_thermal_count=len(thermal_unusual)),indent=2))
