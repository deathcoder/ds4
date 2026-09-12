from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());plan=read(w/'plan.json');inst=read(w/'instrumentation.json')['candidate'];log=(w/'candidate-lifetime.log').read_text();assert read(w/'lifetime-validation-acceptance.json')['passed'];assert log.count('all 129280 logits exact, token ')==128 and 'RS_LIFETIME_ERROR' not in log
rows=[json.loads(l[len('RS_LIFETIME '):]) for l in log.splitlines() if l.startswith('RS_LIFETIME ')];by={(x['position'],x['phase']):x for x in rows};keys={(p,phase) for p in range(4096,32768,1024) for phase in ['entry','registered','submitted','after_pool']}|{(p,'waited') for p in range(7168,32768,4096)};assert len(rows)==119 and set(by)==keys
sizes={q['frontier']:q['one_set_bytes'] for q in plan['lifetime_reference']};last_count=last_bytes=last_dead=0
for x in rows:
 assert x['rows']==1024 and x['frontier']==(x['position']//4096+1)*4096 and x['vm_result']==0
 assert x['registered_count']>=last_count and x['registered_bytes']>=last_bytes and x['dead_count']>=last_dead
 last_count=x['registered_count'];last_bytes=x['registered_bytes'];last_dead=x['dead_count']
 assert x['live_count']==sum(q['count'] for q in x['labels'].values())==x['registered_count']-x['dead_count']
 assert x['scratch_bytes']==sum(q['bytes'] for q in x['labels'].values() if not q['persistent'])
 assert x['persistent_bytes']==sum(q['bytes'] for q in x['labels'].values() if q['persistent'])
 assert set(x['labels']).issubset(inst['names']);assert all(q['persistent']==(n in inst['persistent']) for n,q in x['labels'].items())
 assert x['commands']==sum(x['command_status_counts']) and x['command_status_counts'][5:]==[0,0]
 assert sum(x['command_status_counts'][:4])<=1
 assert x['previous_scratch_bytes']==0
 assert all(q['count']==1 for q in x['labels'].values() if not q['persistent'])
 if x['phase']=='registered':assert x['scratch_bytes']==sizes[x['frontier']] and {n for n,q in x['labels'].items() if not q['persistent']}==set(inst['scratch'])
 if x['phase'] in ['entry','after_pool']:assert x['scratch_bytes']==0
 if x['phase']=='submitted':assert x['commands']==x['position']%4096//1024+1
 if x['phase']=='waited':assert x['commands']==4 and x['command_status_counts'][4]==4
 if x['phase']=='after_pool' and (x['position']+1024)%4096==0:assert x['commands']==0
result=dict(passed=True,criteria_pass=True,tiles=28,snapshots=len(rows),full_fp32_words=16677120,previous_tile_scratch_zero=True,all_tile_pools_drain_scratch=True,max_unfinished_commands=max(sum(x['command_status_counts'][:4]) for x in rows),max_scratch_bytes=max(x['scratch_bytes'] for x in rows),peak_sampled_device_allocated_bytes=max(x['device_allocated_bytes'] for x in rows),peak_sampled_phys_footprint_bytes=max(x['phys_footprint_bytes'] for x in rows),prior_wide_peak_scratch_bytes=9972692932,scope='Mechanism and full-C0 proof only. No engine performance or physical footprint acceptance claim.',log_sha256=hashlib.sha256(log.encode()).hexdigest())
with (w/'diagnostic-selection.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
with (w/'candidate-lifetime-trace.jsonl').open('x') as f:
 for x in rows:f.write(json.dumps(x,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
