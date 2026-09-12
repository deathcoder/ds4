"""Describe every lifetime violation without changing the frozen acceptance audit."""
from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());plan=read(w/'plan.json');inst=read(w/'instrumentation.json')['candidate'];log=(w/'candidate-lifetime.log').read_text();xs=[json.loads(l[12:]) for l in log.splitlines() if l.startswith('RS_LIFETIME ')];by={(x['position'],x['phase']):x for x in xs};keys={(p,phase) for p in range(4096,32768,1024) for phase in ['entry','registered','submitted','after_pool']}|{(p,'waited') for p in range(7168,32768,4096)};assert len(xs)==119 and set(by)==keys
sizes={q['frontier']:q['one_set_bytes'] for q in plan['lifetime_reference']};violations=[];last_count=last_bytes=last_dead=0
for i,x in enumerate(xs):
 assert x['rows']==1024 and x['frontier']==(x['position']//4096+1)*4096 and x['vm_result']==0
 assert x['registered_count']>=last_count and x['registered_bytes']>=last_bytes and x['dead_count']>=last_dead
 last_count=x['registered_count'];last_bytes=x['registered_bytes'];last_dead=x['dead_count']
 assert x['live_count']==sum(q['count'] for q in x['labels'].values())==x['registered_count']-x['dead_count']
 assert x['scratch_bytes']==sum(q['bytes'] for q in x['labels'].values() if not q['persistent'])
 assert x['persistent_bytes']==sum(q['bytes'] for q in x['labels'].values() if q['persistent'])
 assert 0<=x['previous_scratch_bytes']<=x['scratch_bytes']
 assert set(x['labels']).issubset(inst['names']) and all(q['persistent']==(n in inst['persistent']) for n,q in x['labels'].items())
 assert x['commands']==sum(x['command_status_counts']) and x['command_status_counts'][5:]==[0,0]
 if x['phase']=='registered':assert {n for n,q in x['labels'].items() if not q['persistent']}==set(inst['scratch'])
 if x['phase']=='submitted':assert x['commands']==x['position']%4096//1024+1
 if x['phase']=='waited':assert x['commands']==4 and x['command_status_counts'][4]==4
 if x['phase']=='after_pool' and (x['position']+1024)%4096==0:assert x['commands']==0
 rules=dict(at_most_one_unfinished=sum(x['command_status_counts'][:4])<=1,previous_scratch_zero=x['previous_scratch_bytes']==0,single_instance_per_scratch=all(q['count']==1 for q in x['labels'].values() if not q['persistent']),entry_and_pool_scratch_zero=x['scratch_bytes']==0 if x['phase'] in ['entry','after_pool'] else True,registered_exactly_one_set=x['scratch_bytes']==sizes[x['frontier']] if x['phase']=='registered' else True)
 failed=[n for n,v in rules.items() if not v]
 if failed:violations.append(dict(snapshot_index=i,failed=failed,observation=x))
r=dict(structure_passed=True,criteria_pass=False,full_c0_passed=read(w/'lifetime-validation-acceptance.json')['passed'],full_fp32_words=16677120,tiles=28,snapshots=len(xs),violations=violations,max_unfinished_commands=max(sum(x['command_status_counts'][:4]) for x in xs),peak_scratch_bytes=max(x['scratch_bytes'] for x in xs),max_instances=max(q['count'] for x in xs for q in x['labels'].values() if not q['persistent']),first_zero_after_failure=next(dict(snapshot_index=i,position=x['position'],phase=x['phase']) for i,x in enumerate(xs) if i>violations[0]['snapshot_index'] and x['scratch_bytes']==0),interpretation='Completed command status did not coincide with disappearance of all weakly observed buffer objects. Ownership and exact release times are not established; snapshots are non-atomic. Do not relabel this as passing or run engine timing.',log_sha256=hashlib.sha256(log.encode()).hexdigest())
assert violations and len(violations)==4
with (w/'diagnostic-failure-analysis.json').open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
with (w/'candidate-lifetime-trace.jsonl').open('x') as f:
 for x in xs:f.write(json.dumps(x,sort_keys=True)+'\n')
print(json.dumps(dict(structure_passed=True,criteria_pass=False,violations=[dict(index=v['snapshot_index'],position=v['observation']['position'],phase=v['observation']['phase'],failed=v['failed']) for v in violations],max_unfinished_commands=r['max_unfinished_commands'],peak_scratch_bytes=r['peak_scratch_bytes']),indent=2))
