from pathlib import Path
from collections import Counter
import json
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());instrumentation=read(w/'instrumentation.json');result={}
for role,width in [('accepted',64),('wide',1024)]:
 rows=[json.loads(line) for line in (w/(role+'-trace.jsonl')).read_text().splitlines()];by={(x['position'],x['phase']):x for x in rows};assert len(by)==len(rows)
 positions=list(range(4096,32768,width));phases=['entry','registered','submitted','after_pool'];keys={(p,phase) for p in positions for phase in phases}|{(p,'waited') for p in positions if (p+width)%4096==0};assert set(by)==keys,(role,len(by),len(keys),list(set(by)^keys)[:5])
 prior_count=0;prior_bytes=0;prior_dead=0
 for x in rows:
  assert x['rows']==width and x['frontier']==(x['position']//4096+1)*4096
  assert x['vm_result']==0 and x['phys_footprint_bytes']>0
  assert x['registered_count']>=prior_count and x['registered_bytes']>=prior_bytes and x['dead_count']>=prior_dead
  prior_count=x['registered_count'];prior_bytes=x['registered_bytes'];prior_dead=x['dead_count']
  labels=x['labels'];assert set(labels).issubset(instrumentation[role]['names'])
  assert all(q['persistent']==(n in instrumentation[role]['persistent']) for n,q in labels.items())
  assert x['live_count']==sum(q['count'] for q in labels.values())==x['registered_count']-x['dead_count']
  assert x['scratch_bytes']==sum(q['bytes'] for q in labels.values() if not q['persistent'])
  assert x['persistent_bytes']==sum(q['bytes'] for q in labels.values() if q['persistent'])
  assert 0<=x['previous_scratch_bytes']<=x['scratch_bytes']
  assert x['commands']==sum(x['command_status_counts']);assert x['command_status_counts'][5:]==[0,0]
  if x['phase']=='submitted':assert x['commands']==x['position']%4096//width+1
  if x['phase']=='waited':assert x['command_status_counts'][4]==4096//width and x['commands']==4096//width
  if x['phase']=='after_pool' and (x['position']+width)%4096==0:assert x['commands']==0
 def peak(field):
  x=max(rows,key=lambda x:x[field]);return dict(value=x[field],position=x['position'],phase=x['phase'])
 chunks=[]
 for frontier in range(8192,32769,4096):
  block=[x for x in rows if x['frontier']==frontier];reg=[x for x in block if x['phase']=='registered'];end=by[(frontier-width,'after_pool')]
  chunks.append(dict(frontier=frontier,max_scratch_bytes=max(x['scratch_bytes'] for x in block),max_device_allocated_bytes=max(x['device_allocated_bytes'] for x in block),max_phys_footprint_bytes=max(x['phys_footprint_bytes'] for x in block),max_incomplete_commands=max(sum(x['command_status_counts'][:4]) for x in block),max_per_label_instances=max(q['count'] for x in block for q in x['labels'].values()),registered_previous_scratch=[x['previous_scratch_bytes'] for x in reg],end_scratch_bytes=end['scratch_bytes'],end_persistent_bytes=end['persistent_bytes']))
 largest=max(rows,key=lambda x:x['scratch_bytes']);top=sorted([(n,q['bytes'],q['count']) for n,q in largest['labels'].items() if not q['persistent']],key=lambda q:-q[1])[:15]
 result[role]=dict(width=width,tiles=len(positions),snapshots=len(rows),peaks={f:peak(f) for f in ['scratch_bytes','persistent_bytes','previous_scratch_bytes','device_allocated_bytes','phys_footprint_bytes']},registered_count=prior_count,registered_bytes=prior_bytes,dead_count=prior_dead,chunks=chunks,largest_scratch_labels=top)
result['passed']=True;result['scope']='Perturbed object-lifetime observations only; no performance or physical-memory causal conclusion.'
with (w/'audit.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({role:{'tiles':result[role]['tiles'],'peaks':result[role]['peaks'],'end_scratch':[x['end_scratch_bytes'] for x in result[role]['chunks']]} for role in ['accepted','wide']},indent=2))
