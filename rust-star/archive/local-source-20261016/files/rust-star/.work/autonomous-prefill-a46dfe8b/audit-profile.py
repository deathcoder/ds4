from pathlib import Path
from collections import defaultdict
import hashlib,json,math,statistics
w=Path(__file__).resolve().parent
read=lambda n:json.loads((w/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def audit_profile(name):
 plan=read('plan.json')
 x=read(name+'.json')
 assert x==dict(schema='rust-star-long-prefill-chunk-probe-v1',frontier_tokens=32768,internal_hc_collection=False,chunks=[],checked_decode_steps=128,decode_logits_checksum=1716816788919264169,decode_tokens_checksum=11977321378026793466,checked_logits=129280,logits_checksum=2996662110144943749,selected_token=85166,complete_32k_c0_claim=True,throughput_claim=False)
 assert (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
 accept=read('validation-acceptance.json');assert accept['passed'] and all(x['exit_code']==0 for x in accept['runs']) and len(accept['runs'])==1
 p=read(name+'-components.json');assert p['paired_ineligible']
 expected={(c,l) for c in range(4096,32768,4096) for l in range(2,43)}
 stages=defaultdict(lambda:dict(calls=0,gpu_ms=0.,host_elapsed_ms=0.,command_wall_ms=0.))
 parents={name:{} for name in ['attention','tail']}
 for r in p['rows']:
  for metric in ['gpu_ms','host_elapsed_ms','command_wall_ms']:
   assert math.isfinite(r[metric]) and r[metric]>=0
   stages[r['stage']][metric]+=r[metric]
  stages[r['stage']]['calls']+=r['calls']
  if r['stage'] in parents:
   key=r['chunk_start'],r['first_layer'];assert key not in parents[r['stage']]
   assert r['calls']==1 and r['first_layer']==r['last_layer']
   parents[r['stage']][key]=r
 assert all(set(v)==expected for v in parents.values())
 tailnames=['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
 profiles={}
 for kind,total in [('tail',2583),('attention',1883)]:
  t=p[kind+'_components']; assert t['profile_perturbed'] and t['encoder_splitting'] and t['encoder_serialization']=='MTLFence'
  if kind=='attention':assert t['schema']=='rust-star-attention-components-v1'
  groups=defaultdict(list)
  for row in t['rows']:groups[row['chunk_start'],row['layer']].append(row)
  assert len(t['rows'])==total and set(groups)==expected
  components=defaultdict(float);path_totals=defaultdict(float);coverage=[];packets=[]
  for key,g in sorted(groups.items()):
   path='tail' if kind=='tail' else ('indexed' if key[1]%2==0 else 'compressed')
   names=tailnames if kind=='tail' else plan[path+'_components']
   assert [r['component'] for r in g]==names
   if kind=='attention':assert all(r['path']==path for r in g)
   last=-1; ticks=sum(r['end_tick']-r['start_tick'] for r in g);gpu=parents[kind][key]['gpu_ms'];assert gpu>0 and ticks>0
   for r in g:
    assert isinstance(r['start_tick'],int) and isinstance(r['end_tick'],int)
    assert last<r['start_tick']<r['end_tick']<2**64-1
    assert math.isfinite(r['attributed_gpu_ms']) and r['attributed_gpu_ms']>0
    assert abs(gpu*(r['end_tick']-r['start_tick'])/ticks-r['attributed_gpu_ms'])<1e-6
    components[path+'/'+r['component']]+=r['attributed_gpu_ms'];last=r['end_tick']
   assert abs(sum(r['attributed_gpu_ms'] for r in g)-gpu)<1e-6
   path_totals[path]+=gpu
   coverage.append(ticks/(g[-1]['end_tick']-g[0]['start_tick']))
   packets.append(dict(chunk_start=key[0],layer=key[1],path=path,gpu_ms=gpu,components={r['component']:r['attributed_gpu_ms'] for r in g}))
  assert all(0<v<=1 for v in coverage)
  profiles[kind]=dict(packets=len(groups),intervals=total,path_totals_ms=dict(path_totals),component_totals_ms=dict(components),counter_coverage=dict(minimum=min(coverage),median=statistics.median(coverage),maximum=max(coverage)),packet_details=packets)
 prefix=[r for r in p['rows'] if r['stage']=='initial_bootstrap'];assert len(prefix)==1 and (prefix[0]['first_layer'],prefix[0]['last_layer'])==(0,2)
 prefix=[r for r in p['rows'] if r['stage']=='prefix_layer'];assert len(prefix)==40 and [r['first_layer'] for r in prefix]==list(range(3,43))
 assert sum(r['path']=='indexed' for r in profiles['attention']['packet_details'])==147
 assert sum(r['path']=='compressed' for r in profiles['attention']['packet_details'])==140
 ranked=sorted(profiles['attention']['component_totals_ms'].items(),key=lambda kv:kv[1],reverse=True)
 stage_total=sum(v['gpu_ms'] for v in stages.values())
 r=dict(passed=True,complete_32k_c0=True,fp32_words_checked=16677120,stages=dict(stages),profiles=profiles,ranked_attention_components=[dict(component=k,gpu_ms=v,attention_share=v/stages['attention']['gpu_ms'],recorded_stage_share=v/stage_total) for k,v in ranked],profile_sha256=sha(w/(name+'-components.json')),log_sha256=sha(w/(name+'.log')),output_sha256=sha(w/(name+'.json')),engine_speedup_claim=False)
 return r

r=audit_profile('current-32k-profiled')
r['ranked_tail_components']=sorted(r['profiles']['tail']['component_totals_ms'].items(),key=lambda kv:kv[1],reverse=True)
p=w/'profile-audit.json'
if p.exists():assert read(p.name)==r
else:p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(passed=True,stages=r['stages'],tail=r['ranked_tail_components'],attention=r['ranked_attention_components']),indent=2))
