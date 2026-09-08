from pathlib import Path
import json,sys,collections,hashlib
p=Path(sys.argv[1]);r=json.loads(p.read_text());profile=r['long_prefill_profile'];rows=profile['rows']
prefix=[x for x in rows if x['chunk_start']==0]
initial=[x for x in prefix if x['stage']=='initial_bootstrap']
assert len(initial)==1 and initial[0]['first_layer']==0 and initial[0]['last_layer']==2
rebuilt=[x for x in prefix if x['stage']=='prefix_layer']
assert len(rebuilt)==40 and [x['first_layer'] for x in rebuilt]==list(range(3,43))
assert all(x['first_layer']==x['last_layer'] for x in rebuilt)
assert r['context']==32768 and r['gen_tokens']==128
assert not r['paired_protocol_eligible'] and r['selection']['oracle_transcript_match']
assert r['selection']['selected_tokens_checksum']==11977321378026793466
assert {x['chunk_start'] for x in rows}==set(range(0,32768,4096))
for chunk in range(4096,32768,4096):
 covered={layer for x in rows if x['chunk_start']==chunk for layer in range(x['first_layer'],x['last_layer']+1)}
 assert covered==set(range(43)),(chunk,covered)
 for layer in range(2,43):
  stages={x['stage'] for x in rows if x['chunk_start']==chunk and x['first_layer']==x['last_layer']==layer}
  assert {'bind','attention','tail'} <= stages,(chunk,layer,stages)
assert all(x[k]>=0 for x in rows for k in ['host_elapsed_ms','command_wall_ms','gpu_ms'])
for key,components in [('gpu_ms',['prefill_transformer_gpu_ms','prefill_tile_gpu_ms']),('command_wall_ms',['prefill_transformer_wall_ms','prefill_tile_wall_ms'])]:
 expected=sum(r['timing'][k] for k in components);actual=sum(x[key] for x in rows);assert abs(expected-actual)<.01,(key,expected,actual)
groups=collections.defaultdict(lambda:dict(calls=0,host_elapsed_ms=0.,command_wall_ms=0.,gpu_ms=0.))
for row in rows:
 family='prefix' if row['chunk_start']==0 else 'bootstrap' if row['first_layer']!=row['last_layer'] else 'even' if row['first_layer']%2==0 else 'odd'
 g=groups[family+'/'+row['stage']]
 for k in g:g[k]+=row[k]
components=profile['tail_components'];assert components['profile_perturbed'] and components['encoder_splitting']
cr=collections.defaultdict(list)
for x in components['rows']:cr[x['chunk_start'],x['layer']].append(x)
parents={(x['chunk_start'],x['first_layer']):x for x in rows if x['stage']=='tail'}
assert set(cr)==set(parents)=={(c,l) for c in range(4096,32768,4096) for l in range(2,43)}
cn=['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
ct=collections.defaultdict(float)
for key,group in cr.items():
 assert [x['component'] for x in group]==cn
 assert all(0<=x['start_tick']<x['end_tick']<2**64-1 for x in group)
 assert all(b['start_tick']>a['end_tick'] for a,b in zip(group,group[1:]))
 ticks=sum(x['end_tick']-x['start_tick'] for x in group)
 for x in group:
  value=parents[key]['gpu_ms']*(x['end_tick']-x['start_tick'])/ticks
  assert abs(x['attributed_gpu_ms']-value)<1e-6
  ct[x['component']]+=x['attributed_gpu_ms']
 assert abs(sum(x['attributed_gpu_ms'] for x in group)-parents[key]['gpu_ms'])<1e-6
result={'passed':True,'tail_component_rows':len(components['rows']),'tail_component_attributed_gpu_ms':dict(ct),'profile_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(rows),'coverage':'all 43 layers in each continuation chunk, shared bootstrap ranges explicit','group_totals':dict(groups),'prefill_ms':r['metrics']['prefill_ms']}
p.with_name(p.stem+'-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name,g in sorted(groups.items(),key=lambda x:x[1]['gpu_ms'],reverse=True):print(f"{name:28} GPU {g['gpu_ms']/1000:8.3f}s command {g['command_wall_ms']/1000:8.3f}s host-total {g['host_elapsed_ms']/1000:8.3f}s calls {g['calls']}")
print('PASS profile coverage and interval accounting',len(rows),'rows')
