from pathlib import Path
import json,sys,collections,hashlib
p=Path(sys.argv[1]);r=json.loads(p.read_text());profile=r['long_prefill_profile'];rows=profile['rows']
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
result={'passed':True,'profile_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(rows),'coverage':'all 43 layers in each continuation chunk, shared bootstrap ranges explicit','group_totals':dict(groups),'prefill_ms':r['metrics']['prefill_ms']}
p.with_name(p.stem+'-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name,g in sorted(groups.items(),key=lambda x:x[1]['gpu_ms'],reverse=True):print(f"{name:28} GPU {g['gpu_ms']/1000:8.3f}s command {g['command_wall_ms']/1000:8.3f}s host-total {g['host_elapsed_ms']/1000:8.3f}s calls {g['calls']}")
print('PASS profile coverage and interval accounting',len(rows),'rows')
