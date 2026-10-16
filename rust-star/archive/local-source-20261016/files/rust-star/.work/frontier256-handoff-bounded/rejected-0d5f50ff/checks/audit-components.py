from pathlib import Path
from collections import defaultdict
import json, math
w=Path(__file__).resolve().parent
names=['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
for name in ['batch-32k-output','batch-32k-profiled']:
 log=(w/(name+'.log')).read_text()
 assert log.count('all 129280 logits exact, token ')==128
 x=json.loads((w/(name+'.json')).read_text())
 assert x['complete_32k_c0_claim'] is True and x['throughput_claim'] is False
 assert x['checked_decode_steps']==128 and x['checked_logits']==129280
 assert x['logits_checksum']==2996662110144943749
 assert x['decode_logits_checksum']==1716816788919264169
 assert x['decode_tokens_checksum']==11977321378026793466 and x['selected_token']==85166
p=json.loads((w/'components.json').read_text());assert p['paired_ineligible'] is True
prefix=[r for r in p['rows'] if r['chunk_start']==0]
initial=[r for r in prefix if r['stage']=='initial_bootstrap']
assert len(initial)==1 and initial[0]['first_layer']==0 and initial[0]['last_layer']==2
rebuilt=[r for r in prefix if r['stage']=='prefix_layer']
assert len(rebuilt)==40 and [r['first_layer'] for r in rebuilt]==list(range(3,43))
assert all(r['first_layer']==r['last_layer'] for r in rebuilt)
c=p['tail_components'];assert c['profile_perturbed'] is True and c['encoder_splitting'] is True and c['encoder_serialization']=='MTLFence'
parents={(r['chunk_start'],r['first_layer']):r for r in p['rows'] if r['stage']=='tail'}
expected={(chunk,layer) for chunk in range(4096,32768,4096) for layer in range(2,43)}
assert set(parents)==expected
rows=defaultdict(list)
for row in c['rows']:rows[row['chunk_start'],row['layer']].append(row)
assert set(rows)==expected and len(c['rows'])==2583
summary=defaultdict(lambda:{'calls':0,'attributed_gpu_ms':0})
for key,group in rows.items():
 assert [r['component'] for r in group]==names
 last=None
 for row in group:
  a,b=row['start_tick'],row['end_tick'];assert isinstance(a,int) and isinstance(b,int) and 0<=a<b<2**64-1
  assert last is None or a>last
  last=b
  assert math.isfinite(row['attributed_gpu_ms']) and row['attributed_gpu_ms']>0
 total=sum(r['end_tick']-r['start_tick'] for r in group)
 parent=parents[key];assert parent['calls']==1
 assert abs(sum(r['attributed_gpu_ms'] for r in group)-parent['gpu_ms'])<1e-6
 for row in group:
  expected_ms=parent['gpu_ms']*(row['end_tick']-row['start_tick'])/total
  assert abs(row['attributed_gpu_ms']-expected_ms)<1e-6
  k=('even' if key[1]%2==0 else 'odd')+'/'+row['component'];summary[k]['calls']+=1;summary[k]['attributed_gpu_ms']+=row['attributed_gpu_ms']
result={'passed':True,'native_tail_calls':len(rows),'component_rows':len(c['rows']),'normal_and_profiled_full_output_c0':True,'paired_eligible':False,'summary':dict(sorted(summary.items(),key=lambda kv:-kv[1]['attributed_gpu_ms']))}
(w/'component-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
