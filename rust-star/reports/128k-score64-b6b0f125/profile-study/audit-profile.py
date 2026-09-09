from pathlib import Path
from collections import defaultdict
import json,math,sys,statistics
w=Path(__file__).resolve().parent
p=Path(sys.argv[1]) if len(sys.argv)>1 else w/'baseline-profile.json'
x=json.loads(p.read_text())
engine_profile='context' in x
if engine_profile:
 assert x['context']==131072 and x['gen_tokens']==128 and not x['paired_protocol_eligible']
 assert x['selection']==dict(prefill_token=1110,final_token=2945,selected_tokens_checksum=1079188149413743454,oracle_transcript_match=True)
 t=x['timing'];assert t['generation_layer_timing_collection'] and t['generation_stage_counter_collection'] and not t['generation_correctness_collection']
 profile=x['long_prefill_profile']
else:
 profile=x
 c0=json.loads(p.with_name('diagnostic-c0.json').read_text())
 assert c0['external_reference_match'] and c0['frontier_tokens']==131072 and c0['checked_logits']==129280 and c0['checked_decode_steps']==128
assert profile['paired_ineligible'] is True
components=profile['tail_components'];assert components['profile_perturbed'] and components['encoder_splitting'] and components['encoder_serialization']=='MTLFence'
names=['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']
expected={(chunk,layer) for chunk in range(4096,131072,4096) for layer in range(2,43)}
parents={(r['chunk_start'],r['first_layer']):r for r in profile['rows'] if r['stage']=='tail'};assert set(parents)==expected and sum(r['stage']=='tail' for r in profile['rows'])==1271
initial=[r for r in profile['rows'] if r['stage']=='initial_bootstrap'];assert len(initial)==1 and initial[0]['first_layer']==0 and initial[0]['last_layer']==2
prefix=[r for r in profile['rows'] if r['stage']=='prefix_layer'];assert len(prefix)==40 and [r['first_layer'] for r in prefix]==list(range(3,43))
groups=defaultdict(list)
for row in components['rows']:groups[row['chunk_start'],row['layer']].append(row)
assert set(groups)==expected and len(components['rows'])==11439
summary=defaultdict(float);parity=defaultdict(float);coverage=[];shares=defaultdict(list)
for key,rows in groups.items():
 assert [r['component'] for r in rows]==names;last=-1
 for r in rows:
  a,b=r['start_tick'],r['end_tick'];assert isinstance(a,int) and isinstance(b,int) and last<a<b<2**64-1;last=b
  assert math.isfinite(r['attributed_gpu_ms']) and r['attributed_gpu_ms']>0
 total=sum(r['end_tick']-r['start_tick'] for r in rows);parent=parents[key];assert parent['calls']==1
 coverage.append(total/(rows[-1]['end_tick']-rows[0]['start_tick']))
 assert 0<coverage[-1]<=1
 assert abs(sum(r['attributed_gpu_ms'] for r in rows)-parent['gpu_ms'])<1e-6
 for r in rows:
  assert abs(parent['gpu_ms']*(r['end_tick']-r['start_tick'])/total-r['attributed_gpu_ms'])<1e-6
  shares[r['component']].append(r['attributed_gpu_ms']/parent['gpu_ms'])
  summary[r['component']]+=r['attributed_gpu_ms'];parity[('even' if key[1]%2==0 else 'odd')+'/'+r['component']]+=r['attributed_gpu_ms']
stages=defaultdict(lambda:dict(calls=0,gpu_ms=0.,command_wall_ms=0.,host_elapsed_ms=0.))
for row in profile['rows']:
 assert row['calls']>0
 for k in ['gpu_ms','command_wall_ms','host_elapsed_ms']:assert math.isfinite(row[k]) and row[k]>=0;stages[row['stage']][k]+=row[k]
 stages[row['stage']]['calls']+=row['calls']
d={'stages':[]}
if engine_profile:
 d=x['stage_profile'];assert d['profile_perturbed'] and d['encoder_splitting'];assert len(d['stages'])==8
 for role in ['first','steady']:
  for row in d['stages']:assert math.isfinite(row[role+'_gpu_ms']) and row[role+'_gpu_ms']>0
  assert abs(sum(row[role+'_gpu_ms'] for row in d['stages'])-t['gen_'+role+'_transformer_gpu_ms'])<1e-5
  layers=t['gen_'+role+'_layer_gpu_ms'];assert len(layers)==43 and all(math.isfinite(v) and v>0 for v in layers)
  assert abs(sum(layers)-t['gen_'+role+'_transformer_gpu_ms'])<1e-5
by_parity=defaultdict(lambda:dict(calls=0,gpu_ms=0.,command_wall_ms=0.,host_elapsed_ms=0.))
by_chunk=defaultdict(lambda:dict(gpu_ms=0.,command_wall_ms=0.,host_elapsed_ms=0.))
for row in profile['rows']:
 role=('even' if row['first_layer']%2==0 else 'odd') if row['first_layer']==row['last_layer'] else 'multi'
 key=role+'/'+row['stage'];by_parity[key]['calls']+=row['calls']
 for metric in ['gpu_ms','command_wall_ms','host_elapsed_ms']:
  by_parity[key][metric]+=row[metric]
  by_chunk[str(row['chunk_start'])+'/'+row['stage']][metric]+=row[metric]
distribution=lambda values:dict(minimum=min(values),median=statistics.median(values),maximum=max(values))
result=dict(counter_span_coverage=distribution(coverage),tail_component_shares={n:distribution(v) for n,v in shares.items()},passed=True,context=131072,transcript_exact=True,native_tails=len(groups),component_intervals=len(components['rows']),paired_eligible=False,profile_perturbed=True,stages=dict(sorted(stages.items(),key=lambda v:-v[1]['gpu_ms'])),stages_by_parity=dict(sorted(by_parity.items(),key=lambda v:-v[1]['gpu_ms'])),stages_by_chunk=dict(by_chunk),tail_components_gpu_ms=dict(sorted(summary.items(),key=lambda v:-v[1])),tail_components_by_parity_gpu_ms=dict(sorted(parity.items(),key=lambda v:-v[1])),decode_stages=d['stages'])
out=p.with_name(p.stem+'-audit.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='stages_by_chunk'},indent=2))
