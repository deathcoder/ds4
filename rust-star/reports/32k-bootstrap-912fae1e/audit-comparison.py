"""Independent audit of the Rust-to-Rust study; accepts a work/report directory."""
from pathlib import Path
import json,hashlib,statistics,sys,math
p=Path(sys.argv[1]).resolve();r=p/'results'
def read(x):return json.loads(x.read_text())
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
def dist(xs):
 m=statistics.median(xs)
 return dict(count=len(xs),median=m,mad=statistics.median(abs(x-m) for x in xs),min=min(xs),max=max(xs))
raw=read(r/'paired-raw.json');summary=read(r/'paired-summary.json');state=read(r/'state.json');freeze=read(p/'inputs/freeze.json');plan=read(p/'inputs/plan.json')
assert sha(p/'inputs/plan.json')==freeze['plan_sha256']
assert state['phase']=='complete' and len(state['attempts'])==5
pairs=raw['pairs'];assert len(pairs)==5 and all(x['valid'] and x['attempt']==1 for x in pairs)
assert [x['order'] for x in pairs]==['AB','BA','AB','BA','AB']
assert [x['repetition'] for x in pairs]==list(range(1,6))
assert all(x['context']==32768 for x in pairs)
for label in ['host','correctness']:
 file=p/'inputs'/('host-before.json' if label=='host' else 'correctness-manifest.json')
 assert sha(file)==raw[label+'_manifest_sha256']==plan[label+'_manifest_sha256']
assert sha(p/'inputs/acceptance-plan.json')==freeze['acceptance_plan_sha256']
assert read(p/'inputs/acceptance-plan.json')==read(p/'inputs/correctness-manifest.json')['optimization_acceptance']
assert 'Rust-to-Rust' in freeze['comparison']
assert freeze['c0_oracle_source']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert freeze['oracle_commit'].startswith('94565861') and freeze['candidate_commit'].startswith('912fae1e')
ctx=summary['contexts'][0];assert ctx['valid_pairs']==5 and ctx['invalid_pairs']==[]
for role in ('candidate','oracle'):
 identity=raw[role];assert identity==plan[role]['identity']==summary[role]
 assert identity['source_commit']==freeze[role+'_commit'] and identity['executable_sha256']==freeze[role+'_sha256']
 assert 'measure_ruststar.py' in ' '.join(plan[role]['adapter_command'])
 exe=p/'bin'/role/'rust-star'
 if exe.exists():assert sha(exe)==identity['executable_sha256']
 assert identity['model_sha256']==freeze['model_sha256'] and identity['prompt_sha256']==freeze['prompt_sha256']
 cfg=identity['runtime_configuration'];assert cfg['initial_bootstrap_last_completed_layer']==(2 if role=='candidate' else 42)
 assert cfg['initial_bootstrap_output_head']==(role=='oracle') and not cfg['prefill_stage_profile_collection']
 for metric,value in ctx[role].items():assert dist([x[role][metric] for x in pairs])==value
 evidence=[(x['evidence'][role],x[role]) for x in pairs]+[(x,x['metrics']) for x in state['warmups'][role]]
 assert len(evidence)==6
 for record,metrics in evidence:
  d=r/record['output'];m=read(d/'measurement.json');e=read(d/'engine-run.json')
  assert sha(d/'measurement.json')==record['measurement_sha256']
  assert m['status']=='passed' and m['engine']=='rust-star' and m['returncode']==0 and not m['timed_out']
  assert m['metrics']==metrics
  for art in m['artifacts'].values():
   file=d/art['path'];assert sha(file)==art['sha256'] and file.stat().st_size==art['bytes']
  assert e['paired_protocol_eligible'] and e['paired_protocol_blocker'] is None
  assert e['context']==32768 and e['gen_tokens']==128
  assert e['selection']==dict(prefill_token=85166,final_token=14,selected_tokens_checksum=11977321378026793466,oracle_transcript_match=True)
  assert e['long_prefill_profile'] is None and e['stage_profile'] is None and e['prefill_layer_profile'] is None
  for name,value in e['metrics'].items():assert metrics[name]==value
  for rate,count,interval in [('prefill_tps',32768,'prefill_ms'),('gen_tps',128,'gen_ms'),('gen_steady_tps',127,'gen_steady_ms')]:
   assert math.isfinite(metrics[interval]) and metrics[interval]>0
   assert abs(metrics[rate]-count*1000/metrics[interval])<1e-6
  assert abs(metrics['gen_first_ms']+metrics['gen_steady_ms']-metrics['gen_ms'])<1e-6
  t=e['timing'];assert t['generation_command_buffers_per_token']==44 and t['generation_host_waits_per_token']==2
  assert not t['prefill_correctness_collection'] and not t['generation_correctness_collection']
  assert not t['generation_stage_counter_collection'] and not t['prefill_layer_timing_collection']
  assert t['model_residency_queue_attached'] and t['model_residency_allocations']==t['model_view_count']>0
  accounted=t['model_view_warm_wall_ms']+t['prefill_host_overhead_ms']+sum(t[f'prefill_{stage}_wall_ms'] for stage in ['tile','transformer','output_head','handoff'])
  assert abs(accounted-metrics['prefill_ms'])<1e-3
  assert abs(metrics['process_wall_ms']-metrics['prefill_ms']-metrics['gen_ms']-metrics['process_overhead_ms'])<1e-6
for metric,value in ctx['candidate_over_oracle'].items():assert dist([x['candidate'][metric]/x['oracle'][metric] for x in pairs])==value
lat=[x['candidate']['prefill_ms']/x['oracle']['prefill_ms'] for x in pairs]
result={'passed':True,'comparison':'optimized Rust versus accepted Rust (generic oracle slot); not a new Rust-versus-DwarfStar result','valid_pairs':5,'invalid_attempts':0,'warmups':2,'raw_sha256':sha(r/'paired-raw.json'),'summary_sha256':sha(r/'paired-summary.json'),'paired_prefill_latency_ratio':dist(lat),'prefill_improved_pairs':sum(x<1 for x in lat),'prefill_acceptance_passed':statistics.median(lat)<1 and sum(x<1 for x in lat)>=4,'paired_decode_throughput_ratio':dist([x['candidate']['gen_steady_tps']/x['oracle']['gen_steady_tps'] for x in pairs]),'pairs':[dict(repetition=x['repetition'],order=x['order'],candidate_prefill_s=x['candidate']['prefill_ms']/1000,baseline_prefill_s=x['oracle']['prefill_ms']/1000,prefill_ratio=ratio,candidate_decode_tps=x['candidate']['gen_steady_tps'],baseline_decode_tps=x['oracle']['gen_steady_tps'],decode_ratio=x['candidate']['gen_steady_tps']/x['oracle']['gen_steady_tps']) for x,ratio in zip(pairs,lat)]}
(p/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
