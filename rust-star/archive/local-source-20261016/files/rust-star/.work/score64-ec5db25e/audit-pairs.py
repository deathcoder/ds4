from pathlib import Path
import hashlib,json,statistics,subprocess,sys
w=Path(__file__).resolve().parent;p=Path(json.loads((w/'paired-location.json').read_text())['work']);root=w.parents[2];out=p/'results'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(v):
 m=statistics.median(v);return {'count':len(v),'median':m,'mad':statistics.median(abs(x-m) for x in v),'min':min(v),'max':max(v)}
raw=read(out/'paired-raw.json');summary=read(out/'paired-summary.json');state=read(out/'state.json');freeze=read(p/'inputs/freeze.json');c0=read(w/'checks/c0-128k.json')
for name,h in read(p/'inputs/correctness-manifest.json')['measurement_tools_sha256'].items():assert sha(root/'rust-star'/name)==h,name
for name,h in read(w/'candidate-build.json')['sources'].items():assert sha(root/name)==h,name
assert state['phase']=='complete' and sha(p/'inputs/plan.json')==freeze['plan_sha256'];assert raw['configuration']['contexts']==[131072] and raw['configuration']['repetitions']==5 and raw['correctness_class']=='C0';assert len(raw['pairs'])==5
assert len(state['attempts'])==5 and [v['repetition'] for v in state['attempts']]==list(range(1,6))
assert all(v['status']=='passed' and v['attempt']==1 and v['context']==131072 for v in state['attempts'])
for attempt in state['attempts']:
 assert set(attempt['engines'])=={'candidate','oracle'}
 assert all(e['status']=='passed' and e['adapter_returncode']==0 and not e['timed_out'] for e in attempt['engines'].values())
assert [x['order'] for x in raw['pairs']]==['AB','BA','AB','BA','AB'] and all(x['valid'] and x['attempt']==1 for x in raw['pairs'])
assert raw['host_manifest_sha256']==sha(p/'inputs/host-before.json') and raw['correctness_manifest_sha256']==sha(p/'inputs/correctness-manifest.json')
def verify_measurement(directory,role):
 r=read(directory/'measurement.json');assert r['status']=='passed' and r['context']==131072 and r['gen_tokens']==128
 for a in r['artifacts'].values():assert sha(directory/a['path'])==a['sha256'] and (directory/a['path']).stat().st_size==a['bytes']
 if role in ('candidate','oracle'):
  x=read(directory/'engine-run.json');assert x['paired_protocol_eligible'] and x['selection']['oracle_transcript_match'];assert x['selection']['prefill_token']==1110 and x['selection']['final_token']==2945 and x['selection']['selected_tokens_checksum']==c0['decode_tokens_checksum']
  assert not x['timing']['prefill_correctness_collection'] and not x['timing']['generation_correctness_collection']
  assert not x['timing']['generation_layer_timing_collection'] and not x['timing']['generation_stage_counter_collection']
  assert x.get('long_prefill_profile') is None and x.get('stage_profile') is None
 return r
for role in ['oracle','candidate']:
 exe=p/'bin'/role/'rust-star';assert sha(exe)==raw[role]['executable_sha256']==freeze[role+'_sha256']
 for pair in raw['pairs']:
  e=pair['evidence'][role];d=out/e['output'];r=verify_measurement(d,role);assert sha(d/'measurement.json')==e['measurement_sha256'];assert r['metrics']==pair[role]
 for metric,expected in summary['contexts'][0][role].items():assert dist([x[role][metric] for x in raw['pairs']])==expected
 assert len(state['warmups'][role])==1
 r=state['warmups'][role][0];assert r['status']=='passed' and r['attempt']==1 and r['adapter_returncode']==0 and not r['timed_out'];d=out/r['output'];m=verify_measurement(d,role);assert sha(d/'measurement.json')==r['measurement_sha256'] and m['metrics']==r['metrics']
for metric,expected in summary['contexts'][0]['candidate_over_oracle'].items():assert dist([x['candidate'][metric]/x['oracle'][metric] for x in raw['pairs']])==expected
subprocess.run([sys.executable,str(root/'rust-star/summarize_paired_benchmark.py'),str(out/'paired-raw.json'),'--json',str(p/'independent-summary.json')],check=True)
assert (p/'independent-summary.json').read_bytes()==(out/'paired-summary.json').read_bytes()
metrics=[k for k,v in raw['pairs'][0]['candidate'].items() if isinstance(v,(float,int)) and all(isinstance(x[role].get(k),(float,int)) for x in raw['pairs'] for role in ['candidate','oracle'])]
extended={role:{k:dist([x[role][k] for x in raw['pairs']]) for k in metrics} for role in ['candidate','oracle']}
extended['candidate_over_oracle']={k:dist([x['candidate'][k]/x['oracle'][k] for x in raw['pairs']]) for k in metrics if all(x['oracle'][k]>0 for x in raw['pairs'])}
(p/'extended-summary.json').write_text(json.dumps(dict(contexts=[extended]),indent=2)+'\n')
result={'passed':True,'pairs':5,'warmups':2,'failed_attempts':0,'statistics_independently_recomputed':True,'measurement_tools_unchanged':True,'runtime_sources_unchanged':True,'all_measurement_artifacts_rehashed':True,'all_native_transcripts_match':True,'plan_sha256':freeze['plan_sha256'],'memory_gib':{r:dist([x[r]['process_peak_bytes']/2**30 for x in raw['pairs']]) for r in ['oracle','candidate']},'prefill_seconds':{r:dist([x[r]['prefill_ms']/1000 for x in raw['pairs']]) for r in ['oracle','candidate']}}
(p/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

acceptance=read(p/'inputs/acceptance-plan.json')
assert sha(p/'inputs/acceptance-plan.json')==freeze['acceptance_plan_sha256']==read(w/'candidate-build.json')['acceptance_plan_sha256']
assert raw['candidate']['source_commit']==freeze['candidate_commit'] and raw['oracle']['source_commit']==freeze['oracle_commit']
assert freeze['oracle_commit']==read(w/'baseline.json')['baseline_source_commit']
assert freeze['c0_oracle_source']==read(w/'preflight.json')['oracle_source']
ratios=extended['candidate_over_oracle'];faster=sum(x['candidate']['prefill_ms']<x['oracle']['prefill_ms'] for x in raw['pairs'])
criteria=dict(prefill_median=ratios['prefill_ms']['median']<acceptance['prefill_median_candidate_baseline_ratio_must_be_less_than'],prefill_pairs=faster>=acceptance['prefill_faster_pairs_minimum'],steady_decode=ratios['gen_steady_tps']['median']>=acceptance['steady_decode_tps_median_candidate_baseline_ratio_minimum'],peak_rss=ratios['process_peak_bytes']['median']<=acceptance['peak_rss_median_candidate_baseline_ratio_maximum'])
verdict=dict(accepted=all(criteria.values()),criteria=criteria,prefill_faster_pairs=faster,prefill_latency_ratio=ratios['prefill_ms'],steady_decode_throughput_ratio=ratios['gen_steady_tps'],first_step_latency_ratio=ratios['gen_first_ms'],peak_rss_ratio=ratios['process_peak_bytes'],acceptance_plan_sha256=freeze['acceptance_plan_sha256'],scope='128K/128 Rust-to-Rust development study; unchanged oracle-v5 is separate')
(p/'optimization-verdict.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict,indent=2))
