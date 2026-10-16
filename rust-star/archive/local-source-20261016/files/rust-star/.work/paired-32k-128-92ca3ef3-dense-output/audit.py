from pathlib import Path
import hashlib,json,math,statistics,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];out=w/'results'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dist(values):
 m=statistics.median(values);return {'count':len(values),'median':m,'mad':statistics.median(abs(x-m) for x in values),'min':min(values),'max':max(values)}
raw=read(out/'paired-raw.json');summary=read(out/'paired-summary.json');state=read(out/'state.json');freeze=read(w/'inputs/freeze.json');plan=read(w/'inputs/plan.json')
valid_pairs=[pair for pair in raw['pairs'] if pair['valid']]
invalid_raw=[pair for pair in raw['pairs'] if not pair['valid']]
assert len(valid_pairs)==5 and len(invalid_raw)==1
assert (invalid_raw[0]['context'],invalid_raw[0]['repetition'],invalid_raw[0]['attempt'])==(32768,4,1)
assert invalid_raw[0]['invalid_reason'].startswith('Host shutdown/restart interrupted pair 4 attempt 1')
assert state['phase']=='complete' and sha(w/'inputs/plan.json')==freeze['plan_sha256']
assert raw['configuration']['contexts']==[32768] and raw['configuration']['repetitions']==5 and raw['correctness_class']=='C0'
assert raw['host_manifest_sha256']==sha(w/'inputs/host-before.json') and raw['correctness_manifest_sha256']==sha(w/'inputs/correctness-manifest.json')
assert len(valid_pairs)==5 and [p['order'] for p in valid_pairs]==['AB','BA','AB','BA','AB']
assert all(pair['valid'] and pair['attempt']==(2 if pair['repetition']==4 else 1) for pair in valid_pairs)
invalid=[a for a in state['attempts'] if a['status']=='invalid']
assert len(invalid)==1
bad=invalid[0]
assert (bad['context'],bad['repetition'],bad['attempt'])==(32768,4,1)
assert bad['invalid_reason'].startswith('Host shutdown/restart interrupted pair 4 attempt 1')
interruption=read(w/'inputs/restart-interruption.json')
snapshot=w/'inputs/state-before-restart-recovery.json'
assert sha(snapshot)==interruption['interrupted_state_sha256']
assert interruption['confirmed_no_benchmark_processes'] and interruption['candidate_and_oracle_binaries_rehashed']
before=read(snapshot)['attempts'][-1]
assert before['status']=='running' and before['engines']['oracle']['status']=='running'
assert {k:v for k,v in bad.items() if k not in ('status','invalid_reason','invalidated_at_utc')}=={k:v for k,v in before.items() if k!='status'}
e=bad['engines']['candidate'];directory=out/e['output'];measurement=directory/'measurement.json'
assert sha(measurement)==e['measurement_sha256']
m=read(measurement);assert m['status']=='passed' and m['metrics']==e['metrics']
for artifact in m['artifacts'].values():
 file=directory/artifact['path'];assert file.stat().st_size==artifact['bytes'] and sha(file)==artifact['sha256']

for role in ['oracle','candidate']:
 exe=w/'bin'/role/('ds4-bench' if role=='oracle' else 'rust-star');assert sha(exe)==raw[role]['executable_sha256']==freeze[role+'_sha256']
 for p in valid_pairs:
  evidence=p['evidence'][role];directory=out/evidence['output'];measurement=directory/'measurement.json';r=read(measurement)
  assert sha(measurement)==evidence['measurement_sha256'] and r['status']=='passed'
  for key,v in p[role].items():assert r['metrics'][key]==v,(key,v,r['metrics'][key])
  for a in r['artifacts'].values():
   path=directory/a['path'];assert path.stat().st_size==a['bytes'] and sha(path)==a['sha256']
  if role=='candidate':
   e=read(directory/'engine-run.json');assert e['paired_protocol_eligible'] is True and e['selection']['oracle_transcript_match'] is True and e['selection']['selected_tokens_checksum']==11977321378026793466
 context=summary['contexts'][0]
 for metric,expected in context[role].items():assert dist([p[role][metric] for p in valid_pairs])==expected
for metric,expected in context['candidate_over_oracle'].items():assert dist([p['candidate'][metric]/p['oracle'][metric] for p in valid_pairs])==expected
subprocess.run(['python3',str(root/'rust-star/summarize_paired_benchmark.py'),str(out/'paired-raw.json'),'--json',str(w/'independent-summary.json')],check=True)
assert (w/'independent-summary.json').read_bytes()==(out/'paired-summary.json').read_bytes()
for role in ['oracle','candidate']:
 assert len(state['warmups'][role])==1
 record=state['warmups'][role][0]
 assert record['status']=='passed' and record['attempt']==1 and record['adapter_returncode']==0 and not record['timed_out']
 directory=out/record['output'];measurement=directory/'measurement.json';r=read(measurement)
 assert sha(measurement)==record['measurement_sha256'] and r['metrics']==record['metrics'] and r['status']=='passed'
 for artifact in r['artifacts'].values():
  path=directory/artifact['path'];assert path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256']
 if role=='candidate':
  e=read(directory/'engine-run.json');assert e['paired_protocol_eligible'] and e['selection']['oracle_transcript_match'] and e['selection']['selected_tokens_checksum']==11977321378026793466
failure=read(w/'profile-failure.json')
assert failure['exit_code']==1 and not (w/'profile32k.json').exists()
assert sha(w/'profile32k.log')==failure['log_sha256'] and 'invalid tail timestamps' in (w/'profile32k.log').read_text()
observations=[json.loads(s) for s in (w/'host-observations.jsonl').read_text().splitlines()]
unusual=[r for r in observations if 'No thermal warning level has been recorded' not in r['thermal'] or 'No performance warning level has been recorded' not in r['thermal']]
result={'schema':'rust-star-paired-independent-audit-v1','passed':True,'pairs':5,'invalid_attempts':1,'all_measurement_artifacts_rehashed':True,'warmups_verified':2,'profile_stage_shares_verified':False,'profile_status':'failed','profile_failure_sha256':sha(w/'profile-failure.json'),'all_candidate_transcripts_exact':True,'independent_statistics_match':True,'official_summary_byte_identical':True,'plan_sha256':freeze['plan_sha256'],'raw_sha256':sha(out/'paired-raw.json'),'summary_sha256':sha(out/'paired-summary.json'),'profile_sha256':None,'host_observations':len(observations),'unusual_thermal_observations':unusual,'memory_distributions':{role:dist([p[role]['process_peak_bytes']/2**30 for p in valid_pairs]) for role in ['oracle','candidate']},'prefill_wall_distributions':{role:dist([p[role]['prefill_ms']/1000 for p in valid_pairs]) for role in ['oracle','candidate']},'prefill_timing_medians':{key:statistics.median(read(out/p['evidence']['candidate']['output']/'engine-run.json')['timing'][key] for p in valid_pairs) for key in ['prefill_transformer_gpu_ms','prefill_transformer_wall_ms','prefill_transformer_setup_ms','prefill_tile_gpu_ms','prefill_tile_wall_ms','prefill_host_overhead_ms','prefill_unattributed_host_ms']},'claim_scope':'C0 32K/128 development baseline; not the 256K primary headline'}
(w/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS independent audit')
