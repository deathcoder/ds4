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
assert state['phase']=='complete' and sha(w/'inputs/plan.json')==freeze['plan_sha256']
assert raw['configuration']['contexts']==[32768] and raw['configuration']['repetitions']==5 and raw['correctness_class']=='C0'
assert raw['host_manifest_sha256']==sha(w/'inputs/host-before.json') and raw['correctness_manifest_sha256']==sha(w/'inputs/correctness-manifest.json')
assert len(raw['pairs'])==5 and [p['order'] for p in raw['pairs']]==['AB','BA','AB','BA','AB']
assert all(p['valid'] and p['attempt']==1 for p in raw['pairs'])
for role in ['oracle','candidate']:
 exe=w/'bin'/role/('ds4-bench' if role=='oracle' else 'rust-star');assert sha(exe)==raw[role]['executable_sha256']==freeze[role+'_sha256']
 for p in raw['pairs']:
  evidence=p['evidence'][role];directory=out/evidence['output'];measurement=directory/'measurement.json';r=read(measurement)
  assert sha(measurement)==evidence['measurement_sha256'] and r['status']=='passed'
  for key,v in p[role].items():assert r['metrics'][key]==v,(key,v,r['metrics'][key])
  for a in r['artifacts'].values():
   path=directory/a['path'];assert path.stat().st_size==a['bytes'] and sha(path)==a['sha256']
  if role=='candidate':
   e=read(directory/'engine-run.json');assert e['paired_protocol_eligible'] is True and e['selection']['oracle_transcript_match'] is True and e['selection']['selected_tokens_checksum']==11977321378026793466
 context=summary['contexts'][0]
 for metric,expected in context[role].items():assert dist([p[role][metric] for p in raw['pairs']])==expected
for metric,expected in context['candidate_over_oracle'].items():assert dist([p['candidate'][metric]/p['oracle'][metric] for p in raw['pairs']])==expected
subprocess.run(['python3',str(root/'rust-star/summarize_paired_benchmark.py'),str(out/'paired-raw.json'),'--json',str(w/'independent-summary.json')],check=True)
assert (w/'independent-summary.json').read_bytes()==(out/'paired-summary.json').read_bytes()
profile=read(w/'profile32k.json');assert profile['context']==32768 and profile['selection']['oracle_transcript_match'] and not profile['paired_protocol_eligible']
observations=[json.loads(s) for s in (w/'host-observations.jsonl').read_text().splitlines()]
unusual=[r for r in observations if 'No thermal warning level has been recorded' not in r['thermal'] or 'No performance warning level has been recorded' not in r['thermal']]
result={'schema':'rust-star-paired-independent-audit-v1','passed':True,'pairs':5,'invalid_attempts':0,'all_measurement_artifacts_rehashed':True,'all_candidate_transcripts_exact':True,'independent_statistics_match':True,'official_summary_byte_identical':True,'plan_sha256':freeze['plan_sha256'],'raw_sha256':sha(out/'paired-raw.json'),'summary_sha256':sha(out/'paired-summary.json'),'profile_sha256':sha(w/'profile32k.json'),'host_observations':len(observations),'unusual_thermal_observations':unusual,'memory_distributions':{role:dist([p[role]['process_peak_bytes']/2**30 for p in raw['pairs']]) for role in ['oracle','candidate']},'prefill_wall_distributions':{role:dist([p[role]['prefill_ms']/1000 for p in raw['pairs']]) for role in ['oracle','candidate']},'prefill_timing_medians':{key:statistics.median(read(out/p['evidence']['candidate']['output']/'engine-run.json')['timing'][key] for p in raw['pairs']) for key in ['prefill_transformer_gpu_ms','prefill_transformer_wall_ms','prefill_transformer_setup_ms','prefill_tile_gpu_ms','prefill_tile_wall_ms','prefill_host_overhead_ms','prefill_unattributed_host_ms']},'claim_scope':'C0 32K/128 development baseline; not the 256K primary headline'}
(w/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS independent audit')
