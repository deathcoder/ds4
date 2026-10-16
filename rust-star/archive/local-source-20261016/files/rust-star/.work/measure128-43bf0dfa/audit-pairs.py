from pathlib import Path
import hashlib,json,statistics,subprocess,sys
w=Path(__file__).resolve().parent;p=Path(json.loads((w/'paired-location.json').read_text())['work']);root=w.parents[2];out=p/'results'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(v):
 m=statistics.median(v);return {'count':len(v),'median':m,'mad':statistics.median(abs(x-m) for x in v),'min':min(v),'max':max(v)}
raw=read(out/'paired-raw.json');summary=read(out/'paired-summary.json');state=read(out/'state.json');freeze=read(p/'inputs/freeze.json');c0=read(w/'checks/c0-128k.json')
for name,h in read(p/'inputs/oracle-metal-hashes.json').items():assert sha(p/'bin/oracle'/name)==h
assert state['phase']=='complete' and sha(p/'inputs/plan.json')==freeze['plan_sha256'];assert raw['configuration']['contexts']==[131072] and raw['configuration']['repetitions']==5 and raw['correctness_class']=='C0';assert len(raw['pairs'])==5
assert [x['order'] for x in raw['pairs']]==['AB','BA','AB','BA','AB'] and all(x['valid'] and x['attempt']==1 for x in raw['pairs'])
assert raw['host_manifest_sha256']==sha(p/'inputs/host-before.json') and raw['correctness_manifest_sha256']==sha(p/'inputs/correctness-manifest.json')
def verify_measurement(directory,role):
 r=read(directory/'measurement.json');assert r['status']=='passed' and r['context']==131072 and r['gen_tokens']==128
 for a in r['artifacts'].values():assert sha(directory/a['path'])==a['sha256'] and (directory/a['path']).stat().st_size==a['bytes']
 if role=='candidate':
  x=read(directory/'engine-run.json');assert x['paired_protocol_eligible'] and x['selection']['oracle_transcript_match'];assert x['selection']['prefill_token']==1110 and x['selection']['final_token']==2945 and x['selection']['selected_tokens_checksum']==c0['decode_tokens_checksum']
  assert not x['timing']['prefill_correctness_collection'] and not x['timing']['generation_correctness_collection']
 return r
for role in ['oracle','candidate']:
 exe=p/'bin'/role/('ds4-bench' if role=='oracle' else 'candidate-rust-star');assert sha(exe)==raw[role]['executable_sha256']==freeze[role+'_sha256']
 for pair in raw['pairs']:
  e=pair['evidence'][role];d=out/e['output'];r=verify_measurement(d,role);assert sha(d/'measurement.json')==e['measurement_sha256'];assert r['metrics']==pair[role]
 for metric,expected in summary['contexts'][0][role].items():assert dist([x[role][metric] for x in raw['pairs']])==expected
 assert len(state['warmups'][role])==1
 r=state['warmups'][role][0];assert r['status']=='passed' and r['attempt']==1 and r['adapter_returncode']==0 and not r['timed_out'];d=out/r['output'];m=verify_measurement(d,role);assert sha(d/'measurement.json')==r['measurement_sha256'] and m['metrics']==r['metrics']
for metric,expected in summary['contexts'][0]['candidate_over_oracle'].items():assert dist([x['candidate'][metric]/x['oracle'][metric] for x in raw['pairs']])==expected
subprocess.run([sys.executable,str(root/'rust-star/summarize_paired_benchmark.py'),str(out/'paired-raw.json'),'--json',str(p/'independent-summary.json')],check=True)
assert (p/'independent-summary.json').read_bytes()==(out/'paired-summary.json').read_bytes()
result={'passed':True,'pairs':5,'warmups':2,'failed_attempts':0,'statistics_independently_recomputed':True,'all_measurement_artifacts_rehashed':True,'all_native_transcripts_match':True,'plan_sha256':freeze['plan_sha256'],'memory_gib':{r:dist([x[r]['process_peak_bytes']/2**30 for x in raw['pairs']]) for r in ['oracle','candidate']},'prefill_seconds':{r:dist([x[r]['prefill_ms']/1000 for x in raw['pairs']]) for r in ['oracle','candidate']}}
(p/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
