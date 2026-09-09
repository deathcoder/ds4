from pathlib import Path
import json,hashlib,struct,math,statistics,subprocess,sys
r=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fnv(raw):
 h=0xcbf29ce484222325
 for (v,) in struct.iter_unpack('<I',raw):h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
def dist(v):
 m=statistics.median(v);return dict(count=len(v),median=m,mad=statistics.median(abs(x-m) for x in v),min=min(v),max=max(v))
hashes=read(r/'artifact-hashes.json')['artifacts'];files={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file() and p.name!='artifact-hashes.json' and '__pycache__' not in p.parts}
assert files==set(hashes)
for n,x in hashes.items():assert sha(r/n)==x['sha256'] and (r/n).stat().st_size==x['bytes'],n
assert 'inconsistent measurement metadata' in (r/'rejected/5f3b9672/checks/retained-128k.log').read_text()
assert 'test result: FAILED. 0 passed; 1 failed' in (r/'rejected/5f3b9672/writer-regression-before-fix.log').read_text()
b=read(r/'candidate-build.json')
for n,h in b['sources'].items():assert sha(r/'candidate-source'/n)==h
assert '317 passed; 0 failed' in (r/'tests-final.log').read_text();assert 'Ran 94 tests' in (r/'python-tests.log').read_text() and (r/'python-tests.log').read_text().rstrip().endswith('OK')
ref=r/'reference-131072';m=read(ref/'manifest.json');assert m['frontier']==131072 and m['repeat_processes']==2
for n,x in m['artifacts'].items():assert sha(ref/n)==x['sha256'] and (ref/n).stat().st_size==x['bytes']
raw=(ref/'logits.f32le.bin').read_bytes();tokens=(ref/'selected.u32le.bin').read_bytes();selected=list(struct.unpack('<129I',tokens));assert len(raw)==129*129280*4 and selected[0]==1110 and selected[-1]==2945
row_hashes=[]
for j,token in enumerate(selected):
 row=raw[j*129280*4:(j+1)*129280*4];v=struct.unpack('<129280f',row);assert all(map(math.isfinite,v)) and max(range(len(v)),key=v.__getitem__)==token!=1;row_hashes.append(hashlib.sha256(row).hexdigest())
for rep in ['a','b']:
 x=read(r/f'oracle-131072-{rep}-verification.json');assert x['row_sha256']==row_hashes and x['selected_tokens']==selected and x['all_logits_sha256']==sha(ref/'logits.f32le.bin')
c0=read(r/'checks/c0-128k.json');assert c0['prefill_logits_checksum']==fnv(raw[:129280*4]) and c0['decode_logits_checksum']==fnv(raw[129280*4:]) and c0['decode_tokens_checksum']==fnv(tokens[4:])
assert hashlib.sha256((ref/'prompt.u32le.bin').read_bytes()[:32768*4]).hexdigest()=='874c479e2cde7b231dfa4a963d5f1fbe9dd9e60c2842f869e2a117d4860902a0'
fixture=r/'candidate-source/rust-star/fixtures/measurement-131072-v5'
for n in ['prompt.u32le.bin','selected.u32le.bin']:assert (fixture/n).read_bytes()==(ref/n).read_bytes()
subprocess.run([sys.executable,str(r/'audit-gates.py'),str(r)],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,str(r/'checks/audit-components.py')],check=True,stdout=subprocess.DEVNULL)
for n in ['gates-audit.json','checks/component-audit.json']:assert sha(r/n)==hashes[n]['sha256']
rawp=read(r/'results/paired-raw.json');summary=read(r/'results/paired-summary.json');state=read(r/'results/state.json');freeze=read(r/'inputs/freeze.json')
assert state['phase']=='complete' and rawp['configuration']['contexts']==[131072] and rawp['configuration']['repetitions']==5 and rawp['correctness_class']=='C0'
assert len(state['attempts'])==5 and [v['repetition'] for v in state['attempts']]==list(range(1,6))
assert all(v['status']=='passed' and v['attempt']==1 and v['context']==131072 for v in state['attempts'])
for attempt in state['attempts']:
 assert set(attempt['engines'])=={'candidate','oracle'}
 assert all(e['status']=='passed' and e['adapter_returncode']==0 and not e['timed_out'] for e in attempt['engines'].values())
assert len(rawp['pairs'])==5 and [p['order'] for p in rawp['pairs']]==['AB','BA','AB','BA','AB'];assert all(p['valid'] and p['attempt']==1 for p in rawp['pairs'])
assert rawp['host_manifest_sha256']==sha(r/'inputs/host-before.json') and rawp['correctness_manifest_sha256']==sha(r/'inputs/correctness-manifest.json')
correctness=read(r/'inputs/correctness-manifest.json');assert correctness['candidate_commit']==b['commit'] and correctness['candidate_executable_sha256']==b['executable_sha256']
assert correctness['candidate_build_sha256']==sha(r/'candidate-build.json') and correctness['fixture_manifest_sha256']==sha(fixture/'manifest.json')
assert (r/'independent-summary.json').read_bytes()==(r/'results/paired-summary.json').read_bytes()
assert correctness['oracle_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4' and correctness['oracle_executable_sha256']=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
for name,h in correctness['candidate_gates'].items():assert sha(r/'checks'/(name+'.json'))==h
assert read(r/'final-model-verification.json')['model_sha256']==correctness['model_sha256']=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
final_inputs=read(r/'final-model-verification.json');assert final_inputs['passed']
for k,v in final_inputs.items():
 if k not in ['verified_utc','passed']:assert v==read(r/'preflight.json')['inputs'][k],k
assert read(r/'independent-audit.json')['passed']
def check_measurement(d,role):
 x=read(d/'measurement.json');assert x['status']=='passed' and x['context']==131072 and x['gen_tokens']==128
 for a in x['artifacts'].values():assert sha(d/a['path'])==a['sha256'] and (d/a['path']).stat().st_size==a['bytes']
 if role=='candidate':
  v=read(d/'engine-run.json');assert v['paired_protocol_eligible'] and v['selection']['oracle_transcript_match'] and v['selection']['selected_tokens_checksum']==c0['decode_tokens_checksum'];assert v['selection']['prefill_token']==1110 and v['selection']['final_token']==2945
  assert not v['timing']['prefill_correctness_collection'] and not v['timing']['generation_correctness_collection']
 return x
for role in ['candidate','oracle']:
 assert rawp[role]['executable_sha256']==freeze[role+'_sha256']
 for p in rawp['pairs']:
  e=p['evidence'][role];d=r/'results'/e['output'];x=check_measurement(d,role);assert sha(d/'measurement.json')==e['measurement_sha256'] and x['metrics']==p[role]
 for metric,v in summary['contexts'][0][role].items():assert dist([p[role][metric] for p in rawp['pairs']])==v
 assert len(state['warmups'][role])==1;w=state['warmups'][role][0];assert w['attempt']==1 and w['status']=='passed' and not w['timed_out'];d=r/'results'/w['output'];x=check_measurement(d,role);assert sha(d/'measurement.json')==w['measurement_sha256'] and x['metrics']==w['metrics']
for metric,v in summary['contexts'][0]['candidate_over_oracle'].items():assert dist([p['candidate'][metric]/p['oracle'][metric] for p in rawp['pairs']])==v
extended=read(r/'extended-summary.json')['contexts'][0]
for role in ['candidate','oracle']:
 for metric,v in extended[role].items():assert dist([p[role][metric] for p in rawp['pairs']])==v
for metric,v in extended['candidate_over_oracle'].items():assert dist([p['candidate'][metric]/p['oracle'][metric] for p in rawp['pairs']])==v
print(f'PASS: {len(files)} artifacts; 317 Rust/94 Python tests, full 128K C0, retained lifecycle, five pairs/two warmups, all distributions')
