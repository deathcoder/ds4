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
assert read(r/'freeze-attempt-1.json')['model_work_started'] is False
subprocess.run([sys.executable,str(r/'audit-stopped-attempt.py'),str(r)],check=True)
b=read(r/'candidate-build.json')
for n,h in b['sources'].items():assert sha(r/'candidate-source'/n)==h
assert '317 passed; 0 failed' in (r/'tests.log').read_text();assert 'Ran 94 tests' in (r/'python-tests.log').read_text() and (r/'python-tests.log').read_text().rstrip().endswith('OK')
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
assert correctness['c0_oracle_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4' and correctness['c0_oracle_executable_sha256']=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
for name,h in correctness['candidate_gates'].items():assert sha(r/'checks'/(name+'.json'))==h
assert read(r/'final-model-verification.json')['model_sha256']==correctness['model_sha256']=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
final_inputs=read(r/'final-model-verification.json');assert final_inputs['passed']
for k,v in final_inputs.items():
 if k not in ['verified_utc','passed']:assert v==read(r/'preflight.json')[k],k
assert read(r/'independent-audit.json')['passed']
for n,h in correctness['measurement_tools_sha256'].items():assert sha(r/'measurement-tools'/n)==h
sys.dont_write_bytecode=True;sys.path.insert(0,str(r/'measurement-tools'))
from ruststar_measurement_lib import parse_engine_run
def check_measurement(d,role):
 x=read(d/'measurement.json');assert x['status']=='passed' and x['context']==131072 and x['gen_tokens']==128
 for a in x['artifacts'].values():assert sha(d/a['path'])==a['sha256'] and (d/a['path']).stat().st_size==a['bytes']
 if role in ('candidate','oracle'):
  v=read(d/'engine-run.json');assert v['paired_protocol_eligible'] and v['selection']['oracle_transcript_match'] and v['selection']['selected_tokens_checksum']==c0['decode_tokens_checksum'];assert v['selection']['prefill_token']==1110 and v['selection']['final_token']==2945
  assert not v['timing']['prefill_correctness_collection'] and not v['timing']['generation_correctness_collection']
  assert not v['timing']['generation_layer_timing_collection'] and not v['timing']['generation_stage_counter_collection']
  assert v.get('long_prefill_profile') is None and v.get('stage_profile') is None
 parsed=parse_engine_run(d/'engine-run.json',context=131072,gen_tokens=128)
 parsed['process_peak_bytes']=x['process_peak_bytes'];parsed['process_wall_ms']=x['process_wall_ms']
 residual=x['process_wall_ms']-parsed['prefill_ms']-parsed['gen_ms'];assert residual>=-max(5.0,x['process_wall_ms']*.01)
 parsed['process_overhead_ms']=max(0.0,residual);assert parsed==x['metrics']
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

assert read(r/'checks/batch-12k.json')['chunks']==read(r/'prior-controls/batch-12k.json')['chunks']
for actual,prior in zip(read(r/'checks/batch-32k-internal.json')['chunks'],read(r/'prior-controls/batch-32k-internal.json')['chunks']):
 if actual['end']<=24576:assert actual==prior
assert read(r/'checks/c0-128k-profiled.json')==c0==read(r/'profile-study/diagnostic-c0.json')
diagnostic_build=read(r/'profile-study/diagnostic-build.json')
for n,h in diagnostic_build['sources'].items():assert sha(r/'diagnostic-source'/n)==h
study=r/'profile-study'
commands=[
 [sys.executable,str(r/'checks/audit-profile128.py'),str(r/'checks/components-128k.json'),str(r/'checks/c0-128k-profiled.json')],
 [sys.executable,str(study/'audit-profile.py'),str(study/'diagnostic-profile.json')],
 [sys.executable,str(study/'audit-rejected-profile.py')],
 [sys.executable,str(study/'audit-score-tile.py')],
]
for command in commands:subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
for n in ['checks/components-128k-audit.json','profile-study/diagnostic-profile-audit.json','profile-study/rejected-profile-audit.json','profile-study/score-tile-audit.json']:assert sha(r/n)==hashes[n]['sha256'],n
import re
code_lines=lambda text:[line.strip() for line in re.sub(r'//[^\n]*','',text).splitlines() if line.strip()]
assert code_lines((r/'candidate-source/rust-star/runtime/src/indexer_scores.metal').read_text())==code_lines((study/'score-tile-candidate.metal').read_text())
assert sha(r/'candidate-source/metal/dsv4_misc.metal')==read(study/'score-tile-source-pins.json')['source_sha256']
baseline=read(r/'inputs/baseline-build.json')
assert baseline['commit']==correctness['baseline_commit']==freeze['oracle_commit']==rawp['oracle']['source_commit']
assert baseline['executable_sha256']==correctness['baseline_executable_sha256']==freeze['oracle_sha256']
assert sha(r/'inputs/baseline-report-manifest.json')==correctness['baseline_report_manifest_sha256']
assert sha(r/'inputs/baseline-build.json')==read(r/'inputs/baseline-report-manifest.json')['artifacts']['candidate-build.json']['sha256']
assert sha(r/'inputs/protocol-scope.json')==correctness['protocol_scope_sha256']
acceptance=read(r/'inputs/acceptance-plan.json');assert sha(r/'inputs/acceptance-plan.json')==freeze['acceptance_plan_sha256']==b['acceptance_plan_sha256']==correctness['acceptance_plan_sha256']
ratios=extended['candidate_over_oracle'];faster=sum(p['candidate']['prefill_ms']<p['oracle']['prefill_ms'] for p in rawp['pairs'])
criteria=dict(prefill_median=ratios['prefill_ms']['median']<acceptance['prefill_median_candidate_baseline_ratio_must_be_less_than'],prefill_pairs=faster>=acceptance['prefill_faster_pairs_minimum'],steady_decode=ratios['gen_steady_tps']['median']>=acceptance['steady_decode_tps_median_candidate_baseline_ratio_minimum'],peak_rss=ratios['process_peak_bytes']['median']<=acceptance['peak_rss_median_candidate_baseline_ratio_maximum'])
verdict=read(r/'optimization-verdict.json');assert verdict['criteria']==criteria and verdict['accepted']==all(criteria.values()) and verdict['prefill_faster_pairs']==faster
for field,metric in [('prefill_latency_ratio','prefill_ms'),('steady_decode_throughput_ratio','gen_steady_tps'),('first_step_latency_ratio','gen_first_ms'),('peak_rss_ratio','process_peak_bytes')]:assert verdict[field]==ratios[metric]
assert read(r/'disposition.json')['optimization_retained']==verdict['accepted']
print(f'PASS: {len(files)} artifacts; 317 Rust/94 Python tests, ten live gates, 128K profiles, isolated score controls, five Rust-to-Rust pairs/two warmups, all distributions; optimization retained={verdict["accepted"]}')
