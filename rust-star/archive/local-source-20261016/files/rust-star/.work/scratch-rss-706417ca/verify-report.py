"""Read-only portable evidence verification; never launches GPU/model work."""
from pathlib import Path
import datetime,gzip,hashlib,json,subprocess,sys,tempfile
root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=read(root/'manifest.json');assert manifest['schema']=='rust-star-scratch-rss-report-v1'
with tempfile.TemporaryDirectory(prefix='scratch-rss-verify-') as temp:
 w=Path(temp)
 for name,e in manifest['artifacts'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert not Path(e['stored']).is_absolute() and '..' not in Path(e['stored']).parts
  payload=(root/e['stored']).read_bytes();assert sha(payload)==e['stored_sha256'],name
  data=gzip.decompress(payload) if e['stored'].endswith('.gz') else payload
  assert len(data)==e['bytes'] and sha(data)==e['sha256'],name
  dest=w/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 p=read(w/'plan.json');freeze=read(w/'freeze.json');bins=read(w/'binary-identities.json')
 for name,h in freeze['files'].items():
  assert (bins[name]['sha256'] if name in bins else sha((w/name).read_bytes()))==h,name
 assert read(w/'runs/final-pins.json')['files']==freeze['files']
 assert read(w/'runs/final-pins.json')['passed']
 state=read(w/'final-runtime-state.json');assert state['passed'] and not state['runtime_changed'] and not state['active_gpu_processes']
 assert state['sources']==p['runtime_state']['sources'] and len(state['sources'])==42
 assert state['default_executable_sha256']==p['runtime_state']['default_executable_sha256']
 assert state['default_executable_sha256']=='2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4'
 assert state['model_runs']==0
 assert read(w/'inventory.json')==p['inventory']
 assert len(p['inventory'])==105 and sum(x['bytes'] for x in p['inventory'])==2493173233
 assert p['orders']==['ABCDE','BCDEA','CDEAB','DEABC','EABCD']
 assert p['engine_retention_gates']==dict(all_runs_exact_transcript=True,all_five_pairs_prefill_faster=True,separate_ab_ba_median_prefill_ratio_below=.99,median_steady_decode_tps_ratio_at_least=.97,separate_order_steady_decode_tps_ratio_at_least=.95,median_peak_rss_ratio_at_most=1.02,median_peak_memory_footprint_ratio_at_most=1.15)
 def execute(script,success=True):
  x=subprocess.run([sys.executable,str(w/script)],cwd=w,capture_output=True,text=True)
  if success:assert x.returncode==0,(script,x.stdout,x.stderr)
  else:assert x.returncode!=0 and 'AssertionError' in x.stderr,(script,x.stdout,x.stderr)
  return x
 proof_before=(w/'source-proof.json').read_bytes();execute('source-proof.py')
 assert (w/'source-proof.json').read_bytes()==proof_before
 source=read(w/'source-proof.json');assert source['passed'] and source['selected_bytes']==536870912
 assert source['candidate_sha256']==sha((w/'prior-candidate-metal_shim.m').read_bytes())=='a41d4f3cfd9323ae8ed8702c3c10ef8ba700ba9aa3c3ab51eaa391541d98cf59'
 proof=[json.loads(x[7:]) for x in (w/'proof.log').read_text().splitlines() if x.startswith('RESULT ')]
 assert proof==[dict(cases=24,checked_bytes=2147549248,guard_bytes=24576,negative_controls=12,passed=True)]
 assert read(w/'proof-status.json')['exit_code']==0 and read(w/'build-status.json')['exit_code']==0
 failed=(w/'probe-attempt1.m').read_text();fixed=(w/'probe.m').read_text()
 assert failed.replace('assert([@[@"A",@"B",@"C",@"D",@"E"] containsObject:mode]);','assert(([@[@"A",@"B",@"C",@"D",@"E"] containsObject:mode]));')==fixed
 assert 'error:' in (w/'build-attempt1.log').read_text()
 assert read(w/'build-attempt1-failure.json')['runtime_changes'] is False
 progress=read(w/'runs/progress.json');previous=freeze['created_utc']
 for e in progress:
  assert e==read(w/'runs'/e['name']/'status.json')
  assert e['executable_sha256']==freeze['files']['probe'] and e['exit_code']==0
  assert datetime.datetime.fromisoformat(previous)<=datetime.datetime.fromisoformat(e['started_utc'])<=datetime.datetime.fromisoformat(e['ended_utc'])
  previous=e['ended_utc']
 assert datetime.datetime.fromisoformat(previous)<=datetime.datetime.fromisoformat(read(w/'runs/final-pins.json')['utc'])
 execute('audit.py');a=read(w/'audit.json')
 assert a['checked_bytes']==249317323300 and a['processes']==25 and a['tiles']==100
 assert all(a['gates'].values()) and a['select_for_future_candidate']
 assert a['model_runs']==0 and not a['engine_acceptance']
 assert all(v['final_live_bytes']==0 and v['max_live_bytes']==2493173233 for v in a['raw'].values())
 assert all(not any(v['fresh_previous_at_entry']) for v in a['raw'].values() if v['mode'] in 'AB')
 # The optimistic frozen audit did not enforce one explicit proof requirement.
 assert 'after-copy mutation' in p['preconditions'][1]
 gpu_proof=fixed.split('if([mode isEqual:@"proof"]) {',1)[1].split('assert(([@[@"A"',1)[0]
 assert gpu_proof.count('encode(e,writer,')==1 and 'encode(e,writer,src,' in gpu_proof
 assert (w/'probe.metal').read_text().count('kernel void ')==2
 d=read(w/'decision.json')
 assert d['closed'] and d['measurement_criteria_pass'] and d['full_copy_source_proof_pass'] and d['guarded_copy_proof_pass']
 assert not d['prospective_protocol_pass'] and not d['select_for_integration'] and not d['engine_acceptance']
 assert not d['runtime_changed'] and d['model_runs']==0
 # Meaningful auditor negative controls without launching new measurements.
 logfile=w/'runs/1-E/run.log';original=logfile.read_text();lines=original.splitlines(keepends=True)
 for i,line in enumerate(lines):
  if line.startswith('SNAPSHOT '):
   row=json.loads(line[9:])
   if row['phase']=='initialized':
    row['cpu_clear_bytes']+=1;lines[i]='SNAPSHOT '+json.dumps(row)+'\n';break
 else:raise AssertionError('missing initialization snapshot')
 logfile.write_text(''.join(lines));execute('audit.py',False);logfile.write_text(original)
 logfile=w/'runs/1-D/run.log';original=logfile.read_text();lines=original.splitlines(keepends=True)
 for i,line in enumerate(lines):
  if line.startswith('IDENTITIES '):
   row=json.loads(line[11:])
   if row['tile']==1:
    row['ids']['q_cur_buffer']+=1;lines[i]='IDENTITIES '+json.dumps(row)+'\n';break
 else:raise AssertionError('missing second tile identities')
 logfile.write_text(''.join(lines));execute('audit.py',False);logfile.write_text(original)
 print(json.dumps(dict(verified=True,artifacts=len(manifest['artifacts']),processes=25,tiles=100,checked_bytes=a['checked_bytes'],auditor_negative_controls=2,measurement_criteria_pass=True,prospective_protocol_pass=False,engine_acceptance=False,model_runs=0),indent=2))
