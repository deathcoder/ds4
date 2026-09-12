"""Portable evidence audit: Python3.9+, no GPU/model/compiler required."""
from pathlib import Path
import json,hashlib,gzip,tempfile,subprocess,sys,shutil,re,datetime
base=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();manifest=read(base/'manifest.json')
with tempfile.TemporaryDirectory(prefix='rust-star-selective-clear-') as temp:
 root=Path(temp);w=root/'rust-star/.work/selective-clear-db1de4a9';w.mkdir(parents=True)
 for name,e in manifest['artifacts'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  assert not Path(e['stored']).is_absolute() and '..' not in Path(e['stored']).parts
  payload=(base/e['stored']).read_bytes();assert sha(payload)==e['stored_sha256'],name
  data=gzip.decompress(payload) if e['stored'].endswith('.gz') else payload
  assert len(data)==e['bytes'] and sha(data)==e['sha256'],name
  path=w/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 for f in (w/'adapters').iterdir():shutil.copy2(f,root/'rust-star'/f.name)
 plan=read(w/'plan.json');build=read(w/'candidate-build.json');ep=read(w/'engine-plan.json');bins=read(w/'binary-identities.json')
 assert ep['retention_criteria']==plan['engine_retention_gates']==read(w/'prior-engine-plan.json')['retention_criteria']
 assert ep['orders']==['AB','BA','AB','BA','AB']
 assert build['passed'] and all(x['exit_code']==0 for x in build['runs'])
 assert '320 passed; 0 failed' in (w/'host-tests.log').read_text()
 for n,h in build['files'].items():assert (sha((w/n).read_bytes()) if (w/n).exists() else bins[n]['sha256'])==h,n
 assert sha((w/'engine-plan.json').read_bytes())==build['engine_plan_sha256']
 # Prior failed diagnostic remains failed; this milestone supplies a new proof.
 assert not read(w/'prior-diagnostic-decision.json')['prospective_protocol_pass']
 mp=read(w/'mutation-proof-acceptance.json');assert mp['passed'] and mp['runtime_unchanged']
 assert mp['result']==dict(passed=True,cases=24,checked_bytes=2147549248,guard_bytes=24576,shortened_copy_negative_controls=12,missing_mutation_negative_controls=12)
 assert [json.loads(x[7:]) for x in (w/'mutation-proof.log').read_text().splitlines() if x.startswith('RESULT ')]==[mp['result']]
 assert read(w/'mutation-proof-status.json')['exit_code']==read(w/'mutation-build-status.json')['exit_code']==0
 for n,h in mp['files'].items():assert sha((w/n).read_bytes())==h,n
 assert read(w/'mutation-proof-status.json')['executable_sha256']==bins['mutation-proof']['sha256']
 host=(w/'mutation-proof.m').read_text();metal=(w/'mutation-proof.metal').read_text()
 assert host.count('encode(e,mutator,')==3 and 'newFunctionWithName:@"mutate_existing"' in host
 assert '*p = *p ^ uint4(0x9e3779b9u);' in metal and 'v^0x9e3779b9u' in metal
 # The only behavioral edit relative to the prior pool is its clear predicate.
 prior=read(w/'prior-build.json');candidate=(w/'candidate-metal_shim.m').read_text();check=candidate
 for e in reversed(read(w/'source-edits.json')['patches']):assert check.count(e['after'])==1;check=check.replace(e['after'],e['before'])
 assert check==(w/'prior-metal_shim.m').read_text()
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:
  assert sha((w/('baseline-'+n)).read_bytes())==plan['source_reference']['sources']['rust-star/runtime/src/'+n]
  assert sha((w/('prior-'+n)).read_bytes())==prior['sources']['rust-star/runtime/src/'+n]
  assert sha((w/('candidate-'+n)).read_bytes())==build['sources']['rust-star/runtime/src/'+n]
 for n in ['metal.rs','long_prefill.rs']:assert (w/('candidate-'+n)).read_bytes()==(w/('prior-'+n)).read_bytes()
 for n,h in build['sources'].items():
  if Path(n).name not in ['metal.rs','long_prefill.rs','metal_shim.m']:assert h==plan['source_reference']['sources'][n]
 check=(w/'candidate-instrumented.m').read_text();assert sha(check.encode())==build['diagnostic_sources']['rust-star/runtime/src/metal_shim.m']
 for e in reversed(read(w/'instrumentation.json')['patches']):assert check.count(e['after'])==1;check=check.replace(e['after'],e['before'])
 assert check==candidate
 helper=(w/'pool-helper.inc').read_text();interface,body=helper.split('// The caller checks prior',1);body='// The caller checks prior'+body
 assert interface in candidate and body in candidate and interface in (w/'pool-proof.m').read_text() and body in (w/'pool-proof.m').read_text()
 assert read(w/'pool-proof.log')==dict(passed=True,selected_policy_checked_bytes=1073741824,chunks=7,tiles=28,labels=105,compared_bytes=62043240,guard_bytes=192360,identity_reuses=2205,invalid_checkouts_rejected=84,final_and_error_cleanup=True)
 a=candidate.index('int rust_star_metal_run_prefill_layer0_boundary(');b=candidate.index('\nint ',a+4);f=candidate[a:b]
 calls=re.findall(r'bootstrap_scratch_buffer\(context.device, scratch_pool,\s*@"(\w+)"',f);assert len(calls)==105 and set(calls)==set(plan['pool_contract']['labels'])
 assert not [(k,v) for k,v in re.findall(r'context\.(\w+)\s*=\s*(\w+_buffer)\s*;',f) if v in calls]
 def recompute(script,outputs):
  saved={n:(w/n).read_bytes() for n in outputs}
  for n in outputs:(w/n).unlink()
  x=subprocess.run([sys.executable,str(w/script)],cwd=root,capture_output=True,text=True);assert x.returncode==0,(script,x.stderr)
  for n,data in saved.items():assert (w/n).read_bytes()==data,(script,n)
 recompute('source-proof.py',['source-proof.json']);assert read(w/'source-proof.json')['selected_bytes']==536870912
 recompute('audit-ring.py',['ring-audit.json'])
 router=read(w/'router-probe.log');assert router['passed'] and router['compared_words']==7840 and router['guard_words']==96
 expected=read(w/'oracle-32k.json');c0=0
 if (w/'candidate-12k-internal.json').exists():assert read(w/'candidate-12k-internal.json')==read(w/'oracle-12k.json')
 for name in ['candidate-32k-output','candidate-32k-profiled','candidate-lifetime']:
  if (w/(name+'.json')).exists() and read(w/(name+'-status.json'))['exit_code']==0:
   assert read(w/(name+'.json'))==expected and (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128;c0+=16677120
 for n in ['input-before.json','input-after.json']:
  p=read(w/n);assert p['passed'] and p['inputs']==plan['source_reference']['inputs'] and p['fixtures']==plan['source_reference']['fixtures'] and len(p['fixtures'])==7669 and p['sources']==build['sources']
 if (w/'profile-audit.json').exists():recompute('audit-profile.py',['profile-audit.json'])
 decision=read(w/'decision.json');state=read(w/'final-runtime-state.json');poolpass=(w/'diagnostic-selection.json').exists();enginepass=(w/'engine/audit.json').exists()
 if poolpass:
  recompute('audit-lifetime.py',['diagnostic-selection.json']);pool=read(w/'diagnostic-selection.json');assert pool['criteria_pass'] and pool['elided_checkouts']==112
 if enginepass:
  assert poolpass and c0==50031360
  freeze=read(w/'engine-precondition-freeze.json');final=read(w/'engine-precondition-final.json');assert final['passed'] and final['files']==freeze['files']
  for n,h in freeze['files'].items():assert sha((w/n).read_bytes())==h,n
  recompute('engine-audit.py',['engine/audit.json']);recompute('audit-phases.py',['passive-phase-audit.json']);a=read(w/'engine/audit.json');assert a['raw_runs']==12 and a['timed_pairs']==5
  assert decision['retained']==all(a['gates'].values())
 else:assert not decision['retained'] and any((w/n).exists() for n in ['failure.json','lifetime-failure.json','engine/failure.json'])
 assert state['candidate_retained']==decision['retained']
 assert state['sources']==(build['sources'] if decision['retained'] else plan['source_reference']['sources'])
 assert state['default_executable_sha256']==(build['executable_sha256'] if decision['retained'] else state['prior_default_executable_sha256'])
 assert bins['prior-default-rust-star']['sha256']==state['prior_default_executable_sha256']
 if poolpass:
  log=w/'candidate-lifetime.log';original=log.read_text()
  for field,value in [('prior_status',2),('elided',False)]:
   line=next(l for l in original.splitlines() if l.startswith('RS_POOL_CHECKOUT ') and json.loads(l[len('RS_POOL_CHECKOUT '):])['prior_status']==4 and json.loads(l[len('RS_POOL_CHECKOUT '):])['elided'])
   row=json.loads(line[len('RS_POOL_CHECKOUT '):]);row[field]=value;log.write_text(original.replace(line,'RS_POOL_CHECKOUT '+json.dumps(row),1))
   bad=subprocess.run([sys.executable,str(w/'audit-lifetime.py')],capture_output=True,text=True);assert bad.returncode!=0 and 'AssertionError' in bad.stderr
  log.write_text(original)
 print(json.dumps(dict(passed=True,artifacts=len(manifest['artifacts']),full_fp32_words=c0,mutation_proof_passed=True,pool_contract_passed=poolpass,engine_completed=enginepass,retained=decision['retained'],corrupted_pool_records_rejected=2 if poolpass else 0),indent=2))
