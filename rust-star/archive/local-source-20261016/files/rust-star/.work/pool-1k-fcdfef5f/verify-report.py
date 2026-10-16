#!/usr/bin/env python3
"""Portable re-audit of scratch-pool evidence; Python3.9+, no model/GPU/Rust."""
from pathlib import Path
import json,hashlib,gzip,tempfile,subprocess,sys,shutil,re
base=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();manifest=read(base/'manifest.json')
with tempfile.TemporaryDirectory(prefix='rust-star-pool-') as temp:
 root=Path(temp);w=root/'rust-star/.work/pool-1k-fcdfef5f';w.mkdir(parents=True)
 for name,e in manifest['artifacts'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  payload=(base/e['stored']).read_bytes();assert sha(payload)==e['stored_sha256'],name
  data=gzip.decompress(payload) if e['stored'].endswith('.gz') else payload
  assert len(data)==e['bytes'] and sha(data)==e['sha256'],name
  path=w/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 for p in (w/'adapters').iterdir():shutil.copy2(p,root/'rust-star'/p.name)
 plan=read(w/'plan.json');build=read(w/'candidate-build.json');engineplan=read(w/'engine-plan.json');bins=read(w/'binary-identities.json')
 assert plan['baseline_commit']=='fcdfef5f51d44deb0dfab1ce94eba61c3d903acb'
 assert engineplan['retention_criteria']==read(w/'prior-engine-plan.json')['retention_criteria']
 assert engineplan['orders']==['AB','BA','AB','BA','AB']
 assert build['passed'] and all(x['exit_code']==0 for x in build['runs']);assert '320 passed; 0 failed' in (w/'host-tests.log').read_text()
 for n,h in build['files'].items():assert (sha((w/n).read_bytes()) if (w/n).exists() else bins[n]['sha256'])==h,n
 assert sha((w/'engine-plan.json').read_bytes())==build['engine_plan_sha256']
 prior=read(w/'prior-build.json');edit=read(w/'source-edits.json');candidate=(w/'candidate-metal_shim.m').read_text();check=candidate
 for e in reversed(edit['patches']):assert check.count(e['after'])==1;check=check.replace(e['after'],e['before'])
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
 a=candidate.index('int rust_star_metal_run_prefill_layer0_boundary(');b=candidate.index('\nint ',a+4);f=candidate[a:b];names=set(plan['pool_contract']['labels']);calls=re.findall(r'bootstrap_scratch_buffer\(context.device, scratch_pool,\s*@"(\w+)"',f);assert len(calls)==105 and set(calls)==names
 assignments=re.findall(r'context\.(\w+)\s*=\s*(\w+_buffer)\s*;',f);assert not [(k,v) for k,v in assignments if v in names]
 helper=(w/'pool-helper.inc').read_text();interface,body=helper.split('// newBufferWithLength',1);body='// newBufferWithLength'+body
 assert interface in candidate and body in candidate and interface in (w/'pool-proof.m').read_text() and body in (w/'pool-proof.m').read_text()
 proof=read(w/'pool-proof.log');assert proof==dict(passed=True,chunks=7,tiles=28,labels=105,compared_bytes=62043240,guard_bytes=192360,identity_reuses=2205,invalid_checkouts_rejected=84,final_and_error_cleanup=True)
 router=read(w/'router-probe.log');assert router['passed'] and router['compared_words']==7840 and router['guard_words']==96
 expected=read(w/'oracle-32k.json');assert expected['complete_32k_c0_claim'] and expected['checked_decode_steps']==128 and expected['decode_logits_checksum']==1716816788919264169
 c0=0
 if (w/'candidate-12k-internal.json').exists():assert read(w/'candidate-12k-internal.json')==read(w/'oracle-12k.json')
 for name in ['candidate-32k-output','candidate-32k-profiled','candidate-lifetime']:
  if (w/(name+'.json')).exists():
   assert read(w/(name+'.json'))==expected and (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
   assert read(w/(name+'-status.json'))['exit_code']==0;c0+=16677120
 for n in ['input-before.json','input-after.json']:
  p=read(w/n);assert p['passed'] and p['inputs']==plan['source_reference']['inputs'] and p['fixtures']==plan['source_reference']['fixtures'] and len(p['fixtures'])==7669 and p['sources']==build['sources']
 def recompute(script,outputs):
  saved={n:(w/n).read_bytes() for n in outputs}
  for n in outputs:(w/n).unlink()
  subprocess.run([sys.executable,str(w/script)],cwd=root,check=True,stdout=subprocess.PIPE)
  for n,data in saved.items():assert (w/n).read_bytes()==data,(script,n)
 recompute('audit-ring.py',['ring-audit.json'])
 if (w/'profile-audit.json').exists():recompute('audit-profile.py',['profile-audit.json'])
 amendment=read(w/'audit-representation-amendment.json');original=(w/'audit-lifetime.py').read_text();corrected=(w/'audit-pool-v2.py').read_text()
 assert original.replace(amendment['before'],amendment['after'])==corrected
 assert sha(original.encode())==amendment['original_sha256'] and sha(corrected.encode())==amendment['corrected_sha256']
 assert read(w/'status-representation.log')=={'no_prior_status':2**64-1,'NSUIntegerMax':2**64-1}
 assert not amendment['runtime_changes'] and not amendment['observer_changes'] and amendment['inference_repeats']==0 and not amendment['capacity_or_performance_threshold_changes']
 original_failure=subprocess.run([sys.executable,str(w/'audit-lifetime.py')],cwd=root,capture_output=True,text=True)
 assert original_failure.returncode!=0 and 'prior_status' in original_failure.stderr and 'AssertionError' in original_failure.stderr
 decision=read(w/'decision.json');state=read(w/'final-runtime-state.json');pool_pass=(w/'diagnostic-selection.json').exists();engine_pass=(w/'engine/audit.json').exists()
 if pool_pass:
  assert read(w/'diagnostic-selection.json')['criteria_pass'];recompute('audit-pool-v2.py',['diagnostic-selection.json'])
 else:
  assert not engine_pass and not decision['retained'];assert (w/'lifetime-failure.json').exists() or (w/'failure.json').exists()
  if (w/'candidate-lifetime.json').exists():
   failure=subprocess.run([sys.executable,str(w/'audit-pool-v2.py')],cwd=root,capture_output=True,text=True);assert failure.returncode!=0
 if engine_pass:
  freeze=read(w/'engine-precondition-freeze.json');final=read(w/'engine-precondition-final.json');assert final['passed'] and final['files']==freeze['files']
  for n,h in freeze['files'].items():assert sha((w/n).read_bytes())==h,n
  recompute('engine-audit.py',['engine/audit.json']);recompute('audit-phases.py',['passive-phase-audit.json']);a=read(w/'engine/audit.json');assert a['raw_runs']==12 and a['timed_pairs']==5
  assert decision['retained']==all(a['gates'].values())
 assert state['candidate_retained']==decision['retained']
 assert state['sources']==(build['sources'] if decision['retained'] else plan['source_reference']['sources'])
 assert state['default_executable_sha256']==(build['executable_sha256'] if decision['retained'] else state['prior_default_executable_sha256'])
 assert bins['prior-default-rust-star']['sha256']==state['prior_default_executable_sha256']
 if pool_pass:
  log=w/'candidate-lifetime.log';original=log.read_text();line=next(l for l in original.splitlines() if l.startswith('RS_POOL_CHECKOUT ') and json.loads(l[len('RS_POOL_CHECKOUT '):])['prior_status']==4);row=json.loads(line[len('RS_POOL_CHECKOUT '):]);row['prior_status']=2;log.write_text(original.replace(line,'RS_POOL_CHECKOUT '+json.dumps(row),1));(w/'diagnostic-selection.json').unlink();bad=subprocess.run([sys.executable,str(w/'audit-pool-v2.py')],capture_output=True,text=True);assert bad.returncode!=0
 print(json.dumps(dict(passed=True,artifacts=len(manifest['artifacts']),full_fp32_words=c0,pool_contract_passed=pool_pass,engine_completed=engine_pass,retained=decision['retained'],pool_auditor_rejects_corruption=pool_pass),indent=2))
