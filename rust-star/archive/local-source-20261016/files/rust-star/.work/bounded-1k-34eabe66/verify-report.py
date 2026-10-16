#!/usr/bin/env python3
"""Recompute the bounded-1K milestone evidence without GPU/model/toolchain."""
from pathlib import Path
import json,hashlib,gzip,tempfile,subprocess,sys,shutil
base=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();manifest=read(base/'manifest.json')
with tempfile.TemporaryDirectory(prefix='rust-star-bounded-') as temp:
 root=Path(temp);w=root/'rust-star/.work/bounded-1k-34eabe66';w.mkdir(parents=True)
 for name,e in manifest['artifacts'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  payload=(base/e['stored']).read_bytes();assert sha(payload)==e['stored_sha256'],name
  data=gzip.decompress(payload) if e['stored'].endswith('.gz') else payload
  assert len(data)==e['bytes'] and sha(data)==e['sha256'],name
  p=w/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 plan=read(w/'plan.json');build=read(w/'candidate-build.json');source=read(w/'source-audit.json');engine_plan=read(w/'engine-plan.json')
 assert plan['baseline_commit']=='34eabe66b7abeb5d70a2b0e31c0dc04cc1663a99'
 assert engine_plan['retention_criteria']==read(w/'prior-1k-engine-plan.json')['retention_criteria']
 assert engine_plan['orders']==['AB','BA','AB','BA','AB']
 assert build['passed'] and all(x['exit_code']==0 for x in build['runs'])
 assert '320 passed; 0 failed' in (w/'host-tests.log').read_text()
 for n,h in build['files'].items():assert sha((w/n).read_bytes())==h,n
 assert sha((w/'engine-plan.json').read_bytes())==build['engine_plan_sha256']
 prior=read(w/'prior-1k-build.json');edit=read(w/'scheduling-edit.json')
 c=(w/'candidate-metal_shim.m').read_text();assert c.count(edit['after'])==1 and c.replace(edit['after'],edit['before'])==(w/'prior-metal_shim.m').read_text()
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:
  assert sha((w/('baseline-'+n)).read_bytes())==plan['source_reference']['sources']['rust-star/runtime/src/'+n]
  assert sha((w/('prior-'+n)).read_bytes())==prior['sources']['rust-star/runtime/src/'+n]
  assert sha((w/('candidate-'+n)).read_bytes())==build['sources']['rust-star/runtime/src/'+n]
 for n in ['metal.rs','long_prefill.rs']:assert (w/('candidate-'+n)).read_bytes()==(w/('prior-'+n)).read_bytes()
 for n,h in build['sources'].items():
  if Path(n).name not in ['metal.rs','long_prefill.rs','metal_shim.m']:assert h==plan['source_reference']['sources'][n]
 inst=read(w/'instrumentation.json');subprocess.run([sys.executable,str(w/'prepare-observer.py')],check=True,stdout=subprocess.PIPE);assert read(w/'instrumentation.json')==inst
 assert sha((w/'candidate-instrumented.m').read_bytes())==build['diagnostic_sources']['rust-star/runtime/src/metal_shim.m']
 expected=read(w/'oracle-32k.json');assert expected['complete_32k_c0_claim'] and expected['checked_decode_steps']==128 and expected['decode_logits_checksum']==1716816788919264169 and expected['logits_checksum']==2996662110144943749
 assert read(w/'candidate-12k-internal.json')==read(w/'oracle-12k.json')
 for name in ['candidate-32k-output','candidate-32k-profiled','candidate-lifetime']:
  assert read(w/(name+'.json'))==expected and (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
  assert read(w/(name+'-status.json'))['exit_code']==0
 for n in ['input-before.json','input-after.json']:
  p=read(w/n);assert p['passed'] and p['inputs']==plan['source_reference']['inputs'] and p['fixtures']==plan['source_reference']['fixtures'] and len(p['fixtures'])==7669 and p['sources']==build['sources']
 def recompute(script,outputs):
  saved={n:(w/n).read_bytes() for n in outputs}
  for n in outputs:(w/n).unlink()
  subprocess.run([sys.executable,str(w/script)],cwd=root,check=True,stdout=subprocess.PIPE)
  for n,data in saved.items():assert (w/n).read_bytes()==data,(script,n)
 recompute('audit-profile.py',['profile-audit.json'])
 failed=subprocess.run([sys.executable,str(w/'audit-lifetime.py')],cwd=root,capture_output=True,text=True)
 assert failed.returncode!=0 and 'scratch_bytes' in failed.stderr and 'AssertionError' in failed.stderr
 assert not (w/'diagnostic-selection.json').exists()
 recompute('diagnostic-failure-analysis.py',['diagnostic-failure-analysis.json','candidate-lifetime-trace.jsonl'])
 recompute('audit-ring.py',['ring-audit.json'])
 failure=read(w/'diagnostic-failure-analysis.json');decision=read(w/'decision.json');state=read(w/'final-runtime-state.json')
 assert failure['structure_passed'] and not failure['criteria_pass'] and failure['max_unfinished_commands']==1
 assert [v['snapshot_index'] for v in failure['violations']]==[54,55,56,57]
 assert failure['violations'][0]['observation']['scratch_bytes']==1977044465
 assert failure['violations'][1]['observation']['previous_scratch_bytes']==1420731848
 assert failure['violations'][2]['observation']['previous_scratch_bytes']==5242880
 assert not decision['retained'] and decision['engine_runs']==0 and not (w/'engine').exists()
 assert state['sources']==plan['source_reference']['sources'] and not state['candidate_retained']
 assert state['default_executable_sha256']==state['prior_default_executable_sha256']=='2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4'
 assert read(w/'binary-identities.json')['prior-default-rust-star']['sha256']==state['default_executable_sha256']
 router=read(w/'router-probe.log');assert router['passed'] and router['compared_words']==7840 and router['guard_words']==96
 proof=[json.loads(l[12:]) for l in (w/'observer-proof.log').read_text().splitlines() if l.startswith('RS_LIFETIME ')]
 assert [(x['scratch_bytes'],x['persistent_bytes'],x['live_count']) for x in proof]==[(4096,8192,2),(0,8192,1),(0,0,0)]
 print(json.dumps(dict(passed=True,artifacts=len(manifest['artifacts']),full_fp32_words=50031360,profile_intervals=4466,original_lifetime_failure_reproduced=True,failure_snapshots=4,engine_runs=0,runtime_restored=True),indent=2))
