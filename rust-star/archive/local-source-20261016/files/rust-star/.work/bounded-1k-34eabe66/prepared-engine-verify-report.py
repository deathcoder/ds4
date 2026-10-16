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
 for p in (w/'adapters').iterdir():shutil.copy2(p,root/'rust-star'/p.name)
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
 recompute('audit-lifetime.py',['diagnostic-selection.json','candidate-lifetime-trace.jsonl'])
 recompute('audit-ring.py',['ring-audit.json'])
 recompute('engine-audit.py',['engine/audit.json'])
 recompute('audit-phases.py',['passive-phase-audit.json'])
 a=read(w/'engine/audit.json');decision=read(w/'decision.json');state=read(w/'final-runtime-state.json');assert decision['retained']==all(a['gates'].values())==state['candidate_retained']
 assert state['sources']==(build['sources'] if decision['retained'] else plan['source_reference']['sources'])
 assert state['default_executable_sha256']==(build['executable_sha256'] if decision['retained'] else state['prior_default_executable_sha256'])
 # A corrupt one-in-flight observation must fail in a temporary copy.
 log=w/'candidate-lifetime.log';original=log.read_text();line=next(l for l in original.splitlines() if l.startswith('RS_LIFETIME '));row=json.loads(line[len('RS_LIFETIME '):]);row['previous_scratch_bytes']=1;log.write_text(original.replace(line,'RS_LIFETIME '+json.dumps(row),1));(w/'diagnostic-selection.json').unlink();(w/'candidate-lifetime-trace.jsonl').unlink();p=subprocess.run([sys.executable,str(w/'audit-lifetime.py')],capture_output=True,text=True);assert p.returncode!=0
 print(json.dumps(dict(passed=True,artifacts=len(manifest['artifacts']),full_fp32_words=50031360,profile_intervals=4466,engine_pairs=5,all_gates=a['gates'],retained=decision['retained'],lifetime_auditor_rejects_corruption=True),indent=2))
