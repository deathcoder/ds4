from pathlib import Path
import json,hashlib,subprocess,tempfile,shutil,sys
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')['files'];bins=read(w/'binary-identities.json')
for n,x in m.items():
 b=(w/n).read_bytes();assert sha(b)==x['stored_sha256'] and len(b)==x['stored_bytes'],n
refs=read(w/'prior-report-references.json');prior={}
for n,ref in refs.items():
 p=w.parent/n;assert sha((p/'artifacts.json').read_bytes())==ref['manifest_sha256'];prior[n]=p
base_report=prior['msrv-c0-5834fe56'];base_manifest=read(base_report/'artifacts.json')['files'];plan=read(w/'plan.json');build=read(w/'candidate-build.json');decision=read(w/'decision.json');assert decision['closed'] and build['passed'];ops=read(w/'source-edits.json');assert len(ops)==3
for n,h in build['files'].items():assert (bins[n]['sha256'] if n in bins else m[n]['original_sha256'])==h,n
assert bins['candidate-rust-star']['sha256']==build['executable_sha256'];assert bins['baseline-rust-star']['sha256']==build['baseline_executable_sha256']==plan['source_reference']['executable_sha256']
assert build['baseline_executable_sha256']==read(base_report/'completion.json')['executable_sha256']
for n,h in build['sources'].items():
 if n in ops:
  name=Path(n).name;pre=(w/('pre-format-'+name)).read_text();s=pre
  for e in reversed(ops[n]):assert s.count(e['after'])==1;s=s.replace(e['after'],e['before'])
  assert sha(s.encode())==plan['source_reference']['sources'][n];assert s==(w/('baseline-'+name)).read_text()
  formatted=subprocess.check_output(['rustup','run','1.74.0','rustfmt','--edition','2021','--emit','stdout'],input=pre,text=True) if n.endswith('.rs') else pre
  assert formatted==(w/('candidate-'+name)).read_text();assert sha(formatted.encode())==h
 else:assert h==plan['source_reference']['sources'][n]==base_manifest['sources/'+n]['original_sha256'],n
assert len(build['sources'])==42 and build['unchanged_other_source_count']==39
assert read(w/'router-proof.json')['rows']==[32,64,1024];assert read(w/'router-proof.json')['compared_words']==7840;assert read(w/'router-proof.json')['guard_words']==96
assert m['ffn_router.metal']['original_sha256']==build['sources']['rust-star/runtime/src/ffn_router.metal'];assert '320 passed; 0 failed' in (w/'host-tests.log').read_text()
for n in ['prepare-source','format','ring-audit','host-tests','release-build','format-check','router-build','router-probe']:assert read(w/(n+'-status.json'))['exit_code']==0
prepared=read(w/'prepared-plan.json');assert plan['diagnostic_selection']==prepared['diagnostic_selection'];ep=read(w/'engine-plan.json');assert ep['retention_criteria']==read(w/'prepared-engine-plan.json')['retention_criteria'];assert ep['orders']==['AB','BA','AB','BA','AB'];assert ep['retention_criteria']['median_peak_memory_footprint_ratio_at_most']==1.15
for name in ['input-before.json','input-after.json']:
 x=read(w/name);assert x['passed'];assert x['inputs']==plan['source_reference']['inputs'];assert x['fixtures']==plan['source_reference']['fixtures'];assert x['sources']==build['sources'];assert x['candidate_executable_sha256']==build['executable_sha256'];assert x['baseline_executable_sha256']==build['baseline_executable_sha256']
assert read(w/'input-before.json')['model_stat']==read(w/'input-after.json')['model_stat']
validation=read(w/'validation-acceptance.json') if (w/'validation-acceptance.json').exists() else None
if validation:
 assert validation['passed'] and len(validation['runs'])==3 and validation['full_fp32_words_total']==33354240
 expected12=read(prior['wide-bootstrap-c7adfb22']/'candidate-12k-internal.json');expected32=read(base_report/'current-32k-normal.json')
 for n,expected in [('candidate-12k-internal',expected12),('candidate-32k-output',expected32),('candidate-32k-profiled',expected32)]:
  assert read(w/(n+'-status.json'))['exit_code']==0;assert read(w/(n+'.json'))==expected
  if '32k' in n:assert (w/(n+'.log')).read_text().count('all 129280 logits exact, token ')==128
else:
 assert (w/'failure.json').exists() and not decision['retained']
with tempfile.TemporaryDirectory() as d:
 root=Path(d);p=root/'rust-star/.work/study';p.mkdir(parents=True)
 for n in ['audit-ring.py']:shutil.copyfile(w/n,p/n)
 run=subprocess.run([sys.executable,str(p/'audit-ring.py')],capture_output=True,text=True);assert run.returncode==0,run.stderr;assert read(p/'ring-audit.json')==read(w/'ring-audit.json')
 if (w/'profile-audit.json').exists():
  for n in ['plan.json','validation-acceptance.json','candidate-32k-profiled.json','candidate-32k-profiled.log','candidate-32k-profiled-components.json','audit-profile.py']:shutil.copyfile(w/n,p/n)
  run=subprocess.run([sys.executable,str(p/'audit-profile.py')],capture_output=True,text=True);assert run.returncode==0,run.stderr
  actual=read(p/'profile-audit.json');expected=read(w/'profile-audit.json')
  for k,n in [('log_sha256','candidate-32k-profiled.log'),('output_sha256','candidate-32k-profiled.json'),('profile_sha256','candidate-32k-profiled-components.json')]:assert expected[k]==m[n]['original_sha256'];expected[k]=m[n]['stored_sha256']
  assert actual==expected
 if (w/'engine/audit.json').exists():
  for n in ['engine-plan.json','candidate-build.json','validation-acceptance.json','engine-audit.py','audit-phases.py']:shutil.copyfile(w/n,p/n)
  shutil.copytree(w/'engine',p/'engine');(p/'engine/audit.json').unlink()
  for n in ['ruststar_measurement_lib.py','dwarfstar_measurement_lib.py','artifact_lib.py']:shutil.copyfile(w/'adapters'/n,root/'rust-star'/n)
  run=subprocess.run([sys.executable,str(p/'engine-audit.py')],capture_output=True,text=True);assert run.returncode==0,run.stderr
  actual=read(p/'engine/audit.json');expected=read(w/'engine/audit.json')
  assert expected['plan_sha256']==m['engine-plan.json']['original_sha256'];expected['plan_sha256']=m['engine-plan.json']['stored_sha256']
  ef=read(w/'engine/freeze.json')
  for n,h in ef['files'].items():assert m[n]['original_sha256']==h,n
  for n,h in ef['adapter_files'].items():assert m['adapters/'+Path(n).name]['original_sha256']==h,n
  for n,h in expected['raw_files'].items():assert h==m[n]['original_sha256'];expected['raw_files'][n]=m[n]['stored_sha256']
  assert actual==expected;assert decision['retained']==actual['candidate_retained_as_engine_improvement']
  run=subprocess.run([sys.executable,str(p/'audit-phases.py')],capture_output=True,text=True);assert run.returncode==0,run.stderr;actual=read(p/'passive-phase-audit.json');expected=read(w/'passive-phase-audit.json')
  for n,h in expected['raw_files'].items():assert h==m[n]['original_sha256'];expected['raw_files'][n]=m[n]['stored_sha256']
  assert actual==expected
if (w/'diagnostic-selection.json').exists():
 a=read(w/'diagnostic-selection.json');b=read(base_report/'profile-audit.json');c=read(w/'profile-audit.json');rules=plan['diagnostic_selection'];target=rules['target'];raw=c['stages'][target]['gpu_ms']/b['stages'][target]['gpu_ms'];norm=raw*sum(b['stages'][n]['gpu_ms'] for n in rules['controls'])/sum(c['stages'][n]['gpu_ms'] for n in rules['controls']);assert abs(a['raw_aggregate_ratio']-raw)<1e-12 and abs(a['control_normalized_aggregate_ratio']-norm)<1e-12;assert a['criteria_pass']==(raw<.9 and norm<.9)
state=read(w/'final-runtime-state.json');assert state['passed'];expected=build['sources'] if decision['retained'] else plan['source_reference']['sources'];assert state['sources']==expected
print(json.dumps(dict(passed=True,artifacts=len(m),complete_source_inverse=True,retained=decision['retained'],full_fp32_words_checked=33354240 if validation else 0,engine_pairs=5 if (w/'engine/audit.json').exists() else 0,scope='32K only; not DwarfStar comparison'),indent=2))
