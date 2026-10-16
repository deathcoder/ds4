#!/usr/bin/env python3
"""Offline reconstruction, raw-trace audit and full-C0 evidence verification."""
from pathlib import Path
import json,hashlib,gzip,tempfile,subprocess,sys,shutil,re
base=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();manifest=read(base/'manifest.json')
with tempfile.TemporaryDirectory(prefix='rust-star-lifetimes-') as d:
 w=Path(d)
 for name,entry in manifest['artifacts'].items():
  assert Path(name).name==name and Path(entry['stored']).name==entry['stored']
  payload=(base/entry['stored']).read_bytes();assert sha(payload)==entry['stored_sha256'],name
  data=gzip.decompress(payload) if entry['stored'].endswith('.gz') else payload
  assert len(data)==entry['bytes'] and sha(data)==entry['sha256'],name
  (w/name).write_bytes(data)
 plan=read(w/'plan.json');build=read(w/'build.json');audit=read(w/'audit.json');inst=read(w/'instrumentation.json')
 assert plan['baseline_commit']=='bff07722426c0b8719e2d0bee33e4312ec91d377'
 assert read(w/'validation.json')['passed'] and read(w/'validation.json')['full_fp32_words_total']==33354240
 expected=read(w/'oracle-32k.json');assert expected['complete_32k_c0_claim'] and expected['checked_decode_steps']==128 and expected['decode_logits_checksum']==1716816788919264169 and expected['logits_checksum']==2996662110144943749
 for role in ['accepted','wide']:
  assert read(w/(role+'-32k.json'))==expected
  status=read(w/(role+'-status.json'));assert status['exit_code']==0
  log=(w/(role+'-run.log')).read_text();assert log.count('all 129280 logits exact, token ')==128 and 'RS_LIFETIME_ERROR' not in log
  assert '\n'.join(l[len('RS_LIFETIME '):] for l in log.splitlines() if l.startswith('RS_LIFETIME '))+'\n'==(w/(role+'-trace.jsonl')).read_text()
  supplement=read(w/'supplemental-analysis.json')['roles'][role]
  assert int(re.search(r'(\d+)\s+peak memory footprint',log)[1])==supplement['whole_process_peak_footprint_bytes']
  assert int(re.search(r'(\d+)\s+maximum resident set size',log)[1])==supplement['whole_process_peak_rss_bytes']
  trace=[json.loads(l) for l in (w/(role+'-trace.jsonl')).read_text().splitlines()]
  scratch=set(inst[role]['scratch']);assert len(scratch)==105
  assert all({n for n,q in x['labels'].items() if not q['persistent']}==scratch for x in trace if x['phase']=='registered')
  assert all(x['previous_scratch_bytes']==0 for x in trace if x['phase']=='waited')
  for entry in supplement['one_set_per_frontier']:
   first=next(x for x in trace if x['phase']=='registered' and x['frontier']==entry['frontier'])
   assert first['scratch_bytes']==entry['one_set_bytes'] and first['previous_scratch_bytes']==0
   assert all(q['count']==1 for q in first['labels'].values() if not q['persistent'])

  for n,h in build['roles'][role]['sources'].items():
   if n=='rust-star/runtime/src/metal_shim.m':assert sha((w/(role+'-instrumented.m')).read_bytes())==h
   elif Path(n).name in ['metal.rs','long_prefill.rs']:assert sha((w/(role+'-'+Path(n).name)).read_bytes())==h
   else:assert plan['sources'][n]==h
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:
  assert sha((w/('accepted-'+n)).read_bytes())==plan['sources']['rust-star/runtime/src/'+n]
  assert sha((w/('wide-'+n)).read_bytes())==read(w/'prior-wide-build.json')['sources']['rust-star/runtime/src/'+n]
 for n,h in read(w/'protocol-freeze.json').items():assert sha((w/n).read_bytes())==h,n
 for n in ['input-before.json','input-after.json']:
  p=read(w/n);assert p['passed'] and p['fixture_count']==7669
  assert p['inputs']==read(w/'reference-inputs.json')['inputs']
 assert read(w/'input-before.json')['fixture_manifest_sha256']==read(w/'input-after.json')['fixture_manifest_sha256']
 assert read(w/'final-runtime-state.json')['sources']==plan['sources']
 assert read(w/'final-runtime-state.json')['default_executable_sha256']==plan['default_executable_sha256']
 assert build['passed'] and all(x['exit_code']==0 for x in build['runs'])
 proof=[json.loads(l[len('RS_LIFETIME '):]) for l in (w/'observer-proof.log').read_text().splitlines() if l.startswith('RS_LIFETIME ')]
 assert [(x['scratch_bytes'],x['persistent_bytes'],x['live_count']) for x in proof]==[(4096,8192,2),(0,8192,1),(0,0,0)]
 subprocess.run([sys.executable,str(w/'prepare.py')],check=True,stdout=subprocess.PIPE)
 assert read(w/'instrumentation.json')==inst
 for role in ['accepted','wide']:assert sha((w/(role+'-instrumented.m')).read_bytes())==inst[role]['instrumented_sha256']
 (w/'audit.json').unlink()
 original_failure=subprocess.run([sys.executable,str(w/'audit.py')],capture_output=True,text=True)
 assert original_failure.returncode!=0 and 'AssertionError' in original_failure.stderr
 correction=read(w/'width-correction.json')
 assert sha((w/'audit.py').read_bytes())==correction['original_auditor_sha256']
 assert sha((w/'audit-v2.py').read_bytes())==correction['corrected_auditor_sha256']
 assert (w/'audit.py').read_text().replace(correction['before'],correction['after'])==(w/'audit-v2.py').read_text()
 assert 'const PRODUCTION_TILE_ROWS: usize = 64;' in (w/'accepted-metal.rs').read_text()
 subprocess.run([sys.executable,str(w/'audit-v2.py')],check=True,stdout=subprocess.PIPE);assert read(w/'audit.json')==audit
 # Adversarial auditor checks use copied raw evidence, never modify preserved runs.
 original=(w/'wide-trace.jsonl').read_text();lines=original.splitlines();cases={}
 cases['missing_snapshot']='\n'.join(lines[1:])+'\n'
 for name,mutate in [('wrong_live_bytes',lambda x:x.update(scratch_bytes=x['scratch_bytes']+1)),('command_error',lambda x:x.update(command_status_counts=[0,0,0,0,0,1,0],commands=1))]:
  data=[json.loads(l) for l in lines];mutate(data[1]);cases[name]='\n'.join(json.dumps(x) for x in data)+'\n'
 for name,bad in cases.items():
  (w/'wide-trace.jsonl').write_text(bad);(w/'audit.json').unlink(missing_ok=True)
  result=subprocess.run([sys.executable,str(w/'audit-v2.py')],capture_output=True,text=True);assert result.returncode!=0,name
 (w/'wide-trace.jsonl').write_text(original)
 print(json.dumps(dict(passed=True,artifacts=len(manifest['artifacts']),full_fp32_words=33354240,source_reconstruction=True,trace_audit_recomputed=True,adversarial_auditor_cases=list(cases)),indent=2))
