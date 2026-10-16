from pathlib import Path
import json,hashlib,re,subprocess,sys
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')['files']
for n,e in m.items():
 raw=(w/n).read_bytes();assert len(raw)==e['stored_bytes'] and sha(raw)==e['stored_sha256'],n;assert b'/Users/'+b'deathcodevision' not in raw,n
 if not e['redacted']:assert e['stored_sha256']==e['original_sha256'] and e['stored_bytes']==e['original_bytes'],n
original=lambda n:m[n]['original_sha256']
ref=read(w/'accepted-source-reference.json');prior=w/ref['report'];assert sha((prior/'artifacts.json').read_bytes())==ref['manifest_sha256'];pm=read(prior/'artifacts.json')['files'];base=ref['accepted_sources'];assert len(base)==42 and base==read(prior/'restoration.json')['sources']
for n,h in base.items():
 key=('baseline-sources/' if 'baseline-sources/'+n in pm else 'candidate-sources/')+n;assert sha((prior/key).read_bytes())==pm[key]['stored_sha256'] and pm[key]['original_sha256']==h,n
p=read(w/'plan.json');f=read(w/'freeze.json');c=read(w/'completion.json');assert c['passed'] and c['sources']==f['sources'] and c['files']==f['files'];assert c['all_other_41_sources_unchanged']
for n,h in f['files'].items():assert original(n)==h,n
assert p['prior_failure_sha256']==f['prior_failure_sha256']==original('first-attempt/failure.json')
assert c['accepted_performance_executable_sha256']==read(prior/'restoration.json')['executable_sha256'];assert all(c['sources'][n]==h for n,h in base.items() if n!='rust-star/runtime/src/metal.rs')
# Reproduce the complete formatted source inverse, using the declared compiler
# formatter only. This does not compile/link or run the model.
s=(w/'candidate-metal.rs').read_text();helper=(w/'helper.rs').read_text();assert original('candidate-metal.rs')==p['candidate_source_sha256']==c['sources']['rust-star/runtime/src/metal.rs'];assert sha(helper.encode())==p['helper_sha256'];assert s.count(helper)==1 and s.count('.array_refs()')==239 and s.count('.array_muts()')==39
inverse=s.replace(helper,'').replace('.array_refs()','.each_ref()').replace('.array_muts()','.each_mut()')
fmt=['rustfmt','+1.74.0','--edition','2021','--config','skip_children=true'];actual=subprocess.run(fmt,input=inverse,text=True,capture_output=True,check=True).stdout;assert actual==(w/'inverse-metal.rs').read_text();assert sha(actual.encode())==p['prior_source_sha256']==read(w/'first-attempt/candidate-freeze.json')['sources']['rust-star/runtime/src/metal.rs']
assert actual.count('.map_or(true, ')==2;original_source=actual.replace('.map_or(true, ','.is_none_or(');assert sha(original_source.encode())==base['rust-star/runtime/src/metal.rs'];assert original_source==(prior/'baseline-sources/rust-star/runtime/src/metal.rs').read_text()
pre=read(w/'first-attempt/preflight.json');assert pre['sources']==base and pre['run_sha256']==original('first-attempt/run.py');assert pre['observation_sha256']==original('first-attempt/observation.json')
first=read(w/'first-attempt/failure.json');assert [x['name'] for x in first['runs']]==['install-toolchain','accepted-check-failure','format','host-tests']
for x in first['runs']:
 assert x==read(w/('first-attempt/'+x['name']+'-status.json'));assert x['log_sha256']==original('first-attempt/'+x['name']+'.log');assert x['exit_code']==(101 if x['name'] in ['accepted-check-failure','host-tests'] else 0)
log=(w/'first-attempt/accepted-check-failure.log').read_text();assert log.count('no method named `is_none_or`')==2 and log.count("error[E0658]: use of unstable library feature 'array_methods'")==157
log=(w/'first-attempt/host-tests.log').read_text();assert log.count("error[E0658]: use of unstable library feature 'array_methods'")==278
assert read(w/'first-attempt/toolchain.json')['rustc'].startswith('rustc 1.74.0 (79e9716c9 2023-11-13)')
assert [x['name'] for x in c['runs']]==['format','host-tests','release-build','python-contracts','fixture-writer','fixture-reader']
for x in c['runs']:assert x==read(w/(x['name']+'-status.json')) and x['exit_code']==0 and x['log_sha256']==original(x['name']+'.log')
assert '320 passed; 0 failed' in (w/'host-tests.log').read_text();assert re.search(r'Finished .?release', (w/'release-build.log').read_text());log=(w/'python-contracts.log').read_text();count=re.search(r'Ran (\d+) tests',log);assert count and log.rstrip().endswith('OK')
assert original('support/artifact_lib.py')==pm['support/artifact_lib.py']['original_sha256'];sys.path.insert(0,str(w/'support'));from artifact_lib import load_logit_artifact,compare_logit_artifacts
fixture=load_logit_artifact(w/'writer-fixture.json');result=compare_logit_artifacts(fixture,fixture);assert json.loads(json.dumps(result))==read(w/'writer-reader-comparison.json') and result['c0_exact'] and result['vocab']==5
print(json.dumps(dict(passed=True,artifacts=len(m),source_inverse_exact=True,changed_source_files=1,unchanged_other_sources=41,rust_version='1.74.0',host_tests=320,python_tests=int(count[1]),synthetic_fixture_words=5,full_model_or_performance_claim=False),indent=2))
