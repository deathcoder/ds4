from pathlib import Path
import json,hashlib,re,subprocess,sys,os
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')['files']
for n,e in m.items():
 raw=(w/n).read_bytes();assert len(raw)==e['stored_bytes'] and sha(raw)==e['stored_sha256'];assert b'/Users/'+b'deathcodevision' not in raw,n
 if not e['redacted']:assert e['stored_sha256']==e['original_sha256'] and e['stored_bytes']==e['original_bytes']
original=lambda n:m[n]['original_sha256']
ref=read(w/'accepted-source-reference.json');prior=w/ref['report'];assert sha((prior/'artifacts.json').read_bytes())==ref['manifest_sha256'];pm=read(prior/'artifacts.json')['files'];base=ref['accepted_sources'];assert base==read(prior/'restoration.json')['sources'] and len(base)==42
n=ref['base_metal_source'];raw=(prior/n).read_bytes();assert sha(raw)==pm[n]['stored_sha256']==base['rust-star/runtime/src/metal.rs'];old=raw.decode();edit=read(w/'source-edit.json');assert edit['file']=='rust-star/runtime/src/metal.rs' and edit['count']==2 and edit['before']=='.is_none_or(' and edit['after']=='.map_or(true, ';assert old.count(edit['before'])==2 and edit['after'] not in old;new=old.replace(edit['before'],edit['after']);assert sha(old.encode())==edit['before_sha256'] and sha(new.encode())==edit['after_sha256']
pre=read(w/'preflight.json');f=read(w/'candidate-freeze.json');c=read(w/'completion.json');assert pre['sources']==base and c['passed'] and c['sources']==f['sources'];assert c['sources'][edit['file']]==edit['after_sha256'];assert all(c['sources'][n]==h for n,h in base.items() if n!=edit['file']);assert pre['run_sha256']==original('run.py') and pre['observation_sha256']==original('observation.json');assert f['patch_sha256']==original('source.patch') and f['edit_sha256']==original('source-edit.json');assert c['accepted_performance_executable_sha256']==read(prior/'restoration.json')['executable_sha256'];assert c['all_other_41_sources_unchanged']
expected=['install-toolchain','accepted-check-failure','format','host-tests','release-build','python-contracts','fixture-writer','fixture-reader'];assert [x['name'] for x in c['runs']]==expected
for x in c['runs']:
 assert x==read(w/(x['name']+'-status.json'));assert x['log_sha256']==original(x['name']+'.log');assert x['exit_code']==(101 if x['name']=='accepted-check-failure' else 0)
log=(w/'accepted-check-failure.log').read_text();assert log.count('no method named `is_none_or`')==2 and 'E0599' in log
assert read(w/'toolchain.json')['rustc'].startswith('rustc 1.74.0 (79e9716c9 2023-11-13)');assert '320 passed; 0 failed' in (w/'host-tests.log').read_text();assert 'Finished release' in (w/'release-build.log').read_text() or 'Finished `release`' in (w/'release-build.log').read_text();p=(w/'python-contracts.log').read_text();assert re.search(r'Ran \d+ tests',p) and p.rstrip().endswith('OK')
sys.path.insert(0,str(w/'support'));from artifact_lib import load_logit_artifact,compare_logit_artifacts
fixture=load_logit_artifact(w/'writer-fixture.json');comparison=compare_logit_artifacts(fixture,fixture);assert comparison==read(w/'writer-reader-comparison.json');assert comparison['c0_exact'] and comparison['vocab']==5
assert (w/'support/write_logit_fixture.rs').read_text().count('synthetic-fixture')==1
print(json.dumps(dict(passed=True,artifacts=len(m),rust_version='1.74.0',host_tests=320,python_tests=int(re.search(r'Ran (\d+) tests',p).group(1)),unchanged_other_sources=41,synthetic_fixture_words=5,full_model_or_performance_claim=False),indent=2))
