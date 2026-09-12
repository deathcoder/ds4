from pathlib import Path
import json,hashlib,subprocess,sys,datetime,os,re,statistics
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();manifest=read(w/'artifacts.json')['files'];bins=read(w/'binary-identities.json')
for n,e in manifest.items():
 raw=(w/n).read_bytes();assert len(raw)==e['stored_bytes'] and sha(raw)==e['stored_sha256'],n;assert b'/Users/'+b'deathcodevision' not in raw,n
 if not e['redacted']:assert e['original_sha256']==e['stored_sha256'] and e['original_bytes']==e['stored_bytes'],n
def original(n):return bins[n]['sha256'] if n in bins else manifest[n]['original_sha256']
plan=read(w/'plan.json');baseplan=read(w/'baseline/plan.json');build=read(w/'candidate-build.json');initial=read(w/'baseline/preflight.json');final=read(w/'completion-pins.json')
assert initial['passed'] and final['passed'] and final['inputs']==initial['inputs']
assert final['sources']==build['sources'] and final['candidate_executable_sha256']==build['executable_sha256']==original('candidate-rust-star')
assert final['baseline_executable_sha256']==initial['executable_sha256']==original('baseline-rust-star')==original('baseline/current-rust-star')
assert original('candidate-build.json')==final['candidate_build_sha256']
for n,h in build['files'].items():assert original(n)==h,n
for n,h in read(w/'baseline/freeze.json')['files'].items():assert original('baseline/'+n)==h,n
for n,h in read(w/'engine/freeze.json')['files'].items():assert original(n)==h,n
for n,h in read(w/'engine/freeze.json')['adapter_files'].items():assert original('support/'+Path(n).name)==h,n
ops=read(w/'source-edits.json');base_sources=baseplan['accepted_pins']['sources']
for n,h in build['sources'].items():
 assert original('candidate-sources/'+n)==h,n
 if n not in ops:assert h==base_sources[n]
for n,edits in ops.items():
 s=(w/'candidate-sources'/n).read_text()
 for e in reversed(edits):assert s.count(e['after'])==1,n;s=s.replace(e['after'],e['before'])
 assert s==(w/'baseline-sources'/n).read_text(),n;assert original('baseline-sources/'+n)==base_sources[n]
assert len(build['sources'])==42 and len(ops)==3 and build['unchanged_other_source_count']==39
restored=read(w/'restoration.json');assert restored['passed'] and restored['sources']==base_sources and set(restored['restored_files'])==set(ops) and not restored['retained_candidate'];assert restored['executable_sha256']==original('baseline-rust-star')
# The first attempt remains a recorded rejection. Reconstruct its complete
# host from the candidate by reversing only the four dispatch repairs.
first=read(w/'failed-attempt/build-attempt-1.json');assert first['exit_code']==101
assert "use of undeclared identifier 'raw_cache_target_row'" in (w/'failed-attempt/tests-attempt-1.log').read_text()
for n,edits in read(w/'failed-attempt/source-edits-attempt-1.json').items():
 # Rust was formatted before compilation and then unchanged by the repair.
 # Reconstruct the unformatted C host; use the exact compiled Rust snapshots.
 if n.endswith('.rs'):s=(w/'candidate-sources'/n).read_text()
 else:
  s=(w/'baseline-sources'/n).read_text()
  for e in edits:assert s.count(e['before'])==1,n;s=s.replace(e['before'],e['after'])
 assert sha(s.encode())==first['sources'][n],n
failed=read(w/'failed-attempt/candidate-build.json');oldhost=(w/'candidate-sources/rust-star/runtime/src/metal_shim.m').read_text()
for e in reversed(ops['rust-star/runtime/src/metal_shim.m'][-4:]):assert oldhost.count(e['after'])==1;oldhost=oldhost.replace(e['after'],e['before'])
assert not manifest['candidate-sources/rust-star/runtime/src/metal_shim.m']['redacted'];assert sha(oldhost.encode())==failed['sources']['rust-star/runtime/src/metal_shim.m']
for n,h in failed['files'].items():assert original('failed-attempt/'+n)==h,n
failure=read(w/'failed-attempt/candidate-12k-internal-status.json');assert failure['exit_code']==1 and failure['executable_sha256']==failed['executable_sha256']==original('failed-attempt/candidate-rust-star')
log=(w/'failed-attempt/candidate-12k-internal.log').read_text();assert 'chunk 4096 layer 0 final HC mismatch: actual=0x18a5296a9c248c1c expected=0x8bda50b3f5ec5e19' in log
assert not (w/'failed-attempt/candidate-32k-output-status.json').exists();assert not (w/'failed-attempt/engine').exists()
probe=read(w/'router-probe.json');assert probe==dict(passed=True,hash_max_threads=1024,clamp_max_threads=1024,compared_words=29344,guard_words=96,rows=[32,64,4096],repaired_group=256,rejected_group=4096)
assert read(w/'router-repair-audit.json')['probe_executable_sha256']==original('router-probe')
assert '320 passed; 0 failed' in (w/'tests.log').read_text() and 'Finished `release`' in (w/'build.log').read_text()
accept=read(w/'validation-acceptance.json');assert accept['passed'] and accept['internal_12k'] and len(accept['runs'])==3 and accept['full_fp32_words_total']==33354240
assert read(w/'candidate-12k-internal.json')==read(w/'support/expected-12k.json')
for stage in ['prefix','chunk 4096','chunk 8192']:
 for layer in range(43):assert f'rust-star: {stage} layer {layer} exact' in (w/'candidate-12k-internal.log').read_text()
for n in ['candidate-32k-output','candidate-32k-profiled']:assert read(w/(n+'.json'))==read(w/'baseline/current-32k-profiled.json')
for r in accept['runs']:assert r['exit_code']==0 and r['executable_sha256']==build['executable_sha256'];assert read(w/(r['name']+'-status.json'))==r
# Recompute every profile packet and interval from raw evidence using only
# explicit portability changes to the frozen original audit source.
def hash_code(base):
 return "manifest=json.loads(("+base+"/'artifacts.json').read_text())['files']\ndef sha(p):\n key=str(p.relative_to("+base+"));e=manifest[key]\n assert hashlib.sha256(p.read_bytes()).hexdigest()==e['stored_sha256']\n return e['original_sha256']"
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for prefix in ['', 'baseline/']:
 s=(w/(prefix+'audit-profile.py')).read_text().replace('sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()',hash_code('w' if not prefix else 'w.parent'))
 s=s.replace("\np=w/'profile-audit.json'","\nr=json.loads(json.dumps(r))\np=w/'profile-audit.json'");assert s==(w/(prefix+'portable-audit-profile.py')).read_text();subprocess.run([sys.executable,str(w/(prefix+'portable-audit-profile.py'))],check=True,stdout=subprocess.DEVNULL,env=env)
b=read(w/'baseline/profile-audit.json');c=read(w/'profile-audit.json');sel=read(w/'diagnostic-selection.json');rule=plan['diagnostic_selection'];bt=b['stages'][rule['target']]['gpu_ms'];ct=c['stages'][rule['target']]['gpu_ms'];bc=sum(b['stages'][n]['gpu_ms'] for n in rule['controls']);cc=sum(c['stages'][n]['gpu_ms'] for n in rule['controls']);assert sel['raw_aggregate_ratio']==ct/bt and sel['control_normalized_aggregate_ratio']==(ct/cc)/(bt/bc);assert sel['criteria_pass']==(ct/bt<rule['raw_aggregate_ratio_below'] and (ct/cc)/(bt/bc)<rule['control_normalized_aggregate_ratio_below']);assert sel['criteria_pass']
s=(w/'engine-audit.py').read_text().replace('root=w.parents[3]','root=w').replace("sys.path.insert(0,str(root/'rust-star'))","sys.path.insert(0,str(w/'support'))").replace('read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()','read=lambda p:json.loads(p.read_text())\n'+hash_code('w'));assert s==(w/'portable-engine-audit.py').read_text();subprocess.run([sys.executable,str(w/'portable-engine-audit.py')],check=True,stdout=subprocess.DEVNULL,env=env)
subprocess.run([sys.executable,str(w/'audit-footprint.py')],check=True,stdout=subprocess.DEVNULL,env=env)
subprocess.run([sys.executable,str(w/'audit-phases.py')],check=True,stdout=subprocess.DEVNULL,env=env)
a=read(w/'engine/audit.json');assert a['execution_passed'] and a['raw_runs']==12 and a['timed_pairs']==5
rows=read(w/'engine/progress.json')
for prev,nxt in zip(rows,rows[1:]):assert datetime.datetime.fromisoformat(prev['finished_utc'])<datetime.datetime.fromisoformat(nxt['started_utc'])
for row in rows:
 p=w/'engine'/row['name'];assert read(p/'preflight.json')['source_count']==42;obs=[json.loads(s) for s in (p/'host-observations.jsonl').read_text().splitlines()];assert obs[0]['phase']=='before' and obs[-1]['phase']=='after';assert all(o['phase']=='active' for o in obs[1:-1]);times=[datetime.datetime.fromisoformat(o['utc']) for o in obs];assert times==sorted(times) and times[0]<datetime.datetime.fromisoformat(row['started_utc']) and times[-1]<datetime.datetime.fromisoformat(row['finished_utc']);assert all(o[k+'_exit']==0 for o in obs for k in ['thermal','swap','vm_stat'])
print(json.dumps(dict(passed=True,artifacts=len(manifest),preserved_failed_model_runs=1,router_words=29344,full_model_fp32_words=50031360,engine_runs=12,timed_pairs=5,retained=a['candidate_retained_as_engine_improvement']),indent=2))
