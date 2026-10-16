from pathlib import Path
import json,hashlib,tempfile,shutil,subprocess,sys
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')['files']
for n,x in m.items():
 b=(w/n).read_bytes();assert sha(b)==x['stored_sha256'] and len(b)==x['stored_bytes'],n
f=read(w/'freeze.json');c=read(w/'completion.json');assert f['passed'] and c['passed']
for k in ['inputs','sources','fixtures','executable_sha256','model_stat']:assert f[k]==c[k],k
assert f['inputs']['oracle_source']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4';assert f['inputs']['model_sha256']=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
assert len(f['sources'])==42
for n,h in f['sources'].items():assert m['sources/'+n]['original_sha256']==h,n
for n,h in f['files'].items():assert m[n]['original_sha256']==h,n
assert read(w/'binary-identity.json')['sha256']==f['executable_sha256']
br=read(w/'build-reference.json');b=w.parent/br['report'].split('/')[-1];assert sha((b/'artifacts.json').read_bytes())==br['manifest_sha256'];bc=read(b/'completion.json');assert bc['msrv_executable_sha256']==f['executable_sha256'];assert bc['sources']==f['sources']
pr=read(w/'profile-audit-reference.json');old=w.parent/pr['report'].split('/')[-1];assert sha((old/'artifacts.json').read_bytes())==pr['manifest_sha256']
derivation=read(w/'profile-audit-derivation.json');original=(old/'audit-profile.py').read_text();assert sha(original.encode())==derivation['prior_sha256']
for before,after in derivation['edits']:assert before in original;original=original.replace(before,after)
assert original==(w/'audit-profile.py').read_text();assert sha(original.encode())==derivation['current_sha256']
v=read(w/'validation-acceptance.json');assert v['passed'] and len(v['runs'])==2 and v['full_fp32_words_total']==33354240
expected=read(w/'expected-output.json')
for name in ['current-32k-normal','current-32k-profiled']:
 s=read(w/(name+'-status.json'));assert s['exit_code']==0 and s['executable_sha256']==f['executable_sha256'];assert read(w/(name+'.json'))==expected
 assert (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
 assert read(w/(name+'-preflight.json'))['passed']
assert read(w/'final-source-pins.json')['passed'];a=read(w/'profile-audit.json');assert a['passed'] and not a['engine_speedup_claim']
with tempfile.TemporaryDirectory() as d:
 p=Path(d)
 for n in ['plan.json','validation-acceptance.json','current-32k-profiled.json','current-32k-profiled.log','current-32k-profiled-components.json','audit-profile.py']:shutil.copyfile(w/n,p/n)
 run=subprocess.run([sys.executable,str(p/'audit-profile.py')],capture_output=True,text=True);assert run.returncode==0,run.stderr
 actual=read(p/'profile-audit.json');expected_a=json.loads(json.dumps(a))
 for k,n in [('log_sha256','current-32k-profiled.log'),('output_sha256','current-32k-profiled.json'),('profile_sha256','current-32k-profiled-components.json')]:
  assert expected_a[k]==m[n]['original_sha256'];expected_a[k]=m[n]['stored_sha256']
 assert actual==expected_a
print(json.dumps(dict(passed=True,artifacts=len(m),source_files=42,fixture_pins=len(f['fixtures']),full_fp32_words_checked=33354240,profile_intervals=4466,current_rust_174_c0=True,performance_claim=False),indent=2))
