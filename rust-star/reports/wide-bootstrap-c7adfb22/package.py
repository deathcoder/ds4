from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[3];out=root/'rust-star/reports/wide-bootstrap-c7adfb22';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert read(w/'engine/audit.json')['execution_passed'] and read(w/'completion-pins.json')['passed'];out.mkdir(exist_ok=False);entries={};bins={}
private=[(str(w),'$WORK'),(str(w.parent),'$CAMPAIGN'),(str(root),'$REPO'),('$MODEL','$MODEL'),('$USER','$USER')]
def emit(n,raw,origin):
 stored=raw.decode()
 for a,b in private:stored=stored.replace(a,b)
 data=stored.encode();p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);entries[n]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=data!=raw)
def copy(p,n):
 raw=p.read_bytes()
 if p.name in ['candidate-rust-star','baseline-rust-star','current-rust-star','router-probe']:
  bins[n]=dict(sha256=sha(raw),bytes=len(raw),reason='Executable omitted; source/build/validation identities retained.');return
 emit(n,raw,'work/'+n)
for p in sorted(w.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:copy(p,str(p.relative_to(w)))
for p in sorted(w.parent.iterdir()):
 if p.is_file():copy(p,'baseline/'+p.name)
for p in sorted((w.parent/'wide-bootstrap').rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:copy(p,'failed-attempt/'+str(p.relative_to(w.parent/'wide-bootstrap')))
build=read(w/'candidate-build.json');base=read(w.parent/'plan.json')['accepted_pins']['sources']
for n in build['sources']:emit('candidate-sources/'+n,subprocess.check_output(['git','show',build['commit']+':'+n],cwd=root),'git/'+build['commit']+'/'+n)
for n in read(w/'source-edits.json'):emit('baseline-sources/'+n,subprocess.check_output(['git','show','4ced6ee8:'+n],cwd=root),'git/4ced6ee8/'+n)
emit('support/expected-12k.json',subprocess.check_output(['git','show','4ced6ee8:rust-star/reports/128k-score64-b6b0f125/prior-controls/batch-12k.json'],cwd=root),'prior independently pinned 12K internal target')
for n in ['ruststar_measurement_lib.py','dwarfstar_measurement_lib.py','artifact_lib.py']:emit('support/'+n,(root/'rust-star'/n).read_bytes(),'adapter/'+n)
def original_hash_code(base):
 return "manifest=json.loads(("+base+"/'artifacts.json').read_text())['files']\ndef sha(p):\n key=str(p.relative_to("+base+"));e=manifest[key]\n assert hashlib.sha256(p.read_bytes()).hexdigest()==e['stored_sha256']\n return e['original_sha256']"
s=(out/'engine-audit.py').read_text().replace('root=w.parents[3]','root=w').replace("sys.path.insert(0,str(root/'rust-star'))","sys.path.insert(0,str(w/'support'))").replace('read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()', 'read=lambda p:json.loads(p.read_text())\n'+original_hash_code('w'))
emit('portable-engine-audit.py',s.encode(),'derived/engine-audit.py: portable adapter import and manifest-verified original hashes')
for prefix in ['', 'baseline/']:
 s=(out/(prefix+'audit-profile.py')).read_text().replace('sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()',original_hash_code('w' if not prefix else 'w.parent'))
 emit(prefix+'portable-audit-profile.py',s.encode(),'derived/'+prefix+'audit-profile.py: manifest-verified original hashes')
emit('binary-identities.json',(json.dumps(bins,indent=2)+'\n').encode(),'generated executable identities')
(out/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-wide-bootstrap-artifacts-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries),indent=2)+'\n');print('Packaged',len(entries),'artifacts',out)
