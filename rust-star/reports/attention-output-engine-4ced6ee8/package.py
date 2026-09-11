from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];out=root/'rust-star/reports/attention-output-engine-4ced6ee8'
read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert read(w/'engine/audit.json')['execution_passed']
assert read(w/'completion-pins.json')['passed']
out.mkdir(exist_ok=False);entries={};omitted={}
private=[(str(w),'$WORK'),(str(root),'$REPO'),('$MODEL','$MODEL'),('$USER','$USER')]
def emit(name,raw,origin):
 text=raw.decode();stored=text
 for a,b in private:stored=stored.replace(a,b)
 data=stored.encode();p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 entries[name]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=raw!=data)
exclude={'probe','integration/compile-shader','integration/candidate-rust-star','integration/baseline-rust-star'}
for p in sorted(w.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 n=str(p.relative_to(w))
 if n in exclude:
  raw=p.read_bytes();omitted[n]=dict(sha256=sha(raw),bytes=len(raw),reason='Executable omitted; source, build and validation evidence retained.');continue
 emit(n,p.read_bytes(),'work/'+n)
for n in read(w/'integration/candidate-build.json')['sources']:
 raw=subprocess.check_output(['git','show','4ced6ee8:'+n],cwd=root);emit('candidate-sources/'+n,raw,'git/4ced6ee8/'+n)
for n in ['rust-star/runtime/src/metal_shim.m','rust-star/runtime/build.rs']:
 emit('baseline-sources/'+n,subprocess.check_output(['git','show','08ebd1aa:'+n],cwd=root),'git/08ebd1aa/'+n)
prev=root/'rust-star/.work/attention-output-half-input-038dded4'
for n in ['plan.json','audit.json','prepare.py','production-encoding.m','production-arguments.m']:
 emit('prior-screen/'+n,(prev/n).read_bytes(),'prior-screen/'+n)
for n in ['ruststar_measurement_lib.py','dwarfstar_measurement_lib.py','artifact_lib.py']:
 emit('support/'+n,(root/'rust-star'/n).read_bytes(),'adapter/'+n)
# Keep frozen engine-audit.py verbatim except ordinary path redaction. This
# separate portable derivative changes only adapter import and raw-hash lookup.
s=(out/'engine-audit.py').read_text()
s=s.replace("sys.path.insert(0,str(root/'rust-star'))","sys.path.insert(0,str(w/'support'))")
s=s.replace("read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()", "read=lambda p:json.loads(p.read_text())\nmanifest=read(w/'artifacts.json')['files']\ndef sha(p):\n key=str(p.relative_to(w));entry=manifest[key]\n assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['stored_sha256']\n return entry['original_sha256']")
emit('portable-engine-audit.py',s.encode(),'derived/engine-audit.py with portable adapter path and manifest-verified original hashes')
emit('binary-identities.json',(json.dumps(omitted,indent=2)+'\n').encode(),'generated executable identities')
(out/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-attention-output-artifacts-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries),indent=2)+'\n')
print('Packaged',len(entries),'artifacts',out)
