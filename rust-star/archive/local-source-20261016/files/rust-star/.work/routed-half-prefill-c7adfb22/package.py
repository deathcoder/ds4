from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];out=root/'rust-star/reports/routed-half-neutral-24c5d03e';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();assert read(w/'execution-complete.json')['passed'] and not read(w/'execution-complete.json')['selected'];assert not (w/'timing').exists();out.mkdir(exist_ok=False);files={}
def emit(n,raw,origin):
 data=raw.replace(str(w).encode(),b'$WORK').replace(str(root).encode(),b'$REPO').replace(b'/Users/deathcodevision',b'$USER');p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);files[n]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=data!=raw)
for p in sorted(w.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 if p.name=='probe':continue
 emit(str(p.relative_to(w)),p.read_bytes(),'completed frozen local study')
for n,h in read(w/'source-audit.json')['sources'].items():
 raw=(root/n).read_bytes();assert sha(raw)==h;emit('sources/'+n,raw,'pinned production source')
prior=root/'rust-star/reports/wide-bootstrap-c7adfb22';emit('host-source-reference.json',(json.dumps(dict(report='../wide-bootstrap-c7adfb22',manifest_sha256=sha((prior/'artifacts.json').read_bytes()),preparation='candidate-sources/rust-star/runtime/src/metal_shim.m',execution='baseline-sources/rust-star/runtime/src/metal_shim.m'),indent=2)+'\n').encode(),'prior immutable full host sources')
emit('binary-identity.json',(json.dumps(dict(sha256=sha((w/'probe').read_bytes()),bytes=(w/'probe').stat().st_size,omitted=True),indent=2)+'\n').encode(),'compiled probe identity')
(out/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-routed-half-neutral-artifacts-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2)+'\n');print(out)
