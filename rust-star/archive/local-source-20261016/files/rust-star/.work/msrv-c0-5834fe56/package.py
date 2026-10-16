from pathlib import Path
import json,hashlib,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];out=root/'rust-star/reports/msrv-c0-5834fe56';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();assert read(w/'completion.json')['passed'];out.mkdir(exist_ok=False);files={}
def emit(n,raw,origin):
 data=raw.replace(str(w).encode(),b'$WORK').replace(str(root).encode(),b'$REPO').replace(b'/Users/deathcodevision',b'$USER');p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);files[n]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=data!=raw)
for p in sorted(w.rglob('*')):
 if p.is_file() and p.name!='current-rust-star' and '__pycache__' not in p.parts:emit(str(p.relative_to(w)),p.read_bytes(),'completed frozen local full-C0 validation')
for n,h in read(w/'freeze.json')['sources'].items():
 raw=(root/n).read_bytes();assert sha(raw)==h;emit('sources/'+n,raw,'frozen runtime source')
prior=root/'rust-star/reports/msrv-174-179e0c24';emit('build-reference.json',(json.dumps(dict(report='../msrv-174-179e0c24',manifest_sha256=sha((prior/'artifacts.json').read_bytes())),indent=2)+'\n').encode(),'compiler compatibility build provenance')
prior_profile=root/'rust-star/reports/wide-bootstrap-c7adfb22';emit('profile-audit-reference.json',(json.dumps(dict(report='../wide-bootstrap-c7adfb22',manifest_sha256=sha((prior_profile/'artifacts.json').read_bytes())),indent=2)+'\n').encode(),'unchanged component audit provenance')
emit('binary-identity.json',(json.dumps(dict(sha256=sha((w/'current-rust-star').read_bytes()),bytes=(w/'current-rust-star').stat().st_size,omitted=True),indent=2)+'\n').encode(),'frozen Rust 1.74 executable identity')
(out/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-msrv-c0-artifacts-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2)+'\n');print(out)
