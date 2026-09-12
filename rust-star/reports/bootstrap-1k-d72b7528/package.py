from pathlib import Path
import json,hashlib,datetime
root=Path.cwd();w=root/'rust-star/.work/bootstrap-1k-current';out=root/'rust-star/reports/bootstrap-1k-d72b7528';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();assert read(w/'decision.json')['closed'];out.mkdir(exist_ok=False);files={};bins={}
def emit(n,raw,origin):
 data=raw.replace(str(w).encode(),b'$WORK').replace(str(root).encode(),b'$REPO').replace(b'$USER',b'$USER');p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);files[n]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=data!=raw)
for p in sorted(w.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 if p.name in ['candidate-rust-star','baseline-rust-star','router-probe']:
  bins[str(p.relative_to(w))]=dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size,omitted=True);continue
 emit(str(p.relative_to(w)),p.read_bytes(),'closed local 1K-bootstrap study')
if (w/'engine/freeze.json').exists():
 for n,h in read(w/'engine/freeze.json')['adapter_files'].items():
  raw=(root/n).read_bytes();assert sha(raw)==h;emit('adapters/'+Path(n).name,raw,'frozen strict measurement adapter')
emit('binary-identities.json',(json.dumps(bins,indent=2)+'\n').encode(),'frozen executable identities')
refs={}
for name in ['msrv-c0-5834fe56','wide-bootstrap-c7adfb22']:
 p=root/'rust-star/reports'/name;refs[name]=dict(report='../'+name,manifest_sha256=sha((p/'artifacts.json').read_bytes()))
emit('prior-report-references.json',(json.dumps(refs,indent=2)+'\n').encode(),'immutable baseline and prior-rejection provenance')
(out/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-1k-bootstrap-report-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2)+'\n');print(out)
