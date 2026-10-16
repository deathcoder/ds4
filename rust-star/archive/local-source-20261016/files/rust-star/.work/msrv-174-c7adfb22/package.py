from pathlib import Path
import json,hashlib,datetime,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];out=root/'rust-star/reports/msrv-174-179e0c24';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert read(w/'completion.json')['passed'];out.mkdir(exist_ok=False);entries={}
private=[(str(w),'$WORK'),(str(root),'$REPO'),('/Users/deathcodevision','$USER')]
def emit(n,raw,origin):
 text=raw.decode()
 for a,b in private:text=text.replace(a,b)
 data=text.encode();p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);entries[n]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=data!=raw)
for p in sorted(w.iterdir()):
 if p.is_file():emit(p.name,p.read_bytes(),'completed work/'+p.name)
for n in ['artifact_lib.py','compare_logits.py']:emit('support/'+n,(root/'rust-star'/n).read_bytes(),'runtime Python artifact adapter/'+n)
emit('support/write_logit_fixture.rs',(root/'rust-star/runtime/examples/write_logit_fixture.rs').read_bytes(),'runtime fixture writer source')
prior=root/'rust-star/reports/wide-bootstrap-c7adfb22';ref=dict(report='../wide-bootstrap-c7adfb22',manifest_sha256=sha((prior/'artifacts.json').read_bytes()),base_metal_source='baseline-sources/rust-star/runtime/src/metal.rs',accepted_sources=read(prior/'restoration.json')['sources'],source_reference=read(prior/'restoration.json')['source_reference']);emit('accepted-source-reference.json',(json.dumps(ref,indent=2)+'\n').encode(),'previous verified restoration and complete source evidence')
(out/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-msrv-artifacts-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries),indent=2)+'\n');print(out)
