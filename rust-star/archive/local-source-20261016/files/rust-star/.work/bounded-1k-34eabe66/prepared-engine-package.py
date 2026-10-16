from pathlib import Path
import json,hashlib,gzip
w=Path(__file__).resolve().parent;r=w.parents[2];out=r/'rust-star/reports/bounded-1k-34eabe66';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert read(w/'decision.json')['closed'];out.mkdir(exist_ok=False);entries={};bins={}
def emit(name,data):
 stored=name+'.gz' if len(data)>65536 else name;payload=gzip.compress(data,mtime=0) if stored!=name else data;p=out/stored;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload);entries[name]=dict(stored=stored,sha256=sha(data),bytes=len(data),stored_sha256=sha(payload))
for p in sorted(w.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name in ['package-execution.log','verify-execution.log','report-manifest.json','milestone-completion.json']:continue
 name=str(p.relative_to(w))
 if p.name in ['baseline-rust-star','candidate-rust-star','diagnostic-rust-star','prior-default-rust-star','router-probe','observer-proof']:bins[name]=dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size);continue
 emit(name,p.read_bytes())
for n,h in read(w/'engine/freeze.json')['adapter_files'].items():
 data=(r/n).read_bytes();assert sha(data)==h;emit('adapters/'+Path(n).name,data)
emit('oracle-32k.json',(r/'rust-star/reports/msrv-c0-5834fe56/current-32k-normal.json').read_bytes());emit('oracle-12k.json',(r/'rust-star/reports/128k-score64-b6b0f125/prior-controls/batch-12k.json').read_bytes());emit('binary-identities.json',(json.dumps(bins,indent=2)+'\n').encode())
manifest=dict(schema='rust-star-bounded-1k-report-v1',artifacts=entries);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Packaged',len(entries),'artifacts',out)
