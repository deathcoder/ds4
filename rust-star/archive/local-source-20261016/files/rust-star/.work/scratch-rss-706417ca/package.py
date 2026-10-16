from pathlib import Path
import json,hashlib,gzip
w=Path(__file__).resolve().parent;r=w.parents[2];out=r/'rust-star/reports/scratch-rss-706417ca'
read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert read(w/'decision.json')['closed'];out.mkdir(exist_ok=False);entries={};bins={}
def emit(name,data):
 stored=name+'.gz' if len(data)>65536 else name;payload=gzip.compress(data,mtime=0) if stored!=name else data
 dest=out/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
 entries[name]=dict(stored=stored,sha256=sha(data),bytes=len(data),stored_sha256=sha(payload))
for f in sorted(w.rglob('*')):
 if not f.is_file() or '__pycache__' in f.parts or f.name in ['package-execution.log','verify-execution.log','report-manifest.json','milestone-completion.json']:continue
 name=str(f.relative_to(w))
 if name=='probe':bins[name]=dict(sha256=sha(f.read_bytes()),bytes=f.stat().st_size);continue
 emit(name,f.read_bytes())
emit('binary-identities.json',(json.dumps(bins,indent=2)+'\n').encode())
manifest=dict(schema='rust-star-scratch-rss-report-v1',artifacts=entries)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(w/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Packaged',len(entries),'artifacts')
