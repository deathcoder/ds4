from pathlib import Path
import json,hashlib,gzip,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];dest=root/'rust-star/reports/dwarfstar-32k-e0bec446';dest.mkdir(exist_ok=False)
artifacts={}
for p in sorted(w.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name in ['verify.py','package.py','README.md','manifest.json']:continue
 n=str(p.relative_to(w));data=p.read_bytes();compressed=len(data)>65536;stored='evidence/'+n+('.gz' if compressed else '');payload=gzip.compress(data,mtime=0) if compressed else data
 target=dest/stored;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
 artifacts[n]=dict(stored=stored,compressed=compressed,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),stored_sha256=hashlib.sha256(payload).hexdigest())
(dest/'manifest.json').write_text(json.dumps(dict(schema='rust-star-portable-reference-comparison-v1',artifacts=artifacts),indent=2)+'\n')
for n in ['verify.py','README.md']:shutil.copy2(w/n,dest/n)
print(dest,len(artifacts))
