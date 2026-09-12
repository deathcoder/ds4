from pathlib import Path
import json,hashlib,gzip,shutil
w=Path(__file__).resolve().parent;r=w.parents[2];out=r/'rust-star/reports/bootstrap-lifetimes-bff07722';out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest();entries={}
for p in sorted(w.iterdir()):
 if not p.is_file() or p.name in {'accepted-trace','wide-trace','observer-proof','package-execution.log','verify-execution.log','report-manifest.json'}:continue
 data=p.read_bytes();name=p.name
 if len(data)>65536:
  stored=name+'.gz';payload=gzip.compress(data,mtime=0)
 else:stored=name;payload=data
 (out/stored).write_bytes(payload);entries[name]=dict(stored=stored,sha256=sha(data),bytes=len(data),stored_sha256=sha(payload))
manifest=dict(schema='rust-star-bootstrap-lifetime-report-v1',artifacts=entries)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(w/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Packaged',len(entries),'artifacts in',out)
