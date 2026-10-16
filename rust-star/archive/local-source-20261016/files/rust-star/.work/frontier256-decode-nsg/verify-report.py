from pathlib import Path
import json,hashlib,subprocess,sys
sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(r/'artifact-hashes.json')['artifacts'];files={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file() and p.name!='artifact-hashes.json' and '__pycache__' not in p.parts}
assert files==set(a),(files-set(a),set(a)-files)
for n,v in a.items():assert sha(r/n)==v['sha256'] and (r/n).stat().st_size==v['bytes'],n
before=sha(r/'independent-audit.json');subprocess.run([sys.executable,str(r/'audit-results.py'),str(r)],check=True);assert sha(r/'independent-audit.json')==before
final=read(r/'final-model-verification.json');assert final['passed']
for k,v in final.items():
 if k not in ['verified_utc','passed']:assert v==read(r/'preflight.json')[k],k
for n,v in a.items():assert sha(r/n)==v['sha256'],n
print(f'PASS: {len(files)} artifacts; 318 Rust/94 Python tests, repeated 192K/256K oracle references, rejected predecessor and separate oracle GPU regression, eleven corrected native gates, three full 256K output cycles and two-context retained reset/residency validation; no performance claim')
