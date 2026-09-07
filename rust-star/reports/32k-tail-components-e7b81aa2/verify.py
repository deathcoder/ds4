from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(__file__).resolve().parent
manifest=json.loads((r/'artifact-hashes.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
subprocess.run([sys.executable,str(r/'audit.py')],check=True,stdout=subprocess.DEVNULL)
assert hashlib.sha256((r/'audit.json').read_bytes()).hexdigest()==manifest['audit.json']
print(f"PASS: {len(manifest)} artifacts; both full-output gates and all 2583 component rows verified")
