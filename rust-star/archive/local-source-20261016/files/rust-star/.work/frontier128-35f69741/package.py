from pathlib import Path
import subprocess,json,hashlib,sys
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(w/'candidate-build.json');assert read(w/'validation-acceptance.json')['passed'];r=root/'rust-star/reports'/('128k-c0-'+b['commit'][:8]);assert not r.exists();r.mkdir()
def clone(a,b):b.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(a),str(b)],check=True)
def tree(a,b):
 for f in sorted(a.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:clone(f,b/f.relative_to(a))
for name in ['inputs.json','candidate-build.json','tests.log','tests-final.log','release.log','capture-progress.json','capture-acceptance.json','validation-progress.json','validation-acceptance.json','capture.py','validate.py','build.py','observe.py','capture.log','validate.log','host-observations.jsonl','final-model-verification.json','final-input-verification.py','source-audit.md']:
 clone(w/name,r/name)
for n in [65536,98304,131072]:
 tree(w/f'reference-{n}',r/f'reference-{n}')
 for rep in ['a','b']:
  d=w/f'oracle-{n}-{rep}'
  for name in ['console.log','benchmark.csv','all-logits.f32le.bin','selected.u32le.bin','verification.json']:clone(d/name,r/d.name/name)
tree(w/'checks',r/'checks');clone(w/'verify-report.py',r/'verify.py');clone(w/'report-README.md',r/'README.md')
for name,h in b['sources'].items():
 data=subprocess.check_output(['git','show',b['commit']+':'+name],cwd=root);assert hashlib.sha256(data).hexdigest()==h;p=r/'candidate-source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
(r/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','35f69741',b['commit'],'--','rust-star/runtime'],cwd=root))
artifacts={str(p.relative_to(r)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(r.rglob('*')) if p.is_file()};(r/'artifact-hashes.json').write_text(json.dumps(dict(schema='rust-star-artifact-hashes-v1',artifacts=artifacts),indent=2)+'\n')
subprocess.run([sys.executable,str(r/'verify.py')],check=True);print(r)
