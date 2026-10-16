from pathlib import Path
import json,hashlib,subprocess,sys
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());b=read(w/'candidate-build.json')
assert read(w/'independent-audit.json')['passed'] and read(w/'final-model-verification.json')['passed'] and (w/'observation-stop').exists()
r=root/'rust-star/reports'/('256k-c0-'+b['commit'][:8]);assert not r.exists();r.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clone(a,b):b.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(a),str(b)],check=True)
def tree(a,b):
 for f in sorted(a.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:clone(f,b/f.relative_to(a))
for name in ['milestone-plan.json','candidate-build.json','inputs.json','preflight.json','preflight.log','capacity-probe.json','capacity-analysis.json','capacity-provenance.json','capacity_probe.m','source-audit.md','tests.log','build.log','python-tests.log','cli-checks.json','host-current.json','host-provenance.json','host-observations.jsonl','observe-memory.py','memory-observations.jsonl','memory-initial-snapshot.json','memory-initial-snapshot.txt','capture-progress.json','capture-acceptance.json','capture.log','validation-progress.json','validation-acceptance.json','validation.log','independent-audit.json','final-model-verification.json','capture.py','validate.py','freeze.py','observe.py','preflight.py','host-provenance.py','after-capture.py','after-capture.log','audit-results.py','final-input-verification.py','render-report.py']:
 clone(w/name,r/name)
for n in [196608,262144]:
 tree(w/f'reference-{n}',r/f'reference-{n}')
 for rep in ['a','b']:
  for name in ['verification.json','selected.u32le.bin','all-logits.f32le.bin','benchmark.csv','console.log']:clone(w/f'oracle-{n}-{rep}'/name,r/f'oracle-{n}-{rep}'/name)
ref=root/'rust-star/reports/128k-c0-0eb484ea';tree(ref/'reference-131072',r/'reference-131072')
for rep in ['a','b']:clone(ref/f'oracle-131072-{rep}/verification.json',r/f'oracle-131072-{rep}-verification.json')
for name in ['batch-12k.json','batch-32k-internal.json']:clone(root/'rust-star/reports/32k-memory-de6d2c11/correctness'/name,r/'prior-controls'/name)
clone(root/'rust-star/reports/128k-score64-b6b0f125/inputs/host-before.json',r/'prior-host.json')
clone(root/'rust-star/reports/128k-score64-b6b0f125/results/paired-raw.json',r/'prior-128k-paired-raw.json')
clone(root/'rust-star/reports/128k-score64-b6b0f125/artifact-hashes.json',r/'prior-128k-artifact-hashes.json')
tree(w/'checks',r/'checks')
for name,h in b['sources'].items():
 data=subprocess.check_output(['git','show',b['commit']+':'+name],cwd=root);assert hashlib.sha256(data).hexdigest()==h;t=r/'candidate-source'/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(data)
(r/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','a3010176',b['commit'],'--','rust-star/runtime'],cwd=root))
clone(w/'report-README.md',r/'README.md');clone(w/'verify-report.py',r/'verify.py')
redactions={}
for f in r.rglob('*'):
 if not f.is_file() or 'candidate-source' in f.parts or f.suffix in ('.bin','.patch'):continue
 data=f.read_bytes()
 try:text=data.decode()
 except UnicodeDecodeError:continue
 redacted=text.replace('/Users/deathcodevision/dev/ds4/gguf/','$MODEL_DIRECTORY/').replace(str(root),'$WORKTREE').replace('/Users/deathcodevision/.rustup/','$RUSTUP/').replace('/Users/deathcodevision/','$USER_DIRECTORY/')
 if redacted!=text:redactions[str(f.relative_to(r))]={'raw_sha256':hashlib.sha256(data).hexdigest(),'stored_sha256':hashlib.sha256(redacted.encode()).hexdigest()};f.write_text(redacted)
(r/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
artifacts={str(f.relative_to(r)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(r.rglob('*')) if f.is_file()}
(r/'artifact-hashes.json').write_text(json.dumps({'schema':'rust-star-artifact-hashes-v1','artifacts':artifacts},indent=2)+'\n')
subprocess.run([sys.executable,str(r/'verify.py')],check=True);print(r)
