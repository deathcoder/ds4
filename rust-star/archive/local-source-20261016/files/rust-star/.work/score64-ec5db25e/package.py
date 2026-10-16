from pathlib import Path
import json,hashlib,subprocess,sys
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());b=read(w/'candidate-build.json');p=Path(read(w/'paired-location.json')['work']);study=root/'rust-star/.work/profile128-9b4d7d0d'
assert read(p/'independent-audit.json')['passed'] and read(w/'gates-audit.json')['passed'] and read(w/'final-model-verification.json')['passed']
assert (study/'observation-stop').exists() and read(w/'disposition.json')['optimization_retained']==read(p/'optimization-verdict.json')['accepted']
r=root/'rust-star/reports'/('128k-score64-'+b['commit'][:8]);assert not r.exists();r.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clone(a,b):b.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(a),str(b)],check=True)
def tree(a,b):
 for f in sorted(a.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:clone(f,b/f.relative_to(a))
for name in ['candidate-build.json','baseline.json','tests.log','build.log','python-tests.log','preflight.json','preflight.log','validation-progress.json','validation-acceptance.json','gates-audit.json','final-model-verification.json','disposition.json','freeze-attempt-1.json','optimization-acceptance.json','profile-study.json','freeze.py','validate.py','audit-gates.py','audit-pairs.py','prepare-paired.py','campaign.py','preflight.py','final-input-verification.py','campaign.log']:
 clone(w/name,r/name)
clone(study/'host-observations.jsonl',r/'host-observations.jsonl');clone(study/'observe.py',r/'observe.py')
tree(w/'checks',r/'checks');tree(p/'results',r/'results')
for name in ['acceptance-plan.json','correctness-manifest.json','host-before.json','host-after-pairs.json','freeze.json','baseline-build.json','protocol-scope.json']:clone(p/'inputs'/name,r/'inputs'/name)
clone(root/'rust-star/reports/128k-measure-ef37e566/artifact-hashes.json',r/'inputs/baseline-report-manifest.json')
for name in ['independent-audit.json','independent-summary.json','extended-summary.json','optimization-verdict.json','runner.log']:clone(p/name,r/name)
reference=root/'rust-star/reports/128k-c0-0eb484ea';tree(reference/'reference-131072',r/'reference-131072')
for rep in ['a','b']:clone(reference/f'oracle-131072-{rep}/verification.json',r/f'oracle-131072-{rep}-verification.json')
clone(reference/'capture-acceptance.json',r/'prior-capture-acceptance.json')
for name in ['batch-12k.json','batch-32k-internal.json']:clone(root/'rust-star/reports/32k-memory-de6d2c11/correctness'/name,r/'prior-controls'/name)
for name in ['baseline.json','baseline-profile.log','baseline-profile-status.json','run-baseline-profile.py','diagnostic-build.json','diagnostic-c0.json','diagnostic-profile.json','diagnostic-profile-audit.json','diagnostic-profile-status.json','diagnostic-profile.log','diagnostic-host-tests.log','profile-fence-tests.log','profile-fence-build.log','diagnostic-cli-checks.json','preflight.json','preflight.log','preflight.py','audit-profile.py','run-diagnostic-profile.py','counter-order-probe.m','counter-order-unfenced.jsonl','counter-order-fenced.jsonl','audit-rejected-profile.py','rejected-profile-audit.json','score-tile-probe.m','score-tile-probe-build.log','score-tile-probe.jsonl','score-tile-probe.log','score-tile-baseline.metal','score-tile-candidate.metal','score-tile-combined.metal','score-tile-source-pins.json','score-tile-hypothesis.json','score-tile-audit.json','audit-score-tile.py','probe-commands.json']:
 clone(study/name,r/'profile-study'/name)
clone(w/'report-README.md',r/'README.md');clone(w/'verify-report.py',r/'verify.py')
for build,directory in [(b,'candidate-source'),(read(study/'diagnostic-build.json'),'diagnostic-source')]:
 for name,h in build['sources'].items():
  data=subprocess.check_output(['git','show',build['commit']+':'+name],cwd=root);assert hashlib.sha256(data).hexdigest()==h;t=r/directory/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(data)
for name in subprocess.check_output(['git','ls-tree','-r','--name-only',b['commit'],'rust-star/fixtures/measurement-131072-v5'],cwd=root,text=True).splitlines():
 t=r/'candidate-source'/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(subprocess.check_output(['git','show',b['commit']+':'+name],cwd=root))
(r/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','ec5db25e',b['commit'],'--','rust-star/runtime'],cwd=root))
(r/'diagnostic-implementation.patch').write_bytes(subprocess.check_output(['git','diff','ef37e566','ec5db25e','--','rust-star/runtime'],cwd=root))
if (w/'restoration').exists():tree(w/'restoration',r/'restoration')
redactions={}
for f in r.rglob('*'):
 if not f.is_file() or any(part in f.parts for part in ('candidate-source','diagnostic-source','results')) or f.suffix in ('.bin','.patch'):continue
 data=f.read_bytes()
 try:text=data.decode()
 except UnicodeDecodeError:continue
 redacted=text.replace('/Users/deathcodevision/dev/ds4/gguf/','$MODEL_DIRECTORY/').replace(str(root),'$WORKTREE').replace('/Users/deathcodevision/.rustup/','$RUSTUP/').replace('/Users/deathcodevision/','$USER_DIRECTORY/')
 if redacted!=text:redactions[str(f.relative_to(r))]={'raw_sha256':hashlib.sha256(data).hexdigest(),'stored_sha256':hashlib.sha256(redacted.encode()).hexdigest()};f.write_text(redacted)
(r/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
for name in ['correctness-manifest.json','host-before.json','freeze.json','acceptance-plan.json','baseline-build.json','protocol-scope.json']:assert (r/'inputs'/name).read_bytes()==(p/'inputs'/name).read_bytes(),name
artifacts={str(f.relative_to(r)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(r.rglob('*')) if f.is_file()}
(r/'artifact-hashes.json').write_text(json.dumps(dict(schema='rust-star-artifact-hashes-v1',artifacts=artifacts),indent=2)+'\n')
subprocess.run([sys.executable,str(r/'verify.py')],check=True);print(r)
