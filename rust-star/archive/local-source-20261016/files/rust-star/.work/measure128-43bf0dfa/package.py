from pathlib import Path
import json,hashlib,subprocess,sys
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());b=read(w/'candidate-build.json');p=Path(read(w/'paired-location.json')['work'])
assert read(p/'independent-audit.json')['passed'];assert read(w/'gates-audit.json')['passed'];assert read(w/'final-model-verification.json')['passed'];assert (w/'observation-stop').exists()
r=root/'rust-star/reports'/('128k-measure-'+b['commit'][:8]);assert not r.exists();r.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clone(a,b):b.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(a),str(b)],check=True)
def tree(a,b):
 for f in sorted(a.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:clone(f,b/f.relative_to(a))
for name in ['candidate-build.json','tests-final.log','release.log','python-tests.log','preflight.json','validation-progress.json','validation-acceptance.json','gates-audit.json','host-observations.jsonl','final-model-verification.json','build.py','validate.py','audit-gates.py','audit-pairs.py','observe.py','prepare-paired.py','campaign.py','final-input-verification.py','build.log','validate.log','campaign.log']:
 clone(w/name,r/name)
tree(w/'checks',r/'checks');tree(p/'results',r/'results')
for name in ['acceptance-plan.json','correctness-manifest.json','host-before.json','host-after-pairs.json','freeze.json','oracle-metal-hashes.json']:clone(p/'inputs'/name,r/'inputs'/name)
for name in ['independent-audit.json','independent-summary.json','runner.log']:clone(p/name,r/name)
reference=root/'rust-star/reports/128k-c0-0eb484ea';tree(reference/'reference-131072',r/'reference-131072')
for rep in ['a','b']:clone(reference/f'oracle-131072-{rep}/verification.json',r/f'oracle-131072-{rep}-verification.json')
clone(reference/'capture-acceptance.json',r/'prior-capture-acceptance.json')
clone(w/'report-README.md',r/'README.md');clone(w/'verify-report.py',r/'verify.py')
for name,h in b['sources'].items():
 data=subprocess.check_output(['git','show',b['commit']+':'+name],cwd=root);assert hashlib.sha256(data).hexdigest()==h;t=r/'candidate-source'/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(data)
for name in subprocess.check_output(['git','ls-tree','-r','--name-only',b['commit'],'rust-star/fixtures/measurement-131072-v5'],cwd=root,text=True).splitlines():
 t=r/'candidate-source'/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(subprocess.check_output(['git','show',b['commit']+':'+name],cwd=root))
(r/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','43bf0dfa',b['commit'],'--','rust-star/runtime','rust-star/fixtures/measurement-131072-v5','rust-star/tests/test_ruststar_measurement.py'],cwd=root))
# Preserve already-sanitized measurement artifacts byte-for-byte. Redact local
# command/script/build paths elsewhere and retain raw source digests explicitly.
private=str(root);redactions={}
for f in r.rglob('*'):
 if not f.is_file() or 'candidate-source' in f.parts or 'results' in f.parts or f.suffix=='.bin' or f.name in ['implementation.patch']:continue
 data=f.read_bytes()
 try:text=data.decode()
 except UnicodeDecodeError:continue
 redacted=text.replace('/Users/deathcodevision/dev/ds4/gguf/', '$MODEL_DIRECTORY/').replace(private,'$WORKTREE').replace('/Users/deathcodevision/.rustup/', '$RUSTUP/').replace('/Users/deathcodevision/', '$USER_DIRECTORY/')
 if redacted!=text:redactions[str(f.relative_to(r))]={'raw_sha256':hashlib.sha256(data).hexdigest(),'stored_sha256':hashlib.sha256(redacted.encode()).hexdigest()};f.write_text(redacted)
(r/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
# Files with semantic hashes must remain the exact original JSON. These contain
# relative source paths and public identities, rather than private command paths.
for name in ['correctness-manifest.json','host-before.json','freeze.json','acceptance-plan.json']:assert (r/'inputs'/name).read_bytes()==(p/'inputs'/name).read_bytes(),name
artifacts={str(f.relative_to(r)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(r.rglob('*')) if f.is_file()}
(r/'artifact-hashes.json').write_text(json.dumps(dict(schema='rust-star-artifact-hashes-v1',artifacts=artifacts),indent=2)+'\n')
subprocess.run([sys.executable,str(r/'verify.py')],check=True);print(r)
