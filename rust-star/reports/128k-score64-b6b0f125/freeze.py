from pathlib import Path
import subprocess,hashlib,json,re,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];study=root/'rust-star/.work/profile128-9b4d7d0d'
assert not subprocess.check_output(['git','diff','HEAD','--','rust-star/runtime'],cwd=root)
assert '317 passed; 0 failed' in (w/'tests.log').read_text()
assert 'Ran 94 tests' in (w/'python-tests.log').read_text() and (w/'python-tests.log').read_text().rstrip().endswith('OK')
assert 'Finished `release` profile' in (w/'build.log').read_text()
code_lines=lambda s:[line.strip() for line in re.sub(r'//[^\n]*','',s).splitlines() if line.strip()]
assert code_lines((root/'rust-star/runtime/src/indexer_scores.metal').read_text())==code_lines((study/'score-tile-candidate.metal').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for source,name in [(root/'rust-star/.work/runtime-target/release/rust-star','candidate-rust-star'),(study/'baseline-rust-star','baseline-rust-star')]:
 assert not (w/name).exists();subprocess.run(['cp','-c',str(source),str(w/name)],check=True)
base=json.loads((study/'baseline.json').read_text());assert sha(w/'baseline-rust-star')==base['executable_sha256'];(w/'baseline.json').write_text(json.dumps(base,indent=2)+'\n')
files=subprocess.check_output(['git','ls-files','rust-star/runtime'],cwd=root,text=True).splitlines()
external=sorted({str((root/'rust-star/runtime'/s).resolve().relative_to(root)) for s in re.findall(r'manifest\.join\("(\.\./\.\./metal/[^\"]+)"\)',(root/'rust-star/runtime/build.rs').read_text())})
assert len(external)==11
info=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),executable_sha256=sha(w/'candidate-rust-star'),sources={n:sha(root/n) for n in files+external},source_manifest_scope='all tracked runtime files plus imported root Metal sources',isolated_score_audit_sha256=sha(study/'score-tile-audit.json'),acceptance_plan_sha256=sha(w/'optimization-acceptance.json'),built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(w/'candidate-build.json').write_text(json.dumps(info,indent=2)+'\n');print('FROZEN',info['commit'],info['executable_sha256'],flush=True)
