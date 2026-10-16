"""Assemble evidence only after measurements and a human-readable retention decision."""
from pathlib import Path
import subprocess,json,hashlib,shutil,sys
w=Path(__file__).resolve().parent;root=w.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clone(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(a),str(b)],check=True)
def tree(a,b):
 for p in sorted(a.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:clone(p,b/p.relative_to(a))
assert read(w/'measurement-complete.json')['status']=='complete'
assert (w/'retention-decision.json').is_file() and (w/'report-README.md').is_file()
p=Path(read(w/'paired-location.json')['work']);r=root/'rust-star/reports/32k-memory-de6d2c11';assert not r.exists();r.mkdir()
reference=root/'rust-star/reports/32k-dense-output-92ca3ef3/correctness'
clone(reference/'conformance-v5.json',r/'oracle-evidence/conformance-v5.json')
for folder in ['fixtures/prefill-frontier-32768-v5','fixtures/prefill-decode-32768-v5','long-prefill-chunks-v5']:
 tree(reference/folder,r/'oracle-evidence'/folder)
accepted=root/'rust-star/reports/32k-bootstrap-912fae1e'
for name in ['artifact-hashes.json','batch-build.json','independent-audit.json','retention-decision.json']:
 clone(accepted/name,r/'accepted-evidence'/name)
phase=w.with_name('memory-prefill-291eb611')
for name in ['batch-build.json','batch-validation.json','batch-tests-fixed.log','batch-release.log','batch-12k.json','batch-12k.log']:
 clone(phase/name,r/'phase1'/name)
for directory in ['inputs','results']:tree(p/directory,r/directory)
for name in ['independent-audit.json','profile32k.json','profile32k.log','profile32k-analysis.json','profile-failure.json','host-observations.jsonl','load-observations.jsonl','runner.log','milestone.log','observer.log','run-milestone.py','observe-load.py']:
 if (p/name).exists():clone(p/name,r/name)
for name in ['batch-build.json','accepted-build.json','host-audit.json','capacity-probe.json','capacity-provenance.json','capacity-build.log','capacity-probe-run.log','capacity-analysis.json','analyze-capacity.py','analyze-host.py','retention-decision.json','audit-comparison.py','analyze-profile.py','prepare-comparison.py','validate-batch.py','build-and-validate.py','follow-through.py']:
 clone(w/name,r/name)
clone(w/'verify-report.py',r/'verify.py');clone(w/'report-README.md',r/'README.md')
clone(w/'capacity_probe.m',r/'capacity_probe.m')
for name in ['batch-validation.json','batch-tests-fixed.log','batch-release.log','components.json','component-audit.json','audit-components.py']:
 clone(w/name,r/'correctness'/name)
for x in read(w/'batch-validation.json'):
 for ext in ['json','log']:clone(w/(x['name']+'.'+ext),r/'correctness'/(x['name']+'.'+ext))
for name,digest in read(w/'batch-build.json')['sources'].items():
 payload=subprocess.check_output(['git','show',read(w/'batch-build.json')['commit']+':'+name],cwd=root)
 assert hashlib.sha256(payload).hexdigest()==digest
 target=r/'candidate-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
(r/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','912fae1e','de6d2c11','--','rust-star/runtime','rust-star/diagnostics'],cwd=root))
for name in ['run_paired_benchmark.py', 'paired_runner_lib.py', 'paired_benchmark_lib.py', 'measure_ruststar.py', 'ruststar_measurement_lib.py', 'dwarfstar_measurement_lib.py']:
 if (root/'rust-star'/name).exists():clone(root/'rust-star'/name,r/'measurement-tools'/name)
artifacts={str(f.relative_to(r)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(r.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
(r/'artifact-hashes.json').write_text(json.dumps({'schema':'rust-star-artifact-hashes-v1','artifacts':artifacts},indent=2)+'\n')
subprocess.run([sys.executable,str(r/'verify.py')],check=True)
print(r,flush=True)
