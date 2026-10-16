from pathlib import Path
import json,hashlib,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];old=w.with_name('frontier256-decode-nsg')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def clone(a,b):
 assert not b.exists(),b
 b.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(a),str(b)],check=True)
def tree(a,b):
 for f in sorted(a.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:clone(f,b/f.relative_to(a))
reuse=read(old/'reused-artifact-provenance.json')
for name,info in reuse.items():
 if name.startswith('rejected-'):continue
 p=old/name;assert sha(p)==info['sha256'];clone(p,w/name)
# Preserve the preceding attempt as a distinct rejected evidence set.
rejected=w/'rejected-0d5f50ff';rejected.mkdir()
for name in ['candidate-build.json','failed-attempt.json','validation-progress.json','validation.log','milestone-plan.json','host-current.json','host-provenance.json','host-observations.jsonl','memory-observations.jsonl','observe.py','observe-memory.py','freeze.py','validate.py','tests.log','build.log','python-tests.log','preflight.json','cli-checks.json','source-audit.md','probe-verification.json']:
 clone(old/name,rejected/name)
tree(old/'checks',rejected/'checks');tree(old/'regression',rejected/'regression')
for name,h in read(old/'candidate-build.json')['sources'].items():
 p=rejected/'candidate-source'/name;p.parent.mkdir(parents=True,exist_ok=True)
 data=subprocess.check_output(['git','show',read(old/'candidate-build.json')['commit']+':'+name],cwd=root);assert hashlib.sha256(data).hexdigest()==h;p.write_bytes(data)
(rejected/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','c1047756','0d5f50ff','--','rust-star/runtime'],cwd=root))
tree(old/'rejected-c1047756',w/'rejected-c1047756')
# Existing scripts are templates; identity and report assumptions are updated before use.
for name in ['preflight.py','host-provenance.py','observe.py','observe-memory.py','watch-status.py','validate.py','freeze.py','final-input-verification.py','audit-results.py','render-report.py','package.py','verify-report.py','update-completion-docs.py','source-audit.md']:
 clone(old/name,w/name)
(w/'checks').mkdir();clone(old/'checks/audit-components.py',w/'checks/audit-components.py')
if not (w/'capture-acceptance.json').exists():clone(old/'capture-acceptance.json',w/'capture-acceptance.json')
plan=read(old/'milestone-plan.json');plan['rejected_candidates']=[plan.pop('rejected_candidate'),read(old/'candidate-build.json')['commit']];plan['corrective_scope']='Preserve corrected oracle-v5 decode dispatch; above 128K transfer one layer per completed blit and retire source ownership without detaching the model residency set. Regress actual copy/reset path and rerun all eleven gates.'
(w/'milestone-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(w/'reused-artifact-provenance.json').write_text(json.dumps({n:v for n,v in reuse.items() if not n.startswith('rejected-')},indent=2)+'\n')
print('Workspace prepared; no validation or acceptance claim')
