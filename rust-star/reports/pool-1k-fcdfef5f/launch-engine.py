from pathlib import Path
import json,hashlib,subprocess,sys,datetime,traceback
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert read(w/'diagnostic-selection.json')['criteria_pass'];assert read(w/'lifetime-validation-acceptance.json')['passed'];assert read(w/'validation-acceptance.json')['passed'];assert read(w/'profile-audit.json')['passed'];assert read(w/'pool-proof.log')['passed'];assert read(w/'ownership-source-audit.json')['passed']
a=read(w/'audit-representation-amendment.json');assert (w/'audit-lifetime.py').read_text().replace(a['before'],a['after'])==(w/'audit-pool-v2.py').read_text();assert not a['runtime_changes'] and not a['observer_changes'] and not a['capacity_or_performance_threshold_changes']
files=['diagnostic-selection.json','audit-pool-v2.py','audit-lifetime.py','audit-representation-amendment.json','status-representation.log','pool-proof.log','ownership-source-audit.json','scope-proof.md','lifetime-validation-acceptance.json','profile-audit.json','lifetime-failure.json','validation-driver-failure.json']
freeze={n:sha(w/n) for n in files}
with (w/'engine-precondition-freeze.json').open('x') as f:json.dump(dict(created_utc=utc(),files=freeze,pool_auditor='audit-pool-v2.py',representation_correction_only=True),f,indent=2)
try:
 with (w/'engine-execution.log').open('x') as f:result=subprocess.run([sys.executable,str(w/'engine-runner.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT)
 assert result.returncode==0,result.returncode
 assert all(sha(w/n)==h for n,h in freeze.items())
 with (w/'phase-audit-execution.log').open('x') as f:result=subprocess.run([sys.executable,str(w/'audit-phases.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT)
 assert result.returncode==0
 with (w/'engine-precondition-final.json').open('x') as f:json.dump(dict(passed=True,utc=utc(),files=freeze),f,indent=2)
 print('Complete12 engine runs and passive phase audit',flush=True)
except BaseException:
 with (w/'engine-launch-failure.json').open('x') as f:json.dump(dict(utc=utc(),traceback=traceback.format_exc()),f,indent=2)
 raise
