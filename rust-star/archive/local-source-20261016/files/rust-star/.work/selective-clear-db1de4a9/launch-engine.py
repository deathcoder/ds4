from pathlib import Path
import json,hashlib,subprocess,sys,datetime,traceback
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
for n in ['mutation-proof-acceptance.json','source-proof.json','ownership-source-audit.json','validation-acceptance.json','lifetime-validation-acceptance.json','profile-audit.json','input-before.json']:assert read(w/n)['passed'],n
assert read(w/'diagnostic-selection.json')['criteria_pass'] and read(w/'diagnostic-selection.json')['elided_checkouts']==112
assert read(w/'pool-proof.log')['passed'] and read(w/'pool-proof.log')['selected_policy_checked_bytes']==1073741824
assert read(w/'engine-plan.json')['retention_criteria']==read(w/'plan.json')['engine_retention_gates']
files=['mutation-proof-acceptance.json','mutation-proof.m','mutation-proof.metal','mutation-proof.log','mutation-proof-status.json','source-proof.json','source-proof.py','audit-lifetime.py','pool-proof.log','ownership-source-audit.json','scope-proof.md','diagnostic-selection.json','lifetime-validation-acceptance.json','validation-acceptance.json','profile-audit.json']
freeze={n:sha(w/n) for n in files}
with (w/'engine-precondition-freeze.json').open('x') as f:json.dump(dict(created_utc=utc(),files=freeze),f,indent=2)
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
