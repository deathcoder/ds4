from pathlib import Path
import subprocess,sys,json,datetime,traceback
w=Path(__file__).resolve().parent;r=w.parents[2]
try:
 for name,args in [('input-before',['full-input-pins.py','input-before.json']),('validation',['run-validation.py']),('lifetime',['run-lifetime.py'])]:
  print('START',name,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
  with (w/(name+'-execution.log')).open('x') as f:p=subprocess.run([sys.executable,str(w/args[0])]+args[1:],cwd=r,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,(name,p.returncode)
  print('PASS',name,flush=True)
except BaseException:
 with (w/'validation-driver-failure.json').open('x') as f:json.dump(dict(traceback=traceback.format_exc()),f,indent=2)
 raise
