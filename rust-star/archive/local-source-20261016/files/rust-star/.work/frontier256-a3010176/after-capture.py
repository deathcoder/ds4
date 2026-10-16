from pathlib import Path
import json,subprocess,sys,time
w=Path(__file__).resolve().parent
assert json.loads((w/'candidate-build.json').read_text())['commit']=='c1047756a9186b1c60b6ee5dd323ae81ffbbabc3'
while not (w/'capture-acceptance.json').exists():
 if (w/'capture-progress.json').exists():
  assert all(x['exit']==0 for x in json.loads((w/'capture-progress.json').read_text())), 'oracle process failed'
 if 'Traceback' in (w/'capture.log').read_text():raise RuntimeError('oracle capture stopped; preserve and inspect capture.log')
 time.sleep(45)
assert json.loads((w/'capture-acceptance.json').read_text())['passed']
with (w/'validation.log').open('x') as log:
 subprocess.run([sys.executable,str(w/'validate.py')],cwd=w.parents[2],stdout=log,stderr=subprocess.STDOUT,check=True)
print('VALIDATION COMPLETE',flush=True)
