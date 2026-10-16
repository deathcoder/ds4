from pathlib import Path
import json,subprocess,time
w=Path(__file__).resolve().parent;r=w.parents[2]
while True:
 log=(w/'build-and-validate.log').read_text()
 if 'ALL batch correctness and lifecycle gates passed' in log:break
 if 'Traceback (most recent call last)' in log:raise SystemExit('Build or validation failed; preserve and diagnose')
 time.sleep(30)
for script in ['audit-components.py','prepare-paired.py']:
 print('START',script,flush=True);subprocess.run(['python3','-u',str(w/script)],cwd=r,check=True)
p=Path(json.loads((w/'paired-location.json').read_text())['work'])
with (w/'paired-run.log').open('w') as f:
 runner=subprocess.Popen(['python3','-u',str(p/'run-milestone.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT)
 observer=subprocess.Popen(['python3','-u',str(p/'observe-load.py')],cwd=r,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 code=runner.wait();observer.wait(timeout=60)
 if code:raise SystemExit(code)
for args in [['python3',str(w/'analyze-profile.py'),str(p/'profile32k.json')],['python3',str(p/'audit.py')]]:
 print('START',args,flush=True);subprocess.run(args,cwd=r,check=True)
(w/'measurement-complete.json').write_text(json.dumps({'passed':True,'work':str(p),'remaining':'Assess measured benefit; independently export/verify report; update journal/docs and retain or reject candidate.'},indent=2)+'\n')
print('MEASUREMENTS AND INDEPENDENT AUDIT COMPLETE; retention decision/report pending',flush=True)
