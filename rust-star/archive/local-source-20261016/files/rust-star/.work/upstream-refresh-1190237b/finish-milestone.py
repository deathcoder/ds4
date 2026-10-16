from pathlib import Path
import json,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[2]
while True:
 log=(w/'build-and-validate.log').read_text()
 if 'ALL batch correctness and lifecycle gates passed' in log:break
 if 'Traceback (most recent call last)' in log:raise SystemExit('Validation failed; preserve evidence and diagnose before measuring')
 time.sleep(30)
print('ALL live gates passed; preparing frozen paired plan',flush=True)
subprocess.run(['python3','-u',str(w/'prepare-paired.py')],cwd=root,check=True)
paired=Path(json.loads((w/'paired-location.json').read_text())['work'])
print('START timed paired block',paired,flush=True)
with (w/'paired-run.log').open('w') as log:
 runner=subprocess.Popen(['python3','-u',str(paired/'run-milestone.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 observer=subprocess.Popen(['python3','-u',str(paired/'observe-load.py')],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 code=runner.wait();observer.wait(timeout=60)
 if code:raise SystemExit(code)
for args in [ ['python3',str(w/'analyze-profile.py'),str(paired/'profile32k.json')],['python3',str(paired/'audit.py')],['python3',str(w/'export-report.py')],['python3',str(w/'render-report.py')],['python3',str(w/'hash-report.py')]]:
 print('START',args,flush=True);subprocess.run(args,cwd=root,check=True)
report=Path(json.loads((w/'report-location.json').read_text())['path']);subprocess.run(['python3',str(report/'verify.py')],cwd=root,check=True)
(w/'measurement-report-complete.json').write_text(json.dumps({'passed':True,'report':str(report),'remaining':'Update current docs and journal, review final changes, commit.'},indent=2)+'\n')
print('MEASUREMENT AND REPORT COMPLETE',report,flush=True)
