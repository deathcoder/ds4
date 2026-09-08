from pathlib import Path
import subprocess,sys,time,json,hashlib,datetime,os
w=Path(__file__).resolve().parent;root=w.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def run(args,name):
 print('START',name,flush=True)
 with (w/(name+'.log')).open('x') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 print('END',name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(f'{name} exited {r.returncode}')
 return r
try:
 while True:
  if not (w/'batch-validation.json').exists():
   time.sleep(45);continue
  checks=read(w/'batch-validation.json')
  if any(x['exit'] for x in checks):raise RuntimeError('Live correctness gate failed')
  if len(checks)==7 and 'ALL batch correctness and lifecycle gates passed' in (w/'build-and-validate.log').read_text():break
  time.sleep(45)
 run([sys.executable,str(w/'audit-components.py')],'component-audit-run')
 source=w/'capacity_probe.m';exe=w/'capacity-probe';output=w/'capacity-probe.json'
 run([str(exe),str(output)],'capacity-probe-run')
 data=read(output);assert data['passed'] and data['context_tokens']==131072
 write(w/'capacity-provenance.json',dict(source_sha256=sha(source),executable_sha256=sha(exe),output_sha256=sha(output),compiler=subprocess.check_output(['clang','--version'],text=True),build_command=['clang','-fobjc-arc','-O2','-framework','Foundation','-framework','Metal','capacity_probe.m','-o','capacity-probe'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 run([sys.executable,str(w/'prepare-comparison.py')],'prepare-comparison')
 p=Path(read(w/'paired-location.json')['work'])
 with (p/'milestone.log').open('x') as log:
  runner=subprocess.Popen([sys.executable,str(p/'run-milestone.py')],stdout=log,stderr=subprocess.STDOUT)
  with (p/'observer.log').open('x') as observer_log:
   observer=subprocess.Popen([sys.executable,str(p/'observe-load.py')],stdout=observer_log,stderr=subprocess.STDOUT)
   code=runner.wait()
   try:observer.wait(timeout=55)
   except subprocess.TimeoutExpired:observer.terminate();observer.wait()
 print('MEASUREMENT/PROFILE EXIT',code,flush=True)
 if not (p/'results/state.json').exists() or read(p/'results/state.json')['phase']!='complete':raise RuntimeError(f'Paired comparison incomplete (exit {code})')
 run([sys.executable,str(w/'audit-comparison.py'),str(p)],'comparison-audit-run')
 if code:
  assert not (p/'profile32k.json').exists()
  write(p/'profile-failure.json',dict(status='failed',exit_code=code,log_sha256=sha(p/'profile32k.log'),profile_artifact_written=False,attempts=1))
 else:run([sys.executable,str(w/'analyze-profile.py'),str(p/'profile32k.json')],'profile-analysis-run')
 write(w/'measurement-complete.json',dict(status='complete',paired_work=str(p),profile_status='passed' if code==0 else 'failed',stop_after_report=True))
 print('Ready for retention decision and report',flush=True)
except Exception as e:
 write(w/'follow-through-failure.json',dict(error=str(e),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 raise
