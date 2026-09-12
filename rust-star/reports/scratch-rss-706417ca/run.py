from pathlib import Path
import json,hashlib,subprocess,datetime,os,time,sys,traceback
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();plan=read(w/'plan.json')
def idle():
 p=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines();assert not [x for x in p if Path(x.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','diagnostic-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server','probe'}]
def sourcepins():
 assert all(sha(r/n)==h for n,h in plan['runtime_state']['sources'].items());assert sha(r/'rust-star/.work/runtime-target/release/rust-star')==plan['runtime_state']['default_executable_sha256']
idle();sourcepins();assert read(w/'source-proof.json')['passed'];assert read(w/'proof-status.json')['exit_code']==0
proof=[json.loads(x[7:]) for x in (w/'proof.log').read_text().splitlines() if x.startswith('RESULT ')];assert proof==[dict(cases=24,checked_bytes=2147549248,guard_bytes=24576,negative_controls=12,passed=True)]
files=['plan.json','inventory.json','probe.m','probe.metal','probe','audit.py','run.py','source-proof.py','source-proof.json','proof.log','proof-status.json'];freeze={n:sha(w/n) for n in files}
with (w/'freeze.json').open('x') as f:json.dump(dict(created_utc=utc(),files=freeze),f,indent=2)
(w/'runs').mkdir();progress=[];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
try:
 for block,order in enumerate(plan['orders'],1):
  for mode in order:
   idle();sourcepins();assert all(sha(w/n)==h for n,h in freeze.items());name=f'{block}-{mode}';d=w/'runs'/name;d.mkdir();start=time.monotonic();started=utc();cmd=['/usr/bin/time','-l',str(w/'probe'),mode,str(w/'inventory.json'),str(w/'probe.metal')]
   observation={}
   for key,c in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage'])]:
    x=subprocess.run(c,capture_output=True,text=True);observation[key]=dict(exit_code=x.returncode,stdout=x.stdout,stderr=x.stderr)
   (d/'host-before.json').write_text(json.dumps(observation,indent=2)+'\n')
   with (d/'run.log').open('x') as log:
    print('START',name,started,flush=True);p=subprocess.run(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
   entry=dict(name=name,mode=mode,block=block,exit_code=p.returncode,started_utc=started,ended_utc=utc(),wall_seconds=time.monotonic()-start,executable_sha256=freeze['probe']);(d/'status.json').write_text(json.dumps(entry,indent=2)+'\n');assert p.returncode==0,entry
   result=[json.loads(x[7:]) for x in (d/'run.log').read_text().splitlines() if x.startswith('RESULT ')];assert len(result)==1 and result[0]['passed'] and result[0]['checked_bytes']==9972692932
   progress.append(entry);(w/'runs/progress.json').write_text(json.dumps(progress,indent=2)+'\n');print('PASS',name,flush=True)
 sourcepins();assert all(sha(w/n)==h for n,h in freeze.items());(w/'runs/final-pins.json').write_text(json.dumps(dict(passed=True,files=freeze,utc=utc()),indent=2)+'\n')
 subprocess.run([sys.executable,str(w/'audit.py')],check=True)
except BaseException:
 (w/'failure.json').write_text(json.dumps(dict(utc=utc(),traceback=traceback.format_exc(),completed=progress),indent=2)+'\n');raise
