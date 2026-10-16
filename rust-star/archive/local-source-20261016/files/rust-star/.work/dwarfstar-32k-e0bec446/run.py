from pathlib import Path
import json,subprocess,sys,os,time,datetime
from pins import pins,idle,sha
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());plan=read(w/'plan.json');utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert read(w/'input-before.json')['passed'];d=w/'engine';d.mkdir()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def observe(f,phase):
 x=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
 for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
  r=subprocess.run(cmd,capture_output=True,text=True);x[n]=r.stdout;x[n+'_exit']=r.returncode
 f.write(json.dumps(x)+'\n');f.flush()
queue=[('warmup-A','A',None),('warmup-B','B',None)]
for pair,order in enumerate(plan['orders'],1):
 for role in order:queue.append((f'pair-{pair}-{role}',role,pair))
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};records=[]
for name,role,pair in queue:
 idle();run=d/name;run.mkdir();write(run/'preflight.json',pins())
 cmd=[sys.executable,str(w/'adapters'/('measure_dwarfstar.py' if role=='A' else 'measure_ruststar.py')),'--executable',plan['executables'][role],'--model',plan['model'],'--context','32768','--gen-tokens','128','--output',str(run/'measurement'),'--timeout-seconds','1800']
 if role=='A':cmd+=['--prompt',plan['prompt']]
 with (run/'host.jsonl').open('x') as host,(run/'adapter.log').open('x') as log:
  observe(host,'before');start=time.monotonic();started=utc();p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  write(d/'active.json',dict(name=name,pid=p.pid,started_utc=started));print('START',name,started,flush=True);last=start
  while p.poll() is None:
   time.sleep(1)
   if time.monotonic()-last>=30:observe(host,'active');last=time.monotonic()
  observe(host,'after')
 status=dict(name=name,role=role,pair=pair,warmup=pair is None,started_utc=started,finished_utc=utc(),exit_code=p.returncode,adapter_wall_seconds=time.monotonic()-start,executable_sha256=plan['executable_sha256'][role])
 write(run/'status.json',status)
 if p.returncode:
  write(d/'failure.json',status);raise SystemExit(1)
 m=read(run/'measurement/measurement.json');assert m['status']=='passed';status['metrics']=m['metrics'];records.append(status);write(d/'progress.json',records)
 print('PASS',name,'prefill_s',m['metrics']['prefill_ms']/1000,'steady_tps',m['metrics']['gen_steady_tps'],flush=True)
write(d/'final-pins.json',pins());write(d/'complete.json',dict(passed=True,runs=len(records),utc=utc()))
