from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[2]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
freeze=read(w/'freeze.json');plan=read(w/'plan.json')
assert read(w/'edge-audit.json')['passed']
def pins():
 for n,h in freeze['files'].items():assert sha(w/n)==h,n
 for n,h in plan['runtime_sources'].items():assert sha(root/n)==h,n
 return dict(passed=True,utc=utc(),frozen_files=len(freeze['files']),runtime_sources=len(plan['runtime_sources']))
def observe(f,phase):
 row=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
 for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
  s=subprocess.run(cmd,capture_output=True,text=True);row[n]=s.stdout.strip();row[n+'_exit']=s.returncode
 f.write(json.dumps(row)+'\n');f.flush()
for session in plan['sessions']:
 verified=pins();d=w/f'session-{session}';d.mkdir()
 processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
 assert not [r for r in processes if Path(r.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server','probe','edge-probe']]
 (d/'preflight.json').write_text(json.dumps(verified,indent=2)+'\n')
 with (d/'host-observations.jsonl').open('x') as host,(d/'probe.jsonl').open('x') as out,(d/'run.log').open('x') as log:
  observe(host,'before');started=time.monotonic();start_utc=utc()
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
  p=subprocess.Popen(['/usr/bin/time','-l',str(w/'probe'),str(w/'combined.metal'),'timing',str(session)],cwd=root,stdout=out,stderr=log,env=env)
  print(f'START session-{session} pid={p.pid} {start_utc}',flush=True);last=started
  while p.poll() is None:
   time.sleep(1)
   if time.monotonic()-last>=30:observe(host,'active');last=time.monotonic()
  observe(host,'after')
  status=dict(exit_code=p.returncode,started_utc=start_utc,finished_utc=utc(),elapsed_seconds=time.monotonic()-started)
  match=re.search(r'(\d+)\s+maximum resident set size',(d/'run.log').read_text());status['peak_rss_bytes']=int(match[1]) if match else None
  (d/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
  if p.returncode:
   (d/'failure.json').write_text(json.dumps(status,indent=2)+'\n');raise SystemExit(p.returncode)
 assert json.loads((d/'probe.jsonl').read_text().splitlines()[-1])['passed']
 (d/'final-pins.json').write_text(json.dumps(pins(),indent=2)+'\n')
subprocess.run(['python3',str(w/'audit.py')],check=True)
