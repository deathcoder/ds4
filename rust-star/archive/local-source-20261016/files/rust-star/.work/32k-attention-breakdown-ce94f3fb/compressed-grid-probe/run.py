from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[3]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(w.parent/'audit.json')['passed'] and read(w.parent/'final-pins.json')['passed']
for n,h in read(w/'freeze.json')['files'].items():assert sha(w/n)==h,n
for p,h in read(w.parent/'candidate-build.json')['sources'].items():assert sha(root/p)==h,p
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
assert not [r for r in processes if Path(r.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server','probe']]
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def observe(f,phase):
 r=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
 for name,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
  x=subprocess.run(cmd,capture_output=True,text=True);r[name]=x.stdout.strip();r[name+'_exit']=x.returncode
 f.write(json.dumps(r)+'\n');f.flush()
with (w/'host-observations.jsonl').open('x') as host,(w/'probe.jsonl').open('x') as out,(w/'run.log').open('x') as log:
 observe(host,'before');started=time.monotonic();start_utc=utc()
 env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
 p=subprocess.Popen(['/usr/bin/time','-l',str(w/'probe'),str(w/'combined.metal')],cwd=root,stdout=out,stderr=log,env=env)
 (w/'active-run.json').write_text(json.dumps(dict(pid=p.pid,started_utc=start_utc),indent=2)+'\n')
 last=time.monotonic()
 while p.poll() is None:
  time.sleep(1)
  if time.monotonic()-last>=30:observe(host,'active');last=time.monotonic()
 observe(host,'after')
 r=dict(exit_code=p.returncode,started_utc=start_utc,finished_utc=utc(),elapsed_seconds=time.monotonic()-started)
 rss=re.search(r'(\d+)\s+maximum resident set size',(w/'run.log').read_text());r['peak_rss_bytes']=int(rss[1]) if rss else None
 (w/'status.json').write_text(json.dumps(r,indent=2)+'\n')
 if p.returncode:
  (w/'failure.json').write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(p.returncode)
assert json.loads((w/'probe.jsonl').read_text().splitlines()[-1])['passed']
print(json.dumps(r,indent=2))
