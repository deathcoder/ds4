from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time
w=Path(__file__).resolve().parent;parent=w.parent;root=parent.parents[2]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
f=read(w/'run-freeze.json')
for n,h in f['files'].items():assert sha(w/n)==h,n
for n,h in read(parent/'edge-freeze.json')['files'].items():assert sha(parent/n)==h,n
for n,h in read(parent/'plan.json')['runtime_sources'].items():assert sha(root/n)==h,n
assert read(parent/'cpu-trace.json')['passed']
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
assert not [r for r in processes if Path(r.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server','probe','edge-probe']]
started=time.monotonic();start=utc();env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
with (w/'probe.jsonl').open('x') as out,(w/'run.log').open('x') as log:
 p=subprocess.run(['/usr/bin/time','-l',str(w/'edge-probe'),str(parent/'combined.metal')],stdout=out,stderr=log,env=env,cwd=root)
status=dict(exit_code=p.returncode,started_utc=start,finished_utc=utc(),elapsed_seconds=time.monotonic()-started)
m=re.search(r'(\d+)\s+maximum resident set size',(w/'run.log').read_text());status['peak_rss_bytes']=int(m[1]) if m else None
(w/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if p.returncode:
 (w/'failure.json').write_text(json.dumps(status,indent=2)+'\n');raise SystemExit(p.returncode)
for n,h in read(parent/'edge-freeze.json')['files'].items():assert sha(parent/n)==h,n
subprocess.run(['python3',str(w/'audit.py')],check=True)
