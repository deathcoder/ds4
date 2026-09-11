from pathlib import Path
import json,hashlib,subprocess,datetime,time,os
w=Path(__file__).resolve().parent;r=w.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
for n,h in read(w/'edge-freeze.json')['files'].items():assert sha(w/n)==h,n
for n,h in read(w/'plan.json')['runtime_sources'].items():assert sha(r/n)==h,n
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
assert not [x for x in processes if Path(x.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server','probe']]
d=w/'edges';d.mkdir();begin=time.monotonic();started=utc();env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
with (d/'probe.jsonl').open('x') as out,(d/'run.log').open('x') as log:p=subprocess.run(['/usr/bin/time','-l',str(w/'probe'),str(w/'combined.metal'),'edges','0'],cwd=r,stdout=out,stderr=log,env=env)
status=dict(exit_code=p.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-begin)
(d/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(status)
if p.returncode:raise SystemExit(p.returncode)
subprocess.run(['python3',str(w/'edge-audit.py')],check=True)
