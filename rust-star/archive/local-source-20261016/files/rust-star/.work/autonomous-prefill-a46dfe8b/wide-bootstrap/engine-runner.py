from pathlib import Path
import json,datetime,hashlib,subprocess,os,time,re,sys
w=Path(__file__).resolve().parent;root=w.parents[3];d=w/'engine';c=w
sys.path.insert(0,str(root/'rust-star'))
from ruststar_measurement_lib import parse_engine_run
read=lambda p:json.loads(p.read_text())
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=read(w/'engine-plan.json');build=read(c/'candidate-build.json')
assert read(c/'validation-acceptance.json')['passed']
assert read(w/'diagnostic-selection.json')['criteria_pass']
assert sha(w/'engine-plan.json')==build['engine_plan_sha256']
d.mkdir()
freeze=dict(created_utc=utc(),files={p:sha(w/p) for p in ['engine-plan.json','engine-runner.py','engine-audit.py','candidate-build.json','validation-acceptance.json']},adapter_files={p:sha(root/p) for p in ['rust-star/ruststar_measurement_lib.py','rust-star/dwarfstar_measurement_lib.py','rust-star/artifact_lib.py']})
(d/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');model_stat=model.stat()
exe={'A':c/'baseline-rust-star','B':c/'candidate-rust-star'}
expected={'A':plan['baseline_executable_sha256'],'B':build['executable_sha256']}
def pins():
 assert (model.stat().st_size,model.stat().st_mtime_ns)==(model_stat.st_size,model_stat.st_mtime_ns)
 for p,h in freeze['files'].items():assert sha(w/p)==h,p
 for p,h in freeze['adapter_files'].items():assert sha(root/p)==h,p
 for p,h in build['sources'].items():assert sha(root/p)==h,p
 for role,h in expected.items():assert sha(exe[role])==h,role
 return dict(passed=True,utc=utc(),source_count=len(build['sources']))
def observe(f,phase):
 record=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
 for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
  p=subprocess.run(cmd,capture_output=True,text=True);record[n]=p.stdout.strip();record[n+'_exit']=p.returncode
 f.write(json.dumps(record)+'\n');f.flush()
queue=[('warmup-A','A',None),('warmup-B','B',None)]
for pair,order in enumerate(plan['orders'],1):
 for role in order:queue.append((f'pair-{pair}-{role}',role,pair))
records=[]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
for name,role,pair in queue:
 run=d/name;run.mkdir();(run/'preflight.json').write_text(json.dumps(pins(),indent=2)+'\n')
 processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
 assert not [r for r in processes if Path(r.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe']]
 with (run/'host-observations.jsonl').open('x') as host,(run/'run.log').open('x') as log:
  observe(host,'before');start=time.monotonic();started=utc()
  cmd=['/usr/bin/time','-l',str(exe[role]),'engine-measure',str(model),'--context','32768','--gen-tokens','128','--json',str(run/'engine-run.json')]
  p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  (d/'active-run.json').write_text(json.dumps(dict(name=name,pid=p.pid,started_utc=started),indent=2)+'\n');print('START',name,started,flush=True)
  last=start
  while p.poll() is None:
   time.sleep(1)
   if time.monotonic()-last>=30:observe(host,'active');last=time.monotonic()
  observe(host,'after')
  status=dict(name=name,role=role,pair=pair,order=plan['orders'][pair-1] if pair else None,warmup=pair is None,exit_code=p.returncode,started_utc=started,finished_utc=utc(),process_wall_seconds=time.monotonic()-start,executable_sha256=expected[role])
  match=re.search(r'(\d+)\s+maximum resident set size',(run/'run.log').read_text());status['peak_rss_bytes']=int(match[1]) if match else None
  (run/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if p.returncode or not match:
   (d/'failure.json').write_text(json.dumps(status,indent=2)+'\n');raise SystemExit(1)
  try:metrics=parse_engine_run(run/'engine-run.json',context=32768,gen_tokens=128)
  except Exception as e:
   (d/'failure.json').write_text(json.dumps(dict(**status,error=str(e)),indent=2)+'\n');raise
  (run/'parsed-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
  records.append(dict(**status,metrics=metrics));(d/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
  print('PASS',name,'prefill_s',metrics['prefill_ms']/1000,'steady_tps',metrics['gen_steady_tps'],flush=True)
(d/'final-pins.json').write_text(json.dumps(pins(),indent=2)+'\n')
(d/'execution-complete.json').write_text(json.dumps(dict(passed=True,runs=len(records),finished_utc=utc()),indent=2)+'\n')
subprocess.run([sys.executable,str(w/'engine-audit.py')],check=True)
