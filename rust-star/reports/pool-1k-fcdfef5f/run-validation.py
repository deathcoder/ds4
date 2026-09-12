from pathlib import Path
import json,hashlib,subprocess,datetime,time,os,re,traceback
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
build=read(w/'candidate-build.json');model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');exe=w/'candidate-rust-star';initial_stat=model.stat();records=[]
expected12=read(root/'rust-star/reports/128k-score64-b6b0f125/prior-controls/batch-12k.json');expected32=read(root/'rust-star/reports/attention-output-engine-4ced6ee8/integration/candidate-32k-output.json')
def pins():
 assert (model.stat().st_size,model.stat().st_mtime_ns)==(initial_stat.st_size,initial_stat.st_mtime_ns)
 assert sha(exe)==build['executable_sha256']
 for n,h in build['sources'].items():assert sha(root/n)==h,n
 for n,h in build['files'].items():assert sha(w/n)==h,n
 return dict(passed=True,utc=utc(),source_count=len(build['sources']))
def observe(f,phase):
 d=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
 for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
  p=subprocess.run(cmd,capture_output=True,text=True);d[n]=p.stdout.strip();d[n+'_exit']=p.returncode
 f.write(json.dumps(d)+'\n');f.flush()
try:
 for name,ctx,profile in [('candidate-12k-internal',12288,False),('candidate-32k-output',32768,False),('candidate-32k-profiled',32768,True)]:
  (w/(name+'-preflight.json')).write_text(json.dumps(pins(),indent=2)+'\n')
  processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines();assert not [s for s in processes if Path(s.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe']]
  cmd=['/usr/bin/time','-l',str(exe),'long-prefill-chunk-probe',str(model),'--context',str(ctx),'--json',str(w/(name+'.json'))]
  if ctx==32768:cmd+=['--output-only']
  if profile:cmd+=['--profile-json',str(w/(name+'-components.json'))]
  env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
  with (w/(name+'.log')).open('x') as log,(w/(name+'-host.jsonl')).open('x') as host:
   observe(host,'before');started=utc();start=time.monotonic();p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);(w/'active-run.json').write_text(json.dumps(dict(name=name,pid=p.pid,started_utc=started),indent=2)+'\n');print('START',name,started,flush=True);last=start
   while p.poll() is None:
    time.sleep(1)
    if time.monotonic()-last>=30:observe(host,'active');last=time.monotonic()
   observe(host,'after');s=dict(name=name,exit_code=p.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-start,executable_sha256=sha(exe));m=re.search(r'(\d+)\s+maximum resident set size',(w/(name+'.log')).read_text());s['peak_rss_bytes']=int(m[1]) if m else None;(w/(name+'-status.json')).write_text(json.dumps(s,indent=2)+'\n');assert p.returncode==0,s
   assert read(w/(name+'.json'))==(expected12 if ctx==12288 else expected32)
   if ctx==32768:assert (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
   records.append(s);(w/'validation-progress.json').write_text(json.dumps(records,indent=2)+'\n');print('PASS',name,flush=True)
 (w/'final-source-pins.json').write_text(json.dumps(pins(),indent=2)+'\n');(w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,runs=records,full_fp32_words_per_full_run=16677120,full_fp32_words_total=33354240,internal_12k=True),indent=2)+'\n')
 subprocess.run(['python3',str(w/'audit-profile.py')],check=True)
except BaseException:
 (w/'failure.json').write_text(json.dumps(dict(utc=utc(),completed=records,traceback=traceback.format_exc()),indent=2)+'\n');raise
