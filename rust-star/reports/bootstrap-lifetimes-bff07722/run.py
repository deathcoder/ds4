from pathlib import Path
import json,hashlib,subprocess,datetime,time,os,re,traceback,sys
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (w/n).open('x') as f:f.write(json.dumps(x,indent=2)+'\n')
plan=read(w/'plan.json');build=read(w/'build.json');ref=read(w/'reference-inputs.json');model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');oracle=r/'rust-star/.work/oracle-v5-source'
def idle():
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','accepted-trace','wide-trace','ds4','ds4-bench','ds4-server'}]
def pins(full=False):
 assert {n:sha(r/n) for n in plan['sources']}==plan['sources']
 assert sha(r/'rust-star/.work/runtime-target/release/rust-star')==plan['default_executable_sha256']
 for role in ['accepted','wide']:assert sha(w/(role+'-trace'))==build['roles'][role]['executable_sha256']
 if (w/'protocol-freeze.json').exists():
  for n,h in read(w/'protocol-freeze.json').items():assert sha(w/n)==h,n
 result=dict(passed=True,utc=utc(),model_stat=dict(size=model.stat().st_size,mtime_ns=model.stat().st_mtime_ns))
 assert result['model_stat']==ref['model_stat']
 if full:
  idle();print('Full input hashes, GPU idle',flush=True)
  inputs=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(r/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(oracle/'ds4-bench'),cli_sha256=sha(oracle/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=oracle,text=True).strip());assert inputs==ref['inputs'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=oracle,text=True).strip()
  fixtures={n:sha(r/n) for n in ref['fixtures']};assert fixtures==ref['fixtures'];result.update(inputs=inputs,fixture_count=len(fixtures),fixture_manifest_sha256=hashlib.sha256(json.dumps(fixtures,sort_keys=True).encode()).hexdigest())
 return result
records=[]
try:
 idle();save('protocol-freeze.json',{p.name:sha(p) for p in w.iterdir() if p.is_file() and p.suffix in {'.py','.inc','.m','.rs','.json'} and p.name!='protocol-freeze.json'});save('input-before.json',pins(True))
 expected=read(r/'rust-star/reports/msrv-c0-5834fe56/current-32k-normal.json')
 for role in ['accepted','wide']:
  idle();save(role+'-preflight.json',pins())
  args=['/usr/bin/time','-l',str(w/(role+'-trace')),'long-prefill-chunk-probe',str(model),'--context','32768','--json',str(w/(role+'-32k.json')),'--output-only'];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
  with (w/(role+'-run.log')).open('x') as log,(w/(role+'-host.jsonl')).open('x') as host:
   started=utc();start=time.monotonic();p=subprocess.Popen(args,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT);print('START',role,p.pid,started,flush=True)
   def observe():
    o=dict(utc=utc(),load=list(os.getloadavg()))
    for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage'])]:
     q=subprocess.run(cmd,capture_output=True,text=True);o[n]=q.stdout.strip();o[n+'_exit']=q.returncode
    host.write(json.dumps(o)+'\n');host.flush()
   observe();last=time.monotonic()
   while p.poll() is None:
    time.sleep(1)
    if time.monotonic()-last>=30:observe();last=time.monotonic()
   observe();status=dict(role=role,exit_code=p.returncode,args=args,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-start);save(role+'-status.json',status);assert p.returncode==0,status
  log=(w/(role+'-run.log')).read_text();assert read(w/(role+'-32k.json'))==expected;assert log.count('all 129280 logits exact, token ')==128;assert 'RS_LIFETIME_ERROR' not in log
  with (w/(role+'-trace.jsonl')).open('x') as f:
   for line in log.splitlines():
    if line.startswith('RS_LIFETIME '):f.write(line[len('RS_LIFETIME '):]+'\n')
  records.append(status);print('PASS full C0',role,flush=True)
 save('input-after.json',pins(True));save('validation.json',dict(passed=True,runs=records,full_fp32_words_total=33354240))
 subprocess.run([sys.executable,str(w/'audit.py')],cwd=r,check=True)
except BaseException:
 save('run-failure.json',dict(utc=utc(),completed=records,traceback=traceback.format_exc()));raise
