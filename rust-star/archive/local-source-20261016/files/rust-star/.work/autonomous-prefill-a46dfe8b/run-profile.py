from pathlib import Path
import json,hashlib,subprocess,datetime,time,os,traceback,re
w=Path(__file__).resolve().parent;root=w.parents[2]
read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=read(w/'plan.json');pins=plan['accepted_pins'];model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');o=root/'rust-star/.work/oracle-v5-source'
try:
 inputs=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip())
 for k,v in inputs.items():assert pins[k]==v,k
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
 sources={n:sha(root/n) for n in pins['sources']};assert sources==pins['sources']
 original=root/'rust-star/.work/attention-output-milestone-08ebd1aa/integration/candidate-rust-star';assert sha(original)==plan['executable_sha256']
 exe=w/'current-rust-star';assert not exe.exists();subprocess.run(['/bin/cp','-c',str(original),str(exe)],check=True);exe.chmod(0o555);assert sha(exe)==plan['executable_sha256']
 (w/'preflight.json').write_text(json.dumps(dict(passed=True,utc=utc(),inputs=inputs,sources=sources,executable_sha256=sha(exe)),indent=2)+'\n')
 tools={k:subprocess.check_output(cmd,text=True) for k,cmd in {'os':['sw_vers'],'machine':['sysctl','hw.model','hw.memsize','hw.physicalcpu','hw.logicalcpu','machdep.cpu.brand_string'],'clang':['xcrun','clang','--version'],'rustc':['rustc','--version'],'cargo':['cargo','--version'],'metal':['xcrun','metal','--version']}.items()}
 (w/'toolchain.json').write_text(json.dumps(tools,indent=2)+'\n')
 frozen={n:sha(w/n) for n in ['plan.json','run-profile.py','audit-profile.py','audit-derivation.json','current-rust-star','preflight.json','toolchain.json']};(w/'freeze.json').write_text(json.dumps(dict(utc=utc(),files=frozen),indent=2)+'\n')
 processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
 assert not [line for line in processes if Path(line.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe']]
 def observe(f,phase):
  row=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
  for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
   p=subprocess.run(cmd,capture_output=True,text=True);row[n]=p.stdout.strip();row[n+'_exit']=p.returncode
  f.write(json.dumps(row)+'\n');f.flush()
 env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};name='current-32k-profiled'
 with (w/(name+'.log')).open('x') as log,(w/'host-observations.jsonl').open('x') as host:
  observe(host,'before');start=time.monotonic();started=utc()
  cmd=['/usr/bin/time','-l',str(exe),'long-prefill-chunk-probe',str(model),'--context','32768','--output-only','--json',str(w/(name+'.json')),'--profile-json',str(w/(name+'-components.json'))]
  p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);(w/'active-run.json').write_text(json.dumps(dict(name=name,pid=p.pid,started_utc=started),indent=2)+'\n');print('START',name,started,flush=True);last=start
  while p.poll() is None:
   time.sleep(1)
   if time.monotonic()-last>=30:observe(host,'active');last=time.monotonic()
  observe(host,'after');status=dict(name=name,exit_code=p.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-start,executable_sha256=sha(exe))
  m=re.search(r'(\d+)\s+maximum resident set size',(w/(name+'.log')).read_text());status['peak_rss_bytes']=int(m[1]) if m else None
  (w/'status.json').write_text(json.dumps(status,indent=2)+'\n');assert p.returncode==0,status
  expected=read(root/'rust-star/reports/attention-output-engine-4ced6ee8/integration/candidate-32k-output.json');assert read(w/(name+'.json'))==expected
  assert (w/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
  for n,h in sources.items():assert sha(root/n)==h,n
  for n,h in frozen.items():assert sha(w/n)==h,n
  (w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,runs=[status],full_fp32_words_per_run=16677120),indent=2)+'\n')
 subprocess.run(['python3',str(w/'audit-profile.py')],check=True)
except BaseException:
 (w/'failure.json').write_text(json.dumps(dict(utc=utc(),traceback=traceback.format_exc()),indent=2)+'\n');raise
