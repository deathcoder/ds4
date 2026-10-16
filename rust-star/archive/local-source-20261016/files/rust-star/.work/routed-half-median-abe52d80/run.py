from pathlib import Path
import json,hashlib,subprocess,datetime,os,time,traceback,sys
w=Path(__file__).resolve().parent;root=w.parents[2];prior=root/'rust-star/.work/routed-half-prefill-c7adfb22';read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def no_gpu():
 names={'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe','router-probe'}
 ps=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines();assert not [s for s in ps if Path(s.strip().split(maxsplit=1)[-1]).name in names]
def host(f,phase):
 row=dict(utc=utc(),phase=phase,load=list(os.getloadavg()))
 for n,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
  p=subprocess.run(cmd,capture_output=True,text=True);row[n]=p.stdout.strip();row[n+'_exit']=p.returncode
 f.write(json.dumps(row)+'\n');f.flush()
def host_fragments():
 s=(root/'rust-star/runtime/src/metal_shim.m').read_text();s=s[s.index('int rust_star_metal_run_prefill_layer2_continuation_tail('):];a=s.index('        rust_star_q8_mm_id_map_args routed_map_args =');b=s.index('        rust_star_q8_mm_id_args routed_down_args =',a);assert s[a:b]==(w/'production-arguments.m').read_text();a=s.index('        [encoder setComputePipelineState:context.routedBatchMapPipeline]');b=s.index('        [encoder setComputePipelineState:context.routedBatchDownPipeline]',a);assert s[a:b]==(w/'production-encoding.m').read_text()
# This is a hard scheduling gate, not a timer: finish and close the prior
# whole-engine decision before compiling or dispatching this next experiment.
assert read(prior/'milestone-completion.json')['passed']
assert read(prior/'execution-complete.json')['passed'] and not read(prior/'execution-complete.json')['selected']
assert not (prior/'timing').exists()
no_gpu();source=read(w/'source-audit.json');host_fragments()
for n,h in source['sources'].items():assert sha(root/n)==h,n
assert sha(w/'candidate.metal')==source['candidate_sha256'] and sha(w/'combined.metal')==source['combined_sha256']
code=(w/'candidate.metal').read_text()
for e in reversed(read(w/'source-edits.json')):assert code.count(e['after'])==e['count'];code=code.replace(e['after'],e['before'])
assert code==(root/'rust-star/runtime/src/routed_prefill.metal').read_text()
cmd=['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'probe.m'),'-o',str(w/'probe')]
with (w/'build.log').open('x') as f:
 status=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT).returncode
(w/'build-status.json').write_text(json.dumps(dict(command=cmd,exit_code=status,utc=utc()),indent=2)+'\n');assert status==0
(w/'toolchain.json').write_text(json.dumps({n:subprocess.check_output(c,text=True) for n,c in {'os':['sw_vers'],'clang':['clang','--version'],'metal':['xcrun','metal','--version'],'machine':['sysctl','hw.model','hw.memsize','machdep.cpu.brand_string']}.items()},indent=2)+'\n')
files=['plan.json','prepare.py','prepare-audits.py','prepare-protocol.py','protocol-source-edits.json','protocol-derivation.json','probe.m','probe','source-edits.json','source-audit.json','candidate.metal','combined.metal','arguments.h','production-arguments.m','production-encoding.m','audit.py','audit-derivation.json','audit-edges.py','run.py','build.log','build-status.json','toolchain.json']
freeze=dict(utc=utc(),accepted_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),accepted_host_sha256=sha(root/'rust-star/runtime/src/metal_shim.m'),prior_milestone_sha256=sha(prior/'milestone-completion.json'),sources=source['sources'],files={n:sha(w/n) for n in files});(w/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
def pins():
 for n,h in freeze['files'].items():assert sha(w/n)==h,n
 for n,h in freeze['sources'].items():assert sha(root/n)==h,n
 assert sha(root/'rust-star/runtime/src/metal_shim.m')==freeze['accepted_host_sha256'];host_fragments();return dict(passed=True,utc=utc(),source_count=len(freeze['sources']),files=len(freeze['files']))
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};progress=[]
try:
 for mode in ['edges','aa-only','timing']:
  if mode!='edges':assert read(w/'edge-audit.json')['passed']
  if mode=='timing':assert read(w/'aa-only/audit.json')['contemporary_null_screen_pass']
  d=w/mode;d.mkdir();phase={n:sha(w/n) for n in (['edge-audit.json','aa-only/audit.json'] if mode=='timing' else ['edge-audit.json'] if mode=='aa-only' else [])};(d/'phase-freeze.json').write_text(json.dumps(phase,indent=2)+'\n')
  for session in ([0] if mode=='edges' else [0,1]):
   no_gpu();p=d if mode=='edges' else d/f'session-{session}';p.mkdir(exist_ok=mode=='edges');(p/'preflight.json').write_text(json.dumps(pins(),indent=2)+'\n')
   for n,h in phase.items():assert sha(w/n)==h,n
   with (p/'probe.jsonl').open('x') as out,(p/'stderr.log').open('x') as err,(p/'host.jsonl').open('x') as observations:
    host(observations,'before');start=time.monotonic();started=utc();run=subprocess.Popen([str(w/'probe'),str(w/'combined.metal'),mode,str(session)],cwd=root,env=env,stdout=out,stderr=err);(w/'active-run.json').write_text(json.dumps(dict(mode=mode,session=session,pid=run.pid,started_utc=started),indent=2)+'\n');print('START',mode,session,started,flush=True);last=start
    while run.poll() is None:
     time.sleep(1)
     if time.monotonic()-last>=30:host(observations,'active');last=time.monotonic()
    host(observations,'after');status=dict(mode=mode,session=session,exit_code=run.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-start,probe_sha256=sha(w/'probe'));(p/'status.json').write_text(json.dumps(status,indent=2)+'\n');assert run.returncode==0,status
   progress.append(status);(w/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print('PASS',mode,session,flush=True)
  audit=[sys.executable,str(w/'audit-edges.py')] if mode=='edges' else [sys.executable,str(w/'audit.py'),mode]
  with (d/'audit.log').open('x') as f:subprocess.run(audit,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  if mode!='edges' and not read(d/'audit.json')['contemporary_null_screen_pass']:
   (w/'selection-stop.json').write_text(json.dumps(dict(utc=utc(),mode=mode,reason='Neutrality criteria failed; no further GPU work or integration authorized by this plan.'),indent=2)+'\n');print('STOP: neutral screen failed',mode,flush=True);break
 (w/'final-pins.json').write_text(json.dumps(pins(),indent=2)+'\n');(w/'execution-complete.json').write_text(json.dumps(dict(passed=True,utc=utc(),runs=progress,selected=read(w/'timing/audit.json')['candidate_selected_for_later_c0'] if (w/'timing/audit.json').exists() else False),indent=2)+'\n')
except BaseException:
 (w/'failure.json').write_text(json.dumps(dict(utc=utc(),completed=progress,traceback=traceback.format_exc()),indent=2)+'\n');raise
