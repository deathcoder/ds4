from pathlib import Path
import json,hashlib,subprocess,datetime,os,sys,time,traceback
w=Path(__file__).resolve().parent;root=w.parents[2];prior=root/'rust-star/.work/msrv-174-c7adfb22';read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,o):
 with p.open('x') as f:f.write(json.dumps(o,indent=2)+'\n')
def no_gpu():
 names={'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe','router-probe'}
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in names]
base=read(prior/'preflight.json')['sources'];sources=lambda:{n:sha(root/n) for n in base};candidate=sources();plan=read(w/'plan.json');assert candidate['rust-star/runtime/src/metal.rs']==plan['candidate_source_sha256'];assert sum(candidate[k]!=base[k] for k in base)==1;assert sha(prior/'failure.json')==plan['prior_failure_sha256'];no_gpu()
files=['plan.json','helper.rs','candidate-metal.rs','inverse-metal.rs','source.patch','run.py','format-candidate.log','format-inverse.log'];freeze=dict(utc=utc(),sources=candidate,files={n:sha(w/n) for n in files},prior_failure_sha256=sha(prior/'failure.json'),reuse_build_directory=str((prior/'target').relative_to(root)));save(w/'freeze.json',freeze)
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None);env['CARGO_TARGET_DIR']=str(prior/'target');env['PYTHONDONTWRITEBYTECODE']='1';env['RUST_STAR_FORCE_NATIVE_REBUILD']='msrv-174-c7adfb22';runs=[]
def pins():
 assert sources()==candidate
 for n,h in freeze['files'].items():assert sha(w/n)==h,n
 return True
commands=[('format',['cargo','+1.74.0','fmt','--manifest-path','rust-star/runtime/Cargo.toml','--check']),('host-tests',['cargo','+1.74.0','test','--locked','--manifest-path','rust-star/runtime/Cargo.toml']),('release-build',['cargo','+1.74.0','build','--locked','--release','--manifest-path','rust-star/runtime/Cargo.toml']),('python-contracts',[sys.executable,'-m','unittest','discover','-s','rust-star/tests','-v']),('fixture-writer',['cargo','+1.74.0','run','--locked','--quiet','--manifest-path','rust-star/runtime/Cargo.toml','--example','write_logit_fixture','--',str(w/'writer-fixture.json')]),('fixture-reader',[sys.executable,'rust-star/compare_logits.py',str(w/'writer-fixture.json'),str(w/'writer-fixture.json'),'--json',str(w/'writer-reader-comparison.json')])]
try:
 for name,cmd in commands:
  no_gpu();pins();start=time.monotonic();started=utc();print('START',name,started,flush=True)
  with (w/(name+'.log')).open('x') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  row=dict(name=name,command=cmd,exit_code=p.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-start,log_sha256=sha(w/(name+'.log')));save(w/(name+'-status.json'),row);runs.append(row);print('END',name,p.returncode,flush=True);assert p.returncode==0,name
 pins();perf=root/'rust-star/.work/runtime-target/release/rust-star';expected=read(root/'rust-star/reports/wide-bootstrap-c7adfb22/restoration.json')['executable_sha256'];assert sha(perf)==expected
 save(w/'completion.json',dict(passed=True,utc=utc(),sources=candidate,runs=runs,files=freeze['files'],all_other_41_sources_unchanged=True,accepted_performance_executable_sha256=expected,msrv_executable_sha256=sha(prior/'target/release/rust-star'),scope='Local macOS Rust 1.74 compatibility. No remote CI, fresh full-model C0 or performance claim.'))
except BaseException:
 save(w/'failure.json',dict(utc=utc(),runs=runs,traceback=traceback.format_exc()));raise
