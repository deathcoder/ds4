from pathlib import Path
import json,hashlib,subprocess,datetime,os,sys,time,traceback
w=Path(__file__).resolve().parent;root=w.parents[2];prior=root/'rust-star/.work/autonomous-prefill-a46dfe8b/wide-bootstrap-router-bound';read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
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
assert read(prior/'milestone-completion.json')['passed'] and read(prior/'completion-pins.json')['passed'];no_gpu()
base=read(prior.parent/'plan.json')['accepted_pins']['sources'];assert all(sha(root/n)==h for n,h in base.items()),'Restore accepted runtime before this compatibility repair'
sources=lambda:{n:sha(root/n) for n in base}
performance_sha=read(prior/'completion-pins.json')['baseline_executable_sha256'];assert sha(prior/'baseline-rust-star')==performance_sha
save(w/'preflight.json',dict(utc=utc(),sources=sources(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),prior_milestone_sha256=sha(prior/'milestone-completion.json'),run_sha256=sha(w/'run.py'),observation_sha256=sha(w/'observation.json')))
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None);env['CARGO_TARGET_DIR']=str(w/'target');env['PYTHONDONTWRITEBYTECODE']='1';env['RUST_STAR_FORCE_NATIVE_REBUILD']='msrv-174-c7adfb22';runs=[]
def run(name,cmd):
 no_gpu();started=utc();begin=time.monotonic();print('START',name,started,flush=True)
 with (w/(name+'.log')).open('x') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,exit_code=p.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-begin,log_sha256=sha(w/(name+'.log')));save(w/(name+'-status.json'),row);runs.append(row);print('END',name,p.returncode,flush=True);return p.returncode
try:
 assert run('install-toolchain',['rustup','toolchain','install','1.74.0','--profile','minimal','--component','rustfmt'])==0
 save(w/'toolchain.json',{n:subprocess.check_output(c,cwd=root,env=env,text=True).strip() for n,c in {'rustc':['rustc','+1.74.0','-vV'],'cargo':['cargo','+1.74.0','--version'],'rustfmt':['rustfmt','+1.74.0','--version'],'os':['sw_vers'],'clang':['clang','--version']}.items()})
 cargo=['cargo','+1.74.0'];manifest=['--locked','--manifest-path','rust-star/runtime/Cargo.toml']
 status=run('accepted-check-failure',cargo+['check']+manifest);log=(w/'accepted-check-failure.log').read_text();assert status!=0 and log.count('no method named `is_none_or`')==2 and 'E0599' in log;assert sources()==base
 n='rust-star/runtime/src/metal.rs';p=root/n;s=p.read_text();assert s.count('.is_none_or(')==2;fixed=s.replace('.is_none_or(','.map_or(true, ');p.write_text(fixed)
 save(w/'source-edit.json',dict(file=n,before_sha256=hashlib.sha256(s.encode()).hexdigest(),after_sha256=sha(p),before='.is_none_or(',after='.map_or(true, ',count=2))
 (w/'source.patch').write_text(subprocess.check_output(['git','diff','--',n],cwd=root,text=True));candidate=sources();assert sum(candidate[k]!=base[k] for k in base)==1
 save(w/'candidate-freeze.json',dict(utc=utc(),sources=candidate,patch_sha256=sha(w/'source.patch'),edit_sha256=sha(w/'source-edit.json')))
 commands=[('format',cargo+['fmt','--manifest-path','rust-star/runtime/Cargo.toml','--check']),('host-tests',cargo+['test']+manifest),('release-build',cargo+['build','--release']+manifest),('python-contracts',[sys.executable,'-m','unittest','discover','-s','rust-star/tests','-v']),('fixture-writer',cargo+['run','--quiet']+manifest+['--example','write_logit_fixture','--',str(w/'writer-fixture.json')]),('fixture-reader',[sys.executable,'rust-star/compare_logits.py',str(w/'writer-fixture.json'),str(w/'writer-fixture.json'),'--json',str(w/'writer-reader-comparison.json')])]
 for name,cmd in commands:assert sources()==candidate;assert run(name,cmd)==0,name
 assert sources()==candidate;assert sha(prior/'baseline-rust-star')==performance_sha;edit=read(w/'source-edit.json');assert p.read_text().replace(edit['after'],edit['before'])==s
 save(w/'completion.json',dict(passed=True,utc=utc(),sources=candidate,all_other_41_sources_unchanged=True,accepted_performance_executable_sha256=performance_sha,msrv_executable_sha256=sha(w/'target/release/rust-star'),runs=runs,scope='Local macOS Rust 1.74 build/host/artifact compatibility only; not remote CI, full-model C0 or performance evidence.'))
except BaseException:
 save(w/'failure.json',dict(utc=utc(),runs=runs,traceback=traceback.format_exc()));raise
