from pathlib import Path
import json,hashlib,subprocess,os,datetime,traceback
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan=read(w/'plan.json');records=[]
def save(n,x):
 with (w/n).open('x') as f:f.write(json.dumps(x,indent=2)+'\n')
def idle():
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','diagnostic-rust-star','baseline-rust-star','current-rust-star','accepted-trace','wide-trace','ds4','ds4-bench','ds4-server'}]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None);env['CARGO_TARGET_DIR']=str(r/'rust-star/.work/msrv-174-c7adfb22/target')
def run(name,args):
 idle();print('START',name,flush=True)
 with (w/(name+'.log')).open('x') as f:p=subprocess.run(args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(name=name,args=args,exit_code=p.returncode,log_sha256=sha(w/(name+'.log')));records.append(rec);save(name+'-status.json',rec);assert p.returncode==0,rec;print('PASS',name,flush=True)
try:
 idle();assert read(w/'mutation-proof-acceptance.json')['passed'];assert read(w/'source-proof.json')['passed'];assert all(sha(r/n)==h for n,h in plan['source_reference']['sources'].items());save('pre-build-freeze.json',{p.name:sha(p) for p in w.iterdir() if p.is_file() and not p.name.endswith('execution.log')})
 prior=read(w/'prior-build.json');edit=read(w/'source-edits.json');check=(w/'candidate-metal_shim.m').read_text()
 for e in reversed(edit['patches']):assert check.count(e['after'])==1;check=check.replace(e['after'],e['before'])
 assert check==(w/'prior-metal_shim.m').read_text()
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:
  assert sha(w/('prior-'+n))==prior['sources']['rust-star/runtime/src/'+n]
  (r/'rust-star/runtime/src'/n).write_bytes((w/('candidate-'+n)).read_bytes())
 for n,h in plan['source_reference']['sources'].items():
  if Path(n).name not in ['metal.rs','long_prefill.rs','metal_shim.m']:assert sha(r/n)==h,n
 env['RUST_STAR_FORCE_NATIVE_REBUILD']='selective-clear-production'
 run('pool-proof-build',['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'pool-proof.m'),'-o',str(w/'pool-proof')])
 run('pool-proof',[str(w/'pool-proof')]);assert read(w/'pool-proof.log')['passed'] and read(w/'pool-proof.log')['selected_policy_checked_bytes']==1073741824
 run('ring-audit',['python3',str(w/'audit-ring.py')])
 run('host-tests',['cargo','+1.74.0','test','--locked','--manifest-path','rust-star/runtime/Cargo.toml']);assert '320 passed; 0 failed' in (w/'host-tests.log').read_text()
 run('release-build',['cargo','+1.74.0','build','--release','--locked','--manifest-path','rust-star/runtime/Cargo.toml'])
 run('format-check',['cargo','+1.74.0','fmt','--check','--manifest-path','rust-star/runtime/Cargo.toml'])
 subprocess.run(['cp','-c',str(Path(env['CARGO_TARGET_DIR'])/'release/rust-star'),str(w/'candidate-rust-star')],check=True);(w/'candidate-rust-star').chmod(0o555)
 candidate_sources={n:sha(r/n) for n in plan['source_reference']['sources']}
 run('router-build',['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'router-probe.m'),'-o',str(w/'router-probe')]);run('router-probe',[str(w/'router-probe'),str(w/'ffn_router.metal')]);rp=read(w/'router-probe.log');assert rp['passed'] and rp['compared_words']==7840 and rp['guard_words']==96
 run('observer-build',['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'observer-proof.m'),'-o',str(w/'observer-proof')]);run('observer-proof',[str(w/'observer-proof')]);assert 'observer weak ownership and deduplication PASS' in (w/'observer-proof.log').read_text()
 (r/'rust-star/runtime/src/metal_shim.m').write_bytes((w/'candidate-instrumented.m').read_bytes());env['RUST_STAR_FORCE_NATIVE_REBUILD']='selective-clear-diagnostic';run('diagnostic-build',['cargo','+1.74.0','build','--release','--locked','--manifest-path','rust-star/runtime/Cargo.toml'])
 subprocess.run(['cp','-c',str(Path(env['CARGO_TARGET_DIR'])/'release/rust-star'),str(w/'diagnostic-rust-star')],check=True);(w/'diagnostic-rust-star').chmod(0o555)
 diagnostic_sources={n:sha(r/n) for n in plan['source_reference']['sources']};(r/'rust-star/runtime/src/metal_shim.m').write_bytes((w/'candidate-metal_shim.m').read_bytes());assert {n:sha(r/n) for n in candidate_sources}==candidate_sources
 (w/'implementation.patch').write_text(subprocess.check_output(['git','diff','--','rust-star/runtime'],cwd=r,text=True))
 save('source-audit.json',dict(passed=True,baseline_sources=plan['source_reference']['sources'],candidate_sources=candidate_sources,diagnostic_sources=diagnostic_sources,new_change_inverse_exact=True,inherited_rust_files_identical=True,other_sources_unchanged=39))
 files={p.name:sha(p) for p in w.iterdir() if p.is_file() and p.name not in ['baseline-rust-star','candidate-rust-star','diagnostic-rust-star','build-execution.log','router-probe','observer-proof','pool-proof']}
 save('candidate-build.json',dict(passed=True,executable_sha256=sha(w/'candidate-rust-star'),diagnostic_executable_sha256=sha(w/'diagnostic-rust-star'),baseline_executable_sha256=sha(w/'baseline-rust-star'),sources=candidate_sources,diagnostic_sources=diagnostic_sources,files=files,engine_plan_sha256=sha(w/'engine-plan.json'),runs=records,toolchain='Rust1.74.0'))
 print('FROZEN production and diagnostic candidates; production source active',flush=True)
except BaseException:
 save('build-failure.json',dict(traceback=traceback.format_exc(),runs=records))
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:(r/'rust-star/runtime/src'/n).write_bytes((w/('baseline-'+n)).read_bytes())
 raise
