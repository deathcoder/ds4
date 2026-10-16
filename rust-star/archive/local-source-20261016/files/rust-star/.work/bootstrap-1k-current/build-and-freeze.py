from pathlib import Path
import json,hashlib,subprocess,datetime,os,time,traceback
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):
 with (w/n).open('x') as f:f.write(json.dumps(x,indent=2)+'\n')
def no_gpu():
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','probe','router-probe','ds4','ds4-bench','ds4-server'}]
plan=read(w/'plan.json');base=plan['source_reference'];records=[];env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None);env['CARGO_TARGET_DIR']=str(root/'rust-star/.work/msrv-174-c7adfb22/target');env['RUST_STAR_FORCE_NATIVE_REBUILD']='bootstrap-1k-current';env['PYTHONDONTWRITEBYTECODE']='1'
def run(name,cmd):
 no_gpu();started=utc();start=time.monotonic();print('START',name,started,flush=True)
 with (w/(name+'.log')).open('x') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(name=name,command=cmd,exit_code=p.returncode,started_utc=started,finished_utc=utc(),elapsed_seconds=time.monotonic()-start,log_sha256=sha(w/(name+'.log')));save(name+'-status.json',rec);records.append(rec);assert p.returncode==0,rec;print('PASS',name,flush=True)
try:
 no_gpu();assert all(sha(root/n)==h for n,h in base['sources'].items());assert sha(w/'baseline-rust-star')==base['executable_sha256']
 save('pre-edit-freeze.json',dict(utc=utc(),files={str(p.relative_to(w)):sha(p) for p in w.iterdir() if p.is_file()},sources=base['sources']))
 run('prepare-source',['python3',str(w/'prepare-source.py')]);ops=read(w/'source-edits.json')
 for n in ops:(w/('pre-format-'+Path(n).name)).write_bytes((root/n).read_bytes())
 run('format',['cargo','+1.74.0','fmt','--manifest-path','rust-star/runtime/Cargo.toml'])
 for n,es in ops.items():
  p=root/n;pre=(w/('pre-format-'+p.name)).read_text();s=pre
  for e in reversed(es):assert s.count(e['after'])==1,(n,e['after'][:100]);s=s.replace(e['after'],e['before'])
  assert hashlib.sha256(s.encode()).hexdigest()==base['sources'][n],n
  formatted=subprocess.check_output(['rustup','run','1.74.0','rustfmt','--edition','2021','--emit','stdout'],input=pre,text=True) if p.suffix=='.rs' else pre
  assert formatted==p.read_text(),n
  (w/('candidate-'+p.name)).write_bytes(p.read_bytes())
 for n,h in base['sources'].items():
  if n not in ops:assert sha(root/n)==h,n
 save('source-audit.json',dict(passed=True,changed_files=list(ops),unchanged_source_count=39,complete_inverse_before_formatting=True,rust_174_formatting_exact=True,baseline_sources=base['sources'],candidate_sources={n:sha(root/n) for n in base['sources']}))
 run('ring-audit',['python3',str(w/'audit-ring.py')])
 run('host-tests',['cargo','+1.74.0','test','--locked','--manifest-path','rust-star/runtime/Cargo.toml']);assert '320 passed; 0 failed' in (w/'host-tests.log').read_text()
 run('release-build',['cargo','+1.74.0','build','--release','--locked','--manifest-path','rust-star/runtime/Cargo.toml'])
 run('format-check',['cargo','+1.74.0','fmt','--check','--manifest-path','rust-star/runtime/Cargo.toml'])
 run('router-build',['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'router-probe.m'),'-o',str(w/'router-probe')])
 run('router-probe',[str(w/'router-probe'),str(w/'ffn_router.metal')]);rp=read(w/'router-probe.log');assert rp['passed'] and rp['rows']==[32,64,1024] and rp['compared_words']==7840 and rp['guard_words']==96;save('router-proof.json',rp)
 exe=w/'candidate-rust-star';subprocess.run(['cp','-c',str(Path(env['CARGO_TARGET_DIR'])/'release/rust-star'),str(exe)],check=True);exe.chmod(0o555)
 (w/'implementation.patch').write_text(subprocess.check_output(['git','diff',plan['baseline_commit'],'--','rust-star/runtime'],cwd=root,text=True))
 sources={n:sha(root/n) for n in base['sources']};assert sources==read(w/'source-audit.json')['candidate_sources']
 files={str(p.relative_to(w)):sha(p) for p in w.iterdir() if p.is_file() and p.name not in ['candidate-rust-star','baseline-rust-star','execution.log']}
 save('candidate-build.json',dict(passed=True,utc=utc(),commit=plan['baseline_commit'],uncommitted_candidate=True,executable_sha256=sha(exe),baseline_executable_sha256=sha(w/'baseline-rust-star'),sources=sources,files=files,engine_plan_sha256=sha(w/'engine-plan.json'),baseline_profile_audit_sha256=plan['profile_reference_sha256'],toolchain='Rust1.74.0',unchanged_other_source_count=39,runs=records));print('FROZEN candidate after complete build/source/router/ring checks',flush=True)
except BaseException:
 save('build-failure.json',dict(utc=utc(),runs=records,traceback=traceback.format_exc()));raise
