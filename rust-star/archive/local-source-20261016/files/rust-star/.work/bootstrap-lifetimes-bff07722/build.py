from pathlib import Path
import json,hashlib,subprocess,os,datetime,traceback
w=Path(__file__).resolve().parent;r=w.parents[2];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan=read(w/'plan.json');records=[]
def save(n,x):
 with (w/n).open('x') as f:f.write(json.dumps(x,indent=2)+'\n')
def idle():
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','accepted-trace','wide-trace','ds4','ds4-bench','ds4-server'}]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None);env['CARGO_TARGET_DIR']=str(r/'rust-star/.work/msrv-174-c7adfb22/target')
def run(name,args):
 idle();print('START',name,flush=True)
 with (w/(name+'.log')).open('x') as f:p=subprocess.run(args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(name=name,args=args,exit_code=p.returncode,log_sha256=sha(w/(name+'.log')));records.append(rec);save(name+'-status.json',rec);assert p.returncode==0,rec
 print('PASS',name,flush=True)
try:
 idle();assert all(sha(r/n)==h for n,h in plan['sources'].items());save('pre-build-freeze.json',{p.name:sha(p) for p in w.iterdir() if p.is_file()})
 run('observer-build',['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(w/'observer-proof.m'),'-o',str(w/'observer-proof')])
 run('observer-proof',[str(w/'observer-proof')])
 rows=[json.loads(l.removeprefix('RS_LIFETIME ')) for l in (w/'observer-proof.log').read_text().splitlines() if l.startswith('RS_LIFETIME ')];assert [(q['scratch_bytes'],q['persistent_bytes'],q['live_count']) for q in rows]==[(4096,8192,2),(0,8192,1),(0,0,0)]
 results={}
 for role in ['accepted','wide']:
  for n in ['metal.rs','long_prefill.rs']:(r/'rust-star/runtime/src'/n).write_bytes((w/(role+'-'+n)).read_bytes())
  (r/'rust-star/runtime/src/metal_shim.m').write_bytes((w/(role+'-instrumented.m')).read_bytes())
  env['RUST_STAR_FORCE_NATIVE_REBUILD']='bootstrap-lifetimes-'+role
  run(role+'-release-build',['cargo','+1.74.0','build','--release','--locked','--manifest-path','rust-star/runtime/Cargo.toml'])
  subprocess.run(['cp','-c',str(Path(env['CARGO_TARGET_DIR'])/'release/rust-star'),str(w/(role+'-trace'))],check=True);(w/(role+'-trace')).chmod(0o555)
  results[role]=dict(executable_sha256=sha(w/(role+'-trace')),sources={n:sha(r/n) for n in plan['sources']})
 save('build.json',dict(passed=True,roles=results,runs=records))
except BaseException:
 save('build-failure.json',dict(traceback=traceback.format_exc(),runs=records));raise
finally:
 for n in ['metal.rs','long_prefill.rs','metal_shim.m']:(r/'rust-star/runtime/src'/n).write_bytes((w/('accepted-'+n)).read_bytes())
 assert all(sha(r/n)==h for n,h in plan['sources'].items())
 print('Accepted runtime source restored',flush=True)
