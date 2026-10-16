from pathlib import Path
import subprocess, json, hashlib, os, datetime
w=Path(__file__).resolve().parent;r=w.parents[2]
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env['CARGO_TARGET_DIR']=str(r/'rust-star/.work/runtime-target')
log=(w/'tests-final.log').read_text();assert 'test result: ok. 312 passed' in log,log[-1500:]
with (w/'build.log').open('w') as f:subprocess.run(['cargo','build','--release','--manifest-path',str(r/'rust-star/runtime/Cargo.toml')],env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
assert not subprocess.check_output(['git','diff','HEAD','--','rust-star/runtime'],cwd=r)
exe=w/'rust-star';assert not exe.exists();subprocess.run(['cp','-c',str(r/'rust-star/.work/runtime-target/release/rust-star'),str(exe)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=subprocess.check_output(['git','ls-files','rust-star/runtime'],cwd=r,text=True).splitlines()
(w/'build.json').write_text(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'executable_sha256':sha(exe),'sources':{n:sha(r/n) for n in names if (r/n).is_file()}},indent=2)+'\n')
plan=json.loads((r/'rust-star/.work/paired-32k-128-89e32745-tail/inputs/plan.json').read_text());args=plan['candidate']['adapter_command'];model=args[args.index('--model')+1]
for name,extra in [('profiled',['--profile-json',str(w/'components.json')]),('normal',[])]:
 print('START',name,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 with (w/(name+'.log')).open('w') as f:
  proc=subprocess.run([str(exe),'long-prefill-chunk-probe',model,'--context','32768','--output-only','--json',str(w/(name+'.json'))]+extra,env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT)
 print('EXIT',name,proc.returncode,flush=True)
 if proc.returncode:raise SystemExit(proc.returncode)
print('COMPONENT AND NORMAL C0 GATES PASSED',flush=True)
