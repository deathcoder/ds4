from pathlib import Path
import subprocess,os,json,hashlib
w=Path(__file__).resolve().parent;root=w.parents[2];env=dict(os.environ);env['CARGO_TARGET_DIR']=str(root/'rust-star/.work/runtime-target')
assert not subprocess.check_output(['git','diff','HEAD','--','rust-star/runtime'],cwd=root)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
for name,kind in [('tests-final','test'),('release','build')]:
 print('START',name,flush=True)
 with (w/(name+'.log')).open('x') as log:subprocess.run(['cargo',kind,'--release','--manifest-path',str(root/'rust-star/runtime/Cargo.toml')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
exe=w/'candidate-rust-star';subprocess.run(['cp','-c',str(root/'rust-star/.work/runtime-target/release/rust-star'),str(exe)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=subprocess.check_output(['git','ls-files','rust-star/runtime'],cwd=root,text=True).splitlines()
assert not subprocess.check_output(['git','diff','HEAD','--','rust-star/runtime'],cwd=root)
(w/'candidate-build.json').write_text(json.dumps(dict(commit=commit,tree=subprocess.check_output(['git','rev-parse',commit+'^{tree}'],cwd=root,text=True).strip(),executable_sha256=sha(exe),sources={n:sha(root/n) for n in files}),indent=2)+'\n');print('BUILD PASS',flush=True)
