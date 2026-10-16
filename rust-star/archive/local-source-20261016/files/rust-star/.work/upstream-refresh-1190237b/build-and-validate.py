from pathlib import Path
import json,subprocess,hashlib,os,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];assert json.loads((w/'conformance-v5.json').read_text())['passed'];env=dict(os.environ);env['CARGO_TARGET_DIR']=str(root/'rust-star/.work/runtime-target')
for name,args in [('batch-tests-fixed',['cargo','test','--manifest-path',str(root/'rust-star/runtime/Cargo.toml'),'--release']),('batch-release',['cargo','build','--manifest-path',str(root/'rust-star/runtime/Cargo.toml'),'--release'])]:
 print('START',name,flush=True)
 with (w/(name+'.log')).open('w') as f:subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
exe=root/'rust-star/.work/runtime-target/release/rust-star';target=w/'batch-rust-star';assert not target.exists();subprocess.run(['cp','-c',str(exe),str(target)],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();names=subprocess.check_output(['git','ls-files','rust-star/runtime'],cwd=root,text=True).splitlines();sources={name:sha(root/name) for name in names if (root/name).is_file()};assert not subprocess.check_output(['git','diff','--name-only','HEAD','--','rust-star/runtime','rust-star/fixtures'],cwd=root,text=True).strip()
(w/'batch-build.json').write_text(json.dumps({'commit':commit,'tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),'executable_sha256':sha(target),'sources':sources},indent=2)+'\n')
subprocess.run(['python3','-u',str(w/'validate-batch.py')],cwd=root,check=True)
