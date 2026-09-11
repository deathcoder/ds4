from pathlib import Path
import hashlib,json,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda n:json.loads((w/n).read_text())
assert read('shader-compile.json')['passed']
assert read('preflight.json')['passed'] and read('source-audit.json')['passed']
assert '320 passed; 0 failed' in (w/'tests.log').read_text()
assert 'Finished `release`' in (w/'build.log').read_text()
assert not subprocess.check_output(['git','diff','--name-only','HEAD','--','rust-star/runtime','metal'],cwd=root,text=True).strip()
prior=json.loads((root/'rust-star/.work/32k-attention-breakdown-ce94f3fb/candidate-build.json').read_text())
sources={p:sha(root/p) for p in [*prior['sources'],'rust-star/runtime/src/compressed_prefill.metal']}
for p,h in sources.items():assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p],cwd=root)).hexdigest()==h,p
subprocess.run(['cp','-c',str(root/'rust-star/.work/runtime-target/release/rust-star'),str(w/'candidate-rust-star')],check=True)
(w/'candidate-rust-star').chmod(0o555)
r=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),executable_sha256=sha(w/'candidate-rust-star'),sources=sources)
for field,name in [('plan_sha256','plan.json'),('tests_log_sha256','tests.log'),('build_log_sha256','build.log'),('preflight_sha256','preflight.json'),('source_audit_sha256','source-audit.json'),('toolchain_sha256','toolchain.json'),('shader_compilation_sha256','shader-compile.json'),('runner_sha256','validate.py'),('audit_script_sha256','audit.py')]:r[field]=sha(w/name)
r['frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(w/'candidate-build.json').write_text(json.dumps(r,indent=2)+'\n')
(w/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','2f06a2e8','HEAD','--','rust-star/runtime'],cwd=root))
subprocess.run(['cp','-c',str(root/'rust-star/.work/32k-attention-breakdown-ce94f3fb/candidate-rust-star'),str(w/'baseline-rust-star')],check=True)
assert sha(w/'baseline-rust-star')==read('plan.json')['baseline_executable_sha256']
print('FROZEN',r['commit'],r['executable_sha256'])
