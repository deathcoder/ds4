from pathlib import Path
import json,hashlib,subprocess,datetime,re
w=Path(__file__).resolve().parent;root=w.parents[2];d=w/'integration'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(p.read_text())
assert read(w/'timing/audit.json')['candidate_selected_for_later_c0']
assert read(d/'preflight.json')['passed'] and read(d/'source-audit.json')['passed'] and read(d/'shader-compile.json')['passed']
assert '320 passed; 0 failed' in (d/'tests.log').read_text()
assert 'Finished `release`' in (d/'build.log').read_text()
# Invert every recorded edit, then compare the ENTIRE host to frozen baseline.
s=(root/'rust-star/runtime/src/metal_shim.m').read_text()
for op in reversed(read(d/'host-edits.json')):
 assert s.count(op['after'])==1;s=s.replace(op['after'],op['before'])
assert s==subprocess.check_output(['git','show','08ebd1aa:rust-star/runtime/src/metal_shim.m'],cwd=root).decode()
b=read(d/'build-edits.json');assert (root/'rust-star/runtime/build.rs').read_text()==b['after']
assert b['before']==subprocess.check_output(['git','show','08ebd1aa:rust-star/runtime/build.rs'],cwd=root).decode()
shader=(root/'rust-star/runtime/src/attention_output_half.metal').read_text().replace('rust_star_attention_output_half_t','probe_q8_half_t').replace('rust_star_attention_output_half','probe_q8_output_half')
assert shader==(w/'candidate-instantiation.metal').read_text()
prior=read(w/'plan.json')['runtime_sources'];changed={'rust-star/runtime/src/metal_shim.m','rust-star/runtime/build.rs'}
for name,h in prior.items():
 if name not in changed:assert sha(root/name)==h,name
names=[*prior,'rust-star/runtime/src/attention_output_half.metal'];sources={p:sha(root/p) for p in names}
for p,h in sources.items():assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p],cwd=root)).hexdigest()==h,p
# Verify standalone compilation input equals the build-generated Objective-C string.
incs=list((root/'rust-star/.work/runtime-target/release/build').glob('rust-star-*/out/moe_output_source.inc'))
matches=[]
for p in incs:
 lines=p.read_text().splitlines();payload=''.join(json.loads(line.strip()[1:]) for line in lines if line.strip().startswith('@"'))
 if payload==(d/'assembled.metal').read_text():matches.append(p)
assert matches,'no production-generated library matches compiled source'
subprocess.run(['cp','-c',str(root/'rust-star/.work/runtime-target/release/rust-star'),str(d/'candidate-rust-star')],check=True)
subprocess.run(['cp','-c',str(root/read(d/'plan.json')['baseline_executable_source']),str(d/'baseline-rust-star')],check=True)
for n in ['candidate-rust-star','baseline-rust-star']:(d/n).chmod(0o555)
assert sha(d/'baseline-rust-star')==read(d/'plan.json')['baseline_executable_sha256']
r=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),executable_sha256=sha(d/'candidate-rust-star'),sources=sources,production_assembly_matches_build=True,complete_host_restoration_pass=True,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
for k,n in [('plan_sha256','plan.json'),('tests_log_sha256','tests.log'),('build_log_sha256','build.log'),('preflight_sha256','preflight.json'),('source_audit_sha256','source-audit.json'),('shader_compilation_sha256','shader-compile.json'),('runner_sha256','validate.py')]:r[k]=sha(d/n)
r['engine_plan_sha256']=sha(w/'engine-plan.json');r['parent_plan_sha256']=sha(w/'plan.json');r['candidate_selection_audit_sha256']=sha(w/'timing/audit.json')
(d/'candidate-build.json').write_text(json.dumps(r,indent=2)+'\n')
(d/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','08ebd1aa','HEAD','--','rust-star/runtime'],cwd=root))
print('FROZEN',r['commit'],r['executable_sha256'])
