"""Verify the portable evidence; does not load the model or execute a GPU job."""
from pathlib import Path
import hashlib,json,gzip,runpy,re
w=Path(__file__).resolve().parent
read=lambda n:json.loads((w/n).read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
m=read('artifacts.json');red=read('redactions.json')
actual={str(p.relative_to(w)) for p in w.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifacts.json'}
assert actual==set(m['files'])
for n,h in m['files'].items():assert sha((w/n).read_bytes())==h,n
for n,v in red.items():assert sha((w/n).read_bytes())==v['stored_sha256'],n
def rawsha(n):return red[n]['original_sha256'] if n in red else sha((w/n).read_bytes())
b=read('candidate-build.json');old=read('baseline-build.json');plan=read('plan.json');a=read('source-audit.json');pre=read('preflight.json');final=read('final-pins.json')
assert b['commit'].startswith('9d35953f') and old['commit']==plan['runtime_baseline']
for field,n in [('plan_sha256','plan.json'),('tests_log_sha256','tests.log'),('build_log_sha256','build.log'),('preflight_sha256','preflight.json'),('source_audit_sha256','source-audit.json'),('toolchain_sha256','toolchain.json'),('shader_compilation_sha256','shader-compile.json'),('runner_sha256','validate.py')]:assert b[field]==rawsha(n),n
assert b['audit_script_sha256']==rawsha('audit-original.py')
assert '320 passed; 0 failed' in (w/'tests.log').read_text()
assert 'Finished `release`' in (w/'build.log').read_text()
assert final['passed'] and final['sources']==b['sources'] and len(b['sources'])==41
for k,v in pre.items():
 if k not in ['passed','verified_utc']:assert final[k]==v,k
assert final['candidate_executable_sha256']==b['executable_sha256']
assert plan['baseline_executable_sha256']==pre['baseline_executable_sha256']==old['executable_sha256']
assert a['passed'] and read('shader-compile.json')['passed']
ck=(w/'candidate-shader.metal').read_text();pk=(w/'isolated-candidate.metal').read_text()
assert ck.replace('rust_star_compressed_prefill','probe_flash_skip_runs')==pk
assert sha(pk.encode())==a['prior_kernel_sha256']
assert sha(ck.encode())==a['candidate_kernel_sha256']==b['sources']['rust-star/runtime/src/compressed_prefill.metal']
# Reconstruct and hash the entire original host; every unrelated byte is covered.
s=gzip.decompress((w/'candidate-host.m.gz').read_bytes()).decode()
assert sha(s.encode())==b['sources']['rust-star/runtime/src/metal_shim.m']
s=s.replace('\n@property(nonatomic, strong) id<MTLComputePipelineState> compressedNativePipeline;\n@property(nonatomic, strong) id<MTLComputePipelineState> compressedNativeKvpadPipeline;','')
for n in ['native-pipeline-helper.txt','native-predicate.txt']:
 t=(w/n).read_text();assert s.count(t)==1;s=s.replace(t,'')
t=(w/'candidate-dispatch.txt').read_text();assert s.count(t)==1;s=s.replace(t,(w/'original-dispatch.txt').read_text())
assert sha(s.encode())==old['sources']['rust-star/runtime/src/metal_shim.m']
# Match both new specialization constant tables to their original NSG8 tables.
helper=(w/'native-pipeline-helper.txt').read_text();compiler=(w/'compile-shader.m').read_text()
for line in helper.splitlines():
 if '[constants setConstantValue:' in line:assert line.strip() in compiler
for padded,table in [(False,'continuationConstants'),(True,'continuationKvpadConstants')]:
 values=dict(enabled=True,disabled=False,headDim=512,continuationNsg=8,head_dim=512,nsg=8,has_pad=padded)
 pattern=r'\['+table+r' setConstantValue:&(\w+) type:(\w+) atIndex:(\d+)\]'
 before={int(i):(t,values[v]) for v,t,i in re.findall(pattern,s)}
 after={int(i):(t,values[v]) for v,t,i in re.findall(r'\[constants setConstantValue:&(\w+) type:(\w+) atIndex:(\d+)\]',helper)}
 assert len(before)==len(after)==9 and before==after
s=(w/'candidate-build.rs').read_text();assert sha(s.encode())==b['sources']['rust-star/runtime/build.rs']
s=s.replace('    println!("cargo:rerun-if-changed=src/compressed_prefill.metal");\n','').replace('            manifest.join("src/compressed_prefill.metal"),\n','')
assert sha(s.encode())==old['sources']['rust-star/runtime/build.rs']
for n,h in old['sources'].items():
 if n not in ['rust-star/runtime/build.rs','rust-star/runtime/src/metal_shim.m']:assert b['sources'][n]==h,n
parts=[]
for n,h in a['assembly_inputs'].items():
 t=(w/'source-inputs'/n).read_text();assert sha(t.encode())==h==b['sources'][n]
 parts.append(''.join(line+'\n' for line in t.splitlines()))
assert sha(''.join(parts).encode())==a['assembled_sha256']==rawsha('assembled.metal')
assert ck in (w/'assembled.metal').read_text()
accept=read('validation-acceptance.json');assert accept['passed'] and len(accept['runs'])==3
prev=b['frozen_utc']
for rec,n in zip(accept['runs'],plan['live_sequence']):
 assert rec['name']==n and rec['exit_code']==0
 assert prev<rec['started_utc']<rec['finished_utc'];prev=rec['finished_utc']
 assert rec['executable_sha256']==(plan['baseline_executable_sha256'] if n.startswith('baseline') else b['executable_sha256'])
assert prev<final['verified_utc']
runpy.run_path(str(w/'audit.py'),run_name='__main__')
print(f'PASS: {len(m["files"])} artifacts; full source restoration, frozen input/build bindings, three C0 runs and strict profiles')
