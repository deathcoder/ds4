"""Verify this report without model weights or executable binaries."""
from pathlib import Path
import hashlib,json,runpy
w=Path(__file__).resolve().parent
read=lambda n:json.loads((w/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read('artifacts.json')
actual={str(p.relative_to(w)) for p in w.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifacts.json'}
assert actual==set(m['files'])
for name,h in m['files'].items():assert sha(w/name)==h,name
redactions=read('redactions.json')
for n,r in redactions.items():assert sha(w/n)==r['stored_sha256'],n
rawsha=lambda n:redactions[n]['original_sha256'] if n in redactions else sha(w/n)
b=read('candidate-build.json'); plan=read('plan.json'); final=read('final-pins.json'); pre=read('preflight.json')
assert b['commit']=='4552043579bb86b6f94e682e3a154937322169ee'
for key,n in [('plan_sha256','plan.json'),('tests_log_sha256','tests.log'),('build_log_sha256','build.log'),('preflight_sha256','preflight.json'),('source_audit_sha256','source-audit.json'),('shader_compilation_sha256','shader-compile.json'),('reused_host_tests_sha256','reused-host-tests.json')]:assert b[key]==rawsha(n),n
assert '318 passed; 0 failed' in (w/'tests.log').read_text()
assert final['passed'] and final['sources']==b['sources']
for k,v in pre.items():
 if k not in ['passed','verified_utc']:assert final[k]==v,k
assert final['candidate_executable_sha256']==b['executable_sha256']
assert plan['baseline_executable_sha256']==pre['baseline_executable_sha256']
accept=read('validation-acceptance.json'); assert accept['passed'] and len(accept['runs'])==3
for r,n in zip(accept['runs'],plan['live_sequence']):
 assert r['name']==n and r['exit_code']==0
 assert r['executable_sha256']==(plan['baseline_executable_sha256'] if n.startswith('baseline') else b['executable_sha256'])
a=read('source-audit.json'); base=(w/'baseline-kernel.metal').read_text(); candidate=(w/'candidate-shader.metal').read_text()
body=candidate[candidate.index('template<typename'):candidate.index('// IQ2 uses')].rstrip()+'\n'
restored=body.replace('rust_star_routed_prefill_gate_up_impl','kernel_mul_mm_id_pair_swiglu_f16_impl').replace('work_index = tgpig.y','work_index = tgpig.x').replace('r0 = tgpig.x*NR0','r0 = tgpig.y*NR0')
assert restored==base
assert candidate.count('<block_iq2_xxs, 16, dequantize_iq2_xxs>')==2
assert candidate in (w/'assembled.metal').read_text()
assert read('shader-compile.json')['passed']
assert sha(w/'candidate-shader.metal')==a['candidate_shader_sha256']==b['sources']['rust-star/runtime/src/routed_prefill.metal']
assert sha(w/'baseline-kernel.metal')==a['baseline_kernel_sha256']
failed=read('rejected/failure.json');assert failed['exit_code']==1
assert 'QK_NL' in (w/'rejected/candidate-32k-output.log').read_text()
runpy.run_path(str(w/'audit.py'),run_name='__main__')
print(f'PASS: {len(m["files"])} report artifacts; build/input/source bindings; failed attempt; full outputs and profiles')
