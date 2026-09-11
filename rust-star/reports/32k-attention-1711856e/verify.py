"""Verify the complete evidence package without model weights or binaries."""
from pathlib import Path
import contextlib,hashlib,io,json,re,runpy
w=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(w/'artifacts.json')['files']
actual={str(p.relative_to(w)) for p in w.rglob('*') if p.is_file() and p.name!='artifacts.json' and '__pycache__' not in p.parts}
assert actual==set(m),'Artifact inventory differs'
for n,h in m.items():assert sha(w/n)==h,n
redactions=read(w/'redactions.json')
for n,r in redactions.items():assert sha(w/n)==r['stored_sha256'],n
rawsha=lambda p:redactions[str(p.relative_to(w))]['original_sha256'] if str(p.relative_to(w)) in redactions else sha(p)
b=read(w/'candidate-build.json');pre=read(w/'preflight.json');final=read(w/'final-pins.json');completion=read(w/'completion-pins.json')
assert b['commit']=='1711856ea309f70877102d15facb4e1243a9fdcc'
assert final['passed'] and completion['passed'] and final['sources']==b['sources']
assert final['candidate_executable_sha256']==completion['runtime_executable_sha256']==b['executable_sha256']
assert completion['runtime_source_count']==len(b['sources'])==40
for k,v in pre.items():
 if k not in ['passed','verified_utc']:assert final[k]==v,k
for field,n in [('plan_sha256','plan.json'),('tests_log_sha256','tests.log'),('build_log_sha256','build.log'),('preflight_sha256','preflight.json'),('source_audit_sha256','source-audit.json'),('toolchain_sha256','toolchain.json')]:assert b[field]==rawsha(w/n),n
assert '320 passed; 0 failed' in (w/'tests.log').read_text()
assert read(w/'validation-acceptance.json')['runs'][0]['executable_sha256']==b['executable_sha256']
# Removing only guarded diagnostic statements restores both executor bodies.
a=read(w/'source-audit.json');normalize=lambda s:[l.strip() for l in s.splitlines() if l.strip()]
for kind,count in [('indexed',9),('compressed',4)]:
 base=(w/(kind+'-baseline.m')).read_text();candidate=(w/(kind+'-diagnostic.m')).read_text()
 assert rawsha(w/(kind+'-baseline.m'))==a['paths'][kind]['baseline_sha256']
 assert rawsha(w/(kind+'-diagnostic.m'))==a['paths'][kind]['diagnostic_sha256']
 restored,n=re.subn(r'id<MTLCounterSampleBuffer> attention_samples =.*?id<MTLComputeCommandEncoder> encoder = stage_compute_encoder\(command, attention_samples, 0u\);','id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];',candidate,flags=re.S);assert n==1
 restored,n=re.subn(r'^[ ]*if \(attention_samples\) \{\n.*?^[ ]*\}\n','',restored,flags=re.S|re.M);assert n==count
 assert normalize(restored)==normalize(base)
p=w/'compressed-grid-probe';s=read(p/'source-audit.json')
for n,h in s['sources'].items():assert rawsha(p/'source-inputs'/n)==h==b['sources'][n],n
flash=(p/'source-inputs/metal/flash_attn.metal').read_text();start=flash.index('// Batched FlashAttention for prompt/prefill rows.');end=flash.index('#undef FA_NONVEC_TYPES',start)+len('#undef FA_NONVEC_TYPES')
base=(p/'baseline-wrapper.metal').read_text();candidate=(p/'candidate-wrapper.metal').read_text();assert flash[start:end]+'\n'==base
old='#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg';new=old.replace('tgpig,','uint3(tgpig.y, tgpig.x, tgpig.z),')
restored=candidate.replace(new,old).replace('probe_flash_head_first_512_t','flash_attn_ext_dk512_t').replace('probe_flash_head_first_256_t','flash_attn_ext_dk256_t').replace('probe_flash_head_first_impl','kernel_flash_attn_ext').replace('"probe_flash_head_first"','"kernel_flash_attn_ext_f16_dk512_dv512"').replace('"probe_flash_head_first_256"','"kernel_flash_attn_ext_f16_dk256_dv256"');assert restored==base
combined='\n'.join((p/'source-inputs'/n).read_text() for n in s['sources'])+'\n'+candidate
assert (p/'combined.metal').read_text()==combined
assert rawsha(p/'combined.metal')==s['combined_sha256']
assert read(p/'plan.json')['parent_audit_sha256']==rawsha(w/'audit.json')
# The null control changes pipeline/grid identity and labels, retaining all timing/check code.
control=(p/'aa-control/probe.m').read_text()
control=control.replace(' for(int padded=0;padded<2;padded++)p[padded][1]=p[padded][0];\n','').replace('[e dispatchThreadgroups:MTLSizeMake(ROWS/8,HEADS,1) threadsPerThreadgroup:', '[e dispatchThreadgroups:transposed?MTLSizeMake(HEADS,ROWS/8,1):MTLSizeMake(ROWS/8,HEADS,1) threadsPerThreadgroup:').replace('rust-star-compressed-grid-control-v1','rust-star-compressed-grid-probe-v1').replace('A/A null control: same original pipeline and dispatch in both roles; no candidate or engine claim','synthetic isolated kernel; no engine claim')
assert control==(p/'probe.m').read_text()
assert read(p/'aa-control/plan.json')['candidate_screen_sha256']==rawsha(p/'audit.json')
for folder in [p,p/'aa-control']:
 for n,h in read(folder/'freeze.json')['files'].items():
  if n!='probe':assert rawsha(folder/n)==h,str(folder/n)
 assert read(folder/'status.json')['exit_code']==0
assert completion['probe_freeze_sha256']==rawsha(p/'freeze.json')
assert completion['null_control_freeze_sha256']==rawsha(p/'aa-control/freeze.json')
for script in [w/'audit.py',p/'audit.py',p/'aa-control/audit.py']:
 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(script),run_name='__main__')
assert read(p/'audit.json')['screening_passed'] is False
assert read(p/'aa-control/audit.json')['null_control'] is True
print(json.dumps(dict(passed=True,artifacts=len(m),full_model_fp32_words=16677120,attention_intervals=1883,tail_intervals=2583,candidate_kernel_fp32_comparisons=4697620480,null_control_fp32_comparisons=4697620480,candidate_screening_passed=False,engine_speedup_claim=False),indent=2))
