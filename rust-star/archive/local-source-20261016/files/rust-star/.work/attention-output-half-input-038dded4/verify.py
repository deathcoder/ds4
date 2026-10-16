"""Portable source/protocol/evidence verification; no model or GPU work."""
from pathlib import Path
import hashlib,json,runpy,re
w=Path(__file__).resolve().parent
read=lambda n:json.loads((w/n).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read('artifacts.json');red=read('redactions.json')
actual={str(p.relative_to(w)) for p in w.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifacts.json'}
assert actual==set(m['files'])
for n,h in m['files'].items():assert sha(w/n)==h,n
for n,v in red.items():assert sha(w/n)==v['stored_sha256'],n
def rawsha(n):return red[n]['original_sha256'] if n in red else sha(w/n)
p=read('plan.json');a=read('source-audit.json');f=read('freeze.json');ef=read('edge-freeze.json');ex=read('edge-execution-freeze.json');final=read('completion-pins.json')
for freeze in [f,ef,ex]:
 for n,h in freeze['files'].items():
  if n=='probe':assert h==read('binary-identity.json')['sha256']
  else:assert h==rawsha(n),n
assert p['runtime_sources']==a['runtime_sources']==final['runtime_sources'] and len(p['runtime_sources'])==41
assert final['passed'] and final['frozen_files_verified'] and a['passed']
assert p['source_audit_sha256']==rawsha('source-audit.json')
assert p['parent_protocol_audit_sha256']==rawsha('parent-protocol-audit.json')
assert ef['frozen_utc']<read('edges/status.json')['started_utc']<read('edges/status.json')['finished_utc']<f['frozen_utc']
prev=f['frozen_utc']
for session in [0,1]:
 status=read(f'session-{session}/status.json');assert status['exit_code']==0
 assert prev<status['started_utc']<status['finished_utc'];prev=status['finished_utc']
 for n in ['preflight.json','final-pins.json']:assert read(f'session-{session}/'+n)['passed']
assert prev<final['verified_utc']
# Exact assembly from unchanged production source inputs and one instantiation.
parts=[]
for n,h in a['sources'].items():
 t=w/'source-inputs'/n;assert sha(t)==h==p['runtime_sources'][n];parts.append(t.read_text())
extra=(w/'candidate-instantiation.metal').read_text();assert rawsha('candidate-instantiation.metal')==a['candidate_instantiation_sha256']
assert '\n'.join(parts)+'\n'+extra==(w/'combined.metal').read_text()
assert sha(w/'combined.metal')==a['combined_sha256']
dense=(w/'source-inputs/metal/dense.metal').read_text();moe=(w/'source-inputs/metal/moe.metal').read_text()
line=next(l for l in dense.splitlines() if 'host_name("kernel_mul_mm_q8_0_f32")' in l);assert line==a['dense_baseline_instantiation']
params=line[line.index('kernel_mul_mm<'):].replace('float, float2x4>;','half, half2x4>;');assert extra.splitlines()[-1].endswith(params)
lo=[next(l for l in moe.splitlines() if 'host_name("kernel_mul_mm_id_q8_0_'+t+'")' in l) for t in ['f32','f16']]
get=lambda line:line[line.index('kernel_mul_mm_id<'):]
assert get(lo[0]).replace('float, float4x4, float, float2x4>;','half, half4x4, half, half2x4>;')==get(lo[1])
# The grouped export also changes dormant T0 weight-load types. Q8 weights
# are neither float4x4 nor half4x4, and constant 700 is false in both roles.
body=moe[moe.index('kernel void kernel_mul_mm_id('):moe.index('// Address-table variant used by SSD streaming.')]
assert body.count('T0')==2
assert 'is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp' in body
assert '*((device T0 *) x + i)' in body
# Four identical ABI structs extracted from the same production assembly.
text='\n'.join(parts);names=['ds4_metal_args_mul_mm','ds4_metal_args_mul_mm_id_map0','ds4_metal_args_mul_mm_id','ds4_metal_args_dsv4_hc_expand']
structs=[re.search(r'struct '+n+r' \{.*?\n\};',text,re.S)[0] for n in names]
assert '#include <stdint.h>\n'+'\n'.join(structs)+'\n'==(w/'arguments.h').read_text()
assert rawsha('arguments.h')==a['arguments_sha256']
assert rawsha('production-encoding.m')==a['production_encoding_sha256']
assert p['extra_fp16_scratch_bytes']==4096*(32768+8192)*2==335544320
assert p['logical_rhs_traffic']['baseline_bytes']==16*1024**3
assert p['logical_rhs_traffic']['candidate_bytes']==int(8.9375*1024**3)
runpy.run_path(str(w/'edge-audit.py'),run_name='__main__')
runpy.run_path(str(w/'audit.py'),run_name='__main__')
print(f'PASS: {len(m["files"])} artifacts; source assembly/ABI/instantiations, frozen protocol, edge and paired timing audits')
