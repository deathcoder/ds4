"""Verify portable null-protocol evidence; no GPU or model needed."""
from pathlib import Path
import contextlib,hashlib,io,json,runpy
w=Path(__file__).resolve().parent
read=lambda p:json.loads((w/p).read_text())
sha=lambda p:hashlib.sha256((w/p).read_bytes()).hexdigest()
inventory=read('artifacts.json')['files']
actual={str(p.relative_to(w)) for p in w.rglob('*') if p.is_file() and p.name!='artifacts.json' and '__pycache__' not in p.parts}
assert actual==set(inventory),'Inventory differs'
for n,h in inventory.items():assert sha(n)==h,n
redactions=read('redactions.json')
for n,r in redactions.items():assert sha(n)==r['stored_sha256'],n
rawsha=lambda n:redactions[n]['original_sha256'] if n in redactions else sha(n)
plan=read('plan.json');freeze=read('freeze.json');completion=read('completion-pins.json')
assert completion['passed'] and completion['runtime_changed'] is False
assert completion['freeze_sha256']==rawsha('freeze.json')
assert completion['runtime_sources']==plan['runtime_sources'] and len(plan['runtime_sources'])==40
assert completion['runtime_commit']==plan['runtime_commit']=='1711856ea309f70877102d15facb4e1243a9fdcc'
assert completion['probe_executable_sha256']==freeze['files']['probe']
for n,h in freeze['files'].items():
 if n!='probe':assert rawsha(n)==h,n
assert plan['parent_aa_audit_sha256']==rawsha('parent-shape-audit.json')
assert plan['created_utc']<freeze['frozen_utc']<read('session-0/status.json')['started_utc']
assert read('session-0/status.json')['finished_utc']<read('session-1/status.json')['started_utc']<read('session-1/status.json')['finished_utc']<completion['verified_utc']
for session in [0,1]:
 for n in ['preflight.json','final-pins.json']:
  p=read(f'session-{session}/{n}');assert p['passed'] and p['runtime_sources']==40 and p['frozen_files']==len(freeze['files'])
parent=read('parent/source-audit.json')
for n,h in parent['sources'].items():assert rawsha('source-inputs/'+n)==h==plan['runtime_sources'][n],n
combined='\n'.join((w/'source-inputs'/n).read_text() for n in parent['sources'])+'\n'+(w/'parent/candidate-wrapper.metal').read_text()
assert combined==(w/'combined.metal').read_text() and rawsha('combined.metal')==parent['combined_sha256']
# ABI structs and pipeline/grid are unchanged. The unused old wrapper remains
# in the original combined library but is never chosen for either timed role.
flash=(w/'source-inputs/metal/flash_attn.metal').read_text();structs=[]
for name in ['ds4_metal_args_flash_attn_ext_pad','ds4_metal_args_flash_attn_ext_blk','ds4_metal_args_flash_attn_ext']:
 a=flash.index('struct '+name+' {');b=flash.index('\n};',a)+3;structs.append(flash[a:b])
assert (w/'arguments.h').read_text()=='#include <stdint.h>\n'+'\n\n'.join(structs)+'\n'
s=(w/'probe.m').read_text();old=(w/'parent/aa-control/probe.m').read_text()
assert rawsha('parent/aa-control/probe.m')==read('source-audit.json')['parent_probe_sha256']
for head,oldend,newend in [('static void prepare_and_verify','int main','int main'),('static uint64_t verify','static double dispatch','static id<MTLCommandBuffer> encode')]:
 assert old[old.index(head):old.index(oldend,old.index(head))]==s[s.index(head):s.index(newend,s.index(head))]
a=old.index('static double dispatch(');b=old.index('static void prepare_and_verify',a)
expected=old[a:b].replace('static double dispatch(', 'static id<MTLCommandBuffer> encode(').replace('[e endEncoding];completed(c);double ms=(c.GPUEndTime-c.GPUStartTime)*1000.0;require(isfinite(ms)&&ms>0,"GPU interval");return ms;','[e endEncoding];return c;')
a=s.index('static id<MTLCommandBuffer> encode');b=s.index('static double interval',a)
assert s[a:b].strip()==expected.strip()
assert 'for(int padded=0;padded<2;padded++)p[padded][1]=p[padded][0];' in s
assert 'p[padded][role]=pipeline(d,l,role?@"probe_flash_head_first":@"kernel_flash_attn_ext_f16_dk512_dv512"' in s
assert 'transposed?' not in s
assert 'int warm=warmed?4:0, samples=warmed?3:1;' in s
assert 'for(id<MTLCommandBuffer> c in cmd)[c commit];\n [cmd.lastObject waitUntilCompleted];' in s
assert 'timing[role]=slot(q,p[padded][role],&args,inputs,out[role],warmed);' in s
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(w/'audit.py'),run_name='__main__')
a=read('audit.json')
print(json.dumps(dict(passed=True,artifacts=len(inventory),sessions=a['sessions'],pairs=a['total_pairs'],fp32_words_compared=a['fp32_words_compared'],equal_warm_null_screen_pass=a['equal_warm_null_screen_pass'],runtime_changed=False,engine_speedup_claim=False),indent=2))
