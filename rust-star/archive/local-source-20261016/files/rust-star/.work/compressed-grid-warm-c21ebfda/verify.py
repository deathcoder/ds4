"""Verify portable candidate/null-screen evidence without a model or GPU."""
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
assert plan['parent_protocol_audit_sha256']==rawsha('parent/protocol-audit.json')
assert read('parent/protocol-audit.json')['equal_warm_null_screen_pass']
assert plan['created_utc']<freeze['frozen_utc']<read('session-0/status.json')['started_utc']
assert read('session-0/status.json')['finished_utc']<read('session-1/status.json')['started_utc']<read('session-1/status.json')['finished_utc']<completion['verified_utc']
for session in [0,1]:
 for n in ['preflight.json','final-pins.json']:
  p=read(f'session-{session}/{n}');assert p['passed'] and p['runtime_sources']==40 and p['frozen_files']==len(freeze['files'])
hardware=read('hardware.json');assert hardware['sppci_model']=='Apple M1 Ultra' and hardware['sppci_cores']=='48' and hardware['memory_bytes']==128*1024**3
parent=read('parent/source-audit.json')
for n,h in parent['sources'].items():assert rawsha('source-inputs/'+n)==h==plan['runtime_sources'][n],n
candidate=(w/'parent/candidate-wrapper.metal').read_text()
combined='\n'.join((w/'source-inputs'/n).read_text() for n in parent['sources'])+'\n'+candidate
assert combined==(w/'combined.metal').read_text()
assert rawsha('combined.metal')==parent['combined_sha256']==plan['candidate']['combined_sha256']
assert rawsha('parent/candidate-wrapper.metal')==plan['candidate']['wrapper_sha256']
flash=(w/'source-inputs/metal/flash_attn.metal').read_text();structs=[]
for name in ['ds4_metal_args_flash_attn_ext_pad','ds4_metal_args_flash_attn_ext_blk','ds4_metal_args_flash_attn_ext']:
 a=flash.index('struct '+name+' {');b=flash.index('\n};',a)+3;structs.append(flash[a:b])
assert (w/'arguments.h').read_text()=='#include <stdint.h>\n'+'\n\n'.join(structs)+'\n'
start=flash.index('// Batched FlashAttention for prompt/prefill rows.');end=flash.index('#undef FA_NONVEC_TYPES',start)+len('#undef FA_NONVEC_TYPES')
original=flash[start:end]+'\n'
old='#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg';new=old.replace('tgpig,','uint3(tgpig.y, tgpig.x, tgpig.z),')
restored=candidate.replace(new,old).replace('probe_flash_head_first_512_t','flash_attn_ext_dk512_t').replace('probe_flash_head_first_256_t','flash_attn_ext_dk256_t').replace('probe_flash_head_first_impl','kernel_flash_attn_ext').replace('"probe_flash_head_first"','"kernel_flash_attn_ext_f16_dk512_dv512"').replace('"probe_flash_head_first_256"','"kernel_flash_attn_ext_f16_dk256_dv256"')
assert restored==original,'Candidate changes more than wrapper names/grid coordinates'
s=(w/'probe.m').read_text();old=(w/'parent/probe.m').read_text()
assert rawsha('parent/probe.m')==read('source-audit.json')['parent_probe_sha256']
part=lambda t,a,b:t[t.index(a):t.index(b,t.index(a))]
for head,tail in [('static void prepare_and_verify','int main'),('static uint64_t verify','static id<MTLCommandBuffer> encode')]:assert part(old,head,tail)==part(s,head,tail)
encoded=part(s,'static id<MTLCommandBuffer> encode','static double interval').replace('transposed?MTLSizeMake(HEADS,ROWS/8,1):MTLSizeMake(ROWS/8,HEADS,1)','MTLSizeMake(ROWS/8,HEADS,1)')
assert encoded==part(old,'static id<MTLCommandBuffer> encode','static double interval')
slot=part(s,'static NSDictionary *slot','static void emit')
restored=slot.replace('BOOL transposed) {\n int warm=4, samples=3;','BOOL warmed) {\n int warm=warmed?4:0, samples=warmed?3:1;').replace('encode(q,p,a,in,out,transposed)','encode(q,p,a,in,out,NO)').replace('@"transposed":@(transposed),','')
assert restored==part(old,'static NSDictionary *slot','static void emit')
assert 'p[padded][1]=p[padded][0]' not in s
assert 'p[padded][role]=pipeline(d,l,role?@"probe_flash_head_first":@"kernel_flash_attn_ext_f16_dk512_dv512"' in s
assert 'timing[role]=slot(q,p[padded][candidate?role:0],&args,inputs,out[role],candidate&&role);' in s
assert 'c=encode(q,p,a,in,out,transposed);completed(c);return interval(c);' in s
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(w/'audit.py'),run_name='__main__')
a=read('audit.json')
print(json.dumps(dict(passed=True,artifacts=len(inventory),sessions=a['sessions'],pairs=a['total_pairs'],fp32_words_compared=a['fp32_words_compared'],contemporary_null_screen_pass=a['contemporary_null_screen_pass'],candidate_speed_criteria_pass=a['candidate_speed_criteria_pass'],candidate_selected_for_later_c0=a['candidate_selected_for_later_c0'],runtime_changed=False,engine_speedup_claim=False),indent=2))
