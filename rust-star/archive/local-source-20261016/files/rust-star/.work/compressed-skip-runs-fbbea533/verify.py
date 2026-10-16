"""Verify portable run-skip evidence without model weights or GPU access."""
from pathlib import Path
import contextlib,hashlib,io,json,re,runpy
w=Path(__file__).resolve().parent
read=lambda p:json.loads((w/p).read_text());sha=lambda p:hashlib.sha256((w/p).read_bytes()).hexdigest()
inventory=read('artifacts.json')['files']
actual={str(p.relative_to(w)) for p in w.rglob('*') if p.is_file() and p.name!='artifacts.json' and '__pycache__' not in p.parts}
assert actual==set(inventory),'Inventory differs'
for n,h in inventory.items():assert sha(n)==h,n
redactions=read('redactions.json')
for n,r in redactions.items():assert sha(n)==r['stored_sha256'],n
def rawsha(n):
 n=str((w/n).resolve().relative_to(w.resolve()))
 return redactions[n]['original_sha256'] if n in redactions else sha(n)
plan=read('plan.json');freeze=read('freeze.json');edge=read('edge-freeze.json');completion=read('completion-pins.json')
assert completion['passed'] and completion['runtime_changed'] is False
assert completion['freeze_sha256']==rawsha('freeze.json') and completion['edge_freeze_sha256']==rawsha('edge-freeze.json')
assert completion['runtime_sources']==plan['runtime_sources'] and len(plan['runtime_sources'])==40
assert completion['runtime_commit']==plan['runtime_commit']=='1711856ea309f70877102d15facb4e1243a9fdcc'
assert completion['probe_executable_sha256']==freeze['files']['probe']==edge['files']['probe']
assert completion['edge_executable_sha256']==freeze['files']['edges/edge-probe']==edge['files']['edges/edge-probe']
for f in [freeze,edge]:
 for n,h in f['files'].items():
  if n not in ['probe','edges/edge-probe']:assert rawsha(n)==h,n
for n,h in read('edges/run-freeze.json')['files'].items():assert rawsha('edges/'+n)==h,n
assert plan['parent_aa_audit_sha256']==rawsha('parent-shape-audit.json')
assert plan['parent_protocol_audit_sha256']==rawsha('parent/protocol-audit.json')
assert plan['predecessor_candidate_audit_sha256']==rawsha('parent/predecessor-audit.json')
assert read('parent/protocol-audit.json')['equal_warm_null_screen_pass']
assert read('parent/predecessor-audit.json')['candidate_selected_for_later_c0'] is False
assert plan['created_utc']<edge['frozen_utc']<read('edges/run-freeze.json')['frozen_utc']<read('edges/status.json')['started_utc']<read('edges/status.json')['finished_utc']<freeze['frozen_utc']<read('session-0/status.json')['started_utc']
assert read('session-0/status.json')['finished_utc']<read('session-1/status.json')['started_utc']<read('session-1/status.json')['finished_utc']<completion['verified_utc']
for session in [0,1]:
 for n in ['preflight.json','final-pins.json']:
  p=read(f'session-{session}/{n}');assert p['passed'] and p['runtime_sources']==40 and p['frozen_files']==len(freeze['files'])
hardware=read('hardware.json');assert hardware['sppci_model']=='Apple M1 Ultra' and hardware['sppci_cores']=='48' and hardware['memory_bytes']==128*1024**3
source=read('source-audit.json')
for n,h in source['sources'].items():assert rawsha('source-inputs/'+n)==h==plan['runtime_sources'][n],n
candidate=(w/'candidate-kernel.metal').read_text();baseline=(w/'baseline-kernel.metal').read_text()
assert rawsha('candidate-kernel.metal')==source['candidate_kernel_sha256']==plan['candidate']['kernel_sha256']
assert rawsha('baseline-kernel.metal')==source['baseline_kernel_sha256']
combined='\n'.join((w/'source-inputs'/n).read_text() for n in source['sources'])+'\n'+candidate
assert combined==(w/'combined.metal').read_text() and rawsha('combined.metal')==source['combined_sha256']==plan['candidate']['combined_sha256']
flash=(w/'source-inputs/metal/flash_attn.metal').read_text();start=flash.index('// DS4 non-vector FlashAttention.');end=flash.index('#undef FA_NONVEC_TYPES',start)+len('#undef FA_NONVEC_TYPES')
assert flash[start:end]+'\n'==baseline
assert candidate.count(source['insertion'])==1
restored=candidate.replace(source['insertion'],source['anchor'])
for a,b in reversed(source['renames']):restored=re.sub(r'\b'+b+r'\b',a,restored)
assert restored==baseline,'Candidate changes more than the integer run-skip insertion'
structs=[]
for name in ['ds4_metal_args_flash_attn_ext_pad','ds4_metal_args_flash_attn_ext_blk','ds4_metal_args_flash_attn_ext']:
 a=flash.index('struct '+name+' {');b=flash.index('\n};',a)+3;structs.append(flash[a:b])
assert (w/'arguments.h').read_text()=='#include <stdint.h>\n'+'\n\n'.join(structs)+'\n'
assert rawsha('edges/arguments.h')==rawsha('arguments.h')
s=(w/'probe.m').read_text();old=(w/'parent/probe.m').read_text()
assert rawsha('parent/probe.m')==source['parent_probe_sha256']
restored=s.replace('probe_flash_skip_runs','probe_flash_head_first').replace('candidate_variant','transposed').replace('rust-star-compressed-skip-runs-v1','rust-star-compressed-grid-warm-v1').replace('[e dispatchThreadgroups:MTLSizeMake(ROWS/8,HEADS,1) threadsPerThreadgroup:','[e dispatchThreadgroups:transposed?MTLSizeMake(HEADS,ROWS/8,1):MTLSizeMake(ROWS/8,HEADS,1) threadsPerThreadgroup:')
assert restored==old,'Production-shape harness changes beyond symbol/grid/tags'
expected=s[:s.index('int main(')].replace('ROWS=4096, HEADS=64','ROWS=16, HEADS=2').replace('static uint64_t verify(', 'static BOOL expect_zero=NO;\nstatic uint64_t verify(').replace('require(nonzero>n/2&&varied,"degenerate attention output");','if(expect_zero)require(nonzero==0,"all-masked output is not zero");else require(nonzero>n/2&&varied,"degenerate attention output");').replace(' require(counts[0]&&counts[1]&&counts[2],"missing skipped/partial/full mask-block coverage");','')
es=(w/'edges/probe.m').read_text();assert es[:es.index('int main(')].strip()==expected.strip()
for script in ['cpu_trace.py','edges/audit.py','audit.py']:
 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(w/script),run_name='__main__')
a=read('audit.json');e=read('edges/audit.json');c=read('cpu-trace.json')
print(json.dumps(dict(passed=True,artifacts=len(inventory),integer_trace_cases=c['cases'],gpu_edge_cases=e['cases'],edge_fp32_words=e['compared_fp32_words'],sessions=a['sessions'],pairs=a['total_pairs'],production_shape_fp32_words=a['fp32_words_compared'],contemporary_null_screen_pass=a['contemporary_null_screen_pass'],candidate_speed_criteria_pass=a['candidate_speed_criteria_pass'],candidate_selected_for_later_c0=a['candidate_selected_for_later_c0'],runtime_changed=False,engine_speedup_claim=False),indent=2))
