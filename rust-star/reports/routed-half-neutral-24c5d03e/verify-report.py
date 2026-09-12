from pathlib import Path
import json,hashlib,re,subprocess,sys,os
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')['files'];binary=read(w/'binary-identity.json')
for n,e in m.items():
 b=(w/n).read_bytes();assert len(b)==e['stored_bytes'] and sha(b)==e['stored_sha256'],n;assert b'/Users/'+b'deathcodevision' not in b,n
 if not e['redacted']:assert e['stored_sha256']==e['original_sha256']
original=lambda n:binary['sha256'] if n=='probe' else m[n]['original_sha256']
f=read(w/'freeze.json');src=read(w/'source-audit.json');plan=read(w/'plan.json');done=read(w/'execution-complete.json');assert done['passed'] and not done['selected'];assert not (w/'timing').exists();assert read(w/'final-pins.json')['passed']
for n,h in f['files'].items():assert original(n)==h,n
for n,h in src['sources'].items():assert original('sources/'+n)==h,n
assert src['sources']==f['sources'];assert read(w/'build-status.json')['exit_code']==0
kernel=(w/'candidate.metal').read_text();restored=kernel
for e in reversed(read(w/'source-edits.json')):assert restored.count(e['after'])==e['count'];restored=restored.replace(e['after'],e['before'])
assert restored==(w/'sources/rust-star/runtime/src/routed_prefill.metal').read_text();assert sha(restored.encode())==src['original_kernel_sha256'];assert sha(kernel.encode())==src['candidate_sha256'];combined='\n'.join((w/'sources'/n).read_text() for n in src['sources'])+'\n'+kernel;assert combined==(w/'combined.metal').read_text();assert sha(combined.encode())==src['combined_sha256']
names=['ds4_metal_args_mul_mm_id_map0','ds4_metal_args_mul_mm_id','ds4_metal_dsv4_moe_swiglu_weight_args'];header='#include <stdint.h>\n'+'\n'.join(re.search(r'struct '+n+r' \{.*?\n\};',combined,re.S)[0] for n in names)+'\n';assert header==(w/'arguments.h').read_text() and sha(header.encode())==src['arguments_sha256']
ref=read(w/'host-source-reference.json');prior=w/ref['report'];assert sha((prior/'artifacts.json').read_bytes())==ref['manifest_sha256'];pm=read(prior/'artifacts.json')['files']
for k,expected in [('preparation',src['host_sha256']),('execution',f['accepted_host_sha256'])]:
 n=ref[k];host=(prior/n).read_bytes();assert sha(host)==pm[n]['stored_sha256']==expected;tail=host.decode().split('int rust_star_metal_run_prefill_layer2_continuation_tail(',1)[1]
 a=tail.index('        rust_star_q8_mm_id_map_args routed_map_args =');b=tail.index('        rust_star_q8_mm_id_args routed_down_args =',a);assert tail[a:b]==(w/'production-arguments.m').read_text();a=tail.index('        [encoder setComputePipelineState:context.routedBatchMapPipeline]');b=tail.index('        [encoder setComputePipelineState:context.routedBatchDownPipeline]',a);assert tail[a:b]==(w/'production-encoding.m').read_text()
# These audits have no path-bearing result fields or redacted hash inputs.
for n in ['plan.json','probe.m','candidate.metal','combined.metal','edge-audit.json','edges/probe.jsonl','edges/status.json','aa-only/session-0/probe.jsonl','aa-only/session-1/probe.jsonl']:assert not m[n]['redacted'],n
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for args in [['audit-edges.py'],['audit.py','aa-only'],['diagnose-variance.py']]:subprocess.run([sys.executable,str(w/args[0]),*args[1:]],check=True,stdout=subprocess.DEVNULL,env=env)
edge=read(w/'edge-audit.json');a=read(w/'aa-only/audit.json');assert edge['passed'] and edge['fp16_words_compared']==64536576;assert a['passed'] and a['correctness_passed'] and not a['contemporary_null_screen_pass'] and not a['candidate_selected_for_later_c0'];assert sum(x['null_screen_pass'] for x in a['statistics'])==1
assert [(x['mode'],x['session']) for x in done['runs']]==[('edges',0),('aa-only',0),('aa-only',1)]
for x in done['runs']:assert x['exit_code']==0 and x['probe_sha256']==binary['sha256']
assert read(w/'selection-stop.json')['mode']=='aa-only';assert a['fp16_words_compared']==7851737088
unchanged=read(w/'runtime-unchanged.json');assert unchanged['passed'] and unchanged['commit']==f['accepted_commit'];assert len(unchanged['sources'])==42
for n,h in f['sources'].items():assert unchanged['sources'][n]==h
assert unchanged['sources']['rust-star/runtime/src/metal_shim.m']==f['accepted_host_sha256']
print(json.dumps(dict(passed=True,artifacts=len(m),source_inverse_exact=True,fp16_words_compared=edge['fp16_words_compared']+a['fp16_words_compared'],aa_pairs=72,neutral_cells_passed=1,neutral_cells_total=6,candidate_timing_run=False,runtime_changed=False),indent=2))
