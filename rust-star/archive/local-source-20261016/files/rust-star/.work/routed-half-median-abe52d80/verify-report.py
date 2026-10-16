from pathlib import Path
import json,hashlib,re,subprocess,sys,os
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')['files'];binary=read(w/'binary-identity.json')
for n,e in m.items():
 b=(w/n).read_bytes();assert len(b)==e['stored_bytes'] and sha(b)==e['stored_sha256'],n;assert b'/Users/'+b'deathcodevision' not in b,n
 if not e['redacted']:assert e['stored_sha256']==e['original_sha256']
original=lambda n:binary['sha256'] if n=='probe' else m[n]['original_sha256']
f=read(w/'freeze.json');src=read(w/'source-audit.json');plan=read(w/'plan.json');done=read(w/'execution-complete.json');assert done['passed'];assert read(w/'final-pins.json')['passed']
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
for args in [['audit-edges.py'],['audit.py','aa-only']]+([['audit.py','timing']] if (w/'timing/audit.json').exists() else []):subprocess.run([sys.executable,str(w/args[0]),*args[1:]],check=True,stdout=subprocess.DEVNULL,env=env)
edge=read(w/'edge-audit.json');a=read(w/'aa-only/audit.json');assert edge['passed'] and edge['fp16_words_compared']==64536576;assert a['passed'] and a['correctness_passed']
phases=[('edges',0),('aa-only',0),('aa-only',1)];words=edge['fp16_words_compared']+a['fp16_words_compared'];timing=None
if (w/'timing/audit.json').exists():
 assert a['contemporary_null_screen_pass'];timing=read(w/'timing/audit.json');assert timing['passed'] and timing['correctness_passed'];words+=timing['fp16_words_compared'];phases.extend([('timing',0),('timing',1)]);assert done['selected']==timing['candidate_selected_for_later_c0']
else:assert not a['contemporary_null_screen_pass'] and not done['selected']
assert [(x['mode'],x['session']) for x in done['runs']]==phases
for x in done['runs']:
 assert x['exit_code']==0 and x['probe_sha256']==binary['sha256'];path=w/'edges' if x['mode']=='edges' else w/x['mode']/('session-'+str(x['session']));assert read(path/'status.json')==x
unchanged=read(w/'runtime-unchanged.json');assert unchanged['passed'] and unchanged['commit']==f['accepted_commit'];assert len(unchanged['sources'])==42
for n,h in f['sources'].items():assert unchanged['sources'][n]==h
assert unchanged['sources']['rust-star/runtime/src/metal_shim.m']==f['accepted_host_sha256']
# Independently reverse each complete harness/audit/runner change to the
# preserved original protocol, and prove the shader and all gates are unchanged.
oldref=read(w/'prior-protocol-reference.json');old=w/oldref['report'];assert sha((old/'artifacts.json').read_bytes())==oldref['manifest_sha256']==plan['protocol_change']['prior_manifest_sha256'];oldm=read(old/'artifacts.json')['files'];proof=read(w/'protocol-derivation.json')
for n,edits in read(w/'protocol-source-edits.json').items():
 code=(w/n).read_text();assert original(n)==proof['files'][n]['after_sha256']
 for e in reversed(edits):assert code.count(e['after'])==e['count'];code=code.replace(e['after'],e['before'])
 assert code==(old/n).read_text();assert sha(code.encode())==oldm[n]['stored_sha256']==proof['files'][n]['before_sha256']
assert (w/'candidate.metal').read_bytes()==(old/'candidate.metal').read_bytes();assert (w/'combined.metal').read_bytes()==(old/'combined.metal').read_bytes();oldplan=read(old/'plan.json')
for k in ['neutral_screen','candidate_selection','shape_orders','case_seeds','pair_orders','a_buffers','shape','edge_rows']:assert plan[k]==oldplan[k],k
assert plan['conditioning_dispatches']==8 and plan['samples_per_warmed_slot']==12 and plan['slot_estimator']=='median'
print(json.dumps(dict(passed=True,artifacts=len(m),kernel_and_protocol_inverse_exact=True,fp16_words_compared=words,aa_pairs=a['total_pairs'],mixed_pairs=timing['total_pairs'] if timing else 0,initial_neutral_pass=a['contemporary_null_screen_pass'],mixed_neutral_pass=timing['contemporary_null_screen_pass'] if timing else None,candidate_selected=done['selected'],overall_candidate_ratio=timing['overall_candidate_median'] if timing else None,runtime_changed=False),indent=2))
