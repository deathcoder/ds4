from pathlib import Path
import json,struct,sys,math
w=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());checks=w/'checks'
a=read(w/'validation-acceptance.json');assert a['passed'] and len(a['checks'])==10 and all(x['exit']==0 and x['peak_rss_bytes']>0 for x in a['checks'])
assert read(checks/'c0-128k-profiled.json')==read(checks/'c0-128k.json')
assert (checks/'c0-128k-profiled.log').read_text().count('all 129280 logits exact, token ')==128
assert read(checks/'components-128k-audit.json')['passed']
assert read(checks/'batch-8k.json')['complete_8k_transformer_claim'] and read(checks/'batch-8k.json')['output_logits_c0_claim']
x=read(checks/'batch-8k.json');assert x['sparse_transition']['score_kernel']=='rust_star_indexer_scores_tiled64' and x['layer4_prefix_complete']['sparse_attention']['kernels'][0]=='rust_star_indexer_scores_tiled64'
assert 'kernel_dsv4_indexer_scores_tiled' not in (checks/'batch-8k.json').read_text()
assert read(checks/'batch-12k.json')['internal_hc_collection'] and read(checks/'batch-12k.json')['frontier_tokens']==12288
for name in ['batch-32k-output','batch-32k-internal','batch-32k-profiled']:
 x=read(checks/(name+'.json'));assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['checked_logits']==129280
 assert x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169 and x['decode_tokens_checksum']==11977321378026793466
assert read(checks/'batch-2k.json')['output_head']['full_logits_c0_bitwise_match']
x=read(checks/'c0-128k.json');assert x['frontier_tokens']==131072 and x['checked_logits']==129280 and x['checked_decode_steps']==128 and x['external_reference_match'] and not x['throughput_claim']
assert (checks/'c0-128k.log').read_text().count('all 129280 logits exact, token ')==128
for n in [32768,131072]:
 x=read(checks/f'retained-{n//1024}k.json');assert x['context']==n and x['gen_tokens']==128 and not x['paired_protocol_eligible']
 assert x['lifecycle']['cycles']==2
 for key in ['same_process','same_metal_context','first_load_fully_charged','request_state_reset_between_cycles','model_warm_reused_on_retained_cycle','model_residency_reused_on_retained_cycle']:assert x['lifecycle'][key] is True,key
 assert x['selection']['both_oracle_transcripts_match']
 if n==131072:
  c0=read(checks/'c0-128k.json');assert x['selection']['prefill_token']==1110 and x['selection']['final_token']==2945 and x['selection']['selected_tokens_checksum']==c0['decode_tokens_checksum']
 else:assert x['selection']['prefill_token']==85166 and x['selection']['final_token']==14 and x['selection']['selected_tokens_checksum']==11977321378026793466
 for key in ['model_warm_ms','model_view_warm_wall_ms','model_view_warm_gpu_ms']:assert x['first_load'][key]>0 and x['retained'][key]==0
 for role in ['first_load','retained']:
  d=x[role]
  for key in ['prefill_ms','gen_ms','gen_first_ms','gen_steady_ms','prefill_tps','gen_tps','gen_steady_tps']:assert math.isfinite(d[key]) and d[key]>0
  assert abs(d['prefill_ms']+d['model_warm_ms']-d['prompt_with_model_warm_ms'])<1e-5
  assert abs(d['gen_first_ms']+d['gen_steady_ms']-d['gen_ms'])<1e-5
assert read(checks/'component-audit.json')['passed']
(w/'gates-audit.json').write_text(json.dumps(dict(passed=True,full_128k_c0=True,retained_contexts=[32768,131072],retained_cycles_per_context=2,retained_validation='exact greedy transcript and reset/residency invariants; independent fresh full-logit C0',native_checks=10),indent=2)+'\n')
print('PASS gate and lifecycle audit')
