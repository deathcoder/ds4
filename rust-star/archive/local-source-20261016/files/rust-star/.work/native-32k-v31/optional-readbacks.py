from pathlib import Path
import re
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
def edit(name,fn):
 global s
 start=s.index(name+'(');start=s.rfind('\n',0,start)+1
 # Locate definition, not call/prototype: chosen names first definitions below.
 end=s.find('\nint ',start+1)
 if end<0:end=len(s)
 body=s[start:end];new=fn(body);assert new!=body,name;s=s[:start]+new+s[end:]
def optional_memcpy(body,names):
 for name in names:
  body=body.replace('memcpy('+name+',','if ('+name+') memcpy('+name+',')
 return body
# Private indexed tile definition is unique.
edit('run_prefill_layer2_indexed_tile',lambda b:optional_memcpy(b.replace('!q_batch || !scores || !topk || !kqv_back ||','!q_batch ||'),['scores','topk','kqv_back']))
def tail(b):
 a=b.index('        (diagnostic_tile &&\n');z=b.index(' || !result)',a)
 b=b[:a]+'        !result)'+b[z+len(' || !result)'):]
 return optional_memcpy(b,['attention_low','attention_output','after_attention_hc','ffn_current','ffn_norm','router_logits','router_probs','router_selected','router_weights','routed_mid','routed_output','shared_output','after_ffn_hc'])
edit('rust_star_metal_run_prefill_layer2_continuation_tail',tail)
def ingress(b):
 b=b.replace(' || !hc_attn_pre','').replace('''        !attn_norm || !q_lora || !q_lora_norm || !kv_raw || !kv_norm ||
        !q_current || !kv_rope || !kv_current || !result)''','''        !result)''')
 return optional_memcpy(b,['hc_attn_pre','attn_norm','q_lora','q_lora_norm','kv_raw','kv_norm','q_current','kv_rope','kv_current'])
edit('rust_star_metal_run_prefill_layer3_continuation_ingress',ingress)
def checks(b):
 b=b.replace('!checksums || checksum_capacity < checksum_count','(checksums && checksum_capacity < checksum_count)').replace('!checksums ||\n        checksum_capacity < checksum_count','(checksums && checksum_capacity < checksum_count)')
 a=b.index('        checksums[0] =');z=b.index(';',b.index('checksums[',a))
 # Enclose the contiguous checksum assignment block, ending before context/result updates.
 match=re.search(r'\n        (?:context\.|memset\(result)',b[a:]);assert match
 z=a+match.start()
 b=b[:a]+'        if (checksums) {\n'+b[a:z]+'\n        }'+b[z:]
 return b
for name in ['rust_star_metal_run_prefill_layer4_prefix_sparse_batch','rust_star_metal_run_prefill_layer4_continuation_boundary','rust_star_metal_finish_prefill_layer5_continuation']:edit(name,checks)
# New completion APIs permit null checksums for timing-only execution.
for name in ['rust_star_metal_finish_long_prefill_layer','rust_star_metal_finish_long_prefill_prefix']:
 def f(b):
  b=b.replace(' || !checksum','').replace('*checksum = fnv1a_u32_words','if (checksum) *checksum = fnv1a_u32_words')
  return b
 edit(name,f)
def begin(b):return b.replace(' || !checksums','').replace('checksums[layer] = fnv1a_u32_words','if (checksums) checksums[layer] = fnv1a_u32_words')
edit('rust_star_metal_begin_long_prefill_prefix',begin)
# Report the Q-only command separately so aggregate GPU time remains complete.
def q(b):
 b=b.replace('uint64_t q_b_bytes, char *error','uint64_t q_b_bytes, rust_star_metal_sparse_indexed_result *result, char *error').replace('if (!opaque_context || !model_mapping) return 0;', 'if (!opaque_context || !model_mapping || !result) return 0;')
 b=b.replace('''        context.prefillLayer2ContinuationQ = q_batch;
''','''        context.prefillLayer2ContinuationQ = q_batch;
        result->dispatches += 2u;
        result->wrapped_model_ranges += 1u;
        result->pointer_matches += q_b_match ? 1u : 0u;
        result->wall_ms += monotonic_ms()-wall_start;
        result->gpu_ms += gpu_elapsed_ms(command);
''');return b
edit('rust_star_metal_prepare_long_prefill_layer2_q',q)
p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();line=[x for x in s.splitlines() if 'int rust_star_metal_prepare_long_prefill_layer2_q(' in x][0];s=s.replace(line,line.replace('uint64_t, char *','uint64_t, rust_star_metal_sparse_indexed_result *, char *'));p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();start=s.index('        fn rust_star_metal_prepare_long_prefill_layer2_q(');end=s.index('        fn ',start+10);b=s[start:end].replace('            error: *mut c_char,','            result: *mut RawSparseIndexedResult,\n            error: *mut c_char,');s=s[:start]+b+s[end:];p.write_text(s)
