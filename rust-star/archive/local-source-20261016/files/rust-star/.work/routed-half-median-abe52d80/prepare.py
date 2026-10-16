from pathlib import Path
import json,hashlib,re
w=Path(__file__).resolve().parent;root=w.parents[2];sha=lambda b:hashlib.sha256(b).hexdigest();source=root/'rust-star/runtime/src/routed_prefill.metal';s=source.read_text();original=s;ops=[]
for before,after in [('rust_star_routed_prefill_gate_up','probe_routed_half_gate_up'),('    device const float * y = (device const float *)(src1','    device const half * y = (device const half *)(src1'),('                (half2x4)(*((device float2x4 *) y));','                *((device half2x4 *) y);')]:
 count=s.count(before);assert count>0
 if 'device' in before:assert count==1
 s=s.replace(before,after);ops.append(dict(before=before,after=after,count=count))
restored=s
for op in reversed(ops):assert restored.count(op['after'])==op['count'];restored=restored.replace(op['after'],op['before'])
assert restored==original;(w/'candidate.metal').write_text(s);(w/'source-edits.json').write_text(json.dumps(ops,indent=2)+'\n')
parts=['rust-star/runtime/src/moe_output_prefix.metal','metal/flash_attn.metal','metal/dense.metal','metal/glu.metal','metal/moe.metal','rust-star/runtime/src/routed_prefill.metal','rust-star/runtime/src/attention_output_half.metal','metal/dsv4_hc.metal','metal/cpy.metal'];raws=[(root/n).read_text() for n in parts];combined='\n'.join(raws)+'\n'+s;(w/'combined.metal').write_text(combined)
names=['ds4_metal_args_mul_mm_id_map0','ds4_metal_args_mul_mm_id','ds4_metal_dsv4_moe_swiglu_weight_args'];structs=[]
for name in names:
 m=re.search(r'struct '+name+r' \{.*?\n\};',combined,re.S);assert m;structs.append(m[0])
(w/'arguments.h').write_text('#include <stdint.h>\n'+'\n'.join(structs)+'\n')
# Extract the exact production map/gate encoding and arguments for review.
host=(root/'rust-star/runtime/src/metal_shim.m').read_text();tail=host[host.index('int rust_star_metal_run_prefill_layer2_continuation_tail('):];a=tail.index('        rust_star_q8_mm_id_map_args routed_map_args =');b=tail.index('        rust_star_q8_mm_id_args routed_down_args =',a);(w/'production-arguments.m').write_text(tail[a:b]);a=tail.index('        [encoder setComputePipelineState:context.routedBatchMapPipeline]');b=tail.index('        [encoder setComputePipelineState:context.routedBatchDownPipeline]',a);(w/'production-encoding.m').write_text(tail[a:b])
r=dict(passed=True,source_scope='Source-only preparation; no GPU compilation/execution or runtime edits',original_kernel_sha256=sha(original.encode()),sources={n:sha((root/n).read_bytes()) for n in parts},host_sha256=sha((root/'rust-star/runtime/src/metal_shim.m').read_bytes()),candidate_sha256=sha(s.encode()),combined_sha256=sha(combined.encode()),arguments_sha256=sha((w/'arguments.h').read_bytes()),inverse_restores_entire_kernel=True,all_arithmetic_and_dispatch_grid_unchanged=True)
(w/'source-audit.json').write_text(json.dumps(r,indent=2)+'\n');print('Candidate kernel and actual production argument/encoding provenance prepared; no runtime changes')
