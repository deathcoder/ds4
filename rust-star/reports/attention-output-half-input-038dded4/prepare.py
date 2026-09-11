from pathlib import Path
import json,hashlib,re,subprocess,datetime
w=Path(__file__).resolve().parent;r=w.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((r/'rust-star/reports/32k-compressed-skip-9d35953f/candidate-build.json').read_text())
for n,h in b['sources'].items():assert sha(r/n)==h,n
text=(r/'rust-star/runtime/build.rs').read_text();section=text[text.index('            manifest.join("src/moe_output_prefix.metal")'):text.index('        "kMoeOutputSource"')]
inputs=[str((r/'rust-star/runtime'/x).resolve().relative_to(r.resolve())) for x in re.findall(r'manifest.join\("([^"\n]+)"\)',section)]+['metal/cpy.metal']
extra='''// Same dense kernel template and accumulator types; only the RHS load type
// changes from float/float2x4 to half/half2x4 after explicit conversion.
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, half, half2x4>) probe_q8_half_t;
template [[host_name("probe_q8_output_half")]] kernel probe_q8_half_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, half, half2x4>;
'''
(w/'candidate-instantiation.metal').write_text(extra)
(w/'combined.metal').write_text('\n'.join((r/p).read_text() for p in inputs)+'\n'+extra)
names=['ds4_metal_args_mul_mm','ds4_metal_args_mul_mm_id_map0','ds4_metal_args_mul_mm_id','ds4_metal_args_dsv4_hc_expand']
alltext='\n'.join((r/p).read_text() for p in inputs);structs=[]
for name in names:
 m=re.search(r'struct '+name+r' \{.*?\n\};',alltext,re.S);assert m,name;structs.append(m[0])
(w/'arguments.h').write_text('#include <stdint.h>\n'+'\n'.join(structs)+'\n')
# Freeze the actual production encoding excerpt for review and source proof.
host=(r/'rust-star/runtime/src/metal_shim.m').read_text();start=host.index('int rust_star_metal_run_prefill_layer2_continuation_tail(')
part=host[start:];i=part.index('        [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline]');j=part.index('        if (tail_samples) {',i)
(w/'production-encoding.m').write_text(part[i:j])
# Arithmetic is unchanged: both grouped instances share exactly one imported
# body, and dense RHS instantiation differs only in the last two template types.
moe=(r/'metal/moe.metal').read_text();dense=(r/'metal/dense.metal').read_text()
assert 'kernel_mul_mm_id_q8_0_f16' in moe
line=next(l for l in dense.splitlines() if 'host_name("kernel_mul_mm_q8_0_f32")' in l)
params=line[line.index('kernel_mul_mm<'):].replace('float, float2x4>;','half, half2x4>;')
assert extra.splitlines()[-1].endswith(params)
(w/'source-audit.json').write_text(json.dumps(dict(passed=True,runtime_sources=b['sources'],sources={p:sha(r/p) for p in inputs},combined_sha256=sha(w/'combined.metal'),candidate_instantiation_sha256=sha(w/'candidate-instantiation.metal'),arguments_sha256=sha(w/'arguments.h'),production_encoding_sha256=sha(w/'production-encoding.m'),dense_baseline_instantiation=line,scope='No imported kernel body or runtime source changed. Two explicit contiguous FP32-to-FP16 conversions plus half-RHS specializations, preserving original accumulator types, order and all other dispatches.'),indent=2)+'\n')
old=json.loads((r/'rust-star/.work/compressed-skip-runs-fbbea533/plan.json').read_text())
plan=dict(schema='rust-star-attention-output-half-input-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),runtime_commit=b['commit'],runtime_sources=b['sources'],source_audit_sha256=sha(w/'source-audit.json'),candidate='Preconvert both Q8 projection RHS tensors to FP16 once; include both conversion dispatches in the complete attention-output chain timing.',baseline_sequence=['group_map','grouped_q8_f32_rhs','dense_q8_f32_rhs','hc_expand4'],candidate_sequence=['convert_heads_f16','group_map','grouped_q8_f16_rhs','convert_low_f16','dense_q8_f16_rhs','hc_expand4'],runtime_shape=dict(rows=4096,groups=8,group_input_width=4096,group_output_width=1024,dense_input_width=8192,dense_output_width=4096,hc_streams=4),extra_fp16_scratch_bytes=4096*(32768+8192)*2,logical_rhs_traffic=dict(baseline_bytes=16*512*1024**2+64*128*1024**2,candidate_bytes=16*256*1024**2+64*64*1024**2+(512+256+128+64)*1024**2,warning='Logical per-tile input loads and conversion traffic only, not measured DRAM traffic; ignores caches and weight/output traffic.'),edge_rows=[1,31,32,33,127,128,129,1024],edge_patterns=['random','rounding-boundary mixture'],edge_pairs_per_case=2,sessions=[0,1],shape_orders=[[0,1,2],[2,1,0]],timed_cases=[0,1,2],timed_rows=4096,case_seeds=[0x2917a301,0x2917a302,0x2917a303],pairs_per_protocol_per_shape=12,pair_orders=old['pair_orders'],a_buffers=old['a_buffers'],protocol_first=old['protocol_first'],protocols=old['protocols'],null_screen=old['null_screen'],candidate_screen=old['candidate_screen'],checks='Full FP32 low projection, dense output and HC output equality, finite checks, deterministic hashes, all buffer guards, CPU verification of every group map, CPU half conversion checks on initial pairs, full immutable-input hashes before/after. Final outputs of repeated slots are checked; overwritten intermediate dispatch outputs are not retained.',diagnostics='Before timed pairs, one sequential per-stage diagnostic chain for each variant at each main case. These split command-buffer observations identify local stage cost but are excluded from speed acceptance.',scope_limits=old['scope_limits'],parent_protocol_audit_sha256=sha(r/'rust-star/reports/compressed-aa-protocol-a6bf9e24/audit.json'),interpretation='Synthetic complete attention-output chain, equal-warm baseline/candidate slots with contemporaneous A/A. Existing thresholds are inherited as screening criteria, not presumed neutral on this new chain. No runtime integration or full-model run in this step.')
(w/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared source, ABI, original dispatch excerpt and prospective complete-chain plan')
