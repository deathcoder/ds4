from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2]
assert json.loads((w/'timing/audit.json').read_text())['candidate_selected_for_later_c0']
out=w/'integration';out.mkdir()
original=subprocess.check_output(['git','show','08ebd1aa:rust-star/runtime/src/metal_shim.m'],cwd=root).decode()
p=root/'rust-star/runtime/src/metal_shim.m';assert p.read_text()==original
s=original;ops=[]
def replace(a,b,scope=False):
 global s
 if scope:
  start=s.index('int rust_star_metal_run_prefill_layer2_continuation_tail(');end=s.index('\nint ',start+5)
  fragment=s[start:end];assert fragment.count(a)==1,(a[:100],fragment.count(a));s=s[:start]+fragment.replace(a,b)+s[end:]
 else:
  assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b)
 ops.append(dict(before=a,after=b,scope='continuation_tail' if scope else 'context_property'))
replace('@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputBatchLowPipeline;', '@property(nonatomic, strong) id<MTLComputePipelineState> attentionOutputBatchLowPipeline;\n@property(nonatomic, strong) id<MTLComputePipelineState> attentionNativeHalfLowPipeline;\n@property(nonatomic, strong) id<MTLComputePipelineState> attentionNativeHalfOutputPipeline;')
replace('        const BOOL copy_routed_mid = !native_chunk;', '''        // Only native continuation chunks use preconverted RHS tensors.
        // Prefix, legacy tiled prefill and decode retain their original dispatches.
        const BOOL native_attention_half = native_chunk && chunk_start != 0u &&
            tile_start == chunk_start;
        const BOOL copy_routed_mid = !native_chunk;''',True)
needle='        if (native_chunk && !context.routedNativePairSwigluPipeline) {'
helper='''        if (native_attention_half && (!context.attentionNativeHalfLowPipeline ||
            !context.attentionNativeHalfOutputPipeline)) {
            NSError *compile_error = nil;
            bool bounds = false;
            MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&bounds type:MTLDataTypeBool atIndex:700];
            [constants setConstantValue:&bounds type:MTLDataTypeBool atIndex:701];
            id<MTLFunction> low = [context.moeOutputLibrary
                newFunctionWithName:@"kernel_mul_mm_id_q8_0_f16"
                constantValues:constants error:&compile_error];
            id<MTLFunction> output = [context.moeOutputLibrary
                newFunctionWithName:@"rust_star_attention_output_half"
                constantValues:constants error:&compile_error];
            if (!low || !output) return fail_with_message(error, error_bytes,
                compile_error ? compile_error.localizedDescription : @"missing half-input attention output");
            context.attentionNativeHalfLowPipeline = [context.device
                newComputePipelineStateWithFunction:low error:&compile_error];
            context.attentionNativeHalfOutputPipeline = [context.device
                newComputePipelineStateWithFunction:output error:&compile_error];
            if (!context.attentionNativeHalfLowPipeline || !context.attentionNativeHalfOutputPipeline)
                return fail_with_message(error, error_bytes, compile_error.localizedDescription);
        }
'''
replace(needle,helper+needle,True)
needle='        RUST_STAR_NEW_CONTINUATION_TAIL_BUFFER(output_buffer, output_bytes);'
replace(needle,needle+'''
        id<MTLBuffer> heads_half_buffer = native_attention_half
            ? persistent_buffer(context, @"prefill_transient_tail_attention_heads_half",
                (NSUInteger)rows*q_dim*sizeof(uint16_t), error, error_bytes) : nil;
        id<MTLBuffer> low_half_buffer = native_attention_half
            ? persistent_buffer(context, @"prefill_transient_tail_attention_low_half",
                (NSUInteger)rows*8u*output_rank*sizeof(uint16_t), error, error_bytes) : nil;
        if (native_attention_half && (!heads_half_buffer || !low_half_buffer))
            return fail_with_message(error, error_bytes, @"failed to allocate half-input attention scratch");''',True)
needle='        rust_star_hc_expand_args attention_hc_args = {'
replace(needle,'''        if (native_attention_half) {
            low_args.nb10 /= 2u;
            low_args.nb11 /= 2u;
            low_args.nb12 /= 2u;
            low_args.nb13 /= 2u;
            output_args.nb10 /= 2u;
            output_args.nb11 /= 2u;
            output_args.nb12 /= 2u;
            output_args.nb13 /= 2u;
        }
'''+needle,True)
def conversion(count,src,offset,dst):
 return f'''        if (native_attention_half) {{
            const uint32_t count = {count};
            [encoder setComputePipelineState:context.cpyF32F16Pipeline];
            [encoder setBytes:&count length:sizeof(count) atIndex:0];
            [encoder setBuffer:{src} offset:{offset} atIndex:1];
            [encoder setBuffer:{dst} offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake((count+1023u)/1024u,1,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }}

'''
needle='        [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline];'
replace(needle,conversion('rows*q_dim','heads_buffer','heads_offset','heads_half_buffer')+needle,True)
replace('        [encoder setComputePipelineState:context.attentionOutputBatchLowPipeline];','''        [encoder setComputePipelineState:native_attention_half
            ? context.attentionNativeHalfLowPipeline : context.attentionOutputBatchLowPipeline];''',True)
replace('        [encoder setBuffer:heads_buffer offset:heads_offset atIndex:2];','''        [encoder setBuffer:native_attention_half ? heads_half_buffer : heads_buffer
                     offset:native_attention_half ? 0u : heads_offset atIndex:2];''',True)
needle='        [encoder setComputePipelineState:context.q8PrefillPipeline];\n        [encoder setBytes:&output_args length:sizeof(output_args) atIndex:0];'
replace(needle,conversion('rows*8u*output_rank','attention_low_buffer','0','low_half_buffer')+'''        [encoder setComputePipelineState:native_attention_half
            ? context.attentionNativeHalfOutputPipeline : context.q8PrefillPipeline];
        [encoder setBytes:&output_args length:sizeof(output_args) atIndex:0];''',True)
replace('        [encoder setBuffer:attention_low_buffer offset:0 atIndex:2];','        [encoder setBuffer:native_attention_half ? low_half_buffer : attention_low_buffer offset:0 atIndex:2];',True)
restored=s
for op in reversed(ops):
 assert restored.count(op['after'])==1
 restored=restored.replace(op['after'],op['before'])
assert restored==original
p.write_text(s)
(out/'host-edits.json').write_text(json.dumps(ops,indent=2)+'\n')
b=root/'rust-star/runtime/build.rs';old=b.read_text();new=old.replace('    println!("cargo:rerun-if-changed=src/attention_output.metal");','    println!("cargo:rerun-if-changed=src/attention_output.metal");\n    println!("cargo:rerun-if-changed=src/attention_output_half.metal");')
needle='            manifest.join("src/routed_prefill.metal"),'
assert new.count(needle)==1
new=new.replace(needle,needle+'\n            manifest.join("src/attention_output_half.metal"),');b.write_text(new)
(out/'build-edits.json').write_text(json.dumps(dict(before=old,after=new),indent=2)+'\n')
shader=(w/'candidate-instantiation.metal').read_text().replace('probe_q8_half_t','rust_star_attention_output_half_t').replace('probe_q8_output_half','rust_star_attention_output_half')
(root/'rust-star/runtime/src/attention_output_half.metal').write_text(shader)
(out/'source-audit.json').write_text(json.dumps(dict(passed=True,host_restores_exactly_to_starting_source=True,host_operations=len(ops),new_shader_is_tested_instantiation_with_names_only=True,continuation_only=True,extra_scratch_bytes=335544320),indent=2)+'\n')
print('Integration applied; exact host restoration and shader identity pass')
