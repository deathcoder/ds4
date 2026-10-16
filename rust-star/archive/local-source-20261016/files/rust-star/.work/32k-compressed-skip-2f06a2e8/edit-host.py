from pathlib import Path
w=Path(__file__).resolve().parent;r=w.parents[2]
m=r/'rust-star/runtime/src/metal_shim.m';s=m.read_text();needle='@property(nonatomic, strong) id<MTLComputePipelineState> flashNonvecNsg8KvpadPipeline;';assert s.count(needle)==1
s=s.replace(needle,needle+'\n@property(nonatomic, strong) id<MTLComputePipelineState> compressedNativePipeline;\n@property(nonatomic, strong) id<MTLComputePipelineState> compressedNativeKvpadPipeline;')
helper='''// Lazily specialize only the native compressed continuation dispatch. Prefix,
// legacy continuation and decode retain the imported FlashAttention pipelines.
static int ensure_native_compressed_prefill_pipelines(
    RustStarMetalContext *context, char *error, size_t error_bytes)
{
    if (context.compressedNativePipeline && context.compressedNativeKvpadPipeline) return 1;
    NSError *compile_error = nil;
    bool enabled = true, disabled = false;
    int32_t head_dim = 512, nsg = 8;
    id<MTLComputePipelineState> pipelines[2];
    for (unsigned padded = 0; padded < 2; padded++) {
        bool has_pad = padded != 0;
        MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:300];
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:301];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:302];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:303];
        [constants setConstantValue:&has_pad type:MTLDataTypeBool atIndex:304];
        [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:310];
        [constants setConstantValue:&head_dim type:MTLDataTypeInt atIndex:320];
        [constants setConstantValue:&head_dim type:MTLDataTypeInt atIndex:321];
        [constants setConstantValue:&nsg type:MTLDataTypeInt atIndex:322];
        id<MTLFunction> function = [context.attentionIngressLibrary
            newFunctionWithName:@"rust_star_compressed_prefill"
                 constantValues:constants error:&compile_error];
        pipelines[padded] = function ? [context.device
            newComputePipelineStateWithFunction:function error:&compile_error] : nil;
        if (!pipelines[padded]) return fail_with_message(error, error_bytes,
            [NSString stringWithFormat:@"native compressed prefill pipeline failed: %@", compile_error]);
    }
    context.compressedNativePipeline = pipelines[0];
    context.compressedNativeKvpadPipeline = pipelines[1];
    return 1;
}

'''
needle='int rust_star_metal_run_prefill_layer3_continuation_batch_attention(';assert s.count(needle)==1;s=s.replace(needle,helper+needle)
start=s.index(needle);end=s.index('\nint rust_star_metal_begin_prefill_layer5_continuation(',start);body=s[start:end]
needle2='        if (!ensure_attention_ingress_pipelines(context, error, error_bytes)) return 0;';assert body.count(needle2)==1
addition='''
        const BOOL native_chunk = context.prefillLayerChunkStart &&
            position_start == chunk_start && position_start != 0u;
        if (native_chunk && !ensure_native_compressed_prefill_pipelines(context, error, error_bytes)) return 0;'''
body=body.replace(needle2,needle2+addition)
oldpipe='''        [encoder setComputePipelineState:key_rows%64u != 0u
            ? context.flashNonvecNsg8KvpadPipeline
            : context.flashNonvecNsg8Pipeline];'''
newpipe='''        [encoder setComputePipelineState:native_chunk
            ? (key_rows%64u != 0u ? context.compressedNativeKvpadPipeline : context.compressedNativePipeline)
            : (key_rows%64u != 0u ? context.flashNonvecNsg8KvpadPipeline : context.flashNonvecNsg8Pipeline)];''';assert body.count(oldpipe)==1;body=body.replace(oldpipe,newpipe);s=s[:start]+body+s[end:];m.write_text(s)
for name,t in [('native-pipeline-helper.txt',helper),('native-predicate.txt',addition),('original-dispatch.txt',oldpipe),('candidate-dispatch.txt',newpipe)]: (w/name).write_text(t)
print('Narrow dispatch wired')
