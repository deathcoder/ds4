from pathlib import Path
root=Path(__file__).resolve().parents[3]
p=root/'rust-star/runtime/src/metal_shim.m';s=p.read_text()
anchor='@property(nonatomic, strong) id<MTLCommandBuffer> prefillLayerTailCommand;'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'''
@property(nonatomic, strong) id<MTLCounterSampleBuffer> prefillLayerAttentionSamples;
@property(nonatomic, strong) id<MTLCommandBuffer> prefillLayerAttentionCommand;
@property(nonatomic, strong) NSNumber *prefillLayerAttentionSampleCount;''')
start=s.index('int rust_star_metal_enable_prefill_tail_profiling(');end=s.index('\nint rust_star_metal_copy_prefill_tail_ticks(',start)
enable=s[start:end].replace('enable_prefill_tail_profiling','enable_prefill_attention_profiling').replace('prefillLayerTail','prefillLayerAttention').replace('native prefill tail timestamps','native prefill attention timestamps')
enable=enable.replace('context.prefillLayerAttentionCommand = nil;', 'context.prefillLayerAttentionCommand = nil;\n        context.prefillLayerAttentionSampleCount = nil;')
start=s.index('int rust_star_metal_copy_prefill_tail_ticks(');end=s.index('\nint rust_star_metal_enable_prefill_layer_profiling(',start)
copy=s[start:end].replace('copy_prefill_tail_ticks','copy_prefill_attention_ticks').replace('void *opaque_context, uint64_t *ticks, char *error, size_t error_bytes','void *opaque_context, uint32_t sample_count, uint64_t *ticks,\n    char *error, size_t error_bytes').replace('prefillLayerTail','prefillLayerAttention').replace('tail timing','attention timing').replace('native tail','native attention').replace('tail timestamps','attention timestamps').replace('18u','sample_count')
copy=copy.replace('if (!opaque_context || !ticks)', 'if (!opaque_context || !ticks || (sample_count != 8u && sample_count != 18u))')
copy=copy.replace('if (!context.prefillLayerAttentionSamples || !context.prefillLayerAttentionCommand ||', 'if (!context.prefillLayerAttentionSamples || !context.prefillLayerAttentionCommand ||\n            context.prefillLayerAttentionSampleCount.unsignedIntValue != sample_count ||')
copy=copy.replace('context.prefillLayerAttentionCommand = nil;', 'context.prefillLayerAttentionCommand = nil;\n        context.prefillLayerAttentionSampleCount = nil;')
s=s[:end]+'\n'+enable+'\n'+copy+s[end:]
# Bind all encoder changes to exact statement boundaries without modifying dispatches.
def instrument(s,start_marker,end_marker,indexed):
 start=s.index(start_marker);end=s.index(end_marker,start);f=s[start:end]
 indent='    ' if indexed else '        '
 native='retain_batch' if indexed else 'context.prefillLayerChunkStart && position_start == chunk_start && position_start != 0u'
 original=indent+'id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];'
 assert f.count(original)==1
 replacement=indent+f'''id<MTLCounterSampleBuffer> attention_samples = ({native})
{indent}    ? context.prefillLayerAttentionSamples : nil;
{indent}id<MTLFence> attention_fence = attention_samples ? [context.device newFence] : nil;
{indent}if (attention_samples && (!attention_fence || context.prefillLayerAttentionCommand))
{indent}    return fail_with_message(error, error_bytes,
{indent}        @"attention profile requires a fence and consumed previous packet");
{indent}id<MTLComputeCommandEncoder> encoder = stage_compute_encoder(command, attention_samples, 0u);'''
 f=f.replace(original,replacement)
 if indexed:
  boundaries=[('[encoder setComputePipelineState:context.ropeTailPipeline];',1),('[encoder setComputePipelineState:context.f16AlignedPrefillPipeline];\n    [encoder setBytes:&weight_projection',2),('[encoder setComputePipelineState:context.indexerScoreBatchPipeline];',3),('const uint32_t topk_dispatches = encode_indexer_topk',4),('[encoder setComputePipelineState:context.indexedTopkSortPipeline];',5),('uint32_t comp_elements = compressed_capacity*attention_width;',6),('[encoder setComputePipelineState:context.indexedAttentionPrefillPipeline];',7)]
  # Two RoPE occurrences: the first is indexer transformation, final is inverse.
  final='[encoder setComputePipelineState:context.ropeTailPipeline];\n    [encoder setBytes:&inverse_rope'
  boundaries.append((final,8))
 else:
  boundaries=[('if (key_rows%64u != 0u) {',1),('[encoder setComputePipelineState:key_rows%64u != 0u',2),('[encoder setComputePipelineState:context.ropeTailPipeline];',3)]
 for anchor,stage in boundaries:
  if indexed and stage==1:
   assert f.count(anchor)==2
  else:assert f.count(anchor)==1,(anchor,f.count(anchor))
  block=f'''if (attention_samples) {{
{indent}    encoder = next_tail_profile_encoder(encoder, command,
{indent}        attention_samples, attention_fence, {stage}u);
{indent}    if (!encoder) return fail_with_message(error, error_bytes,
{indent}        @"failed to create attention profile encoder");
{indent}}}

{indent}'''
  f=f.replace(anchor,block+anchor,1)
 anchor=indent+'if (!command_succeeded(command, error, error_bytes)) return 0;'
 assert f.count(anchor)==1
 f=f.replace(anchor,anchor+f'''
{indent}if (attention_samples) {{
{indent}    context.prefillLayerAttentionCommand = command;
{indent}    context.prefillLayerAttentionSampleCount = @({18 if indexed else 8}u);
{indent}}}''')
 return s[:start]+f+s[end:]
s=instrument(s,'static int rust_star_metal_run_prefill_layer2_indexed_tile(', '\nint rust_star_metal_run_prefill_layer2_continuation_sparse_tile(',True)
s=instrument(s,'int rust_star_metal_run_prefill_layer3_continuation_batch_attention(', '\nint rust_star_metal_begin_prefill_layer5_continuation(',False)
s=s.replace(' * perturbing diagnostic orders their boundaries with a fence. */',' * perturbing diagnostic orders their boundaries with a fence. Native attention\n * component diagnostics use the same ordering helper. */')
p.write_text(s)
p=root/'rust-star/runtime/src/metal_shim.h';s=p.read_text();anchor='int rust_star_metal_enable_prefill_layer_profiling('
assert s.count(anchor)==1
s=s.replace(anchor,'''/* Diagnostic attention counters: 18 indexed or 8 compressed samples. */
int rust_star_metal_enable_prefill_attention_profiling(
    void *opaque_context, char *error, size_t error_bytes);
int rust_star_metal_copy_prefill_attention_ticks(
    void *opaque_context, uint32_t sample_count, uint64_t *ticks,
    char *error, size_t error_bytes);

'''+anchor);p.write_text(s)
p=root/'rust-star/runtime/src/metal.rs';s=p.read_text();anchor='        fn rust_star_metal_copy_chained_layer_stage_ticks('
assert s.count(anchor)==1
s=s.replace(anchor,'''        fn rust_star_metal_enable_prefill_attention_profiling(
            context: *mut c_void,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_copy_prefill_attention_ticks(
            context: *mut c_void,
            sample_count: u32,
            ticks: *mut u64,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
'''+anchor);p.write_text(s)
