from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();s=s.replace('@property(nonatomic, strong) id<MTLComputePipelineState> indexerTopkStream512Pipeline;\n','');s=s.replace('    id<MTLFunction> topk_stream512 =\n        [library newFunctionWithName:@"kernel_topk_stream512"];\n','');s=s.replace(' || !topk_stream512','');s=s.replace('    context.indexerTopkStream512Pipeline =\n        [context.device newComputePipelineStateWithFunction:topk_stream512 error:&compile_error];\n','');s=s.replace('!context.indexerTopkStream512Pipeline || ','')
idx=s.index('} rust_star_argsort_merge_args;')+len('} rust_star_argsort_merge_args;')
helper='''

/* Port of upstream c0a6119f's deterministic block sort/merge selector.
 * Preserve its tied-score permutation; callers sort selected indices into
 * chronological order before attention. The command retains scratch buffers. */
static uint32_t encode_indexer_topk(
    RustStarMetalContext *context, id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> scores, id<MTLBuffer> selected,
    rust_star_argsort_args args, char *error, size_t error_bytes)
{
    const uint32_t columns = (uint32_t)args.ne00;
    const uint32_t rows = (uint32_t)args.ne01;
    const uint32_t top_k = (uint32_t)args.top_k;
    if (columns == 0 || rows == 0 || top_k == 0 || top_k > columns ||
        scores.length < (NSUInteger)columns*rows*sizeof(float) ||
        selected.length < (NSUInteger)top_k*rows*sizeof(int32_t)) {
        fail_with_message(error, error_bytes, @"invalid deterministic top-k buffers");
        return 0;
    }
    NSUInteger limit = context.indexerTopkPipeline.maxTotalThreadsPerThreadgroup;
    if (!limit) limit = 256;
    uint32_t threads = 1;
    while (threads < columns && threads*2u <= limit) threads *= 2u;
    const uint32_t blocks = (columns+threads-1u)/threads;
    const uint32_t block_top_k = MIN(top_k, threads);
    const uint32_t last = columns-(blocks-1u)*threads;
    const uint32_t work_width = blocks == 1u ? top_k :
        (blocks-1u)*block_top_k+MIN(last,block_top_k);
    const NSUInteger scratch_half = (NSUInteger)work_width*rows*sizeof(int32_t);
    id<MTLBuffer> scratch = blocks == 1u ? nil :
        [context.device newBufferWithLength:2u*scratch_half
                                  options:MTLResourceStorageModeShared];
    if (blocks > 1u && !scratch) {
        fail_with_message(error, error_bytes, @"failed to allocate deterministic top-k scratch");
        return 0;
    }
    args.ne0 = work_width;
    args.top_k = block_top_k;
    [encoder setComputePipelineState:context.indexerTopkPipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:scores offset:0 atIndex:1];
    [encoder setBuffer:blocks == 1u ? selected : scratch offset:0 atIndex:2];
    [encoder setThreadgroupMemoryLength:threads*(sizeof(int32_t)+sizeof(float)) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)blocks*rows,1,1)
         threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    uint32_t dispatches = 1;
    NSUInteger current = 0, next = scratch_half;
    for (uint32_t length = block_top_k; length < work_width; length *= 2u) {
        const uint32_t merges = (work_width+2u*length-1u)/(2u*length);
        NSUInteger merge_threads = context.indexerTopkMergePipeline.maxTotalThreadsPerThreadgroup;
        if (!merge_threads || merge_threads > 512u) merge_threads = 512u;
        merge_threads = MIN(merge_threads,length);
        rust_star_argsort_merge_args merge = {
            .ne00=columns, .ne01=rows, .ne02=1, .ne03=1,
            .nb00=sizeof(float), .nb01=(uint64_t)columns*sizeof(float),
            .nb02=(uint64_t)columns*rows*sizeof(float),
            .nb03=(uint64_t)columns*rows*sizeof(float),
            .ne0=work_width, .ne1=rows, .ne2=1, .ne3=1,
            .top_k=merges == 1u ? top_k : work_width, .len=length,
        };
        [encoder setComputePipelineState:context.indexerTopkMergePipeline];
        [encoder setBytes:&merge length:sizeof(merge) atIndex:0];
        [encoder setBuffer:scores offset:0 atIndex:1];
        [encoder setBuffer:scratch offset:current atIndex:2];
        [encoder setBuffer:merges == 1u ? selected : scratch
                   offset:merges == 1u ? 0 : next atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)merges*rows,1,1)
             threadsPerThreadgroup:MTLSizeMake(merge_threads,1,1)];
        NSUInteger swap = current; current = next; next = swap;
        dispatches++;
    }
    return dispatches;
}
'''
s=s[:idx]+helper+s[idx:]
a=s.index('            [encoder setComputePipelineState:context.indexerTopkStream512Pipeline];');b=s.index('            [encoder setComputePipelineState:context.indexedTopkSortPipeline];',a);s=s[:a]+'''            if (!encode_indexer_topk(context, encoder,
                    layer2_indexer_score_buffer, layer2_indexer_topk_buffer,
                    layer2_topk_args, error, error_bytes)) {
                [encoder endEncoding];
                return 0;
            }

'''+s[b:]
a=s.index('    [encoder setComputePipelineState:context.indexerTopkStream512Pipeline];');b=s.index('    [encoder setComputePipelineState:context.indexedTopkSortPipeline];',a);s=s[:a]+'''    const uint32_t topk_dispatches = encode_indexer_topk(context, encoder,
        score_buffer, topk_buffer, topk_args, error, error_bytes);
    if (!topk_dispatches) {
        [encoder endEncoding];
        return 0;
    }

'''+s[b:];a=s.index('const uint32_t topk_dispatches = encode_indexer_topk');i=s.index('result->dispatches += 10u;',a);s=s[:i]+s[i:].replace('result->dispatches += 10u;', 'result->dispatches += 9u+topk_dispatches;',1);s=s.replace('choices: aligned F16 projections, tiled indexer scores, streaming top-512,','choices: aligned F16 projections, tiled indexer scores, deterministic top-512,');p.write_text(s)
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();needle='''            if tile.dispatches != 10 || tile.wrapped_model_ranges != 3 || tile.pointer_matches != 3
''';assert needle in s;s=s.replace(needle,'''            // 1024-column sorted blocks retain 512 entries; each merge doubles
            // their width until the final 512-index selection is produced.
            let blocks = ((start + 4096) / 4).div_ceil(1024);
            let merges = blocks.next_power_of_two().trailing_zeros();
            if tile.dispatches != 10 + merges || tile.wrapped_model_ranges != 3 || tile.pointer_matches != 3
''',1);p.write_text(s)
