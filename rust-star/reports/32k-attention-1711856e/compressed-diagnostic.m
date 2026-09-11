int rust_star_metal_run_prefill_layer3_continuation_batch_attention(
    void *opaque_context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t sinks_offset,
    uint64_t sinks_bytes,
    uint32_t position_start,
    rust_star_metal_sparse_indexed_result *result,
    char *error,
    size_t error_bytes)
{
    const uint32_t chunk_start = prefill_chunk_start(opaque_context);
    const uint32_t chunk_end = chunk_start + 4096u;
    enum {
        rows = 4096,
        width = 512,
        ratio = 128,
        q_dim = 32768,
        n_head = 64,
        head_dim = 512,
    };
    if (!opaque_context || !model_mapping || !result ||
        (position_start != 0u && position_start != chunk_start) ||
        sinks_bytes != n_head*sizeof(float)) {
        return fail_with_message(error, error_bytes,
            @"prefill layer-3 continuation batch attention received an invalid input");
    }

    @autoreleasepool {
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        const uint32_t first_position = position_start;
        const uint32_t raw_start = position_start == 0u ? 0u : position_start-128u;
        const uint32_t raw_rows = position_start == 0u ? 4096u : 4224u;
        const uint32_t compressed_rows = (position_start+rows)/ratio;
        const uint32_t key_rows = raw_rows+compressed_rows;
        const NSUInteger row_bytes = width*sizeof(float);
        const NSUInteger staged_row_bytes = width*sizeof(uint16_t);
        const NSUInteger staged_bytes = (NSUInteger)key_rows*staged_row_bytes;
        const NSUInteger mask_bytes = (NSUInteger)rows*key_rows*sizeof(uint16_t);
        const NSUInteger q_bytes = (NSUInteger)rows*q_dim*sizeof(float);
        const NSUInteger pad_bytes = key_rows%64u != 0u
            ? (NSUInteger)n_head*(2u*width*sizeof(uint16_t)+
                (NSUInteger)rows*sizeof(uint16_t))
            : 1u;
        const uint32_t transient_base = context.prefillLayerChunkStart ? chunk_start : 0u;
        if (first_position < transient_base)
            return fail_with_message(error, error_bytes, @"query position precedes its chunk storage");
        const NSUInteger full_q_bytes =
            (NSUInteger)(first_position-transient_base+rows)*q_dim*sizeof(float);
        const NSUInteger full_kv_bytes =
            (NSUInteger)(raw_start+raw_rows)*row_bytes;
        const NSUInteger compressed_bytes =
            (NSUInteger)compressed_rows*row_bytes;
        const NSUInteger key_blocks = (key_rows+63u)/64u;
        const NSUInteger query_blocks = (rows+7u)/8u;
        const NSUInteger block_bytes = key_blocks*query_blocks;
        if (context.prefillKvRows != chunk_end ||
            !context.prefillLayer3FullQ ||
            context.prefillLayer3FullQ.length < full_q_bytes ||
            !context.prefillLayer3FullKv ||
            context.prefillLayer3FullKv.length < full_kv_bytes ||
            !context.prefillLayer3AttnCompressed ||
            context.prefillLayer3AttnCompressed.length < compressed_bytes) {
            return fail_with_message(error, error_bytes,
                @"prefill layer-3 continuation batch attention requires the complete 8K Q/KV state");
        }
        if (!ensure_attention_ingress_pipelines(context, error, error_bytes)) return 0;

        NSUInteger sinks_inner = 0u;
        BOOL sinks_match = NO;
        id<MTLBuffer> sinks_buffer = wrap_model_range(
            context, model_mapping, model_bytes, sinks_offset, sinks_bytes,
            &sinks_inner, &sinks_match, error, error_bytes);
        if (!sinks_buffer) return 0;

#define RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(name, bytes) \
        id<MTLBuffer> name = [context.device newBufferWithLength:(bytes) \
            options:MTLResourceStorageModeShared]
        RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(staged_kv_buffer, staged_bytes);
        RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(mask_buffer, mask_bytes);
        RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(block_buffer, block_bytes);
        RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(pad_buffer, pad_bytes);
        RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(heads_buffer, q_bytes);
        RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER(position_buffer,
            rows*sizeof(int32_t));
#undef RUST_STAR_NEW_L3_BATCH_ATTN_BUFFER
        if (!staged_kv_buffer || !mask_buffer || !block_buffer ||
            !pad_buffer || !heads_buffer || !position_buffer) {
            return fail_with_message(error, error_bytes,
                @"failed to allocate layer-3 continuation batch-attention buffers");
        }

        uint16_t *mask = mask_buffer.contents;
        for (uint32_t query = 0; query < rows; query++) {
            const uint32_t global_query = first_position+query;
            uint16_t *mask_row = mask+(NSUInteger)query*key_rows;
            for (uint32_t key = 0; key < raw_rows; key++) {
                const uint32_t global_key = raw_start+key;
                const BOOL visible = global_key <= global_query &&
                    global_query-global_key < ratio;
                mask_row[key] = visible ? 0u : 0xfc00u;
            }
            const uint32_t visible_compressed = (global_query+1u)/ratio;
            for (uint32_t key = 0; key < compressed_rows; key++) {
                mask_row[raw_rows+key] =
                    key < visible_compressed ? 0u : 0xfc00u;
            }
        }
        int32_t *positions = position_buffer.contents;
        for (uint32_t row = 0; row < rows; row++) {
            positions[row] = first_position+(int32_t)row;
        }

        rust_star_flash_pad_args pad_args = {
            .ne11=(int32_t)key_rows, .ne_12_2=1, .ne_12_3=1,
            .nb11=staged_row_bytes,
            .nb12=staged_bytes, .nb13=staged_bytes,
            .nb21=staged_row_bytes,
            .nb22=staged_bytes, .nb23=staged_bytes,
            .ne31=rows, .ne32=1, .ne33=1,
            .nb31=key_rows*sizeof(uint16_t),
            .nb32=mask_bytes, .nb33=mask_bytes,
        };
        rust_star_flash_blk_args block_args = {
            .ne01=rows, .ne30=key_rows,
            .ne31=rows, .ne32=1, .ne33=1,
            .nb31=key_rows*sizeof(uint16_t),
            .nb32=mask_bytes, .nb33=mask_bytes,
        };
        const uint64_t head_f32_bytes = head_dim*sizeof(float);
        rust_star_flash_vec_args attention_args = {
            .ne01=rows, .ne02=n_head, .ne03=1,
            .nb01=n_head*head_f32_bytes, .nb02=head_f32_bytes,
            .nb03=q_bytes,
            .ne11=key_rows, .ne_12_2=1, .ne_12_3=1, .ns10=head_dim,
            .nb11=staged_row_bytes, .nb12=staged_bytes, .nb13=staged_bytes,
            .ns20=head_dim, .nb21=staged_row_bytes,
            .nb22=staged_bytes, .nb23=staged_bytes,
            .ne31=rows, .ne32=1, .ne33=1,
            .nb31=key_rows*sizeof(uint16_t),
            .nb32=mask_bytes, .nb33=mask_bytes,
            .ne1=n_head, .ne2=rows, .ne3=1,
            .scale=1.0f/sqrtf((float)head_dim),
            .max_bias=0.0f, .m0=0.0f, .m1=0.0f,
            .n_head_log2=0, .logit_softcap=0.0f,
        };
        rust_star_rope_tail_args inverse_args = {
            .ne00=head_dim, .ne01=n_head, .ne02=rows, .ne03=1,
            .nb00=sizeof(float), .nb01=head_f32_bytes,
            .nb02=n_head*head_f32_bytes, .nb03=q_bytes,
            .nb0=sizeof(float), .nb1=head_f32_bytes,
            .nb2=n_head*head_f32_bytes, .nb3=q_bytes,
            .n_dims=64, .mode=0, .n_ctx_orig=65536, .inverse=1,
            .freq_base=160000.0f, .freq_scale=1.0f/16.0f,
            .ext_factor=1.0f,
            .attn_factor=1.0f/(1.0f+0.1f*logf(16.0f)),
            .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
        };

        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLCounterSampleBuffer> attention_samples = (context.prefillLayerChunkStart && position_start == chunk_start && position_start != 0u)
            ? context.prefillLayerAttentionSamples : nil;
        id<MTLFence> attention_fence = attention_samples ? [context.device newFence] : nil;
        if (attention_samples && (!attention_fence || context.prefillLayerAttentionCommand))
            return fail_with_message(error, error_bytes,
                @"attention profile requires a fence and consumed previous packet");
        id<MTLComputeCommandEncoder> encoder = stage_compute_encoder(command, attention_samples, 0u);
        if (!command || !encoder) return fail_with_message(error, error_bytes,
            @"failed to create layer-3 continuation batch-attention command");
#define RUST_STAR_ENCODE_L3_BATCH_COPY(input, input_offset, output_offset, elements) do { \
        uint32_t copy_elements = (elements); \
        [encoder setComputePipelineState:context.cpyF32F16Pipeline]; \
        [encoder setBytes:&copy_elements length:sizeof(copy_elements) atIndex:0]; \
        [encoder setBuffer:(input) offset:(input_offset) atIndex:1]; \
        [encoder setBuffer:staged_kv_buffer offset:(output_offset) atIndex:2]; \
        [encoder dispatchThreadgroups:MTLSizeMake((copy_elements+1023u)/1024u,1,1) \
             threadsPerThreadgroup:MTLSizeMake(256,1,1)]; \
    } while (0)
        RUST_STAR_ENCODE_L3_BATCH_COPY(
            context.prefillLayer3FullKv, (NSUInteger)raw_start*row_bytes,
            0u, raw_rows*width);
        RUST_STAR_ENCODE_L3_BATCH_COPY(
            context.prefillLayer3AttnCompressed, 0u,
            (NSUInteger)raw_rows*staged_row_bytes, compressed_rows*width);
#undef RUST_STAR_ENCODE_L3_BATCH_COPY

        if (attention_samples) {
            encoder = next_tail_profile_encoder(encoder, command,
                attention_samples, attention_fence, 1u);
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create attention profile encoder");
        }

        if (key_rows%64u != 0u) {
            [encoder setComputePipelineState:context.flashPad64Pipeline];
            [encoder setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
            [encoder setBuffer:staged_kv_buffer offset:0 atIndex:1];
            [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
            [encoder setBuffer:mask_buffer offset:0 atIndex:3];
            [encoder setBuffer:pad_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(64,1,1)
                 threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }

        [encoder setComputePipelineState:context.flashBlkPipeline];
        [encoder setBytes:&block_args length:sizeof(block_args) atIndex:0];
        [encoder setBuffer:mask_buffer offset:0 atIndex:1];
        [encoder setBuffer:block_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(key_blocks,query_blocks,1)
             threadsPerThreadgroup:MTLSizeMake(32,1,1)];

        if (attention_samples) {
            encoder = next_tail_profile_encoder(encoder, command,
                attention_samples, attention_fence, 2u);
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create attention profile encoder");
        }

        [encoder setComputePipelineState:key_rows%64u != 0u
            ? context.flashNonvecNsg8KvpadPipeline
            : context.flashNonvecNsg8Pipeline];
        [encoder setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
        [encoder setBuffer:context.prefillLayer3FullQ
                     offset:(NSUInteger)(first_position-transient_base)*q_dim*sizeof(float)
                    atIndex:1];
        [encoder setBuffer:staged_kv_buffer offset:0 atIndex:2];
        [encoder setBuffer:staged_kv_buffer offset:0 atIndex:3];
        [encoder setBuffer:mask_buffer offset:0 atIndex:4];
        [encoder setBuffer:sinks_buffer offset:sinks_inner atIndex:5];
        [encoder setBuffer:pad_buffer offset:0 atIndex:6];
        [encoder setBuffer:block_buffer offset:0 atIndex:7];
        [encoder setBuffer:heads_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:28672u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(query_blocks,n_head,1)
             threadsPerThreadgroup:MTLSizeMake(32,8,1)];

        if (attention_samples) {
            encoder = next_tail_profile_encoder(encoder, command,
                attention_samples, attention_fence, 3u);
            if (!encoder) return fail_with_message(error, error_bytes,
                @"failed to create attention profile encoder");
        }

        [encoder setComputePipelineState:context.ropeTailPipeline];
        [encoder setBytes:&inverse_args length:sizeof(inverse_args) atIndex:0];
        [encoder setBuffer:heads_buffer offset:0 atIndex:1];
        [encoder setBuffer:position_buffer offset:0 atIndex:2];
        [encoder setBuffer:heads_buffer offset:0 atIndex:3];
        [encoder setBuffer:heads_buffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(n_head,rows,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];

        const double wall_start = monotonic_ms();
        [command commit];
        if (!command_succeeded(command, error, error_bytes)) return 0;
        if (attention_samples) {
            context.prefillLayerAttentionCommand = command;
            context.prefillLayerAttentionSampleCount = @(8u);
        }
        context.prefillLayer3ContinuationHeads = heads_buffer;
        result->dispatches += key_rows%64u != 0u ? 6u : 5u;
        result->wrapped_model_ranges += 1u;
        result->pointer_matches += sinks_match ? 1u : 0u;
        result->wall_ms += monotonic_ms()-wall_start;
        result->gpu_ms += gpu_elapsed_ms(command);
        return 1;
    }
}
