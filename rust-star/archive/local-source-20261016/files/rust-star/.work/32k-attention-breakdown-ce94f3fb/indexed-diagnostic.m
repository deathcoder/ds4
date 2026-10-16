static int rust_star_metal_run_prefill_layer2_indexed_tile(
    RustStarMetalContext *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t indexer_q_offset,
    uint64_t indexer_q_bytes,
    uint64_t indexer_weight_offset,
    uint64_t indexer_weight_bytes,
    uint64_t sinks_offset,
    uint64_t sinks_bytes,
    uint32_t position_start,
    id<MTLBuffer> q_batch,
    uint32_t q_batch_start,
    id<MTLBuffer> q_norm_buffer,
    uint32_t q_norm_start,
    id<MTLBuffer> attn_norm_buffer,
    uint32_t attn_norm_start,
    id<MTLBuffer> full_kv_buffer,
    id<MTLBuffer> attn_compressed_buffer,
    id<MTLBuffer> indexer_compressed_buffer,
    id<MTLBuffer> *heads_out,
    float *scores,
    int32_t *topk,
    float *kqv_back,
    rust_star_metal_sparse_indexed_result *result,
    char *error,
    size_t error_bytes)
{
    enum {
        batch_tokens = 4096,
        raw_capacity = 4352,
        top_k = 512,
        attention_heads = 64,
        attention_width = 512,
        indexer_width = 128,
        q_rank = 1024,
        n_embd = 4096,
    };
    // Each kernel keeps its original reduction/tile geometry. Batching query
    // rows amortizes bindings, scratch allocation, raw-ring reconstruction and
    // compressed-KV conversion across the complete native chunk.
    const BOOL retain_batch = context.prefillLayerEvenBatchRows != nil;
    const uint32_t tile_rows = retain_batch ? batch_tokens : 32u;
    const uint32_t chunk_start = prefill_chunk_start((__bridge void *)context);
    const uint32_t chunk_end = chunk_start+batch_tokens;
    const uint32_t raw_start = chunk_start-128u;
    const uint32_t compressed_capacity = chunk_end/4u;
    if (position_start < chunk_start || position_start > chunk_end-tile_rows ||
        (position_start-chunk_start)%tile_rows != 0u) {
        return fail_with_message(error, error_bytes,
            @"prefill layer-2 indexed tile start is invalid");
    }
    if (!ensure_attention_ingress_pipelines(context, error, error_bytes) ||
        !ensure_sparse_indexed_pipelines(context, error, error_bytes) ||
        !ensure_moe_output_pipelines(context, error, error_bytes)) return 0;
    if (!q_batch ||
        q_batch.length < (NSUInteger)batch_tokens*attention_heads*
            attention_width*sizeof(float) ||
        !q_norm_buffer || !attn_norm_buffer || !full_kv_buffer ||
        !attn_compressed_buffer || !indexer_compressed_buffer ||
        q_batch_start > position_start || q_norm_start > position_start ||
        attn_norm_start > position_start ||
        full_kv_buffer.length < chunk_end*attention_width*sizeof(float) ||
        attn_compressed_buffer.length <
            (NSUInteger)compressed_capacity*attention_width*sizeof(float) ||
        indexer_compressed_buffer.length <
            (NSUInteger)compressed_capacity*indexer_width*sizeof(float)) {
        return fail_with_message(error, error_bytes,
            @"prefill layer-2 indexed tile retained buffers are undersized");
    }

    NSUInteger inner[3] = {0,0,0};
    BOOL matches[3] = {NO,NO,NO};
    id<MTLBuffer> q_weights = wrap_model_range(
        context, model_mapping, model_bytes, indexer_q_offset, indexer_q_bytes,
        &inner[0], &matches[0], error, error_bytes);
    id<MTLBuffer> weight_weights = wrap_model_range(
        context, model_mapping, model_bytes, indexer_weight_offset,
        indexer_weight_bytes, &inner[1], &matches[1], error, error_bytes);
    id<MTLBuffer> sinks_buffer = wrap_model_range(
        context, model_mapping, model_bytes, sinks_offset, sinks_bytes,
        &inner[2], &matches[2], error, error_bytes);
    if (!q_weights || !weight_weights || !sinks_buffer) return 0;

    const NSUInteger q_row_bytes =
        attention_heads*attention_width*sizeof(float);
    const uint32_t raw_span = 4224u+(position_start-chunk_start);
    const NSUInteger q_batch_offset =
        (NSUInteger)(position_start-q_batch_start)*q_row_bytes;
    const NSUInteger indexer_q_row_bytes =
        attention_heads*indexer_width*sizeof(float);
    const NSUInteger weight_row_bytes = attention_heads*sizeof(float);
    const NSUInteger score_row_bytes = compressed_capacity*sizeof(float);
    const NSUInteger topk_row_bytes = top_k*sizeof(int32_t);
    const NSUInteger kv_row_bytes = attention_width*sizeof(float);
    const NSUInteger raw_bytes = raw_capacity*kv_row_bytes;
    const NSUInteger comp_f16_bytes =
        compressed_capacity*attention_width*sizeof(uint16_t);

#define RS_NEW_INDEXED_TILE_BUFFER(name, bytes) \
    id<MTLBuffer> name = [context.device newBufferWithLength:(bytes) \
        options:MTLResourceStorageModeShared]
    RS_NEW_INDEXED_TILE_BUFFER(indexer_q_buffer, tile_rows*indexer_q_row_bytes);
    RS_NEW_INDEXED_TILE_BUFFER(indexer_weight_buffer, tile_rows*weight_row_bytes);
    RS_NEW_INDEXED_TILE_BUFFER(score_buffer, tile_rows*score_row_bytes);
    RS_NEW_INDEXED_TILE_BUFFER(topk_buffer, tile_rows*topk_row_bytes);
    RS_NEW_INDEXED_TILE_BUFFER(sorted_buffer, tile_rows*topk_row_bytes);
    RS_NEW_INDEXED_TILE_BUFFER(raw_ring, raw_bytes);
    RS_NEW_INDEXED_TILE_BUFFER(comp_f16, comp_f16_bytes);
    id<MTLBuffer> heads_buffer = retain_batch ? context.prefillLayerEvenBatchHeads :
        [context.device newBufferWithLength:tile_rows*q_row_bytes
            options:MTLResourceStorageModeShared];
    RS_NEW_INDEXED_TILE_BUFFER(position_buffer, tile_rows*sizeof(int32_t));
#undef RS_NEW_INDEXED_TILE_BUFFER
    if (!indexer_q_buffer || !indexer_weight_buffer || !score_buffer ||
        !topk_buffer || !sorted_buffer || !raw_ring || !comp_f16 ||
        !heads_buffer || !position_buffer) {
        return fail_with_message(error, error_bytes,
            @"failed to allocate prefill layer-2 indexed tile buffers");
    }

    /* State after the complete second chunk has been stored in the physical
     * 4352-row ring. Causal bounds reconstruct the requested per-query view. */
    for (uint32_t position = chunk_end-raw_capacity; position < chunk_end; position++) {
        memcpy((uint8_t *)raw_ring.contents+(NSUInteger)(position%raw_capacity)*kv_row_bytes,
               (const uint8_t *)full_kv_buffer.contents+(NSUInteger)position*kv_row_bytes,
               kv_row_bytes);
    }
    for (uint32_t row = 0; row < tile_rows; row++) {
        ((int32_t *)position_buffer.contents)[row] =
            (int32_t)(position_start+row);
    }

    rust_star_q8_mm_args q_projection = {
        .ne00=q_rank, .ne02=1,
        .nb01=q_rank*sizeof(uint16_t),
        .nb02=indexer_q_bytes, .nb03=indexer_q_bytes,
        .ne12=1, .nb10=sizeof(float),
        .nb11=q_rank*sizeof(float),
        .nb12=(uint64_t)tile_rows*q_rank*sizeof(float),
        .nb13=(uint64_t)tile_rows*q_rank*sizeof(float),
        .ne0=attention_heads*indexer_width, .ne1=tile_rows, .r2=1, .r3=1,
    };
    rust_star_q8_mm_args weight_projection = {
        .ne00=n_embd, .ne02=1,
        .nb01=n_embd*sizeof(uint16_t),
        .nb02=indexer_weight_bytes, .nb03=indexer_weight_bytes,
        .ne12=1, .nb10=sizeof(float),
        .nb11=n_embd*sizeof(float),
        .nb12=(uint64_t)tile_rows*n_embd*sizeof(float),
        .nb13=(uint64_t)tile_rows*n_embd*sizeof(float),
        .ne0=attention_heads, .ne1=tile_rows, .r2=1, .r3=1,
    };
    rust_star_rope_tail_args indexer_rope = {
        .ne00=indexer_width, .ne01=attention_heads,
        .ne02=tile_rows, .ne03=1,
        .nb00=sizeof(float), .nb01=indexer_width*sizeof(float),
        .nb02=indexer_q_row_bytes,
        .nb03=(uint64_t)tile_rows*indexer_q_row_bytes,
        .nb0=sizeof(float), .nb1=indexer_width*sizeof(float),
        .nb2=indexer_q_row_bytes,
        .nb3=(uint64_t)tile_rows*indexer_q_row_bytes,
        .n_dims=64, .mode=0, .n_ctx_orig=65536, .inverse=0,
        .freq_base=160000.0f, .freq_scale=1.0f/16.0f,
        .ext_factor=1.0f,
        .attn_factor=1.0f/(1.0f+0.1f*logf(16.0f)),
        .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
    };
    rust_star_indexer_qat_args qat = {
        .n_rows=tile_rows*attention_heads, .head_dim=indexer_width,
        .row_stride=indexer_width*sizeof(float),
    };
    rust_star_indexer_score_args score_args = {
        .n_comp=compressed_capacity, .n_tokens=tile_rows,
        .n_head=attention_heads, .head_dim=indexer_width,
        .pos0=position_start, .ratio=4,
        .q_token_stride=indexer_q_row_bytes,
        .q_head_stride=indexer_width*sizeof(float),
        .weights_token_stride=weight_row_bytes,
        .index_row_stride=indexer_width*sizeof(float),
        .score_token_stride=score_row_bytes,
        .scale=1.0f/sqrtf(8192.0f),
    };
    rust_star_argsort_args topk_args = {
        .ne00=compressed_capacity, .ne01=tile_rows, .ne02=1, .ne03=1,
        .nb00=sizeof(float), .nb01=score_row_bytes,
        .nb02=(uint64_t)tile_rows*score_row_bytes,
        .nb03=(uint64_t)tile_rows*score_row_bytes,
        .ne0=top_k, .ne1=tile_rows, .ne2=1, .ne3=1, .top_k=top_k,
    };
    rust_star_topk_mask_args sort_args = {
        .ne00=top_k, .ne01=tile_rows,
        .nb00=sizeof(int32_t), .nb01=topk_row_bytes,
        .ne0=top_k, .ne1=tile_rows,
        .nb0=sizeof(int32_t), .nb1=topk_row_bytes,
    };
    rust_star_indexed_attention_args attention_args = {
        .n_tokens=batch_tokens, .n_head=attention_heads,
        .n_raw=raw_span, .raw_cap=raw_capacity, .raw_start=raw_start,
        .n_comp=compressed_capacity, .top_k=top_k,
        .pos0=position_start, .window=128, .ratio=4,
        .comp_kv_f16=1, .n_splits=1,
        .q_token_stride=q_row_bytes,
        .q_head_stride=attention_width*sizeof(float),
        .raw_row_stride=kv_row_bytes,
        .comp_row_stride=attention_width*sizeof(uint16_t),
        .topk_token_stride=topk_row_bytes,
        .dst_token_stride=q_row_bytes,
        .dst_head_stride=attention_width*sizeof(float),
        .scale=1.0f/sqrtf(512.0f),
    };
    rust_star_rope_tail_args inverse_rope = {
        .ne00=attention_width, .ne01=attention_heads,
        .ne02=tile_rows, .ne03=1,
        .nb00=sizeof(float), .nb01=attention_width*sizeof(float),
        .nb02=q_row_bytes, .nb03=(uint64_t)tile_rows*q_row_bytes,
        .nb0=sizeof(float), .nb1=attention_width*sizeof(float),
        .nb2=q_row_bytes, .nb3=(uint64_t)tile_rows*q_row_bytes,
        .n_dims=64, .mode=0, .n_ctx_orig=65536, .inverse=1,
        .freq_base=160000.0f, .freq_scale=1.0f/16.0f,
        .ext_factor=1.0f,
        .attn_factor=1.0f/(1.0f+0.1f*logf(16.0f)),
        .beta_fast=32.0f, .beta_slow=1.0f, .src2=false,
    };

    id<MTLCommandBuffer> command = [context.queue commandBuffer];
    id<MTLCounterSampleBuffer> attention_samples = (retain_batch)
        ? context.prefillLayerAttentionSamples : nil;
    id<MTLFence> attention_fence = attention_samples ? [context.device newFence] : nil;
    if (attention_samples && (!attention_fence || context.prefillLayerAttentionCommand))
        return fail_with_message(error, error_bytes,
            @"attention profile requires a fence and consumed previous packet");
    id<MTLComputeCommandEncoder> encoder = stage_compute_encoder(command, attention_samples, 0u);
    if (!command || !encoder) return fail_with_message(error, error_bytes,
        @"failed to create prefill layer-2 indexed tile encoder");

    [encoder setComputePipelineState:context.f16AlignedPrefillPipeline];
    [encoder setBytes:&q_projection length:sizeof(q_projection) atIndex:0];
    [encoder setBuffer:q_weights offset:inner[0] atIndex:1];
    [encoder setBuffer:q_norm_buffer
                offset:(NSUInteger)(position_start-q_norm_start)*q_rank*sizeof(float)
               atIndex:2];
    [encoder setBuffer:indexer_q_buffer offset:0 atIndex:3];
    [encoder setThreadgroupMemoryLength:6144u atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(tile_rows/32u,128,1)
         threadsPerThreadgroup:MTLSizeMake(128,1,1)];

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 1u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    [encoder setComputePipelineState:context.ropeTailPipeline];
    [encoder setBytes:&indexer_rope length:sizeof(indexer_rope) atIndex:0];
    [encoder setBuffer:indexer_q_buffer offset:0 atIndex:1];
    [encoder setBuffer:position_buffer offset:0 atIndex:2];
    [encoder setBuffer:indexer_q_buffer offset:0 atIndex:3];
    [encoder setBuffer:indexer_q_buffer offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(attention_heads,tile_rows,1)
         threadsPerThreadgroup:MTLSizeMake(128,1,1)];

    [encoder setComputePipelineState:context.indexerQatPipeline];
    [encoder setBytes:&qat length:sizeof(qat) atIndex:0];
    [encoder setBuffer:indexer_q_buffer offset:0 atIndex:1];
    [encoder setThreadgroupMemoryLength:256u*sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(tile_rows*attention_heads,1,1)
         threadsPerThreadgroup:MTLSizeMake(128,1,1)];

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 2u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    [encoder setComputePipelineState:context.f16AlignedPrefillPipeline];
    [encoder setBytes:&weight_projection length:sizeof(weight_projection) atIndex:0];
    [encoder setBuffer:weight_weights offset:inner[1] atIndex:1];
    [encoder setBuffer:attn_norm_buffer
                offset:(NSUInteger)(position_start-attn_norm_start)*n_embd*sizeof(float)
               atIndex:2];
    [encoder setBuffer:indexer_weight_buffer offset:0 atIndex:3];
    [encoder setThreadgroupMemoryLength:6144u atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(tile_rows/32u,1,1)
         threadsPerThreadgroup:MTLSizeMake(128,1,1)];

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 3u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    [encoder setComputePipelineState:context.indexerScoreBatchPipeline];
    [encoder setBytes:&score_args length:sizeof(score_args) atIndex:0];
    [encoder setBuffer:indexer_q_buffer offset:0 atIndex:1];
    [encoder setBuffer:indexer_weight_buffer offset:0 atIndex:2];
    [encoder setBuffer:indexer_compressed_buffer offset:0 atIndex:3];
    [encoder setBuffer:score_buffer offset:0 atIndex:4];
    dispatch_prefill_indexer_scores(encoder, compressed_capacity, tile_rows);

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 4u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    const uint32_t topk_dispatches = encode_indexer_topk(context, encoder,
        score_buffer, topk_buffer, topk_args, error, error_bytes);
    if (!topk_dispatches) {
        [encoder endEncoding];
        return 0;
    }

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 5u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    [encoder setComputePipelineState:context.indexedTopkSortPipeline];
    [encoder setBytes:&sort_args length:sizeof(sort_args) atIndex:0];
    [encoder setBuffer:topk_buffer offset:0 atIndex:1];
    [encoder setBuffer:sorted_buffer offset:0 atIndex:2];
    [encoder setThreadgroupMemoryLength:topk_row_bytes atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(tile_rows,1,1)
         threadsPerThreadgroup:MTLSizeMake(top_k,1,1)];

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 6u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    uint32_t comp_elements = compressed_capacity*attention_width;
    [encoder setComputePipelineState:context.cpyF32F16Pipeline];
    [encoder setBytes:&comp_elements length:sizeof(comp_elements) atIndex:0];
    [encoder setBuffer:attn_compressed_buffer offset:0 atIndex:1];
    [encoder setBuffer:comp_f16 offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake((comp_elements+1023u)/1024u,1,1)
         threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 7u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    [encoder setComputePipelineState:context.indexedAttentionPrefillPipeline];
    [encoder setBytes:&attention_args length:sizeof(attention_args) atIndex:0];
    [encoder setBuffer:q_batch offset:q_batch_offset atIndex:1];
    [encoder setBuffer:raw_ring offset:0 atIndex:2];
    [encoder setBuffer:comp_f16 offset:0 atIndex:3];
    [encoder setBuffer:sorted_buffer offset:0 atIndex:4];
    [encoder setBuffer:sinks_buffer offset:inner[2] atIndex:5];
    [encoder setBuffer:heads_buffer offset:0 atIndex:6];
    [encoder setThreadgroupMemoryLength:128u*4u*sizeof(uint16_t) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(tile_rows,8,1)
         threadsPerThreadgroup:MTLSizeMake(32,8,1)];

    if (attention_samples) {
        encoder = next_tail_profile_encoder(encoder, command,
            attention_samples, attention_fence, 8u);
        if (!encoder) return fail_with_message(error, error_bytes,
            @"failed to create attention profile encoder");
    }

    [encoder setComputePipelineState:context.ropeTailPipeline];
    [encoder setBytes:&inverse_rope length:sizeof(inverse_rope) atIndex:0];
    [encoder setBuffer:heads_buffer offset:0 atIndex:1];
    [encoder setBuffer:position_buffer offset:0 atIndex:2];
    [encoder setBuffer:heads_buffer offset:0 atIndex:3];
    [encoder setBuffer:heads_buffer offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(attention_heads,tile_rows,1)
         threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder endEncoding];

    // Native attention writes the complete tail input directly; no per-tile
    // snapshot blits or duplicate heads allocation are needed.
    const uint32_t retained_rows = context.prefillLayerEvenBatchRows.unsignedIntValue;
    if (retain_batch && (retained_rows != 0u || position_start != chunk_start ||
        heads_buffer.length != 4096ull*q_row_bytes))
        return fail_with_message(error, error_bytes, @"invalid long-prefill attention batch");

    const double wall_start = monotonic_ms();
    [command commit];
    if (!command_succeeded(command, error, error_bytes)) return 0;
    if (attention_samples) {
        context.prefillLayerAttentionCommand = command;
        context.prefillLayerAttentionSampleCount = @(18u);
    }
    const float diagnostic_q0 = ((const float *)indexer_q_buffer.contents)[0];
    const float diagnostic_w0 = ((const float *)indexer_weight_buffer.contents)[0];
    const float diagnostic_score0 = ((const float *)score_buffer.contents)[0];
    if (diagnostic_score0 == 0.0f) {
        return fail_with_message(error, error_bytes,
            [NSString stringWithFormat:
                @"prefill layer-2 indexed tile produced a zero first score (q0=%a w0=%a)",
                diagnostic_q0, diagnostic_w0]);
    }
    if (scores) memcpy(scores, score_buffer.contents, tile_rows*score_row_bytes);
    if (topk) memcpy(topk, topk_buffer.contents, tile_rows*topk_row_bytes);
    if (kqv_back) memcpy(kqv_back, heads_buffer.contents, tile_rows*q_row_bytes);
    if (retain_batch) context.prefillLayerEvenBatchRows = @(retained_rows+tile_rows);
    if (heads_out) *heads_out = heads_buffer;
    result->dispatches += 9u+topk_dispatches;
    result->wrapped_model_ranges += 3u;
    for (uint32_t index = 0; index < 3u; index++) {
        if (matches[index]) result->pointer_matches++;
    }
    result->wall_ms += monotonic_ms()-wall_start;
    result->gpu_ms += gpu_elapsed_ms(command);
    return 1;
}
