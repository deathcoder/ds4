from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();code=r'''
int rust_star_metal_begin_long_prefill_chunk(void *opaque_context,
    uint32_t chunk_start, char *error, size_t error_bytes)
{
    if (!opaque_context || chunk_start < 8192u || chunk_start >= 32768u || chunk_start%4096u)
        return fail_with_message(error, error_bytes, @"invalid long-prefill chunk start");
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    if (context.prefillKvRows != chunk_start || context.prefillLayerRemainingCompleted.unsignedIntValue != 42u)
        return fail_with_message(error, error_bytes, @"long-prefill chunk requires its complete predecessor");
    for (uint32_t layer = 0; layer < 43u; layer++) {
        if (!context.prefillLayerChunkStates[@(layer)]) save_prefill_chunk_state(context, layer, layer);
        if (!context.prefillLayerChunkStates[@(layer)][@"FullKv"])
            return fail_with_message(error, error_bytes, @"long-prefill layer history is missing");
    }
    context.prefillLayerChunkStart = @(chunk_start);
    context.prefillLayerRemainingCompleted = nil;
    context.prefillLayerChunkHc = nil;
    /* Later chunks need history, not past query/HC activations. Clear the
     * named aliases too, so the dictionary is the sole retained history. */
    for (uint32_t layer = 3; layer < 43u; layer++) {
        for (NSString *role in @[@"FullQ", @"FullQNorm", @"FullNorm", @"InputHc",
                @"AttnSplit", @"AfterAttentionHc", @"AfterFfnHc", @"PrefixAfterFfnHc"]) {
            NSString *key = [NSString stringWithFormat:@"prefillLayer%u%@", layer, role];
            if ([context respondsToSelector:NSSelectorFromString(key)]) [context setValue:nil forKey:key];
        }
    }
    context.prefillLayer2ContinuationFullAfterFfnHc = nil;
    context.prefillLayer3ContinuationFullAfterFfnHc = nil;
    context.prefillLayer4ContinuationFullAfterFfnHc = nil;
    context.prefillLayer5ContinuationFullAfterFfnHc = nil;
    return 1;
}

int rust_star_metal_bind_long_prefill_layer(void *opaque_context,
    uint32_t layer, char *error, size_t error_bytes)
{
    if (!opaque_context || layer < 3u || layer > 42u)
        return fail_with_message(error, error_bytes, @"invalid long-prefill layer");
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    if (!context.prefillLayerChunkHc || context.prefillLayerChunkHc.length != 4096ull*16384u*sizeof(float))
        return fail_with_message(error, error_bytes, @"long-prefill layer requires its predecessor HC");
    const uint32_t slot = layer%2u ? 3u : 4u;
    NSDictionary *state = context.prefillLayerChunkStates[@(layer)];
    if (!state[@"FullKv"] || !state[@"AttnCompressed"] || !state[@"AttnStateKv"] || !state[@"AttnStateScore"])
        return fail_with_message(error, error_bytes, @"long-prefill compressor history is incomplete");
    for (NSString *role in state) {
        NSString *key = [NSString stringWithFormat:@"prefillLayer%u%@", slot, role];
        [context setValue:state[role] forKey:key];
    }
    const NSUInteger end = prefill_chunk_start(opaque_context)+4096u;
    NSDictionary *widths = slot == 3u ? @{@"FullQ":@32768, @"AttnSplit":@24}
        : @{@"FullQ":@32768, @"FullQNorm":@1024, @"FullNorm":@4096, @"InputHc":@16384, @"AttnSplit":@24};
    for (NSString *role in widths) {
        id<MTLBuffer> buffer = [context.device newBufferWithLength:end*[widths[role] unsignedIntegerValue]*sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        if (!buffer) return fail_with_message(error, error_bytes, @"long-prefill transient allocation failed");
        [context setValue:buffer forKey:[NSString stringWithFormat:@"prefillLayer%u%@", slot, role]];
    }
    if (slot == 3u) {
        context.prefillLayer4ContinuationFullAfterFfnHc = context.prefillLayerChunkHc;
        context.prefillLayer3ContinuationFullNorm = nil;
        context.prefillLayer3ContinuationHeads = nil;
        context.prefillLayer5ContinuationAfterAttentionHc = nil;
        context.prefillLayer5ContinuationFullAfterFfnHc = nil;
    } else {
        if (!state[@"IndexerCompressed"] || !state[@"IndexerStateKv"] || !state[@"IndexerStateScore"])
            return fail_with_message(error, error_bytes, @"long-prefill indexer history is incomplete");
        context.prefillLayer3ContinuationFullAfterFfnHc = context.prefillLayerChunkHc;
        context.prefillLayer4ContinuationFullAfterFfnHc = nil;
        context.prefillLayer4ContinuationNorm = nil;
        context.prefillLayer4ContinuationHeads = nil;
    }
    return 1;
}

int rust_star_metal_finish_long_prefill_layer(void *opaque_context,
    uint32_t layer, uint64_t *checksum, char *error, size_t error_bytes)
{
    if (!opaque_context || !checksum || layer > 42u)
        return fail_with_message(error, error_bytes, @"invalid long-prefill layer completion");
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    const NSUInteger hc_bytes = 4096ull*16384u*sizeof(float);
    if (layer < 2u) {
        id<MTLBuffer> hc = layer == 0u ? context.prefillLayer1InputHc : context.prefillLayer2InputHc;
        NSUInteger offset = (NSUInteger)prefill_chunk_start(opaque_context)*16384u*sizeof(float);
        if (!hc || hc.length < offset+hc_bytes)
            return fail_with_message(error, error_bytes, @"bootstrap HC history is incomplete");
        *checksum = fnv1a_u32_words((const uint8_t *)hc.contents+offset, hc_bytes/sizeof(float));
        save_prefill_chunk_state(context, layer, layer);
        return 1;
    }
    uint32_t slot = layer == 2u ? 2u : layer%2u ? 3u : 4u;
    id<MTLBuffer> hc = layer == 2u ? context.prefillLayer2ContinuationFullAfterFfnHc
        : layer%2u ? context.prefillLayer5ContinuationFullAfterFfnHc
        : context.prefillLayer4ContinuationFullAfterFfnHc;
    if (!hc || hc.length != hc_bytes)
        return fail_with_message(error, error_bytes, @"complete long-prefill layer HC is unavailable");
    save_prefill_chunk_state(context, layer, slot);
    context.prefillLayerChunkHc = hc;
    *checksum = fnv1a_u32_words(hc.contents, hc_bytes/sizeof(float));
    if (layer == 42u) {
        context.prefillLayer42AfterFfnHc = hc;
        context.prefillLayerRemainingCompleted = @42;
    }
    return 1;
}
'''
# Extract only the validated production Q-B/head-normalization operations.
a=s.index('int rust_star_metal_run_prefill_layer2_sparse_transition(');b=s.index('        NSUInteger q_b_inner = 0u;',a);c=s.index('        const double batch_gpu_ms = gpu_elapsed_ms(command);',b)
body=s[b:c]
code+='''\nint rust_star_metal_prepare_long_prefill_layer2_q(void *opaque_context,
    const void *model_mapping, uint64_t model_bytes, uint64_t q_b_offset,
    uint64_t q_b_bytes, char *error, size_t error_bytes)
{
    if (!opaque_context || !model_mapping) return 0;
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    const uint32_t continuation_start = prefill_chunk_start(opaque_context);
    enum { continuation_rows = 4096, q_dim = 32768, q_rank = 1024 };
    if (!context.prefillLayer2FullQNorm || context.prefillKvRows != continuation_start+continuation_rows)
        return fail_with_message(error, error_bytes, @"long-prefill layer-2 Q requires the complete bootstrap");
'''+body+'''        context.prefillLayer2ContinuationQ = q_batch;
        return 1;
}
'''
s+='\n'+code;p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();i=s.index('int rust_star_metal_release_remaining_prefill_transients(');s=s[:i]+'''int rust_star_metal_begin_long_prefill_chunk(void *context, uint32_t chunk_start, char *error, size_t error_bytes);
int rust_star_metal_bind_long_prefill_layer(void *context, uint32_t layer, char *error, size_t error_bytes);
int rust_star_metal_finish_long_prefill_layer(void *context, uint32_t layer, uint64_t *checksum, char *error, size_t error_bytes);
int rust_star_metal_prepare_long_prefill_layer2_q(void *context, const void *model_mapping, uint64_t model_bytes, uint64_t q_b_offset, uint64_t q_b_bytes, char *error, size_t error_bytes);

''' +s[i:];p.write_text(s)
