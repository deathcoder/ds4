from pathlib import Path
import json
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();s=s.replace('layer_index == 5u && tile_start == chunk_start && rows == 4096u;', 'layer_index == 5u && tile_start != 0u && tile_start == chunk_start && rows == 4096u;',1)
s+='''
/* Rebuild the 4K prefix with the same sparse/Flash schedules as later chunks.
 * The older monolithic 4K schedule is a bootstrap, not a C0 prefix target. */
int rust_star_metal_begin_long_prefill_prefix(void *opaque_context,
    uint64_t *checksums, char *error, size_t error_bytes)
{
    if (!opaque_context || !checksums) return 0;
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    if (context.prefillKvRows != 4096u || !context.prefillLayer3InputHc)
        return fail_with_message(error, error_bytes, @"exact prefix requires completed layers 0--2");
    context.prefillLayerChunkStart = @0;
    context.prefillLayerChunkStates = [NSMutableDictionary dictionary];
    NSArray *hc = @[context.prefillLayer1InputHc, context.prefillLayer2InputHc, context.prefillLayer3InputHc];
    for (uint32_t layer = 0; layer < 3u; layer++) {
        id<MTLBuffer> buffer = hc[layer];
        if (buffer.length != 4096ull*16384u*sizeof(float))
            return fail_with_message(error, error_bytes, @"invalid bootstrap HC extent");
        checksums[layer] = fnv1a_u32_words(buffer.contents, 4096ull*16384u);
        save_prefill_chunk_state(context, layer, layer);
    }
    context.prefillLayerChunkHc = context.prefillLayer3InputHc;
    return 1;
}

int rust_star_metal_bind_long_prefill_prefix(void *opaque_context,
    uint32_t layer, char *error, size_t error_bytes)
{
    if (!opaque_context || layer < 3u || layer > 42u) return 0;
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    if (!context.prefillLayerChunkHc || context.prefillLayerChunkStart.unsignedIntValue != 0u)
        return fail_with_message(error, error_bytes, @"prefix layer requires preceding HC");
    const uint32_t slot = layer%2u ? 5u : 4u;
    NSDictionary *widths = slot == 5u
        ? @{@"AttnStateKv":@65536, @"AttnStateScore":@65536}
        : @{@"FullQ":@(4096ull*32768), @"FullKv":@(4096ull*512),
            @"InputHc":@(4096ull*16384), @"AttnSplit":@(4096ull*24),
            @"AttnCompressed":@(1024ull*512), @"AttnStateKv":@8192,
            @"AttnStateScore":@8192, @"IndexerCompressed":@(1024ull*128),
            @"IndexerStateKv":@2048, @"IndexerStateScore":@2048};
    for (NSString *role in widths) {
        NSUInteger count = [widths[role] unsignedIntegerValue];
        id<MTLBuffer> buffer = [context.device newBufferWithLength:count*sizeof(float) options:MTLResourceStorageModeShared];
        if (!buffer) return fail_with_message(error, error_bytes, @"prefix buffer allocation failed");
        memset(buffer.contents, 0, buffer.length);
        if ([role hasSuffix:@"StateScore"])
            for (NSUInteger i = 0; i < count; i++) ((uint32_t *)buffer.contents)[i] = 0xff800000u;
        [context setValue:buffer forKey:[NSString stringWithFormat:@"prefillLayer%u%@",slot,role]];
    }
    if (slot == 5u) {
        context.prefillLayer4AfterFfnHc = context.prefillLayerChunkHc;
        context.prefillLayer3ContinuationHeads = nil;
    } else {
        context.prefillLayer3AfterFfnHc = context.prefillLayerChunkHc;
        context.prefillLayer4FullQNorm = nil;
        context.prefillLayer4FullNorm = nil;
        context.prefillLayer4PrefixHeads = nil;
    }
    return 1;
}

int rust_star_metal_finish_long_prefill_prefix(void *opaque_context,
    uint32_t layer, uint64_t *checksum, char *error, size_t error_bytes)
{
    if (!opaque_context || !checksum || layer < 3u || layer > 42u) return 0;
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    uint32_t slot = layer%2u ? 5u : 4u;
    id<MTLBuffer> hc = slot == 5u ? context.prefillLayer5AfterFfnHc : context.prefillLayer4AfterFfnHc;
    if (!hc || hc.length != 4096ull*16384u*sizeof(float))
        return fail_with_message(error, error_bytes, @"prefix layer HC is incomplete");
    save_prefill_chunk_state(context, layer, slot);
    context.prefillLayerChunkHc = hc;
    *checksum = fnv1a_u32_words(hc.contents, 4096ull*16384u);
    if (layer == 42u) {
        context.prefillLayer42AfterFfnHc = hc;
        context.prefillLayerRemainingCompleted = @42;
    }
    return 1;
}
''';p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();pos=s.index('int rust_star_metal_begin_long_prefill_chunk');s=s[:pos]+'''int rust_star_metal_begin_long_prefill_prefix(void *, uint64_t *, char *, size_t);
int rust_star_metal_bind_long_prefill_prefix(void *, uint32_t, char *, size_t);
int rust_star_metal_finish_long_prefill_prefix(void *, uint32_t, uint64_t *, char *, size_t);
'''+s[pos:];p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();pos=s.index('        fn rust_star_metal_begin_long_prefill_chunk');s=s[:pos]+'''        fn rust_star_metal_begin_long_prefill_prefix(context: *mut c_void, checksums: *mut u64, error: *mut c_char, error_bytes: usize) -> i32;
        fn rust_star_metal_bind_long_prefill_prefix(context: *mut c_void, layer: u32, error: *mut c_char, error_bytes: usize) -> i32;
        fn rust_star_metal_finish_long_prefill_prefix(context: *mut c_void, layer: u32, checksum: *mut u64, error: *mut c_char, error_bytes: usize) -> i32;
'''+s[pos:];p.write_text(s)
w=Path('rust-star/.work/remaining-layers-v30/all-layers');a=json.loads((w/'prefix-a/hashes.json').read_text());b=json.loads((w/'prefix-b/hashes.json').read_text());assert a==b
p=Path('rust-star/runtime/src/long_prefill_checksums.rs');s=p.read_text();s+='\nconst LONG_PREFILL_PREFIX_CHECKSUMS: [u64; 43] = ['+', '.join('0x'+a[f'oracle_hc_ffn_post-{l}_pos0.bin']['fnv1a_u32'] for l in range(43))+'];\n';p.write_text(s)
