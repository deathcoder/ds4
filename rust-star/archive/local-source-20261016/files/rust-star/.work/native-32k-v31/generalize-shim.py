from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();Path('rust-star/.work/native-32k-v31/metal_shim.baseline.m').write_text(s)
pos=s.index('@property(nonatomic,strong) NSNumber *prefillLayerRemainingCompleted;') if '@property(nonatomic,strong) NSNumber *prefillLayerRemainingCompleted;' in s else s.index('@property(nonatomic, strong) NSNumber *prefillLayerRemainingCompleted;')
s=s[:pos]+'''@property(nonatomic, strong) NSNumber *prefillLayerChunkStart;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSDictionary *> *prefillLayerChunkStates;
@property(nonatomic, strong) id<MTLBuffer> prefillLayerChunkHc;
''' +s[pos:]
i=s.index('static id<MTLBuffer> retained_prefill_layer_buffer(')
s=s[:i]+'''/* A chunk keeps absolute positions in KV/compressor history. The existing 8K
 * diagnostic defaults to its second chunk; later chunks set this explicitly. */
static uint32_t prefill_chunk_start(void *opaque_context)
{
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    return context.prefillLayerChunkStart ? context.prefillLayerChunkStart.unsignedIntValue : 4096u;
}

static void save_prefill_chunk_state(RustStarMetalContext *context,
                                    uint32_t layer, uint32_t slot)
{
    if (!context.prefillLayerChunkStates)
        context.prefillLayerChunkStates = [NSMutableDictionary dictionary];
    NSMutableDictionary *state = [NSMutableDictionary dictionary];
    for (NSString *role in @[@"FullKv", @"AttnCompressed", @"AttnStateKv",
            @"AttnStateScore", @"IndexerCompressed", @"IndexerStateKv", @"IndexerStateScore"]) {
        NSString *key = [NSString stringWithFormat:@"prefillLayer%u%@", slot, role];
        if ([context respondsToSelector:NSSelectorFromString(key)]) {
            id value = [context valueForKey:key];
            if (value) state[role] = value;
        }
    }
    context.prefillLayerChunkStates[@(layer)] = state;
}

''' +s[i:]
# Scoped function transformation avoids confusing model widths with context extents.
def edit(name,func):
 global s
 a=s.index(name+'(');a=s.rfind('\n',0,a)+1
 b=s.index('\n}\n',a)+3
 s=s[:a]+func(s[a:b])+s[b:]
def geometry(t):
 t=t.replace('{\n    enum {','{\n    const uint32_t chunk_start = prefill_chunk_start(opaque_context);\n    const uint32_t chunk_end = chunk_start + 4096u;\n    enum {',1)
 t=t.replace('position_start != 4096u','position_start != chunk_start').replace('tile_start < 4096u','tile_start < chunk_start').replace('tile_start > 8160u','tile_start > chunk_end-32u').replace('(tile_start-4096u)','(tile_start-chunk_start)').replace('tile_start-4096u','tile_start-chunk_start')
 t=t.replace('context.prefillKvRows != 8192u','context.prefillKvRows != chunk_end')
 t=t.replace('(NSUInteger)8192u*','(NSUInteger)chunk_end*')
 return t
for name in ['rust_star_metal_run_prefill_layer3_continuation_ingress','rust_star_metal_run_prefill_layer3_continuation_batch_compressor','rust_star_metal_run_prefill_layer3_continuation_batch_attention','rust_star_metal_run_prefill_layer4_continuation_boundary','rust_star_metal_run_prefill_layer2_continuation_tail']:
 edit(name,geometry)
edit('rust_star_metal_run_prefill_layer4_continuation_boundary',lambda t:t.replace('(NSUInteger)2048u*attn_compressed_row_bytes','(NSUInteger)(chunk_end/4u)*attn_compressed_row_bytes').replace('(NSUInteger)2048u*indexer_compressed_row_bytes','(NSUInteger)(chunk_end/4u)*indexer_compressed_row_bytes'))
edit('rust_star_metal_run_prefill_layer2_continuation_tail',lambda t:t.replace('tile_start == 4096u','tile_start == chunk_start'))
edit('rust_star_metal_run_prefill_layer3_continuation_batch_attention',lambda t:t.replace('position_start == 0u ? 0u : 3968u','position_start == 0u ? 0u : position_start-128u').replace('position_start == 0u ? 32u : 64u','(position_start+rows)/ratio'))
edit('rust_star_metal_prepare_prefill_layer5_continuation_tile',lambda t:t.replace('enum { rows = 32, continuation_start = 4096, hc_dim = 16384 };','const uint32_t continuation_start = prefill_chunk_start(opaque_context);\n    enum { rows = 32, hc_dim = 16384 };').replace('tile_start > 8160u','tile_start > continuation_start+4096u-rows'))
edit('rust_star_metal_finish_prefill_layer5_continuation',lambda t:t.replace('compressed_offset_rows = 32,','').replace('{\n    enum {','{\n    const uint32_t compressed_offset_rows = prefill_chunk_start(opaque_context)/128u;\n    enum {',1))
# Preserve layers whose named slots are subsequently reused by the 8K diagnostic.
for name,layer in [('rust_star_metal_checksum_prefill_layer3_continuation_hc',3),('rust_star_metal_checksum_prefill_layer4_continuation_hc',4),('rust_star_metal_checksum_prefill_layer5_continuation_tail',5)]:
 edit(name,lambda t,layer=layer:t.replace('        return 1;','        if (!context.prefillLayerChunkStart) save_prefill_chunk_state(context, '+str(layer)+', '+str(layer)+');\n        return 1;'))
# Bootstrap storage already grows from its passed frontier.
edit('rust_star_metal_run_prefill_layer0_boundary',lambda t:t.replace('(prefill_rows != 2048u && prefill_rows != 4096u &&\n         prefill_rows != 8192u)','(prefill_rows != 2048u && (prefill_rows < 4096u ||\n         prefill_rows > 32768u || prefill_rows%4096u != 0u))'))
# Indexed attention uses the production 4352-row physical ring at any chunk.
def indexed(t):
 t=t.replace('        raw_start = 3968,\n','').replace('        compressed_capacity = 2048,\n','')
 t=t.replace('    if (position_start < 4096u || position_start > 8160u ||\n        (position_start-4096u)%tile_rows != 0u)', '    const uint32_t chunk_start = prefill_chunk_start((__bridge void *)context);\n    const uint32_t chunk_end = chunk_start+batch_tokens;\n    const uint32_t raw_start = chunk_start-128u;\n    const uint32_t compressed_capacity = chunk_end/4u;\n    if (position_start < chunk_start || position_start > chunk_end-tile_rows ||\n        (position_start-chunk_start)%tile_rows != 0u)')
 t=t.replace('8192u*attention_width','chunk_end*attention_width').replace('(position_start-4096u)','(position_start-chunk_start)')
 a=t.index('    memcpy(raw_ring.contents,');b=t.index('    for (uint32_t row = 0; row < tile_rows;',a)
 t=t[:a]+'''    for (uint32_t position = chunk_end-raw_capacity; position < chunk_end; position++) {
        memcpy((uint8_t *)raw_ring.contents+(NSUInteger)(position%raw_capacity)*kv_row_bytes,
               (const uint8_t *)full_kv_buffer.contents+(NSUInteger)position*kv_row_bytes,
               kv_row_bytes);
    }
''' +t[b:]
 t=t.replace('MTLSizeMake(64,4,1)','MTLSizeMake((compressed_capacity+31u)/32u,4,1)')
 return t
edit('rust_star_metal_run_prefill_layer2_indexed_tile',indexed)
edit('rust_star_metal_run_prefill_layer2_continuation_sparse_tile',lambda t:t.replace('            4096u,','            prefill_chunk_start(opaque_context),'))
# Only a completed chunk can expose its last HC row to the output head.
edit('rust_star_metal_run_output_head',lambda t:t.replace('const BOOL complete_second_chunk = prefill_rows == 8192u &&','const BOOL complete_second_chunk = prefill_rows >= 8192u &&\n                prefill_rows <= 32768u && prefill_rows%4096u == 0u &&'))
p.write_text(s)
