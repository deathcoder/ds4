from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
a='@property(nonatomic, strong) id<MTLBuffer> prefillLayer2ContinuationHeads;'
s=s.replace(a,a+'''
// One chunk of inverse-RoPE attention heads, reused across even executor layers.
@property(nonatomic, strong) id<MTLBuffer> prefillLayerEvenBatchHeads;
@property(nonatomic, strong) NSNumber *prefillLayerEvenBatchRows;''',1)
a=s.index('static int rust_star_metal_run_prefill_layer2_indexed_tile(');b=s.index('int rust_star_metal_run_prefill_layer2_continuation_sparse_tile(',a);part=s[a:b]
needle='    const double wall_start = monotonic_ms();\n    [command commit];'
assert part.count(needle)==1
part=part.replace(needle,'''    // Retain completed per-token arithmetic in the same command. The FFN tail
    // consumes the whole chunk only after every tile has filled its own rows.
    const BOOL retain_batch = context.prefillLayerEvenBatchRows != nil;
    const uint32_t retained_rows = context.prefillLayerEvenBatchRows.unsignedIntValue;
    if (retain_batch) {
        const uint32_t chunk_start = prefill_chunk_start((__bridge void *)context);
        if (retained_rows > 4096u-tile_rows || tile_start != chunk_start+retained_rows ||
            context.prefillLayerEvenBatchHeads.length != 4096ull*q_row_bytes)
            return fail_with_message(error, error_bytes, @"nonconsecutive long-prefill attention tile");
        id<MTLBlitCommandEncoder> retain = [command blitCommandEncoder];
        if (!retain) return fail_with_message(error, error_bytes, @"cannot retain long-prefill attention tile");
        [retain copyFromBuffer:heads_buffer sourceOffset:0
                     toBuffer:context.prefillLayerEvenBatchHeads
            destinationOffset:(NSUInteger)retained_rows*q_row_bytes size:tile_rows*q_row_bytes];
        [retain endEncoding];
    }

'''+needle,1)
part=part.replace('    if (heads_out) *heads_out = heads_buffer;','    if (retain_batch) context.prefillLayerEvenBatchRows = @(retained_rows+tile_rows);\n    if (heads_out) *heads_out = heads_buffer;',1)
s=s[:a]+part+s[b:]
a=s.index('int rust_star_metal_run_prefill_layer2_continuation_tail(')
begin='''int rust_star_metal_begin_long_prefill_even_attention(
    void *opaque_context, char *error, size_t error_bytes)
{
    @autoreleasepool {
        if (!opaque_context) return fail_with_message(error, error_bytes, @"null long-prefill context");
        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
        if (!context.prefillLayerChunkStart || context.prefillLayerEvenBatchRows ||
            context.prefillKvRows != context.prefillLayerChunkStart.unsignedIntValue+4096u)
            return fail_with_message(error, error_bytes, @"long-prefill attention batch is not ready");
        const NSUInteger bytes = 4096ull*32768u*sizeof(float);
        if (!context.prefillLayerEvenBatchHeads)
            context.prefillLayerEvenBatchHeads = [context.device newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
        if (context.prefillLayerEvenBatchHeads.length != bytes)
            return fail_with_message(error, error_bytes, @"cannot allocate long-prefill attention batch");
        context.prefillLayerEvenBatchRows = @0;
        return 1;
    }
}

'''
s=s[:a]+begin+s[a:]
a=s.index('int rust_star_metal_run_prefill_layer2_continuation_tail(');b=s.index('\nint ',a+5);part=s[a:b]
part=part.replace('    const BOOL production_batch =','    const BOOL production_even_batch = (layer_index == 2u || layer_index == 4u) &&\n        tile_start != 0u && tile_start == chunk_start && rows == 4096u;\n    const BOOL production_batch =',1)
part=part.replace('        production_layer5_prefix || production_layer5_batch;','        production_layer5_prefix || production_layer5_batch || production_even_batch;',1)
part=part.replace('        id<MTLBuffer> heads_buffer = production_layer5_prefix ||','        if (production_even_batch && context.prefillLayerEvenBatchRows.unsignedIntValue != 4096u)\n            return fail_with_message(error, error_bytes, @"long-prefill tail requires all attention rows");\n        id<MTLBuffer> heads_buffer = production_even_batch ? context.prefillLayerEvenBatchHeads : production_layer5_prefix ||',1)
needle='        result->wall_ms += monotonic_ms()-wall_start;'
assert part.count(needle)==1
part=part.replace(needle,'        if (production_even_batch) context.prefillLayerEvenBatchRows = nil;\n'+needle,1)
s=s[:a]+part+s[b:];p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();a='int rust_star_metal_begin_long_prefill_chunk(';i=s.index(a);s=s[:i]+'int rust_star_metal_begin_long_prefill_even_attention(void *context, char *error, size_t error_bytes);\n'+s[i:];p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();a='        fn rust_star_metal_begin_long_prefill_chunk(';i=s.index(a);s=s[:i]+'''        fn rust_star_metal_begin_long_prefill_even_attention(
            context: *mut c_void, error: *mut c_char, error_bytes: usize,
        ) -> i32;
'''+s[i:];p.write_text(s)
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();a=s.index('fn run_prefill_chunk_layer');b=s.index('fn rebuild_long_prefill_prefix');part=s[a:b]
a=part.index('        let mut tail: Vec<Vec<f32>>');b=part.index('        profile.record(',a);part=part[:a]+part[b:]
needle='        for position in (start..start + 4096).step_by(32) {'
# Insert only in the even branch, after host scratch setup.
a=part.index('        let scratch_start');i=part.index(needle,a);part=part[:i]+'''        checked!(rust_star_metal_begin_long_prefill_even_attention(
            context.0, error.as_mut_ptr(), error.len()
        ));
'''+part[i:]
a=part.index('            let stage_start = profile.start();\n            let before = (tile.wall_ms, tile.gpu_ms);');b=part.index('            raw.wall_ms += tile.wall_ms;',a)
part=part[:a]+'''            if tile.dispatches != 10 || tile.wrapped_model_ranges != 3 || tile.pointer_matches != 3 {
                return Err(Error::invalid("even chunk attention schedule mismatch"));
            }
'''+part[b:]
needle='            raw.gpu_ms += tile.gpu_ms;\n        }'
tail='''
        let stage_start = profile.start();
        let mut tail = RawSparseIndexedResult::default();
        checked!(rust_star_metal_run_prefill_layer2_continuation_tail(
            context.0, model.mapping_pointer(), model.bytes(), &prepared.weights,
            if layer == 2 { 2 } else { 4 }, start, 4096,
'''+''.join('            std::ptr::null_mut(),\n' for _ in range(13))+'''            &mut tail, error.as_mut_ptr(), error.len()
        ));
        profile.record(start, layer, layer, "tail", stage_start, (0.0, 0.0), (tail.wall_ms, tail.gpu_ms));
        if tail.dispatches != 26 || tail.wrapped_model_ranges != 14 || tail.pointer_matches != 14 {
            return Err(Error::invalid("even chunk batched tail schedule mismatch"));
        }
        raw.wall_ms += tail.wall_ms;
        raw.gpu_ms += tail.gpu_ms;
'''
assert part.count(needle)==1;part=part.replace(needle,needle+tail,1)
a=s.index('fn run_prefill_chunk_layer');b=s.index('fn rebuild_long_prefill_prefix');s=s[:a]+part+s[b:];p.write_text(s)
