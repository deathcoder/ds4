from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(s.count(a),a)
 s=s.replace(a,b)
replace('        const uint32_t compressor_rows = prefill_rows/compressor_ratio;', '''        // Native continuation retains history only for KV/compressor state.
        // HC and normalized inputs are consumed within the current 4K chunk.
        const NSUInteger bootstrap_base = context.prefillLayerChunkStart
            ? context.prefillLayerChunkStart.unsignedIntValue : 0u;
        const NSUInteger bootstrap_rows = context.prefillLayerChunkStart
            ? 4096u : prefill_rows;
        if (position_start < bootstrap_base ||
            (NSUInteger)position_start-bootstrap_base+rows > bootstrap_rows)
            return fail_with_message(error, error_bytes,
                @"bootstrap transient rows exceed the active chunk");
        const NSUInteger bootstrap_row = (NSUInteger)position_start-bootstrap_base;
        const uint32_t compressor_rows = prefill_rows/compressor_ratio;''')
for prop,width in [('prefillLayer1InputHc','hc_dim'),('prefillLayer2FullQNorm','q_rank'),('prefillLayer2FullAttnNorm','n_embd'),('prefillLayer2InputHc','hc_dim'),('prefillLayer2AttnSplit','mix_hc')]:
 replace(f'RUST_STAR_GROW_PREFILL_BUFFER({prop},\n                prefill_rows*{width}', f'RUST_STAR_GROW_PREFILL_BUFFER({prop},\n                bootstrap_rows*{width}')
for var,width,macro,indent in [('layer1_input_hc_buffer','hc_dim','L1',20),('layer2_full_q_norm_buffer','q_rank','L2',24),('layer2_full_attn_norm_buffer','n_embd','L2',24),('layer2_input_hc_buffer','hc_dim','L2',24),('layer2_attn_split_buffer','mix_hc','L2',24)]:
 replace(f'RUST_STAR_NEW_{macro}_BUFFER({var},\n'+(' '*indent)+f'prefill_rows*{width}',f'RUST_STAR_NEW_{macro}_BUFFER({var},\n'+(' '*indent)+f'bootstrap_rows*{width}')
 replace(f'toBuffer:{var}\n                   destinationOffset:(NSUInteger)position_start*{width}',f'toBuffer:{var}\n                   destinationOffset:bootstrap_row*{width}')
replace('NSUInteger offset = (NSUInteger)prefill_chunk_start(opaque_context)*16384u*sizeof(float);', 'NSUInteger offset = context.prefillLayerChunkStart ? 0u\n                : (NSUInteger)prefill_chunk_start(opaque_context)*16384u*sizeof(float);')
start=s.index('int rust_star_metal_prepare_long_prefill_layer2_q(');end=s.index('/* Rebuild the 4K prefix',start)
a=s[start:end];old='offset:(NSUInteger)continuation_start*q_rank*sizeof(float)';assert a.count(old)==1
s=s[:start]+a.replace(old,'offset:context.prefillLayerChunkStart ? 0u\n                            : (NSUInteger)continuation_start*q_rank*sizeof(float)')+s[end:]
start=s.index('int rust_star_metal_run_prefill_layer2_continuation_sparse_tile(');end=s.index('/* Executes the first-chunk layer-4',start)
a=s[start:end]
for prop in ['prefillLayer2FullQNorm','prefillLayer2FullAttnNorm']:
 old=f'context.{prop}, 0u,';assert a.count(old)==1
 a=a.replace(old,f'context.{prop},\n            context.prefillLayerChunkStart ? prefill_chunk_start(opaque_context) : 0u,')
s=s[:start]+a+s[end:]
replace('const NSUInteger transient_row = native_chunk && layer_index >= 3u', 'const NSUInteger transient_row = native_chunk && layer_index >= 2u')
replace(': layer3 ? 0u : (NSUInteger)tile_start*hc_row_bytes;', ': layer3 ? 0u : transient_row*hc_row_bytes;')
p.write_text(s)
