from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
def modify(name,fn):
 global s
 a=s.index('\nint '+name+'(');b=s.find('\nint ',a+1);b=len(s) if b<0 else b
 body=s[a:b];changed=fn(body);assert changed!=body,name;s=s[:a]+changed+s[b:]
def repl(s,a,b,count=1):
 assert s.count(a)==count,(a,s.count(a),count)
 return s.replace(a,b)
def bootstrap(f):
 f=repl(f,'for (uint32_t index = 0; index < 1216u; index++) {\n            model_buffers[index]', 'const uint32_t model_range_count = last_layer == 2u ? 16u : 1216u;\n        for (uint32_t index = 0; index < model_range_count; index++) {\n            model_buffers[index]')
 # Later-layer variables still exist for the complete path, but have no allocation
 # or initializer in the partial path. No placeholder buffers are published.
 old='''#define RUST_STAR_NEW_L2_ATTN_BUFFER(name, bytes) \\
        id<MTLBuffer> name = prefill_layer_buffer( \\
            context, #name, (bytes), collect_outputs, error, error_bytes)'''
 new='''#define RUST_STAR_NEW_L2_ATTN_BUFFER(name, bytes) \\
        id<MTLBuffer> name = (last_layer == 2u && strncmp(#name, "layer", 5u) == 0) \\
            ? nil : prefill_layer_buffer( \\
                context, #name, (bytes), collect_outputs, error, error_bytes)'''
 f=repl(f,old,new)
 f=repl(f,'!shared_out_buffer || !after_ffn_hc_buffer || !layer3_flat_hc_buffer ||','!shared_out_buffer || !after_ffn_hc_buffer ||\n            (last_layer == 42u && (!layer3_flat_hc_buffer ||')
 f=repl(f,'!layer42_shared_out_buffer || !layer42_after_ffn_hc_buffer\n)', '!layer42_shared_out_buffer || !layer42_after_ffn_hc_buffer))\n)')
 a=f.index('        uint16_t *layer3_mask =');b=f.index('        int32_t *positions =',a)
 f=f[:a]+'        if (last_layer == 42u) {\n'+f[a:b]+'        }\n'+f[b:]
 a=f.index('        int32_t *layer3_compressed_positions =');b=f.index('        rust_star_q8_mm_args q_b_args =',a)
 f=f[:a]+'        if (last_layer == 42u) {\n'+f[a:b]+'        }\n\n'+f[b:]
 f=repl(f,'for (uint32_t index = 0; index < 1216u; index++)\n                pointer_matches', 'for (uint32_t index = 0; index < model_range_count; index++)\n                pointer_matches')
 f=repl(f,'/* Later-layer views are still wrapped by the shared setup above. */\n            result->wrapped_model_ranges = 1218u;', '/* Only layer-2 execution weights and its two indexer weights are wrapped. */\n            result->wrapped_model_ranges = model_range_count + 2u;')
 return f
modify('rust_star_metal_run_prefill_layer2_attention',bootstrap)
def binding(f):
 return repl(f,'const NSUInteger end = prefill_chunk_start(opaque_context)+4096u;', '/* Query, normalized input and HC/split are chunk-local activations.\n         * FullKv/compressed/state buffers above retain absolute history. */\n        const NSUInteger end = 4096u;')
modify('rust_star_metal_bind_long_prefill_layer',binding)
def odd_ingress(f):
 f=repl(f,'const NSUInteger full_q_bytes = (NSUInteger)chunk_end*q_dim*sizeof(float);', 'const uint32_t transient_base = context.prefillLayerChunkStart ? chunk_start : 0u;\n        const NSUInteger full_q_bytes = (NSUInteger)(chunk_end-transient_base)*q_dim*sizeof(float);')
 f=repl(f,'(NSUInteger)chunk_end*mix_hc*sizeof(float);','(NSUInteger)(chunk_end-transient_base)*mix_hc*sizeof(float);')
 f=repl(f,'destinationOffset:(NSUInteger)tile_start*q_dim*sizeof(float)', 'destinationOffset:(NSUInteger)(tile_start-transient_base)*q_dim*sizeof(float)')
 f=repl(f,'destinationOffset:(NSUInteger)tile_start*mix_hc*sizeof(float)', 'destinationOffset:(NSUInteger)(tile_start-transient_base)*mix_hc*sizeof(float)')
 return f
modify('rust_star_metal_run_prefill_layer3_continuation_ingress',odd_ingress)
def odd_attention(f):
 f=repl(f,'const NSUInteger full_q_bytes =\n            (NSUInteger)(first_position+rows)*q_dim*sizeof(float);', 'const uint32_t transient_base = context.prefillLayerChunkStart ? chunk_start : 0u;\n        if (first_position < transient_base)\n            return fail_with_message(error, error_bytes, @"query position precedes its chunk storage");\n        const NSUInteger full_q_bytes =\n            (NSUInteger)(first_position-transient_base+rows)*q_dim*sizeof(float);')
 return repl(f,'offset:(NSUInteger)first_position*q_dim*sizeof(float)', 'offset:(NSUInteger)(first_position-transient_base)*q_dim*sizeof(float)')
modify('rust_star_metal_run_prefill_layer3_continuation_batch_attention',odd_attention)
def even_boundary(f):
 needle='const NSUInteger full_hc_bytes = (NSUInteger)chunk_end*hc_dim*sizeof(float);'
 f=repl(f,needle,'const uint32_t transient_base = context.prefillLayerChunkStart ? position_start : 0u;\n        const NSUInteger transient_rows = chunk_end-transient_base;\n        const NSUInteger full_hc_bytes = transient_rows*hc_dim*sizeof(float);')
 for dim in ['mix_hc','q_dim','q_rank','n_embd']:
  f=repl(f,'(NSUInteger)chunk_end*'+dim+'*sizeof(float);','transient_rows*'+dim+'*sizeof(float);')
 for dim in ['hc_dim','mix_hc','q_dim','q_rank','n_embd']:
  f=repl(f,'destinationOffset:(NSUInteger)position_start*'+dim+'*sizeof(float)', 'destinationOffset:(NSUInteger)(position_start-transient_base)*'+dim+'*sizeof(float)')
 return f
modify('rust_star_metal_run_prefill_layer4_continuation_boundary',even_boundary)
def even_sparse(f):
 f=repl(f,'id<MTLBuffer> heads = nil;', 'id<MTLBuffer> heads = nil;\n        const uint32_t transient_base = context.prefillLayerChunkStart\n            ? prefill_chunk_start(opaque_context) : 0u;')
 for name in ['FullQ','FullQNorm','FullNorm']:f=repl(f,'context.prefillLayer4'+name+', 0u,','context.prefillLayer4'+name+', transient_base,')
 return f
modify('rust_star_metal_run_prefill_layer4_continuation_sparse_tile',even_sparse)
def tail(f):
 f=repl(f,'const NSUInteger input_hc_offset = layer5', 'const NSUInteger transient_row = native_chunk && layer_index >= 3u\n            ? tile_start-chunk_start : tile_start;\n        const NSUInteger input_hc_offset = layer5')
 f=repl(f,'? (NSUInteger)tile_start*hc_row_bytes : full_layer3_batch', '? transient_row*hc_row_bytes : full_layer3_batch')
 a=f.index('        const NSUInteger split_offset =');b=f.index('        const NSUInteger token_offset =',a)
 part=f[a:b].replace('(NSUInteger)tile_start*mix_hc','transient_row*mix_hc');f=f[:a]+part+f[b:]
 return f
modify('rust_star_metal_run_prefill_layer2_continuation_tail',tail)
p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();s=repl(s,'let expected_model_ranges = if prefill_rows == 4096 { 1218 } else { 1216 };', 'let expected_model_ranges = if bootstrap { 18 } else if prefill_rows == 4096 { 1218 } else { 1216 };');p.write_text(s)
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text()
s=repl(s,'.map(|width| vec![0.0; 4096 * width])','.map(|width| if collect_checks { vec![0.0; 4096 * width] } else { Vec::new() })')
s=repl(s,'let mut scores = vec![0.0; 4096 * ((start + 4096) / 4) as usize];','let mut scores = if collect_checks { vec![0.0; 4096 * ((start + 4096) / 4) as usize] } else { Vec::new() };')
s=repl(s,'let mut topk = vec![0; 4096 * 512];','let mut topk = if collect_checks { vec![0; 4096 * 512] } else { Vec::new() };')
s=repl(s,'let mut heads = vec![0.0; 4096 * 32768];','let mut heads = if collect_checks { vec![0.0; 4096 * 32768] } else { Vec::new() };')
p.write_text(s)
