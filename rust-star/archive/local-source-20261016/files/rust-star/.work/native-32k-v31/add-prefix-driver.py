from pathlib import Path
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();pos=s.index('fn run_long_prefill_in_context(');s=s[:pos]+'''fn rebuild_long_prefill_prefix(model: &MappedModel, context: &Context,
    prepared: &[PreparedPrefillChunkLayer]) -> Result<()> {
    let mut error = [0 as c_char; ERROR_BYTES];
    macro_rules! checked {
        ($call:expr) => {
            if unsafe { $call } == 0 { return Err(Error::invalid(error_text(&error))); }
        };
    }
    let mut actual = [0; 43];
    checked!(rust_star_metal_begin_long_prefill_prefix(context.0, actual.as_mut_ptr(), error.as_mut_ptr(), error.len()));
    for layer in 0..43 {
        if layer >= 3 {
            let p = &prepared[layer as usize - 2];
            checked!(rust_star_metal_bind_long_prefill_prefix(context.0, layer, error.as_mut_ptr(), error.len()));
            let mut raw = RawSparseIndexedResult::default();
            if layer % 2 == 1 {
                checked!(rust_star_metal_run_prefill_layer5_prefix_ingress(context.0, model.mapping_pointer(), model.bytes(), &p.weights, &mut raw, error.as_mut_ptr(), error.len()));
                checked!(rust_star_metal_run_prefill_layer3_continuation_batch_compressor(context.0, model.mapping_pointer(), model.bytes(), &p.compressor, 0, &mut raw, error.as_mut_ptr(), error.len()));
                checked!(rust_star_metal_run_prefill_layer3_continuation_batch_attention(context.0, model.mapping_pointer(), model.bytes(), p.weights.attn_sinks_offset, p.weights.attn_sinks_bytes, 0, &mut raw, error.as_mut_ptr(), error.len()));
            } else {
                let mut boundary = [0; 10];
                checked!(rust_star_metal_run_prefill_layer4_continuation_boundary(context.0, model.mapping_pointer(), model.bytes(), &p.weights, &p.compressor, 0, boundary.as_mut_ptr(), boundary.len(), &mut raw, error.as_mut_ptr(), error.len()));
                let mut sparse = [0; 5];
                checked!(rust_star_metal_run_prefill_layer4_prefix_sparse_batch(context.0, model.mapping_pointer(), model.bytes(), p.indexer_q.0, p.indexer_q.1, p.indexer_weight.0, p.indexer_weight.1, p.weights.attn_sinks_offset, p.weights.attn_sinks_bytes, sparse.as_mut_ptr(), sparse.len(), &mut raw, error.as_mut_ptr(), error.len()));
            }
            checked!(rust_star_metal_run_prefill_layer2_continuation_tail(
                context.0, model.mapping_pointer(), model.bytes(), &p.weights,
                if layer % 2 == 1 { 5 } else { 4 }, 0, 4096,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), &mut raw, error.as_mut_ptr(), error.len()));
            checked!(rust_star_metal_finish_long_prefill_prefix(context.0, layer, &mut actual[layer as usize], error.as_mut_ptr(), error.len()));
        }
        if actual[layer as usize] != LONG_PREFILL_PREFIX_CHECKSUMS[layer as usize] {
            return Err(Error::invalid(format!("4K prefix layer {layer} HC differs: actual={:#018x} expected={:#018x}",actual[layer as usize],LONG_PREFILL_PREFIX_CHECKSUMS[layer as usize])));
        }
        eprintln!("rust-star: prefix layer {layer} exact");
    }
    Ok(())
}

'''+s[pos:];needle='''        .collect::<Result<Vec<_>>>()?;
    let mut chunks = Vec::new();''';s=s.replace(needle,'''        .collect::<Result<Vec<_>>>()?;
    rebuild_long_prefill_prefix(model, &context, &prepared)?;
    let mut chunks = Vec::new();''',1);p.write_text(s)
