from pathlib import Path
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();s='''fn prefill_output_ptr<T>(output: &mut [T], collect: bool) -> *mut T {
    if collect { output.as_mut_ptr() } else { std::ptr::null_mut() }
}

'''+s
start=s.index('fn run_prefill_chunk_layer(');end=s.index('fn rebuild_long_prefill_prefix(',start);b=s[start:end]
b=b.replace('''    prepared: &PreparedPrefillChunkLayer,
) -> Result<()>''','''    prepared: &PreparedPrefillChunkLayer,
    collect_checks: bool,
) -> Result<(f64, f64)>''')
for name in ['inputs','tail']:
 import re
 b=re.sub(rf'{name}\[(\d+)\]\.as_mut_ptr\(\)',rf'prefill_output_ptr(&mut {name}[\1], collect_checks)',b)
for name in ['scores','topk','heads','selected','boundary']:
 b=b.replace(name+'.as_mut_ptr()',f'prefill_output_ptr(&mut {name}, collect_checks)')
b=b.replace('&mut compressed,','prefill_output_ptr(std::slice::from_mut(&mut compressed), collect_checks),')
b=b.replace('''                prepared.weights.q_b_bytes,
                error.as_mut_ptr(),''','''                prepared.weights.q_b_bytes,
                &mut raw,
                error.as_mut_ptr(),''')
b=b.replace('''        }
    }
    Ok(())
}''','''            raw.wall_ms += tile.wall_ms;
            raw.gpu_ms += tile.gpu_ms;
        }
    }
    Ok((raw.wall_ms, raw.gpu_ms))
}''',1)
s=s[:start]+b+s[end:]
start=s.index('fn rebuild_long_prefill_prefix(');end=s.index('fn run_long_prefill_in_context(',start);b=s[start:end]
b=b.replace('''    prepared: &[PreparedPrefillChunkLayer],
) -> Result<()>''','''    prepared: &[PreparedPrefillChunkLayer],
    collect_checks: bool,
) -> Result<(f64, f64)>''')
b=b.replace('''    let mut actual = [0; 43];''','''    let mut total = (0.0, 0.0);
    let mut actual = [0; 43];''',1)
for name in ['actual','boundary','sparse']:
 b=b.replace(name+'.as_mut_ptr()',f'prefill_output_ptr(&mut {name}, collect_checks)')
b=b.replace('''            let mut raw = RawSparseIndexedResult::default();''','''            let mut raw = RawSparseIndexedResult::default();
            let mut preceding = (0.0, 0.0);''',1)
b=b.replace('''                let mut sparse = [0; 5];''','''                preceding = (raw.wall_ms, raw.gpu_ms);
                let mut sparse = [0; 5];''',1)
b=b.replace('''                &mut actual[layer as usize],''','''                prefill_output_ptr(std::slice::from_mut(&mut actual[layer as usize]), collect_checks),''')
b=b.replace('''        }
        if actual[layer as usize] != LONG_PREFILL_PREFIX_CHECKSUMS[layer as usize] {''','''            total.0 += preceding.0 + raw.wall_ms;
            total.1 += preceding.1 + raw.gpu_ms;
        }
        if collect_checks && actual[layer as usize] != LONG_PREFILL_PREFIX_CHECKSUMS[layer as usize] {''',1)
b=b.replace('''        eprintln!("rust-star: prefix layer {layer} exact");''','''        if collect_checks { eprintln!("rust-star: prefix layer {layer} exact"); }''')
b=b.replace('''    Ok(())''','''    Ok(total)''',1);s=s[:start]+b+s[end:]
s=s.replace('''fn run_long_prefill_in_context(
    model: &MappedModel,
    frontier: u32,
) -> Result<(Context, LongPrefillChunkProbeReport)>''','''fn run_long_prefill_with_context(
    model: &MappedModel,
    frontier: u32,
    retained_context: Option<Context>,
    collect_checks: bool,
) -> Result<(Context, LongPrefillChunkProbeReport, NativePrefillTiming)>''',1)
s=s.replace('''    let context = Context::new()?;
    let (context, diagnostic, _, _)''','''    let context_started = Instant::now();
    let context = match retained_context { Some(context) => context, None => Context::new()? };
    let context_ms = context_started.elapsed().as_secs_f64()*1000.0;
    let (context, diagnostic, _, mut timing)''',1).replace('''        context,
        0.0,
        false,''','''        context,
        context_ms,
        false,''',1)
s=s.replace('''    eprintln!("rust-star: native 4096-token bootstrap ready for exact prefix rebuild");
    let prepared''','''    if collect_checks { eprintln!("rust-star: native 4096-token bootstrap ready for exact prefix rebuild"); }
    let prefix_started = Instant::now();
    let prepared''',1)
s=s.replace('''    rebuild_long_prefill_prefix(model, &context, &prepared)?;''','''    let prefix = rebuild_long_prefill_prefix(model, &context, &prepared, collect_checks)?;
    timing.transformer_wall_ms += prefix.0;
    timing.transformer_gpu_ms += prefix.1;
    timing.transformer_setup_ms += (prefix_started.elapsed().as_secs_f64()*1000.0-prefix.0).max(0.0);''',1)
s=s.replace('''        run_prefill_layers012_compressor_loop_probe_in_context(''','''        let (_, bootstrap_timing) = run_prefill_layers012_compressor_loop_probe_in_context(''',1)
needle='''        let mut actual = [0; 43];
        for layer in 0..43 {''';s=s.replace(needle,'''        timing.bootstrap_setup_ms += bootstrap_timing.bootstrap_setup_ms;
        timing.tile_wall_ms += bootstrap_timing.tile_wall_ms;
        timing.tile_gpu_ms += bootstrap_timing.tile_gpu_ms;
        let mut actual = [0; 43];
        for layer in 0..43 {''',1)
s=s.replace('''                run_prefill_chunk_layer(''','''                let started = Instant::now();
                let layer_timing = run_prefill_chunk_layer(''',1).replace('''                    &prepared[layer as usize - 2],
                )?;
            }''','''                    &prepared[layer as usize - 2],
                    collect_checks,
                )?;
                timing.transformer_wall_ms += layer_timing.0;
                timing.transformer_gpu_ms += layer_timing.1;
                timing.transformer_setup_ms += (started.elapsed().as_secs_f64()*1000.0-layer_timing.0).max(0.0);
            }''',1)
s=s.replace('''                    &mut actual[layer as usize],''','''                    prefill_output_ptr(std::slice::from_mut(&mut actual[layer as usize]), collect_checks),''',1)
s=s.replace('''            if actual[layer as usize] != expected[layer as usize] {''','''            if collect_checks && actual[layer as usize] != expected[layer as usize] {''',1).replace('''            eprintln!("rust-star: chunk {start} layer {layer} exact");''','''            if collect_checks { eprintln!("rust-star: chunk {start} layer {layer} exact"); }''',1)
s=s.replace('''        if end == 8192 {
            // Keep''','''        if end == 8192 && collect_checks {
            // Keep''',1).replace('''        } else {
            chunks.push((end, actual));''','''        } else if end > 8192 && collect_checks {
            chunks.push((end, actual));''',1)
s=s.replace('''    let mut head = PreparedOutputHead::new(model)?;
    let (selected_token, _) = run_sampling_output_head(model, &context, &mut head)?;''','''    let head_started = Instant::now();
    let mut head = PreparedOutputHead::new(model)?;
    timing.output_head_setup_ms += head_started.elapsed().as_secs_f64()*1000.0;
    let head_started = Instant::now();
    let (selected_token, gpu_ms) = if collect_checks {
        run_sampling_output_head(model, &context, &mut head)?
    } else { run_top1_output_head(model, &context, &head)? };
    timing.output_head_wall_ms += head_started.elapsed().as_secs_f64()*1000.0;
    timing.output_head_gpu_ms += gpu_ms;''',1)
s=s.replace('''    if frontier == 32768 {
        if head.logits''','''    if frontier == 32768 && collect_checks {
        if head.logits''',1).replace('''        if frontier == 32768 {
            run_long_prefill_decode''','''        if frontier == 32768 && collect_checks {
            run_long_prefill_decode''',1)
s=s.replace('''            logits_checksum: checksum_f32(&head.logits),
        },
    ))''','''            logits_checksum: if collect_checks { checksum_f32(&head.logits) } else { 0 },
        },
        timing,
    ))''',1).replace('''Ok(run_long_prefill_in_context(model, frontier)?.1)''','''Ok(run_long_prefill_with_context(model, frontier, None, true)?.1)''',1)
p.write_text(s)
