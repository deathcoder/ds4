from pathlib import Path
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();needle='''    pub chunks: Vec<(u32, [u64; 43])>,
''';s=s.replace(needle,needle+'''    pub checked_decode_steps: u32,
    pub decode_logits_checksum: u64,
    pub decode_tokens_checksum: u64,
''',1);needle='''    } else if report.checked_logits != 0 {
        return Err(Error::invalid("partial frontier cannot claim 32K logits"));
    }
''';s=s.replace(needle,'''        let (tokens, logits) = prefill_decode_32768_fixture()?;
        if report.checked_decode_steps != 128 || report.decode_logits_checksum != checksum_f32(&logits)
            || report.decode_tokens_checksum != checksum_u32(&tokens[1..]) {
            return Err(Error::invalid("32K full decode report differs from oracle"));
        }
    } else if report.checked_logits != 0 || report.checked_decode_steps != 0
        || report.decode_logits_checksum != 0 || report.decode_tokens_checksum != 0 {
        return Err(Error::invalid("partial frontier cannot claim 32K logits or decode"));
    }
''',1)
needle='''    writeln!(output,"],\\n  \\"checked_logits''';# use exact line separately
old=[line for line in s.splitlines() if 'writeln!(output,"],' in line and 'complete_32k_c0_claim' in line][0]
new=old.replace('\\n  \\"complete_32k_c0_claim', '\\n  \\"complete_32k_c0_claim') # explicit insert before checked_logits
new=old.replace('\\n  \\"checked_logits', '\\n  \\"checked_logits')
# Actual Rust source contains escaped quotes; replace by literal raw strings.
new=old.replace(r'\n  \"checked_logits\": {}',r'\n  \"checked_decode_steps\": {},\n  \"decode_logits_checksum\": {},\n  \"decode_tokens_checksum\": {},\n  \"checked_logits\": {}').replace('report.checked_logits,report.logits_checksum', 'report.checked_decode_steps,report.decode_logits_checksum,report.decode_tokens_checksum,report.checked_logits,report.logits_checksum')
assert new!=old;s=s.replace(old,new,1)
needle='''fn prefill_frontier_32768_fixture()''';pos=s.index(needle);s=s[:pos]+'''fn prefill_decode_32768_fixture() -> Result<(Vec<u32>, Vec<f32>)> {
    let tokens = decode_u32_fixture(include_bytes!("../../fixtures/prefill-decode-32768-v3/token-ids.u32le.bin"), "32K decode token transcript")?;
    let logits = decode_f32_fixture(include_bytes!("../../fixtures/prefill-decode-32768-v3/decode-logits.f32le.bin"), "32K full decode logits")?;
    if tokens.len() != 129 || tokens[0] != 85166 || tokens[128] != 14 || logits.len() != 128*129280 {
        return Err(Error::invalid("invalid 32K decode fixture geometry"));
    }
    for (step, row) in logits.chunks_exact(129280).enumerate() {
        if lowest_id_argmax(row)? != tokens[step+1] {
            return Err(Error::invalid("32K decode fixture transcript differs from full logits"));
        }
    }
    Ok((tokens, logits))
}

'''+s[pos:];s=s.replace('''            chunks: super::LONG_PREFILL_CHUNK_CHECKSUMS[..2].to_vec(),
''','''            chunks: super::LONG_PREFILL_CHUNK_CHECKSUMS[..2].to_vec(),
            checked_decode_steps: 0,
            decode_logits_checksum: 0,
            decode_tokens_checksum: 0,
''',1);p.write_text(s)
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();needle='''    Ok((
        context,
        LongPrefillChunkProbeReport {''';s=s.replace(needle,'''    let (checked_decode_steps, decode_logits_checksum, decode_tokens_checksum) = if frontier == 32768 {
        run_long_prefill_decode(model, &context)?
    } else { (0, 0, 0) };
    Ok((
        context,
        LongPrefillChunkProbeReport {''',1);s=s.replace('''            frontier,
            chunks,
''','''            frontier,
            chunks,
            checked_decode_steps,
            decode_logits_checksum,
            decode_tokens_checksum,
''',1)
s+='''
fn run_long_prefill_decode(model: &MappedModel, context: &Context) -> Result<(u32, u64, u64)> {
    let (tokens, expected) = prefill_decode_32768_fixture()?;
    context.adopt_prefill_decoder_state(32768, 32896)?;
    context.prepare_model_residency()?;
    let mut layers = (0..43).map(|layer| PreparedLayerExecution::new_cold_with_capacity(model, layer, 32896)).collect::<Result<Vec<_>>>()?;
    for layer in &mut layers {
        layer.validate_expected = false;
        layer.collect_outputs = false;
    }
    context.prepare_decoder()?;
    let mut head = PreparedOutputHead::new(model)?;
    let mut input = tokens[0];
    let mut actual_logits = Vec::with_capacity(expected.len());
    let mut selected = Vec::with_capacity(128);
    for step in 0..128 {
        let position = 32768+step;
        submit_prepared_layers(model, context, &mut layers, input, position)?;
        run_prepared_layer_iterations(model, context, &mut layers[0], input, position, 0, 1, COMMAND_CHAINED_TIMING, 42)?;
        let (token, _) = run_sampling_output_head(model, context, &mut head)?;
        let row = &expected[step as usize*129280..(step as usize+1)*129280];
        for (index, (actual, expected)) in head.logits.iter().zip(row).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!("32K decode position {position} logit {index} differs: actual={:#010x} expected={:#010x}",actual.to_bits(),expected.to_bits())));
            }
        }
        if token != tokens[step as usize+1] {
            return Err(Error::invalid(format!("32K decode token differs at position {position}")));
        }
        actual_logits.extend_from_slice(&head.logits);
        selected.push(token);
        input = token;
        eprintln!("rust-star: decode position {position} all 129280 logits exact, token {token}");
    }
    Ok((128, checksum_f32(&actual_logits), checksum_u32(&selected)))
}
''';p.write_text(s)
