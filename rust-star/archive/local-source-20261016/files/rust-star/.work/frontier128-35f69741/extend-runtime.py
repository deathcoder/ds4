from pathlib import Path
p=Path('rust-star/runtime/src/lib.rs');s=p.read_text();s+='\npub mod frontier_reference;\n';p.write_text(s)
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();a=s.index('fn run_long_prefill_with_context(');b=s.index('    let (context, context_ms)',a)
old=s[a:b];sig=old[:old.index('    if frontier')];wrapper=sig+'''    run_long_prefill_with_reference(model, frontier, retained_context, collect_checks,
        validate_outputs, collect_profile, None)
}

'''
newsig=sig.replace('fn run_long_prefill_with_context(', 'fn run_long_prefill_with_reference(').replace('    collect_profile: bool,\n', '    collect_profile: bool,\n    reference: Option<&crate::frontier_reference::FrontierReference>,\n')
body=old[old.index('    if frontier'):].replace('frontier > 32768', 'frontier > if reference.is_some() { 131072 } else { 32768 }')
body=body.replace('let (tokens, expected_logits) = if validate_outputs {', '''let (tokens, expected_logits) = if let Some(reference) = reference {
        let prefix = decode_u32_fixture(PREFILL_FRONTIER_32768_TOKEN_IDS_BYTES, "32K prompt tokens")?;
        if reference.frontier() != frontier || reference.tokens[..32768] != prefix {
            return Err(Error::invalid("external reference must extend the pinned 32K prompt"));
        }
        (reference.tokens.clone(), reference.logits[..129280].to_vec())
    } else if validate_outputs {''')
s=s[:a]+wrapper+newsig+body+s[b:]
s=s.replace('if frontier == 32768 && validate_outputs {', 'if (frontier == 32768 || reference.is_some()) && validate_outputs {')
s=s.replace('            run_long_prefill_decode(model, &context, selected_token)?', '''            if let Some(reference) = reference {
                if reference.selected.len() == 1 { (0, 0, 0) } else {
                    run_long_prefill_decode_reference(model, &context, selected_token, frontier,
                        &reference.selected, &reference.logits[129280..])?
                }
            } else { run_long_prefill_decode(model, &context, selected_token)? }''')
a=s.index('    let (tokens, expected) = prefill_decode_32768_fixture()?;',s.index('fn run_long_prefill_decode('));b=s.index('// This gate exercises',a)
part=s[a:b];part=part.replace('    let (tokens, expected) = prefill_decode_32768_fixture()?;\n', '''    let (tokens, expected) = prefill_decode_32768_fixture()?;
    run_long_prefill_decode_reference(model, context, first_token, 32768, &tokens, &expected)
}

fn run_long_prefill_decode_reference(model: &MappedModel, context: &Context,
    first_token: u32, frontier: u32, tokens: &[u32], expected: &[f32]) -> Result<(u32, u64, u64)> {
''').replace('context.adopt_prefill_decoder_state(32768, 32896)?;', 'context.adopt_prefill_decoder_state(frontier, frontier + 128)?;').replace('model, layer, 32896)', 'model, layer, frontier + 128)').replace('let position = 32768 + step;', 'let position = frontier + step;').replace('32K decode','frontier decode').replace('native 32K prefill','native prefill')
s=s[:a]+part+s[b:]
s+='''
/// Compare all output words with explicitly supplied vectors. The evidence
/// runner must authenticate their oracle/model provenance separately.
pub fn run_long_prefill_reference_probe(model: &MappedModel,
    reference: &crate::frontier_reference::FrontierReference) -> Result<LongPrefillChunkProbeReport> {
    Ok(run_long_prefill_with_reference(model, reference.frontier(), None, false, true, false, Some(reference))?.1)
}
''';p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();s=s.replace('if !matches!(prefill_rows, 2_048 | 4_096 | 32_768)', 'if !(matches!(prefill_rows, 2_048 | 4_096) || ((32_768..=131_072).contains(&prefill_rows) && prefill_rows % 4096 == 0))',1)
a=s.index('mod imp {',s.index('#[cfg(not(target_os = "macos"))]'));a+=len('mod imp {');s=s[:a]+'''
    pub fn run_long_prefill_reference_probe(_model: &crate::mapping::MappedModel,
        _reference: &crate::frontier_reference::FrontierReference) -> crate::Result<super::LongPrefillChunkProbeReport> {
        Err(crate::Error::invalid("external frontier comparison requires macOS Metal"))
    }
'''+s[a:]
s=s.replace('    run_long_prefill_bootstrap_probe, run_long_prefill_chunk_probe,', '    run_long_prefill_bootstrap_probe, run_long_prefill_chunk_probe, run_long_prefill_reference_probe,');p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();s=s.replace('prefill_rows > 32768u || prefill_rows%4096u', 'prefill_rows > 131072u || prefill_rows%4096u',1).replace('const BOOL complete_long_prefill = prefill_rows == 32768u &&\n            context.prefillLayerChunkStart.unsignedIntValue == 28672u &&', 'const BOOL complete_long_prefill = prefill_rows >= 32768u && prefill_rows <= 131072u &&\n            prefill_rows%4096u == 0u &&\n            context.prefillLayerChunkStart.unsignedIntValue == prefill_rows-4096u &&',1).replace('completed 2K, 4K, or 32K prefill', 'completed 2K, 4K, or aligned 32K--128K prefill',1).replace('prefillKvRows records the global frontier through 32K.', 'prefillKvRows records the global frontier through 128K.',1).replace('prefill_rows <= 32768u && prefill_rows%4096u', 'prefill_rows <= 131072u && prefill_rows%4096u',1).replace('chunk_start >= 32768u || chunk_start%4096u', 'chunk_start >= 131072u || chunk_start%4096u',1);p.write_text(s)
