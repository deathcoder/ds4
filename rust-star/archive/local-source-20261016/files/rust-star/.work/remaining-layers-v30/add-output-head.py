from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();needle='@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLBuffer>> *prefillLayerRemainingPrefixHc;';s=s.replace(needle,needle+'\n@property(nonatomic, strong) NSNumber *prefillLayerRemainingCompleted;')
for name in ['rust_star_metal_finish_prefill_remaining_odd_rebuild','rust_star_metal_finish_prefill_remaining_even_rebuild']:
 a=s.index('int '+name+'(');b=s.index('\nint ',a+4);block=s[a:b];pos=block.rfind('        return 1;');block=block[:pos]+'        context.prefillLayerRemainingCompleted = @(layer);\n'+block[pos:];s=s[:a]+block+s[b:]
old='''            const NSUInteger prefill_hc_bytes = prefill_rows*hc_dim*sizeof(float);
            input_hc = context.prefillLayer42AfterFfnHc;
            if ((prefill_rows != 2048u && prefill_rows != 4096u) ||
                input_hc.length != prefill_hc_bytes) {'''
new='''            /* The completed second chunk retains its 4096 HC rows, while
             * prefillKvRows records the global 8192-token frontier. */
            const BOOL complete_second_chunk = prefill_rows == 8192u &&
                context.prefillLayerRemainingCompleted.unsignedIntValue == 42u;
            const NSUInteger resident_rows = complete_second_chunk ? 4096u : prefill_rows;
            const NSUInteger prefill_hc_bytes = resident_rows*hc_dim*sizeof(float);
            input_hc = context.prefillLayer42AfterFfnHc;
            if ((!complete_second_chunk && prefill_rows != 2048u && prefill_rows != 4096u) ||
                input_hc.length != prefill_hc_bytes) {'''
assert old in s;s=s.replace(old,new);s=s.replace('input_hc_offset = (prefill_rows-1u)*hc_dim*sizeof(float);','input_hc_offset = (resident_rows-1u)*hc_dim*sizeof(float);');p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();at=s.index('// Immutable oracle targets, installed only');s=s[:at]+'''#[derive(Clone, Debug)]
pub struct RemainingPrefillOutputReport {
    pub checked_logits: u32,
    pub logits_checksum: u64,
    pub selected_token: u32,
}

'''+s[at:];s=s.replace('    pub remaining_layers: Vec<RemainingPrefillLayerReport>,','    pub remaining_layers: Vec<RemainingPrefillLayerReport>,\n    pub remaining_output: Option<RemainingPrefillOutputReport>,',1)
s=s.replace('''        let remaining_layers = run_remaining_prefill_layers(model, &context)?;
        drop(context);''','''        let remaining_layers = run_remaining_prefill_layers(model, &context)?;
        let remaining_output = if remaining_layers.last().map(|r| r.layer) == Some(42) {
            Some(run_remaining_prefill_output(model, &context)?)
        } else { None };
        drop(context);''',1)
s=s.replace('''        Ok(LongPrefillContinuationBootstrapProbeReport {
            remaining_layers,''','''        Ok(LongPrefillContinuationBootstrapProbeReport {
            remaining_layers,
            remaining_output,''',1)
a=s.index('    if report.remaining_layers.len()');b=s.index('    write!(',a)
v='''    let complete = report.remaining_layers.last().map(|r| r.layer) == Some(42);
    if complete != report.remaining_output.is_some() {
        return Err(Error::invalid("8K output requires all remaining layers"));
    }
    if let Some(head) = &report.remaining_output {
        let (_, expected) = prefill_frontier_8192_fixture()?;
        if head.checked_logits as usize != expected.len()
            || head.logits_checksum != checksum_f32(&expected)
            || head.selected_token != 77179 {
            return Err(Error::invalid("8K output report differs from pinned full logits"));
        }
    }
''';s=s[:b]+v+s[b:]
a=s.index('    output.write_all(rendered.as_bytes())?;',s.index('pub fn write_long_prefill_continuation_bootstrap_probe_json'))
v='''    let rendered = if let Some(head) = &report.remaining_output {
        let insertion = format!("  \\\"remaining_output\\\": {{\\\"checked_logits\\\": {}, \\\"logits_checksum\\\": {}, \\\"selected_token\\\": {}, \\\"c0_bitwise_match\\\": true}},\\n", head.checked_logits, head.logits_checksum, head.selected_token);
        rendered.replacen(checksums_boundary, &(insertion + checksums_boundary), 1)
            .replace("\\\"output_logits_c0_claim\\\": false", "\\\"output_logits_c0_claim\\\": true")
    } else { rendered };
''';s=s[:a]+v+s[a:]
a=s.index('let report = LongPrefillContinuationBootstrapProbeReport {',s.index('mod tests'));a+=len('let report = LongPrefillContinuationBootstrapProbeReport {');s=s[:a]+'''
            remaining_output: if REMAINING_PREFILL_CHECKSUMS.last().map(|r| r.0) == Some(42) {
                Some(RemainingPrefillOutputReport { checked_logits: 129280, logits_checksum: checksum_f32(&prefill_frontier_8192_fixture().unwrap().1), selected_token: 77179 })
            } else { None },'''+s[a:];p.write_text(s)
p=Path('rust-star/runtime/src/remaining_prefill.rs');s=p.read_text();s+='''
fn run_remaining_prefill_output(model: &MappedModel, context: &Context) -> Result<RemainingPrefillOutputReport> {
    let (_, expected) = prefill_frontier_8192_fixture()?;
    let mut prepared = PreparedOutputHead::new(model)?;
    let (selected_token, _) = run_sampling_output_head(model, context, &mut prepared)?;
    if prepared.logits.len() != expected.len() {
        return Err(Error::invalid("8K output vocabulary length differs from oracle"));
    }
    for (index, (actual, expected)) in prepared.logits.iter().zip(&expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(Error::invalid(format!("8K full-logit C0 mismatch at {index}: actual={:#010x} expected={:#010x}", actual.to_bits(), expected.to_bits())));
        }
    }
    Ok(RemainingPrefillOutputReport { checked_logits: expected.len() as u32, logits_checksum: checksum_f32(&prepared.logits), selected_token })
}
''';p.write_text(s)
