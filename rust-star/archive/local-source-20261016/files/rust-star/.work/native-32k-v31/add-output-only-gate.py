from pathlib import Path
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text().replace('''pub struct LongPrefillChunkProbeReport {
    pub frontier: u32,''','''pub struct LongPrefillChunkProbeReport {
    pub frontier: u32,
    pub internal_hc_collection: bool,''',1).replace('''        || report.chunks.len() != ((report.frontier - 8192) / 4096) as usize''','''        || (report.internal_hc_collection && report.chunks.len() != ((report.frontier - 8192) / 4096) as usize)
        || (!report.internal_hc_collection && (report.frontier != 32768 || !report.chunks.is_empty()))''',1)
old=[l for l in s.splitlines() if 'write!(output,"{{' in l and 'rust-star-long-prefill-chunk-probe-v1' in l][0];new=old.replace(r'\n  \"chunks\": [',r'\n  \"internal_hc_collection\": {},\n  \"chunks\": [').replace('report.frontier)?;', 'report.frontier,report.internal_hc_collection)?;');assert new!=old;s=s.replace(old,new,1)
s=s.replace('''            frontier: 16384,
            chunks:''','''            frontier: 16384,
            internal_hc_collection: true,
            chunks:''',1)
s=s.replace('''run_long_prefill_with_context(model, context_tokens, retained_context, false)?''','''run_long_prefill_with_context(model, context_tokens, retained_context, false, false)?''',1)
# Non-macOS counterpart and public export.
start=s.index('    pub fn run_long_prefill_chunk_probe(',s.index('#[cfg(not(target_os = "macos"))]'))
end=s.index('\n    pub fn ',start+10)
stub='''
    pub fn run_long_prefill_output_probe(_model: &MappedModel) -> Result<LongPrefillChunkProbeReport> {
        Err(Error::invalid("long-prefill output probe requires macOS Metal"))
    }
''';s=s[:end]+stub+s[end:]
s=s.replace('''    run_long_prefill_bootstrap_probe, run_long_prefill_chunk_probe,''','''    run_long_prefill_bootstrap_probe, run_long_prefill_chunk_probe, run_long_prefill_output_probe,''',1);p.write_text(s)
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text();a=s.index('fn run_long_prefill_with_context(');b=s.index('pub fn run_long_prefill_chunk_probe',a);part=s[a:b].replace('''    collect_checks: bool,
)''','''    collect_checks: bool,
    validate_outputs: bool,
)''',1).replace('''    let (selected_token, gpu_ms) = if collect_checks {''','''    let (selected_token, gpu_ms) = if validate_outputs {''',1).replace('frontier == 32768 && collect_checks','frontier == 32768 && validate_outputs').replace('''            frontier,
            chunks,''','''            frontier,
            internal_hc_collection: collect_checks,
            chunks,''',1).replace('''logits_checksum: if collect_checks''','''logits_checksum: if validate_outputs''');s=s[:a]+part+s[b:];s=s.replace('run_long_prefill_with_context(model, frontier, None, true)?','run_long_prefill_with_context(model, frontier, None, true, true)?',1);s+='''
// This gate exercises the timing path's prefill without internal readbacks,
// then verifies every output word before making any measurement claim.
pub fn run_long_prefill_output_probe(model: &MappedModel) -> Result<LongPrefillChunkProbeReport> {
    Ok(run_long_prefill_with_context(model, 32768, None, false, true)?.1)
}
''';p.write_text(s)
p=Path('rust-star/runtime/src/main.rs');s=p.read_text();a=s.index('    if command == "long-prefill-chunk-probe"');b=s.index('    if command == "long-prefill-bootstrap-probe"',a);part=s[a:b];part=part.replace('''        let mut frontier = 32768;''','''        let mut frontier = 32768;
        let mut output_only = false;''',1);# inferred type might be explicit
if 'let mut output_only' not in part:
 part=part.replace('''        let mut frontier = 32768_u32;''','''        let mut frontier = 32768_u32;
        let mut output_only = false;''',1)
assert 'let mut output_only' in part
part=part.replace('''                Some("--context") =>''','''                Some("--output-only") => output_only = true,
                Some("--context") =>''',1).replace('''expected --context or --json''','''expected --context, --json, or --output-only''')
old='''        let report = rust_star_runtime::metal::run_long_prefill_chunk_probe(&model, frontier)?;''';assert old in part;part=part.replace(old,'''        let report = if output_only {
            if frontier != 32768 { return Err(Error::invalid("--output-only requires context 32768")); }
            rust_star_runtime::metal::run_long_prefill_output_probe(&model)?
        } else { rust_star_runtime::metal::run_long_prefill_chunk_probe(&model, frontier)? };''',1);s=s[:a]+part+s[b:];s=s.replace('long-prefill-chunk-probe MODEL.gguf [--context N] [--json PATH]', 'long-prefill-chunk-probe MODEL.gguf [--context N] [--output-only] [--json PATH]',1);p.write_text(s)
