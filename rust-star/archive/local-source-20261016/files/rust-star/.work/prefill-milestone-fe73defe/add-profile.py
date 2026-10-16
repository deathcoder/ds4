from pathlib import Path
p=Path('rust-star/runtime/src/long_prefill.rs');s=p.read_text()
helper='''// Stage totals observe existing completed command buffers; they add no GPU waits.
// Bootstrap ranges are explicit because their shared scheduler spans several layers.
#[derive(Default)]
struct LongPrefillProfile {
    enabled: bool,
    rows: Vec<(u32, u32, u32, &'static str, u32, f64, f64, f64)>,
}
impl LongPrefillProfile {
    fn start(&self) -> Option<Instant> { self.enabled.then(Instant::now) }
    fn record(&mut self, chunk: u32, first: u32, last: u32, stage: &'static str,
              start: Option<Instant>, before: (f64, f64), after: (f64, f64)) {
        if let Some(start) = start {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let wall = after.0 - before.0;
            let gpu = after.1 - before.1;
            if let Some(row) = self.rows.iter_mut().rev().find(|r|
                r.0 == chunk && r.1 == first && r.2 == last && r.3 == stage) {
                row.4 += 1; row.5 += elapsed; row.6 += wall; row.7 += gpu;
            } else { self.rows.push((chunk, first, last, stage, 1, elapsed, wall, gpu)); }
        }
    }
    fn json(self) -> Option<String> {
        if !self.enabled { return None; }
        let rows = self.rows.iter().map(|r| format!(
            "{{\\\"chunk_start\\\":{},\\\"first_layer\\\":{},\\\"last_layer\\\":{},\\\"stage\\\":\\\"{}\\\",\\\"calls\\\":{},\\\"host_elapsed_ms\\\":{:.9},\\\"command_wall_ms\\\":{:.9},\\\"gpu_ms\\\":{:.9}}}",
            r.0,r.1,r.2,r.3,r.4,r.5,r.6,r.7)).collect::<Vec<_>>().join(",");
        Some(format!("{{\\\"method\\\":\\\"existing-command-completion-and-host-stage-timers\\\",\\\"paired_ineligible\\\":true,\\\"rows\\\":[{rows}]}}"))
    }
}

'''
s=helper+s
s=s.replace('    collect_checks: bool,\n) -> Result<(f64, f64)> {','    collect_checks: bool,\n    profile: &mut LongPrefillProfile,\n) -> Result<(f64, f64)> {')
# Continuation stages, using cumulative raw counters (sparse tile starts at zero).
start=s.index('fn run_prefill_chunk_layer');end=s.index('fn rebuild_long_prefill_prefix');part=s[start:end]
part=part.replace('    if layer % 2 == 1 {','    let stage_start = profile.start();\n    if layer % 2 == 1 {',1)
needle='        checked!(\n            rust_star_metal_run_prefill_layer3_continuation_batch_compressor('
part=part.replace(needle,'''        profile.record(start, layer, layer, "ingress", stage_start, (0.0, 0.0), (raw.wall_ms, raw.gpu_ms));
        let stage_start = profile.start(); let before = (raw.wall_ms, raw.gpu_ms);
'''+needle,1)
needle='        checked!(\n            rust_star_metal_run_prefill_layer3_continuation_batch_attention('
part=part.replace(needle,'''        profile.record(start, layer, layer, "compressor", stage_start, before, (raw.wall_ms, raw.gpu_ms));
        let stage_start = profile.start(); let before = (raw.wall_ms, raw.gpu_ms);
'''+needle,1)
needle='        checked!(rust_star_metal_run_prefill_layer2_continuation_tail('
part=part.replace(needle,'''        profile.record(start, layer, layer, "attention", stage_start, before, (raw.wall_ms, raw.gpu_ms));
        let stage_start = profile.start(); let before = (raw.wall_ms, raw.gpu_ms);
'''+needle,1)
part=part.replace('        let padding_dispatch =','        profile.record(start, layer, layer, "tail", stage_start, before, (raw.wall_ms, raw.gpu_ms));\n        let padding_dispatch =',1)
part=part.replace('        let mut scores =','        profile.record(start, layer, layer, "boundary", stage_start, (0.0, 0.0), (raw.wall_ms, raw.gpu_ms));\n        let mut scores =',1)
part=part.replace('            checked!(sparse(','            let stage_start = profile.start();\n            checked!(sparse(',1)
needle='            checked!(rust_star_metal_run_prefill_layer2_continuation_tail('
part=part.replace(needle,'''            profile.record(start, layer, layer, "attention", stage_start, (0.0, 0.0), (tile.wall_ms, tile.gpu_ms));
            let stage_start = profile.start(); let before = (tile.wall_ms, tile.gpu_ms);
'''+needle,1)
part=part.replace('            if tile.dispatches != 36','            profile.record(start, layer, layer, "tail", stage_start, before, (tile.wall_ms, tile.gpu_ms));\n            if tile.dispatches != 36',1)
s=s[:start]+part+s[end:]
# Prefix whole-layer stages preserve existing arithmetic and counters.
a=s.index('fn rebuild_long_prefill_prefix');b=s.index('fn run_long_prefill_with_context');part=s[a:b]
part=part.replace('        if layer >= 3 {','        if layer >= 3 {\n            let stage_start = profile.start();',1)
part=part.replace('            total.0 += preceding.0 + raw.wall_ms;','            profile.record(0, layer, layer, "prefix_layer", stage_start, (0.0, 0.0), (preceding.0 + raw.wall_ms, preceding.1 + raw.gpu_ms));\n            total.0 += preceding.0 + raw.wall_ms;',1)
s=s[:a]+part+s[b:]
s=s.replace('    validate_outputs: bool,\n) -> Result<(Context, LongPrefillChunkProbeReport, NativePrefillTiming)> {','    validate_outputs: bool,\n    collect_profile: bool,\n) -> Result<(Context, LongPrefillChunkProbeReport, NativePrefillTiming)> {',1)
s=s.replace('    let (context, diagnostic, _, mut timing) =','    let mut profile = LongPrefillProfile { enabled: collect_profile, ..Default::default() };\n    let stage_start = profile.start();\n    let (context, diagnostic, _, mut timing) =',1)
s=s.replace('    if diagnostic.is_some() {','    profile.record(0, 0, 42, "initial_bootstrap", stage_start, (0.0, 0.0), (timing.tile_wall_ms + timing.transformer_wall_ms, timing.tile_gpu_ms + timing.transformer_gpu_ms));\n    if diagnostic.is_some() {',1)
s=s.replace('rebuild_long_prefill_prefix(model, &context, &prepared, collect_checks)?','rebuild_long_prefill_prefix(model, &context, &prepared, collect_checks, &mut profile)?',1)
s=s.replace('        let (_, bootstrap_timing) =','        let stage_start = profile.start();\n        let (_, bootstrap_timing) =',1)
s=s.replace('        timing.bootstrap_setup_ms += bootstrap_timing.bootstrap_setup_ms;','        profile.record(start, 0, 2, "bootstrap_ingress", stage_start, (0.0, 0.0), (bootstrap_timing.tile_wall_ms, bootstrap_timing.tile_gpu_ms));\n        timing.bootstrap_setup_ms += bootstrap_timing.bootstrap_setup_ms;',1)
s=s.replace('                    collect_checks,\n                )?;','                    collect_checks,\n                    &mut profile,\n                )?;',1)
# Attach before returning the timing; inspect exact return marker.
needle='    Ok((\n        context,\n        LongPrefillChunkProbeReport'
assert needle in s
s=s.replace(needle,'    timing.long_prefill_profile = profile.json();\n'+needle,1)
s=s.replace('None, true, true)?','None, true, true, false)?').replace('None, false, true)?','None, false, true, false)?')
p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text()
s=s.replace('    pub prefill_transformer_layer_gpu_ms: Option<Vec<f64>>,','    pub long_prefill_profile: Option<String>,\n    pub prefill_transformer_layer_gpu_ms: Option<Vec<f64>>,',1)
s=s.replace('        transformer_layer_gpu_ms: Option<Vec<f64>>,','        long_prefill_profile: Option<String>,\n        transformer_layer_gpu_ms: Option<Vec<f64>>,',1)
s=s.replace('                    retained_context,\n                    false,\n                    false,\n                )?;','                    retained_context,\n                    false,\n                    false,\n                    collect_steady_layer_times,\n                )?;',1)
s=s.replace('            prefill_transformer_layer_gpu_ms: prefill_timing.transformer_layer_gpu_ms,','            long_prefill_profile: prefill_timing.long_prefill_profile,\n            prefill_transformer_layer_gpu_ms: prefill_timing.transformer_layer_gpu_ms,',1)
s=s.replace('            prefill_transformer_layer_gpu_ms: None,','            long_prefill_profile: None,\n            prefill_transformer_layer_gpu_ms: None,',1)
s=s.replace('    let paired_protocol_eligible = !report.prefill_correctness_collection','    let paired_protocol_eligible = report.long_prefill_profile.is_none() && !report.prefill_correctness_collection',1)
s=s.replace('    let paired_protocol_blocker = if report.prefill_correctness_collection {','    let paired_protocol_blocker = if report.long_prefill_profile.is_some() {\n        "\\\"native long-prefill stage timer collection\\\""\n    } else if report.prefill_correctness_collection {',1)
s=s.replace('\\\"stage_profile\\\": {},\\n','\\\"long_prefill_profile\\\": {},\\n  \\\"stage_profile\\\": {},\\n',1)
s=s.replace('        prefill_layer_profile,\n        stage_profile,','        prefill_layer_profile,\n        report.long_prefill_profile.as_deref().unwrap_or("null"),\n        stage_profile,',1)
p.write_text(s)
