from pathlib import Path
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text().replace('''    if report.context != 2048
        || report.gen_tokens != 128''','''    if !matches!(report.context, 2048 | 32768)
        || report.gen_tokens != 128''',1).replace('''        || report.prefill_selected_token != 15342
''','''        || report.prefill_selected_token != if report.context == 32768 { 85166 } else { 15342 }
''',1)
needle='''    let prefill_tps = f64::from(report.context)''';pos=s.index(needle);s=s[:pos]+'''    if report.context == 32768 {
        let tokens = decode_u32_fixture(include_bytes!("../../fixtures/prefill-decode-32768-v3/token-ids.u32le.bin"), "32K engine transcript")?;
        if report.final_selected_token != 14 || report.selected_tokens_checksum != checksum_u32(&tokens[1..]) {
            return Err(Error::invalid("32K engine transcript differs from the pinned oracle"));
        }
    }
'''+s[pos:]
s=s.replace('''        if context_tokens != 2048 || gen_tokens != 128 {''','''        if !matches!(context_tokens, 2048 | 32768) || gen_tokens != 128 {''',1).replace('the initial Rust Star engine measurement supports exactly context=2048 and gen_tokens=128','Rust Star engine measurement supports context=2048 or 32768 and gen_tokens=128',1)
start=s.index('    fn run_engine_measurement_cycle(');end=s.index('    fn run_position_advancing_probe(',start);b=s[start:end];b=b.replace('''        let (oracle_inputs, _, _) = prefill_decode_frontier_4099_fixture()?;''','''        let oracle_inputs = if context_tokens == 32768 {
            decode_u32_fixture(include_bytes!("../../fixtures/prefill-decode-32768-v3/token-ids.u32le.bin"), "32K engine transcript")?
        } else { prefill_decode_frontier_4099_fixture()?.0 };''',1)
b=b.replace('''            match retained_context {''','''            if context_tokens == 32768 {
                let (context, report, timing) = run_long_prefill_with_context(model, context_tokens, retained_context, false)?;
                (context, None, report.selected_token, timing)
            } else { match retained_context {''',1).replace('''            };
        if prefill_report.is_some()''','''            }};
        if prefill_report.is_some()''',1)
b=b.replace('''            prefill_layer_timing_collection: collect_steady_layer_times,''','''            prefill_layer_timing_collection: collect_steady_layer_times && context_tokens == 2048,''',1);s=s[:start]+b+s[end:];p.write_text(s)
