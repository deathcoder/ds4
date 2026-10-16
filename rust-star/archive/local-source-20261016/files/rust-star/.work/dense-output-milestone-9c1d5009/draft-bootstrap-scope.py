"""Prepare a reviewable next-step patch; default mode never edits runtime files."""
from pathlib import Path
import difflib,sys
w=Path(__file__).resolve().parent;r=w.parents[2];changes={}
def source(name):return (r/name).read_text()
def save(name,old,new):
 assert old!=new,name
 changes[name]=(old,new)
name='rust-star/runtime/src/metal_shim.h';old=source(name);s=old
end=s.index('} rust_star_metal_prefill_layer2_attention_result;')
s=s[:end]+'    uint32_t last_completed_layer;\n    uint32_t indexer_dispatches;\n'+s[end:]
a=s.index('int rust_star_metal_run_prefill_layer2_attention(');b=s.index(');',a)
part=s[a:b];assert part.count('    uint32_t collect_outputs,')==1
part=part.replace('    uint32_t collect_outputs,','    uint32_t collect_outputs,\n    uint32_t last_layer,');s=s[:a]+part+s[b:];save(name,old,s)
name='rust-star/runtime/src/metal_shim.m';old=source(name);s=old
a=s.index('int rust_star_metal_run_prefill_layer2_attention(');b=s.index('\nint ',a+4);part=s[a:b]
part=part.replace('    uint32_t collect_outputs,','    uint32_t collect_outputs,\n    uint32_t last_layer,',1)
marker='    if (!opaque_context || !model_mapping || !weights || !result ||'
assert part.count(marker)==1
part=part.replace(marker,'''    if ((last_layer != 2u && last_layer != 42u) ||
        (last_layer == 2u && (prefill_rows != 4096u || collect_outputs)))
        return fail_with_message(error, error_bytes, @"invalid native prefill completion scope");
'''+marker,1)
marker='        const BOOL prefill_layer_profiling = context.prefillLayerProfiling;'
part=part.replace(marker,marker+'''
        if (last_layer == 2u && prefill_layer_profiling) {
            [encoder endEncoding];
            return fail_with_message(error, error_bytes, @"bootstrap scope uses outer stage profiling");
        }''',1)
marker='        uint32_t compressed_elements = compressed_rows*head_dim;\n        if (rows == 4096u) {'
assert part.count(marker)==1
part=part.replace(marker,'        uint32_t layer2_indexer_dispatches = 0u;\n'+marker,1)
marker='''            if (!encode_indexer_topk(context, encoder,
                    layer2_indexer_score_buffer, layer2_indexer_topk_buffer,
                    layer2_topk_args, error, error_bytes)) {'''
assert part.count(marker)==1
part=part.replace(marker,'''            layer2_indexer_dispatches = encode_indexer_topk(context, encoder,
                layer2_indexer_score_buffer, layer2_indexer_topk_buffer,
                layer2_topk_args, error, error_bytes);
            if (!layer2_indexer_dispatches) {''',1)
marker='        RUST_STAR_PREFILL_LAYER_BOUNDARY(1u);';assert part.count(marker)==1
part=part.replace(marker,'''        /* Long prefill rebuilds layers 3--42 with its accepted prefix schedule.
         * Retain only the completed layer-2 HC here; no partial output head is run. */
        if (last_layer == 2u) {
            [encoder endEncoding];
            const double bootstrap_wall_start = monotonic_ms();
            [command commit];
            if (!command_succeeded(command, error, error_bytes)) return 0;
            context.prefillLayer3InputHc = after_ffn_hc_buffer;
            uint32_t pointer_matches = 0u;
            for (uint32_t index = 0; index < 1216u; index++)
                pointer_matches += matches[index] ? 1u : 0u;
            pointer_matches += layer2_indexer_matches[0] ? 1u : 0u;
            pointer_matches += layer2_indexer_matches[1] ? 1u : 0u;
            result->rows = rows;
            result->raw_kv_rows = raw_rows;
            result->compressed_kv_rows = compressed_rows;
            result->dispatches = 36u + layer2_indexer_dispatches;
            /* Later-layer views are still wrapped by the shared setup above. */
            result->wrapped_model_ranges = 1218u;
            result->pointer_matches = pointer_matches;
            result->layer3_compressed_kv_rows = 0u;
            result->wall_ms = monotonic_ms()-bootstrap_wall_start;
            result->gpu_ms = gpu_elapsed_ms(command);
            result->last_completed_layer = 2u;
            result->indexer_dispatches = layer2_indexer_dispatches;
            return 1;
        }

'''+marker,1)
marker='        result->dispatches = rows == 4096u ? 2396u : 2391u;'
assert part.count(marker)==1
part=part.replace(marker,marker+'\n        result->last_completed_layer = 42u;\n        result->indexer_dispatches = layer2_indexer_dispatches;',1)
s=s[:a]+part+s[b:];save(name,old,s)
name='rust-star/runtime/src/metal.rs';old=source(name);s=old
a=s.index('    struct RawPrefillLayer2AttentionResult {');b=s.index('\n    }',a)
s=s[:b]+'\n        last_completed_layer: u32,\n        indexer_dispatches: u32,'+s[b:]
a=s.index('        fn rust_star_metal_run_prefill_layer2_attention(');b=s.index(') -> i32;',a)
part=s[a:b];assert part.count('            collect_outputs: u32,')==1
part=part.replace('            collect_outputs: u32,','            collect_outputs: u32,\n            last_layer: u32,');s=s[:a]+part+s[b:]
a=s.index('    fn run_prefill_layers012_attention_loop_with_context(');b=s.index('    fn run_sparse_indexed_attention_fixture_probe(',a);part=s[a:b]
part=part.replace('fn run_prefill_layers012_attention_loop_with_context(', 'fn run_prefill_layers012_attention_loop_with_scope(',1)
part=part.replace('        tokens: Option<&[u32]>,','        tokens: Option<&[u32]>,\n        scope: NativePrefillScope,',1)
part=part.replace('        u32,\n        NativePrefillTiming,','        Option<u32>,\n        NativePrefillTiming,',1)
part=part.replace('''        if profile_layers {''','''        let bootstrap = scope == NativePrefillScope::Bootstrap012;
        if bootstrap && (prefill_rows != 4096 || collect_outputs || profile_layers) {
            return Err(Error::invalid("invalid initial bootstrap scope"));
        }
        if profile_layers {''',1)
part=part.replace('                u32::from(collect_outputs),\n                &mut raw,','                u32::from(collect_outputs),\n                scope as u32,\n                &mut raw,',1)
part=part.replace('        let expected_dispatches = if prefill_rows == 4096 { 2396 } else { 2391 };','''        let expected_dispatches = if bootstrap {
            36 + raw.indexer_dispatches
        } else if prefill_rows == 4096 { 2396 } else { 2391 };
        let valid_indexer_count = if prefill_rows == 4096 {
            // One block sort plus at most ten merges for 1024 input columns.
            (1..=11).contains(&raw.indexer_dispatches)
        } else { raw.indexer_dispatches == 0 };''',1)
part=part.replace('            || raw.layer3_compressed_kv_rows != prefill_rows / 128','''            || raw.last_completed_layer != scope as u32
            || !valid_indexer_count
            || raw.layer3_compressed_kv_rows != if bootstrap { 0 } else { prefill_rows / 128 }''',1)
marker='''        if !collect_outputs {
            let output_head_setup_started = Instant::now();''';assert part.count(marker)==1
part=part.replace(marker,'''        if bootstrap {
            timing.transformer_wall_ms = raw.wall_ms;
            timing.transformer_gpu_ms = raw.gpu_ms;
            return Ok((context, None, None, timing));
        }
'''+marker,1)
part=part.replace('return Ok((context, None, selected_token, timing));','return Ok((context, None, Some(selected_token), timing));',1)
assert part.count('            selected_token,\n            timing,')==1
part=part.replace('            selected_token,\n            timing,','            Some(selected_token),\n            timing,',1)
wrapper='''    #[derive(Clone, Copy, PartialEq, Eq)]
    #[repr(u32)]
    enum NativePrefillScope {
        Bootstrap012 = 2,
        Complete = 42,
    }

    fn run_prefill_layers012_attention_loop_with_context(
        model: &MappedModel, context: Context, context_setup_ms: f64,
        collect_outputs: bool, profile_layers: bool, prefill_rows: u32,
        tokens: Option<&[u32]>,
    ) -> Result<(Context, Option<PrefillLayers012AttentionLoopProbeReport>, u32, NativePrefillTiming)> {
        let (context, report, selected, timing) = run_prefill_layers012_attention_loop_with_scope(
            model, context, context_setup_ms, collect_outputs, profile_layers,
            prefill_rows, tokens, NativePrefillScope::Complete)?;
        Ok((context, report, selected.ok_or_else(|| Error::invalid("complete prefill omitted selection"))?, timing))
    }

    fn run_long_prefill_initial_bootstrap(
        model: &MappedModel, context: Context, context_setup_ms: f64, tokens: &[u32],
    ) -> Result<(Context, NativePrefillTiming)> {
        let (context, report, selected, timing) = run_prefill_layers012_attention_loop_with_scope(
            model, context, context_setup_ms, false, false, 4096,
            Some(tokens), NativePrefillScope::Bootstrap012)?;
        if report.is_some() || selected.is_some() {
            return Err(Error::invalid("partial bootstrap unexpectedly produced a model output"));
        }
        Ok((context, timing))
    }

'''
s=s[:a]+wrapper+part+s[b:];save(name,old,s)
name='rust-star/runtime/src/long_prefill.rs';old=source(name);s=old
marker='''    let (context, diagnostic, _, mut timing) = run_prefill_layers012_attention_loop_with_context(
        model,
        context,
        context_ms,
        false,
        false,
        4096,
        Some(&tokens),
    )?;''';assert s.count(marker)==1
s=s.replace(marker,'''    let (context, mut timing) =
        run_long_prefill_initial_bootstrap(model, context, context_ms, &tokens)?;''',1)
s=s.replace('        42,\n        "initial_bootstrap",','        2,\n        "initial_bootstrap",',1)
marker='''    if diagnostic.is_some() {
        return Err(Error::invalid(
            "long-prefill prefix unexpectedly collected diagnostics",
        ));
    }
''';assert s.count(marker)==1;s=s.replace(marker,'',1);save(name,old,s)
patch=''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+name,tofile='b/'+name)) for name,(a,b) in changes.items())
(w/'proposed-bootstrap-scope.patch').write_text(patch)
if '--apply' in sys.argv:
 assert (w/'measurement-complete.json').exists(), 'Finish the active measurements before changing runtime'
 for name,(a,b) in changes.items():assert (r/name).read_text()==a
 for name,(a,b) in changes.items():(r/name).write_text(b)
 print('APPLIED: next candidate still requires tests/build/full live gates/measurement')
else:
 print('DRAFT ONLY; runtime unchanged:',len(patch.splitlines()),'patch lines')
