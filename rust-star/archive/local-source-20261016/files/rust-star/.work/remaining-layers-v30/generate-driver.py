from pathlib import Path
import re
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text()
fields='''
#[derive(Clone, Debug)]
pub struct RemainingPrefillLayerReport {
    pub layer: u32,
    pub prefix_boundary: Vec<u64>,
    pub continuation_boundary: Vec<u64>,
    pub prefix_tail: [u64; 3],
    pub continuation_tail: [u64; 3],
    pub dispatches: u64,
    pub model_mappings: u64,
    pub pointer_matches: u64,
}

// Immutable oracle targets, installed only after independent captures agree.
include!("remaining_prefill_checksums.rs");
'''
at=s.index('#[derive',s.index('pub struct LongPrefillContinuationBootstrapProbeReport')-100) if False else s.index('pub struct LongPrefillContinuationBootstrapProbeReport')
# Insert types before the report's derive attributes.
at=s.rfind('#[derive',0,at);s=s[:at]+fields+'\n'+s[at:]
a=s.index('pub struct LongPrefillContinuationBootstrapProbeReport');b=s.index('\n}',a);s=s[:b]+'\n    pub remaining_layers: Vec<RemainingPrefillLayerReport>,'+s[b:]
# Runtime call and report handoff.
needle='''            run_prefill_layer2_sparse_transition_probe(model, &context)?;
        drop(context);'''
assert s.count(needle)==1
s=s.replace(needle,'''            run_prefill_layer2_sparse_transition_probe(model, &context)?;
        let remaining_layers = run_remaining_prefill_layers(model, &context)?;
        drop(context);''')
a=s.index('        Ok(LongPrefillContinuationBootstrapProbeReport {');s=s[:a]+s[a:].replace('        Ok(LongPrefillContinuationBootstrapProbeReport {','        Ok(LongPrefillContinuationBootstrapProbeReport {\n            remaining_layers,',1)
# Existing serializer unit report gains matching targets.
a=s.index('let report = LongPrefillContinuationBootstrapProbeReport {',s.index('mod tests'))
s=s[:a]+s[a:].replace('let report = LongPrefillContinuationBootstrapProbeReport {','let report = LongPrefillContinuationBootstrapProbeReport {\n            remaining_layers: Vec::new(),',1)
# FFI entries, private include inside imp module.
prototypes=Path('rust-star/.work/remaining-layers-v30/binding-names.txt').read_text()
ffi=''.join('        fn '+name+'(context: *mut c_void, layer: u32, error: *mut c_char, error_bytes: usize) -> i32;\n' for name in re.findall(r'int (\w+)\(',prototypes))
a=s.index('        fn rust_star_metal_begin_prefill_layer13_prefix_rebuild');s=s[:a]+ffi+s[a:]
a=s.index('    struct Context(*mut c_void);');s=s[:a]+'    include!("remaining_prefill.rs");\n\n'+s[a:]
p.write_text(s)
base=s
out='''// Shared execution for the remaining fixed-model transformer layers.
// The arithmetic and dispatch order follow the validated layer-13/14 paths.
fn run_remaining_prefill_layers(model: &MappedModel, context: &Context) -> Result<Vec<RemainingPrefillLayerReport>> {
    let mut reports = Vec::new();
    for &(layer, prefix, continuation, prefix_tail, continuation_tail) in REMAINING_PREFILL_CHECKSUMS {
        let result = if layer%2 == 1 {
            run_remaining_odd(model, context, layer, prefix, continuation, prefix_tail, continuation_tail)
        } else {
            run_remaining_even(model, context, layer, prefix, continuation, prefix_tail, continuation_tail)
        }.map_err(|error| Error::invalid(format!("remaining layer {layer}: {error}")))?;
        eprintln!("rust-star: complete retained layer {layer} exact");
        reports.push(result);
    }
    Ok(reports)
}
'''
for n,end_marker in [(13,'        let layer14_begin_succeeded'),(14,'        layer23_loop.wall_ms += layer6_prefix_raw.wall_ms')]:
 odd=n==13;label='odd' if odd else 'even'
 # Weight lookups and struct assembly from the validated current layer.
 a=base.index(f'        let layer{n}_hc_fn',base.index('fn run_prefill_layer2_sparse_transition_probe'))
 b=base.index('        let layer12_hc_fn',a) if odd else base.index('        let tail_weights',a)
 lookups=base[a:b]
 # layer14 lookup ends at tail_weights and contains only layer14.
 lookups=re.sub(r'"blk\.'+str(n)+r'\.([^"\n]+)"',lambda m:'&format!("blk.{layer}.'+m[1]+'")',lookups)
 a=base.index(f'        let layer{n}_weights = RawPrefillLayerWeights',a)
 b=base.index('        let layer12_weights',a) if odd else base.index('        let expected_q_current',a)
 weights=base[a:b]
 # Extract the entire validated executor block unchanged apart from bind calls / expected hashes.
 marker=f'        let layer{n}_prefix_begin_succeeded' if odd else f'        let layer{n}_begin_succeeded'
 a=base.index(marker);b=base.index(end_marker,a);body=base[a:b]
 names=re.findall(r'int (\w+)\(',prototypes)
 for name in names:
  if f'remaining_{label}' in name:
   old=name.replace(f'remaining_{label}',f'layer{n}')
   body=body.replace(old+'(\n                context.0,',name+'(\n                context.0,\n                layer,')
 body=body.replace(f'layer-{n}',f'{label}-layer')
 consts=''
 for kind,size,expr in [('PREFIX_BOUNDARY',5 if odd else 10,'prefix'),('CONTINUATION_BOUNDARY',10,'continuation'),('PREFIX_TAIL',3,'prefix_tail'),('CONTINUATION_TAIL',3,'continuation_tail')]:
  if odd and kind=='CONTINUATION_BOUNDARY':continue
  name=f'PREFILL_LAYER{n}_{kind}_CHECKSUMS';value=expr if 'TAIL' in kind else expr+'.try_into().map_err(|_| Error::invalid("invalid oracle boundary width"))?'
  consts+=f'        let {name}: [u64; {size}] = {value};\n'
 if odd:
  consts+='''        let PREFILL_LAYER13_CONTINUATION_INGRESS_CHECKSUMS: [u64; 4] = continuation[..4].try_into().unwrap();
        let PREFILL_LAYER13_CONTINUATION_COMPRESSED_CHECKSUM: u64 = continuation[4];
'''
 # Local diagnostic tile scratch only; no full model copy.
 scratch=''
 pat=r'        let mut layer'+('5' if odd else '4')+r'_\w+ = vec!\[[^\n]+\n'
 for line in re.findall(pat,base):
  var=re.search(r'let mut (\w+)',line)[1]
  if var in body and line not in scratch:scratch+=line
 prefix=f'layer23_loop.layer{n}_prefix_boundary_checksums.to_vec()'
 continuation=f'layer23_loop.layer{n}_continuation_boundary_checksums.to_vec()' if not odd else '[layer23_loop.layer13_continuation_ingress_checksums.to_vec(), vec![layer23_loop.layer13_continuation_compressed_checksum]].concat()'
 totals=('layer13_prefix_raw.dispatches + layer23_loop.layer13_continuation_dispatches','layer13_prefix_raw.wrapped_model_ranges + layer23_loop.layer13_continuation_wrapped_model_ranges','layer13_prefix_raw.pointer_matches + layer23_loop.layer13_continuation_pointer_matches') if odd else tuple(f'layer14_prefix_raw.{f} + layer14_prefix_complete_raw.{f} + layer14_continuation_raw.{f} + layer23_loop.layer14_continuation_{g}' for f,g in [('dispatches','dispatches'),('wrapped_model_ranges','wrapped_model_ranges'),('pointer_matches','pointer_matches')])
 out+=f'''\n#[allow(non_snake_case)]
fn run_remaining_{label}(model: &MappedModel, context: &Context, layer: u32,
 prefix: &[u64], continuation: &[u64], prefix_tail: [u64;3], continuation_tail: [u64;3]) -> Result<RemainingPrefillLayerReport> {{
 let mut error = [0 as c_char; 1024];
 let mut layer23_loop = RawLayer23ContinuationLoopResult::default();
'''+consts+lookups+weights+scratch+body+f'''        Ok(RemainingPrefillLayerReport {{ layer,
            prefix_boundary: {prefix}, continuation_boundary: {continuation},
            prefix_tail: layer23_loop.layer{n}_prefix_tail_checksums,
            continuation_tail: layer23_loop.layer{n}_continuation_tail_checksums,
            dispatches: ({totals[0]}) as u64,
            model_mappings: ({totals[1]}) as u64,
            pointer_matches: ({totals[2]}) as u64,
        }})
}}
'''
Path('rust-star/runtime/src/remaining_prefill.rs').write_text(out)
Path('rust-star/runtime/src/remaining_prefill_checksums.rs').write_text('type RemainingPrefillChecksums = (u32, &\'static [u64], &\'static [u64], [u64; 3], [u64; 3]);\nconst REMAINING_PREFILL_CHECKSUMS: &[RemainingPrefillChecksums] = &[];\n')
