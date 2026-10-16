from pathlib import Path
import re
src=Path('rust-star/runtime/src/remaining_prefill.rs').read_text();a=src.index('    let layer14_hc_fn =');b=src.index('    let mut layer4_indexer_scores',a);block=src[a:b]
pat=r'let layer14_(\w+)\s*=\s*exact_tensor\(\s*model,\s*&format!\("blk\.\{layer\}\.([^"\n]+)"\),\s*(\d+),\s*&\[([^]]*)\],?\s*\)\?;'
fields=re.findall(pat,block);assert len(fields)==34,len(fields)
code='''// Fixed-model 4096-token continuation chunks after the validated 8K bootstrap.
struct PreparedPrefillChunkLayer {
    weights: RawPrefillLayerWeights,
    compressor: RawPrefillCompressorWeights,
    indexer_q: (u64, u64),
    indexer_weight: (u64, u64),
}
impl PreparedPrefillChunkLayer {
    fn new(model: &MappedModel, layer: u32) -> Result<Self> {
        let tensor = |suffix: &str, ty, dims: &[u64]| -> Result<(u64, u64)> {
            let tensor = exact_tensor(model, &format!("blk.{layer}.{suffix}"), ty, dims)?;
            Ok((tensor.absolute_offset, tensor.bytes))
        };
'''
for name,suffix,ty,dims in fields:
 expr=f'tensor("{suffix}", {ty}, &[{dims}])?'
 if name.startswith('indexer_'):expr=f'if layer % 2 == 0 {{ {expr} }} else {{ (0, 0) }}'
 if name=='attn_ape':expr=f'if layer % 2 == 0 {{ {expr} }} else {{ tensor("{suffix}", 1, &[512,128])? }}'
 if name in ['attn_kv','attn_gate']:expr=f'if layer % 2 == 0 {{ {expr} }} else {{ tensor("{suffix}", 1, &[4096,512])? }}'
 if name=='router_bias':expr=f'if layer == 2 {{ tensor("ffn_gate_tid2eid.weight", 26, &[6,129280])? }} else {{ {expr} }}'
 code+=f'        let {name} = {expr};\n'
a=block.index('    let layer14_weights =');raw=block[a:].replace('layer14_','').replace('.absolute_offset','.0').replace('.bytes','.1');code+=raw+'        Ok(Self { weights, compressor, indexer_q, indexer_weight })\n    }\n}\n'
Path('rust-star/runtime/src/long_prefill.rs').write_text(code)
# Retain the exact 8K context for subsequent chunks. The existing public probe
# still drops it when returning just its report.
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();a=s.index('    pub fn run_long_prefill_continuation_bootstrap_probe(');b=s.index('    pub fn run_long_prefill_sequential_continuation_probe(',a);t=s[a:b];t=t.replace('pub fn run_long_prefill_continuation_bootstrap_probe','fn run_complete_8k_in_context',1).replace('Result<LongPrefillContinuationBootstrapProbeReport>','Result<(Context, LongPrefillContinuationBootstrapProbeReport)>',1).replace('        drop(context);\n','').replace('Ok(LongPrefillContinuationBootstrapProbeReport {','Ok((context, LongPrefillContinuationBootstrapProbeReport {',1);i=t.rfind('        })');t=t[:i]+t[i:].replace('        })','        }))',1);wrapper='''    pub fn run_long_prefill_continuation_bootstrap_probe(model: &MappedModel) -> Result<LongPrefillContinuationBootstrapProbeReport> {
        Ok(run_complete_8k_in_context(model)?.1)
    }

''';s=s[:a]+wrapper+t+s[b:];s=s.replace('    include!("remaining_prefill.rs");','    include!("remaining_prefill.rs");\n    include!("long_prefill.rs");');i=s.index('        fn rust_star_metal_release_remaining_prefill_transients(');s=s[:i]+'''        fn rust_star_metal_begin_long_prefill_chunk(context: *mut c_void, chunk_start: u32, error: *mut c_char, error_bytes: usize) -> c_int;
        fn rust_star_metal_bind_long_prefill_layer(context: *mut c_void, layer: u32, error: *mut c_char, error_bytes: usize) -> c_int;
        fn rust_star_metal_finish_long_prefill_layer(context: *mut c_void, layer: u32, checksum: *mut u64, error: *mut c_char, error_bytes: usize) -> c_int;
        fn rust_star_metal_prepare_long_prefill_layer2_q(context: *mut c_void, model_mapping: *const c_void, model_bytes: u64, q_b_offset: u64, q_b_bytes: u64, error: *mut c_char, error_bytes: usize) -> c_int;
''' +s[i:];p.write_text(s)
