from pathlib import Path
import re
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text()
def rename(x):return x.replace('layer12','layer14').replace('LAYER12','LAYER14').replace('layer-12','layer-14').replace('blk.12.','blk.14.')
def duplicate(start,end,offset=0):
 global s
 a=s.index(start,offset);b=s.index(end,a);part=s[a:b];s=s[:b]+rename(part)+s[b:]
# Fields and mappings, all bounded by the final layer-12 field.
pat=r'(?m)^([ \t]*(?:pub )?layer12_prefix_boundary_dispatches:.*?^[ \t]*(?:pub )?layer12_continuation_tail_checksums:[^\n]*\n)'
s,n=re.subn(pat,lambda m:m[0]+rename(m[0]),s,flags=re.S|re.M);assert n==4,n
# Constants will be populated only from independently verified captures.
duplicate('const PREFILL_LAYER12_PREFIX_BOUNDARY_CHECKSUMS','const PREFILL_LAYER5_CONTINUATION_INGRESS_CHECKSUMS')
duplicate('        || report.layer12_prefix_boundary_dispatches','        || !report.sparse_transition.wall_ms')
duplicate('    let layer12_insertion = format!(', '    let complete_layer5_claim')
duplicate('        fn rust_star_metal_begin_prefill_layer12_rebuild','        fn rust_star_metal_run_prefill_layer4_continuation_boundary')
o=s.index('        let layer12_hc_fn',s.index('        let layer12_hc_fn')+1)
duplicate('        let layer12_hc_fn','        let tail_weights',o)
duplicate('        let layer12_weights = RawPrefillLayerWeights','        let expected_q_current')
a=s.index('        let layer12_begin_succeeded');b=s.index('        let layer13_prefix_begin_succeeded',a) if '        let layer13_prefix_begin_succeeded' in s[a:] else -1
# End immediately before the next layer begins.
if b<0:
 b=s.index('        let layer13_',s.index('"Metal layer-12 rebuild finalization failed:',a))
part=rename(s[a:b]);at=s.index('        layer23_loop.wall_ms += layer6_prefix_raw.wall_ms',b);s=s[:at]+part+s[at:]
s=s.replace('+ layer13_prefix_raw.wall_ms;','+ layer13_prefix_raw.wall_ms\n            + layer14_prefix_raw.wall_ms + layer14_prefix_complete_raw.wall_ms + layer14_continuation_raw.wall_ms;').replace('+ layer13_prefix_raw.gpu_ms;','+ layer13_prefix_raw.gpu_ms\n            + layer14_prefix_raw.gpu_ms + layer14_prefix_complete_raw.gpu_ms + layer14_continuation_raw.gpu_ms;')
s=s.replace('bootstrap-probe-v28','bootstrap-probe-v29')
p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();a=s.index('int rust_star_metal_begin_prefill_layer12_rebuild(');b=s.index('int rust_star_metal_begin_prefill_layer13_prefix_rebuild(',a)
block=rename(s[a:b]).replace('Layer12','Layer14').replace('Layer11','Layer13').replace('layer-11','layer-13')
s=s[:b]+block+s[b:];needle='@property(nonatomic, strong) id<MTLBuffer> prefillLayer14AfterFfnHc;';s=s.replace(needle,needle+'\n@property(nonatomic, strong) id<MTLBuffer> prefillLayer14PrefixAfterFfnHc;');p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();a=s.index('/* Publishes exact retained layer-11 HC and layer-12');b=s.index('/* Consumes one retained production-batch layer-3 HC',a);s=s[:b]+rename(s[a:b]).replace('layer-11','layer-13')+s[b:];p.write_text(s)
