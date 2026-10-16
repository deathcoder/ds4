from pathlib import Path
import re
p=Path('rust-star/runtime/src/metal.rs'); s=p.read_text()
def mapped(t): return t.replace('layer11','layer13').replace('LAYER11','LAYER13').replace('layer-11','layer-13')
def clone(start,end,before=None):
    global s
    a=s.index(start); b=s.index(end,a); block=mapped(s[a:b]); at=s.index(before) if before else b
    s=s[:at]+block+s[at:]
clone('const PREFILL_LAYER11_PREFIX_BOUNDARY_CHECKSUMS:', 'const PREFILL_LAYER12_PREFIX_BOUNDARY_CHECKSUMS:')
# Each of the public report, raw report, conversion, and test initializer has
# one contiguous layer-11 field block. Keep layer-specific accounting explicit.
patterns=[r'    pub layer11_prefix_boundary_dispatches:.*?(?=    pub layer12_prefix_boundary_dispatches:)',r'        layer11_prefix_boundary_dispatches: u32,.*?(?=        layer12_prefix_boundary_dispatches: u32,)',r'            layer11_prefix_boundary_dispatches: layer23_loop.*?(?=            layer12_prefix_boundary_dispatches:)',r'            layer11_prefix_boundary_dispatches: 17,.*?(?=            layer12_prefix_boundary_dispatches:)']
for pat in patterns:
    matches=list(re.finditer(pat,s,re.S)); assert len(matches)==1, (pat,len(matches))
    m=matches[0];s=s[:m.end()]+mapped(m.group())+s[m.end():]
clone('        || report.layer11_prefix_boundary_dispatches != 17','        || report.layer12_prefix_boundary_dispatches != 40')
clone('    let layer11_insertion = format!(', '    let layer12_insertion = format!(')
# Claim insertion must follow the existing layer-12 claim insertion.
a=s.index('    let complete_layer10_claim =');b=s.index('    let complete_layer11_claim =',a)
block=mapped(s[a:b]).replace('layer10','layer12').replace('layer-10','layer-12')
pos=s.index('    output.write_all(rendered.as_bytes())?;',b);s=s[:pos]+block+s[pos:]
clone('        fn rust_star_metal_begin_prefill_layer11_prefix_rebuild(', '        fn rust_star_metal_begin_prefill_layer12_rebuild(')
# Only the continuation runner model setup, rather than the independent 2K path.
a=s.index('        let layer11_hc_fn =', s.index('        let layer10_hc_fn =',s.index('pub fn run_long_prefill_sequential_continuation_probe')))
b=s.index('        let layer12_hc_fn =',a)
s=s[:b]+mapped(s[a:b]).replace('blk.11.','blk.13.')+s[b:]
clone('        let layer11_weights = RawPrefillLayerWeights {','        let layer12_weights = RawPrefillLayerWeights {')
a=s.index('        let layer11_prefix_begin_succeeded =');b=s.index('        let layer12_begin_succeeded =',a)
block=mapped(s[a:b]);pos=s.index('        layer23_loop.wall_ms += layer6_prefix_raw.wall_ms',b)
s=s[:pos]+block+s[pos:]
s=s.replace('+ layer12_continuation_raw.wall_ms;', '+ layer12_continuation_raw.wall_ms\n            + layer13_prefix_raw.wall_ms;').replace('+ layer12_continuation_raw.gpu_ms;', '+ layer12_continuation_raw.gpu_ms\n            + layer13_prefix_raw.gpu_ms;')
s=s.replace('rust-star-long-prefill-continuation-bootstrap-probe-v27','rust-star-long-prefill-continuation-bootstrap-probe-v28')
p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
a=s.index('int rust_star_metal_begin_prefill_layer11_prefix_rebuild(');b=s.index('int rust_star_metal_begin_prefill_layer12_rebuild(',a)
block=mapped(s[a:b]).replace('Layer11','Layer13').replace('Layer10','Layer12').replace('layer-10','layer-12')
pos=s.index('int rust_star_metal_run_prefill_layer4_continuation_boundary(',b);s=s[:pos]+block+s[pos:]
s=s.replace('@property(nonatomic, strong) id<MTLBuffer> prefillLayer13AfterFfnHc;', '@property(nonatomic, strong) id<MTLBuffer> prefillLayer13AfterFfnHc;\n@property(nonatomic, strong) id<MTLBuffer> prefillLayer13PrefixAfterFfnHc;')
p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();a=s.index('/* Publishes exact layer-10 prefix HC');b=s.index('/*',s.index('int rust_star_metal_finish_prefill_layer11_rebuild',a))
block=mapped(s[a:b]).replace('layer-10','layer-12');s=s[:b]+block+s[b:];p.write_text(s)
