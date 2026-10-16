from pathlib import Path
import re
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
needle='@property(nonatomic, strong) id<MTLBuffer> prefillLayer14PrefixAfterFfnHc;'
s=s.replace(needle,needle+'\n@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLBuffer>> *prefillLayerRemainingPrefixHc;')
start=s.index('int rust_star_metal_begin_prefill_layer13_prefix_rebuild(');end=s.index('int rust_star_metal_run_prefill_layer4_continuation_boundary(',start)
odd=s[start:end]
start=s.index('int rust_star_metal_begin_prefill_layer14_rebuild(');end=s.index('int rust_star_metal_begin_prefill_layer13_prefix_rebuild(',start)
even=s[start:end]
helpers='''/* Remaining layers use the same validated odd/even slots. Layer-scoped
 * state stays strongly owned; only the prefix HC handoff needs a new slot. */
static id<MTLBuffer> remaining_layer_buffer(RustStarMetalContext *context,
                                          uint32_t layer, NSString *role)
{
    if (layer >= 15u && [role isEqualToString:@"PrefixAfterFfnHc"])
        return context.prefillLayerRemainingPrefixHc[@(layer)];
    return [context valueForKey:[NSString stringWithFormat:@"prefillLayer%u%@", layer, role]];
}

static void set_remaining_layer_buffer(RustStarMetalContext *context,
                                      uint32_t layer, NSString *role,
                                      id<MTLBuffer> buffer)
{
    if (layer >= 15u && [role isEqualToString:@"PrefixAfterFfnHc"]) {
        if (!context.prefillLayerRemainingPrefixHc)
            context.prefillLayerRemainingPrefixHc = [NSMutableDictionary new];
        if (buffer) context.prefillLayerRemainingPrefixHc[@(layer)] = buffer;
        else [context.prefillLayerRemainingPrefixHc removeObjectForKey:@(layer)];
    } else {
        [context setValue:buffer forKey:[NSString stringWithFormat:@"prefillLayer%u%@", layer, role]];
    }
}

'''
prototypes=[]
for block,base,prev,parity in [(odd,13,12,1),(even,14,13,0)]:
 names=re.findall(r'int (rust_star_metal_\w+)\(',block)
 for name in names:
  new=name.replace(f'layer{base}', 'remaining_odd' if parity else 'remaining_even')
  block=block.replace(name,new)
  prototypes.append(f'int {new}(void *context, uint32_t layer, char *error, size_t error_bytes);')
 block=block.replace('    void *opaque_context,','    void *opaque_context,\n    uint32_t layer,')
 block=block.replace('if (!opaque_context) {',f'if (!opaque_context || layer < 15u || layer > 42u || layer%2u != {parity}u) {{')
 # Rewrite dynamic-layer writes first, then all remaining reads.
 for n,expr in [(base,'layer'),(prev,'layer-1u')]:
  pattern=rf'context\.prefillLayer{n}([A-Za-z]+)\s*=\s*([^;]+);'
  block=re.sub(pattern,lambda m:f'set_remaining_layer_buffer(context, {expr}, @"{m[1]}", {m[2]});',block)
  block=re.sub(rf'context\.prefillLayer{n}([A-Za-z]+)',lambda m:f'remaining_layer_buffer(context, {expr}, @"{m[1]}")',block)
 block=block.replace(f'layer-{base}','remaining layer').replace(f'layer-{prev}','preceding layer')
 helpers+=block
at=s.index('int rust_star_metal_run_prefill_layer4_continuation_boundary(');s=s[:at]+helpers+s[at:];p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text();at=s.index('int rust_star_metal_run_prefill_layer4_continuation_boundary(');s=s[:at]+'\n'.join(prototypes)+'\n\n'+s[at:];p.write_text(s)
Path('rust-star/.work/remaining-layers-v30/binding-names.txt').write_text('\n'.join(prototypes))
