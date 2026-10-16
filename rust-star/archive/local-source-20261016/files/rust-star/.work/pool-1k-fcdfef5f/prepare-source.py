from pathlib import Path
import json,re,hashlib
w=Path(__file__).resolve().parent
p=json.loads((w/'plan.json').read_text()); names=p['pool_contract']['labels']; persistent=p['pool_contract']['persistent_excluded'];s=(w/'prior-metal_shim.m').read_text();patches=[]
def edit(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b);patches.append(dict(before=a,after=b))
interface='''// One initialized scratch set, owned only during four native bootstrap tiles.
@interface RustStarBootstrapScratch : NSObject
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *buffers;
@property(nonatomic, strong) NSMutableSet<NSString *> *checkedOut;
@property(nonatomic, strong) id<MTLCommandBuffer> priorCommand;
@property(nonatomic) NSUInteger bytes;
@property(nonatomic) uint32_t frontier;
@property(nonatomic) uint32_t nextPosition;
@end
@implementation RustStarBootstrapScratch
@end

'''
edit('@interface RustStarMetalContext : NSObject',interface+'@interface RustStarMetalContext : NSObject')
edit('@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *prefillBoundaryCommands;', '@property(nonatomic, strong) RustStarBootstrapScratch *prefillLayerBootstrapScratch;\n@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *prefillBoundaryCommands;')
helper='''// newBufferWithLength returns zeroed storage. Reuse restores that entire
// initial range before the caller performs its existing CPU initialization.
// The caller checks prior GPU completion once, before any checkout in a tile.
static id<MTLBuffer> bootstrap_scratch_buffer(
    id<MTLDevice> device, RustStarBootstrapScratch *pool,
    NSString *label, NSUInteger bytes, const void *source,
    char *error, size_t error_bytes)
{
    if (!pool) {
        return source ? [device newBufferWithBytes:source length:bytes
            options:MTLResourceStorageModeShared] :
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    NSArray<NSString *> *allowed = @[
LABELS
    ];
    if (![allowed containsObject:label] || [pool.checkedOut containsObject:label] ||
        bytes == 0u || bytes > 2493173233ull ||
        (pool.priorCommand && pool.priorCommand.status != MTLCommandBufferStatusCompleted)) {
        fail_with_message(error, error_bytes, @"invalid bootstrap scratch checkout");
        return nil;
    }
    id<MTLBuffer> buffer = pool.buffers[label];
    if (buffer) {
        if (buffer.length != bytes) {
            fail_with_message(error, error_bytes, @"bootstrap scratch capacity changed within chunk");
            return nil;
        }
    } else {
        if (pool.buffers.count >= 105u || pool.bytes > 2493173233ull - bytes) {
            fail_with_message(error, error_bytes, @"bootstrap scratch capacity exceeded");
            return nil;
        }
        buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) {
            fail_with_message(error, error_bytes, @"bootstrap scratch allocation failed");
            return nil;
        }
        pool.buffers[label] = buffer;
        pool.bytes += bytes;
    }
    [pool.checkedOut addObject:label];
    if (source) memcpy(buffer.contents, source, bytes);
    else memset(buffer.contents, 0, bytes);
    return buffer;
}

'''.replace('LABELS',',\n'.join('        @"'+n+'"' for n in names))
(w/'pool-helper.inc').write_text(interface+helper)
edit('static int encode_dispatch(',helper+'static int encode_dispatch(') if 'static int encode_dispatch(' in s else None
# Use the known boundary entry as helper anchor if the generic name is absent.
if helper not in s:edit('int rust_star_metal_run_prefill_layer0_boundary(',helper+'int rust_star_metal_run_prefill_layer0_boundary(')
a=s.index('int rust_star_metal_run_prefill_layer0_boundary(');b=s.index('\nint ',a+4);f=s[a:b];orig=f;local=[]
def change(x,y):
 global f
 pattern = (r'(?<!\w)' if x[0].isalnum() else '') + re.escape(x)
 assert len(re.findall(pattern,f))==1,(x[:80],len(re.findall(pattern,f)));f=re.sub(pattern,lambda _:y,f);local.append(dict(before=x,after=y))
anchor='        RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;'
change(anchor,anchor+'''
        BOOL scratch_success = NO;
        @try {
        if (!native_1k_bootstrap && context.prefillLayerBootstrapScratch) {
            return fail_with_message(error, error_bytes,
                @"bootstrap scratch chunk was not completed");
        }
''')
anchor='        const BOOL retain_kv_state = kv_state_mode == 1u;'
change(anchor,'''        if (native_1k_bootstrap) {
            if (position_start == prefill_rows - 4096u) {
                if (context.prefillLayerBootstrapScratch) {
                    return fail_with_message(error, error_bytes,
                        @"bootstrap scratch owner already exists");
                }
                RustStarBootstrapScratch *created = [RustStarBootstrapScratch new];
                created.buffers = [NSMutableDictionary new];
                created.checkedOut = [NSMutableSet new];
                created.frontier = prefill_rows;
                created.nextPosition = position_start;
                context.prefillLayerBootstrapScratch = created;
            }
            RustStarBootstrapScratch *owner = context.prefillLayerBootstrapScratch;
            if (!owner || owner.frontier != prefill_rows ||
                owner.nextPosition != position_start) {
                return fail_with_message(error, error_bytes,
                    @"bootstrap scratch owner is missing or noncontiguous");
            }
            if (owner.priorCommand &&
                !command_succeeded(owner.priorCommand, error, error_bytes)) return 0;
            [owner.checkedOut removeAllObjects];
        }
        RustStarBootstrapScratch *scratch_pool = context.prefillLayerBootstrapScratch;
'''+anchor)
# Expand only scratch macro call sites, retaining original macro for persistent state.
routed=[]
for macro in ['RUST_STAR_NEW_L1_BUFFER','RUST_STAR_NEW_L2_BUFFER']:
 pattern=re.compile(r'\b'+macro+r'\((\w+),\s*([^;\\]*?)\);',re.S)
 for m in list(pattern.finditer(f)):
  name,expr=m.group(1,2)
  if name in names:
   change(m[0],f'{name} = bootstrap_scratch_buffer(context.device, scratch_pool,\n                @"{name}", ({expr}), NULL, error, error_bytes);');routed.append(name)
# Direct constructors (scratch only); retain fallback call for nonselected widths.
pattern=re.compile(r'(\w+_buffer) = (\[context.device\s+newBufferWith(?:Length:(.*?)\s+options:MTLResourceStorageModeShared|Bytes:(.*?)\s+length:(.*?)\s+options:MTLResourceStorageModeShared)\]);',re.S)
for m in list(pattern.finditer(f)):
 name=m[1]
 if name not in names:continue
 length=m[3] if m[3] is not None else m[5];source=m[4] if m[4] is not None else 'NULL'
 change(m[0],f'{name} = bootstrap_scratch_buffer(context.device, scratch_pool,\n            @"{name}", ({length}), {source}, error, error_bytes);');routed.append(name)
change('[context.device newBufferWithBytes:compressed_positions\n                length:compressor_tile_rows*sizeof(int32_t)\n                options:MTLResourceStorageModeShared]', 'bootstrap_scratch_buffer(context.device, scratch_pool,\n                @"compressed_position_buffer", compressor_tile_rows*sizeof(int32_t),\n                compressed_positions, error, error_bytes)')
routed.append('compressed_position_buffer')
assert sorted(routed)==names,(len(routed),set(names)-set(routed))
anchor='        rust_star_fp8_quantize_args kv_fp8_args = {'
change(anchor,'''        if (scratch_pool && (scratch_pool.checkedOut.count != 105u ||
            scratch_pool.buffers.count != 105u)) {
            return fail_with_message(error, error_bytes,
                @"bootstrap scratch checkout set is incomplete");
        }
'''+anchor)
anchor='            [context.prefillBoundaryCommands addObject:command];'
change(anchor,anchor+'\n            if (scratch_pool) scratch_pool.priorCommand = command;')
anchor='        return 1;\n    }\n}\n'
change(anchor,'''        if (scratch_pool) scratch_pool.nextPosition = position_start + rows;
        scratch_success = YES;
        return 1;
        } @finally {
            if (!scratch_success || position_start + rows == prefill_rows) {
                context.prefillLayerBootstrapScratch = nil;
            }
        }
    }
}
''')
s=s[:a]+f+s[b:];patches.append(dict(before=orig,after=f))
check=s
for x in reversed(patches):assert check.count(x['after'])==1;check=check.replace(x['after'],x['before'])
assert check==(w/'prior-metal_shim.m').read_text()
(w/'candidate-metal_shim.m').write_text(s)
for n in ['metal.rs','long_prefill.rs']:(w/('candidate-'+n)).write_bytes((w/('prior-'+n)).read_bytes())
(w/'source-edits.json').write_text(json.dumps(dict(patches=patches,routed=sorted(routed),inverse_exact=True),indent=2)+'\n')
print('Prepared105 allocations, exact inverse to prior bounded candidate')
