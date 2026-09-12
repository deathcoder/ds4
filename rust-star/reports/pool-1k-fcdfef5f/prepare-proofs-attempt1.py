from pathlib import Path
import json,re
w=Path(__file__).resolve().parent;p=json.loads((w/'plan.json').read_text());s=(w/'candidate-metal_shim.m').read_text();interface=(w/'pool-helper.inc').read_text().split('// newBufferWithLength')[0];helper='// newBufferWithLength'+(w/'pool-helper.inc').read_text().split('// newBufferWithLength',1)[1]
base='''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
static int fail_with_message(char *e, size_t n, NSString *s) { if(e&&n) snprintf(e,n,"%s",s.UTF8String); return 0; }
static int command_succeeded(id<MTLCommandBuffer> c,char *e,size_t n) { [c waitUntilCompleted]; return c.status==MTLCommandBufferStatusCompleted ? 1:fail_with_message(e,n,@"command failed"); }
'''+interface+helper
probe='''
@interface ProbeOwner : NSObject
@property(nonatomic,strong) RustStarBootstrapScratch *prefillLayerBootstrapScratch;
@end
@implementation ProbeOwner
@end
static int cleanup_probe(ProbeOwner *context, BOOL scratch_success,
    uint32_t position_start, uint32_t rows, uint32_t prefill_rows) {
    @try { return scratch_success ? 1:0; }
    @finally {
        if (!scratch_success || position_start + rows == prefill_rows) {
            context.prefillLayerBootstrapScratch = nil;
        }
    }
}
int main(void) { @autoreleasepool {
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();assert(device);
 id<MTLCommandQueue> queue=[device newCommandQueue];assert(queue);
 NSArray<NSString *> *labels=@[LABELS];
 char error[512]={0};NSUInteger checked=0, guards=0, reuse=0, failures=0;
 for(NSUInteger chunk=0;chunk<7;chunk++) { @autoreleasepool {
  ProbeOwner *owner=[ProbeOwner new];RustStarBootstrapScratch *pool=[RustStarBootstrapScratch new];
  pool.buffers=[NSMutableDictionary new];pool.checkedOut=[NSMutableSet new];owner.prefillLayerBootstrapScratch=pool;
  NSMutableDictionary *identities=[NSMutableDictionary new];
  for(NSUInteger tile=0;tile<4;tile++) { @autoreleasepool {
   if(pool.priorCommand) assert(command_succeeded(pool.priorCommand,error,sizeof(error)));
   [pool.checkedOut removeAllObjects];
   NSUInteger index=0;
   id<MTLCommandBuffer> verify=[queue commandBuffer];id<MTLBlitCommandEncoder> blit=[verify blitCommandEncoder];
   NSMutableArray *copies=[NSMutableArray new];NSMutableArray *expected=[NSMutableArray new];
   for(NSString *label in labels) {
    NSUInteger sizes[]={1,3,4,127,4096,4099,65536};NSUInteger n=sizes[index%7];
    NSMutableData *data=[NSMutableData dataWithLength:n];memset(data.mutableBytes,(int)(tile+chunk+31),n);
    BOOL is_copy=[label isEqual:@"token_buffer"]||[label isEqual:@"position_buffer"]||[label isEqual:@"compressed_position_buffer"];
    id<MTLBuffer> b=bootstrap_scratch_buffer(device,pool,label,n,is_copy?data.bytes:NULL,error,sizeof(error));assert(b&&b.length==n);
    if(tile) { assert([identities[label] unsignedLongLongValue]==(uintptr_t)(__bridge void *)b);reuse++; }
    else { for(NSNumber *v in identities.allValues)assert(v.unsignedLongLongValue!=(uintptr_t)(__bridge void *)b);identities[label]=@((uintptr_t)(__bridge void *)b); }
    id<MTLBuffer> dst=[device newBufferWithLength:n+64 options:MTLResourceStorageModeShared];memset(dst.contents,0x7b,n+64);
    // Blit offsets and size must align to four bytes on this backend. For odd
    // lengths, CPU validates the complete odd range; GPU copies an aligned prefix.
    if(n>=4)[blit copyFromBuffer:b sourceOffset:0 toBuffer:dst destinationOffset:32 size:n-(n%4)];
    const uint8_t *v=b.contents;for(NSUInteger j=0;j<n;j++)assert(v[j]==(is_copy?(tile+chunk+31):0));checked+=n;
    [copies addObject:dst];[expected addObject:@{@"n":@(n),@"value":@(is_copy?(tile+chunk+31):0)}];
    index++;
   }
   [blit endEncoding];[verify commit];assert(command_succeeded(verify,error,sizeof(error)));
   for(NSUInteger i=0;i<copies.count;i++) { id<MTLBuffer> dst=copies[i];NSUInteger n=[expected[i][@"n"] unsignedLongLongValue];uint8_t val=[expected[i][@"value"] unsignedIntValue];const uint8_t *v=dst.contents;NSUInteger copied=n-(n%4);for(NSUInteger j=0;j<copied;j++)assert(v[32+j]==val);checked+=copied;for(NSUInteger j=0;j<32;j++)assert(v[j]==0x7b);for(NSUInteger j=32+copied;j<n+64;j++)assert(v[j]==0x7b);guards+=64+(n%4); }
   assert(pool.buffers.count==105&&pool.checkedOut.count==105);
   assert(!bootstrap_scratch_buffer(device,pool,labels[0],1,NULL,error,sizeof(error)));failures++;
   id<MTLCommandBuffer> poison=[queue commandBuffer];id<MTLBlitCommandEncoder> poisonBlit=[poison blitCommandEncoder];
   for(id<MTLBuffer> b in pool.buffers.allValues)[poisonBlit fillBuffer:b range:NSMakeRange(0,b.length) value:0xa5];
   [poisonBlit endEncoding];pool.priorCommand=poison;
   // Not-enqueued prior work must be rejected without modifying storage.
   [pool.checkedOut removeObject:labels[0]];
   assert(!bootstrap_scratch_buffer(device,pool,labels[0],1,NULL,error,sizeof(error)));failures++;
   [poison commit];assert(command_succeeded(poison,error,sizeof(error)));
   for(id<MTLBuffer> b in pool.buffers.allValues){const uint8_t *v=b.contents;for(NSUInteger j=0;j<b.length;j++)assert(v[j]==0xa5);}
  }}
  [pool.checkedOut removeAllObjects];
  assert(!bootstrap_scratch_buffer(device,pool,@"persistent_forbidden",4,NULL,error,sizeof(error)));failures++;
  assert(!bootstrap_scratch_buffer(device,pool,labels[0],2,NULL,error,sizeof(error)));failures++;
  assert(!bootstrap_scratch_buffer(device,pool,labels[0],0,NULL,error,sizeof(error)));failures++;
  assert(!bootstrap_scratch_buffer(device,pool,labels[0],2493173234ull,NULL,error,sizeof(error)));failures++;
  assert(cleanup_probe(owner,YES,0,1024,4096)&&owner.prefillLayerBootstrapScratch==pool);
  assert(cleanup_probe(owner,YES,3072,1024,4096)&&!owner.prefillLayerBootstrapScratch);
  owner.prefillLayerBootstrapScratch=pool;assert(!cleanup_probe(owner,NO,0,1024,4096)&&!owner.prefillLayerBootstrapScratch);
  __weak RustStarBootstrapScratch *weakPool=pool;pool=nil;assert(!weakPool);
 }}
 printf("{\\"passed\\":true,\\"chunks\\":7,\\"tiles\\":28,\\"labels\\":105,\\"compared_bytes\\":%lu,\\"guard_bytes\\":%lu,\\"identity_reuses\\":%lu,\\"invalid_checkouts_rejected\\":%lu,\\"final_and_error_cleanup\\":true}\\n",checked,guards,reuse,failures);
 return 0;
}}
'''.replace('LABELS',','.join('@"'+n+'"' for n in p['pool_contract']['labels']))
(w/'pool-proof.m').write_text(base+probe)
# Instrumentation leaves the production helper byte-identical after reversing
# the one emission block and records only numeric identities, never owners.
emit='''    @autoreleasepool {
        fprintf(stderr, "RS_POOL_CHECKOUT %s\\n", [[NSString alloc] initWithData:
            [NSJSONSerialization dataWithJSONObject:@{@"label":label,
                @"bytes":@(bytes), @"identity":@((uintptr_t)(__bridge void *)buffer),
                @"owner":@((uintptr_t)(__bridge void *)pool),
                @"position":@(pool.nextPosition), @"frontier":@(pool.frontier),
                @"prior_status":@(pool.priorCommand ? pool.priorCommand.status : -1),
                @"copy":@(source != NULL), @"pool_bytes":@(pool.bytes),
                @"pool_count":@(pool.buffers.count)} options:0 error:nil]
            encoding:NSUTF8StringEncoding].UTF8String);
    }
'''
patches=[]
def edit(x,y):
 global s
 assert s.count(x)==1,(x[:60],s.count(x));s=s.replace(x,y);patches.append(dict(before=x,after=y))
edit('    else memset(buffer.contents, 0, bytes);','    else memset(buffer.contents, 0, bytes);\n'+emit)
edit('int rust_star_metal_run_prefill_layer0_boundary(', (w/'observer.inc').read_text()+'\nint rust_star_metal_run_prefill_layer0_boundary(')
snap=lambda phase:f'rs_lifetime_snapshot(context.device, context.prefillBoundaryCommands, @"{phase}", rows, position_start, prefill_rows);'
edit('        RustStarBootstrapScratch *scratch_pool = context.prefillLayerBootstrapScratch;', '        RustStarBootstrapScratch *scratch_pool = context.prefillLayerBootstrapScratch;\n        if (native_1k_bootstrap) '+snap('entry'))
inv=json.loads((w/'prior-instrumentation.json').read_text())['candidate'];block='        if (native_1k_bootstrap) {\n'+''.join(f'            rs_lifetime_register({n}, @"{n}", {"YES" if n in inv["persistent"] else "NO"}, position_start);\n' for n in inv['names'])+'            '+snap('registered')+'\n        }\n'
edit('        rust_star_fp8_quantize_args kv_fp8_args = {',block+'        rust_star_fp8_quantize_args kv_fp8_args = {')
edit('            [context.prefillBoundaryCommands addObject:command];','            [context.prefillBoundaryCommands addObject:command];\n            if (native_1k_bootstrap) '+snap('submitted'))
edit('                context.prefillBoundaryCommands = nil;','                if (native_1k_bootstrap) '+snap('waited')+'\n                context.prefillBoundaryCommands = nil;')
edit('        scratch_success = YES;\n        return 1;', '        scratch_success = YES;')
x='''        } @finally {
            if (!scratch_success || position_start + rows == prefill_rows) {
                context.prefillLayerBootstrapScratch = nil;
            }
        }
    }
}
'''
y=x[:-2]+'''    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    if (native_1k_bootstrap) {
        fprintf(stderr, "RS_POOL_OWNER %u %u %d\\n", position_start, prefill_rows,
            context.prefillLayerBootstrapScratch != nil);
        '''+snap('after_pool')+'''
    }
    return 1;
}
'''
edit(x,y)
check=s
for e in reversed(patches):assert check.count(e['after'])==1;check=check.replace(e['after'],e['before'])
assert check==(w/'candidate-metal_shim.m').read_text()
(w/'candidate-instrumented.m').write_text(s);(w/'instrumentation.json').write_text(json.dumps(dict(patches=patches,inverse_exact=True),indent=2)+'\n')
print('Synthetic proof and reversible pool observer prepared')
