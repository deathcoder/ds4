// Executes the production handoff and reset, with full-size synthetic histories.
#include "../../runtime/src/metal_shim.m"
static void require(BOOL ok, const char *why) { if (!ok) { fprintf(stderr,"FAILED: %s\n",why); exit(1); } }
static uint32_t pattern(uint32_t i, uint32_t seed) { return 0x3f000000u | ((i*1664525u+seed)&0x7fffffu); }
@interface ProbeQueue : NSObject
@property(nonatomic,strong) id<MTLCommandQueue> real;
@property(nonatomic,weak) RustStarMetalContext *context;
@property(nonatomic,strong) NSHashTable *sources;
@property(nonatomic,assign) uint32_t calls;
@property(nonatomic,assign) uint32_t rows;
@end
@implementation ProbeQueue
- (id<MTLCommandBuffer>)commandBuffer {
    uint32_t completed = self.rows > 131072u ? self.calls : 0u;
    require(self.context.prefillLayerChunkStates.count == 43u-completed,"per-command history ownership");
    uint32_t expected = 0u;
    for(uint32_t l=completed;l<43;l++) expected += l<2?1:l%2?4:7;
    require(self.sources.allObjects.count == expected,"completed command must release all source aliases");
    self.calls++;
    return [self.real commandBuffer];
}
- (id)forwardingTargetForSelector:(SEL)selector { return self.real; }
@end
static id<MTLComputePipelineState> fill_pipeline;
static id<MTLBuffer> allocation(RustStarMetalContext *c, NSUInteger bytes) {
    id<MTLBuffer> b=[c.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(b!=nil,"allocation"); return b;
}
static NSArray<NSString*> *roles;
static NSString *destination(uint32_t l,uint32_t r) {
    if(!r)return [NSString stringWithFormat:@"kv_cache_layer_%u",l];
    NSArray *keys=@[@"",@"compressed_kv_cache",@"compressor_state_kv",@"compressor_state_score",@"compressed_indexer_cache",@"indexer_state_kv",@"indexer_state_score"];
    return layer_buffer_key(keys[r],YES,l);
}
static NSUInteger source_words(uint32_t rows,uint32_t l,uint32_t r) {
    if(!r)return (NSUInteger)rows*512;
    uint32_t ratio=l%2?128:4;
    if(r==1)return (NSUInteger)(rows/ratio)*512;
    if(r==4)return (NSUInteger)(rows/4)*128;
    return r<4?(ratio==4?8192:65536):2048;
}
static NSUInteger destination_words(uint32_t rows,uint32_t l,uint32_t r) {
    if(!r)return 128*512;
    if(r==1)return (NSUInteger)((rows+256)/(l%2?128:4)+2)*512;
    if(r==4)return (NSUInteger)((rows+256)/4+2)*128;
    return source_words(rows,l,r);
}
static void prepare(RustStarMetalContext *c,ProbeQueue *q,uint32_t rows,uint32_t cycle) {
    c.prefillKvRows=rows;c.prefillLayerChunkStart=@(rows-4096);c.prefillLayerRemainingCompleted=@42;
    c.prefillLayerChunkStates=[NSMutableDictionary dictionary];
    c.prefillLayer42AfterFfnHc=allocation(c,256);
    c.activationBufferCache[@"prefill_transient_probe"]=allocation(c,256);
    q.sources=[NSHashTable weakObjectsHashTable];q.calls=0;q.rows=rows;
    for(uint32_t l=0;l<43;l++){@autoreleasepool{
        NSMutableDictionary *state=[NSMutableDictionary dictionary];
        id<MTLCommandBuffer> cmd=[q.real commandBuffer];id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
        [e setComputePipelineState:fill_pipeline];
        for(uint32_t r=0;r<(l<2?1:l%2?4:7);r++){
            NSUInteger n=source_words(rows,l,r);uint32_t seed=cycle*739u+l*503u+r*109u;
            id<MTLBuffer> b=allocation(c,n*4);state[roles[r]]=b;[q.sources addObject:b];
            [e setBuffer:b offset:0 atIndex:0];[e setBytes:&seed length:4 atIndex:1];
            [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            id<MTLBuffer> d=allocation(c,destination_words(rows,l,r)*4);memset(d.contents,0xa5,d.length);
            c.activationBufferCache[destination(l,r)]=d;
            // Mirror the final real executor-slot aliases, including overwritten slots.
            uint32_t slot=l<3?l:l%2?3:4;
            [c setValue:b forKey:[NSString stringWithFormat:@"prefillLayer%u%@",slot,roles[r]]];
        }
        [e endEncoding];[cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,"source GPU fill");
        c.prefillLayerChunkStates[@(l)]=state;
    }}
}
static void verify(RustStarMetalContext *c,uint32_t rows,uint32_t cycle,uint64_t *words,uint64_t *tails) {
    for(uint32_t l=0;l<43;l++)for(uint32_t r=0;r<(l<2?1:l%2?4:7);r++){
        id<MTLBuffer> b=c.activationBufferCache[destination(l,r)];uint32_t *p=b.contents;
        NSUInteger copied=r?source_words(rows,l,r):128*512;
        uint32_t offset=r?0:(rows-128)*512,seed=cycle*739u+l*503u+r*109u;
        for(NSUInteger i=0;i<copied;i++)if(p[i]!=pattern((uint32_t)i+offset,seed)){
            fprintf(stderr,"row=%u cycle=%u layer=%u role=%u word=%lu actual=%08x expected=%08x\n",rows,cycle,l,r,(unsigned long)i,p[i],pattern((uint32_t)i+offset,seed));exit(2);
        }
        for(NSUInteger i=copied;i<b.length/4;i++)require(p[i]==0xa5a5a5a5,"untouched destination tail");
        *words+=copied;*tails+=b.length/4-copied;
    }
}
int main(void){@autoreleasepool{
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError *err=nil;
    NSString *src=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void fill(device uint *p [[buffer(0)]], constant uint &seed [[buffer(1)]], uint i [[thread_position_in_grid]]) {p[i]=0x3f000000u|((i*1664525u+seed)&0x7fffffu);}";
    id<MTLLibrary> lib=[device newLibraryWithSource:src options:nil error:&err];
    fill_pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill"] error:&err];require(fill_pipeline!=nil,"fill pipeline");
    roles=@[@"FullKv",@"AttnCompressed",@"AttnStateKv",@"AttnStateScore",@"IndexerCompressed",@"IndexerStateKv",@"IndexerStateScore"];
    for(uint32_t rows=131072;rows<=262144;rows*=2){@autoreleasepool{
        RustStarMetalContext *c=[RustStarMetalContext new];c.device=device;c.activationBufferCache=[NSMutableDictionary dictionary];c.modelViewCache=[NSMutableDictionary dictionary];
        ProbeQueue *q=[ProbeQueue new];q.real=[device newCommandQueue];q.context=c;c.queue=(id<MTLCommandQueue>)q;
        id<MTLBuffer> model=allocation(c,4096);c.modelViewCache[@"synthetic-model"]=model;
        MTLResidencySetDescriptor *desc=[MTLResidencySetDescriptor new];id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:desc error:&err];
        require(residency!=nil,"residency creation");[residency addAllocation:model];[residency commit];[residency requestResidency];[q.real addResidencySet:residency];c.modelResidencySet=residency;c.modelResidencyAttached=YES;
        for(uint32_t cycle=1;cycle<=2;cycle++){@autoreleasepool{
            prepare(c,q,rows,cycle);@autoreleasepool{printf("{\"phase\":\"prepared\",\"rows\":%u,\"cycle\":%u,\"sources\":%lu}\n",rows,cycle,(unsigned long)q.sources.allObjects.count);fflush(stdout);}
            rust_star_metal_prefill_decode_handoff_result result={0};char error[4096]={0};
            int ok=rust_star_metal_adopt_prefill_decoder_state((__bridge void*)c,rows+256,&result,error,sizeof(error));
            if(!ok)fprintf(stderr,"%s\n",error);require(ok,"production handoff");
            uint64_t words=0,tails=0;verify(c,rows,cycle,&words,&tails);
            require(result.layers==43&&result.blit_copies==229&&result.command_buffers==(rows>131072?43:1)&&result.host_waits==result.command_buffers&&q.calls==result.command_buffers,"handoff counters");
            require(result.raw_rows_per_layer==128&&result.ratio4_rows_per_layer==rows/4&&result.ratio128_rows_per_layer==rows/128&&isfinite(result.wall_ms)&&result.wall_ms>0&&isfinite(result.gpu_ms)&&result.gpu_ms>=0,"handoff metadata");
            require(q.sources.allObjects.count==0&&!c.prefillLayerChunkStates&&!c.prefillLayer42AfterFfnHc&&c.prefillKvRows==0&&!c.activationBufferCache[@"prefill_transient_probe"],"all prefill ownership released");
            require(c.modelResidencySet==residency&&c.modelResidencyAttached&&[residency containsAllocation:model]&&c.modelViewCache[@"synthetic-model"]==model,"residency preserved");
            require(rust_star_metal_reset_request_state((__bridge void*)c,error,sizeof(error)),"production reset");
            require(c.activationBufferCache.count==0&&c.modelResidencySet==residency&&c.modelResidencyAttached&&[residency containsAllocation:model]&&c.modelViewCache[@"synthetic-model"]==model,"reset preserves model residency");
            printf("{\"rows\":%u,\"cycle\":%u,\"copied_words\":%llu,\"untouched_tail_words\":%llu,\"copies\":%u,\"commands\":%u,\"waits\":%u,\"exact\":true,\"per_command_source_release\":true,\"residency_preserved\":true,\"reset_passed\":true}\n",rows,cycle,words,tails,result.blit_copies,result.command_buffers,result.host_waits);fflush(stdout);
        }}
        [q.real removeResidencySet:residency];[residency endResidency];[residency removeAllAllocations];c.queue=nil;
    }}printf("{\"passed\":true,\"scope\":\"Production handoff and reset with full-size synthetic histories and small attached model residency allocation; real-model retained C0 required separately\"}\n");return 0;
}}
