// Compiled with the actual native shim; factory and policy under test are not copied.
#include "../../runtime/src/metal_shim.m"

// Verbatim oracle-v5 ds4_metal.m policy, lines 3870-3877.
static uint32_t ds4_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg) {
    uint32_t nsg = 1;
    while (2u * nwg * nsg * ncpsg < n_keys && nsg < 4u) {
        nsg *= 2u;
    }
    return nsg;
}
static id<MTLLibrary> oracle_library;
static id<MTLComputePipelineState> oracle_reduce;
static id<MTLComputePipelineState> reference_pipeline(RustStarMetalContext *c, uint32_t nkeys, uint32_t nsg) {
    bool yes=true,no=false,pad=nkeys%32!=0;int32_t dim=512,groups=nsg,nwg=32;
    MTLFunctionConstantValues *v=[MTLFunctionConstantValues new];
    [v setConstantValue:&yes type:MTLDataTypeBool atIndex:400];
    [v setConstantValue:&yes type:MTLDataTypeBool atIndex:401];
    [v setConstantValue:&no type:MTLDataTypeBool atIndex:402];
    [v setConstantValue:&no type:MTLDataTypeBool atIndex:403];
    [v setConstantValue:&pad type:MTLDataTypeBool atIndex:404];
    [v setConstantValue:&no type:MTLDataTypeBool atIndex:405];
    [v setConstantValue:&dim type:MTLDataTypeInt atIndex:420];
    [v setConstantValue:&dim type:MTLDataTypeInt atIndex:421];
    [v setConstantValue:&groups type:MTLDataTypeInt atIndex:422];
    [v setConstantValue:&nwg type:MTLDataTypeInt atIndex:423];
    NSError *err=nil;id<MTLFunction> f=[oracle_library newFunctionWithName:@"kernel_flash_attn_ext_vec_f16_dk512_dv512" constantValues:v error:&err];
    id<MTLComputePipelineState> p=f?[c.device newComputePipelineStateWithFunction:f error:&err]:nil;
    if(!p){fprintf(stderr,"reference pipeline: %s\n",err.description.UTF8String);exit(2);}return p;
}
static uint32_t rng(uint32_t *s){uint32_t x=*s;x^=x<<13;x^=x>>17;x^=x<<5;return *s=x;}
static float value(uint32_t *s){return ((int32_t)(rng(s)%20001)-10000)/8192.0f;}
static id<MTLBuffer> buffer(RustStarMetalContext *c,NSUInteger bytes){
    id<MTLBuffer> b=[c.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if(!b)exit(3);memset(b.contents,0,b.length);return b;
}
static void dispatch_probe(RustStarMetalContext *c,id<MTLComputePipelineState> p,uint32_t nsg,uint32_t keys,NSArray<id<MTLBuffer>> *b,id<MTLBuffer> output, BOOL reference){
    const uint64_t row=512*2,mask=keys*2ull;
    rust_star_flash_pad_args pad={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=row,.nb12=keys*row,.nb13=keys*row,.nb21=row,.nb22=keys*row,.nb23=keys*row,.ne31=1,.ne32=1,.ne33=1,.nb31=mask,.nb32=mask,.nb33=mask};
    rust_star_flash_vec_args a={.ne01=1,.ne02=64,.ne03=1,.nb01=64*512*4,.nb02=512*4,.nb03=64*512*4,
        .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=row,.nb12=keys*row,.nb13=keys*row,
        .ns20=512,.nb21=row,.nb22=keys*row,.nb23=keys*row,.ne31=1,.ne32=1,.ne33=1,
        .nb31=mask,.nb32=mask,.nb33=mask,.ne1=64,.ne2=1,.ne3=1,.scale=1.0f/sqrtf(512.0f)};
    rust_star_flash_reduce_args r={.nrows=64};
    id<MTLCommandBuffer> cmd=[c.queue commandBuffer];id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
    [e setComputePipelineState:c.flashPadPipeline];[e setBytes:&pad length:sizeof(pad) atIndex:0];
    [e setBuffer:b[1] offset:0 atIndex:1];[e setBuffer:b[1] offset:0 atIndex:2];[e setBuffer:b[2] offset:0 atIndex:3];[e setBuffer:b[4] offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setComputePipelineState:p];[e setBytes:&a length:sizeof(a) atIndex:0];
    [e setBuffer:b[0] offset:0 atIndex:1];[e setBuffer:b[1] offset:0 atIndex:2];[e setBuffer:b[1] offset:0 atIndex:3];[e setBuffer:b[2] offset:0 atIndex:4];
    [e setBuffer:b[3] offset:0 atIndex:5];[e setBuffer:b[4] offset:0 atIndex:6];[e setBuffer:b[5] offset:0 atIndex:7];
    [e setThreadgroupMemoryLength:3328u*nsg atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,64,32) threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
    [e setComputePipelineState:reference?oracle_reduce:c.flashReducePipeline];[e setBytes:&r length:sizeof(r) atIndex:0];
    [e setBuffer:b[5] offset:0 atIndex:1];[e setBuffer:output offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
    [e endEncoding];[cmd commit];[cmd waitUntilCompleted];
    if(cmd.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",cmd.error.description.UTF8String);exit(4);}
}
int main(int argc,const char **argv){@autoreleasepool{
    RustStarMetalContext *c=[RustStarMetalContext new];c.device=MTLCreateSystemDefaultDevice();c.queue=[c.device newCommandQueue];
    char error[4096]={0};if(!ensure_attention_ingress_pipelines(c,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
    if(argc!=2)return 9;
    NSError *err=nil;
    NSString *shader=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];
    NSString *source=[@"#include <metal_stdlib>\nusing namespace metal;\n#define N_SIMDWIDTH 32\n#define MIN(x,y) ((x)<(y)?(x):(y))\n#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n" stringByAppendingString:shader];
    oracle_library=[c.device newLibraryWithSource:source options:[MTLCompileOptions new] error:&err];
    if(!oracle_library){fprintf(stderr,"oracle library: %s\n",err.description.UTF8String);return 10;}
    int32_t dim=512,nwg=32;MTLFunctionConstantValues *v=[MTLFunctionConstantValues new];
    [v setConstantValue:&dim type:MTLDataTypeInt atIndex:500];[v setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];
    id<MTLFunction> f=[oracle_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce" constantValues:v error:&err];
    oracle_reduce=f?[c.device newComputePipelineStateWithFunction:f error:&err]:nil;
    if(!oracle_reduce){fprintf(stderr,"oracle reduce: %s\n",err.description.UTF8String);return 11;}
    printf("{\"scope\":\"synthetic GPU regression using actual native shim factory and oracle-v5 policy, separately compiled unmodified oracle-v5 flash attention and reduction; not model C0 or throughput\",\"device\":\"%s\"}\n",c.device.name.UTF8String);fflush(stdout);
    const uint32_t cases[]={128,1152,1664,2048,2049,2176,2177,4096,4097};
    for(NSUInteger ci=0;ci<sizeof(cases)/sizeof(*cases);ci++){@autoreleasepool{
        uint32_t keys=cases[ci],nsg=flash_vec_decode_nsg(keys),ref=ds4_gpu_flash_attn_vec_nsg(keys,32,32);
        if(nsg!=ref)return 5;
        NSArray *b=@[buffer(c,64*512*4),buffer(c,keys*512*2),buffer(c,keys*2),buffer(c,64*4),buffer(c,2*32*512*2+32*2),buffer(c,64*(512*32+2*32)*4)];
        uint32_t seed=0x8d351f21u+(uint32_t)ci;
        float *q=[b[0] contents],*sinks=[b[3] contents];__fp16 *kv=[b[1] contents];
        for(int i=0;i<64*512;i++)q[i]=value(&seed);
        for(uint32_t i=0;i<keys*512;i++)kv[i]=(__fp16)value(&seed);
        for(int i=0;i<64;i++)sinks[i]=value(&seed);
        id<MTLBuffer> actual=buffer(c,64*512*4),expected=buffer(c,64*512*4),old=buffer(c,64*512*4);
        id<MTLComputePipelineState> p=flash_vec_decode_pipeline(c,nsg,error,sizeof(error));
        if(!p){fprintf(stderr,"%s\n",error);return 6;}
        dispatch_probe(c,p,nsg,keys,b,actual,NO);
        dispatch_probe(c,reference_pipeline(c,keys,ref),ref,keys,b,expected,YES);
        dispatch_probe(c,c.flashVecPipeline,1,keys,b,old,NO);
        uint32_t *a=actual.contents,*e=expected.contents,*o=old.contents,diffs=0;uint64_t hash=0xcbf29ce484222325ull;
        for(int i=0;i<64*512;i++){
            if(a[i]!=e[i]||!isfinite(((float*)actual.contents)[i])){fprintf(stderr,"mismatch keys=%u word=%d actual=%08x expected=%08x\n",keys,i,a[i],e[i]);return 7;}
            diffs+=o[i]!=e[i];hash^=a[i];hash*=0x100000001b3ull;
        }
        if((keys<=2048&&diffs)||(keys>2048&&!diffs))return 8;
        printf("{\"keys\":%u,\"nsg\":%u,\"words\":32768,\"exact\":true,\"old_nsg1_different_words\":%u,\"fnv1a_u32\":%llu}\n",keys,nsg,diffs,(unsigned long long)hash);fflush(stdout);
    }}printf("{\"passed\":true}\n");return 0;
}}
