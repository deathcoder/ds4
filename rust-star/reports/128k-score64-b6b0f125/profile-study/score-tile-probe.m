#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale;
} ScoreArgs;
static uint32_t random_word(uint32_t *state) {
    uint32_t x=*state;x^=x<<13;x^=x>>17;x^=x<<5;return *state=x;
}
static void fill(id<MTLBuffer> buffer,uint32_t *seed) {
    float *p=buffer.contents;for(NSUInteger i=0;i<buffer.length/4;i++) {
        uint32_t bits=0x3f000000u|(random_word(seed)&0x007fffffu);float x;memcpy(&x,&bits,4);p[i]=(x-0.75f)*2.0f;
    }
}
static double dispatch(id<MTLCommandQueue> q,id<MTLComputePipelineState> pipeline,
    ScoreArgs *args,NSArray *inputs,id<MTLBuffer> output,uint32_t columns) {
    id<MTLCommandBuffer> command=[q commandBuffer];id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
    [e setComputePipelineState:pipeline];[e setBytes:args length:sizeof(*args) atIndex:0];
    for(NSUInteger i=0;i<3;i++)[e setBuffer:inputs[i] offset:0 atIndex:i+1];
    [e setBuffer:output offset:0 atIndex:4];
    [e setThreadgroupMemoryLength:(8*128+columns*128)*2+8*columns*4 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((args->n_comp+columns-1)/columns,(args->n_tokens+7)/8,1)
        threadsPerThreadgroup:MTLSizeMake(32,columns/8,1)];
    [e endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",command.error.description.UTF8String);exit(2);}
    double ms=(command.GPUEndTime-command.GPUStartTime)*1000.0;if(!isfinite(ms)||ms<=0)exit(3);return ms;
}
static uint64_t verify(id<MTLBuffer> a,id<MTLBuffer> b,ScoreArgs args) {
    const uint32_t *x=a.contents,*y=b.contents;uint64_t hash=0xcbf29ce484222325ull;
    for(uint32_t row=0;row<args.n_tokens;row++)for(uint32_t col=0;col<args.n_comp;col++) {
        NSUInteger i=(NSUInteger)row*args.n_comp+col;float value;memcpy(&value,&x[i],4);
        BOOL visible=col<MIN((args.pos0+row+1)/args.ratio,args.n_comp);
        if(x[i]!=y[i]||(visible?!isfinite(value):x[i]!=0xff800000u)) {
            fprintf(stderr,"score mismatch/coverage row=%u col=%u baseline=%08x candidate=%08x visible=%d\n",row,col,x[i],y[i],visible);exit(4);
        }
        hash^=x[i];hash*=0x100000001b3ull;
    }return hash;
}
int main(int argc,const char **argv) {@autoreleasepool {
    if(argc!=2){fprintf(stderr,"usage: score-tile-probe COMBINED.metal\n");return 1;}
    NSError *error=nil;NSString *source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLLibrary> library=[device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
    if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 1;}
    NSMutableArray *pipelines=[NSMutableArray array];for(NSString *name in @[@"kernel_dsv4_indexer_scores_tiled",@"rust_star_indexer_scores_tiled64"]) {
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
        if(!p||p.maxTotalThreadsPerThreadgroup<(pipelines.count?256:128)){fprintf(stderr,"pipeline unavailable/undersized\n");return 1;}[pipelines addObject:p];
    }
    printf("{\"device\":\"%s\",\"scope\":\"synthetic isolated indexer scores; not engine throughput\",\"warmups_per_shape_per_kernel\":1,\"large_shape_pairs\":7}\n",device.name.UTF8String);fflush(stdout);
    const uint32_t cases[][3]={{9,1,0},{9,33,120},{9,65,248},{9,1025,4092},{32,1024,4064},{4096,8192,28672},{4096,32768,126976}};
    for(NSUInteger c=0;c<sizeof(cases)/sizeof(cases[0]);c++){@autoreleasepool {
        ScoreArgs args={.n_tokens=cases[c][0],.n_comp=cases[c][1],.pos0=cases[c][2],.n_head=64,.head_dim=128,.ratio=4,
            .q_token_stride=64*128*4,.q_head_stride=128*4,.weights_token_stride=64*4,.index_row_stride=128*4,
            .score_token_stride=cases[c][1]*4ull,.scale=1.0f/sqrtf(128.0f)};
        id<MTLBuffer> q=[device newBufferWithLength:args.n_tokens*args.q_token_stride options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights=[device newBufferWithLength:args.n_tokens*args.weights_token_stride options:MTLResourceStorageModeShared];
        id<MTLBuffer> k=[device newBufferWithLength:args.n_comp*args.index_row_stride options:MTLResourceStorageModeShared];
        id<MTLBuffer> a=[device newBufferWithLength:args.n_tokens*args.score_token_stride options:MTLResourceStorageModeShared];
        id<MTLBuffer> b=[device newBufferWithLength:args.n_tokens*args.score_token_stride options:MTLResourceStorageModeShared];
        if(!q||!weights||!k||!a||!b)return 1;
        uint32_t seed=0x8d351f21u+(uint32_t)c;fill(q,&seed);fill(weights,&seed);fill(k,&seed);
        memset(a.contents,0xff,a.length);memset(b.contents,0xff,b.length);NSArray *inputs=@[q,weights,k];
        dispatch(queue,pipelines[0],&args,inputs,a,32);dispatch(queue,pipelines[1],&args,inputs,b,64);
        uint64_t checksum=verify(a,b,args);
        printf("{\"shape\":%lu,\"tokens\":%u,\"compressed_rows\":%u,\"position\":%u,\"checked_words\":%llu,\"bitwise_match\":true,\"fnv1a_u32\":%llu}\n",(unsigned long)c,args.n_tokens,args.n_comp,args.pos0,(unsigned long long)a.length/4,(unsigned long long)checksum);fflush(stdout);
        if(args.n_tokens==4096)for(int pair=0;pair<7;pair++) {
            double times[2];for(int half=0;half<2;half++){int role=pair%2?1-half:half;times[role]=dispatch(queue,pipelines[role],&args,inputs,role?b:a,role?64:32);}
            printf("{\"shape\":%lu,\"pair\":%d,\"order\":\"%s\",\"baseline_gpu_ms\":%.9f,\"candidate_gpu_ms\":%.9f,\"ratio\":%.9f}\n",(unsigned long)c,pair+1,pair%2?"BA":"AB",times[0],times[1],times[1]/times[0]);fflush(stdout);
        }
        if(verify(a,b,args)!=checksum)return 5;
    }}printf("{\"passed\":true}\n");return 0;
}}
