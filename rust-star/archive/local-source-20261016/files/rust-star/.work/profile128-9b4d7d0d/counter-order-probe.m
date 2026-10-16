#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
int main(int argc,const char **argv) { @autoreleasepool {
    BOOL fenced=argc>1 && strcmp(argv[1],"fenced")==0;
    id<MTLDevice> d=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[d newCommandQueue]; NSError *err=nil;
    NSString *src=@"#include <metal_stdlib>\nusing namespace metal; kernel void work(device float *out [[buffer(0)]],constant uint &iterations [[buffer(1)]], uint i [[thread_position_in_grid]]) {float x=float(i%97)*0.01f; for(uint k=0;k<iterations;k++)x=fma(x,0.99999f,0.00001f);out[i]=x;}";
    id<MTLLibrary> lib=[d newLibraryWithSource:src options:nil error:&err]; id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"work"] error:&err];
    if(!p){fprintf(stderr,"%s\n",err.description.UTF8String);return 1;}
    MTLCounterSampleBufferDescriptor *sd=[MTLCounterSampleBufferDescriptor new]; for(id<MTLCounterSet> s in d.counterSets)if([s.name isEqualToString:MTLCommonCounterSetTimestamp])sd.counterSet=s;
    sd.sampleCount=18;sd.storageMode=MTLStorageModeShared;id<MTLCounterSampleBuffer> samples=[d newCounterSampleBufferWithDescriptor:sd error:&err]; if(!samples)return 2;
    NSMutableArray *buffers=[NSMutableArray array];for(int i=0;i<9;i++)[buffers addObject:[d newBufferWithLength:65536*4 options:MTLResourceStorageModeShared]];
    id<MTLFence> fence=[d newFence]; NSUInteger inversions=0;
    for(int run=0;run<100;run++){@autoreleasepool{
        id<MTLCommandBuffer> cmd=[q commandBuffer];
        for(int stage=0;stage<9;stage++){
            MTLComputePassDescriptor *desc=[MTLComputePassDescriptor computePassDescriptor];desc.sampleBufferAttachments[0].sampleBuffer=samples;desc.sampleBufferAttachments[0].startOfEncoderSampleIndex=2*stage;desc.sampleBufferAttachments[0].endOfEncoderSampleIndex=2*stage+1;
            id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoderWithDescriptor:desc];if(fenced&&stage)[e waitForFence:fence];
            [e setComputePipelineState:p];[e setBuffer:buffers[stage] offset:0 atIndex:0];uint32_t iterations=(stage==5?30000:stage==7?1000:100);[e setBytes:&iterations length:4 atIndex:1];[e dispatchThreadgroups:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];if(fenced)[e updateFence:fence];[e endEncoding];
        }
        [cmd commit];[cmd waitUntilCompleted];if(cmd.status!=MTLCommandBufferStatusCompleted)return 3;
        NSData *data=[samples resolveCounterRange:NSMakeRange(0,18)];if(data.length!=18*sizeof(MTLCounterResultTimestamp))return 4;const MTLCounterResultTimestamp *t=data.bytes;
        BOOL bad=NO;for(int i=0;i<18;i++)if(t[i].timestamp==MTLCounterErrorValue||(i&&t[i].timestamp<=t[i-1].timestamp))bad=YES;
        if(bad){inversions++;printf("{\"run\":%d,\"ticks\":[",run);for(int i=0;i<18;i++)printf("%s%llu",i?",":"",t[i].timestamp);printf("]}\n");}
    }}printf("{\"fenced\":%s,\"runs\":100,\"non_increasing_packets\":%lu}\n",fenced?"true":"false",(unsigned long)inversions);return 0;
}}
