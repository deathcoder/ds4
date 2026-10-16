#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>
static NSString *const kProbeSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"kernel void rust_star_dispatch_probe(device uint *values [[buffer(0)]],\n"
    @"    uint gid [[thread_position_in_grid]]) {\n"
    @"    values[gid] = values[gid] * 1664525u + 1013904223u;\n"
    @"}\n"
    @"kernel void rust_star_touch_u8_stride(device const uchar *src [[buffer(0)]],\n"
    @"    device uchar *out [[buffer(1)]], constant ulong &stride [[buffer(2)]],\n"
    @"    constant ulong &bytes [[buffer(3)]], constant ulong &dst_offset [[buffer(4)]],\n"
    @"    uint gid [[thread_position_in_grid]]) {\n"
    @"    const ulong offset = min((ulong)gid * stride, bytes - 1ul);\n"
    @"    out[dst_offset + gid] = src[offset];\n"
    @"}\n"
    @"kernel void rust_star_argmax_f32(device const float *values [[buffer(0)]],\n"
    @"    device uint *result [[buffer(1)]], constant uint &count [[buffer(2)]],\n"
    @"    uint tid [[thread_index_in_threadgroup]],\n"
    @"    uint lane [[thread_index_in_simdgroup]],\n"
    @"    uint simd_group [[simdgroup_index_in_threadgroup]],\n"
    @"    uint3 group_size [[threads_per_threadgroup]]) {\n"
    @"    float best = -INFINITY;\n"
    @"    uint best_id = UINT_MAX;\n"
    @"    uint invalid = 0u;\n"
    @"    for (uint i = tid; i < count; i += group_size.x) {\n"
    @"        const float value = values[i];\n"
    @"        if (!isfinite(value)) invalid = 1u;\n"
    @"        else if (value > best || (value == best && i < best_id)) { best = value; best_id = i; }\n"
    @"    }\n"
    @"    for (uint offset = 16u; offset != 0u; offset >>= 1u) {\n"
    @"        const float other = simd_shuffle_down(best, offset);\n"
    @"        const uint other_id = simd_shuffle_down(best_id, offset);\n"
    @"        const uint other_invalid = simd_shuffle_down(invalid, offset);\n"
    @"        if (lane + offset < 32u && (other > best || (other == best && other_id < best_id))) {\n"
    @"            best = other; best_id = other_id;\n"
    @"        }\n"
    @"        if (lane + offset < 32u) invalid += other_invalid;\n"
    @"    }\n"
    @"    threadgroup float group_best[32];\n"
    @"    threadgroup uint group_id[32];\n"
    @"    threadgroup uint group_invalid[32];\n"
    @"    if (lane == 0u) { group_best[simd_group] = best; group_id[simd_group] = best_id; group_invalid[simd_group] = invalid; }\n"
    @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    @"    if (simd_group == 0u) {\n"
    @"        const uint groups = (group_size.x + 31u) / 32u;\n"
    @"        best = lane < groups ? group_best[lane] : -INFINITY;\n"
    @"        best_id = lane < groups ? group_id[lane] : UINT_MAX;\n"
    @"        invalid = lane < groups ? group_invalid[lane] : 0u;\n"
    @"        for (uint offset = 16u; offset != 0u; offset >>= 1u) {\n"
    @"            const float other = simd_shuffle_down(best, offset);\n"
    @"            const uint other_id = simd_shuffle_down(best_id, offset);\n"
    @"            const uint other_invalid = simd_shuffle_down(invalid, offset);\n"
    @"            if (lane + offset < 32u && (other > best || (other == best && other_id < best_id))) {\n"
    @"                best = other; best_id = other_id;\n"
    @"            }\n"
    @"            if (lane + offset < 32u) invalid += other_invalid;\n"
    @"        }\n"
    @"        if (lane == 0u) { result[0] = invalid == 0u ? best_id : UINT_MAX; result[1] = invalid; }\n"
    @"    }\n"
    @"}\n";


int main(int argc, char **argv) { @autoreleasepool {
 if(argc!=2)return 2;
 NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
 const uint32_t n=129280,rows=128;
 if(data.length != (NSUInteger)n*rows*4)return 3;
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError *error=nil;
 id<MTLLibrary>lib=[d newLibraryWithSource:kProbeSource options:[MTLCompileOptions new] error:&error];
 id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rust_star_argmax_f32"] error:&error];
 if(!p){fprintf(stderr,"%s",error.description.UTF8String);return 4;}
 id<MTLBuffer>input=[d newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
 id<MTLBuffer>output=[d newBufferWithLength:rows*8 options:MTLResourceStorageModeShared];
 id<MTLCommandQueue>q=[d newCommandQueue];uint32_t expected[128];const float *f=data.bytes;
 for(uint32_t r=0;r<rows;r++){uint32_t best=0;for(uint32_t i=1;i<n;i++)if(f[r*n+i]>f[r*n+best])best=i;expected[r]=best;}
 for(uint32_t rep=0;rep<100;rep++){ @autoreleasepool {
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];
 [e setComputePipelineState:p];
 for(uint32_t r=0;r<rows;r++){
 [e setBuffer:input offset:(NSUInteger)r*n*4 atIndex:0];[e setBuffer:output offset:r*8 atIndex:1];[e setBytes:&n length:4 atIndex:2];
 [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(1024,p.maxTotalThreadsPerThreadgroup),1,1)];}
 [e endEncoding];[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted)return 5;
 uint32_t *actual=output.contents;for(uint32_t r=0;r<rows;r++)if(actual[r*2]!=expected[r]||actual[r*2+1]){printf("FAIL rep=%u row=%u actual=%u expected=%u invalid=%u\n",rep,r,actual[r*2],expected[r],actual[r*2+1]);return 1;}
 }}
 puts("PASS 12800 GPU argmax fixture selections against independent CPU argmax");return 0;
}}
