#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"requirement failed line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main(int argc,const char **argv) { @autoreleasepool {
 REQUIRE(argc==2); NSError *error=nil;
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); REQUIRE(dev);
 NSString *source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error]; REQUIRE(source);
 MTLCompileOptions *options=[MTLCompileOptions new]; options.fastMathEnabled=NO;
 id<MTLLibrary> lib=[dev newLibraryWithSource:source options:options error:&error]; if(!lib)fprintf(stderr,"%s\n",error.description.UTF8String); REQUIRE(lib);
 id<MTLComputePipelineState> hash=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rust_star_router_hash_rows_batch"] error:&error];
 id<MTLComputePipelineState> clamp=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rust_star_router_clamp_sums_batch"] error:&error]; REQUIRE(hash && clamp);
 REQUIRE(hash.maxTotalThreadsPerThreadgroup>=256 && hash.maxTotalThreadsPerThreadgroup<4096);
 REQUIRE(clamp.maxTotalThreadsPerThreadgroup>=256 && clamp.maxTotalThreadsPerThreadgroup<4096);
 id<MTLCommandQueue> queue=[dev newCommandQueue]; REQUIRE(queue);
 id<MTLBuffer> table=[dev newBufferWithLength:129280*6*sizeof(int32_t) options:MTLResourceStorageModeShared];
 for(uint32_t t=0;t<129280;t++)for(uint32_t k=0;k<6;k++)((int32_t*)table.contents)[t*6+k]=(t*17+k*31)%256;
 uint32_t widths[]={32,64,1024}; uint64_t compared=0,guards=0;
 for(unsigned c=0;c<3;c++) { uint32_t rows=widths[c]; NSUInteger group=MIN(rows,256u);
  id<MTLBuffer> tokens=[dev newBufferWithLength:(rows+16)*sizeof(uint32_t) options:MTLResourceStorageModeShared];
  id<MTLBuffer> ids=[dev newBufferWithLength:(rows*6+16)*sizeof(int32_t) options:MTLResourceStorageModeShared];
  id<MTLBuffer> sums=[dev newBufferWithLength:(rows+16)*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> result=[dev newBufferWithLength:(rows+16)*sizeof(float) options:MTLResourceStorageModeShared];
  memset(ids.contents,0xa5,ids.length); memset(result.contents,0xa5,result.length);
  const float values[]={-INFINITY,-3.0f,0.0f,0x1p-15f,0x1p-14f,0x1.000002p-14f,1.0f,27.5f,INFINITY};
  for(uint32_t r=0;r<rows;r++) {((uint32_t*)tokens.contents)[r]=r%11==0?UINT32_MAX:(r*719u)%150000u;((float*)sums.contents)[r]=values[r%9];}
  id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  [e setComputePipelineState:hash];[e setBuffer:table offset:0 atIndex:0];[e setBuffer:tokens offset:0 atIndex:1];[e setBuffer:ids offset:0 atIndex:2];[e dispatchThreads:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
  [e setComputePipelineState:clamp];[e setBuffer:sums offset:0 atIndex:0];[e setBuffer:result offset:0 atIndex:1];[e dispatchThreads:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];REQUIRE(cb.status==MTLCommandBufferStatusCompleted);
  for(uint32_t r=0;r<rows;r++){uint32_t t=MIN(((uint32_t*)tokens.contents)[r],129279u);for(uint32_t k=0;k<6;k++){REQUIRE(((int32_t*)ids.contents)[r*6+k]==((int32_t*)table.contents)[t*6+k]);compared++;}float expected=fmaxf(((float*)sums.contents)[r],0x1p-14f); REQUIRE(memcmp(&expected,(float*)result.contents+r,4)==0);compared++;}
  for(uint32_t g=0;g<16;g++){REQUIRE(((uint32_t*)ids.contents)[rows*6+g]==0xa5a5a5a5);REQUIRE(((uint32_t*)result.contents)[rows+g]==0xa5a5a5a5);guards+=2;}
 }
 printf("{\"passed\":true,\"hash_max_threads\":%lu,\"clamp_max_threads\":%lu,\"compared_words\":%llu,\"guard_words\":%llu,\"rows\":[32,64,1024],\"repaired_group\":256,\"rejected_group\":4096}\n",(unsigned long)hash.maxTotalThreadsPerThreadgroup,(unsigned long)clamp.maxTotalThreadsPerThreadgroup,(unsigned long long)compared,(unsigned long long)guards);
 return 0;
}}
