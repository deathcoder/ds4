#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
typedef struct {int32_t ne00,ne01,ne02,ne03;uint64_t nb00,nb01,nb02,nb03;int32_t ne0,ne1,ne2,ne3,top_k;} Args;
typedef struct {float score;uint32_t id;} Item;
static int compare(const void *a,const void*b){const Item*x=a,*y=b;return x->score>y->score?-1:x->score<y->score?1:x->id<y->id?-1:x->id>y->id;}
int main(int argc,char**argv){@autoreleasepool{
 if(argc<2||argc>4)return 2;NSError *error=nil;
 NSString*source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();MTLCompileOptions*opts=[MTLCompileOptions new];opts.fastMathEnabled=NO;
 id<MTLLibrary>lib=[d newLibraryWithSource:source options:opts error:&error];id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_topk_stream512"] error:&error];
 if(!p){fprintf(stderr,"%s\n",error.description.UTF8String);return 3;}
 id<MTLCommandQueue>q=[d newCommandQueue];const uint32_t rows=32,cols=argc==4?(uint32_t)atoi(argv[3]):3072u,keep=512;
 if(cols<1025||cols>32768)return 2;const uint32_t repetitions=argc>=3?(uint32_t)atoi(argv[2]):2000u;
 id<MTLBuffer>input=[d newBufferWithLength:rows*cols*4 options:MTLResourceStorageModeShared];id<MTLBuffer>output=[d newBufferWithLength:rows*keep*4 options:MTLResourceStorageModeShared];
 uint32_t*expected=malloc(rows*keep*4);Item*items=malloc(cols*sizeof(Item));uint32_t seed=1729;float*f=input.contents;
 for(uint32_t r=0;r<rows;r++){for(uint32_t c=0;c<cols;c++){seed=seed*1664525u+1013904223u;float value=(float)(seed%10000u)/127.f;if(c>cols-64-r)value=-INFINITY;f[r*cols+c]=value;items[c]=(Item){value,c};}qsort(items,cols,sizeof(Item),compare);for(uint32_t k=0;k<keep;k++)expected[r*keep+k]=items[k].id;}
 Args args={.ne00=cols,.ne01=rows,.ne02=1,.ne03=1,.nb00=4,.nb01=cols*4,.ne0=keep,.ne1=rows,.ne2=1,.ne3=1,.top_k=keep};
 for(uint32_t rep=0;rep<repetitions;rep++){@autoreleasepool{
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];[e setComputePipelineState:p];[e setBytes:&args length:sizeof(args) atIndex:0];[e setBuffer:input offset:0 atIndex:1];[e setBuffer:output offset:0 atIndex:2];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"GPU ERROR %s\n",c.error.description.UTF8String);return 4;}
 uint32_t*a=output.contents;for(uint32_t i=0;i<rows*keep;i++)if(a[i]!=expected[i]){printf("FAIL repetition=%u row=%u rank=%u actual=%u expected=%u\n",rep,i/keep,i%keep,a[i],expected[i]);return 1;}
 if(rep%100==0){printf("PASS through repetition %u\n",rep);fflush(stdout);}
 }}printf("PASS %u repetitions, %u rows, %llu selected indices\n",repetitions,repetitions*rows,(unsigned long long)repetitions*rows*keep);free(items);free(expected);return 0;
}}
