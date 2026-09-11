#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arguments.h"
typedef struct ds4_metal_args_flash_attn_ext FlashArgs;
typedef struct ds4_metal_args_flash_attn_ext_pad PadArgs;
typedef struct ds4_metal_args_flash_attn_ext_blk BlockArgs;
enum { ROWS=4096, HEADS=64, WIDTH=512, RAW_ROWS=4224, GUARD=256 };
static void require(BOOL okay,const char *message) {if(!okay){fprintf(stderr,"FAIL: %s\n",message);exit(2);}}
static uint32_t random_word(uint32_t *state) {uint32_t x=*state;x^=x<<13;x^=x>>17;x^=x<<5;return *state=x;}
static float random_float(uint32_t *state) {uint32_t bits=0x3f000000u|(random_word(state)&0x007fffffu);float f;memcpy(&f,&bits,4);return (f-0.75f)*2.0f;}
static void completed(id<MTLCommandBuffer> c) {[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",c.error.description.UTF8String);exit(3);}}
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*n,MTLFunctionConstantValues*v) {
 NSError *error=nil;id<MTLFunction> f=[l newFunctionWithName:n constantValues:v error:&error];
 id<MTLComputePipelineState> p=f?[d newComputePipelineStateWithFunction:f error:&error]:nil;
 if(!p){fprintf(stderr,"pipeline %s: %s\n",n.UTF8String,error.description.UTF8String);exit(4);}return p;
}
static MTLFunctionConstantValues *attention_constants(BOOL padded) {
 MTLFunctionConstantValues *v=[MTLFunctionConstantValues new];BOOL yes=YES,no=NO;int32_t width=WIDTH,nsg=8;
 for(NSUInteger i=300;i<=301;i++)[v setConstantValue:&yes type:MTLDataTypeBool atIndex:i];
 for(NSUInteger i=302;i<=303;i++)[v setConstantValue:&no type:MTLDataTypeBool atIndex:i];
 [v setConstantValue:&padded type:MTLDataTypeBool atIndex:304];[v setConstantValue:&no type:MTLDataTypeBool atIndex:310];
 [v setConstantValue:&width type:MTLDataTypeInt atIndex:320];[v setConstantValue:&width type:MTLDataTypeInt atIndex:321];
 [v setConstantValue:&nsg type:MTLDataTypeInt atIndex:322];return v;
}
static id<MTLBuffer> buffer(id<MTLDevice>d,NSUInteger n) {id<MTLBuffer>b=[d newBufferWithLength:n options:MTLResourceStorageModeShared];require(b!=nil,"buffer allocation");return b;}
static uint64_t hash_inputs(NSArray<id<MTLBuffer>> *inputs) {
 uint64_t h=0xcbf29ce484222325ull;
 for(id<MTLBuffer> b in inputs){const uint8_t*p=b.contents;NSUInteger words=b.length/4;
  for(NSUInteger i=0;i<words;i++){uint32_t v;memcpy(&v,p+4*i,4);h^=v;h*=0x100000001b3ull;}
  for(NSUInteger i=4*words;i<b.length;i++){h^=p[i];h*=0x100000001b3ull;}}
 return h;
}
static void poison(id<MTLBuffer>b) {memset(b.contents,0xa5,GUARD);memset((uint8_t*)b.contents+GUARD,0xff,b.length-2*GUARD);memset((uint8_t*)b.contents+b.length-GUARD,0xa5,GUARD);}
static uint64_t verify(id<MTLBuffer>a,id<MTLBuffer>b) {
 for(id<MTLBuffer> out in @[a,b]){const uint8_t*p=out.contents;for(NSUInteger i=0;i<GUARD;i++)require(p[i]==0xa5&&p[out.length-GUARD+i]==0xa5,"output guard changed");}
 const uint32_t*x=(const uint32_t*)((uint8_t*)a.contents+GUARD),*y=(const uint32_t*)((uint8_t*)b.contents+GUARD);
 NSUInteger n=(a.length-2*GUARD)/4;uint64_t h=0xcbf29ce484222325ull,nonzero=0;BOOL varied=NO;
 for(NSUInteger i=0;i<n;i++){float f;memcpy(&f,x+i,4);if(x[i]!=y[i]||!isfinite(f)){fprintf(stderr,"output mismatch/nonfinite word=%lu baseline=%08x candidate=%08x\n",(unsigned long)i,x[i],y[i]);exit(5);}nonzero+=f!=0.0f;varied|=x[i]!=x[0];h^=x[i];h*=0x100000001b3ull;}
 require(nonzero>n/2&&varied,"degenerate attention output");return h;
}
static double dispatch(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,FlashArgs *a,NSArray<id<MTLBuffer>>*in,id<MTLBuffer>out,BOOL transposed) {
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];require(c&&e,"attention encoder");
 [e setComputePipelineState:p];[e setBytes:a length:sizeof(*a) atIndex:0];
 for(NSUInteger i=0;i<7;i++)[e setBuffer:in[i] offset:0 atIndex:i+1];
 [e setBuffer:out offset:GUARD atIndex:8];[e setThreadgroupMemoryLength:28672 atIndex:0];
 [e dispatchThreadgroups:MTLSizeMake(ROWS/8,HEADS,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];
 [e endEncoding];completed(c);double ms=(c.GPUEndTime-c.GPUStartTime)*1000.0;require(isfinite(ms)&&ms>0,"GPU interval");return ms;
}
static void prepare_and_verify(id<MTLCommandQueue>q,id<MTLComputePipelineState>padp,id<MTLComputePipelineState>blkp,
 id<MTLBuffer>kv,id<MTLBuffer>mask,id<MTLBuffer>pad,id<MTLBuffer>blocks,uint32_t keys,uint64_t counts[3]) {
 const uint64_t stride=WIDTH*2,kvbytes=keys*stride,maskbytes=ROWS*keys*2ull;
 PadArgs pa={.ne11=keys,.ne_12_2=1,.ne_12_3=1,.nb11=stride,.nb12=kvbytes,.nb13=kvbytes,.nb21=stride,.nb22=kvbytes,.nb23=kvbytes,.ne31=ROWS,.ne32=1,.ne33=1,.nb31=keys*2ull,.nb32=maskbytes,.nb33=maskbytes};
 BlockArgs ba={.ne01=ROWS,.ne30=keys,.ne31=ROWS,.ne32=1,.ne33=1,.nb31=keys*2ull,.nb32=maskbytes,.nb33=maskbytes};
 memset(blocks.contents,0xff,blocks.length);memset(pad.contents,0xa5,pad.length);
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];require(c&&e,"preparation encoder");
 if(keys%64){[e setComputePipelineState:padp];[e setBytes:&pa length:sizeof(pa) atIndex:0];[e setBuffer:kv offset:0 atIndex:1];[e setBuffer:kv offset:0 atIndex:2];[e setBuffer:mask offset:0 atIndex:3];[e setBuffer:pad offset:0 atIndex:4];[e dispatchThreadgroups:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];}
 [e setComputePipelineState:blkp];[e setBytes:&ba length:sizeof(ba) atIndex:0];[e setBuffer:mask offset:0 atIndex:1];[e setBuffer:blocks offset:0 atIndex:2];[e dispatchThreadgroups:MTLSizeMake((keys+63)/64,ROWS/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];completed(c);
 const uint16_t*m=mask.contents,*k=kv.contents,*pd=pad.contents;const uint8_t*bl=blocks.contents;
 for(uint32_t qb=0;qb<ROWS/8;qb++)for(uint32_t kb=0;kb<(keys+63)/64;kb++){
  uint32_t visible=0;uint8_t expected=1;
  if(kb*64+64<=keys){for(uint32_t r=qb*8;r<qb*8+8;r++)for(uint32_t col=kb*64;col<kb*64+64;col++)visible+=m[(NSUInteger)r*keys+col]==0;expected=visible==512?2:(visible?1:0);}
  require(bl[(NSUInteger)qb*((keys+63)/64)+kb]==expected,"CPU/GPU mask blocks differ");counts[expected]++;
 }
 if(keys%64){uint32_t rem=keys%64,base=keys-rem;
  for(uint32_t which=0;which<2;which++)for(uint32_t r=0;r<64;r++)for(uint32_t col=0;col<WIDTH;col++)require(pd[(NSUInteger)which*64*WIDTH+r*WIDTH+col]==(r<rem?k[(NSUInteger)(base+r)*WIDTH+col]:0),"CPU/GPU KV padding differs");
  for(uint32_t r=0;r<ROWS;r++)for(uint32_t col=0;col<64;col++)require(pd[2*64*WIDTH+(NSUInteger)r*64+col]==(col<rem?m[(NSUInteger)r*keys+base+col]:0xfbff),"CPU/GPU mask padding differs");
 }
 require(counts[0]&&counts[1]&&counts[2],"missing skipped/partial/full mask-block coverage");
}
int main(int argc,const char**argv){@autoreleasepool{
 require(argc==2,"usage: probe COMBINED.metal");NSError*error=nil;
 NSString*source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];require(source!=nil,"shader source");
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();require([d.name isEqualToString:@"Apple M1 Ultra"],"requires target Apple M1 Ultra");
 id<MTLCommandQueue>q=[d newCommandQueue];require(q!=nil,"command queue");
 id<MTLLibrary>l=[d newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];if(!l){fprintf(stderr,"%s\n",error.description.UTF8String);return 1;}
 id<MTLComputePipelineState> p[2][2];
 for(int padded=0;padded<2;padded++)for(int role=0;role<2;role++){
  p[padded][role]=pipeline(d,l,role?@"probe_flash_head_first":@"kernel_flash_attn_ext_f16_dk512_dv512",attention_constants(padded));
  require(p[padded][role].threadExecutionWidth==32&&p[padded][role].maxTotalThreadsPerThreadgroup>=256,"attention pipeline geometry");
  require(p[padded][role].staticThreadgroupMemoryLength+28672<=d.maxThreadgroupMemoryLength,"threadgroup memory limit");
 }
 for(int padded=0;padded<2;padded++)p[padded][1]=p[padded][0];
 int32_t cache=64,queries=8;BOOL yes=YES;MTLFunctionConstantValues*pc=[MTLFunctionConstantValues new];
 [pc setConstantValue:&yes type:MTLDataTypeBool atIndex:100];[pc setConstantValue:&cache type:MTLDataTypeInt atIndex:125];
 id<MTLComputePipelineState>padp=pipeline(d,l,@"kernel_flash_attn_ext_pad",pc);
 MTLFunctionConstantValues*bc=[MTLFunctionConstantValues new];[bc setConstantValue:&queries type:MTLDataTypeInt atIndex:224];[bc setConstantValue:&cache type:MTLDataTypeInt atIndex:225];
 id<MTLComputePipelineState>blkp=pipeline(d,l,@"kernel_flash_attn_ext_blk",bc);
 printf("{\"schema\":\"rust-star-compressed-grid-control-v1\",\"device\":\"%s\",\"scope\":\"A/A null control: same original pipeline and dispatch in both roles; no candidate or engine claim\",\"rows\":4096,\"heads\":64,\"width\":512,\"threads_per_group\":256,\"threadgroup_bytes\":28672,\"warmups_per_variant\":2,\"pairs_per_timed_case\":7,\"flash_args_bytes\":%lu,\"pad_args_bytes\":%lu,\"block_args_bytes\":%lu}\n",d.name.UTF8String,(unsigned long)sizeof(FlashArgs),(unsigned long)sizeof(PadArgs),(unsigned long)sizeof(BlockArgs));fflush(stdout);
 uint64_t compared=0;uint32_t timed_pairs=0;
 for(uint32_t shape=0;shape<7;shape++){@autoreleasepool{
  const uint32_t start=4096*(shape+1),comp=(start+ROWS)/128,keys=RAW_ROWS+comp;BOOL padded=keys%64!=0;
  const uint64_t qbytes=ROWS*HEADS*WIDTH*4ull,kvbytes=keys*WIDTH*2ull,maskbytes=ROWS*keys*2ull;
  id<MTLBuffer>qb=buffer(d,qbytes),kv=buffer(d,kvbytes),mask=buffer(d,maskbytes),sinks=buffer(d,HEADS*4),pad=buffer(d,padded?2*64*WIDTH*2+ROWS*64*2:1),blocks=buffer(d,(ROWS/8)*((keys+63)/64));
  id<MTLBuffer>a=buffer(d,qbytes+2*GUARD),b=buffer(d,qbytes+2*GUARD);
  uint32_t seed=0x8d351f21u+shape;float*query=qb.contents;__fp16*keys_data=kv.contents;
  for(NSUInteger i=0;i<qbytes/4;i++)query[i]=random_float(&seed);
  for(NSUInteger i=0;i<kvbytes/2;i++)keys_data[i]=(__fp16)random_float(&seed);
  for(uint32_t h=0;h<HEADS;h++)((float*)sinks.contents)[h]=0.05f*((int)(h%11)-5);
  uint16_t*mask_data=mask.contents;
  for(uint32_t r=0;r<ROWS;r++){
   uint32_t global=start+r,visible=0;
   for(uint32_t k=0;k<RAW_ROWS;k++){uint32_t pos=start-128+k;BOOL live=pos<=global&&global-pos<128;mask_data[(NSUInteger)r*keys+k]=live?0:0xfc00;visible+=live;}
   for(uint32_t k=0;k<comp;k++){BOOL live=k<(global+1)/128;mask_data[(NSUInteger)r*keys+RAW_ROWS+k]=live?0:0xfc00;visible+=live;}
   require(visible==128+(global+1)/128,"causal visibility count");
  }
  uint64_t block_counts[3]={0,0,0};prepare_and_verify(q,padp,blkp,kv,mask,pad,blocks,keys,block_counts);
  FlashArgs args={.ne01=ROWS,.ne02=HEADS,.ne03=1,.nb01=HEADS*WIDTH*4ull,.nb02=WIDTH*4,.nb03=qbytes,
   .ne11=keys,.ne_12_2=1,.ne_12_3=1,.ns10=WIDTH,.nb11=WIDTH*2,.nb12=kvbytes,.nb13=kvbytes,
   .ns20=WIDTH,.nb21=WIDTH*2,.nb22=kvbytes,.nb23=kvbytes,.ne31=ROWS,.ne32=1,.ne33=1,.nb31=keys*2ull,.nb32=maskbytes,.nb33=maskbytes,
   .ne1=HEADS,.ne2=ROWS,.ne3=1,.scale=1.0f/sqrtf((float)WIDTH),.max_bias=0,.m0=0,.m1=0,.n_head_log2=0,.logit_softcap=0};
  NSArray<id<MTLBuffer>>*inputs=@[qb,kv,kv,mask,sinks,pad,blocks];uint64_t input_hash=hash_inputs(inputs),checksum=0;
  printf("{\"event\":\"shape\",\"case\":%u,\"chunk_start\":%u,\"key_rows\":%u,\"padded\":%s,\"cpu_gpu_preparation_exact\":true,\"mask_blocks\":[%llu,%llu,%llu],\"input_fnv1a_u32\":%llu,\"output_words\":%llu}\n",shape,start,keys,padded?"true":"false",block_counts[0],block_counts[1],block_counts[2],input_hash,qbytes/4);fflush(stdout);
  for(int warm=0;warm<2;warm++){
   poison(a);poison(b);for(int half=0;half<2;half++){int role=warm%2?1-half:half;dispatch(q,p[padded][role],&args,inputs,role?b:a,role);}
   uint64_t h=verify(a,b);if(warm)require(h==checksum,"warm-up output changed");checksum=h;compared+=qbytes/4;
   printf("{\"event\":\"warmup\",\"case\":%u,\"repetition\":%d,\"order\":\"%s\",\"exact\":true,\"checked_words\":%llu,\"fnv1a_u32\":%llu}\n",shape,warm+1,warm%2?"BA":"AB",qbytes/4,h);fflush(stdout);
  }
  if(start==4096||start==16384||start==28672)for(int pair=0;pair<7;pair++){
   double times[2];poison(a);poison(b);for(int half=0;half<2;half++){int role=pair%2?1-half:half;times[role]=dispatch(q,p[padded][role],&args,inputs,role?b:a,role);}
   require(verify(a,b)==checksum,"timed output changed");compared+=qbytes/4;timed_pairs++;
   printf("{\"event\":\"pair\",\"case\":%u,\"pair\":%d,\"order\":\"%s\",\"baseline_gpu_ms\":%.9f,\"candidate_gpu_ms\":%.9f,\"ratio\":%.9f,\"exact\":true,\"checked_words\":%llu,\"fnv1a_u32\":%llu}\n",shape,pair+1,pair%2?"BA":"AB",times[0],times[1],times[1]/times[0],qbytes/4,checksum);fflush(stdout);
  }
  require(hash_inputs(inputs)==input_hash,"input buffers changed");
  printf("{\"event\":\"shape_complete\",\"case\":%u,\"inputs_unchanged\":true,\"guards_exact\":true}\n",shape);fflush(stdout);
 }}
 printf("{\"event\":\"complete\",\"passed\":true,\"cases\":7,\"timed_pairs\":%u,\"compared_fp32_words\":%llu}\n",timed_pairs,compared);return 0;
}}
