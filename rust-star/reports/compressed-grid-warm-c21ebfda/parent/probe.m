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
static id<MTLCommandBuffer> encode(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,FlashArgs *a,NSArray<id<MTLBuffer>>*in,id<MTLBuffer>out,BOOL transposed) {
 id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];require(c&&e,"attention encoder");
 [e setComputePipelineState:p];[e setBytes:a length:sizeof(*a) atIndex:0];
 for(NSUInteger i=0;i<7;i++)[e setBuffer:in[i] offset:0 atIndex:i+1];
 [e setBuffer:out offset:GUARD atIndex:8];[e setThreadgroupMemoryLength:28672 atIndex:0];
 [e dispatchThreadgroups:MTLSizeMake(ROWS/8,HEADS,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];
 [e endEncoding];return c;
}

static double interval(id<MTLCommandBuffer> c) {
 require(c.status==MTLCommandBufferStatusCompleted,"command completed");
 double ms=(c.GPUEndTime-c.GPUStartTime)*1000.0;
 require(isfinite(ms)&&ms>0,"GPU interval");return ms;
}
static double dispatch(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,FlashArgs *a,NSArray<id<MTLBuffer>>*in,id<MTLBuffer>out,BOOL ignored) {
 id<MTLCommandBuffer> c=encode(q,p,a,in,out,NO);completed(c);return interval(c);
}
// Submit all slot work before waiting. Each role receives the same original
// pipeline, own output buffer, four conditioning dispatches and three samples.
static NSDictionary *slot(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,FlashArgs *a,NSArray<id<MTLBuffer>>*in,id<MTLBuffer>out,BOOL warmed) {
 int warm=warmed?4:0, samples=warmed?3:1;
 NSMutableArray<id<MTLCommandBuffer>> *cmd=[NSMutableArray new];
 double hostBegin=NSProcessInfo.processInfo.systemUptime;
 for(int i=0;i<warm+samples;i++)[cmd addObject:encode(q,p,a,in,out,NO)];
 for(id<MTLCommandBuffer> c in cmd)[c commit];
 [cmd.lastObject waitUntilCompleted];
 double hostEnd=NSProcessInfo.processInfo.systemUptime,total=0;
 NSMutableArray *raw=[NSMutableArray new];
 for(int i=0;i<warm+samples;i++){
  id<MTLCommandBuffer> c=cmd[i];double ms=interval(c);if(i>=warm)total+=ms;
  if(i)require(c.GPUStartTime>=cmd[i-1].GPUEndTime,"queue intervals overlap");
  [raw addObject:@{@"conditioning":@(i<warm),@"start":@(c.GPUStartTime),@"end":@(c.GPUEndTime),@"gpu_ms":@(ms)}];
 }
 return @{@"mean_gpu_ms":@(total/samples),@"host_begin":@(hostBegin),@"host_end":@(hostEnd),@"commands":raw};
}
static void emit(NSDictionary *record) {
 NSError *err=nil;NSData *data=[NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:&err];
 require(data!=nil,"JSON encoding");fwrite(data.bytes,1,data.length,stdout);fputc('\n',stdout);fflush(stdout);
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
 require(argc==3,"usage: probe COMBINED.metal SESSION_0_OR_1");int session=atoi(argv[2]);require(session==0||session==1,"session index");NSError*error=nil;
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
 emit(@{@"schema":@"rust-star-compressed-aa-protocol-v1",@"session":@(session),@"device":d.name,@"rows":@(ROWS),@"heads":@(HEADS),@"width":@(WIDTH),@"pairs_per_protocol_per_shape":@12,@"conditioning_dispatches":@4,@"samples_per_warmed_slot":@3,@"same_pipeline_object":@YES});
 uint64_t compared=0;uint32_t timed_pairs=0;
 for(uint32_t shape_index=0;shape_index<3;shape_index++){@autoreleasepool{
  uint32_t shape=session?6-3*shape_index:3*shape_index;
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

  for(int pair=0;pair<12;pair++)for(int pass=0;pass<2;pass++){
   BOOL warmed=(pair+session+pass)%2, reverse=(pair%4==1||pair%4==2), swapped=(pair/2)%2;
   id<MTLBuffer> out[2]={swapped?b:a,swapped?a:b};
   poison(a);poison(b);NSDictionary *timing[2];
   for(int half=0;half<2;half++){int role=reverse?1-half:half;timing[role]=slot(q,p[padded][role],&args,inputs,out[role],warmed);}
   require(verify(a,b)==checksum,"timed output changed");compared+=qbytes/4;timed_pairs++;
   double ta=[timing[0][@"mean_gpu_ms"] doubleValue],tb=[timing[1][@"mean_gpu_ms"] doubleValue];
   emit(@{@"event":@"pair",@"case":@(shape),@"pair":@(pair+1),@"protocol":warmed?@"equal-warm":@"legacy-balanced",@"order":reverse?@"BA":@"AB",@"a_buffer":swapped?@"b":@"a",@"timing":@[timing[0],timing[1]],@"a_gpu_ms":@(ta),@"b_gpu_ms":@(tb),@"ratio":@(tb/ta),@"exact":@YES,@"checked_words":@(qbytes/4),@"fnv1a_u32":@(checksum)});
  }
  require(hash_inputs(inputs)==input_hash,"input buffers changed");
  printf("{\"event\":\"shape_complete\",\"case\":%u,\"inputs_unchanged\":true,\"guards_exact\":true}\n",shape);fflush(stdout);
 }}
 printf("{\"event\":\"complete\",\"passed\":true,\"cases\":3,\"timed_pairs\":%u,\"compared_fp32_words\":%llu}\n",timed_pairs,compared);return 0;
}}
