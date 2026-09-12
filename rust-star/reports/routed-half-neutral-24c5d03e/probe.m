#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arguments.h"
typedef struct ds4_metal_args_mul_mm_id_map0 MapArgs;
typedef struct ds4_metal_args_mul_mm_id GateArgs;
typedef struct ds4_metal_dsv4_moe_swiglu_weight_args ActArgs;
typedef struct { __fp16 d; uint16_t qs[32]; } IQ2Block;
enum { WIDTH=4096, EXPERTS=256, USED=6, MID=2048, GUARD=256 };
_Static_assert(sizeof(IQ2Block)==66,"IQ2 block ABI");
_Static_assert(sizeof(MapArgs)==48 && sizeof(GateArgs)==104 && sizeof(ActArgs)==48,"Metal argument ABI");
@interface Fixture:NSObject { @public
 uint32_t rows; NSUInteger tpe, workoff, workcap; MapArgs map; GateArgs gate[2]; ActArgs act;
 NSArray<id<MTLBuffer>> *inputs; NSArray<id<MTLBuffer>> *arena[2];
} @end
@implementation Fixture @end
static id<MTLComputePipelineState> mapper, converter, gatep[2];
static void require(BOOL ok,const char *message) { if(!ok){fprintf(stderr,"FAIL: %s\n",message);exit(2);} }
static uint32_t random_word(uint32_t *s) { uint32_t x=*s;x^=x<<13;x^=x>>17;x^=x<<5;return *s=x; }
static float random_float(uint32_t *s) { uint32_t u=0x3f000000u|(random_word(s)&0x7fffffu);float f;memcpy(&f,&u,4);return (f-0.75f)*2; }
static void emit(NSDictionary *r) { NSError *e=nil;NSData *d=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingSortedKeys error:&e];require(d!=nil,"JSON");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout); }
static void *data(id<MTLBuffer> b) { return (uint8_t*)b.contents+GUARD; }
static NSUInteger bytes(id<MTLBuffer> b) { return b.length-2*GUARD; }
static id<MTLBuffer> buffer(id<MTLDevice> d,NSUInteger n) { id<MTLBuffer>b=[d newBufferWithLength:n+2*GUARD options:MTLResourceStorageModeShared];require(b!=nil,"allocation");memset(b.contents,0xa5,b.length);return b; }
static void guards(id<MTLBuffer> b) { const uint8_t *p=b.contents;for(int i=0;i<GUARD;i++)require(p[i]==0xa5 && p[b.length-GUARD+i]==0xa5,"buffer guards"); }
static void poison(NSArray<id<MTLBuffer>> *ar,uint8_t byte) { for(id<MTLBuffer>b in ar)memset(data(b),byte,bytes(b)); }
static uint64_t hash_inputs(NSArray<id<MTLBuffer>> *ar) { uint64_t h=0xcbf29ce484222325ull;for(id<MTLBuffer>b in ar){const uint32_t*p=b.contents;for(NSUInteger i=0;i<b.length/4;i++){h^=p[i];h*=0x100000001b3ull;}const uint8_t*t=b.contents;for(NSUInteger i=b.length/4*4;i<b.length;i++){h^=t[i];h*=0x100000001b3ull;}}return h; }
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*n) { NSError*e=nil;id<MTLFunction>f=[l newFunctionWithName:n];require(f!=nil,n.UTF8String);id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];if(!p)fprintf(stderr,"%s\n",e.description.UTF8String);require(p!=nil,"pipeline");return p; }
static Fixture *fixture(id<MTLDevice>d,uint32_t rows,uint32_t seed,int pattern) {
 Fixture*f=[Fixture new];f->rows=rows;f->tpe=EXPERTS*4;f->workoff=(f->tpe+(NSUInteger)EXPERTS*rows*4+7)&~7ull;f->workcap=((NSUInteger)rows*USED+31*EXPERTS+31)/32;
 f->map=(MapArgs){.ne02=EXPERTS,.ne10=WIDTH,.ne11=1,.nb11=WIDTH*4,.nb12=WIDTH*4,.ne21=rows,.ne20=USED,.nb21=USED*4};
 f->gate[0]=(GateArgs){.ne00=WIDTH,.ne02=EXPERTS,.nb01=1056,.nb02=MID*1056ull,.nb03=EXPERTS*MID*1056ull,.ne11=1,.nb10=4,.nb11=WIDTH*4,.nb12=WIDTH*4,.nb13=(uint64_t)rows*WIDTH*4,.ne20=USED,.ne21=rows,.ne0=MID,.ne1=USED,.r2=1,.r3=1,.tp_rank=0,.tp_world=0,.tp_expert_base=0};
 f->gate[1]=f->gate[0];f->gate[1].nb10/=2;f->gate[1].nb11/=2;f->gate[1].nb12/=2;f->gate[1].nb13/=2;
 f->act=(ActArgs){.width=MID,.rows=rows*USED,.gate_row_stride=MID*4,.up_row_stride=MID*4,.mid_row_stride=MID*2,.weight_stride=4,.write_clamped=0,.clamp_value=10.0f};
 f->inputs=@[buffer(d,EXPERTS*MID*1056ull),buffer(d,EXPERTS*MID*1056ull),buffer(d,(NSUInteger)rows*WIDTH*4),buffer(d,(NSUInteger)rows*USED*4),buffer(d,(NSUInteger)rows*USED*4)];
 for(int which=0;which<2;which++){IQ2Block*p=data(f->inputs[which]);for(NSUInteger b=0;b<bytes(f->inputs[which])/sizeof(*p);b++){p[b].d=(__fp16)(0x1p-9f+(random_word(&seed)&15u)*0x1p-13f);for(int j=0;j<32;j++)p[b].qs[j]=(uint16_t)random_word(&seed);}}
 float*x=data(f->inputs[2]);
 for(NSUInteger i=0;i<(NSUInteger)rows*WIDTH;i++){
  x[i]=random_float(&seed);
  if(pattern==2 || (pattern==1 && i%2==0)){
   uint32_t u=0x3e000000u|(random_word(&seed)&0x807fe000u)|0x1000u;int delta=(int)(i%3)-1;u=(uint32_t)((int64_t)u+delta);memcpy(x+i,&u,4);
   if(i%97==0){const float edges[]={0.0f,-0.0f,0x1p-25f,-0x1p-25f,0x1p-24f,-0x1p-24f,0x1.000002p-25f,0x1.fffffep-26f};x[i]=edges[(i/97)%8];}
  }
 }
 int32_t*ids=data(f->inputs[3]);float*weights=data(f->inputs[4]);uint32_t offset=seed%EXPERTS;
 for(uint32_t r=0;r<rows;r++)for(int k=0;k<USED;k++){ids[r*USED+k]=pattern==1?(r*7+k*5+offset)%32:(r*37+k*31+offset)%EXPERTS;weights[r*USED+k]=0.0625f+fabsf(random_float(&seed))*0.25f;}
 for(int a=0;a<2;a++)f->arena[a]=@[buffer(d,f->workoff+8+f->workcap*8),buffer(d,(NSUInteger)rows*WIDTH*2),buffer(d,(NSUInteger)rows*USED*MID*2)];
 return f;
}
static void encode_chain(id<MTLComputeCommandEncoder>e,Fixture*f,int variant,int ar) {
 NSArray*in=f->inputs;NSArray*tmp=f->arena[ar];
 [e setComputePipelineState:mapper];[e setBytes:&f->map length:sizeof(f->map) atIndex:0];[e setBuffer:in[3] offset:GUARD atIndex:1];[e setBuffer:tmp[0] offset:GUARD atIndex:2];[e setBuffer:tmp[0] offset:GUARD+f->tpe atIndex:3];[e setBuffer:tmp[0] offset:GUARD+f->workoff atIndex:4];[e setThreadgroupMemoryLength:EXPERTS*USED*2 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 if(variant){uint32_t n=f->rows*WIDTH;[e setComputePipelineState:converter];[e setBytes:&n length:sizeof(n) atIndex:0];[e setBuffer:in[2] offset:GUARD atIndex:1];[e setBuffer:tmp[1] offset:GUARD atIndex:2];[e dispatchThreadgroups:MTLSizeMake((n+1023)/1024,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
 [e setComputePipelineState:gatep[variant]];[e setBytes:&f->gate[variant] length:sizeof(GateArgs) atIndex:0];[e setBytes:&f->act length:sizeof(ActArgs) atIndex:1];[e setBuffer:in[0] offset:GUARD atIndex:2];[e setBuffer:in[1] offset:GUARD atIndex:3];[e setBuffer:variant?tmp[1]:in[2] offset:GUARD atIndex:4];[e setBuffer:tmp[0] offset:GUARD atIndex:5];[e setBuffer:tmp[0] offset:GUARD+f->tpe atIndex:6];[e setBuffer:tmp[2] offset:GUARD atIndex:7];[e setBuffer:in[4] offset:GUARD atIndex:8];[e setBuffer:tmp[0] offset:GUARD+f->workoff atIndex:9];[e setThreadgroupMemoryLength:16384 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(32,f->workcap,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
static id<MTLCommandBuffer>encode(id<MTLCommandQueue>q,Fixture*f,int variant,int ar) { id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];require(c&&e,"encoder");encode_chain(e,f,variant,ar);[e endEncoding];return c; }
static double interval(id<MTLCommandBuffer>c) { if(c.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",c.error.description.UTF8String);require(c.status==MTLCommandBufferStatusCompleted,"command completed");double ms=(c.GPUEndTime-c.GPUStartTime)*1000;require(isfinite(ms)&&ms>0,"positive GPU interval");return ms; }
static double complete(id<MTLCommandBuffer>c) { [c commit];[c waitUntilCompleted];return interval(c); }
static void mapcheck(Fixture*f,int a) {
 uint8_t*p=data(f->arena[a][0]);uint32_t*count=(uint32_t*)p,*work=(uint32_t*)(p+f->workoff);int32_t*ids=(int32_t*)(p+f->tpe),*source=data(f->inputs[3]);NSUInteger wi=0;
 for(int expert=0;expert<EXPERTS;expert++){uint32_t n=0;for(uint32_t row=0;row<f->rows;row++)for(int slot=0;slot<USED;slot++)if(source[row*USED+slot]==expert){require(ids[expert*f->rows+n]==(int32_t)(row*USED+slot),"map ID");n++;}require(count[expert]==n,"map count");for(uint32_t t=0;t<n;t+=32){require(work[2+wi*2]==(uint32_t)expert && work[3+wi*2]==t,"map work item");wi++;}}
 require(work[0]==wi && wi<=f->workcap,"map work count");
}
static void halfcheck(Fixture*f,int a) { const float*x=data(f->inputs[2]);const uint16_t*y=data(f->arena[a][1]);for(NSUInteger i=0;i<(NSUInteger)f->rows*WIDTH;i++){__fp16 h=(__fp16)x[i];uint16_t bits;memcpy(&bits,&h,2);if(bits!=y[i]){fprintf(stderr,"half mismatch word %lu: %04x %04x\n",(unsigned long)i,bits,y[i]);exit(5);}} }
static uint64_t verify(Fixture*f) {
 for(int a=0;a<2;a++){for(id<MTLBuffer>b in f->arena[a])guards(b);mapcheck(f,a);}
 const uint16_t*x=data(f->arena[0][2]),*y=data(f->arena[1][2]);const __fp16*v=data(f->arena[0][2]);NSUInteger n=(NSUInteger)f->rows*USED*MID,nz=0;BOOL varied=NO;uint64_t h=0xcbf29ce484222325ull;
 for(NSUInteger i=0;i<n;i++){if(x[i]!=y[i] || !isfinite((float)v[i])){fprintf(stderr,"output mismatch/nonfinite word %lu: %04x %04x\n",(unsigned long)i,x[i],y[i]);exit(6);}nz+=v[i]!=0;varied|=x[i]!=x[0];h^=x[i];h*=0x100000001b3ull;}
 require(nz>n/4 && varied,"nondegenerate output");return h;
}
static NSDictionary*slot(id<MTLCommandQueue>q,Fixture*f,int variant,int ar) {
 int warm=4,samples=3;NSMutableArray<id<MTLCommandBuffer>>*cmd=[NSMutableArray new];double begin=NSProcessInfo.processInfo.systemUptime;
 for(int i=0;i<warm+samples;i++)[cmd addObject:encode(q,f,variant,ar)];for(id<MTLCommandBuffer>c in cmd)[c commit];[cmd.lastObject waitUntilCompleted];double end=NSProcessInfo.processInfo.systemUptime,total=0;NSMutableArray*raw=[NSMutableArray new];
 for(int i=0;i<warm+samples;i++){id<MTLCommandBuffer>c=cmd[i];double ms=interval(c);if(i>=warm)total+=ms;if(i)require(c.GPUStartTime>=cmd[i-1].GPUEndTime,"queue intervals overlap");[raw addObject:@{@"conditioning":@(i<warm),@"start":@(c.GPUStartTime),@"end":@(c.GPUEndTime),@"gpu_ms":@(ms)}];}
 return @{@"candidate_variant":@(variant!=0),@"mean_gpu_ms":@(total/samples),@"host_begin":@(begin),@"host_end":@(end),@"commands":raw};
}
int main(int argc,const char**argv) { @autoreleasepool {
 require(argc==4,"usage: probe combined.metal edges|aa-only|timing session");BOOL edges=strcmp(argv[2],"edges")==0,aa=strcmp(argv[2],"aa-only")==0;require(edges||aa||strcmp(argv[2],"timing")==0,"mode");int session=atoi(argv[3]);require(session==0||session==1,"session");
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[d newCommandQueue];NSError*e=nil;NSString*s=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&e];id<MTLLibrary>l=s?[d newLibraryWithSource:s options:[MTLCompileOptions new] error:&e]:nil;if(!l){fprintf(stderr,"%s\n",e.description.UTF8String);return 1;}
 mapper=pipeline(d,l,@"kernel_mul_mm_id_map0_ne20_6");converter=pipeline(d,l,@"kernel_cpy_contig_f32_f16_4");gatep[0]=pipeline(d,l,@"rust_star_routed_prefill_gate_up");gatep[1]=pipeline(d,l,@"probe_routed_half_gate_up");require(mapper.maxTotalThreadsPerThreadgroup>=256 && converter.maxTotalThreadsPerThreadgroup>=256 && gatep[0].maxTotalThreadsPerThreadgroup>=128 && gatep[1].maxTotalThreadsPerThreadgroup>=128,"pipeline geometry");
 emit(@{@"schema":@"rust-star-routed-half-same-arena-v1",@"device":d.name,@"mode":edges?@"edges":aa?@"aa-only":@"timing",@"session":@(session),@"same_arena_roles":@YES,@"aa_uses_same_pipeline_objects":@YES,@"conditioning_dispatches":@4,@"samples_per_warmed_slot":@3,@"argument_sizes":@[@(sizeof(MapArgs)),@(sizeof(GateArgs)),@(sizeof(ActArgs))]});
 uint32_t erows[]={1,31,32,33,63,64,65,1024};int cases=edges?16:3,timed=0;uint64_t compared=0;
 for(int index=0;index<cases;index++) { @autoreleasepool {
  int shape=edges?index:session?2-index:index,pattern=edges?shape%2:shape;uint32_t rows=edges?erows[shape/2]:4096,seed=0x5417311u+shape;Fixture*f=fixture(d,rows,seed,pattern);uint64_t words=(uint64_t)rows*USED*MID,inputHash=hash_inputs(f->inputs),checksum=0;
  emit(@{@"event":@"shape",@"case":@(shape),@"rows":@(rows),@"seed":@(seed),@"pattern":@(pattern),@"input_fnv1a_u32":@(inputHash),@"output_fp16_words":@(words),@"half_scratch_bytes":@((uint64_t)rows*WIDTH*2)});
  for(int warm=0;warm<2;warm++){poison(f->arena[0],0x7e);poison(f->arena[1],0x7f);for(int half=0;half<2;half++){int role=warm?1-half:half;complete(encode(q,f,role,role));}uint64_t h=verify(f);halfcheck(f,1);if(warm)require(h==checksum,"warmup checksum changed");checksum=h;compared+=words;emit(@{@"event":@"warmup",@"case":@(shape),@"repetition":@(warm+1),@"order":warm?@"BA":@"AB",@"exact":@YES,@"cpu_half_conversion_exact":@YES,@"checked_fp16_words":@(words),@"fnv1a_u16":@(h)});}
  if(!edges)for(int pair=0;pair<12;pair++)for(int pass=0;pass<(aa?1:2);pass++){
   BOOL candidate=!aa&&(pair+session+pass)%2,reverse=pair%4==1||pair%4==2;int active=(pair/2)%2,reference=1-active;poison(f->arena[reference],0x7e);complete(encode(q,f,0,reference));NSDictionary*timing[2];
   for(int half=0;half<2;half++){int role=reverse?1-half:half;poison(f->arena[active],0x7f);timing[role]=slot(q,f,candidate?role:0,active);require(verify(f)==checksum,"per-role protected reference mismatch");compared+=words;}
   timed++;double ta=[timing[0][@"mean_gpu_ms"]doubleValue],tb=[timing[1][@"mean_gpu_ms"]doubleValue];emit(@{@"event":@"pair",@"case":@(shape),@"pair":@(pair+1),@"protocol":candidate?@"candidate":@"aa-control",@"order":reverse?@"BA":@"AB",@"active_buffer":active?@"b":@"a",@"reference_buffer":reference?@"b":@"a",@"timing":@[timing[0],timing[1]],@"a_gpu_ms":@(ta),@"b_gpu_ms":@(tb),@"ratio":@(tb/ta),@"exact":@YES,@"per_role_verified":@YES,@"checked_fp16_words":@(2*words),@"fnv1a_u16":@(checksum)});
  }
  require(hash_inputs(f->inputs)==inputHash,"input buffers changed");for(id<MTLBuffer>b in f->inputs)guards(b);emit(@{@"event":@"shape_complete",@"case":@(shape),@"inputs_unchanged":@YES,@"guards_exact":@YES,@"cpu_gpu_map_exact":@YES});
 }}
 emit(@{@"event":@"complete",@"passed":@YES,@"cases":@(cases),@"timed_pairs":@(timed),@"compared_fp16_words":@(compared)});return 0;
}}
