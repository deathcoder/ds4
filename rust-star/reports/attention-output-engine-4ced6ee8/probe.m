#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arguments.h"
typedef struct ds4_metal_args_mul_mm MatArgs;
typedef struct ds4_metal_args_mul_mm_id LowArgs;
typedef struct ds4_metal_args_mul_mm_id_map0 MapArgs;
typedef struct ds4_metal_args_dsv4_hc_expand HcArgs;
typedef struct { __fp16 d; int8_t qs[32]; } Q8Block;
enum { WIDTH=4096, GROUPS=8, RANK=1024, LOW=8192, HC=16384, HEADS=32768, GUARD=256 };
_Static_assert(sizeof(Q8Block)==34,"Q8 ABI");
_Static_assert(sizeof(MatArgs)==88&&sizeof(LowArgs)==104&&sizeof(MapArgs)==48&&sizeof(HcArgs)==152,"Metal argument ABI");
@interface Fixture:NSObject { @public
 uint32_t rows; NSUInteger tpe,workoff,workcap; MapArgs map; LowArgs low[2]; MatArgs out[2]; HcArgs hc;
 NSArray<id<MTLBuffer>> *inputs; NSArray<id<MTLBuffer>> *arena[2];
} @end
@implementation Fixture @end
static id<MTLComputePipelineState> mapper,converter,hcp,lowp[2][2],outp[2][2];
static void require(BOOL ok,const char*m){if(!ok){fprintf(stderr,"FAIL: %s\n",m);exit(2);}}
static uint32_t random_word(uint32_t*s){uint32_t x=*s;x^=x<<13;x^=x>>17;x^=x<<5;return *s=x;}
static float random_float(uint32_t*s){uint32_t u=0x3f000000u|(random_word(s)&0x7fffffu);float f;memcpy(&f,&u,4);return (f-0.75f)*2;}
static void emit(NSDictionary*r){NSError*e=nil;NSData*d=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingSortedKeys error:&e];require(d!=nil,"JSON serialization");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout);}
static void*data(id<MTLBuffer>b){return (uint8_t*)b.contents+GUARD;}
static NSUInteger bytes(id<MTLBuffer>b){return b.length-2*GUARD;}
static id<MTLBuffer>buffer(id<MTLDevice>d,NSUInteger n){id<MTLBuffer>b=[d newBufferWithLength:n+2*GUARD options:MTLResourceStorageModeShared];require(b!=nil,"allocation");memset(b.contents,0xa5,b.length);return b;}
static void guards(id<MTLBuffer>b){uint8_t*p=b.contents;for(NSUInteger i=0;i<GUARD;i++)require(p[i]==0xa5&&p[b.length-GUARD+i]==0xa5,"guard overwritten");}
static void poison(NSArray<id<MTLBuffer>>*ar){for(id<MTLBuffer>b in ar){memset(b.contents,0xa5,GUARD);memset(data(b),0xff,bytes(b));memset((uint8_t*)b.contents+b.length-GUARD,0xa5,GUARD);}}
static uint64_t hash_inputs(NSArray<id<MTLBuffer>>*ar){uint64_t h=0xcbf29ce484222325ull;for(id<MTLBuffer>b in ar){const uint32_t*p=b.contents;for(NSUInteger i=0;i<b.length/4;i++){h^=p[i];h*=0x100000001b3ull;}const uint8_t*t=b.contents;for(NSUInteger i=b.length/4*4;i<b.length;i++){h^=t[i];h*=0x100000001b3ull;}}return h;}
static id<MTLComputePipelineState>pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*n,MTLFunctionConstantValues*v){NSError*e=nil;id<MTLFunction>f=v?[l newFunctionWithName:n constantValues:v error:&e]:[l newFunctionWithName:n];id<MTLComputePipelineState>p=f?[d newComputePipelineStateWithFunction:f error:&e]:nil;if(!p){fprintf(stderr,"%s: %s\n",n.UTF8String,e.description.UTF8String);exit(3);}require(p.threadExecutionWidth==32,"SIMD width");return p;}
static void weights(id<MTLBuffer>b,uint32_t*seed){Q8Block*p=data(b);for(NSUInteger i=0;i<bytes(b)/34;i++){p[i].d=(__fp16)((1+random_word(seed)%16)/16384.0f);for(int k=0;k<32;k++)p[i].qs[k]=(int8_t)(random_word(seed)&255);}}
static Fixture*fixture(id<MTLDevice>d,uint32_t rows,uint32_t seed,BOOL precision){Fixture*f=[Fixture new];f->rows=rows;f->tpe=GROUPS*4;f->workoff=(f->tpe+(NSUInteger)GROUPS*rows*4+7)&~7ull;f->workcap=((NSUInteger)rows*GROUPS+31*GROUPS+31)/32;
 NSUInteger waBytes=GROUPS*RANK*(WIDTH/32)*34ull,wbBytes=WIDTH*(LOW/32)*34ull;
 id<MTLBuffer>wa=buffer(d,waBytes),wb=buffer(d,wbBytes),x=buffer(d,(NSUInteger)rows*HEADS*4),res=buffer(d,(NSUInteger)rows*HC*4),split=buffer(d,(NSUInteger)rows*24*4),ids=buffer(d,(NSUInteger)rows*GROUPS*4);
 weights(wa,&seed);weights(wb,&seed);
 float*q=data(x),*rr=data(res),*ss=data(split);int32_t*ii=data(ids);
 const float edge[]={0.0f,-0.0f,0x1p-24f,-0x1p-24f,0x1p-25f,-0x1p-25f,0x1p-14f,-0x1p-14f,1.00048828125f,1.00146484375f,-1.00048828125f,2.0009765625f,31.9921875f,-31.9921875f,0.3333333432674408f,-0.3333333432674408f};
 for(NSUInteger i=0;i<(NSUInteger)rows*HEADS;i++){float v=random_float(&seed);q[i]=precision&&i%4==0?edge[(i/4)%16]:v;}
 for(NSUInteger i=0;i<(NSUInteger)rows*HC;i++)rr[i]=random_float(&seed);
 for(uint32_t t=0;t<rows;t++){for(int i=0;i<24;i++)ss[t*24+i]=0.02f*random_float(&seed);for(int i=0;i<4;i++)ss[t*24+4+i]=0.7f+0.1f*random_float(&seed);for(int i=0;i<4;i++)ss[t*24+8+i*4+i]+=0.9f;for(int g=0;g<GROUPS;g++)ii[t*GROUPS+g]=g;}
 f->inputs=@[wa,wb,x,res,split,ids];
 for(int i=0;i<2;i++)f->arena[i]=@[buffer(d,f->workoff+8+f->workcap*8),buffer(d,(NSUInteger)rows*LOW*4),buffer(d,(NSUInteger)rows*WIDTH*4),buffer(d,(NSUInteger)rows*HC*4),buffer(d,(NSUInteger)rows*HEADS*2),buffer(d,(NSUInteger)rows*LOW*2)];
 f->map=(MapArgs){.ne02=8,.ne10=WIDTH,.ne11=8,.nb11=WIDTH*4,.nb12=HEADS*4,.ne21=rows,.ne20=8,.nb21=32};
 for(int role=0;role<2;role++){NSUInteger scalar=role?2:4;
 f->low[role]=(LowArgs){.ne00=WIDTH,.ne02=8,.nb01=(WIDTH/32)*34,.nb02=RANK*(WIDTH/32)*34ull,.nb03=waBytes,.ne11=8,.nb10=scalar,.nb11=WIDTH*scalar,.nb12=HEADS*scalar,.nb13=(NSUInteger)rows*HEADS*scalar,.ne20=8,.ne21=rows,.ne0=RANK,.ne1=8,.r2=1,.r3=1};
 f->out[role]=(MatArgs){.ne00=LOW,.ne02=1,.nb01=(LOW/32)*34,.nb02=wbBytes,.nb03=wbBytes,.ne12=1,.nb10=scalar,.nb11=LOW*scalar,.nb12=(NSUInteger)rows*LOW*scalar,.nb13=(NSUInteger)rows*LOW*scalar,.ne0=WIDTH,.ne1=rows,.r2=1,.r3=1};}
 f->hc=(HcArgs){.n_embd=WIDTH,.n_hc=4,.n_tokens=rows,.nb_block0=4,.nb_block1=WIDTH*4,.nb_add0=4,.nb_add1=WIDTH*4,.nb_res0=4,.nb_res1=WIDTH*4,.nb_res2=HC*4,.nb_post0=4,.nb_post1=24*4,.nb_comb0=4,.nb_comb1=4*4,.nb_comb2=24*4,.nb0=4,.nb1=WIDTH*4,.nb2=HC*4,.has_add=0};
 return f;
}
static void encode_stage(id<MTLComputeCommandEncoder>e,Fixture*f,int variant,int arena,int stage){NSArray*ar=f->arena[arena],*in=f->inputs;BOOL bounded=f->rows%32!=0;
 if(stage==0||stage==3){if(!variant)return;uint32_t n=f->rows*(stage==0?HEADS:LOW);[e setComputePipelineState:converter];[e setBytes:&n length:4 atIndex:0];[e setBuffer:stage==0?in[2]:ar[1] offset:GUARD atIndex:1];[e setBuffer:stage==0?ar[4]:ar[5] offset:GUARD atIndex:2];[e dispatchThreadgroups:MTLSizeMake((n+1023)/1024,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
 else if(stage==1){[e setComputePipelineState:mapper];[e setBytes:&f->map length:sizeof(MapArgs) atIndex:0];[e setBuffer:in[5] offset:GUARD atIndex:1];[e setBuffer:ar[0] offset:GUARD atIndex:2];[e setBuffer:ar[0] offset:GUARD+f->tpe atIndex:3];[e setBuffer:ar[0] offset:GUARD+f->workoff atIndex:4];[e setThreadgroupMemoryLength:128 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(8,1,1)];}
 else if(stage==2){[e setComputePipelineState:lowp[bounded][variant]];[e setBytes:&f->low[variant] length:sizeof(LowArgs) atIndex:0];[e setBuffer:in[0] offset:GUARD atIndex:1];[e setBuffer:variant?ar[4]:in[2] offset:GUARD atIndex:2];[e setBuffer:ar[0] offset:GUARD atIndex:3];[e setBuffer:ar[0] offset:GUARD+f->tpe atIndex:4];[e setBuffer:ar[1] offset:GUARD atIndex:5];[e setBuffer:ar[0] offset:GUARD+f->workoff atIndex:6];[e setThreadgroupMemoryLength:8192 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(f->workcap,RANK/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
 else if(stage==4){[e setComputePipelineState:outp[bounded][variant]];[e setBytes:&f->out[variant] length:sizeof(MatArgs) atIndex:0];[e setBuffer:in[1] offset:GUARD atIndex:1];[e setBuffer:variant?ar[5]:ar[1] offset:GUARD atIndex:2];[e setBuffer:ar[2] offset:GUARD atIndex:3];[e setThreadgroupMemoryLength:bounded?8192:6144 atIndex:0];[e dispatchThreadgroups:MTLSizeMake((f->rows+31)/32,WIDTH/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
 else if(stage==5){[e setComputePipelineState:hcp];[e setBytes:&f->hc length:sizeof(HcArgs) atIndex:0];[e setBuffer:ar[2] offset:GUARD atIndex:1];[e setBuffer:in[3] offset:GUARD atIndex:2];[e setBuffer:in[4] offset:GUARD+4*4 atIndex:3];[e setBuffer:in[4] offset:GUARD+8*4 atIndex:4];[e setBuffer:ar[2] offset:GUARD atIndex:5];[e setBuffer:ar[3] offset:GUARD atIndex:6];[e dispatchThreadgroups:MTLSizeMake((f->rows*WIDTH+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
}
static id<MTLCommandBuffer>encode(id<MTLCommandQueue>q,Fixture*f,int variant,int arena,int stage){id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];require(c&&e,"encoder");if(stage>=0)encode_stage(e,f,variant,arena,stage);else for(int i=0;i<6;i++)encode_stage(e,f,variant,arena,i);[e endEncoding];return c;}
static double interval(id<MTLCommandBuffer>c){require(c.status==MTLCommandBufferStatusCompleted,"command completed");double ms=(c.GPUEndTime-c.GPUStartTime)*1000;require(isfinite(ms)&&ms>0,"positive GPU interval");return ms;}
static double complete(id<MTLCommandBuffer>c){[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",c.error.description.UTF8String);exit(4);}return interval(c);}
static void mapcheck(Fixture*f,int idx){uint8_t*p=data(f->arena[idx][0]);uint32_t*tpe=(uint32_t*)p,*work=(uint32_t*)(p+f->workoff);int32_t*ids=(int32_t*)(p+f->tpe);NSUInteger wi=0;for(int g=0;g<8;g++){require(tpe[g]==f->rows,"map count");for(uint32_t row=0;row<f->rows;row++)require(ids[g*f->rows+row]==(int32_t)(row*8+g),"map id");for(uint32_t t=0;t<f->rows;t+=32){require(work[2+wi*2]==(uint32_t)g&&work[3+wi*2]==t,"map work item");wi++;}}require(work[0]==wi&&wi<=f->workcap,"map work count");}
static void halfcheck(Fixture*f,int idx){for(int which=0;which<2;which++){id<MTLBuffer>src=which?f->arena[idx][1]:f->inputs[2],dst=f->arena[idx][which?5:4];const float*x=data(src);const uint16_t*y=data(dst);for(NSUInteger i=0;i<bytes(src)/4;i++){__fp16 h=(__fp16)x[i];uint16_t bits;memcpy(&bits,&h,2);if(bits!=y[i]){fprintf(stderr,"conversion mismatch %d %lu %04x %04x\n",which,(unsigned long)i,bits,y[i]);exit(5);}}}}
static NSArray*verify(Fixture*f){NSMutableArray*hashes=[NSMutableArray new];for(int a=0;a<2;a++){for(id<MTLBuffer>b in f->arena[a])guards(b);mapcheck(f,a);}for(int part=1;part<=3;part++){const uint32_t*x=data(f->arena[0][part]),*y=data(f->arena[1][part]);NSUInteger n=bytes(f->arena[0][part])/4;uint64_t h=0xcbf29ce484222325ull,nz=0;BOOL varied=NO;for(NSUInteger i=0;i<n;i++){float v;memcpy(&v,x+i,4);if(x[i]!=y[i]||!isfinite(v)){fprintf(stderr,"output mismatch/nonfinite part=%d word=%lu A=%08x B=%08x\n",part,(unsigned long)i,x[i],y[i]);exit(6);}nz+=v!=0;varied|=x[i]!=x[0];h^=x[i];h*=0x100000001b3ull;}require(nz>n/2&&varied,"nondegenerate output");[hashes addObject:@(h)];}return hashes;}
static NSDictionary*slot(id<MTLCommandQueue>q,Fixture*f,int variant,int ar){int warm=4,samples=3;NSMutableArray<id<MTLCommandBuffer>>*cmd=[NSMutableArray new];double begin=NSProcessInfo.processInfo.systemUptime;for(int i=0;i<warm+samples;i++)[cmd addObject:encode(q,f,variant,ar,-1)];for(id<MTLCommandBuffer>c in cmd)[c commit];[cmd.lastObject waitUntilCompleted];double end=NSProcessInfo.processInfo.systemUptime,total=0;NSMutableArray*raw=[NSMutableArray new];for(int i=0;i<warm+samples;i++){id<MTLCommandBuffer>c=cmd[i];double ms=interval(c);if(i>=warm)total+=ms;if(i)require(c.GPUStartTime>=cmd[i-1].GPUEndTime,"queue intervals overlap");[raw addObject:@{@"conditioning":@(i<warm),@"start":@(c.GPUStartTime),@"end":@(c.GPUEndTime),@"gpu_ms":@(ms)}];}return @{@"candidate_variant":@(variant!=0),@"mean_gpu_ms":@(total/samples),@"host_begin":@(begin),@"host_end":@(end),@"commands":raw};}
int main(int argc,const char**argv){@autoreleasepool{if(argc!=4)return 1;BOOL edges=NO;BOOL aaOnly=strcmp(argv[2],"aa-only")==0;int session=atoi(argv[3]);require(session==0||session==1,"session");id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[d newCommandQueue];NSError*e=nil;NSString*s=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&e];id<MTLLibrary>l=s?[d newLibraryWithSource:s options:[MTLCompileOptions new] error:&e]:nil;if(!l){fprintf(stderr,"%s\n",e.description.UTF8String);return 1;}
 mapper=pipeline(d,l,@"kernel_mul_mm_id_map0_ne20_8",nil);converter=pipeline(d,l,@"kernel_cpy_contig_f32_f16_4",nil);hcp=pipeline(d,l,@"kernel_dsv4_hc_expand4",nil);
 for(int bounded=0;bounded<2;bounded++){bool no=false,bc=bounded!=0;MTLFunctionConstantValues*v=[MTLFunctionConstantValues new];[v setConstantValue:&no type:MTLDataTypeBool atIndex:700];[v setConstantValue:&bc type:MTLDataTypeBool atIndex:701];for(int role=0;role<2;role++){lowp[bounded][role]=pipeline(d,l,role?@"kernel_mul_mm_id_q8_0_f16":@"kernel_mul_mm_id_q8_0_f32",v);outp[bounded][role]=pipeline(d,l,role?@"probe_q8_output_half":@"kernel_mul_mm_q8_0_f32",v);require(lowp[bounded][role].maxTotalThreadsPerThreadgroup>=128&&outp[bounded][role].maxTotalThreadsPerThreadgroup>=128,"matmul geometry");}}
 emit(@{@"schema":@"rust-star-attention-output-same-arena-v1",@"device":d.name,@"mode":aaOnly?@"aa-only":@"timing",@"same_arena_roles":@YES,@"session":@(session),@"aa_uses_same_pipeline_objects":@YES,@"conditioning_dispatches":@4,@"samples_per_warmed_slot":@3,@"argument_sizes":@[@(sizeof(MapArgs)),@(sizeof(LowArgs)),@(sizeof(MatArgs)),@(sizeof(HcArgs))]});
 uint32_t erows[]={1,31,32,33,127,128,129,1024};int count=edges?16:3;uint64_t compared=0;int timed=0;
 for(int index=0;index<count;index++){@autoreleasepool{int shape=edges?index:(session?2-index:index);uint32_t rows=edges?erows[shape/2]:4096;BOOL precision=edges?shape%2:NO;uint32_t seed=0x2917a301u+shape;Fixture*f=fixture(d,rows,seed,precision);uint64_t words=(uint64_t)rows*(LOW+WIDTH+HC),inputHash=hash_inputs(f->inputs);NSArray*checksum=nil;
 emit(@{@"event":@"shape",@"case":@(shape),@"rows":@(rows),@"seed":@(seed),@"precision_pattern":@(precision),@"input_fnv1a_u32":@(inputHash),@"output_words":@(words),@"extra_half_scratch_bytes":@((uint64_t)rows*(HEADS+LOW)*2)});
 for(int warm=0;warm<2;warm++){poison(f->arena[0]);poison(f->arena[1]);for(int half=0;half<2;half++){int role=warm?1-half:half;complete(encode(q,f,role,role,-1));}NSArray*h=verify(f);halfcheck(f,1);if(checksum)require([h isEqual:checksum],"warmup checksum changed");checksum=h;compared+=words;emit(@{@"event":@"warmup",@"case":@(shape),@"repetition":@(warm+1),@"order":warm?@"BA":@"AB",@"exact":@YES,@"cpu_half_conversion_exact":@YES,@"checked_words":@(words),@"fnv1a_u32":h});}
 if(!edges){
 for(int pair=0;pair<12;pair++)for(int pass=0;pass<(aaOnly?1:2);pass++){BOOL candidate=!aaOnly&&(pair+session+pass)%2,reverse=pair%4==1||pair%4==2;int active=(pair/2)%2,reference=1-active;
 // Establish protected ORIGINAL reference outside timed slots. Never dispatch into it during either role.
 poison(f->arena[reference]);complete(encode(q,f,0,reference,-1));
 NSDictionary*timing[2];
 for(int half=0;half<2;half++){int role=reverse?1-half:half;poison(f->arena[active]);timing[role]=slot(q,f,candidate?role:0,active);require([verify(f)isEqual:checksum],"per-role protected reference mismatch");compared+=words;}
 timed++;double ta=[timing[0][@"mean_gpu_ms"]doubleValue],tb=[timing[1][@"mean_gpu_ms"]doubleValue];emit(@{@"event":@"pair",@"case":@(shape),@"pair":@(pair+1),@"protocol":candidate?@"candidate":@"aa-control",@"order":reverse?@"BA":@"AB",@"active_buffer":active?@"b":@"a",@"reference_buffer":reference?@"b":@"a",@"timing":@[timing[0],timing[1]],@"a_gpu_ms":@(ta),@"b_gpu_ms":@(tb),@"ratio":@(tb/ta),@"exact":@YES,@"per_role_verified":@YES,@"checked_words":@(2*words),@"fnv1a_u32":checksum});}}

 require(hash_inputs(f->inputs)==inputHash,"input buffers changed");emit(@{@"event":@"shape_complete",@"case":@(shape),@"inputs_unchanged":@YES,@"guards_exact":@YES,@"cpu_gpu_map_exact":@YES});}}
 emit(@{@"event":@"complete",@"passed":@YES,@"cases":@(count),@"timed_pairs":@(timed),@"compared_fp32_words":@(compared)});return 0;}}
