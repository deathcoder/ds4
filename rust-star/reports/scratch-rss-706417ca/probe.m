#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
typedef struct { uint64_t bytes,offset; uint32_t seed,kind; } Args;
@interface WeakBuffer : NSObject
@property(nonatomic,weak) id<MTLBuffer> buffer;
@property(nonatomic) NSUInteger bytes;
@property(nonatomic) NSUInteger tile;
@end
@implementation WeakBuffer
@end
static void emit(NSString *tag,NSDictionary *data) {
    NSError *error=nil;NSData *json=[NSJSONSerialization dataWithJSONObject:data options:NSJSONWritingSortedKeys error:&error];
    assert(json&&!error);printf("%s %.*s\n",tag.UTF8String,(int)json.length,(const char *)json.bytes);fflush(stdout);
}
static void wait_ok(id<MTLCommandBuffer> c) { assert(c);[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"COMMAND_FAILED %s\n",c.error.description.UTF8String);exit(2);} }
static uint32_t seed_for(NSUInteger tile,NSUInteger index) { return (uint32_t)(103u+tile*317u+index*19u); }
static void encode(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,id<MTLBuffer> buffer,Args a,id<MTLBuffer> errors) {
    [e setComputePipelineState:p];[e setBuffer:buffer offset:0 atIndex:0];[e setBytes:&a length:sizeof(a) atIndex:1];
    if(errors)[e setBuffer:errors offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake((a.bytes+15u)/16u,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static NSDictionary *snapshot(id<MTLDevice> device,NSArray<WeakBuffer *> *ledger,NSString *phase,NSUInteger tile,NSUInteger cpuBytes) {
    @autoreleasepool {
        struct task_vm_info vm={0};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
        kern_return_t result=task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&vm,&count);assert(result==KERN_SUCCESS);
        struct rusage usage={0};assert(getrusage(RUSAGE_SELF,&usage)==0);
        NSUInteger live=0,bytes=0,old=0;
        for(WeakBuffer *record in ledger){id<MTLBuffer> b=record.buffer;if(b){live++;bytes+=record.bytes;if(record.tile<tile)old+=record.bytes;}}
        NSDictionary *row=@{@"phase":phase,@"tile":@(tile),@"cpu_clear_bytes":@(cpuBytes),@"rss":@(vm.resident_size),@"footprint":@(vm.phys_footprint),@"rusage_peak_rss":@(usage.ru_maxrss),@"device_allocated":@(device.currentAllocatedSize),@"live_count":@(live),@"live_bytes":@(bytes),@"prior_generation_bytes":@(old),@"vm_result":@(result)};
        emit(@"SNAPSHOT",row);return row;
    }
}
int main(int argc,const char **argv) { @autoreleasepool {
    assert(argc==4);NSString *mode=[NSString stringWithUTF8String:argv[1]];NSError *error=nil;
    NSArray *inventory=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]] options:0 error:&error];assert(inventory.count==105&&!error);
    NSDictionary *copies=@{@"q_cur_buffer":@"q_raw_buffer",@"attention_back_buffer":@"attention_output_buffer",@"next_q_cur_buffer":@"next_q_raw_buffer",@"next_attention_back_buffer":@"next_attention_output_buffer"};
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();assert(device);id<MTLCommandQueue> queue=[device newCommandQueue];assert(queue);
    NSString *source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[3]] encoding:NSUTF8StringEncoding error:&error];assert(source&&!error);
    id<MTLLibrary> library=[device newLibraryWithSource:source options:nil error:&error];if(!library){fprintf(stderr,"SHADER_FAILED %s\n",error.description.UTF8String);return 2;}
    id<MTLComputePipelineState> writer=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"write_pattern"] error:&error];assert(writer&&!error);
    id<MTLComputePipelineState> verifier=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"verify_pattern"] error:&error];assert(verifier&&!error);
    id<MTLBuffer> errors=[device newBufferWithLength:4096 options:MTLResourceStorageModeShared];assert(errors);memset(errors.contents,0,4096);
    emit(@"HEADER",@{@"mode":mode,@"device":device.name,@"page_size":@(getpagesize()),@"args_bytes":@(sizeof(Args))});
    if([mode isEqual:@"proof"]) {
        NSUInteger checked=0,guards=0,negative=0,cases=0;
        for(NSNumber *size in @[@4,@4096,@134217728]) { @autoreleasepool {
            NSUInteger n=size.unsignedLongLongValue;
            for(NSUInteger name=0;name<4;name++) { @autoreleasepool {
                id<MTLBuffer> src=[device newBufferWithLength:n options:MTLResourceStorageModeShared];
                id<MTLBuffer> zeroed=[device newBufferWithLength:n+512 options:MTLResourceStorageModeShared];
                id<MTLBuffer> poisoned=[device newBufferWithLength:n+512 options:MTLResourceStorageModeShared];assert(src&&zeroed&&poisoned);
                for(NSUInteger turn=0;turn<2;turn++) { @autoreleasepool {
                    uint32_t seed=seed_for(turn,name);uint32_t poison=turn?0x5a:0xa5;
                    memset(zeroed.contents,0,n+512);memset(poisoned.contents,(int)poison,n+512);
                    id<MTLCommandBuffer> c=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[c computeCommandEncoder];assert(e);encode(e,writer,src,(Args){n,0,seed,0},nil);[e endEncoding];
                    id<MTLBlitCommandEncoder> blit=[c blitCommandEncoder];assert(blit);[blit copyFromBuffer:src sourceOffset:0 toBuffer:zeroed destinationOffset:256 size:n];[blit copyFromBuffer:src sourceOffset:0 toBuffer:poisoned destinationOffset:256 size:n];[blit endEncoding];
                    e=[c computeCommandEncoder];assert(e);
                    for(NSUInteger role=0;role<2;role++){id<MTLBuffer> dst=role?poisoned:zeroed;encode(e,verifier,dst,(Args){n,256,seed,0},errors);encode(e,verifier,dst,(Args){256,0,role?poison:0,1},errors);encode(e,verifier,dst,(Args){256,n+256,role?poison:0,1},errors);}
                    [e endEncoding];[c commit];wait_ok(c);assert(*(uint32_t *)errors.contents==0);checked+=2*n;guards+=1024;cases++;
                }}
                memset(poisoned.contents,0xa5,n+512);*(uint32_t *)errors.contents=0;
                id<MTLCommandBuffer> c=[queue commandBuffer];id<MTLBlitCommandEncoder> blit=[c blitCommandEncoder];assert(blit);if(n>4)[blit copyFromBuffer:src sourceOffset:0 toBuffer:poisoned destinationOffset:256 size:n-4];[blit endEncoding];
                id<MTLComputeCommandEncoder> e=[c computeCommandEncoder];assert(e);encode(e,verifier,poisoned,(Args){n,256,seed_for(1,name),0},errors);[e endEncoding];[c commit];wait_ok(c);assert(*(uint32_t *)errors.contents>0);negative++;*(uint32_t *)errors.contents=0;
            }}
        }}
        emit(@"RESULT",@{@"passed":@YES,@"cases":@(cases),@"checked_bytes":@(checked),@"guard_bytes":@(guards),@"negative_controls":@(negative)});return 0;
    }
    assert(([@[@"A",@"B",@"C",@"D",@"E"] containsObject:mode]));BOOL reuse=!([mode isEqual:@"A"]||[mode isEqual:@"B"]);BOOL clear=[mode isEqual:@"B"]||[mode isEqual:@"D"]||[mode isEqual:@"E"];
    NSMutableDictionary<NSString *,id<MTLBuffer>> *buffers=nil;NSMutableArray<WeakBuffer *> *ledger=[NSMutableArray new];NSMutableDictionary *identity=[NSMutableDictionary new];NSMutableDictionary *indexes=[NSMutableDictionary new];NSUInteger total=0;
    for(NSUInteger i=0;i<inventory.count;i++){indexes[inventory[i][@"label"]]=@(i);total+=[inventory[i][@"bytes"] unsignedLongLongValue];}assert(total==2493173233ull);
    snapshot(device,ledger,@"baseline",0,0);NSUInteger checked=0;
    for(NSUInteger tile=0;tile<4;tile++){ @autoreleasepool {
        snapshot(device,ledger,@"entry",tile,0);
        if(!buffers)buffers=[NSMutableDictionary new];NSMutableSet *addresses=[NSMutableSet new];
        for(NSDictionary *item in inventory){NSString *label=item[@"label"];NSUInteger n=[item[@"bytes"] unsignedLongLongValue];
            if(!buffers[label]){id<MTLBuffer> b=[device newBufferWithLength:n options:MTLResourceStorageModeShared];assert(b&&b.length==n);buffers[label]=b;WeakBuffer *record=[WeakBuffer new];record.buffer=b;record.bytes=n;record.tile=tile;[ledger addObject:record];}
            NSNumber *address=@((uintptr_t)(__bridge void *)buffers[label]);assert(![addresses containsObject:address]);[addresses addObject:address];
            if(reuse&&tile)assert([identity[label] isEqual:address]);else identity[label]=address;
        }
        assert(buffers.count==105&&addresses.count==105);emit(@"IDENTITIES",@{@"tile":@(tile),@"ids":[identity copy]});snapshot(device,ledger,@"allocated",tile,0);
        NSUInteger cleared=0;
        if(clear)for(NSDictionary *item in inventory){NSString *label=item[@"label"];if([mode isEqual:@"E"]&&copies[label])continue;NSUInteger n=[item[@"bytes"] unsignedLongLongValue];memset(buffers[label].contents,0,n);cleared+=n;}
        snapshot(device,ledger,@"initialized",tile,cleared);
        id<MTLCommandBuffer> c=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[c computeCommandEncoder];assert(e);
        for(NSDictionary *item in inventory){NSString *label=item[@"label"];if(copies[label])continue;NSUInteger n=[item[@"bytes"] unsignedLongLongValue];encode(e,writer,buffers[label],(Args){n,0,seed_for(tile,[indexes[label] unsignedLongLongValue]),0},nil);}[e endEncoding];
        id<MTLBlitCommandEncoder> blit=[c blitCommandEncoder];assert(blit);
        for(NSString *dst in copies){NSString *src=copies[dst];assert(buffers[src].length==buffers[dst].length);[blit copyFromBuffer:buffers[src] sourceOffset:0 toBuffer:buffers[dst] destinationOffset:0 size:buffers[dst].length];}[blit endEncoding];
        e=[c computeCommandEncoder];assert(e);
        for(NSDictionary *item in inventory){NSString *label=item[@"label"];NSString *sourceLabel=copies[label]?:label;NSUInteger n=[item[@"bytes"] unsignedLongLongValue];encode(e,verifier,buffers[label],(Args){n,0,seed_for(tile,[indexes[sourceLabel] unsignedLongLongValue]),0},errors);}
        [e endEncoding];[c commit];wait_ok(c);assert(*(uint32_t *)errors.contents==0);checked+=total;snapshot(device,ledger,@"gpu_completed",tile,cleared);
        if(!reuse)buffers=nil;
    }snapshot(device,ledger,@"after_tile",tile,0);}
    buffers=nil;snapshot(device,ledger,@"released",4,0);
    emit(@"RESULT",@{@"passed":@YES,@"mode":mode,@"tiles":@4,@"buffers":@105,@"logical_bytes":@(total),@"checked_bytes":@(checked),@"copy_destination_bytes":@536870912,@"pool_identities_stable":@(reuse)});return 0;
}}
