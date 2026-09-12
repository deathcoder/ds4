#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
static int fail_with_message(char *e, size_t n, NSString *s) { if(e&&n) snprintf(e,n,"%s",s.UTF8String); return 0; }
static int command_succeeded(id<MTLCommandBuffer> c,char *e,size_t n) { [c waitUntilCompleted]; return c.status==MTLCommandBufferStatusCompleted ? 1:fail_with_message(e,n,@"command failed"); }
// One initialized scratch set, owned only during four native bootstrap tiles.
@interface RustStarBootstrapScratch : NSObject
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *buffers;
@property(nonatomic, strong) NSMutableSet<NSString *> *checkedOut;
@property(nonatomic, strong) id<MTLCommandBuffer> priorCommand;
@property(nonatomic) NSUInteger bytes;
@property(nonatomic) uint32_t frontier;
@property(nonatomic) uint32_t nextPosition;
@end
@implementation RustStarBootstrapScratch
@end

// newBufferWithLength returns zeroed storage. Reuse restores that entire
// initial range before the caller performs its existing CPU initialization.
// The caller checks prior GPU completion once, before any checkout in a tile.
static id<MTLBuffer> bootstrap_scratch_buffer(
    id<MTLDevice> device, RustStarBootstrapScratch *pool,
    NSString *label, NSUInteger bytes, const void *source,
    char *error, size_t error_bytes)
{
    if (!pool) {
        return source ? [device newBufferWithBytes:source length:bytes
            options:MTLResourceStorageModeShared] :
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    NSArray<NSString *> *allowed = @[
        @"after_attention_hc_buffer",
        @"after_ffn_hc_buffer",
        @"attention_back_buffer",
        @"attention_block_buffer",
        @"attention_group_ids_buffer",
        @"attention_group_map_buffer",
        @"attention_low_buffer",
        @"attention_mask_buffer",
        @"attention_out_buffer",
        @"attention_output_buffer",
        @"attention_pad_buffer",
        @"attn_buffer",
        @"collapsed_buffer",
        @"compressed_position_buffer",
        @"embedding_buffer",
        @"ffn_cur_buffer",
        @"ffn_flat_hc_buffer",
        @"ffn_mix_buffer",
        @"ffn_norm_buffer",
        @"ffn_split_buffer",
        @"flat_hc_buffer",
        @"full_kv_half_buffer",
        @"hc_buffer",
        @"kv_half_buffer",
        @"kv_norm_buffer",
        @"kv_norm_snapshot_buffer",
        @"kv_raw_buffer",
        @"kv_rope_buffer",
        @"layer2_attn_packed_kv_buffer",
        @"layer2_attn_packed_score_buffer",
        @"layer2_attn_projected_kv_buffer",
        @"layer2_attn_projected_score_buffer",
        @"layer2_attn_softmax_buffer",
        @"layer2_cur_buffer",
        @"layer2_flat_hc_buffer",
        @"layer2_indexer_packed_kv_buffer",
        @"layer2_indexer_packed_score_buffer",
        @"layer2_indexer_projected_kv_buffer",
        @"layer2_indexer_projected_score_buffer",
        @"layer2_indexer_softmax_buffer",
        @"layer2_kv_norm_buffer",
        @"layer2_kv_norm_snapshot_buffer",
        @"layer2_kv_raw_buffer",
        @"layer2_kv_rope_buffer",
        @"layer2_mix_buffer",
        @"layer2_norm_buffer",
        @"layer2_q_lora_buffer",
        @"layer2_q_norm_buffer",
        @"layer2_split_buffer",
        @"mix_buffer",
        @"next_after_attention_hc_buffer",
        @"next_after_ffn_hc_buffer",
        @"next_attention_back_buffer",
        @"next_attention_low_buffer",
        @"next_attention_out_buffer",
        @"next_attention_output_buffer",
        @"next_cur_buffer",
        @"next_ffn_cur_buffer",
        @"next_ffn_flat_hc_buffer",
        @"next_ffn_mix_buffer",
        @"next_ffn_norm_buffer",
        @"next_ffn_split_buffer",
        @"next_flat_hc_buffer",
        @"next_full_kv_half_buffer",
        @"next_kv_half_buffer",
        @"next_kv_norm_buffer",
        @"next_kv_norm_snapshot_buffer",
        @"next_kv_raw_buffer",
        @"next_kv_rope_buffer",
        @"next_mix_buffer",
        @"next_norm_buffer",
        @"next_q_cur_buffer",
        @"next_q_lora_buffer",
        @"next_q_norm_buffer",
        @"next_q_raw_buffer",
        @"next_raw_cache_buffer",
        @"next_routed_mid_buffer",
        @"next_routed_out_buffer",
        @"next_router_logits_buffer",
        @"next_router_probs_buffer",
        @"next_router_selected_buffer",
        @"next_router_weights_buffer",
        @"next_shared_out_buffer",
        @"next_split_buffer",
        @"position_buffer",
        @"q_buffer",
        @"q_cur_buffer",
        @"q_norm_buffer",
        @"q_raw_buffer",
        @"raw_cache_buffer",
        @"routed_experts_buffer",
        @"routed_map_buffer",
        @"routed_mid_buffer",
        @"routed_out_buffer",
        @"router_logits_buffer",
        @"router_probs_buffer",
        @"router_selected_buffer",
        @"router_weight_sums_buffer",
        @"router_weights_buffer",
        @"shared_gate_buffer",
        @"shared_mid_buffer",
        @"shared_out_buffer",
        @"shared_up_buffer",
        @"split_buffer",
        @"token_buffer"
    ];
    if (![allowed containsObject:label] || [pool.checkedOut containsObject:label] ||
        bytes == 0u || bytes > 2493173233ull ||
        (pool.priorCommand && pool.priorCommand.status != MTLCommandBufferStatusCompleted)) {
        fail_with_message(error, error_bytes, @"invalid bootstrap scratch checkout");
        return nil;
    }
    id<MTLBuffer> buffer = pool.buffers[label];
    if (buffer) {
        if (buffer.length != bytes) {
            fail_with_message(error, error_bytes, @"bootstrap scratch capacity changed within chunk");
            return nil;
        }
    } else {
        if (pool.buffers.count >= 105u || pool.bytes > 2493173233ull - bytes) {
            fail_with_message(error, error_bytes, @"bootstrap scratch capacity exceeded");
            return nil;
        }
        buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) {
            fail_with_message(error, error_bytes, @"bootstrap scratch allocation failed");
            return nil;
        }
        pool.buffers[label] = buffer;
        pool.bytes += bytes;
    }
    [pool.checkedOut addObject:label];
    if (source) memcpy(buffer.contents, source, bytes);
    else memset(buffer.contents, 0, bytes);
    return buffer;
}


@interface ProbeOwner : NSObject
@property(nonatomic,strong) RustStarBootstrapScratch *prefillLayerBootstrapScratch;
@end
@implementation ProbeOwner
@end
static int cleanup_probe(ProbeOwner *context, BOOL scratch_success,
    uint32_t position_start, uint32_t rows, uint32_t prefill_rows) {
    @try { return scratch_success ? 1:0; }
    @finally {
        if (!scratch_success || position_start + rows == prefill_rows) {
            context.prefillLayerBootstrapScratch = nil;
        }
    }
}
int main(void) { @autoreleasepool {
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();assert(device);
 id<MTLCommandQueue> queue=[device newCommandQueue];assert(queue);
 NSArray<NSString *> *labels=@[@"after_attention_hc_buffer",@"after_ffn_hc_buffer",@"attention_back_buffer",@"attention_block_buffer",@"attention_group_ids_buffer",@"attention_group_map_buffer",@"attention_low_buffer",@"attention_mask_buffer",@"attention_out_buffer",@"attention_output_buffer",@"attention_pad_buffer",@"attn_buffer",@"collapsed_buffer",@"compressed_position_buffer",@"embedding_buffer",@"ffn_cur_buffer",@"ffn_flat_hc_buffer",@"ffn_mix_buffer",@"ffn_norm_buffer",@"ffn_split_buffer",@"flat_hc_buffer",@"full_kv_half_buffer",@"hc_buffer",@"kv_half_buffer",@"kv_norm_buffer",@"kv_norm_snapshot_buffer",@"kv_raw_buffer",@"kv_rope_buffer",@"layer2_attn_packed_kv_buffer",@"layer2_attn_packed_score_buffer",@"layer2_attn_projected_kv_buffer",@"layer2_attn_projected_score_buffer",@"layer2_attn_softmax_buffer",@"layer2_cur_buffer",@"layer2_flat_hc_buffer",@"layer2_indexer_packed_kv_buffer",@"layer2_indexer_packed_score_buffer",@"layer2_indexer_projected_kv_buffer",@"layer2_indexer_projected_score_buffer",@"layer2_indexer_softmax_buffer",@"layer2_kv_norm_buffer",@"layer2_kv_norm_snapshot_buffer",@"layer2_kv_raw_buffer",@"layer2_kv_rope_buffer",@"layer2_mix_buffer",@"layer2_norm_buffer",@"layer2_q_lora_buffer",@"layer2_q_norm_buffer",@"layer2_split_buffer",@"mix_buffer",@"next_after_attention_hc_buffer",@"next_after_ffn_hc_buffer",@"next_attention_back_buffer",@"next_attention_low_buffer",@"next_attention_out_buffer",@"next_attention_output_buffer",@"next_cur_buffer",@"next_ffn_cur_buffer",@"next_ffn_flat_hc_buffer",@"next_ffn_mix_buffer",@"next_ffn_norm_buffer",@"next_ffn_split_buffer",@"next_flat_hc_buffer",@"next_full_kv_half_buffer",@"next_kv_half_buffer",@"next_kv_norm_buffer",@"next_kv_norm_snapshot_buffer",@"next_kv_raw_buffer",@"next_kv_rope_buffer",@"next_mix_buffer",@"next_norm_buffer",@"next_q_cur_buffer",@"next_q_lora_buffer",@"next_q_norm_buffer",@"next_q_raw_buffer",@"next_raw_cache_buffer",@"next_routed_mid_buffer",@"next_routed_out_buffer",@"next_router_logits_buffer",@"next_router_probs_buffer",@"next_router_selected_buffer",@"next_router_weights_buffer",@"next_shared_out_buffer",@"next_split_buffer",@"position_buffer",@"q_buffer",@"q_cur_buffer",@"q_norm_buffer",@"q_raw_buffer",@"raw_cache_buffer",@"routed_experts_buffer",@"routed_map_buffer",@"routed_mid_buffer",@"routed_out_buffer",@"router_logits_buffer",@"router_probs_buffer",@"router_selected_buffer",@"router_weight_sums_buffer",@"router_weights_buffer",@"shared_gate_buffer",@"shared_mid_buffer",@"shared_out_buffer",@"shared_up_buffer",@"split_buffer",@"token_buffer"];
 char error[512]={0};NSUInteger checked=0, guards=0, reuse=0, failures=0;
 for(NSUInteger chunk=0;chunk<7;chunk++) { @autoreleasepool {
  ProbeOwner *owner=[ProbeOwner new];RustStarBootstrapScratch *pool=[RustStarBootstrapScratch new];
  pool.buffers=[NSMutableDictionary new];pool.checkedOut=[NSMutableSet new];owner.prefillLayerBootstrapScratch=pool;
  NSMutableDictionary *identities=[NSMutableDictionary new];
  for(NSUInteger tile=0;tile<4;tile++) { @autoreleasepool {
   if(pool.priorCommand) assert(command_succeeded(pool.priorCommand,error,sizeof(error)));
   [pool.checkedOut removeAllObjects];
   NSUInteger index=0;
   id<MTLCommandBuffer> verify=[queue commandBuffer];id<MTLBlitCommandEncoder> blit=[verify blitCommandEncoder];
   NSMutableArray *copies=[NSMutableArray new];NSMutableArray *expected=[NSMutableArray new];
   for(NSString *label in labels) {
    NSUInteger sizes[]={1,3,4,127,4096,4099,65536};NSUInteger n=sizes[index%7];
    NSMutableData *data=[NSMutableData dataWithLength:n];memset(data.mutableBytes,(int)(tile+chunk+31),n);
    BOOL is_copy=[label isEqual:@"token_buffer"]||[label isEqual:@"position_buffer"]||[label isEqual:@"compressed_position_buffer"];
    id<MTLBuffer> b=bootstrap_scratch_buffer(device,pool,label,n,is_copy?data.bytes:NULL,error,sizeof(error));assert(b&&b.length==n);
    if(tile) { assert([identities[label] unsignedLongLongValue]==(uintptr_t)(__bridge void *)b);reuse++; }
    else { for(NSNumber *v in identities.allValues)assert(v.unsignedLongLongValue!=(uintptr_t)(__bridge void *)b);identities[label]=@((uintptr_t)(__bridge void *)b); }
    id<MTLBuffer> dst=[device newBufferWithLength:n+64 options:MTLResourceStorageModeShared];memset(dst.contents,0x7b,n+64);
    // Blit offsets and size must align to four bytes on this backend. For odd
    // lengths, CPU validates the complete odd range; GPU copies an aligned prefix.
    if(n>=4)[blit copyFromBuffer:b sourceOffset:0 toBuffer:dst destinationOffset:32 size:n-(n%4)];
    const uint8_t *v=b.contents;for(NSUInteger j=0;j<n;j++)assert(v[j]==(is_copy?(tile+chunk+31):0));checked+=n;
    [copies addObject:dst];[expected addObject:@{@"n":@(n),@"value":@(is_copy?(tile+chunk+31):0)}];
    index++;
   }
   [blit endEncoding];[verify commit];assert(command_succeeded(verify,error,sizeof(error)));
   for(NSUInteger i=0;i<copies.count;i++) { id<MTLBuffer> dst=copies[i];NSUInteger n=[expected[i][@"n"] unsignedLongLongValue];uint8_t val=[expected[i][@"value"] unsignedIntValue];const uint8_t *v=dst.contents;NSUInteger copied=n-(n%4);for(NSUInteger j=0;j<copied;j++)assert(v[32+j]==val);checked+=copied;for(NSUInteger j=0;j<32;j++)assert(v[j]==0x7b);for(NSUInteger j=32+copied;j<n+64;j++)assert(v[j]==0x7b);guards+=64+(n%4); }
   assert(pool.buffers.count==105&&pool.checkedOut.count==105);
   assert(!bootstrap_scratch_buffer(device,pool,labels[0],1,NULL,error,sizeof(error)));failures++;
   id<MTLCommandBuffer> poison=[queue commandBuffer];id<MTLBlitCommandEncoder> poisonBlit=[poison blitCommandEncoder];
   for(id<MTLBuffer> b in pool.buffers.allValues){NSUInteger n=b.length-(b.length%4);if(n)[poisonBlit fillBuffer:b range:NSMakeRange(0,n) value:0xa5];if(n<b.length)memset((uint8_t *)b.contents+n,0xa5,b.length-n);}
   [poisonBlit endEncoding];pool.priorCommand=poison;
   // Not-enqueued prior work must be rejected without modifying storage.
   [pool.checkedOut removeObject:labels[0]];
   assert(!bootstrap_scratch_buffer(device,pool,labels[0],1,NULL,error,sizeof(error)));failures++;
   [poison commit];assert(command_succeeded(poison,error,sizeof(error)));
   for(id<MTLBuffer> b in pool.buffers.allValues){const uint8_t *v=b.contents;for(NSUInteger j=0;j<b.length;j++)assert(v[j]==0xa5);}
  }}
  [pool.checkedOut removeAllObjects];
  assert(!bootstrap_scratch_buffer(device,pool,@"persistent_forbidden",4,NULL,error,sizeof(error)));failures++;
  assert(!bootstrap_scratch_buffer(device,pool,labels[0],2,NULL,error,sizeof(error)));failures++;
  assert(!bootstrap_scratch_buffer(device,pool,labels[0],0,NULL,error,sizeof(error)));failures++;
  assert(!bootstrap_scratch_buffer(device,pool,labels[0],2493173234ull,NULL,error,sizeof(error)));failures++;
  assert(cleanup_probe(owner,YES,0,1024,4096)&&owner.prefillLayerBootstrapScratch==pool);
  assert(cleanup_probe(owner,YES,3072,1024,4096)&&!owner.prefillLayerBootstrapScratch);
  owner.prefillLayerBootstrapScratch=pool;assert(!cleanup_probe(owner,NO,0,1024,4096)&&!owner.prefillLayerBootstrapScratch);
  __weak RustStarBootstrapScratch *weakPool=pool;pool=nil;assert(!weakPool);
 }}
 printf("{\"passed\":true,\"chunks\":7,\"tiles\":28,\"labels\":105,\"compared_bytes\":%lu,\"guard_bytes\":%lu,\"identity_reuses\":%lu,\"invalid_checkouts_rejected\":%lu,\"final_and_error_cleanup\":true}\n",checked,guards,reuse,failures);
 return 0;
}}
