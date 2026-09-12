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

// The caller checks prior GPU completion before any checkout in a tile.
// Preserve full initial zero/copy state except four exact 1K destinations:
// their entire range is overwritten by an ordered blit before any content read.
// Source buffers and every other scratch initializer retain their prior policy.
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
    else if (!(bytes == 134217728u &&
        ([label isEqualToString:@"q_cur_buffer"] ||
         [label isEqualToString:@"attention_back_buffer"] ||
         [label isEqualToString:@"next_q_cur_buffer"] ||
         [label isEqualToString:@"next_attention_back_buffer"]))) {
        memset(buffer.contents, 0, bytes);
    }
    return buffer;
}

