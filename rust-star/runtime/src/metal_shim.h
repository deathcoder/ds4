#ifndef RUST_STAR_METAL_SHIM_H
#define RUST_STAR_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef struct rust_star_metal_probe_result {
    uint64_t elements;
    uint64_t iterations;
    uint64_t recommended_max_working_set_bytes;
    uint64_t buffer_bytes;
    uint64_t max_total_threads_per_threadgroup;
    uint64_t checksum;
    uint32_t has_unified_memory;
    uint32_t reserved;
    double setup_ms;
    double compile_ms;
    double warmup_wall_ms;
    double warmup_gpu_ms;
    double roundtrip_wall_ms;
    double roundtrip_gpu_ms;
    double batched_wall_ms;
    double batched_gpu_ms;
    char device_name[256];
} rust_star_metal_probe_result;

typedef struct rust_star_metal_embedding_probe_result {
    uint64_t model_bytes;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
    uint64_t page_offset;
    uint64_t buffer_bytes;
    uint64_t inner_offset;
    uint64_t output_elements;
    uint64_t max_buffer_length;
    uint32_t no_copy_pointer_match;
    uint32_t reserved;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_embedding_probe_result;

typedef struct rust_star_metal_projection_probe_result {
    uint64_t model_bytes;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
    uint64_t page_offset;
    uint64_t buffer_bytes;
    uint64_t inner_offset;
    uint64_t input_elements;
    uint64_t output_elements;
    uint64_t max_buffer_length;
    uint32_t no_copy_pointer_match;
    uint32_t simdgroups;
    uint32_t rows_per_threadgroup;
    uint32_t reserved;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_projection_probe_result;

typedef struct rust_star_metal_prefill_q8_probe_result {
    uint64_t model_bytes;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
    uint64_t input_elements_per_row;
    uint64_t output_elements_per_row;
    uint64_t rows;
    uint64_t max_buffer_length;
    uint32_t no_copy_pointer_match;
    uint32_t batch_threads_per_threadgroup;
    uint32_t batch_threadgroups_x;
    uint32_t batch_threadgroups_y;
    double batch_wall_ms;
    double batch_gpu_ms;
    double decode_wall_ms;
    double decode_gpu_ms;
} rust_star_metal_prefill_q8_probe_result;

typedef struct rust_star_metal_prefill_qkv_weights {
    uint64_t q_a_offset, q_a_bytes;
    uint64_t q_a_norm_offset, q_a_norm_bytes;
    uint64_t kv_offset, kv_bytes;
    uint64_t kv_norm_offset, kv_norm_bytes;
    uint64_t q_b_offset, q_b_bytes;
} rust_star_metal_prefill_qkv_weights;

typedef struct rust_star_metal_prefill_qkv_probe_result {
    uint64_t rows;
    uint64_t input_elements_per_row;
    uint64_t q_lora_elements_per_row;
    uint64_t kv_elements_per_row;
    uint64_t q_elements_per_row;
    uint32_t dispatches;
    uint32_t wrapped_model_ranges;
    uint32_t pointer_matches;
    uint32_t position_start;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_prefill_qkv_probe_result;

typedef struct rust_star_metal_prefill_ffn_weights {
    uint64_t hc_fn_offset, hc_fn_bytes;
    uint64_t hc_scale_offset, hc_scale_bytes;
    uint64_t hc_base_offset, hc_base_bytes;
    uint64_t norm_offset, norm_bytes;
    uint64_t router_gate_offset, router_gate_bytes;
    uint64_t router_hash_offset, router_hash_bytes;
    uint64_t routed_gate_offset, routed_gate_bytes;
    uint64_t routed_up_offset, routed_up_bytes;
    uint64_t routed_down_offset, routed_down_bytes;
    uint64_t shared_gate_offset, shared_gate_bytes;
    uint64_t shared_up_offset, shared_up_bytes;
    uint64_t shared_down_offset, shared_down_bytes;
} rust_star_metal_prefill_ffn_weights;

typedef struct rust_star_metal_prefill_attention_ingress_weights {
    uint64_t hc_fn_offset, hc_fn_bytes;
    uint64_t hc_scale_offset, hc_scale_bytes;
    uint64_t hc_base_offset, hc_base_bytes;
    uint64_t norm_offset, norm_bytes;
    uint64_t q_a_offset, q_a_bytes;
} rust_star_metal_prefill_attention_ingress_weights;

typedef struct rust_star_metal_prefill_layer_weights {
    rust_star_metal_prefill_attention_ingress_weights ingress;
    uint64_t q_a_norm_offset, q_a_norm_bytes;
    uint64_t kv_offset, kv_bytes;
    uint64_t kv_norm_offset, kv_norm_bytes;
    uint64_t q_b_offset, q_b_bytes;
    uint64_t attn_sinks_offset, attn_sinks_bytes;
    uint64_t attn_output_a_offset, attn_output_a_bytes;
    uint64_t attn_output_b_offset, attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights ffn;
} rust_star_metal_prefill_layer_weights;

typedef struct rust_star_metal_prefill_kvnorm_weights {
    rust_star_metal_prefill_attention_ingress_weights ingress;
    uint64_t q_a_norm_offset, q_a_norm_bytes;
    uint64_t kv_offset, kv_bytes;
    uint64_t kv_norm_offset, kv_norm_bytes;
} rust_star_metal_prefill_kvnorm_weights;

typedef struct rust_star_metal_prefill_compressor_weights {
    uint64_t attn_ape_offset, attn_ape_bytes;
    uint64_t attn_kv_offset, attn_kv_bytes;
    uint64_t attn_gate_offset, attn_gate_bytes;
    uint64_t attn_norm_offset, attn_norm_bytes;
    uint64_t indexer_ape_offset, indexer_ape_bytes;
    uint64_t indexer_kv_offset, indexer_kv_bytes;
    uint64_t indexer_gate_offset, indexer_gate_bytes;
    uint64_t indexer_norm_offset, indexer_norm_bytes;
} rust_star_metal_prefill_compressor_weights;

typedef struct rust_star_metal_prefill_layer2_attention_weights {
    uint64_t q_b_offset, q_b_bytes;
    uint64_t attn_sinks_offset, attn_sinks_bytes;
    uint64_t attn_output_a_offset, attn_output_a_bytes;
    uint64_t attn_output_b_offset, attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights ffn;
    rust_star_metal_prefill_kvnorm_weights layer3_kvnorm;
    uint64_t layer3_q_b_offset, layer3_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer3_compressor;
    uint64_t layer3_attn_sinks_offset, layer3_attn_sinks_bytes;
    uint64_t layer3_attn_output_a_offset, layer3_attn_output_a_bytes;
    uint64_t layer3_attn_output_b_offset, layer3_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer3_ffn;
    rust_star_metal_prefill_kvnorm_weights layer4_kvnorm;
    uint64_t layer4_q_b_offset, layer4_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer4_compressor;
    uint64_t layer4_attn_sinks_offset, layer4_attn_sinks_bytes;
    uint64_t layer4_attn_output_a_offset, layer4_attn_output_a_bytes;
    uint64_t layer4_attn_output_b_offset, layer4_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer4_ffn;
    rust_star_metal_prefill_kvnorm_weights layer5_kvnorm;
    uint64_t layer5_q_b_offset, layer5_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer5_compressor;
    uint64_t layer5_attn_sinks_offset, layer5_attn_sinks_bytes;
    uint64_t layer5_attn_output_a_offset, layer5_attn_output_a_bytes;
    uint64_t layer5_attn_output_b_offset, layer5_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer5_ffn;
    rust_star_metal_prefill_kvnorm_weights layer6_kvnorm;
    uint64_t layer6_q_b_offset, layer6_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer6_compressor;
    uint64_t layer6_attn_sinks_offset, layer6_attn_sinks_bytes;
    uint64_t layer6_attn_output_a_offset, layer6_attn_output_a_bytes;
    uint64_t layer6_attn_output_b_offset, layer6_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer6_ffn;
    rust_star_metal_prefill_kvnorm_weights layer7_kvnorm;
    uint64_t layer7_q_b_offset, layer7_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer7_compressor;
    uint64_t layer7_attn_sinks_offset, layer7_attn_sinks_bytes;
    uint64_t layer7_attn_output_a_offset, layer7_attn_output_a_bytes;
    uint64_t layer7_attn_output_b_offset, layer7_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer7_ffn;
    rust_star_metal_prefill_kvnorm_weights layer8_kvnorm;
    uint64_t layer8_q_b_offset, layer8_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer8_compressor;
    uint64_t layer8_attn_sinks_offset, layer8_attn_sinks_bytes;
    uint64_t layer8_attn_output_a_offset, layer8_attn_output_a_bytes;
    uint64_t layer8_attn_output_b_offset, layer8_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer8_ffn;
    rust_star_metal_prefill_kvnorm_weights layer9_kvnorm;
    uint64_t layer9_q_b_offset, layer9_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer9_compressor;
    uint64_t layer9_attn_sinks_offset, layer9_attn_sinks_bytes;
    uint64_t layer9_attn_output_a_offset, layer9_attn_output_a_bytes;
    uint64_t layer9_attn_output_b_offset, layer9_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer9_ffn;
    rust_star_metal_prefill_kvnorm_weights layer10_kvnorm;
    uint64_t layer10_q_b_offset, layer10_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer10_compressor;
    uint64_t layer10_attn_sinks_offset, layer10_attn_sinks_bytes;
    uint64_t layer10_attn_output_a_offset, layer10_attn_output_a_bytes;
    uint64_t layer10_attn_output_b_offset, layer10_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer10_ffn;
    rust_star_metal_prefill_kvnorm_weights layer11_kvnorm;
    uint64_t layer11_q_b_offset, layer11_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer11_compressor;
    uint64_t layer11_attn_sinks_offset, layer11_attn_sinks_bytes;
    uint64_t layer11_attn_output_a_offset, layer11_attn_output_a_bytes;
    uint64_t layer11_attn_output_b_offset, layer11_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer11_ffn;
    rust_star_metal_prefill_kvnorm_weights layer12_kvnorm;
    uint64_t layer12_q_b_offset, layer12_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer12_compressor;
    uint64_t layer12_attn_sinks_offset, layer12_attn_sinks_bytes;
    uint64_t layer12_attn_output_a_offset, layer12_attn_output_a_bytes;
    uint64_t layer12_attn_output_b_offset, layer12_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer12_ffn;
    rust_star_metal_prefill_kvnorm_weights layer13_kvnorm;
    uint64_t layer13_q_b_offset, layer13_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer13_compressor;
    uint64_t layer13_attn_sinks_offset, layer13_attn_sinks_bytes;
    uint64_t layer13_attn_output_a_offset, layer13_attn_output_a_bytes;
    uint64_t layer13_attn_output_b_offset, layer13_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer13_ffn;
    rust_star_metal_prefill_kvnorm_weights layer14_kvnorm;
    uint64_t layer14_q_b_offset, layer14_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer14_compressor;
    uint64_t layer14_attn_sinks_offset, layer14_attn_sinks_bytes;
    uint64_t layer14_attn_output_a_offset, layer14_attn_output_a_bytes;
    uint64_t layer14_attn_output_b_offset, layer14_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer14_ffn;
    rust_star_metal_prefill_kvnorm_weights layer15_kvnorm;
    uint64_t layer15_q_b_offset, layer15_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer15_compressor;
    uint64_t layer15_attn_sinks_offset, layer15_attn_sinks_bytes;
    uint64_t layer15_attn_output_a_offset, layer15_attn_output_a_bytes;
    uint64_t layer15_attn_output_b_offset, layer15_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer15_ffn;
    rust_star_metal_prefill_kvnorm_weights layer16_kvnorm;
    uint64_t layer16_q_b_offset, layer16_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer16_compressor;
    uint64_t layer16_attn_sinks_offset, layer16_attn_sinks_bytes;
    uint64_t layer16_attn_output_a_offset, layer16_attn_output_a_bytes;
    uint64_t layer16_attn_output_b_offset, layer16_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer16_ffn;
    rust_star_metal_prefill_kvnorm_weights layer17_kvnorm;
    uint64_t layer17_q_b_offset, layer17_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer17_compressor;
    uint64_t layer17_attn_sinks_offset, layer17_attn_sinks_bytes;
    uint64_t layer17_attn_output_a_offset, layer17_attn_output_a_bytes;
    uint64_t layer17_attn_output_b_offset, layer17_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer17_ffn;
    rust_star_metal_prefill_kvnorm_weights layer18_kvnorm;
    uint64_t layer18_q_b_offset, layer18_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer18_compressor;
    uint64_t layer18_attn_sinks_offset, layer18_attn_sinks_bytes;
    uint64_t layer18_attn_output_a_offset, layer18_attn_output_a_bytes;
    uint64_t layer18_attn_output_b_offset, layer18_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer18_ffn;
    rust_star_metal_prefill_kvnorm_weights layer19_kvnorm;
    uint64_t layer19_q_b_offset, layer19_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer19_compressor;
    uint64_t layer19_attn_sinks_offset, layer19_attn_sinks_bytes;
    uint64_t layer19_attn_output_a_offset, layer19_attn_output_a_bytes;
    uint64_t layer19_attn_output_b_offset, layer19_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer19_ffn;
    rust_star_metal_prefill_kvnorm_weights layer20_kvnorm;
    uint64_t layer20_q_b_offset, layer20_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer20_compressor;
    uint64_t layer20_attn_sinks_offset, layer20_attn_sinks_bytes;
    uint64_t layer20_attn_output_a_offset, layer20_attn_output_a_bytes;
    uint64_t layer20_attn_output_b_offset, layer20_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer20_ffn;
    rust_star_metal_prefill_kvnorm_weights layer21_kvnorm;
    uint64_t layer21_q_b_offset, layer21_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer21_compressor;
    uint64_t layer21_attn_sinks_offset, layer21_attn_sinks_bytes;
    uint64_t layer21_attn_output_a_offset, layer21_attn_output_a_bytes;
    uint64_t layer21_attn_output_b_offset, layer21_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer21_ffn;
    rust_star_metal_prefill_kvnorm_weights layer22_kvnorm;
    uint64_t layer22_q_b_offset, layer22_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer22_compressor;
    uint64_t layer22_attn_sinks_offset, layer22_attn_sinks_bytes;
    uint64_t layer22_attn_output_a_offset, layer22_attn_output_a_bytes;
    uint64_t layer22_attn_output_b_offset, layer22_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer22_ffn;
    rust_star_metal_prefill_kvnorm_weights layer23_kvnorm;
    uint64_t layer23_q_b_offset, layer23_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer23_compressor;
    uint64_t layer23_attn_sinks_offset, layer23_attn_sinks_bytes;
    uint64_t layer23_attn_output_a_offset, layer23_attn_output_a_bytes;
    uint64_t layer23_attn_output_b_offset, layer23_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer23_ffn;
    rust_star_metal_prefill_kvnorm_weights layer24_kvnorm;
    uint64_t layer24_q_b_offset, layer24_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer24_compressor;
    uint64_t layer24_attn_sinks_offset, layer24_attn_sinks_bytes;
    uint64_t layer24_attn_output_a_offset, layer24_attn_output_a_bytes;
    uint64_t layer24_attn_output_b_offset, layer24_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer24_ffn;
    rust_star_metal_prefill_kvnorm_weights layer25_kvnorm;
    uint64_t layer25_q_b_offset, layer25_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer25_compressor;
    uint64_t layer25_attn_sinks_offset, layer25_attn_sinks_bytes;
    uint64_t layer25_attn_output_a_offset, layer25_attn_output_a_bytes;
    uint64_t layer25_attn_output_b_offset, layer25_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer25_ffn;
    rust_star_metal_prefill_kvnorm_weights layer26_kvnorm;
    uint64_t layer26_q_b_offset, layer26_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer26_compressor;
    uint64_t layer26_attn_sinks_offset, layer26_attn_sinks_bytes;
    uint64_t layer26_attn_output_a_offset, layer26_attn_output_a_bytes;
    uint64_t layer26_attn_output_b_offset, layer26_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer26_ffn;
    rust_star_metal_prefill_kvnorm_weights layer27_kvnorm;
    uint64_t layer27_q_b_offset, layer27_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer27_compressor;
    uint64_t layer27_attn_sinks_offset, layer27_attn_sinks_bytes;
    uint64_t layer27_attn_output_a_offset, layer27_attn_output_a_bytes;
    uint64_t layer27_attn_output_b_offset, layer27_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer27_ffn;
    rust_star_metal_prefill_kvnorm_weights layer28_kvnorm;
    uint64_t layer28_q_b_offset, layer28_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer28_compressor;
    uint64_t layer28_attn_sinks_offset, layer28_attn_sinks_bytes;
    uint64_t layer28_attn_output_a_offset, layer28_attn_output_a_bytes;
    uint64_t layer28_attn_output_b_offset, layer28_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer28_ffn;
    rust_star_metal_prefill_kvnorm_weights layer29_kvnorm;
    uint64_t layer29_q_b_offset, layer29_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer29_compressor;
    uint64_t layer29_attn_sinks_offset, layer29_attn_sinks_bytes;
    uint64_t layer29_attn_output_a_offset, layer29_attn_output_a_bytes;
    uint64_t layer29_attn_output_b_offset, layer29_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer29_ffn;
    rust_star_metal_prefill_kvnorm_weights layer30_kvnorm;
    uint64_t layer30_q_b_offset, layer30_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer30_compressor;
    uint64_t layer30_attn_sinks_offset, layer30_attn_sinks_bytes;
    uint64_t layer30_attn_output_a_offset, layer30_attn_output_a_bytes;
    uint64_t layer30_attn_output_b_offset, layer30_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer30_ffn;
    rust_star_metal_prefill_kvnorm_weights layer31_kvnorm;
    uint64_t layer31_q_b_offset, layer31_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer31_compressor;
    uint64_t layer31_attn_sinks_offset, layer31_attn_sinks_bytes;
    uint64_t layer31_attn_output_a_offset, layer31_attn_output_a_bytes;
    uint64_t layer31_attn_output_b_offset, layer31_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer31_ffn;
    rust_star_metal_prefill_kvnorm_weights layer32_kvnorm;
    uint64_t layer32_q_b_offset, layer32_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer32_compressor;
    uint64_t layer32_attn_sinks_offset, layer32_attn_sinks_bytes;
    uint64_t layer32_attn_output_a_offset, layer32_attn_output_a_bytes;
    uint64_t layer32_attn_output_b_offset, layer32_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer32_ffn;
    rust_star_metal_prefill_kvnorm_weights layer33_kvnorm;
    uint64_t layer33_q_b_offset, layer33_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer33_compressor;
    uint64_t layer33_attn_sinks_offset, layer33_attn_sinks_bytes;
    uint64_t layer33_attn_output_a_offset, layer33_attn_output_a_bytes;
    uint64_t layer33_attn_output_b_offset, layer33_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer33_ffn;
    rust_star_metal_prefill_kvnorm_weights layer34_kvnorm;
    uint64_t layer34_q_b_offset, layer34_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer34_compressor;
    uint64_t layer34_attn_sinks_offset, layer34_attn_sinks_bytes;
    uint64_t layer34_attn_output_a_offset, layer34_attn_output_a_bytes;
    uint64_t layer34_attn_output_b_offset, layer34_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer34_ffn;
    rust_star_metal_prefill_kvnorm_weights layer35_kvnorm;
    uint64_t layer35_q_b_offset, layer35_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer35_compressor;
    uint64_t layer35_attn_sinks_offset, layer35_attn_sinks_bytes;
    uint64_t layer35_attn_output_a_offset, layer35_attn_output_a_bytes;
    uint64_t layer35_attn_output_b_offset, layer35_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer35_ffn;
    rust_star_metal_prefill_kvnorm_weights layer36_kvnorm;
    uint64_t layer36_q_b_offset, layer36_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer36_compressor;
    uint64_t layer36_attn_sinks_offset, layer36_attn_sinks_bytes;
    uint64_t layer36_attn_output_a_offset, layer36_attn_output_a_bytes;
    uint64_t layer36_attn_output_b_offset, layer36_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer36_ffn;
    rust_star_metal_prefill_kvnorm_weights layer37_kvnorm;
    uint64_t layer37_q_b_offset, layer37_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer37_compressor;
    uint64_t layer37_attn_sinks_offset, layer37_attn_sinks_bytes;
    uint64_t layer37_attn_output_a_offset, layer37_attn_output_a_bytes;
    uint64_t layer37_attn_output_b_offset, layer37_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer37_ffn;
    rust_star_metal_prefill_kvnorm_weights layer38_kvnorm;
    uint64_t layer38_q_b_offset, layer38_q_b_bytes;
    rust_star_metal_prefill_compressor_weights layer38_compressor;
    uint64_t layer38_attn_sinks_offset, layer38_attn_sinks_bytes;
    uint64_t layer38_attn_output_a_offset, layer38_attn_output_a_bytes;
    uint64_t layer38_attn_output_b_offset, layer38_attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights layer38_ffn;

} rust_star_metal_prefill_layer2_attention_weights;

typedef struct rust_star_metal_prefill_layer2_attention_result {
    uint32_t rows;
    uint32_t raw_kv_rows;
    uint32_t compressed_kv_rows;
    uint32_t dispatches;
    uint32_t wrapped_model_ranges;
    uint32_t pointer_matches;
    uint32_t layer3_compressed_kv_rows;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_prefill_layer2_attention_result;

typedef struct rust_star_metal_prefill_layer_outputs {
    const float *kv_prefix;
    float *q_lora_norm;
    float *kv_norm;
    float *q_cur;
    float *kv_rope;
    float *kv_cur;
    float *attention_output;
    float *attention_back;
    float *attention_low;
    float *attention_out;
    float *after_attention_hc;
    float *ffn_cur;
    float *ffn_norm;
    float *router_logits;
    float *router_probs;
    int32_t *router_selected;
    float *router_weights;
    float *routed_mid;
    float *routed_out;
    float *shared_out;
    float *after_ffn_hc;
} rust_star_metal_prefill_layer_outputs;

typedef struct rust_star_metal_prefill_layer0_weights {
    uint64_t embedding_offset, embedding_bytes;
    uint64_t hc_fn_offset, hc_fn_bytes;
    uint64_t hc_scale_offset, hc_scale_bytes;
    uint64_t hc_base_offset, hc_base_bytes;
    uint64_t attn_norm_offset, attn_norm_bytes;
    rust_star_metal_prefill_qkv_weights qkv;
    uint64_t attn_sinks_offset, attn_sinks_bytes;
    uint64_t attn_output_a_offset, attn_output_a_bytes;
    uint64_t attn_output_b_offset, attn_output_b_bytes;
    rust_star_metal_prefill_ffn_weights ffn;
} rust_star_metal_prefill_layer0_weights;

typedef struct rust_star_metal_prefill_layer0_probe_result {
    uint64_t rows;
    uint64_t input_elements_per_row;
    uint64_t q_lora_elements_per_row;
    uint64_t kv_elements_per_row;
    uint64_t q_elements_per_row;
    uint32_t dispatches;
    uint32_t wrapped_model_ranges;
    uint32_t pointer_matches;
    uint32_t position_start;
    uint32_t raw_cache_rows;
    uint32_t raw_cache_target_row;
    uint32_t raw_cache_guard_rows;
    uint32_t kv_state_mode;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_prefill_layer0_probe_result;

typedef struct rust_star_metal_ingress_probe_result {
    uint64_t model_bytes;
    uint64_t max_buffer_length;
    uint32_t wrapped_model_ranges;
    uint32_t pointer_matches;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_ingress_probe_result;

typedef struct rust_star_metal_layer0_extension {
    uint64_t hc_ffn_fn_offset, hc_ffn_fn_bytes;
    uint64_t hc_ffn_scale_offset, hc_ffn_scale_bytes;
    uint64_t hc_ffn_base_offset, hc_ffn_base_bytes;
    uint64_t ffn_norm_offset, ffn_norm_bytes;
    uint64_t router_gate_offset, router_gate_bytes;
    uint64_t router_aux_offset, router_aux_bytes;
    uint64_t routed_gate_offset, routed_gate_bytes;
    uint64_t routed_up_offset, routed_up_bytes;
    uint64_t routed_down_offset, routed_down_bytes;
    uint64_t shared_gate_offset, shared_gate_bytes;
    uint64_t shared_up_offset, shared_up_bytes;
    uint64_t shared_down_offset, shared_down_bytes;
    uint64_t attn_compressor_ape_offset, attn_compressor_ape_bytes;
    uint64_t attn_compressor_kv_offset, attn_compressor_kv_bytes;
    uint64_t attn_compressor_gate_offset, attn_compressor_gate_bytes;
    uint64_t attn_compressor_norm_offset, attn_compressor_norm_bytes;
    uint64_t indexer_compressor_ape_offset, indexer_compressor_ape_bytes;
    uint64_t indexer_compressor_kv_offset, indexer_compressor_kv_bytes;
    uint64_t indexer_compressor_gate_offset, indexer_compressor_gate_bytes;
    uint64_t indexer_compressor_norm_offset, indexer_compressor_norm_bytes;
    uint64_t indexer_q_offset, indexer_q_bytes;
    uint64_t indexer_weight_offset, indexer_weight_bytes;
    const float *compressor_prime_attn_norm;
    float *compressed_kv_row;
    float *compressed_indexer_row;
    float *kv_norm_pre_rope;
    float *ffn_mixes;
    float *ffn_split;
    float *ffn_cur;
    float *ffn_norm;
    float *router_logits;
    float *router_probs;
    int32_t *selected;
    float *router_weights;
    float *routed_mid;
    float *routed_out;
    float *shared_out;
    float *after_ffn_hc;
    uint32_t warmup_iterations;
    uint32_t measured_iterations;
    double *wall_ms_samples;
    double *gpu_ms_samples;
    uint32_t *repeat_bitwise_matches;
    uint32_t layer_index;
    uint32_t reuse_previous_hc;
    uint32_t command_mode;
    uint32_t chain_final_layer;
    uint32_t position;
    uint32_t initial_state_mode;
    uint32_t context_capacity;
} rust_star_metal_layer0_extension;

int rust_star_metal_create(void **context_out, char *error, size_t error_bytes);

int rust_star_metal_prepare_decoder(void *context, char *error, size_t error_bytes);

int rust_star_metal_run_probe(
    void *context,
    uint64_t elements,
    uint64_t iterations,
    rust_star_metal_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_f16_get_rows(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint32_t n_vocab,
    uint32_t n_embd,
    const uint32_t *tokens,
    uint32_t token_count,
    float *output,
    uint64_t output_elements,
    rust_star_metal_embedding_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_q8_0_projection(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint32_t input_elements,
    uint32_t output_elements,
    const float *input,
    float *output,
    rust_star_metal_projection_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_prefill_q8_boundary(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint32_t input_elements_per_row,
    uint32_t output_elements_per_row,
    uint32_t rows,
    const float *input,
    float *batch_output,
    float *decode_output,
    rust_star_metal_prefill_q8_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_prefill_qkv_boundary(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    const rust_star_metal_prefill_qkv_weights *weights,
    uint32_t rows,
    uint32_t position_start,
    const float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    float *q_cur,
    rust_star_metal_prefill_qkv_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_prefill_layer0_boundary(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    const rust_star_metal_prefill_layer0_weights *weights,
    const rust_star_metal_prefill_attention_ingress_weights *next_ingress,
    const rust_star_metal_prefill_layer_weights *next_layer,
    const rust_star_metal_prefill_layer_outputs *next_outputs,
    const rust_star_metal_prefill_kvnorm_weights *layer2_kvnorm,
    const rust_star_metal_prefill_compressor_weights *layer2_compressors,
    float *layer2_kv_norm_output,
    float *layer2_kv_rope_output,
    float *layer2_kv_cur_output,
    const float *layer2_kv_prefix,
    float *layer2_attn_compressed_output,
    float *layer2_indexer_compressed_output,
    float *layer2_attn_state_kv_output,
    float *layer2_attn_state_score_output,
    float *layer2_indexer_state_kv_output,
    float *layer2_indexer_state_score_output,
    const float *layer2_attn_compressed_prefix,
    const float *layer2_indexer_compressed_prefix,
    uint32_t n_vocab,
    uint32_t rows,
    uint32_t position_start,
    uint32_t kv_state_mode,
    const uint32_t *tokens,
    float *hc_collapsed,
    float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    float *q_cur,
    float *kv_rope,
    float *kv_cur,
    float *raw_cache,
    const float *kv_prefix,
    float *full_kv,
    float *attention_output,
    float *attention_back,
    float *attention_low,
    float *attention_out,
    float *after_attention_hc,
    float *ffn_cur,
    float *ffn_norm,
    float *router_logits,
    float *router_probs,
    int32_t *router_selected,
    float *router_weights,
    float *routed_mid,
    float *routed_out,
    float *shared_out,
    float *after_ffn_hc,
    float *next_hc_collapsed,
    float *next_attn_norm,
    float *next_q_lora,
    rust_star_metal_prefill_layer0_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_ratio128_compressor_replay(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint32_t layer_index,
    uint64_t ape_offset,
    uint64_t ape_bytes,
    uint64_t kv_offset,
    uint64_t kv_bytes,
    uint64_t gate_offset,
    uint64_t gate_bytes,
    uint64_t norm_offset,
    uint64_t norm_bytes,
    const float *activation_sequence,
    uint64_t activation_elements,
    float *output,
    uint64_t output_elements,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_prefill_layer2_attention(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    const rust_star_metal_prefill_layer2_attention_weights *weights,
    float *attention_output,
    float *after_attention_hc,
    float *after_ffn_hc,
    float *ffn_cur_final_tile,
    float *ffn_norm_final_tile,
    int32_t *router_selected_final_tile,
    float *router_weights_final_tile,
    float *routed_out_final_tile,
    float *shared_out_final_tile,
    float *layer3_hc_attn_pre,
    float *layer3_attn_norm,
    float *layer3_q_lora,
    float *layer3_q_lora_norm,
    float *layer3_kv_raw,
    float *layer3_kv_norm,
    float *layer3_q_raw_final_tile,
    float *layer3_q_cur_final_tile,
    float *layer3_kv_rope,
    float *layer3_kv_cur,
    float *layer3_attn_compressed,
    float *layer3_attn_state_kv,
    int32_t *layer3_attn_state_score,
    float *layer3_kqv_out_row0,
    float *layer3_kqv_back_row0,
    float *layer3_attn_low_row0,
    float *layer3_attention_output,
    float *layer3_after_attention_hc,
    float *layer3_after_ffn_hc,
    float *layer3_ffn_cur_final_tile,
    float *layer3_ffn_norm_final_tile,
    int32_t *layer3_router_selected_final_tile,
    float *layer3_router_weights_final_tile,
    float *layer3_routed_out_final_tile,
    float *layer3_shared_out_final_tile,
    float *layer4_hc_attn_pre_final_tile,
    float *layer4_attn_norm_final_tile,
    float *layer4_q_lora_final_tile,
    float *layer4_q_lora_norm_final_tile,
    float *layer4_kv_raw_final_tile,
    float *layer4_kv_norm_final_tile,
    float *layer4_q_raw_final_tile,
    float *layer4_q_cur_final_tile,
    float *layer4_kv_rope_final_tile,
    float *layer4_kv_cur_final_tile,
    float *layer4_attn_compressed,
    float *layer4_attn_state_kv,
    int32_t *layer4_attn_state_score,
    float *layer4_indexer_compressed,
    float *layer4_indexer_state_kv,
    int32_t *layer4_indexer_state_score,
    float *layer4_kqv_out_row0,
    float *layer4_kqv_back_row0,
    float *layer4_attn_low_row0,
    float *layer4_attention_output,
    float *layer4_after_attention_hc,
    float *layer4_after_ffn_hc,
    float *layer4_ffn_cur_final_tile,
    float *layer4_ffn_norm_final_tile,
    int32_t *layer4_router_selected_final_tile,
    float *layer4_router_weights_final_tile,
    float *layer4_routed_out_final_tile,
    float *layer4_shared_out_final_tile,
    float *layer5_hc_attn_pre_final_tile,
    float *layer5_attn_norm_final_tile,
    float *layer5_q_lora_final_tile,
    float *layer5_q_lora_norm_final_tile,
    float *layer5_kv_raw_final_tile,
    float *layer5_kv_norm_final_tile,
    float *layer5_q_raw_final_tile,
    float *layer5_q_cur_final_tile,
    float *layer5_kv_rope_final_tile,
    float *layer5_kv_cur_final_tile,
    float *layer5_attn_compressed,
    float *layer5_attn_state_kv,
    int32_t *layer5_attn_state_score,
    float *layer5_kqv_out_row0,
    float *layer5_kqv_back_row0,
    float *layer5_attn_low_row0,
    float *layer5_attention_output,
    float *layer5_after_attention_hc,
    float *layer5_after_ffn_hc,
    float *layer5_ffn_cur_final_tile,
    float *layer5_ffn_norm_final_tile,
    int32_t *layer5_router_selected_final_tile,
    float *layer5_router_weights_final_tile,
    float *layer5_routed_out_final_tile,
    float *layer5_shared_out_final_tile,
    float *layer6_hc_attn_pre_final_tile,
    float *layer6_attn_norm_final_tile,
    float *layer6_q_lora_final_tile,
    float *layer6_q_lora_norm_final_tile,
    float *layer6_kv_raw_final_tile,
    float *layer6_kv_norm_final_tile,
    float *layer6_q_raw_final_tile,
    float *layer6_q_cur_final_tile,
    float *layer6_kv_rope_final_tile,
    float *layer6_kv_cur_final_tile,
    float *layer6_attn_compressed,
    float *layer6_attn_state_kv,
    int32_t *layer6_attn_state_score,
    float *layer6_indexer_compressed,
    float *layer6_indexer_state_kv,
    int32_t *layer6_indexer_state_score,
    float *layer6_kqv_out_row0,
    float *layer6_kqv_back_row0,
    float *layer6_attn_low_row0,
    float *layer6_attention_output,
    float *layer6_after_attention_hc,
    float *layer6_after_ffn_hc,
    float *layer6_ffn_cur_final_tile,
    float *layer6_ffn_norm_final_tile,
    int32_t *layer6_router_selected_final_tile,
    float *layer6_router_weights_final_tile,
    float *layer6_routed_out_final_tile,
    float *layer6_shared_out_final_tile,
    float *layer7_hc_attn_pre_final_tile,
    float *layer7_attn_norm_final_tile,
    float *layer7_q_lora_final_tile,
    float *layer7_q_lora_norm_final_tile,
    float *layer7_kv_raw_final_tile,
    float *layer7_kv_norm_final_tile,
    float *layer7_q_raw_final_tile,
    float *layer7_q_cur_final_tile,
    float *layer7_kv_rope_final_tile,
    float *layer7_kv_cur_final_tile,
    float *layer7_attn_compressed,
    float *layer7_attn_state_kv,
    int32_t *layer7_attn_state_score,
    float *layer7_kqv_out_row0,
    float *layer7_kqv_back_row0,
    float *layer7_attn_low_row0,
    float *layer7_attention_output,
    float *layer7_after_attention_hc,
    float *layer7_after_ffn_hc,
    float *layer7_ffn_cur_final_tile,
    float *layer7_ffn_norm_final_tile,
    int32_t *layer7_router_selected_final_tile,
    float *layer7_router_weights_final_tile,
    float *layer7_routed_out_final_tile,
    float *layer7_shared_out_final_tile,
    float *layer8_hc_attn_pre_final_tile,
    float *layer8_attn_norm_final_tile,
    float *layer8_q_lora_final_tile,
    float *layer8_q_lora_norm_final_tile,
    float *layer8_kv_raw_final_tile,
    float *layer8_kv_norm_final_tile,
    float *layer8_q_raw_final_tile,
    float *layer8_q_cur_final_tile,
    float *layer8_kv_rope_final_tile,
    float *layer8_kv_cur_final_tile,
    float *layer8_attn_compressed,
    float *layer8_attn_state_kv,
    int32_t *layer8_attn_state_score,
    float *layer8_indexer_compressed,
    float *layer8_indexer_state_kv,
    int32_t *layer8_indexer_state_score,
    float *layer8_kqv_out_row0,
    float *layer8_kqv_back_row0,
    float *layer8_attn_low_row0,
    float *layer8_attention_output,
    float *layer8_after_attention_hc,
    float *layer8_after_ffn_hc,
    float *layer8_ffn_cur_final_tile,
    float *layer8_ffn_norm_final_tile,
    int32_t *layer8_router_selected_final_tile,
    float *layer8_router_weights_final_tile,
    float *layer8_routed_out_final_tile,
    float *layer8_shared_out_final_tile,
    float *layer9_hc_attn_pre_final_tile,
    float *layer9_attn_norm_final_tile,
    float *layer9_q_lora_final_tile,
    float *layer9_q_lora_norm_final_tile,
    float *layer9_kv_raw_final_tile,
    float *layer9_kv_norm_final_tile,
    float *layer9_q_raw_final_tile,
    float *layer9_q_cur_final_tile,
    float *layer9_kv_rope_final_tile,
    float *layer9_kv_cur_final_tile,
    float *layer9_attn_compressed,
    float *layer9_attn_state_kv,
    int32_t *layer9_attn_state_score,
    float *layer9_kqv_out_row0,
    float *layer9_kqv_back_row0,
    float *layer9_attn_low_row0,
    float *layer9_attention_output,
    float *layer9_after_attention_hc,
    float *layer9_after_ffn_hc,
    float *layer9_ffn_cur_final_tile,
    float *layer9_ffn_norm_final_tile,
    int32_t *layer9_router_selected_final_tile,
    float *layer9_router_weights_final_tile,
    float *layer9_routed_out_final_tile,
    float *layer9_shared_out_final_tile,
    float *layer10_hc_attn_pre_final_tile,
    float *layer10_attn_norm_final_tile,
    float *layer10_q_lora_final_tile,
    float *layer10_q_lora_norm_final_tile,
    float *layer10_kv_raw_final_tile,
    float *layer10_kv_norm_final_tile,
    float *layer10_q_raw_final_tile,
    float *layer10_q_cur_final_tile,
    float *layer10_kv_rope_final_tile,
    float *layer10_kv_cur_final_tile,
    float *layer10_attn_compressed,
    float *layer10_attn_state_kv,
    int32_t *layer10_attn_state_score,
    float *layer10_indexer_compressed,
    float *layer10_indexer_state_kv,
    int32_t *layer10_indexer_state_score,
    float *layer10_kqv_out_row0,
    float *layer10_kqv_back_row0,
    float *layer10_attn_low_row0,
    float *layer10_attention_output,
    float *layer10_after_attention_hc,
    float *layer10_after_ffn_hc,
    float *layer10_ffn_cur_final_tile,
    float *layer10_ffn_norm_final_tile,
    int32_t *layer10_router_selected_final_tile,
    float *layer10_router_weights_final_tile,
    float *layer10_routed_out_final_tile,
    float *layer10_shared_out_final_tile,
    float *layer11_hc_attn_pre_final_tile,
    float *layer11_attn_norm_final_tile,
    float *layer11_q_lora_final_tile,
    float *layer11_q_lora_norm_final_tile,
    float *layer11_kv_raw_final_tile,
    float *layer11_kv_norm_final_tile,
    float *layer11_q_raw_final_tile,
    float *layer11_q_cur_final_tile,
    float *layer11_kv_rope_final_tile,
    float *layer11_kv_cur_final_tile,
    float *layer11_attn_compressed,
    float *layer11_attn_state_kv,
    int32_t *layer11_attn_state_score,
    float *layer11_kqv_out_row0,
    float *layer11_kqv_back_row0,
    float *layer11_attn_low_row0,
    float *layer11_attention_output,
    float *layer11_after_attention_hc,
    float *layer11_after_ffn_hc,
    float *layer11_ffn_cur_final_tile,
    float *layer11_ffn_norm_final_tile,
    int32_t *layer11_router_selected_final_tile,
    float *layer11_router_weights_final_tile,
    float *layer11_routed_out_final_tile,
    float *layer11_shared_out_final_tile,
    float *layer12_hc_attn_pre_final_tile,
    float *layer12_attn_norm_final_tile,
    float *layer12_q_lora_final_tile,
    float *layer12_q_lora_norm_final_tile,
    float *layer12_kv_raw_final_tile,
    float *layer12_kv_norm_final_tile,
    float *layer12_q_raw_final_tile,
    float *layer12_q_cur_final_tile,
    float *layer12_kv_rope_final_tile,
    float *layer12_kv_cur_final_tile,
    float *layer12_attn_compressed,
    float *layer12_attn_state_kv,
    int32_t *layer12_attn_state_score,
    float *layer12_indexer_compressed,
    float *layer12_indexer_state_kv,
    int32_t *layer12_indexer_state_score,
    float *layer12_kqv_out_row0,
    float *layer12_kqv_back_row0,
    float *layer12_attn_low_row0,
    float *layer12_attention_output,
    float *layer12_after_attention_hc,
    float *layer12_after_ffn_hc,
    float *layer12_ffn_cur_final_tile,
    float *layer12_ffn_norm_final_tile,
    int32_t *layer12_router_selected_final_tile,
    float *layer12_router_weights_final_tile,
    float *layer12_routed_out_final_tile,
    float *layer12_shared_out_final_tile,
    float *layer13_hc_attn_pre_final_tile,
    float *layer13_attn_norm_final_tile,
    float *layer13_q_lora_final_tile,
    float *layer13_q_lora_norm_final_tile,
    float *layer13_kv_raw_final_tile,
    float *layer13_kv_norm_final_tile,
    float *layer13_q_raw_final_tile,
    float *layer13_q_cur_final_tile,
    float *layer13_kv_rope_final_tile,
    float *layer13_kv_cur_final_tile,
    float *layer13_attn_compressed,
    float *layer13_attn_state_kv,
    int32_t *layer13_attn_state_score,
    float *layer13_kqv_out_row0,
    float *layer13_kqv_back_row0,
    float *layer13_attn_low_row0,
    float *layer13_attention_output,
    float *layer13_after_attention_hc,
    float *layer13_after_ffn_hc,
    float *layer13_ffn_cur_final_tile,
    float *layer13_ffn_norm_final_tile,
    int32_t *layer13_router_selected_final_tile,
    float *layer13_router_weights_final_tile,
    float *layer13_routed_out_final_tile,
    float *layer13_shared_out_final_tile,
    float *layer14_hc_attn_pre_final_tile,
    float *layer14_attn_norm_final_tile,
    float *layer14_q_lora_final_tile,
    float *layer14_q_lora_norm_final_tile,
    float *layer14_kv_raw_final_tile,
    float *layer14_kv_norm_final_tile,
    float *layer14_q_raw_final_tile,
    float *layer14_q_cur_final_tile,
    float *layer14_kv_rope_final_tile,
    float *layer14_kv_cur_final_tile,
    float *layer14_attn_compressed,
    float *layer14_attn_state_kv,
    int32_t *layer14_attn_state_score,
    float *layer14_indexer_compressed,
    float *layer14_indexer_state_kv,
    int32_t *layer14_indexer_state_score,
    float *layer14_kqv_out_row0,
    float *layer14_kqv_back_row0,
    float *layer14_attn_low_row0,
    float *layer14_attention_output,
    float *layer14_after_attention_hc,
    float *layer14_after_ffn_hc,
    float *layer14_ffn_cur_final_tile,
    float *layer14_ffn_norm_final_tile,
    int32_t *layer14_router_selected_final_tile,
    float *layer14_router_weights_final_tile,
    float *layer14_routed_out_final_tile,
    float *layer14_shared_out_final_tile,
    float *layer15_hc_attn_pre_final_tile,
    float *layer15_attn_norm_final_tile,
    float *layer15_q_lora_final_tile,
    float *layer15_q_lora_norm_final_tile,
    float *layer15_kv_raw_final_tile,
    float *layer15_kv_norm_final_tile,
    float *layer15_q_raw_final_tile,
    float *layer15_q_cur_final_tile,
    float *layer15_kv_rope_final_tile,
    float *layer15_kv_cur_final_tile,
    float *layer15_attn_compressed,
    float *layer15_attn_state_kv,
    int32_t *layer15_attn_state_score,
    float *layer15_kqv_out_row0,
    float *layer15_kqv_back_row0,
    float *layer15_attn_low_row0,
    float *layer15_attention_output,
    float *layer15_after_attention_hc,
    float *layer15_after_ffn_hc,
    float *layer15_ffn_cur_final_tile,
    float *layer15_ffn_norm_final_tile,
    int32_t *layer15_router_selected_final_tile,
    float *layer15_router_weights_final_tile,
    float *layer15_routed_out_final_tile,
    float *layer15_shared_out_final_tile,
    float *layer16_hc_attn_pre_final_tile,
    float *layer16_attn_norm_final_tile,
    float *layer16_q_lora_final_tile,
    float *layer16_q_lora_norm_final_tile,
    float *layer16_kv_raw_final_tile,
    float *layer16_kv_norm_final_tile,
    float *layer16_q_raw_final_tile,
    float *layer16_q_cur_final_tile,
    float *layer16_kv_rope_final_tile,
    float *layer16_kv_cur_final_tile,
    float *layer16_attn_compressed,
    float *layer16_attn_state_kv,
    int32_t *layer16_attn_state_score,
    float *layer16_indexer_compressed,
    float *layer16_indexer_state_kv,
    int32_t *layer16_indexer_state_score,
    float *layer16_kqv_out_row0,
    float *layer16_kqv_back_row0,
    float *layer16_attn_low_row0,
    float *layer16_attention_output,
    float *layer16_after_attention_hc,
    float *layer16_after_ffn_hc,
    float *layer16_ffn_cur_final_tile,
    float *layer16_ffn_norm_final_tile,
    int32_t *layer16_router_selected_final_tile,
    float *layer16_router_weights_final_tile,
    float *layer16_routed_out_final_tile,
    float *layer16_shared_out_final_tile,
    float *layer17_hc_attn_pre_final_tile,
    float *layer17_attn_norm_final_tile,
    float *layer17_q_lora_final_tile,
    float *layer17_q_lora_norm_final_tile,
    float *layer17_kv_raw_final_tile,
    float *layer17_kv_norm_final_tile,
    float *layer17_q_raw_final_tile,
    float *layer17_q_cur_final_tile,
    float *layer17_kv_rope_final_tile,
    float *layer17_kv_cur_final_tile,
    float *layer17_attn_compressed,
    float *layer17_attn_state_kv,
    int32_t *layer17_attn_state_score,
    float *layer17_kqv_out_row0,
    float *layer17_kqv_back_row0,
    float *layer17_attn_low_row0,
    float *layer17_attention_output,
    float *layer17_after_attention_hc,
    float *layer17_after_ffn_hc,
    float *layer17_ffn_cur_final_tile,
    float *layer17_ffn_norm_final_tile,
    int32_t *layer17_router_selected_final_tile,
    float *layer17_router_weights_final_tile,
    float *layer17_routed_out_final_tile,
    float *layer17_shared_out_final_tile,
    float *layer18_hc_attn_pre_final_tile,
    float *layer18_attn_norm_final_tile,
    float *layer18_q_lora_final_tile,
    float *layer18_q_lora_norm_final_tile,
    float *layer18_kv_raw_final_tile,
    float *layer18_kv_norm_final_tile,
    float *layer18_q_raw_final_tile,
    float *layer18_q_cur_final_tile,
    float *layer18_kv_rope_final_tile,
    float *layer18_kv_cur_final_tile,
    float *layer18_attn_compressed,
    float *layer18_attn_state_kv,
    int32_t *layer18_attn_state_score,
    float *layer18_indexer_compressed,
    float *layer18_indexer_state_kv,
    int32_t *layer18_indexer_state_score,
    float *layer18_kqv_out_row0,
    float *layer18_kqv_back_row0,
    float *layer18_attn_low_row0,
    float *layer18_attention_output,
    float *layer18_after_attention_hc,
    float *layer18_after_ffn_hc,
    float *layer18_ffn_cur_final_tile,
    float *layer18_ffn_norm_final_tile,
    int32_t *layer18_router_selected_final_tile,
    float *layer18_router_weights_final_tile,
    float *layer18_routed_out_final_tile,
    float *layer18_shared_out_final_tile,
    float *layer19_hc_attn_pre_final_tile,
    float *layer19_attn_norm_final_tile,
    float *layer19_q_lora_final_tile,
    float *layer19_q_lora_norm_final_tile,
    float *layer19_kv_raw_final_tile,
    float *layer19_kv_norm_final_tile,
    float *layer19_q_raw_final_tile,
    float *layer19_q_cur_final_tile,
    float *layer19_kv_rope_final_tile,
    float *layer19_kv_cur_final_tile,
    float *layer19_attn_compressed,
    float *layer19_attn_state_kv,
    int32_t *layer19_attn_state_score,
    float *layer19_kqv_out_row0,
    float *layer19_kqv_back_row0,
    float *layer19_attn_low_row0,
    float *layer19_attention_output,
    float *layer19_after_attention_hc,
    float *layer19_after_ffn_hc,
    float *layer19_ffn_cur_final_tile,
    float *layer19_ffn_norm_final_tile,
    int32_t *layer19_router_selected_final_tile,
    float *layer19_router_weights_final_tile,
    float *layer19_routed_out_final_tile,
    float *layer19_shared_out_final_tile,
    float *layer20_hc_attn_pre_final_tile,
    float *layer20_attn_norm_final_tile,
    float *layer20_q_lora_final_tile,
    float *layer20_q_lora_norm_final_tile,
    float *layer20_kv_raw_final_tile,
    float *layer20_kv_norm_final_tile,
    float *layer20_q_raw_final_tile,
    float *layer20_q_cur_final_tile,
    float *layer20_kv_rope_final_tile,
    float *layer20_kv_cur_final_tile,
    float *layer20_attn_compressed,
    float *layer20_attn_state_kv,
    int32_t *layer20_attn_state_score,
    float *layer20_indexer_compressed,
    float *layer20_indexer_state_kv,
    int32_t *layer20_indexer_state_score,
    float *layer20_kqv_out_row0,
    float *layer20_kqv_back_row0,
    float *layer20_attn_low_row0,
    float *layer20_attention_output,
    float *layer20_after_attention_hc,
    float *layer20_after_ffn_hc,
    float *layer20_ffn_cur_final_tile,
    float *layer20_ffn_norm_final_tile,
    int32_t *layer20_router_selected_final_tile,
    float *layer20_router_weights_final_tile,
    float *layer20_routed_out_final_tile,
    float *layer20_shared_out_final_tile,
    float *layer21_hc_attn_pre_final_tile,
    float *layer21_attn_norm_final_tile,
    float *layer21_q_lora_final_tile,
    float *layer21_q_lora_norm_final_tile,
    float *layer21_kv_raw_final_tile,
    float *layer21_kv_norm_final_tile,
    float *layer21_q_raw_final_tile,
    float *layer21_q_cur_final_tile,
    float *layer21_kv_rope_final_tile,
    float *layer21_kv_cur_final_tile,
    float *layer21_attn_compressed,
    float *layer21_attn_state_kv,
    int32_t *layer21_attn_state_score,
    float *layer21_kqv_out_row0,
    float *layer21_kqv_back_row0,
    float *layer21_attn_low_row0,
    float *layer21_attention_output,
    float *layer21_after_attention_hc,
    float *layer21_after_ffn_hc,
    float *layer21_ffn_cur_final_tile,
    float *layer21_ffn_norm_final_tile,
    int32_t *layer21_router_selected_final_tile,
    float *layer21_router_weights_final_tile,
    float *layer21_routed_out_final_tile,
    float *layer21_shared_out_final_tile,
    float *layer22_hc_attn_pre_final_tile,
    float *layer22_attn_norm_final_tile,
    float *layer22_q_lora_final_tile,
    float *layer22_q_lora_norm_final_tile,
    float *layer22_kv_raw_final_tile,
    float *layer22_kv_norm_final_tile,
    float *layer22_q_raw_final_tile,
    float *layer22_q_cur_final_tile,
    float *layer22_kv_rope_final_tile,
    float *layer22_kv_cur_final_tile,
    float *layer22_attn_compressed,
    float *layer22_attn_state_kv,
    int32_t *layer22_attn_state_score,
    float *layer22_indexer_compressed,
    float *layer22_indexer_state_kv,
    int32_t *layer22_indexer_state_score,
    float *layer22_kqv_out_row0,
    float *layer22_kqv_back_row0,
    float *layer22_attn_low_row0,
    float *layer22_attention_output,
    float *layer22_after_attention_hc,
    float *layer22_after_ffn_hc,
    float *layer22_ffn_cur_final_tile,
    float *layer22_ffn_norm_final_tile,
    int32_t *layer22_router_selected_final_tile,
    float *layer22_router_weights_final_tile,
    float *layer22_routed_out_final_tile,
    float *layer22_shared_out_final_tile,
    float *layer23_hc_attn_pre_final_tile,
    float *layer23_attn_norm_final_tile,
    float *layer23_q_lora_final_tile,
    float *layer23_q_lora_norm_final_tile,
    float *layer23_kv_raw_final_tile,
    float *layer23_kv_norm_final_tile,
    float *layer23_q_raw_final_tile,
    float *layer23_q_cur_final_tile,
    float *layer23_kv_rope_final_tile,
    float *layer23_kv_cur_final_tile,
    float *layer23_attn_compressed,
    float *layer23_attn_state_kv,
    int32_t *layer23_attn_state_score,
    float *layer23_kqv_out_row0,
    float *layer23_kqv_back_row0,
    float *layer23_attn_low_row0,
    float *layer23_attention_output,
    float *layer23_after_attention_hc,
    float *layer23_after_ffn_hc,
    float *layer23_ffn_cur_final_tile,
    float *layer23_ffn_norm_final_tile,
    int32_t *layer23_router_selected_final_tile,
    float *layer23_router_weights_final_tile,
    float *layer23_routed_out_final_tile,
    float *layer23_shared_out_final_tile,
    float *layer24_hc_attn_pre_final_tile,
    float *layer24_attn_norm_final_tile,
    float *layer24_q_lora_final_tile,
    float *layer24_q_lora_norm_final_tile,
    float *layer24_kv_raw_final_tile,
    float *layer24_kv_norm_final_tile,
    float *layer24_q_raw_final_tile,
    float *layer24_q_cur_final_tile,
    float *layer24_kv_rope_final_tile,
    float *layer24_kv_cur_final_tile,
    float *layer24_attn_compressed,
    float *layer24_attn_state_kv,
    int32_t *layer24_attn_state_score,
    float *layer24_indexer_compressed,
    float *layer24_indexer_state_kv,
    int32_t *layer24_indexer_state_score,
    float *layer24_kqv_out_row0,
    float *layer24_kqv_back_row0,
    float *layer24_attn_low_row0,
    float *layer24_attention_output,
    float *layer24_after_attention_hc,
    float *layer24_after_ffn_hc,
    float *layer24_ffn_cur_final_tile,
    float *layer24_ffn_norm_final_tile,
    int32_t *layer24_router_selected_final_tile,
    float *layer24_router_weights_final_tile,
    float *layer24_routed_out_final_tile,
    float *layer24_shared_out_final_tile,
    float *layer25_hc_attn_pre_final_tile,
    float *layer25_attn_norm_final_tile,
    float *layer25_q_lora_final_tile,
    float *layer25_q_lora_norm_final_tile,
    float *layer25_kv_raw_final_tile,
    float *layer25_kv_norm_final_tile,
    float *layer25_q_raw_final_tile,
    float *layer25_q_cur_final_tile,
    float *layer25_kv_rope_final_tile,
    float *layer25_kv_cur_final_tile,
    float *layer25_attn_compressed,
    float *layer25_attn_state_kv,
    int32_t *layer25_attn_state_score,
    float *layer25_kqv_out_row0,
    float *layer25_kqv_back_row0,
    float *layer25_attn_low_row0,
    float *layer25_attention_output,
    float *layer25_after_attention_hc,
    float *layer25_after_ffn_hc,
    float *layer25_ffn_cur_final_tile,
    float *layer25_ffn_norm_final_tile,
    int32_t *layer25_router_selected_final_tile,
    float *layer25_router_weights_final_tile,
    float *layer25_routed_out_final_tile,
    float *layer25_shared_out_final_tile,
    float *layer26_hc_attn_pre_final_tile,
    float *layer26_attn_norm_final_tile,
    float *layer26_q_lora_final_tile,
    float *layer26_q_lora_norm_final_tile,
    float *layer26_kv_raw_final_tile,
    float *layer26_kv_norm_final_tile,
    float *layer26_q_raw_final_tile,
    float *layer26_q_cur_final_tile,
    float *layer26_kv_rope_final_tile,
    float *layer26_kv_cur_final_tile,
    float *layer26_attn_compressed,
    float *layer26_attn_state_kv,
    int32_t *layer26_attn_state_score,
    float *layer26_indexer_compressed,
    float *layer26_indexer_state_kv,
    int32_t *layer26_indexer_state_score,
    float *layer26_kqv_out_row0,
    float *layer26_kqv_back_row0,
    float *layer26_attn_low_row0,
    float *layer26_attention_output,
    float *layer26_after_attention_hc,
    float *layer26_after_ffn_hc,
    float *layer26_ffn_cur_final_tile,
    float *layer26_ffn_norm_final_tile,
    int32_t *layer26_router_selected_final_tile,
    float *layer26_router_weights_final_tile,
    float *layer26_routed_out_final_tile,
    float *layer26_shared_out_final_tile,
    float *layer27_hc_attn_pre_final_tile,
    float *layer27_attn_norm_final_tile,
    float *layer27_q_lora_final_tile,
    float *layer27_q_lora_norm_final_tile,
    float *layer27_kv_raw_final_tile,
    float *layer27_kv_norm_final_tile,
    float *layer27_q_raw_final_tile,
    float *layer27_q_cur_final_tile,
    float *layer27_kv_rope_final_tile,
    float *layer27_kv_cur_final_tile,
    float *layer27_attn_compressed,
    float *layer27_attn_state_kv,
    int32_t *layer27_attn_state_score,
    float *layer27_kqv_out_row0,
    float *layer27_kqv_back_row0,
    float *layer27_attn_low_row0,
    float *layer27_attention_output,
    float *layer27_after_attention_hc,
    float *layer27_after_ffn_hc,
    float *layer27_ffn_cur_final_tile,
    float *layer27_ffn_norm_final_tile,
    int32_t *layer27_router_selected_final_tile,
    float *layer27_router_weights_final_tile,
    float *layer27_routed_out_final_tile,
    float *layer27_shared_out_final_tile,
    float *layer28_hc_attn_pre_final_tile,
    float *layer28_attn_norm_final_tile,
    float *layer28_q_lora_final_tile,
    float *layer28_q_lora_norm_final_tile,
    float *layer28_kv_raw_final_tile,
    float *layer28_kv_norm_final_tile,
    float *layer28_q_raw_final_tile,
    float *layer28_q_cur_final_tile,
    float *layer28_kv_rope_final_tile,
    float *layer28_kv_cur_final_tile,
    float *layer28_attn_compressed,
    float *layer28_attn_state_kv,
    int32_t *layer28_attn_state_score,
    float *layer28_indexer_compressed,
    float *layer28_indexer_state_kv,
    int32_t *layer28_indexer_state_score,
    float *layer28_kqv_out_row0,
    float *layer28_kqv_back_row0,
    float *layer28_attn_low_row0,
    float *layer28_attention_output,
    float *layer28_after_attention_hc,
    float *layer28_after_ffn_hc,
    float *layer28_ffn_cur_final_tile,
    float *layer28_ffn_norm_final_tile,
    int32_t *layer28_router_selected_final_tile,
    float *layer28_router_weights_final_tile,
    float *layer28_routed_out_final_tile,
    float *layer28_shared_out_final_tile,
    float *layer29_hc_attn_pre_final_tile,
    float *layer29_attn_norm_final_tile,
    float *layer29_q_lora_final_tile,
    float *layer29_q_lora_norm_final_tile,
    float *layer29_kv_raw_final_tile,
    float *layer29_kv_norm_final_tile,
    float *layer29_q_raw_final_tile,
    float *layer29_q_cur_final_tile,
    float *layer29_kv_rope_final_tile,
    float *layer29_kv_cur_final_tile,
    float *layer29_attn_compressed,
    float *layer29_attn_state_kv,
    int32_t *layer29_attn_state_score,
    float *layer29_kqv_out_row0,
    float *layer29_kqv_back_row0,
    float *layer29_attn_low_row0,
    float *layer29_attention_output,
    float *layer29_after_attention_hc,
    float *layer29_after_ffn_hc,
    float *layer29_ffn_cur_final_tile,
    float *layer29_ffn_norm_final_tile,
    int32_t *layer29_router_selected_final_tile,
    float *layer29_router_weights_final_tile,
    float *layer29_routed_out_final_tile,
    float *layer29_shared_out_final_tile,
    float *layer30_hc_attn_pre_final_tile,
    float *layer30_attn_norm_final_tile,
    float *layer30_q_lora_final_tile,
    float *layer30_q_lora_norm_final_tile,
    float *layer30_kv_raw_final_tile,
    float *layer30_kv_norm_final_tile,
    float *layer30_q_raw_final_tile,
    float *layer30_q_cur_final_tile,
    float *layer30_kv_rope_final_tile,
    float *layer30_kv_cur_final_tile,
    float *layer30_attn_compressed,
    float *layer30_attn_state_kv,
    int32_t *layer30_attn_state_score,
    float *layer30_indexer_compressed,
    float *layer30_indexer_state_kv,
    int32_t *layer30_indexer_state_score,
    float *layer30_kqv_out_row0,
    float *layer30_kqv_back_row0,
    float *layer30_attn_low_row0,
    float *layer30_attention_output,
    float *layer30_after_attention_hc,
    float *layer30_after_ffn_hc,
    float *layer30_ffn_cur_final_tile,
    float *layer30_ffn_norm_final_tile,
    int32_t *layer30_router_selected_final_tile,
    float *layer30_router_weights_final_tile,
    float *layer30_routed_out_final_tile,
    float *layer30_shared_out_final_tile,
    float *layer31_hc_attn_pre_final_tile,
    float *layer31_attn_norm_final_tile,
    float *layer31_q_lora_final_tile,
    float *layer31_q_lora_norm_final_tile,
    float *layer31_kv_raw_final_tile,
    float *layer31_kv_norm_final_tile,
    float *layer31_q_raw_final_tile,
    float *layer31_q_cur_final_tile,
    float *layer31_kv_rope_final_tile,
    float *layer31_kv_cur_final_tile,
    float *layer31_attn_compressed,
    float *layer31_attn_state_kv,
    int32_t *layer31_attn_state_score,
    float *layer31_kqv_out_row0,
    float *layer31_kqv_back_row0,
    float *layer31_attn_low_row0,
    float *layer31_attention_output,
    float *layer31_after_attention_hc,
    float *layer31_after_ffn_hc,
    float *layer31_ffn_cur_final_tile,
    float *layer31_ffn_norm_final_tile,
    int32_t *layer31_router_selected_final_tile,
    float *layer31_router_weights_final_tile,
    float *layer31_routed_out_final_tile,
    float *layer31_shared_out_final_tile,
    float *layer32_hc_attn_pre_final_tile,
    float *layer32_attn_norm_final_tile,
    float *layer32_q_lora_final_tile,
    float *layer32_q_lora_norm_final_tile,
    float *layer32_kv_raw_final_tile,
    float *layer32_kv_norm_final_tile,
    float *layer32_q_raw_final_tile,
    float *layer32_q_cur_final_tile,
    float *layer32_kv_rope_final_tile,
    float *layer32_kv_cur_final_tile,
    float *layer32_attn_compressed,
    float *layer32_attn_state_kv,
    int32_t *layer32_attn_state_score,
    float *layer32_indexer_compressed,
    float *layer32_indexer_state_kv,
    int32_t *layer32_indexer_state_score,
    float *layer32_kqv_out_row0,
    float *layer32_kqv_back_row0,
    float *layer32_attn_low_row0,
    float *layer32_attention_output,
    float *layer32_after_attention_hc,
    float *layer32_after_ffn_hc,
    float *layer32_ffn_cur_final_tile,
    float *layer32_ffn_norm_final_tile,
    int32_t *layer32_router_selected_final_tile,
    float *layer32_router_weights_final_tile,
    float *layer32_routed_out_final_tile,
    float *layer32_shared_out_final_tile,
    float *layer33_hc_attn_pre_final_tile,
    float *layer33_attn_norm_final_tile,
    float *layer33_q_lora_final_tile,
    float *layer33_q_lora_norm_final_tile,
    float *layer33_kv_raw_final_tile,
    float *layer33_kv_norm_final_tile,
    float *layer33_q_raw_final_tile,
    float *layer33_q_cur_final_tile,
    float *layer33_kv_rope_final_tile,
    float *layer33_kv_cur_final_tile,
    float *layer33_attn_compressed,
    float *layer33_attn_state_kv,
    int32_t *layer33_attn_state_score,
    float *layer33_kqv_out_row0,
    float *layer33_kqv_back_row0,
    float *layer33_attn_low_row0,
    float *layer33_attention_output,
    float *layer33_after_attention_hc,
    float *layer33_after_ffn_hc,
    float *layer33_ffn_cur_final_tile,
    float *layer33_ffn_norm_final_tile,
    int32_t *layer33_router_selected_final_tile,
    float *layer33_router_weights_final_tile,
    float *layer33_routed_out_final_tile,
    float *layer33_shared_out_final_tile,
    float *layer34_hc_attn_pre_final_tile,
    float *layer34_attn_norm_final_tile,
    float *layer34_q_lora_final_tile,
    float *layer34_q_lora_norm_final_tile,
    float *layer34_kv_raw_final_tile,
    float *layer34_kv_norm_final_tile,
    float *layer34_q_raw_final_tile,
    float *layer34_q_cur_final_tile,
    float *layer34_kv_rope_final_tile,
    float *layer34_kv_cur_final_tile,
    float *layer34_attn_compressed,
    float *layer34_attn_state_kv,
    int32_t *layer34_attn_state_score,
    float *layer34_indexer_compressed,
    float *layer34_indexer_state_kv,
    int32_t *layer34_indexer_state_score,
    float *layer34_kqv_out_row0,
    float *layer34_kqv_back_row0,
    float *layer34_attn_low_row0,
    float *layer34_attention_output,
    float *layer34_after_attention_hc,
    float *layer34_after_ffn_hc,
    float *layer34_ffn_cur_final_tile,
    float *layer34_ffn_norm_final_tile,
    int32_t *layer34_router_selected_final_tile,
    float *layer34_router_weights_final_tile,
    float *layer34_routed_out_final_tile,
    float *layer34_shared_out_final_tile,
    float *layer35_hc_attn_pre_final_tile,
    float *layer35_attn_norm_final_tile,
    float *layer35_q_lora_final_tile,
    float *layer35_q_lora_norm_final_tile,
    float *layer35_kv_raw_final_tile,
    float *layer35_kv_norm_final_tile,
    float *layer35_q_raw_final_tile,
    float *layer35_q_cur_final_tile,
    float *layer35_kv_rope_final_tile,
    float *layer35_kv_cur_final_tile,
    float *layer35_attn_compressed,
    float *layer35_attn_state_kv,
    int32_t *layer35_attn_state_score,
    float *layer35_kqv_out_row0,
    float *layer35_kqv_back_row0,
    float *layer35_attn_low_row0,
    float *layer35_attention_output,
    float *layer35_after_attention_hc,
    float *layer35_after_ffn_hc,
    float *layer35_ffn_cur_final_tile,
    float *layer35_ffn_norm_final_tile,
    int32_t *layer35_router_selected_final_tile,
    float *layer35_router_weights_final_tile,
    float *layer35_routed_out_final_tile,
    float *layer35_shared_out_final_tile,
    float *layer36_hc_attn_pre_final_tile,
    float *layer36_attn_norm_final_tile,
    float *layer36_q_lora_final_tile,
    float *layer36_q_lora_norm_final_tile,
    float *layer36_kv_raw_final_tile,
    float *layer36_kv_norm_final_tile,
    float *layer36_q_raw_final_tile,
    float *layer36_q_cur_final_tile,
    float *layer36_kv_rope_final_tile,
    float *layer36_kv_cur_final_tile,
    float *layer36_attn_compressed,
    float *layer36_attn_state_kv,
    int32_t *layer36_attn_state_score,
    float *layer36_indexer_compressed,
    float *layer36_indexer_state_kv,
    int32_t *layer36_indexer_state_score,
    float *layer36_kqv_out_row0,
    float *layer36_kqv_back_row0,
    float *layer36_attn_low_row0,
    float *layer36_attention_output,
    float *layer36_after_attention_hc,
    float *layer36_after_ffn_hc,
    float *layer36_ffn_cur_final_tile,
    float *layer36_ffn_norm_final_tile,
    int32_t *layer36_router_selected_final_tile,
    float *layer36_router_weights_final_tile,
    float *layer36_routed_out_final_tile,
    float *layer36_shared_out_final_tile,
    float *layer37_hc_attn_pre_final_tile,
    float *layer37_attn_norm_final_tile,
    float *layer37_q_lora_final_tile,
    float *layer37_q_lora_norm_final_tile,
    float *layer37_kv_raw_final_tile,
    float *layer37_kv_norm_final_tile,
    float *layer37_q_raw_final_tile,
    float *layer37_q_cur_final_tile,
    float *layer37_kv_rope_final_tile,
    float *layer37_kv_cur_final_tile,
    float *layer37_attn_compressed,
    float *layer37_attn_state_kv,
    int32_t *layer37_attn_state_score,
    float *layer37_kqv_out_row0,
    float *layer37_kqv_back_row0,
    float *layer37_attn_low_row0,
    float *layer37_attention_output,
    float *layer37_after_attention_hc,
    float *layer37_after_ffn_hc,
    float *layer37_ffn_cur_final_tile,
    float *layer37_ffn_norm_final_tile,
    int32_t *layer37_router_selected_final_tile,
    float *layer37_router_weights_final_tile,
    float *layer37_routed_out_final_tile,
    float *layer37_shared_out_final_tile,
    float *layer38_hc_attn_pre_final_tile,
    float *layer38_attn_norm_final_tile,
    float *layer38_q_lora_final_tile,
    float *layer38_q_lora_norm_final_tile,
    float *layer38_kv_raw_final_tile,
    float *layer38_kv_norm_final_tile,
    float *layer38_q_raw_final_tile,
    float *layer38_q_cur_final_tile,
    float *layer38_kv_rope_final_tile,
    float *layer38_kv_cur_final_tile,
    float *layer38_attn_compressed,
    float *layer38_attn_state_kv,
    int32_t *layer38_attn_state_score,
    float *layer38_indexer_compressed,
    float *layer38_indexer_state_kv,
    int32_t *layer38_indexer_state_score,
    float *layer38_kqv_out_row0,
    float *layer38_kqv_back_row0,
    float *layer38_attn_low_row0,
    float *layer38_attention_output,
    float *layer38_after_attention_hc,
    float *layer38_after_ffn_hc,
    float *layer38_ffn_cur_final_tile,
    float *layer38_ffn_norm_final_tile,
    int32_t *layer38_router_selected_final_tile,
    float *layer38_router_weights_final_tile,
    float *layer38_routed_out_final_tile,
    float *layer38_shared_out_final_tile,
    rust_star_metal_prefill_layer2_attention_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_attention_ingress(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint32_t token,
    uint32_t n_vocab,
    uint64_t embedding_offset,
    uint64_t embedding_bytes,
    uint64_t hc_fn_offset,
    uint64_t hc_fn_bytes,
    uint64_t hc_scale_offset,
    uint64_t hc_scale_bytes,
    uint64_t hc_base_offset,
    uint64_t hc_base_bytes,
    uint64_t attn_norm_offset,
    uint64_t attn_norm_bytes,
    uint64_t q_a_offset,
    uint64_t q_a_bytes,
    uint64_t q_a_norm_offset,
    uint64_t q_a_norm_bytes,
    uint64_t kv_offset,
    uint64_t kv_bytes,
    uint64_t kv_norm_offset,
    uint64_t kv_norm_bytes,
    uint64_t q_b_offset,
    uint64_t q_b_bytes,
    uint64_t attn_sinks_offset,
    uint64_t attn_sinks_bytes,
    uint64_t attn_output_a_offset,
    uint64_t attn_output_a_bytes,
    uint64_t attn_output_b_offset,
    uint64_t attn_output_b_bytes,
    float *mixes,
    float *split,
    float *collapsed,
    float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    float *q_cur,
    float *kv_rope,
    float *kv_cur,
    float *cache_rows,
    const float *cache_row0,
    float *attention_raw,
    float *attention_back,
    float *attention_low,
    float *attention_out,
    float *after_attention_hc,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes,
    const rust_star_metal_layer0_extension *layer0);

typedef struct rust_star_metal_sparse_indexed_result {
    uint32_t position;
    uint32_t compressed_rows;
    uint32_t raw_rows;
    uint32_t top_k;
    uint32_t dispatches;
    uint32_t wrapped_model_ranges;
    uint32_t pointer_matches;
    uint32_t split_count;
    double wall_ms;
    double gpu_ms;
} rust_star_metal_sparse_indexed_result;

int rust_star_metal_run_sparse_indexed_attention(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t indexer_q_offset,
    uint64_t indexer_q_bytes,
    uint64_t indexer_weight_offset,
    uint64_t indexer_weight_bytes,
    uint64_t sinks_offset,
    uint64_t sinks_bytes,
    uint32_t position,
    uint32_t compressed_rows,
    const float *q_lora_norm,
    const float *attn_norm,
    const float *q_current,
    const float *raw_cache,
    const float *attention_comp_cache,
    const float *indexer_comp_cache,
    float *indexer_q,
    float *indexer_weights,
    float *indexer_scores,
    int32_t *indexer_topk,
    float *kqv_out,
    float *kqv_back,
    rust_star_metal_sparse_indexed_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_output_head(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t hc_fn_offset,
    uint64_t hc_fn_bytes,
    uint64_t hc_scale_offset,
    uint64_t hc_scale_bytes,
    uint64_t hc_base_offset,
    uint64_t hc_base_bytes,
    uint64_t output_norm_offset,
    uint64_t output_norm_bytes,
    uint64_t output_offset,
    uint64_t output_bytes,
    float *hc_pre,
    float *hc_weights,
    float *hc,
    float *norm,
    float *logits,
    uint32_t collect_intermediates,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes);

/* Diagnostic-only state import for the first production sparse boundary.
 * The payload shapes and position are deliberately fixed to the captured
 * layer-2 position-4099 control. */
int rust_star_metal_seed_retained_sparse_layer2_position4099(
    void *context,
    const float *input_hc,
    const float *raw_cache_prior,
    const float *attention_compressed_prior,
    const float *indexer_compressed_prior,
    const float *attention_state_kv_pre,
    const float *attention_state_score_pre,
    const float *indexer_state_kv_pre,
    const float *indexer_state_score_pre,
    char *error,
    size_t error_bytes);

int rust_star_metal_copy_retained_sparse_layer2_position4099(
    void *context,
    float *indexer_q,
    float *indexer_weights,
    float *indexer_scores,
    int32_t *indexer_topk,
    char *error,
    size_t error_bytes);

/* Diagnostic-only state import/readback for the first three-sort-block,
 * two-merge-pass sparse boundary at layer 2 position 8195. */
int rust_star_metal_seed_retained_sparse_layer2_position8195(
    void *context,
    const float *input_hc,
    const float *raw_cache_prior,
    const float *attention_compressed_prior,
    const float *indexer_compressed_prior,
    const float *attention_state_kv_pre,
    const float *attention_state_score_pre,
    const float *indexer_state_kv_pre,
    const float *indexer_state_score_pre,
    char *error,
    size_t error_bytes);

/* Seeds only the captured raw/compressed cache histories and recurrent
 * layer-2 compressor states. Layers 0 and 1 then execute normally and produce
 * the live HC handoff consumed by layer 2. */
int rust_star_metal_seed_retained_sparse_layers012_position8195(
    void *context,
    const float *layer0_raw_cache_prior,
    const float *layer1_raw_cache_prior,
    const float *layer2_raw_cache_prior,
    const float *attention_compressed_prior,
    const float *indexer_compressed_prior,
    const float *attention_state_kv_pre,
    const float *attention_state_score_pre,
    const float *indexer_state_kv_pre,
    const float *indexer_state_score_pre,
    char *error,
    size_t error_bytes);

/* Seeds one layer's exact pre-position-8195 retained history. Ratio-4
 * attention rows may be sparse and carry their original row indices; the
 * indexer cache remains complete because it determines the 512 visible rows. */
int rust_star_metal_seed_retained_decoder_layer_position8195(
    void *context,
    uint32_t layer_index,
    const float *raw_cache_prior,
    const int32_t *attention_row_indices,
    const float *attention_compressed_prior,
    uint32_t attention_rows,
    const float *attention_state_kv_pre,
    const float *attention_state_score_pre,
    uint32_t attention_state_elements,
    const float *indexer_compressed_prior,
    uint32_t indexer_rows,
    const float *indexer_state_kv_pre,
    const float *indexer_state_score_pre,
    uint32_t indexer_state_elements,
    char *error,
    size_t error_bytes);

int rust_star_metal_copy_retained_sparse_layer2_position8195(
    void *context,
    float *indexer_q,
    float *indexer_weights,
    float *indexer_scores,
    int32_t *indexer_topk,
    char *error,
    size_t error_bytes);

int rust_star_metal_copy_compressed_kv_row(
    void *context,
    uint32_t layer_index,
    uint32_t row_index,
    float *output,
    uint64_t output_elements,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_ffn_router(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t hc_fn_offset,
    uint64_t hc_fn_bytes,
    uint64_t hc_scale_offset,
    uint64_t hc_scale_bytes,
    uint64_t hc_base_offset,
    uint64_t hc_base_bytes,
    uint64_t ffn_norm_offset,
    uint64_t ffn_norm_bytes,
    uint64_t gate_offset,
    uint64_t gate_bytes,
    uint64_t bias_offset,
    uint64_t bias_bytes,
    uint64_t hash_offset,
    uint64_t hash_bytes,
    const float *after_attention_hc,
    float *mixes,
    float *split,
    float *ffn_cur,
    float *ffn_norm,
    float *logits,
    float *probs,
    int32_t *selected,
    float *weights,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes);

int rust_star_metal_run_moe_output(
    void *context,
    const void *model_mapping,
    uint64_t model_bytes,
    uint64_t routed_gate_offset,
    uint64_t routed_gate_bytes,
    uint64_t routed_up_offset,
    uint64_t routed_up_bytes,
    uint64_t routed_down_offset,
    uint64_t routed_down_bytes,
    uint64_t shared_gate_offset,
    uint64_t shared_gate_bytes,
    uint64_t shared_up_offset,
    uint64_t shared_up_bytes,
    uint64_t shared_down_offset,
    uint64_t shared_down_bytes,
    const float *ffn_norm,
    const int32_t *selected,
    const float *weights,
    const float *after_attention_hc,
    const float *split,
    float *routed_mid,
    float *routed_out,
    float *shared_out,
    float *after_ffn_hc,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes);

void rust_star_metal_destroy(void *context);

#endif
