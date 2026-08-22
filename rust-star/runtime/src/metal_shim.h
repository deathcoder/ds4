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
    rust_star_metal_prefill_attention_ingress_weights layer3_ingress;
} rust_star_metal_prefill_layer2_attention_weights;

typedef struct rust_star_metal_prefill_layer2_attention_result {
    uint32_t rows;
    uint32_t raw_kv_rows;
    uint32_t compressed_kv_rows;
    uint32_t dispatches;
    uint32_t wrapped_model_ranges;
    uint32_t pointer_matches;
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
    const float *compressor_prime_attn_norm;
    float *compressed_kv_row;
    float *compressed_indexer_row;
    float *ffn_mixes;
    float *ffn_split;
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
