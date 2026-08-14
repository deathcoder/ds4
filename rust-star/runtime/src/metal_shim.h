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
    uint64_t router_hash_offset, router_hash_bytes;
    uint64_t routed_gate_offset, routed_gate_bytes;
    uint64_t routed_up_offset, routed_up_bytes;
    uint64_t routed_down_offset, routed_down_bytes;
    uint64_t shared_gate_offset, shared_gate_bytes;
    uint64_t shared_up_offset, shared_up_bytes;
    uint64_t shared_down_offset, shared_down_bytes;
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
} rust_star_metal_layer0_extension;

int rust_star_metal_create(void **context_out, char *error, size_t error_bytes);

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
