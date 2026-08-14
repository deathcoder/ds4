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
    float *mixes,
    float *split,
    float *collapsed,
    float *attn_norm,
    float *q_lora,
    float *q_lora_norm,
    float *kv_raw,
    float *kv_norm,
    float *q_raw,
    rust_star_metal_ingress_probe_result *result,
    char *error,
    size_t error_bytes);

void rust_star_metal_destroy(void *context);

#endif
