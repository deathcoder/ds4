#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_log_is_tty(FILE *stream) {
    (void)stream;
    return 0;
}

static void fill_values(float *values, size_t count, uint32_t *state) {
    for (size_t i = 0; i < count; i++) {
        *state = *state * 1664525u + 1013904223u;
        values[i] = (float)(int32_t)*state / 2147483648.0f;
    }
}

int main(void) {
    enum { N_COMP = 8192, N_TOKENS = 64, TOP_K = 512, ITERATIONS = 100 };
    const size_t score_count = (size_t)N_COMP * N_TOKENS;
    const size_t selected_count = (size_t)TOP_K * N_TOKENS;
    float *scores = malloc(score_count * sizeof(*scores));
    uint32_t *reference = malloc(selected_count * sizeof(*reference));
    uint32_t *actual = malloc(selected_count * sizeof(*actual));
    if (!scores || !reference || !actual) return 1;

    uint32_t state = 0x6d2b79f5u;
    fill_values(scores, score_count, &state);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *score_tensor =
        ds4_gpu_tensor_alloc(score_count * sizeof(*scores));
    ds4_gpu_tensor *selected_tensor =
        ds4_gpu_tensor_alloc(selected_count * sizeof(*actual));
    if (!score_tensor || !selected_tensor ||
        !ds4_gpu_tensor_write(score_tensor, 0, scores,
                              score_count * sizeof(*scores))) {
        return 1;
    }

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        if (!ds4_gpu_indexer_topk_tensor(selected_tensor, score_tensor,
                                         N_COMP, N_TOKENS, TOP_K) ||
            !ds4_gpu_tensor_read(selected_tensor, 0, actual,
                                 selected_count * sizeof(*actual))) {
            return 1;
        }
        if (iteration == 0) {
            memcpy(reference, actual, selected_count * sizeof(*reference));
        } else if (memcmp(reference, actual,
                          selected_count * sizeof(*reference)) != 0) {
            for (size_t i = 0; i < selected_count; i++) {
                if (reference[i] != actual[i]) {
                    fprintf(stderr,
                            "top-k differs at iteration=%d element=%zu "
                            "expected=%u actual=%u\n",
                            iteration, i, reference[i], actual[i]);
                    return 1;
                }
            }
        }
    }

    printf("top-k stream repeatability PASS: tokens=%d rows=%d top_k=%d "
           "iterations=%d\n",
           N_TOKENS, N_COMP, TOP_K, ITERATIONS);

    const size_t q_count = (size_t)N_TOKENS * 64u * 128u;
    const size_t weight_count = (size_t)N_TOKENS * 64u;
    const size_t comp_count = (size_t)N_COMP * 128u;
    float *q = malloc(q_count * sizeof(*q));
    float *weights = malloc(weight_count * sizeof(*weights));
    float *comp = malloc(comp_count * sizeof(*comp));
    uint32_t *score_reference = malloc(score_count * sizeof(*score_reference));
    uint32_t *score_actual = malloc(score_count * sizeof(*score_actual));
    if (!q || !weights || !comp || !score_reference || !score_actual) return 1;
    fill_values(q, q_count, &state);
    fill_values(weights, weight_count, &state);
    fill_values(comp, comp_count, &state);

    ds4_gpu_tensor *q_tensor = ds4_gpu_tensor_alloc(q_count * sizeof(*q));
    ds4_gpu_tensor *weight_tensor =
        ds4_gpu_tensor_alloc(weight_count * sizeof(*weights));
    ds4_gpu_tensor *comp_tensor =
        ds4_gpu_tensor_alloc(comp_count * sizeof(*comp));
    if (!q_tensor || !weight_tensor || !comp_tensor ||
        !ds4_gpu_tensor_write(q_tensor, 0, q, q_count * sizeof(*q)) ||
        !ds4_gpu_tensor_write(weight_tensor, 0, weights,
                              weight_count * sizeof(*weights)) ||
        !ds4_gpu_tensor_write(comp_tensor, 0, comp,
                              comp_count * sizeof(*comp))) {
        return 1;
    }

    for (int iteration = 0; iteration < 25; iteration++) {
        if (!ds4_gpu_indexer_scores_decode_batch_tensor(
                score_tensor, q_tensor, weight_tensor, comp_tensor,
                N_COMP, N_TOKENS, 28672u, 64u, 128u, 4u,
                1.0f / sqrtf(64.0f * 128.0f)) ||
            !ds4_gpu_tensor_read(score_tensor, 0, score_actual,
                                 score_count * sizeof(*score_actual))) {
            return 1;
        }
        if (iteration == 0) {
            memcpy(score_reference, score_actual,
                   score_count * sizeof(*score_reference));
        } else if (memcmp(score_reference, score_actual,
                          score_count * sizeof(*score_reference)) != 0) {
            for (size_t i = 0; i < score_count; i++) {
                if (score_reference[i] != score_actual[i]) {
                    fprintf(stderr,
                            "indexer score differs at iteration=%d element=%zu "
                            "expected=0x%08x actual=0x%08x\n",
                            iteration, i, score_reference[i], score_actual[i]);
                    return 1;
                }
            }
        }
    }
    printf("indexer score repeatability PASS: tokens=%d rows=%d iterations=25\n",
           N_TOKENS, N_COMP);

    ds4_gpu_tensor_free(comp_tensor);
    ds4_gpu_tensor_free(weight_tensor);
    ds4_gpu_tensor_free(q_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(score_tensor);
    ds4_gpu_cleanup();
    free(score_actual);
    free(score_reference);
    free(comp);
    free(weights);
    free(q);
    free(actual);
    free(reference);
    free(scores);
    return 0;
}
