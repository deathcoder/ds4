/* Capture DwarfStar's closed-loop decoder continuation after an exact batched
 * prefill.  The input transcript proves every intermediate greedy selection;
 * the first and final logits anchor the state handoff and the first default
 * sparse ratio-4 boundary without retaining a multi-gigabyte logits trace. */

#include "ds4.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PREFILL_TOKENS = 2048 };

static int write_bytes(const char *path, const void *payload, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp || fwrite(payload, 1, bytes, fp) != bytes || fclose(fp) != 0) {
        fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno));
        return 1;
    }
    return 0;
}

static int write_logits(ds4_session *session, ds4_engine *engine,
                        const char *path) {
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits || ds4_session_copy_logits(session, logits, vocab) != vocab) {
        free(logits);
        return 1;
    }
    const int failed = write_bytes(path, logits, (size_t)vocab * sizeof(*logits));
    free(logits);
    return failed;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s MODEL PREFILL_TOKENS_U32 FINAL_POSITION "
                "PREFILL_LAYER_PAYLOAD INPUT_TOKENS FIRST_LOGITS FINAL_LOGITS\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    const long final_position = strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || final_position < PREFILL_TOKENS ||
        final_position > INT32_MAX - 1) {
        fprintf(stderr, "invalid final position\n");
        return 2;
    }

    FILE *token_file = fopen(argv[2], "rb");
    uint32_t raw_tokens[PREFILL_TOKENS];
    int prompt_tokens[PREFILL_TOKENS];
    if (!token_file ||
        fread(raw_tokens, sizeof(raw_tokens[0]), PREFILL_TOKENS, token_file) !=
            PREFILL_TOKENS ||
        fgetc(token_file) != EOF || fclose(token_file) != 0) {
        fprintf(stderr, "cannot read the exact 2K token fixture\n");
        return 1;
    }
    for (size_t index = 0; index < PREFILL_TOKENS; index++) {
        if (raw_tokens[index] > INT32_MAX) {
            fprintf(stderr, "invalid token at index %zu\n", index);
            return 1;
        }
        prompt_tokens[index] = (int)raw_tokens[index];
    }

    ds4_engine_options options = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = (int)final_position + 1,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, (int)final_position + 1) != 0) {
        ds4_engine_close(engine);
        return 1;
    }
    ds4_tokens prompt = {
        .v = prompt_tokens,
        .len = PREFILL_TOKENS,
        .cap = PREFILL_TOKENS,
    };
    char error[256] = "";
    if (ds4_session_sync(session, &prompt, error, sizeof(error)) != 0) {
        fprintf(stderr, "batched prefill failed: %s\n", error);
        return 1;
    }
    const uint64_t payload_bytes =
        ds4_session_layer_payload_bytes(session, 0u, 42u);
    FILE *payload = fopen(argv[4], "wb");
    if (!payload ||
        ds4_session_save_layer_payload(session, payload, 0u, 42u,
                                       error, sizeof(error)) != 0 ||
        fclose(payload) != 0) {
        fprintf(stderr, "cannot write batched-prefill layer payload: %s\n", error);
        return 1;
    }

    const size_t continuation_count =
        (size_t)(final_position - PREFILL_TOKENS + 1);
    uint32_t *input_tokens = malloc(continuation_count * sizeof(*input_tokens));
    if (!input_tokens) return 1;
    int selected = ds4_session_argmax(session);
    for (long position = PREFILL_TOKENS; position <= final_position; position++) {
        const size_t index = (size_t)(position - PREFILL_TOKENS);
        input_tokens[index] = (uint32_t)selected;
        if (ds4_session_eval(session, selected, error, sizeof(error)) != 0) {
            fprintf(stderr, "decode failed at position %ld: %s\n", position, error);
            return 1;
        }
        selected = ds4_session_argmax(session);
        if (position == PREFILL_TOKENS && write_logits(session, engine, argv[6])) {
            return 1;
        }
        if ((position + 1) % 256 == 0 || position == final_position) {
            fprintf(stderr, "batched-prefill continuation: %ld/%ld positions\n",
                    position + 1, final_position + 1);
        }
    }
    if (write_bytes(argv[5], input_tokens,
                    continuation_count * sizeof(*input_tokens)) ||
        write_logits(session, engine, argv[7])) {
        return 1;
    }
    fprintf(stderr,
            "batched-prefill continuation captured: prefill=%d final_position=%ld "
            "final_argmax=%d payload_bytes=%llu\n",
            PREFILL_TOKENS, final_position, selected,
            (unsigned long long)payload_bytes);

    free(input_tokens);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
