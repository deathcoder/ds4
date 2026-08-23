/* Capture the exact DwarfStar state immediately before one retained decoder
 * step, then execute that step and write its logits.  The layer payload is the
 * engine-owned, portable representation used by distributed inference and the
 * server KV cache; unlike ad-hoc tensor hooks it includes every live raw ring,
 * compressed history, and recurrent compressor state. */

#include "ds4.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_token_ids(const char *path, int **tokens_out, uint32_t *count_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open token file: %s\n", strerror(errno));
        return 1;
    }
    size_t cap = 16384u;
    size_t count = 0u;
    int *tokens = malloc(cap * sizeof(*tokens));
    if (!tokens) {
        fclose(fp);
        return 1;
    }
    int first = fgetc(fp);
    while (first != EOF && (first == ' ' || first == '\t' ||
                            first == '\r' || first == '\n')) {
        first = fgetc(fp);
    }
    if (first != '[' && first != EOF) ungetc(first, fp);
    for (;;) {
        long value = 0;
        if (first == '[') {
            if (fscanf(fp, " %ld", &value) != 1) break;
        } else {
            char line[4096];
            if (!fgets(line, sizeof(line), fp)) break;
            char *end = NULL;
            errno = 0;
            value = strtol(line, &end, 10);
            if (errno != 0 || end == line) value = -1;
        }
        if (value < 0 || value > INT32_MAX) {
            fprintf(stderr, "invalid token %zu\n", count + 1u);
            free(tokens);
            fclose(fp);
            return 1;
        }
        if (count == cap) {
            cap *= 2u;
            int *grown = realloc(tokens, cap * sizeof(*tokens));
            if (!grown) {
                free(tokens);
                fclose(fp);
                return 1;
            }
            tokens = grown;
        }
        tokens[count++] = (int)value;
        if (first == '[') {
            int delimiter = fgetc(fp);
            while (delimiter == ' ' || delimiter == '\t' ||
                   delimiter == '\r' || delimiter == '\n') {
                delimiter = fgetc(fp);
            }
            if (delimiter == ']') break;
            if (delimiter != ',') {
                fprintf(stderr, "invalid token-list delimiter\n");
                free(tokens);
                fclose(fp);
                return 1;
            }
        }
    }
    if (ferror(fp) || fclose(fp) != 0 || count > UINT32_MAX) {
        free(tokens);
        return 1;
    }
    *tokens_out = tokens;
    *count_out = (uint32_t)count;
    return 0;
}

static int write_logits(ds4_session *session, ds4_engine *engine, const char *path) {
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits || ds4_session_copy_logits(session, logits, vocab) != vocab) {
        free(logits);
        return 1;
    }
    FILE *fp = fopen(path, "wb");
    const int failed = !fp ||
        fwrite(logits, sizeof(*logits), (size_t)vocab, fp) != (size_t)vocab ||
        fclose(fp) != 0;
    free(logits);
    if (failed) fprintf(stderr, "cannot write logits\n");
    return failed;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: %s MODEL TOKENS PREFIX_TOKENS LAYER_PAYLOAD LOGITS "
                "[LOAD_LAYER_PAYLOAD]\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    long prefix = strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || prefix < 1 || prefix > INT32_MAX - 1) {
        fprintf(stderr, "invalid prefix token count\n");
        return 2;
    }
    int *tokens = NULL;
    uint32_t token_count = 0u;
    if (read_token_ids(argv[2], &tokens, &token_count) != 0) return 1;
    if ((uint32_t)prefix >= token_count) {
        fprintf(stderr, "token file does not include the retained step token\n");
        free(tokens);
        return 1;
    }

    ds4_engine_options options = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = (int)prefix + 1,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) {
        free(tokens);
        return 1;
    }
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, (int)prefix + 1) != 0) {
        ds4_engine_close(engine);
        free(tokens);
        return 1;
    }
    char error[256] = "";
    if (argc == 7) {
        FILE *input = fopen(argv[6], "rb");
        if (!input || fseek(input, 0, SEEK_END) != 0) {
            fprintf(stderr, "cannot open retained layer payload\n");
            return 1;
        }
        const long input_bytes = ftell(input);
        if (input_bytes < 0 || fseek(input, 0, SEEK_SET) != 0 ||
            ds4_session_load_layer_payload(session, input, (uint64_t)input_bytes,
                                           tokens, (uint32_t)prefix, 0u, 42u,
                                           error, sizeof(error)) != 0 ||
            fclose(input) != 0) {
            fprintf(stderr, "cannot load retained layer payload: %s\n", error);
            return 1;
        }
        fprintf(stderr, "retained capture restored %ld-token prefix\n", prefix);
    } else {
        ds4_tokens first = {.v = tokens, .len = 1, .cap = 1};
        if (ds4_session_sync(session, &first, error, sizeof(error)) != 0) {
            fprintf(stderr, "initial prefill failed: %s\n", error);
            return 1;
        }
        for (long index = 1; index < prefix; index++) {
            if (ds4_session_eval(session, tokens[index], error, sizeof(error)) != 0) {
                fprintf(stderr, "decode replay failed at %ld: %s\n", index, error);
                return 1;
            }
            if ((index + 1) % 256 == 0 || index + 1 == prefix) {
                fprintf(stderr, "retained capture prefix: %ld/%ld tokens\n", index + 1, prefix);
            }
        }
    }
    const uint64_t payload_bytes =
        ds4_session_layer_payload_bytes(session, 0u, 42u);
    FILE *payload = fopen(argv[4], "wb");
    if (!payload ||
        ds4_session_save_layer_payload(session, payload, 0u, 42u,
                                       error, sizeof(error)) != 0 ||
        fclose(payload) != 0) {
        fprintf(stderr, "cannot write layer payload: %s\n", error);
        return 1;
    }
    if (ds4_session_eval(session, tokens[prefix], error, sizeof(error)) != 0) {
        fprintf(stderr, "retained decoder step failed: %s\n", error);
        return 1;
    }
    if (write_logits(session, engine, argv[5]) != 0) return 1;
    fprintf(stderr,
            "retained decoder capture: prefix=%ld token=%d argmax=%d payload_bytes=%llu\n",
            prefix, tokens[prefix], ds4_session_argmax(session),
            (unsigned long long)payload_bytes);

    ds4_session_free(session);
    ds4_engine_close(engine);
    free(tokens);
    return 0;
}
