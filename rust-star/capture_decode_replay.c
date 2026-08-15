/* Capture DwarfStar's one-token decoder arithmetic over an external token
 * prefix. This is a diagnostic oracle for separating decoder-state bugs from
 * the deliberately different batched prefill kernels. */

#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) return NULL;
    long bytes = ftell(fp);
    if (bytes < 0 || fseek(fp, 0, SEEK_SET) != 0) return NULL;
    char *text = malloc((size_t)bytes + 1u);
    if (!text) return NULL;
    if (fread(text, 1, (size_t)bytes, fp) != (size_t)bytes || fclose(fp) != 0) {
        free(text);
        return NULL;
    }
    text[bytes] = '\0';
    return text;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL PROMPT TOKENS OUTPUT\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    long token_count = strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || token_count < 1 || token_count > 1048576) {
        fprintf(stderr, "invalid token count\n");
        return 2;
    }
    char *text = read_file(argv[2]);
    if (!text) {
        fprintf(stderr, "cannot read prompt: %s\n", strerror(errno));
        return 1;
    }
    ds4_engine_options options = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = (int)token_count + 1,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, text, &prompt);
    free(text);
    if (prompt.len < token_count) {
        fprintf(stderr, "prompt has %d tokens, need %ld\n", prompt.len, token_count);
        return 1;
    }
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, (uint32_t)token_count + 1u) != 0) return 1;
    ds4_tokens first = {.v = prompt.v, .len = 1, .cap = 1};
    char error[256] = "";
    if (ds4_session_sync(session, &first, error, sizeof(error)) != 0) {
        fprintf(stderr, "initial prefill failed: %s\n", error);
        return 1;
    }
    for (long index = 1; index < token_count; index++) {
        if (ds4_session_eval(session, prompt.v[index], error, sizeof(error)) != 0) {
            fprintf(stderr, "decode replay failed at %ld: %s\n", index, error);
            return 1;
        }
    }
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits || ds4_session_copy_logits(session, logits, vocab) != vocab) return 1;
    FILE *output = fopen(argv[4], "wb");
    if (!output || fwrite(logits, sizeof(*logits), (size_t)vocab, output) != (size_t)vocab ||
        fclose(output) != 0) {
        fprintf(stderr, "cannot write logits\n");
        return 1;
    }
    fprintf(stderr, "decode replay: tokens=%ld argmax=%d\n", token_count,
            ds4_session_argmax(session));
    free(logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
