#include "ds4.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *dspark_path;
    const char *prompt_path;
    const char *system;
    int ctx_size;
    int warmups;
    int runs;
} config;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-warm-prefill-bench --model FILE --prompt-file FILE [options]\n"
            "  --dspark FILE   Load a DSpark sidecar\n"
            "  --ctx N         Session context size (default 256)\n"
            "  --warmups N     Recorded conditioning sessions (default 1)\n"
            "  --runs N        Measured fresh sessions (default 3)\n"
            "  --system TEXT   Chat system prompt\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-warm-prefill-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_positive(const char *text, const char *opt) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!text[0] || *end || value <= 0 || value > INT32_MAX) {
        fprintf(stderr, "ds4-warm-prefill-bench: invalid %s: %s\n", opt, text);
        exit(2);
    }
    return (int)value;
}

static int parse_nonnegative(const char *text, const char *opt) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!text[0] || *end || value < 0 || value > INT32_MAX) {
        fprintf(stderr, "ds4-warm-prefill-bench: invalid %s: %s\n", opt, text);
        exit(2);
    }
    return (int)value;
}

static config parse_options(int argc, char **argv) {
    config cfg = {
        .system = "You are a helpful assistant",
        .ctx_size = 256,
        .warmups = 1,
        .runs = 3,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dspark")) {
            cfg.dspark_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--system")) {
            cfg.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.ctx_size = parse_positive(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--warmups")) {
            cfg.warmups = parse_nonnegative(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--runs")) {
            cfg.runs = parse_positive(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-warm-prefill-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!cfg.model_path || !cfg.prompt_path) {
        usage(stderr);
        exit(2);
    }
    return cfg;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-warm-prefill-bench: failed to open %s: %s\n",
                path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) exit(1);
    const long size = ftell(fp);
    if (size < 0 || fseek(fp, 0, SEEK_SET) != 0) exit(1);
    char *text = malloc((size_t)size + 1u);
    if (!text) exit(1);
    if (fread(text, 1, (size_t)size, fp) != (size_t)size) exit(1);
    if (fclose(fp) != 0) exit(1);
    text[size] = '\0';
    return text;
}

static uint64_t hash_bytes(const void *data, size_t len) {
    const unsigned char *bytes = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int run_session(ds4_engine *engine,
                       const ds4_tokens *prompt,
                       int ctx_size,
                       const char *kind,
                       int run_index,
                       float *logits,
                       int vocab_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4-warm-prefill-bench: failed to create session\n");
        return 1;
    }
    char err[256] = {0};
    const double started = now_sec();
    const int sync_rc = ds4_session_sync(session, prompt, err, sizeof(err));
    const double seconds = now_sec() - started;
    if (sync_rc != 0) {
        fprintf(stderr, "ds4-warm-prefill-bench: sync failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab_size) != vocab_size) {
        fprintf(stderr, "ds4-warm-prefill-bench: failed to copy target logits\n");
        ds4_session_free(session);
        return 1;
    }
    const int argmax = ds4_session_argmax(session);
    const uint64_t hash = hash_bytes(logits, (size_t)vocab_size * sizeof(logits[0]));
    printf("%s,%d,%d,%.9f,%.6f,%d,%016" PRIx64 "\n",
           kind,
           run_index,
           prompt->len,
           seconds,
           seconds > 0.0 ? (double)prompt->len / seconds : 0.0,
           argmax,
           hash);
    fflush(stdout);
    ds4_session_free(session);
    return 0;
}

int main(int argc, char **argv) {
    const config cfg = parse_options(argc, argv);
    ds4_engine_options options = {
        .model_path = cfg.model_path,
        .dspark_path = cfg.dspark_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;

    char *text = read_file(cfg.prompt_path);
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    free(text);
    if (prompt.len <= 0 || prompt.len >= cfg.ctx_size) {
        fprintf(stderr,
                "ds4-warm-prefill-bench: rendered prompt has %d tokens for ctx %d\n",
                prompt.len, cfg.ctx_size);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    const int vocab_size = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab_size * sizeof(logits[0]));
    if (!logits) return 1;
    printf("kind,run,prompt_tokens,prefill_seconds,prefill_tps,argmax,logits_hash\n");
    fflush(stdout);

    int rc = 0;
    for (int i = 0; i < cfg.warmups && rc == 0; i++) {
        rc = run_session(engine, &prompt, cfg.ctx_size,
                         i == 0 ? "cold" : "conditioning",
                         i + 1, logits, vocab_size);
    }
    for (int i = 0; i < cfg.runs && rc == 0; i++) {
        rc = run_session(engine, &prompt, cfg.ctx_size, "warm",
                         i + 1, logits, vocab_size);
    }

    free(logits);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
