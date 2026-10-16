from pathlib import Path
w=Path('rust-star/.work/native-32k-v31');src=w/'oracle-transcript-source';s=(src/'ds4.c').read_text();needle='static void metal_graph_debug_dump_tensor(\n';pos=s.index(needle);s=s[:pos]+'''/* Diagnostic-only snapshots: enqueue a blit without submitting/restarting
 * the inference batch. Read them only after session_sync has completed. */
typedef struct {
    ds4_gpu_tensor *tensor;
    uint64_t words;
    char path[1024];
} rust_star_deferred_dump;
static rust_star_deferred_dump rust_star_deferred_dumps[43];
static unsigned rust_star_deferred_count;
int rust_star_oracle_flush_deferred_dumps(void) {
    if (!ds4_gpu_synchronize()) return 0;
    for (unsigned i = 0; i < rust_star_deferred_count; i++) {
        rust_star_deferred_dump *d = &rust_star_deferred_dumps[i];
        float *data = xmalloc(d->words*sizeof(float));
        if (!ds4_gpu_tensor_read(d->tensor, 0, data, d->words*sizeof(float)) ||
            !write_f32_binary_file(d->path, data, d->words)) return 0;
        free(data);
        ds4_gpu_tensor_free(d->tensor);
    }
    rust_star_deferred_count = 0;
    return 1;
}

'''+s[pos:];needle='''    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;
''';s=s.replace(needle,needle+'''
    if (getenv("RUST_STAR_ORACLE_DEFER_DUMPS")) {
        if (rust_star_deferred_count >= 43u) abort();
        rust_star_deferred_dump *d = &rust_star_deferred_dumps[rust_star_deferred_count++];
        d->words = n_f32;
        d->tensor = ds4_gpu_tensor_alloc(n_f32*sizeof(float));
        if (!d->tensor || !ds4_gpu_tensor_copy(d->tensor, 0, t, 0, n_f32*sizeof(float))) abort();
        snprintf(d->path, sizeof(d->path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        return;
    }
''',1);(src/'ds4_deferred.c').write_text(s)
s=(src/'capture_32k_transcript.c').read_text().replace('int main(int argc, char **argv) {','extern int rust_star_oracle_flush_deferred_dumps(void);\n\nint main(int argc, char **argv) {',1).replace('''    free(raw_tokens);
    free(prompt_tokens);
    if (argc == 9''','''    if (!rust_star_oracle_flush_deferred_dumps()) return 1;
    free(raw_tokens);
    free(prompt_tokens);
    if (argc == 9''',1);(src/'capture_deferred.c').write_text(s)
