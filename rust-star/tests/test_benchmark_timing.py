"""Exercise the actual benchmark loop with deterministic eval/sampling costs."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class BenchmarkTimingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "requires a C compiler")
    def test_prefill_and_decode_charge_sampling_and_commit_outputs(self):
        source = (Path(__file__).resolve().parents[2] / "ds4_bench.c").read_text()
        prefill = source[source.index("        const double prefill_t0 ="):
                         source.index("        const int prefill_tokens =")]
        decode = source[source.index("        const double gen_t0 ="):
                        source.index("        if (cfg.show_output && gen_token_buf")]
        harness = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { int pos; } Session;
static double ticks;
static int selections, evaluations, inputs[3], syncs;
static double bench_now_sec(void) { return ticks; }
static int ds4_session_sync(Session *s, void *p, char *e, size_t n) {
    (void)p; (void)e; (void)n; ticks += 20; syncs++; s->pos = 32768; return 0;
}
static int ds4_session_pos(Session *s) { return s->pos; }
static int ds4_session_ctx(Session *s) { (void)s; return 33000; }
static int ds4_session_argmax_excluding(Session *s, int eos) {
    (void)eos; int tokens[] = {85166,14,9,22}; selections++; ticks += 1;
    return tokens[s->pos - 32768];
}
static int ds4_session_eval(Session *s, int token, char *e, size_t n) {
    (void)e; (void)n; inputs[evaluations++] = token; ticks += 10; s->pos++; return 0;
}
static int write_frontier_logits_json(void *c, void *e, Session *s, int f, int p) {
    (void)c; (void)e; (void)s; (void)f; (void)p; assert(0); return 1;
}
int main(int argc, char **argv) {
    assert(argc == 2); int count = atoi(argv[1]);
    Session state = {0}, *session = &state; void *engine = NULL;
    struct { int gen_tokens, show_output; const char *dump_frontier_logits_dir; }
        cfg = {count, 1, NULL};
    int prefix = 0, eos = 0, frontier = 32768, rc = 0; char err[32] = "";
    do {
'''+prefill+decode+r'''
        assert(rc == 0 && syncs == 1 && evaluations == count);
        assert(selections == (count ? count+1 : 0));
        assert(prefill_sec == (count ? 21 : 20));
        assert(gen_done == count && gen_token_count == count);
        assert(gen_t1-gen_t0 == count*11);
        assert(gen_first_sec == (count ? 11 : 0));
        assert(gen_steady_sec == (count ? (count-1)*11 : 0));
        if (count) {
            assert(inputs[0] == 85166 && inputs[1] == 14 && inputs[2] == 9);
            assert(gen_token_buf[0] == 14 && gen_token_buf[1] == 9 && gen_token_buf[2] == 22);
            assert(next_token == 22);
        }
        free(gen_token_buf);
    } while (0);
    assert(rc == 0); return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); (path / "test.c").write_text(harness)
            subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                            str(path / "test.c"), "-o", str(path / "test")], check=True)
            for count in (0, 3):
                subprocess.run([str(path / "test"), str(count)], check=True)
