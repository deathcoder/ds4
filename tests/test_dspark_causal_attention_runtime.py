#!/usr/bin/env python3
"""Model-free safety tests for the one-layer causal-attention runtime."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_causal_attention_runtime as runtime  # noqa: E402


def exact_records(proposed=5, accepted=3):
    return (
        f"ds4: DSpark causal attention runtime proposed={proposed} layer=41 "
        "attempts=1 successes=1 result=pass\n"
        "ds4: DSpark proposal slab observer layer=41 ratio=128 "
        f"proposed={proposed} raw_rows={proposed}/{proposed} "
        f"attn_prefixes={proposed}/{proposed} index_prefixes=0/0 "
        f"counters={proposed}/{proposed} result=exact\n"
        "ds4: DSpark proposal slab publication layer=41 ratio=128 "
        f"proposed={proposed} accepted={accepted} "
        "prepared=exact result=exact\n"
    ).encode()


class DSparkCausalAttentionRuntimeTests(unittest.TestCase):
    def test_runner_freezes_one_layer_and_two_schedules(self):
        self.assertEqual(runtime.LAYER, 41)
        self.assertEqual(
            runtime.SCHEDULES,
            (("scheduled", "0.75"), ("fixed-k5", "0")),
        )
        env = runtime.clean_env(True)
        self.assertEqual(env["DS4_DSPARK_CAUSAL_ATTN_RUNTIME_LAYER"], "41")
        self.assertEqual(env["DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER"], "41")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS"], "1")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)

    def test_baseline_clears_every_ds4_setting(self):
        old = runtime.os.environ.get("DS4_TEST_SENTINEL")
        runtime.os.environ["DS4_TEST_SENTINEL"] = "1"
        try:
            self.assertFalse(any(
                key.startswith("DS4_") for key in runtime.clean_env()
            ))
        finally:
            if old is None:
                runtime.os.environ.pop("DS4_TEST_SENTINEL", None)
            else:
                runtime.os.environ["DS4_TEST_SENTINEL"] = old

    def test_parser_accepts_exact_runtime_and_partial_publication(self):
        parsed = runtime.parse_diagnostics(exact_records())
        self.assertEqual(parsed["runtime"], [{"proposed": 5}])
        self.assertEqual(parsed["publication"][0]["accepted"], 3)

    def test_parser_rejects_runtime_fallback(self):
        data = exact_records().replace(
            b"attempts=1 successes=1 result=pass",
            b"attempts=1 successes=0 result=fallback",
        )
        with self.assertRaisesRegex(RuntimeError, "fell back"):
            runtime.parse_diagnostics(data)

    def test_parser_rejects_missing_state_record(self):
        data = exact_records().replace(
            b"ds4: DSpark proposal slab observer", b"ignored observer"
        )
        with self.assertRaisesRegex(RuntimeError, "no proposal-slab"):
            runtime.parse_diagnostics(data)

    def test_source_keeps_candidate_opt_in_and_layer_41_only(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        gate_start = source.index("static int dspark_causal_attn_runtime_layer")
        gate_end = source.index(
            "static bool dspark_causal_attn_runtime_encode", gate_start
        )
        gate = source[gate_start:gate_end]
        self.assertIn("DS4_DSPARK_GPU_RUNTIME", gate)
        self.assertIn("DS4_DSPARK_MULTI_COMMIT", gate)
        self.assertIn("DS4_DSPARK_CAUSAL_ATTN_RUNTIME_LAYER", gate)
        self.assertIn("layer != 41", gate)

    def test_source_stages_state_before_exact_vec_and_suffix(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        state_return = source.index(
            "part == METAL_GRAPH_DECODE_LAYER_ATTENTION_STATE"
        )
        attention = source.index("const char *exact_attention_stage", state_return)
        self.assertLess(state_return, attention)
        helper_start = source.index(
            "static bool dspark_causal_attn_runtime_encode"
        )
        helper_end = source.index(
            "static bool dspark_causal_attn_head_shadow_alloc", helper_start
        )
        helper = source[helper_start:helper_end]
        self.assertIn(
            "ds4_gpu_attention_decode_mixed_vec_query_heads_tensor", helper
        )
        self.assertIn("g->batch_heads", helper)
        self.assertIn("ds4_gpu_rope_tail_tensor", helper)
        self.assertIn("metal_graph_exact_attention_suffix_batch", helper)


if __name__ == "__main__":
    unittest.main()
