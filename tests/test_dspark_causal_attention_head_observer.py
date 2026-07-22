#!/usr/bin/env python3
"""Model-free safety and parser tests for causal attention head shadowing."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_causal_attention_head_observer as observer  # noqa: E402


def exact_row(proposed, row):
    return (
        b"ds4: DSpark causal attention head observer layer=41 ratio=128 "
        + f"proposed={proposed} row={row} ".encode()
        + b"max=0 rms=0 rel_l2=0 max_ulp=0 first=65536 "
        + b"batch=0x0p+0 serial=0x0p+0 result=exact\n"
    )


class DSparkCausalAttentionHeadObserverTests(unittest.TestCase):
    def test_runner_freezes_diagnostic_modes(self):
        self.assertEqual(observer.LAYER, 41)
        self.assertEqual(
            observer.SCHEDULES,
            (("scheduled", "0.75"), ("fixed-k5", "0")),
        )
        env = observer.clean_env(41)
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS"], "1")
        self.assertEqual(
            env["DS4_DSPARK_CAUSAL_ATTN_HEAD_OBSERVER_LAYER"], "41"
        )
        self.assertEqual(env["DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER"], "41")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)

    def test_parser_accepts_complete_exact_proposals(self):
        data = b"".join(exact_row(2, row) for row in range(2))
        data += b"".join(exact_row(5, row) for row in range(5))
        parsed = observer.parse_head_observer(data)
        self.assertEqual(parsed, [
            {"proposed": 2, "rows": 2},
            {"proposed": 5, "rows": 5},
        ])

    def test_parser_rejects_missing_row(self):
        data = exact_row(3, 0) + exact_row(3, 2)
        with self.assertRaisesRegex(RuntimeError, "non-contiguous"):
            observer.parse_head_observer(data)

    def test_parser_rejects_bounded_or_drift(self):
        for result in (b"bounded", b"drift"):
            data = exact_row(2, 0) + exact_row(2, 1).replace(
                b"result=exact", b"result=" + result
            )
            with self.assertRaisesRegex(RuntimeError, result.decode()):
                observer.parse_head_observer(data)

    def test_parser_rejects_fallback(self):
        data = (
            b"ds4: DSpark causal attention head observer layer=41 "
            b"ratio=128 proposed=5 result=fallback\n"
        )
        with self.assertRaisesRegex(RuntimeError, "fell back"):
            observer.parse_head_observer(data)

    def test_source_requires_safety_gate_and_ratio128(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        start = source.index("static int dspark_causal_attn_head_observer_layer")
        end = source.index("static bool dspark_causal_attn_head_shadow_alloc", start)
        gate = source[start:end]
        for name in (
            "DS4_DSPARK_GPU_RUNTIME",
            "DS4_DSPARK_MULTI_COMMIT",
            "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS",
            "DS4_DSPARK_CAUSAL_ATTN_HEAD_OBSERVER_LAYER",
        ):
            self.assertIn(name, gate)
        encode_start = source.index(
            "static bool dspark_causal_attn_head_shadow_encode"
        )
        encode_end = source.index(
            "static bool dspark_causal_attn_head_row_bounded", encode_start
        )
        encode = source[encode_start:encode_end]
        self.assertIn("ds4_layer_compress_ratio(il) != 128u", encode)
        self.assertIn("ds4_gpu_attention_decode_mixed_batch_heads_tensor", encode)

    def test_serial_capture_precedes_inverse_rope(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        start = source.index("const bool observe_causal_attn =", 15000)
        end = source.index(
            "DS4_METAL_PROFILE_EXACT_TAIL_STAGE(\"inverse_rope\")", start
        )
        body = source[start:end]
        capture = body.index("g->causal_attn_serial_heads")
        inverse_rope = body.index("ds4_gpu_rope_tail_tensor(g->heads")
        self.assertLess(capture, inverse_rope)

    def test_shadow_outputs_are_not_runtime_inputs(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        report_start = source.index(
            "static bool dspark_causal_attn_head_shadow_report"
        )
        report_end = source.index(
            "static void dspark_proposal_slab_shadow_free", report_start
        )
        report = source[report_start:report_end]
        self.assertNotIn("s->logits", report)
        self.assertNotIn("return false", report)
        call = source.index("(void)dspark_causal_attn_head_shadow_report")
        self.assertGreater(call, report_end)


if __name__ == "__main__":
    unittest.main()
