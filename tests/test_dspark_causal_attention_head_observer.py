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


def exact_vec_row(proposed, row):
    return (
        b"ds4: DSpark causal attention vec control layer=41 ratio=128 "
        + f"proposed={proposed} row={row} ".encode()
        + b"max=0 rms=0 rel_l2=0 max_ulp=0 result=exact\n"
    )


def profile_row(mode, tokens, stage, elapsed):
    return (
        b"ds4: Metal FlashAttention prefill stage "
        + f"mode={mode} tokens={tokens} comp=64 keys=192 ".encode()
        + b"heads=16 dim=512 window=0 ratio=128 "
        + f"{stage}={elapsed:.3f} ms\n".encode()
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

        legacy = observer.clean_env(41, serial_legacy=True)
        self.assertEqual(
            legacy["DS4_METAL_DENSE_MIXED_GATHERED_LEGACY"], "1"
        )
        profile = observer.clean_env(41, vec_profile=True)
        self.assertEqual(profile["DS4_DSPARK_CAUSAL_ATTN_VEC_PROFILE"], "1")
        self.assertEqual(profile["DS4_METAL_FLASH_ATTN_GATHERED_PROFILE"], "1")
        self.assertEqual(profile["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"], "41")

    def test_parser_accepts_complete_exact_proposals(self):
        data = b"".join(exact_row(2, row) for row in range(2))
        data += b"".join(exact_row(5, row) for row in range(5))
        parsed = observer.parse_head_observer(data)
        self.assertEqual(parsed, [
            {"proposed": 2, "rows": 2, "results": ["exact", "exact"]},
            {"proposed": 5, "rows": 5,
             "results": ["exact", "exact", "exact", "exact", "exact"]},
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

    def test_localization_parser_retains_drift(self):
        data = exact_row(2, 0) + exact_row(2, 1).replace(
            b"result=exact", b"result=drift"
        )
        parsed = observer.parse_head_observer(data, require_exact=False)
        self.assertEqual(parsed[0]["results"], ["exact", "drift"])

    def test_rowwise_control_parser(self):
        data = (
            b"ds4: DSpark causal attention rowwise control layer=41 ratio=128 "
            b"proposed=2 row=0 serial_max=1e-6 serial_rms=1e-7 "
            b"serial_rel_l2=1e-7 batch_max=0 batch_rms=0 batch_rel_l2=0 "
            b"serial_result=drift batch_result=exact\n"
        )
        parsed = observer.parse_rowwise_control(data)
        self.assertEqual(parsed[0]["serial_result"], "drift")
        self.assertEqual(parsed[0]["batch_result"], "exact")

    def test_vec_control_requires_exact_when_promoting(self):
        data = exact_vec_row(2, 0) + exact_vec_row(2, 1)
        parsed = observer.parse_vec_control(data, require_exact=True)
        self.assertEqual([item["result"] for item in parsed], ["exact", "exact"])
        with self.assertRaisesRegex(RuntimeError, "causal vec attention drift"):
            observer.parse_vec_control(
                data.replace(b"result=exact", b"result=drift", 1),
                require_exact=True,
            )

    def test_vec_profile_pairs_width_with_serial_calls(self):
        data = b""
        for _ in range(2):
            for stage, elapsed in zip(observer.PROFILE_STAGES, (1.0, 2.0, 3.0)):
                data += profile_row("fused_gather_decode", 1, stage, elapsed)
        for stage, elapsed in zip(observer.PROFILE_STAGES, (1.5, 2.5, 3.5)):
            data += profile_row("causal_vec_query", 2, stage, elapsed)
        parsed = observer.parse_vec_profile(data)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["width"], 2)
        self.assertEqual(parsed[0]["serial_total_ms"], 12.0)
        self.assertEqual(parsed[0]["candidate_total_ms"], 7.5)
        self.assertAlmostEqual(parsed[0]["candidate_serial_ratio"], 0.625)

    def test_vec_profile_rejects_missing_serial_call(self):
        data = b"".join(
            profile_row("fused_gather_decode", 1, stage, 1.0)
            for stage in observer.PROFILE_STAGES
        )
        data += b"".join(
            profile_row("causal_vec_query", 2, stage, 1.0)
            for stage in observer.PROFILE_STAGES
        )
        with self.assertRaisesRegex(RuntimeError, "follows 1 serial calls"):
            observer.parse_vec_profile(data)

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
        self.assertIn(
            "ds4_gpu_attention_decode_mixed_vec_query_heads_tensor", encode
        )
        self.assertIn("g->causal_attn_rowwise_heads", encode)
        self.assertIn("g->causal_attn_vec_heads", encode)

    def test_vec_kernel_specialization_is_query_indexed_only(self):
        metal = (ROOT / "metal/flash_attn.metal").read_text(encoding="utf-8")
        self.assertIn("bool QUERY_KV = false", metal)
        self.assertIn("QUERY_KV ? iq1*args.nb12 : 0", metal)
        self.assertIn(
            'host_name("kernel_flash_attn_ext_vec_query_kv_f16_dk512_dv512")',
            metal,
        )

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
