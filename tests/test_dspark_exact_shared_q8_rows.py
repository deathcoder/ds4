#!/usr/bin/env python3
"""Model-free tests for exact shared-expert Q8 proposal rows."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_comparison as comparison  # noqa: E402


def args():
    return SimpleNamespace(
        binary=Path("/tmp/ds4"),
        model=Path("/tmp/base.gguf"),
        prompt_file=Path("/tmp/prompt.txt"),
        dspark_model=Path("/tmp/dspark.gguf"),
        ctx=256,
        tokens=64,
        fast_verifier=False,
        serial_ffn_ablation=False,
        attention_pre_ablation=False,
        attention_suffix_ablation=False,
        compressor_pair_nr4_ablation=False,
        indexed_attention_rb16_promotion_ablation=False,
        exact_q8_rows_ablation=False,
        attention_inverse_rope_fusion_ablation=False,
        exact_prefix_checkpoint_ablation=False,
        metal_drafter_ablation=False,
        exact_attention_row_views_ablation=False,
        exact_attention_output_nr4_ablation=False,
        exact_attention_output_nr8_ablation=False,
        dense_mixed_direct_ablation=False,
        exact_routed_moe_hybrid_ablation=False,
        exact_q2_down_batch_ablation=False,
        exact_shared_q8_rows_ablation=True,
        stats_only=False,
    )


class ExactSharedQ8RowsTests(unittest.TestCase):
    def test_ablation_compares_default_with_shared_q8_rows(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("legacy_shared_q8_rows", "default_exact"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_legacy_reference_disables_shared_q8_rows(self):
        legacy = comparison.clean_dspark_env(
            "legacy_shared_q8_rows", runtime_stats=False
        )
        promoted = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        self.assertEqual(legacy["DS4_DSPARK_EXACT_SHARED_Q8_ROWS"], "0")
        self.assertNotIn("DS4_DSPARK_EXACT_SHARED_Q8_ROWS", promoted)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", promoted)

    def test_commands_are_metal_and_uninstrumented(self):
        legacy = comparison.command_text(
            args(), "legacy_shared_q8_rows", runtime_stats=False
        )
        promoted = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        self.assertIn("DS4_DSPARK_EXACT_SHARED_Q8_ROWS=0", legacy)
        self.assertNotIn("DS4_DSPARK_EXACT_SHARED_Q8_ROWS", promoted)
        self.assertIn("--backend metal", promoted)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", promoted)

    def test_summary_reports_paired_gain(self):
        rows = [
            {
                "pair": 1,
                "mode": "legacy_shared_q8_rows",
                "generation_tps": 10.0,
            },
            {
                "pair": 1,
                "mode": "default_exact",
                "generation_tps": 11.0,
            },
            {
                "pair": 2,
                "mode": "default_exact",
                "generation_tps": 12.0,
            },
            {
                "pair": 2,
                "mode": "legacy_shared_q8_rows",
                "generation_tps": 10.0,
            },
        ]
        summary = comparison.summarize(
            rows, ("legacy_shared_q8_rows", "default_exact")
        )
        self.assertEqual(
            summary["comparison"], "exact_shared_q8_rows_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Shared-Expert Q8 Rows Promotion Confirmation", report)
        self.assertIn("Promoted shared Q8 rows delta", report)

    def test_kernel_reuses_shared_weights_at_widths_two_through_five(self):
        source = (ROOT / "metal" / "dense.metal").read_text(encoding="utf-8")
        self.assertIn(
            "kernel_dsv4_shared_gate_up_swiglu_q8_0_exact_rows_impl", source
        )
        for width in range(2, 6):
            self.assertIn(f"exact_rows_impl<{width}>;", source)
        self.assertIn("float sumg[NROWS][NR0]", source)
        self.assertIn("float sumu[NROWS][NR0]", source)

    def test_default_row_loop_is_retained_as_fallback(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        self.assertIn("metal_graph_exact_shared_q8_rows_enabled()", source)
        self.assertIn("enabled = !value || !value[0]", source)
        self.assertIn(
            "ds4_gpu_shared_gate_up_swiglu_q8_0_exact_rows_tensor", source
        )
        self.assertIn("for (uint32_t row = 0; ok && row < n_tokens; row++)", source)

    def test_correctness_matrix_requires_both_stage_traces(self):
        source = (ROOT / "tests" / "dspark_gpu_candidates_correctness.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("DS4_TEST_DSPARK_EXACT_SHARED_Q8_ROWS", source)
        self.assertIn("DS4_DSPARK_EXACT_SHARED_Q8_ROWS_TRACE=1", source)
        self.assertIn("stage=(gate_up|down) result=pass", source)


if __name__ == "__main__":
    unittest.main()
