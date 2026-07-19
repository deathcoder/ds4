#!/usr/bin/env python3
"""Model-free tests for the exact routed-MoE hybrid candidate."""

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
        exact_routed_moe_hybrid_ablation=True,
        stats_only=False,
    )


class ExactRoutedMoEHybridTests(unittest.TestCase):
    def test_ablation_compares_default_with_hybrid(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "exact_routed_moe_hybrid"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_candidate_enables_hybrid(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "exact_routed_moe_hybrid", runtime_stats=False
        )
        self.assertNotIn("DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID", reference)
        self.assertEqual(
            candidate["DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID"], "1"
        )
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_commands_are_metal_and_uninstrumented(self):
        reference = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        candidate = comparison.command_text(
            args(), "exact_routed_moe_hybrid", runtime_stats=False
        )
        self.assertNotIn("DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID", reference)
        self.assertIn(
            "DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID=1", candidate
        )
        self.assertIn("--backend metal", candidate)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {
                "pair": 1,
                "mode": "exact_routed_moe_hybrid",
                "generation_tps": 11.0,
            },
            {
                "pair": 2,
                "mode": "exact_routed_moe_hybrid",
                "generation_tps": 12.0,
            },
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "exact_routed_moe_hybrid")
        )
        self.assertEqual(
            summary["comparison"], "exact_routed_moe_hybrid_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Routed-MoE Hybrid Ablation", report)
        self.assertIn("hybrid delta", report)

    def test_hybrid_keeps_one_row_direct_down_arithmetic(self):
        source = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        self.assertIn("const bool exact_down_rows_active", source)
        self.assertIn("for (uint32_t row = 0; ok && row < n_tokens; row++)", source)
        self.assertIn("ds4_gpu_encode_mul_mv_id_sum6(", source)
        self.assertIn("!exact_down_rows_active", source)


if __name__ == "__main__":
    unittest.main()
