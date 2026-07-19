#!/usr/bin/env python3
"""Model-free tests for exact compressor projection prebatching."""

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
        exact_shared_q8_rows_ablation=False,
        exact_compressor_pre_batch_ablation=True,
        stats_only=False,
    )


class ExactCompressorPreBatchTests(unittest.TestCase):
    def test_ablation_compares_default_with_prebatch_candidate(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("legacy_compressor_pre_batch", "default_exact"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_legacy_reference_disables_prebatch(self):
        legacy = comparison.clean_dspark_env(
            "legacy_compressor_pre_batch", runtime_stats=False
        )
        promoted = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        self.assertEqual(
            legacy["DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH"], "0"
        )
        self.assertNotIn("DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH", promoted)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", promoted)

    def test_commands_are_metal_and_uninstrumented(self):
        legacy = comparison.command_text(
            args(), "legacy_compressor_pre_batch", runtime_stats=False
        )
        promoted = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        self.assertIn("DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=0", legacy)
        self.assertNotIn("DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH", promoted)
        self.assertIn("--backend metal", promoted)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", promoted)
        self.assertNotIn("TRACE", promoted)

    def test_summary_reports_paired_gain(self):
        rows = [
            {
                "pair": 1,
                "mode": "legacy_compressor_pre_batch",
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
                "mode": "legacy_compressor_pre_batch",
                "generation_tps": 10.0,
            },
        ]
        summary = comparison.summarize(
            rows, ("legacy_compressor_pre_batch", "default_exact")
        )
        self.assertEqual(
            summary["comparison"], "exact_compressor_pre_batch_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Prebatch Promotion Confirmation", report)
        self.assertIn("Promoted compressor prebatch delta", report)

    def test_paired_f16_projection_supports_exact_widths(self):
        source = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        self.assertIn("n_tok == 0 || n_tok > 5", source)
        self.assertIn("mv_args.ne11 = (int32_t)n_tok", source)
        self.assertIn("(NSUInteger)n_tok,\n                                              1)", source)

    def test_prebatch_preserves_serial_compressor_updates(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        self.assertIn(
            "metal_graph_exact_compressor_pre_batch_prepare", source
        )
        self.assertIn("enabled = !value || !value[0]", source)
        self.assertIn("g->batch_index_comp_kv", source)
        self.assertIn("g->batch_index_comp_sc", source)
        self.assertIn("g->exact_comp_kv_pre", source)
        self.assertIn("g->exact_index_comp_kv_pre", source)
        self.assertIn("ok ? \"ok\" : \"fallback\"", source)

    def test_correctness_matrix_requires_both_compression_ratios(self):
        source = (ROOT / "tests" / "dspark_gpu_candidates_correctness.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("DS4_TEST_DSPARK_EXACT_COMPRESSOR_PRE_BATCH", source)
        self.assertIn("DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH_TRACE=1", source)
        self.assertIn("ratio=128 result=ok", source)
        self.assertIn("ratio=4 result=ok", source)


if __name__ == "__main__":
    unittest.main()
