#!/usr/bin/env python3
"""Model-free tests for the dense-mixed vector-prepare candidate."""

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
        dense_mixed_nwg8_ablation=False,
        dense_mixed_split_source_ablation=False,
        dense_mixed_vector_prepare_ablation=True,
        stats_only=False,
    )


class DenseMixedVectorPrepareTests(unittest.TestCase):
    def test_ablation_compares_scalar_with_vector_prepare(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "dense_mixed_vector_prepare"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_candidate_enables_vector_prepare(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "dense_mixed_vector_prepare", runtime_stats=False
        )
        self.assertNotIn("DS4_METAL_DENSE_MIXED_VECTOR_PREPARE", reference)
        self.assertEqual(
            candidate["DS4_METAL_DENSE_MIXED_VECTOR_PREPARE"], "1"
        )
        self.assertNotIn(
            "DS4_METAL_DENSE_MIXED_SPLIT_SOURCE_PARITY", candidate
        )
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_commands_are_metal_and_uninstrumented(self):
        reference = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        candidate = comparison.command_text(
            args(), "dense_mixed_vector_prepare", runtime_stats=False
        )
        self.assertNotIn("DS4_METAL_DENSE_MIXED_VECTOR_PREPARE", reference)
        self.assertIn(
            "DS4_METAL_DENSE_MIXED_VECTOR_PREPARE=1", candidate
        )
        self.assertIn("--backend metal", candidate)
        self.assertNotIn("SPLIT_SOURCE_PARITY", candidate)
        self.assertNotIn("VECTOR_PREPARE_TRACE", candidate)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {
                "pair": 1,
                "mode": "dense_mixed_vector_prepare",
                "generation_tps": 11.0,
            },
            {
                "pair": 2,
                "mode": "dense_mixed_vector_prepare",
                "generation_tps": 12.0,
            },
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "dense_mixed_vector_prepare")
        )
        self.assertEqual(
            summary["comparison"], "dense_mixed_vector_prepare_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Dense-Mixed Vector Prepare Ablation", report)
        self.assertIn("Vector prepare delta", report)

    def test_candidate_reuses_the_parity_proven_kernel(self):
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        shader = (ROOT / "metal/dsv4_misc.metal").read_text(encoding="utf-8")

        self.assertIn("kernel_dsv4_dense_mixed_split_source_view_f16", host)
        self.assertIn("device      half4 *dst_kv", shader)
        self.assertIn("if (need_scalar_prepare)", host)
        self.assertIn("use_vector_prepare && split_source_parity", host)
        self.assertIn("use_vector_prepare ? 128u : 32u", host)


if __name__ == "__main__":
    unittest.main()
