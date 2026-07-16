#!/usr/bin/env python3
"""Model-free tests for the direct dense-mixed attention candidate."""

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
        dense_mixed_direct_ablation=True,
        stats_only=False,
    )


class DenseMixedDirectTests(unittest.TestCase):
    def test_ablation_compares_gathered_with_direct(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "dense_mixed_direct"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_candidate_enables_direct_route(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "dense_mixed_direct", runtime_stats=False
        )
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", reference)
        self.assertEqual(candidate["DS4_METAL_DENSE_MIXED_DIRECT"], "1")
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT_TRACE", candidate)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_commands_are_metal_and_uninstrumented(self):
        reference = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        candidate = comparison.command_text(
            args(), "dense_mixed_direct", runtime_stats=False
        )
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", reference)
        self.assertIn("DS4_METAL_DENSE_MIXED_DIRECT=1", candidate)
        self.assertIn("--backend metal", candidate)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT_TRACE", candidate)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {"pair": 1, "mode": "dense_mixed_direct", "generation_tps": 11.0},
            {"pair": 2, "mode": "dense_mixed_direct", "generation_tps": 12.0},
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "dense_mixed_direct")
        )
        self.assertEqual(summary["comparison"], "dense_mixed_direct_ablation")
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Dense-Mixed Direct Attention Ablation", report)
        self.assertIn("Direct attention delta", report)

    def test_kernel_reads_ring_and_compressed_cache_directly(self):
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        shader = (ROOT / "metal/dsv4_misc.metal").read_text(encoding="utf-8")

        self.assertIn('getenv("DS4_METAL_DENSE_MIXED_DIRECT")', host)
        self.assertIn(
            "kernel_dsv4_dense_mixed_attention_heads8_rb16_direct",
            host,
        )
        self.assertIn(
            "kernel void kernel_dsv4_dense_mixed_attention_heads8_rb16_direct",
            shader,
        )
        self.assertIn(
            "(args.raw_start + i + r) % args.raw_cap",
            shader,
        )
        self.assertIn("for (uint i = 0; i < args.n_comp; i += 16u)", shader)
        self.assertIn("inverse_rope == NULL", host)


if __name__ == "__main__":
    unittest.main()
