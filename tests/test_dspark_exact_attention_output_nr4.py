#!/usr/bin/env python3
"""Model-free tests for the exact attention-output NR4 gate."""

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
        exact_attention_output_nr4_ablation=True,
        stats_only=False,
    )


class ExactAttentionOutputNr4Tests(unittest.TestCase):
    def test_ablation_modes_keep_default_exact_as_reference(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("legacy_attention_output_nr2", "default_exact"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_reference_disables_promoted_nr4(self):
        reference = comparison.clean_dspark_env(
            "legacy_attention_output_nr2", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        self.assertEqual(reference["DS4_DSPARK_EXACT_ATTN_OUT_NR4"], "0")
        self.assertNotIn("DS4_DSPARK_EXACT_ATTN_OUT_NR4", candidate)
        self.assertNotIn("DS4_DSPARK_EXACT_ATTN_OUT_NR4_TRACE", reference)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_commands_are_metal_and_uninstrumented(self):
        reference = comparison.command_text(
            args(), "legacy_attention_output_nr2", runtime_stats=False
        )
        candidate = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        self.assertIn("DS4_DSPARK_EXACT_ATTN_OUT_NR4=0", reference)
        self.assertNotIn("DS4_DSPARK_EXACT_ATTN_OUT_NR4", candidate)
        self.assertIn("--backend metal", candidate)
        self.assertNotIn("DS4_DSPARK_EXACT_ATTN_OUT_NR4_TRACE", reference)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {
                "pair": 1,
                "mode": "legacy_attention_output_nr2",
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
                "mode": "legacy_attention_output_nr2",
                "generation_tps": 10.0,
            },
        ]
        summary = comparison.summarize(
            rows, ("legacy_attention_output_nr2", "default_exact")
        )
        self.assertEqual(
            summary["comparison"], "exact_attention_output_nr4_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("NR4 Promotion Confirmation", report)
        self.assertIn("Promoted NR4 delta", report)

    def test_kernels_preserve_q8_reduction_shape(self):
        metal_host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        moe = (ROOT / "metal/moe.metal").read_text(encoding="utf-8")
        hc = (ROOT / "metal/dsv4_hc.metal").read_text(encoding="utf-8")

        self.assertIn(
            'ds4_gpu_env_bool("DS4_DSPARK_EXACT_ATTN_OUT_NR4")',
            metal_host,
        )
        self.assertIn("enabled = configured < 0 ? 1 : configured", metal_host)
        self.assertIn(
            "kernel_dsv4_attn_out_low_q8_0_f32_impl<4>", moe
        )
        self.assertIn(
            "kernel_dsv4_q8_hc_expand4_q8_0_impl<4>", hc
        )
        self.assertIn("kernel_mul_mv_q8_0_f32_impl<NR0", moe)


if __name__ == "__main__":
    unittest.main()
