#!/usr/bin/env python3
"""Model-free tests for the one-layer causal-attention ablation."""

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
        ctx=16384,
        tokens=128,
        fast_verifier=False,
        serial_ffn_ablation=False,
        attention_pre_ablation=False,
        attention_suffix_ablation=False,
        compressor_pair_nr4_ablation=False,
        indexed_attention_rb16_promotion_ablation=False,
        exact_q8_rows_ablation=False,
        attention_inverse_rope_fusion_ablation=False,
        exact_prefix_checkpoint_ablation=False,
        causal_attention_layer41_ablation=True,
        stats_only=False,
    )


class CausalAttentionAblationTests(unittest.TestCase):
    def test_ablation_compares_default_with_layer41_runtime(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "causal_attention_layer41"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_candidate_enables_layer41_runtime(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "causal_attention_layer41", runtime_stats=False
        )
        key = "DS4_DSPARK_CAUSAL_ATTN_RUNTIME_LAYER"
        self.assertNotIn(key, reference)
        self.assertEqual(candidate[key], "41")
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_candidate_command_is_explicit_metal_without_stats(self):
        text = comparison.command_text(args(), "causal_attention_layer41")
        self.assertIn("DS4_DSPARK_CAUSAL_ATTN_RUNTIME_LAYER=41", text)
        self.assertIn("--backend metal", text)
        self.assertIn(
            f"--dspark {Path('/tmp/dspark.gguf').resolve()}", text
        )
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", text)

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {
                "pair": 1,
                "mode": "causal_attention_layer41",
                "generation_tps": 11.0,
            },
            {
                "pair": 2,
                "mode": "causal_attention_layer41",
                "generation_tps": 12.0,
            },
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "causal_attention_layer41")
        )
        self.assertEqual(
            summary["comparison"], "causal_attention_layer41_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("One-Layer Causal Attention Ablation", report)
        self.assertIn("Layer-41 causal attention delta", report)


if __name__ == "__main__":
    unittest.main()
