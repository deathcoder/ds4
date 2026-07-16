#!/usr/bin/env python3
"""Model-free tests for the exact DSpark prefix-checkpoint ablation."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speed-bench"))
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
        exact_prefix_checkpoint_ablation=True,
    )


class ExactPrefixCheckpointTests(unittest.TestCase):
    def test_ablation_modes_are_replay_then_checkpoint(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "exact_prefix_checkpoint"),
        )

    def test_only_candidate_enables_checkpoint(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "exact_prefix_checkpoint", runtime_stats=False
        )
        self.assertNotIn("DS4_DSPARK_EXACT_PREFIX_CHECKPOINT", reference)
        self.assertEqual(
            candidate["DS4_DSPARK_EXACT_PREFIX_CHECKPOINT"], "1"
        )
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_command_is_metal_and_labels_checkpoint(self):
        candidate = comparison.command_text(
            args(), "exact_prefix_checkpoint", runtime_stats=False
        )
        self.assertIn("DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=1", candidate)
        self.assertIn("--backend metal", candidate)
        self.assertIn("--dspark ", candidate)
        self.assertIn("dspark.gguf", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {
                "pair": 1,
                "mode": "exact_prefix_checkpoint",
                "generation_tps": 12.0,
            },
            {"pair": 2, "mode": "exact_prefix_checkpoint",
             "generation_tps": 11.0},
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "exact_prefix_checkpoint")
        )
        self.assertEqual(
            summary["comparison"], "exact_prefix_checkpoint_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        self.assertIn("Prefix-Checkpoint", comparison.format_report(summary))


if __name__ == "__main__":
    unittest.main()
