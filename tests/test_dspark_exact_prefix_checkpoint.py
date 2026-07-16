#!/usr/bin/env python3
"""Model-free tests for the exact DSpark prefix-checkpoint promotion gate."""

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
    def test_ablation_modes_are_legacy_replay_then_promoted_default(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("replay_partial_accept", "default_exact"),
        )

    def test_only_reference_disables_checkpoint(self):
        reference = comparison.clean_dspark_env(
            "replay_partial_accept", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        self.assertEqual(
            reference["DS4_DSPARK_EXACT_PREFIX_CHECKPOINT"], "0"
        )
        self.assertNotIn("DS4_DSPARK_EXACT_PREFIX_CHECKPOINT", candidate)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_commands_are_metal_and_reference_disables_checkpoint(self):
        reference = comparison.command_text(
            args(), "replay_partial_accept", runtime_stats=False
        )
        candidate = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        self.assertIn("DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=0", reference)
        self.assertNotIn("DS4_DSPARK_EXACT_PREFIX_CHECKPOINT", candidate)
        self.assertIn("--backend metal", candidate)
        self.assertIn("--dspark ", candidate)
        self.assertIn("dspark.gguf", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {
                "pair": 1,
                "mode": "replay_partial_accept",
                "generation_tps": 10.0,
            },
            {
                "pair": 1,
                "mode": "default_exact",
                "generation_tps": 12.0,
            },
            {"pair": 2, "mode": "default_exact", "generation_tps": 11.0},
            {
                "pair": 2,
                "mode": "replay_partial_accept",
                "generation_tps": 10.0,
            },
        ]
        summary = comparison.summarize(
            rows, ("replay_partial_accept", "default_exact")
        )
        self.assertEqual(
            summary["comparison"], "exact_prefix_checkpoint_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Promotion Confirmation", report)
        self.assertIn("Promoted exact prefix-checkpoint", report)


if __name__ == "__main__":
    unittest.main()
