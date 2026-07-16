#!/usr/bin/env python3
"""Model-free tests for the persistent-KV DSpark Metal drafter gate."""

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
        metal_drafter_ablation=True,
        stats_only=False,
    )


class MetalDrafterTests(unittest.TestCase):
    def test_ablation_modes_keep_default_exact_as_reference(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "metal_drafter"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_candidate_enables_metal_drafter(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "metal_drafter", runtime_stats=False
        )
        self.assertNotIn("DS4_DSPARK_METAL_DRAFTER", reference)
        self.assertEqual(candidate["DS4_DSPARK_METAL_DRAFTER"], "1")
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_commands_are_metal_and_uninstrumented(self):
        reference = comparison.command_text(
            args(), "default_exact", runtime_stats=False
        )
        candidate = comparison.command_text(
            args(), "metal_drafter", runtime_stats=False
        )
        self.assertNotIn("DS4_DSPARK_METAL_DRAFTER", reference)
        self.assertIn("DS4_DSPARK_METAL_DRAFTER=1", candidate)
        self.assertIn("--backend metal", candidate)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {"pair": 1, "mode": "metal_drafter", "generation_tps": 12.0},
            {"pair": 2, "mode": "metal_drafter", "generation_tps": 11.0},
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "metal_drafter")
        )
        self.assertEqual(summary["comparison"], "metal_drafter_ablation")
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Persistent-KV Metal Drafter Ablation", report)
        self.assertIn("Metal drafter delta", report)

    def test_runtime_gate_is_default_off_in_source(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        self.assertIn('getenv("DS4_DSPARK_METAL_DRAFTER")', source)
        self.assertIn("gpu_metal_drafter_fallbacks", source)
        self.assertIn(
            "ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor",
            source,
        )

    def test_stats_only_summary_omits_throughput(self):
        rows = []
        for mode, metal in (("default_exact", False), ("metal_drafter", True)):
            row = {
                "mode": mode,
                "stats_emitted": 64,
                "stats_proposals": 14,
                "stats_multi_attempts": 13,
                "stats_target_evals": 14,
                "stats_target_eval_tokens": 64,
                "stats_metal_drafter_attempts": 14 if metal else 0,
                "stats_metal_drafter_successes": 14 if metal else 0,
                "stats_metal_drafter_fallbacks": 0,
                "stats_generation_bridge_ms": 16 if metal else 12,
                "stats_generation_stage0_ms": 40 if metal else 80,
                "stats_generation_stage1_ms": 40 if metal else 80,
                "stats_generation_stage2_ms": 40 if metal else 80,
                "stats_generation_head_ms": 8,
                "stats_generation_chain_ms": 64,
                "stats_generation_sidecar_ms": 208 if metal else 324,
            }
            rows.append(row)
        summary = comparison.summarize_metal_drafter_stats(rows)
        report = comparison.format_report(summary)
        self.assertEqual(summary["comparison"], "metal_drafter_stats")
        self.assertIn("throughput values are intentionally omitted", report)
        self.assertNotIn("t/s", report)
        self.assertGreater(summary["stages_saved_ms_per_emitted"], 0)


if __name__ == "__main__":
    unittest.main()
