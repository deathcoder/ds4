#!/usr/bin/env python3
"""Model-free tests for the upstream router-weight batch fusion candidate."""

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
        dense_mixed_vector_prepare_ablation=False,
        router_weights_batch_fusion_ablation=True,
        stats_only=False,
    )


class RouterWeightsBatchFusionTests(unittest.TestCase):
    def test_ablation_compares_default_with_batch_fusion(self):
        self.assertEqual(
            comparison.benchmark_modes(args()),
            ("default_exact", "router_weights_batch_fusion"),
        )
        self.assertFalse(comparison.throughput_runtime_stats_enabled(args()))

    def test_only_candidate_enables_fusion(self):
        reference = comparison.clean_dspark_env(
            "default_exact", runtime_stats=False
        )
        candidate = comparison.clean_dspark_env(
            "router_weights_batch_fusion", runtime_stats=False
        )
        key = "DS4_METAL_ENABLE_ROUTER_WEIGHTS_BATCH_FUSION"
        self.assertNotIn(key, reference)
        self.assertEqual(candidate[key], "1")
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", candidate)
        self.assertNotIn("TRACE", " ".join(candidate))

    def test_summary_reports_paired_gain(self):
        rows = [
            {"pair": 1, "mode": "default_exact", "generation_tps": 10.0},
            {
                "pair": 1,
                "mode": "router_weights_batch_fusion",
                "generation_tps": 11.0,
            },
            {
                "pair": 2,
                "mode": "router_weights_batch_fusion",
                "generation_tps": 12.0,
            },
            {"pair": 2, "mode": "default_exact", "generation_tps": 10.0},
        ]
        summary = comparison.summarize(
            rows, ("default_exact", "router_weights_batch_fusion")
        )
        self.assertEqual(
            summary["comparison"], "router_weights_batch_fusion_ablation"
        )
        self.assertAlmostEqual(summary["paired_speedup_median"], 1.15)
        report = comparison.format_report(summary)
        self.assertIn("Router-Weight Batch Fusion Ablation", report)
        self.assertIn("Batch-fusion delta", report)

    def test_exact_override_remains_after_generic_fusion(self):
        graph = (ROOT / "ds4.c").read_text(encoding="utf-8")
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        shader = (ROOT / "metal/dsv4_misc.metal").read_text(encoding="utf-8")

        generic = graph.index("ds4_gpu_router_select_batch_tensor")
        exact = graph.index("ds4_gpu_dsv4_router_weights_decode_rows_tensor")
        self.assertLess(generic, exact)
        self.assertIn("kernel_dsv4_router_weights_batch", host)
        self.assertIn(
            "DS4_METAL_ENABLE_ROUTER_WEIGHTS_BATCH_FUSION", host
        )
        self.assertIn("kernel void kernel_dsv4_router_weights_batch", shader)


if __name__ == "__main__":
    unittest.main()
