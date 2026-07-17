#!/usr/bin/env python3
"""Model-free tests for the post-promotion cumulative cost audit."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_cumulative_cost_audit as audit  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class DSparkCumulativeCostAuditTests(unittest.TestCase):
    @staticmethod
    def args():
        return type("Args", (), {
            "binary": Path("/tmp/ds4"),
            "model": Path("/tmp/base.gguf"),
            "dspark_model": Path("/tmp/dspark.gguf"),
            "ctx": 16384,
            "tokens": 128,
            "nothink": True,
            "fast_verifier": False,
            "exact_head_batch": False,
            "confidence_threshold": audit.THRESHOLD,
        })()

    def test_policy_and_source_are_frozen(self):
        self.assertEqual(audit.THRESHOLD, "0.75")
        self.assertEqual(audit.TASK_COUNT, 32)
        self.assertEqual(
            audit.CUMULATIVE_SOURCE_COMMIT,
            "f0edb16884aafd7e8ce95054da4d9a07117f5719",
        )

    def test_command_enables_stats_without_experimental_routes(self):
        command = common.command_text(
            self.args(), Path("/tmp/prompt.txt"), "runtime",
            stats=True, confidence_threshold=audit.THRESHOLD,
        )
        self.assertIn("DS4_DSPARK_GPU_RUNTIME_STATS=1", command)
        self.assertIn("DS4_DSPARK_CONFIDENCE_THRESHOLD=0.75", command)
        self.assertNotIn("DS4_DSPARK_ORACLE_TRACE", command)
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", command)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", command)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_GATHERED_LEGACY", command)

    def test_reference_identity_requires_clean_promoted_cumulative_run(self):
        args = self.args()
        selection = {"indices_zero_based": list(range(32))}
        summary = {
            "sample_count": 32,
            "threshold": "0.75",
            "selection": selection,
        }
        metadata = {
            "experiment": "dspark_humaneval_cumulative_throughput",
            "git_commit": audit.CUMULATIVE_SOURCE_COMMIT,
            "git_status_tracked": "",
            "config": {
                "ctx": 16384,
                "tokens": 128,
                "temperature": 0,
                "seed": 1,
                "nothink": True,
                "threshold": "0.75",
                "instrumented": False,
                "measured_pairs_per_task": 1,
                "alternating_order": True,
                "global_warmup_pairs": 2,
                "promoted_defaults": True,
            },
            "binary": {"path": "/tmp/ds4"},
            "base_model": {"path": "/tmp/base.gguf"},
            "dspark_model": {"path": "/tmp/dspark.gguf"},
        }
        audit.validate_reference_identity(args, summary, metadata, selection)
        metadata["config"]["promoted_defaults"] = False
        with self.assertRaisesRegex(SystemExit, "promoted_defaults mismatch"):
            audit.validate_reference_identity(args, summary, metadata, selection)

    def test_report_names_post_promotion_cumulative_budget(self):
        summary = {
            "reference_paired_ratio_geometric_mean": 0.8826,
            "aggregate": {
                "baseline_ms_per_emitted": 40.0,
                "runtime_ms_per_emitted": 45.0,
                "runtime_deficit_ms_per_emitted": 5.0,
                "target_ms_per_emitted": 35.0,
                "sidecar_ms_per_emitted": 5.0,
                "residual_ms_per_emitted": 5.0,
                "pooled_runtime_ratio": 0.8889,
                "target_share_of_runtime": 35.0 / 45.0,
                "sidecar_share_of_runtime": 5.0 / 45.0,
                "target_scale_for_parity": 30.0 / 35.0,
                "accounted_target_scale_for_parity": 1.0,
                "prefill_sidecar_ms": 10.0,
                "sidecar_outside_scheduler_ms": 2.0,
                "target_evals_per_emitted": 0.4,
                "target_positions_per_eval": 3.0,
                "batch_attempts": 1,
                "batch_full": 1,
                "batch_partial": 0,
                "batch_fallbacks": 0,
            },
            "verifier_widths": {},
            "scheduler_widths": {},
            "tasks": {},
        }
        report = audit.render_report(summary)
        self.assertIn("Post-Promotion Exact Verifier Cost Audit", report)
        self.assertIn("frozen uninstrumented cumulative artifact", report)
        self.assertIn("| baseline | current DSpark |", report)
        self.assertIn("Frozen cumulative geometric mean", report)


if __name__ == "__main__":
    unittest.main()
