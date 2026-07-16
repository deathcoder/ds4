#!/usr/bin/env python3
"""Model-free tests for the threshold-0.75 exact-verifier cost audit."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_threshold075_cost_audit as audit  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class DSparkThreshold075CostAuditTests(unittest.TestCase):
    @staticmethod
    def blank_stats():
        values = {field: 0 for field in common.INT_STATS | common.FLOAT_STATS}
        for field in common.INT_ARRAY_STATS:
            values[field] = [0] * 6
        for field in common.FLOAT_ARRAY_STATS:
            values[field] = [0.0] * 6
        return values

    def test_policy_is_frozen(self):
        self.assertEqual(audit.THRESHOLD, "0.75")
        self.assertEqual(audit.TASK_COUNT, 32)

    def test_command_enables_only_stats_and_threshold(self):
        args = type("Args", (), {
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
        command = common.command_text(
            args, Path("/tmp/prompt.txt"), "runtime",
            stats=True, confidence_threshold=audit.THRESHOLD,
        )
        self.assertIn("DS4_DSPARK_GPU_RUNTIME_STATS=1", command)
        self.assertIn("DS4_DSPARK_CONFIDENCE_THRESHOLD=0.75", command)
        self.assertNotIn("DS4_DSPARK_ORACLE_TRACE", command)
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", command)

    def test_task_metrics_calibrate_target_reduction(self):
        row = self.blank_stats()
        row.update({
            "emitted": 100,
            "target_evals": 25,
            "target_eval_tokens": 100,
            "target_eval_ms": 4000.0,
            "generation_sidecar_ms": 500.0,
            "batch_attempts": 25,
            "batch_full": 20,
            "batch_partial": 5,
        })
        context = {
            "record": {"source_index": 7},
            "sample": {
                "acceptance_verify_rate": 0.7,
                "paired_ratio": 0.8,
            },
            "baseline_tps": 20.0,
            "runtime_tps": 16.0,
        }
        item = audit.task_metrics(row, context)
        self.assertAlmostEqual(item["baseline_ms"], 5000.0)
        self.assertAlmostEqual(item["runtime_ms"], 6250.0)
        self.assertAlmostEqual(item["target_ms_per_emitted"], 40.0)
        self.assertAlmostEqual(item["sidecar_ms_per_emitted"], 5.0)
        self.assertAlmostEqual(item["residual_ms_per_emitted"], 17.5)
        self.assertAlmostEqual(item["target_positions_per_eval"], 4.0)
        self.assertAlmostEqual(item["target_scale_for_parity"], 0.6875)
        self.assertAlmostEqual(
            item["accounted_target_scale_for_parity"], 1.125
        )

    def test_summary_aggregates_widths_and_batch_outcomes(self):
        rows = []
        tasks = {}
        for index, task in enumerate(("humaneval_000", "humaneval_001")):
            row = self.blank_stats()
            row.update({
                "prompt": task,
                "emitted": 10,
                "target_evals": 3,
                "target_eval_tokens": 12,
                "target_eval_ms": 120.0,
                "generation_sidecar_ms": 20.0,
                "batch_attempts": 3,
                "batch_full": 2,
                "batch_partial": 1,
                "scheduler_width_rounds": [0, 0, 1, 1, 1, 0],
                "scheduler_width_committed": [0, 0, 2, 3, 5, 0],
                "scheduler_width_sidecar_ms": [0, 0, 4, 6, 10, 0],
                "verify_width_evals": [0, 0, 1, 1, 1, 0],
                "verify_width_positions": [0, 0, 2, 4, 6, 0],
                "verify_width_target_ms": [0, 0, 20, 40, 60, 0],
            })
            rows.append(row)
            tasks[task] = {
                "record": {"source_index": index},
                "sample": {
                    "acceptance_verify_rate": 0.7,
                    "paired_ratio": 0.8,
                },
                "baseline_tps": 20.0,
                "runtime_tps": 16.0,
            }
        reference = {
            "tasks": tasks,
            "summary": {"paired_ratio_geometric_mean": 0.8},
        }
        summary = audit.summarize(rows, reference)
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["batch_attempts"], 6)
        self.assertEqual(aggregate["batch_full"], 4)
        self.assertEqual(aggregate["batch_partial"], 2)
        self.assertAlmostEqual(
            summary["verifier_widths"]["4"]["target_ms_per_position"], 10.0
        )
        self.assertAlmostEqual(
            summary["scheduler_widths"]["4"]["progress_per_round"], 5.0
        )

    def test_report_omits_throughput_and_explains_calibration(self):
        row = self.blank_stats()
        row.update({
            "prompt": "humaneval_000",
            "emitted": 10,
            "target_evals": 2,
            "target_eval_tokens": 8,
            "target_eval_ms": 100.0,
            "generation_sidecar_ms": 20.0,
            "batch_attempts": 2,
            "batch_full": 1,
            "batch_partial": 1,
            "scheduler_width_rounds": [0, 0, 0, 0, 0, 2],
            "scheduler_width_committed": [0, 0, 0, 0, 0, 10],
            "scheduler_width_sidecar_ms": [0, 0, 0, 0, 0, 20],
            "verify_width_evals": [0, 0, 0, 0, 2, 0],
            "verify_width_positions": [0, 0, 0, 0, 8, 0],
            "verify_width_target_ms": [0, 0, 0, 0, 100, 0],
        })
        reference = {
            "tasks": {
                "humaneval_000": {
                    "record": {"source_index": 0},
                    "sample": {
                        "acceptance_verify_rate": 0.7,
                        "paired_ratio": 0.8,
                    },
                    "baseline_tps": 20.0,
                    "runtime_tps": 16.0,
                }
            },
            "summary": {"paired_ratio_geometric_mean": 0.8},
        }
        report = audit.render_report(audit.summarize([row], reference))
        self.assertIn("throughput values are intentionally omitted", report)
        self.assertIn("End-to-end-calibrated target scale", report)
        self.assertIn("cross-run difference", report)
        self.assertIn("No fresh baseline", report)


if __name__ == "__main__":
    unittest.main()
