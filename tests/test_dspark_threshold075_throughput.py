#!/usr/bin/env python3
"""Model-free tests for the full threshold-0.75 HumanEval confirmation."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_threshold075_throughput as confirmation  # noqa: E402
import run_dspark_humaneval_throughput as throughput  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class DSparkThreshold075ThroughputTests(unittest.TestCase):
    def test_policy_and_decision_gates_are_frozen(self):
        self.assertEqual(confirmation.THRESHOLD, "0.75")
        self.assertEqual(confirmation.SAMPLE_COUNT, 32)
        self.assertEqual(confirmation.MIN_MOVEMENT_GEOMEAN, 1.05)
        self.assertEqual(confirmation.MIN_IMPROVED_TASKS, 24)
        self.assertEqual(confirmation.MIN_TASK_MOVEMENT, 0.80)
        self.assertEqual(confirmation.NEAR_PARITY_GEOMEAN, 0.95)
        self.assertEqual(confirmation.NEAR_PARITY_FASTER_TASKS, 8)

    def test_measured_order_alternates(self):
        self.assertEqual(
            throughput.measured_order(1), ("baseline", "runtime")
        )
        self.assertEqual(
            throughput.measured_order(2), ("runtime", "baseline")
        )

    def test_runtime_environment_is_uninstrumented_threshold_075(self):
        env = common.benchmark_env(
            "runtime", False, confidence_threshold=confirmation.THRESHOLD
        )
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)
        self.assertNotIn("DS4_DSPARK_ACCEPTANCE_AUDIT", env)
        self.assertNotIn("DS4_DSPARK_ORACLE_TRACE", env)

    @staticmethod
    def records(count=32):
        return [
            {"label": f"humaneval_{index:03d}", "source_index": index}
            for index in range(count)
        ]

    @staticmethod
    def reference(records, prior_ratio=0.8):
        return {
            "tasks": {
                record["label"]: {
                    "prior": {
                        "acceptance_verify_rate": 0.7,
                        "paired_ratio": prior_ratio,
                    }
                }
                for record in records
            }
        }

    @staticmethod
    def rows(records, ratio, prior_ratio=0.8):
        rows = []
        for position, record in enumerate(records, start=1):
            order = "-".join(throughput.measured_order(position))
            rows.extend([
                {
                    "prompt": record["label"],
                    "mode": "baseline",
                    "generation_tps": 20.0,
                    "pair_order": order,
                },
                {
                    "prompt": record["label"],
                    "mode": "runtime",
                    "generation_tps": 20.0 * ratio,
                    "pair_order": order,
                },
            ])
        return rows

    def test_summary_passes_confirmation_and_near_parity(self):
        records = self.records()
        summary = confirmation.summarize(
            self.rows(records, ratio=0.96),
            records,
            self.reference(records, prior_ratio=0.8),
        )
        self.assertTrue(summary["confirmation_gate"]["pass"])
        self.assertTrue(summary["next_path_gate"]["near_parity"])
        self.assertEqual(
            summary["next_path_gate"]["next_path"],
            "audit_threshold_075_acceptance_and_costs",
        )

    def test_summary_freezes_scheduler_when_below_parity(self):
        records = self.records()
        summary = confirmation.summarize(
            self.rows(records, ratio=0.88),
            records,
            self.reference(records, prior_ratio=0.8),
        )
        self.assertTrue(summary["confirmation_gate"]["pass"])
        self.assertFalse(summary["next_path_gate"]["near_parity"])
        self.assertEqual(
            summary["next_path_gate"]["next_path"],
            "freeze_scheduler_and_optimize_exact_verifier",
        )

    def test_confirmation_rejects_broad_historical_regression(self):
        records = self.records()
        summary = confirmation.summarize(
            self.rows(records, ratio=0.78),
            records,
            self.reference(records, prior_ratio=0.8),
        )
        self.assertFalse(summary["confirmation_gate"]["pass"])

    def test_report_distinguishes_authoritative_and_historical_ratios(self):
        records = self.records()
        summary = confirmation.summarize(
            self.rows(records, ratio=0.88),
            records,
            self.reference(records, prior_ratio=0.8),
        )
        report = confirmation.render_report(summary)
        self.assertIn("paired ratios are authoritative", report)
        self.assertIn("descriptive cross-run context", report)
        self.assertIn("No DSpark stats", report)


if __name__ == "__main__":
    unittest.main()
