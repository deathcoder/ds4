#!/usr/bin/env python3
"""Model-free tests for cumulative HumanEval throughput reassessment."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_cumulative_throughput as cumulative  # noqa: E402


class HumanEvalCumulativeThroughputTests(unittest.TestCase):
    def test_protocol_and_gates_are_frozen(self):
        self.assertEqual(cumulative.THRESHOLD, "0.75")
        self.assertEqual(cumulative.SAMPLE_COUNT, 32)
        self.assertEqual(cumulative.MIN_MOVEMENT_GEOMEAN, 1.05)
        self.assertEqual(cumulative.MIN_IMPROVED_TASKS, 24)
        self.assertEqual(cumulative.MIN_TASK_MOVEMENT, 0.90)
        self.assertEqual(cumulative.NEAR_PARITY_GEOMEAN, 0.95)
        self.assertEqual(cumulative.PARITY_GEOMEAN, 1.00)

    def test_runtime_uses_only_promoted_defaults(self):
        env = cumulative.runtime_env()
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", env)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_GATHERED_LEGACY", env)
        self.assertNotIn("DS4_DSPARK_EXACT_ROUTED_MOE_HYBRID", env)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)

    @staticmethod
    def records():
        return [
            {"label": f"humaneval_{index:03d}", "source_index": index}
            for index in range(cumulative.SAMPLE_COUNT)
        ]

    @staticmethod
    def historical(records, ratio=0.86):
        return {
            "summary": {"paired_ratio_geometric_mean": ratio},
            "tasks": {
                record["label"]: {
                    "acceptance_verify_rate": 0.7,
                    "paired_ratio": ratio,
                }
                for record in records
            },
        }

    @staticmethod
    def rows(records, ratio):
        rows = []
        for position, record in enumerate(records, start=1):
            order = "-".join(cumulative.throughput.measured_order(position))
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

    def test_summary_passes_movement_and_reaches_near_parity(self):
        records = self.records()
        summary = cumulative.summarize(
            self.rows(records, 0.96), records, self.historical(records)
        )
        self.assertTrue(summary["movement_gate"]["pass"])
        self.assertEqual(summary["parity_gate"]["outcome"], "near_parity")
        self.assertEqual(summary["paired_ratio_improved_tasks"], 32)

    def test_summary_distinguishes_parity(self):
        records = self.records()
        summary = cumulative.summarize(
            self.rows(records, 1.02), records, self.historical(records)
        )
        self.assertEqual(
            summary["parity_gate"]["outcome"], "parity_or_speedup"
        )
        self.assertEqual(summary["runtime_faster_tasks"], 32)

    def test_summary_rejects_weak_movement(self):
        records = self.records()
        summary = cumulative.summarize(
            self.rows(records, 0.88), records, self.historical(records)
        )
        self.assertFalse(summary["movement_gate"]["pass"])
        self.assertEqual(
            summary["parity_gate"]["outcome"], "below_near_parity"
        )

    def test_report_is_explicitly_uninstrumented(self):
        records = self.records()
        summary = cumulative.summarize(
            self.rows(records, 0.96), records, self.historical(records)
        )
        report = cumulative.render_report(summary)
        self.assertIn("current paired ratios are authoritative", report)
        self.assertIn("pins confidence threshold 0.75", report)
        self.assertIn("No DSpark stats", report)

if __name__ == "__main__":
    unittest.main()
