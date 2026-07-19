#!/usr/bin/env python3
"""Model-free tests for compressor-prebatch outlier adjudication."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_compressor_pre_batch_outlier as adjudication  # noqa: E402


class HumanEvalCompressorPreBatchOutlierTests(unittest.TestCase):
    def test_protocol_and_gate_are_frozen(self):
        self.assertEqual(adjudication.TASK, "humaneval_095")
        self.assertEqual(adjudication.MEASURED_PAIRS, 6)
        self.assertEqual(adjudication.WARMUP_PAIRS, 2)
        self.assertEqual(adjudication.MIN_MEDIAN_RATIO, 1.005)
        self.assertEqual(adjudication.MIN_GEOMEAN_RATIO, 1.005)
        self.assertEqual(adjudication.MIN_WINS, 4)
        self.assertEqual(adjudication.ORIGINAL_TASK_FLOOR, 0.95)
        self.assertEqual(adjudication.MIN_PAIRS_ABOVE_FLOOR, 5)

    def test_measured_order_is_exactly_balanced(self):
        orders = [
            adjudication.measured_order(pair)
            for pair in range(1, adjudication.MEASURED_PAIRS + 1)
        ]
        self.assertEqual(
            orders.count(("default_exact", "exact_compressor_pre_batch")), 3
        )
        self.assertEqual(
            orders.count(("exact_compressor_pre_batch", "default_exact")), 3
        )

    @staticmethod
    def rows(ratios):
        rows = []
        for pair_number, ratio in enumerate(ratios, start=1):
            order = "-".join(adjudication.measured_order(pair_number))
            rows.extend([
                {
                    "pair_number": pair_number,
                    "mode": "default_exact",
                    "generation_tps": 20.0,
                    "pair_order": order,
                },
                {
                    "pair_number": pair_number,
                    "mode": "exact_compressor_pre_batch",
                    "generation_tps": 20.0 * ratio,
                    "pair_order": order,
                },
            ])
        return rows

    def test_summary_passes_replicated_positive_result(self):
        summary = adjudication.summarize(
            self.rows([1.008, 1.012, 1.010, 1.015, 1.007, 1.011])
        )
        self.assertTrue(summary["adjudication_gate"]["pass"])
        self.assertEqual(summary["candidate_wins"], 6)
        self.assertEqual(summary["pairs_above_original_floor"], 6)

    def test_summary_rejects_unresolved_floor_regression(self):
        summary = adjudication.summarize(
            self.rows([1.01, 0.94, 0.93, 0.92, 1.02, 1.01])
        )
        self.assertFalse(summary["adjudication_gate"]["pass"])
        self.assertLess(summary["pairs_above_original_floor"], 5)

    def test_summary_rejects_weak_typical_gain(self):
        summary = adjudication.summarize(
            self.rows([1.003, 1.003, 1.003, 1.003, 1.003, 1.003])
        )
        self.assertFalse(summary["adjudication_gate"]["pass"])
        self.assertLess(summary["paired_ratio_median"], 1.005)

    def test_report_keeps_original_gate_failure_explicit(self):
        summary = adjudication.summarize(
            self.rows([1.008, 1.012, 1.010, 1.015, 1.007, 1.011])
        )
        report = adjudication.render_report(summary)
        self.assertIn("32-task gate remains a formal failure", report)
        self.assertIn("paired ratios are authoritative", report)
        self.assertIn("Require median paired ratio at least `1.005x`", report)
        self.assertIn("No DSpark stats", report)


if __name__ == "__main__":
    unittest.main()
