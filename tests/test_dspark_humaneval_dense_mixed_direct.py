#!/usr/bin/env python3
"""Model-free tests for HumanEval direct dense-mixed confirmation."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_dense_mixed_direct as confirmation  # noqa: E402


class HumanEvalDenseMixedDirectTests(unittest.TestCase):
    def test_protocol_and_gate_are_frozen(self):
        self.assertEqual(confirmation.THRESHOLD, "0.75")
        self.assertEqual(confirmation.SAMPLE_COUNT, 32)
        self.assertEqual(confirmation.MODES, ("gathered", "fused_gather"))
        self.assertEqual(confirmation.MIN_GEOMEAN, 1.02)
        self.assertEqual(confirmation.MIN_WINS, 24)
        self.assertEqual(confirmation.MIN_TASK_RATIO, 0.95)
        self.assertEqual(confirmation.LOW_ACCEPTANCE_MAX, 0.65)
        self.assertEqual(confirmation.MIN_LOW_ACCEPTANCE_GEOMEAN, 1.00)

    def test_order_alternates(self):
        self.assertEqual(
            confirmation.mode_order(1), ("gathered", "fused_gather")
        )
        self.assertEqual(
            confirmation.mode_order(2), ("fused_gather", "gathered")
        )

    def test_only_fused_gather_mode_enables_candidate(self):
        gathered = confirmation.mode_env("gathered")
        direct = confirmation.mode_env("fused_gather")
        self.assertEqual(
            gathered["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75"
        )
        self.assertEqual(
            gathered["DS4_METAL_DENSE_MIXED_GATHERED_LEGACY"], "1"
        )
        self.assertEqual(direct["DS4_METAL_DENSE_MIXED_DIRECT"], "1")
        self.assertNotIn(
            "DS4_METAL_DENSE_MIXED_GATHERED_LEGACY", direct
        )
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT_TRACE", direct)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", direct)
        with self.assertRaises(ValueError):
            confirmation.mode_env("unknown")

    @staticmethod
    def records():
        return [
            {"label": f"humaneval_{index:03d}", "source_index": index}
            for index in range(32)
        ]

    @staticmethod
    def reference(records, acceptance=0.7):
        return {
            "tasks": {
                record["label"]: {"acceptance_verify_rate": acceptance}
                for record in records
            }
        }

    @staticmethod
    def rows(records, ratio):
        rows = []
        for position, record in enumerate(records, start=1):
            order = "-".join(confirmation.mode_order(position))
            rows.extend([
                {
                    "prompt": record["label"],
                    "mode": "gathered",
                    "generation_tps": 20.0,
                    "pair_order": order,
                },
                {
                    "prompt": record["label"],
                    "mode": "fused_gather",
                    "generation_tps": 20.0 * ratio,
                    "pair_order": order,
                },
            ])
        return rows

    def test_summary_passes_broad_positive_result(self):
        records = self.records()
        summary = confirmation.summarize(
            self.rows(records, 1.03),
            records,
            self.reference(records, acceptance=0.60),
        )
        self.assertTrue(summary["promotion_gate"]["pass"])
        self.assertEqual(summary["direct_faster_tasks"], 32)
        self.assertAlmostEqual(
            summary["low_acceptance_geometric_mean"], 1.03
        )

    def test_summary_rejects_low_acceptance_regression(self):
        records = self.records()
        rows = self.rows(records, 1.03)
        for row in rows:
            if row["mode"] == "fused_gather" and row["prompt"].endswith("000"):
                row["generation_tps"] = 18.0
        reference = self.reference(records, acceptance=0.70)
        reference["tasks"]["humaneval_000"]["acceptance_verify_rate"] = 0.60
        summary = confirmation.summarize(rows, records, reference)
        self.assertFalse(summary["promotion_gate"]["pass"])
        self.assertLess(summary["paired_ratio_minimum"], 0.95)

    def test_report_is_explicitly_uninstrumented(self):
        records = self.records()
        summary = confirmation.summarize(
            self.rows(records, 1.03),
            records,
            self.reference(records, acceptance=0.60),
        )
        report = confirmation.render_report(summary)
        self.assertIn("paired ratios are authoritative", report)
        self.assertIn("acceptance versus paired ratio: n/a", report)
        self.assertIn("No DSpark stats", report)
        self.assertIn("Promotion Gate", report)


if __name__ == "__main__":
    unittest.main()
