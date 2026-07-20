#!/usr/bin/env python3
"""Model-free tests for the measured-cost DSpark scheduler audit."""

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_dspark_cost_aware_scheduler as audit  # noqa: E402


class DSparkCostAwareSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.costs = {
            "fixed_ms_per_round": 5.0,
            "target_ms_per_eval": {1: 10.0, 2: 15.0, 3: 20.0, 4: 25.0, 5: 30.0},
            "cycle_ms": {0: 15.0, 2: 20.0, 3: 25.0, 4: 30.0, 5: 35.0},
        }

    def record(self, accepted, confidences, sample="task", round_number=1):
        return {
            "sample": sample,
            "round": round_number,
            "accepted": accepted,
            "confidences": tuple(confidences),
        }

    def test_runtime_progress_preserves_one_token_fallback_semantics(self):
        self.assertEqual(audit.runtime_progress(5, 0), 1)
        self.assertEqual(audit.runtime_progress(5, 1), 1)
        self.assertEqual(audit.runtime_progress(5, 2), 3)
        self.assertEqual(audit.runtime_progress(1, 5), 2)

    def test_expected_progress_uses_conditional_survival(self):
        value = audit.expected_progress((0.8, 0.5, 0.1, 0.1, 0.1), 2)
        self.assertAlmostEqual(value, 1.0 + 0.8 + 0.8 * 0.5)
        self.assertEqual(
            audit.expected_progress((1.0, 1.0, 1.0, 1.0, 1.0), 0),
            1.0,
        )

    def test_static_threshold_keeps_deepspec_strict_prefix_rule(self):
        self.assertEqual(audit.selected_prefix((0.9, 0.75, 0.7), 0.75), 2)
        self.assertEqual(
            audit.selected_prefix((0.9, math.nextafter(0.75, 0.0)), 0.75),
            1,
        )

    def test_cost_aware_policy_excludes_useless_k1_route(self):
        width = audit.cost_aware_width(
            (0.95, 0.9, 0.8, 0.7, 0.6), self.costs["cycle_ms"]
        )
        self.assertIn(width, audit.CANDIDATE_WIDTHS)
        self.assertNotEqual(width, 1)

    def test_summary_compares_policy_with_static_threshold(self):
        records = [
            self.record(5, (0.95, 0.9, 0.8, 0.7, 0.6)),
            self.record(0, (0.8, 0.2, 0.2, 0.2, 0.2), round_number=2),
        ]
        summary = audit.analyze(records, self.costs)
        self.assertEqual(
            summary["analysis"],
            "dspark_measured_cost_scheduler_local_counterfactual",
        )
        self.assertEqual(summary["candidate_widths"], [0, 2, 3, 4, 5])
        self.assertGreater(summary["realized_round_oracle"]["ratio_vs_static"], 0.0)
        report = audit.render_report(summary)
        self.assertIn("Measured-Cost Scheduler Audit", report)
        self.assertIn("| policy | K=0 | K=1 | K=2", report)
        self.assertIn("not a throughput prediction", report)


if __name__ == "__main__":
    unittest.main()
