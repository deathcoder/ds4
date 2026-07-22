#!/usr/bin/env python3
"""Model-free tests for the DSpark pre-sidecar router study."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_dspark_pre_sidecar_router as router  # noqa: E402


def record(round_number, accepted, baseline=100.0, dspark=90.0):
    selected = max(accepted, 1)
    return {
        "task": "task",
        "round": round_number,
        "proposed": 5,
        "selected": selected,
        "verified": selected,
        "accepted": accepted,
        "committed": max(1, accepted),
        "baseline_ms": baseline,
        "accounted_dspark_ms": dspark,
        "current_profitable": dspark < baseline,
        "confidences": (0.9, 0.8, 0.7, 0.6, 0.5),
    }


class DSparkPreSidecarRouterTests(unittest.TestCase):
    def test_terminal_capacity_can_verify_less_than_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "analysis": router.ORACLE_ANALYSIS,
                "threshold": router.THRESHOLD,
                "tasks": {"task": {}},
                "aggregate": {
                    "rounds": 1,
                    "baseline_ms": 100.0,
                    "accounted_dspark_ms": 90.0,
                    "profitable_rounds": 1,
                },
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            fields = (
                "task", "round", "proposed", "selected", "verified",
                "accepted", "committed", "baseline_ms",
                "accounted_dspark_ms", "current_profitable", "confidences",
            )
            with (run_dir / "rounds.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "task": "task",
                    "round": 1,
                    "proposed": 5,
                    "selected": 5,
                    "verified": 3,
                    "accepted": 3,
                    "committed": 3,
                    "baseline_ms": 100.0,
                    "accounted_dspark_ms": 90.0,
                    "current_profitable": "true",
                    "confidences": "0.9,0.8,0.7,0.6,0.5",
                })
            _, by_task, _, _ = router.load_oracle_run(run_dir)
            self.assertEqual(by_task["task"][0]["verified"], 3)
            self.assertEqual(by_task["task"][0]["selected"], 5)

    def test_previous_count_policy_is_strictly_lagged(self):
        policy = {
            "kind": "previous_count",
            "feature": "accepted",
            "threshold": 4.0,
        }
        self.assertTrue(router.policy_routes(policy, [record(1, 5)]))
        self.assertFalse(router.policy_routes(policy, [record(1, 3)]))

    def test_evaluation_routes_first_round_to_baseline(self):
        rows = [record(1, 5, dspark=1.0), record(2, 5, dspark=50.0)]
        policy = {
            "name": "previous accepted >= 5",
            "kind": "previous_count",
            "feature": "accepted",
            "threshold": 5.0,
        }
        metrics = router.evaluate_policy(
            {"task": rows}, ["task"], {"task": policy}
        )
        self.assertEqual(metrics["routed_rounds"], 1)
        self.assertAlmostEqual(metrics["policy_ms"], 150.0)
        self.assertAlmostEqual(metrics["pooled_ratio"], 200.0 / 150.0)

    def test_transition_summary_measures_profit_persistence(self):
        rows = [
            record(1, 0, dspark=110.0),
            record(2, 5, dspark=90.0),
            record(3, 5, dspark=90.0),
        ]
        result = router.transition_summary({"task": rows})
        self.assertEqual(result["unprofitable_to_profitable"], 1)
        self.assertEqual(result["profitable_to_profitable"], 1)
        self.assertEqual(result["profitable_after_unprofitable"], 1.0)
        self.assertEqual(result["profitable_after_profitable"], 1.0)

    def test_report_marks_failed_predictor_gate(self):
        summary = {
            "reference": {
                "current_accounted_ratio": 0.9,
                "perfect_route_oracle_ratio": 1.03,
            },
            "round_count": 10,
            "candidate_count": 2,
            "best_in_sample": {
                "policy": {"name": "lagged"},
                "pooled_ratio": 1.001,
                "task_geometric_ratio": 1.0,
                "minimum_task_ratio": 0.99,
                "routed_rounds": 2,
                "total_rounds": 10,
            },
            "leave_one_task_out": {
                "pooled_ratio": 1.0,
                "task_geometric_ratio": 1.0,
                "minimum_task_ratio": 0.99,
                "routed_rounds": 2,
                "total_rounds": 10,
                "tasks_faster": 0,
                "tasks_equal": 1,
                "tasks_slower": 0,
            },
            "transitions": {
                "unprofitable_to_unprofitable": 1,
                "unprofitable_to_profitable": 1,
                "profitable_to_unprofitable": 1,
                "profitable_to_profitable": 1,
                "profitable_after_unprofitable": 0.5,
                "profitable_after_profitable": 0.5,
            },
            "folds": [{"policy": "lagged"}],
            "promotion_gate": {
                "passed": False,
                "checks": {
                    "pooled_ratio_at_least_1_01": False,
                    "geometric_ratio_at_least_1_01": False,
                    "minimum_task_ratio_at_least_0_95": True,
                    "route_share_at_least_0_02": True,
                },
            },
            "caveats": ["not a throughput measurement"],
        }
        report = router.render_report(summary)
        self.assertIn("**FAIL**", report)
        self.assertIn("skipped-round observation assumption", report)
        self.assertIn("not a throughput measurement", report)


if __name__ == "__main__":
    unittest.main()
