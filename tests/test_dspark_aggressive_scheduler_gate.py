#!/usr/bin/env python3
"""Model-free tests for the aggressive HumanEval scheduler gate."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_aggressive_scheduler_gate as gate  # noqa: E402


class DSparkAggressiveSchedulerGateTests(unittest.TestCase):
    def test_tasks_and_thresholds_are_frozen(self):
        self.assertEqual(
            gate.TASKS,
            (
                "humaneval_152",
                "humaneval_032",
                "humaneval_000",
                "humaneval_121",
                "humaneval_131",
                "humaneval_137",
                "humaneval_011",
                "humaneval_079",
            ),
        )
        self.assertEqual(gate.THRESHOLDS["threshold_0455"], "0.455")
        self.assertEqual(gate.THRESHOLDS["threshold_075"], "0.75")
        self.assertEqual(gate.THRESHOLDS["threshold_085"], "0.85")

    def test_four_mode_order_is_balanced_across_tasks(self):
        orders = [
            gate.measured_order(index) for index in range(len(gate.TASKS))
        ]
        for mode in gate.MODES:
            positions = [order.index(mode) for order in orders]
            self.assertEqual(sorted(positions), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_mode_environments_are_uninstrumented(self):
        baseline = gate.mode_env("baseline")
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME", baseline)
        for mode, threshold in (
            ("threshold_0455", "0.455"),
            ("threshold_075", "0.75"),
            ("threshold_085", "0.85"),
        ):
            env = gate.mode_env(mode)
            self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
            self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
            self.assertEqual(
                env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], threshold
            )
            self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)
            self.assertNotIn("DS4_DSPARK_ACCEPTANCE_AUDIT", env)
            self.assertNotIn("DS4_DSPARK_ORACLE_TRACE", env)

    def test_candidate_gate_uses_predeclared_rules(self):
        passing = gate.candidate_gate(
            [1.08, 1.07, 1.06, 1.09, 1.05, 1.10, 1.04, 0.95]
        )
        self.assertTrue(passing["pass_geometric_mean"])
        self.assertTrue(passing["pass_task_wins"])
        self.assertTrue(passing["pass_minimum"])
        failing = gate.candidate_gate(
            [1.20, 1.10, 1.05, 1.04, 1.03, 1.02, 0.89, 0.88]
        )
        self.assertFalse(failing["pass_minimum"])

    @staticmethod
    def synthetic_reference():
        return {
            "tasks": {
                task: {
                    "prior": {
                        "acceptance_verify_rate": 0.7,
                        "paired_ratio": 0.8,
                    }
                }
                for task in gate.TASKS
            }
        }

    def synthetic_rows(self, threshold_075=11.0, threshold_085=10.5):
        rows = []
        for index, task in enumerate(gate.TASKS):
            order = "-".join(gate.measured_order(index))
            rows.extend([
                {
                    "task": task, "mode": "baseline",
                    "generation_tps": 12.0, "order": order,
                },
                {
                    "task": task, "mode": "threshold_0455",
                    "generation_tps": 10.0, "order": order,
                },
                {
                    "task": task, "mode": "threshold_075",
                    "generation_tps": threshold_075, "order": order,
                },
                {
                    "task": task, "mode": "threshold_085",
                    "generation_tps": threshold_085, "order": order,
                },
            ])
        return rows

    def test_summary_selects_lower_threshold_when_both_pass_close(self):
        summary = gate.summarize(
            self.synthetic_rows(), self.synthetic_reference()
        )
        self.assertEqual(
            summary["promotion_gate"]["passing_candidates"],
            ["threshold_075", "threshold_085"],
        )
        self.assertEqual(
            summary["promotion_gate"]["selected_candidate"],
            "threshold_075",
        )

    def test_summary_selects_high_threshold_only_with_margin(self):
        summary = gate.summarize(
            self.synthetic_rows(threshold_075=10.5, threshold_085=11.0),
            self.synthetic_reference(),
        )
        self.assertEqual(
            summary["promotion_gate"]["selected_candidate"],
            "threshold_085",
        )

    def test_report_states_balancing_and_no_instrumentation(self):
        summary = gate.summarize(
            self.synthetic_rows(), self.synthetic_reference()
        )
        report = gate.render_report(summary)
        self.assertIn("balanced exactly", report)
        self.assertIn("No DSpark stats", report)
        self.assertIn("Selected candidate", report)


if __name__ == "__main__":
    unittest.main()
