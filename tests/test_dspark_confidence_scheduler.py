#!/usr/bin/env python3
"""Model-free tests for the DSpark confidence-prefix analyzer."""

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speed-bench"))
import analyze_dspark_confidence_scheduler as scheduler  # noqa: E402
import run_dspark_humaneval_scheduler_ablation as ablation  # noqa: E402
import run_dspark_humaneval_throughput as throughput  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class ConfidenceSchedulerTests(unittest.TestCase):
    def setUp(self):
        data = (
            b"ds4: DSpark acceptance trace round=1 proposed=5 accepted=3 "
            b"truncated=0 confidences=0.9,0.8,0.7,0.6,0.5\n"
        )
        self.record = scheduler.parse_trace(
            data, Path("synthetic.stderr"), "sample"
        )[0]

    def test_deepspec_threshold_is_strict(self):
        confidences = self.record["confidences"]
        self.assertEqual(scheduler.selected_prefix(confidences, 0.7), 3)
        self.assertEqual(
            scheduler.selected_prefix(
                confidences, math.nextafter(0.7, math.inf)
            ),
            2,
        )

    def test_fixed_and_oracle_accounting(self):
        fixed = scheduler.policy_metrics([self.record], threshold=0.0)
        oracle = scheduler.policy_metrics([self.record], oracle=True)
        self.assertEqual(fixed["local_progress"], 4)
        self.assertEqual(fixed["target_positions"], 5)
        self.assertEqual(oracle["local_progress"], 4)
        self.assertEqual(oracle["target_positions"], 3)

    def test_zero_draft_still_requires_target_eval(self):
        zero = scheduler.policy_metrics([self.record], threshold=1.0)
        self.assertEqual(zero["local_progress"], 1)
        self.assertEqual(zero["target_positions"], 1)
        self.assertEqual(zero["target_eval_amplification"], 4)

    def test_truncated_record_is_identified(self):
        data = (
            b"ds4: DSpark acceptance trace round=1 proposed=2 accepted=1 "
            b"truncated=1 confidences=0.4,0.3\n"
        )
        record = scheduler.parse_trace(
            data, Path("synthetic.stderr"), "sample"
        )[0]
        self.assertTrue(record["truncated"])

    def test_rejects_confidence_count_mismatch(self):
        data = (
            b"ds4: DSpark acceptance trace round=1 proposed=2 accepted=1 "
            b"truncated=0 confidences=0.4\n"
        )
        with self.assertRaisesRegex(RuntimeError, "confidence count mismatch"):
            scheduler.parse_trace(data, Path("synthetic.stderr"), "sample")


class ConfidenceSchedulerAblationTests(unittest.TestCase):
    def test_three_period_order_is_balanced(self):
        orders = [ablation.measured_order(0, pair) for pair in range(1, 4)]
        for mode in ablation.MODES:
            positions = [order.index(mode) for order in orders]
            self.assertEqual(sorted(positions), [0, 1, 2])

    def test_modes_set_only_predeclared_thresholds(self):
        fixed = ablation.mode_env("fixed_k5")
        conservative = ablation.mode_env("threshold_038")
        balanced = ablation.mode_env("threshold_0455")
        self.assertEqual(
            fixed["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0"
        )
        self.assertEqual(
            conservative["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.38"
        )
        self.assertEqual(
            balanced["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.455"
        )
        for env in (fixed, conservative, balanced):
            self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
            self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
            self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)

    def test_summary_pairs_candidates_with_fixed_mode(self):
        rows = []
        rates = {"humaneval_152": 0.528, "humaneval_079": 0.839}
        for task in ablation.TASKS:
            for pair in range(1, ablation.PAIRS + 1):
                rows.extend([
                    {"task": task, "pair": pair, "mode": "fixed_k5",
                     "generation_tps": 10.0},
                    {"task": task, "pair": pair, "mode": "threshold_038",
                     "generation_tps": 11.0},
                    {"task": task, "pair": pair, "mode": "threshold_0455",
                     "generation_tps": 9.0},
                ])
        reference = {
            "tasks": {
                task: {
                    "sample": {
                        "acceptance_verify_rate": rates[task],
                        "paired_ratio": 0.5,
                    }
                }
                for task in ablation.TASKS
            }
        }
        summary = ablation.summarize(rows, reference)
        self.assertAlmostEqual(
            summary["aggregate"]["threshold_038"]["paired_ratio_median"],
            1.1,
        )
        self.assertAlmostEqual(
            summary["aggregate"]["threshold_0455"]["paired_ratio_median"],
            0.9,
        )


class ConfidenceSchedulerThroughputTests(unittest.TestCase):
    @staticmethod
    def scheduler_summary():
        return {
            "analysis": "deepspec_confidence_prefix_local_counterfactual",
            "samples": 32,
            "block_size": 5,
            "in_sample_policies": {
                "0.975": {
                    "threshold": 0.455891937,
                    "progress_retention": 0.976,
                }
            },
            "leave_one_task_out": {
                "0.975": {
                    "retention_floor": 0.975,
                    "threshold_median": 0.455,
                    "threshold_minimum": 0.455,
                    "threshold_maximum": 0.460,
                    "progress_retention": 0.975,
                }
            },
        }

    def test_shared_runtime_env_sets_only_requested_threshold(self):
        env = common.benchmark_env(
            "runtime", False, confidence_threshold="0.455"
        )
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.455")
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)
        baseline = common.benchmark_env("baseline", False)
        self.assertNotIn("DS4_DSPARK_CONFIDENCE_THRESHOLD", baseline)

    def test_promoted_and_fixed_threshold_constants_are_distinct(self):
        self.assertEqual(common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD, "0.455")
        self.assertEqual(common.DSPARK_FIXED_CONFIDENCE_THRESHOLD, "0")

    def test_shared_env_rejects_invalid_thresholds(self):
        for threshold in ("bad", "nan", "inf", "-0.1", "1.1"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    common.benchmark_env(
                        "runtime", False, confidence_threshold=threshold
                    )
        with self.assertRaisesRegex(ValueError, "runtime mode"):
            common.benchmark_env(
                "baseline", False, confidence_threshold="0.455"
            )

    def test_frozen_scheduler_reference_is_accepted(self):
        policy = throughput.validate_scheduler_summary(
            self.scheduler_summary()
        )
        self.assertEqual(policy["threshold"], "0.455")
        self.assertEqual(policy["retention_floor"], 0.975)

    def test_scheduler_reference_rejects_threshold_drift(self):
        summary = self.scheduler_summary()
        summary["leave_one_task_out"]["0.975"]["threshold_median"] = 0.46
        with self.assertRaisesRegex(ValueError, "threshold median"):
            throughput.validate_scheduler_summary(summary)


if __name__ == "__main__":
    unittest.main()
