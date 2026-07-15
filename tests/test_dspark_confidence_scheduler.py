#!/usr/bin/env python3
"""Model-free tests for the DSpark confidence-prefix analyzer."""

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speed-bench"))
import analyze_dspark_confidence_scheduler as scheduler  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
