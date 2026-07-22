#!/usr/bin/env python3
"""Model-free tests for the exact-verifier gap-closure ledger."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_dspark_gap_closure as gap  # noqa: E402


def evidence_summaries():
    cost = {
        "analysis": "dspark_humaneval_cumulative_exact_verifier_cost",
        "threshold": "0.75",
        "task_count": 32,
        "aggregate": {
            "baseline_ms_per_emitted": 40.0,
            "runtime_ms_per_emitted": 45.0,
            "runtime_deficit_ms_per_emitted": 5.0,
            "target_ms_per_emitted": 40.0,
            "accounted_target_scale_for_parity": 0.9,
        },
        "verifier_widths": {
            "1": {"target_time_share": 0.20},
            "2": {"target_time_share": 0.08},
            "3": {"target_time_share": 0.09},
            "4": {"target_time_share": 0.08},
            "5": {"target_time_share": 0.55},
        },
    }
    layer = {
        "analysis": "dspark_post_promotion_width_stratified_exact_layer",
        "task": "humaneval_079",
        "widths": [2, 3, 4, 5],
        "sampled_width_totals": {
            "5": {
                "sampled_layer_ms_per_row": 3.0,
                "sampled_stage_ms_per_row": {
                    "attention_pre_batch": 0.6,
                    "attention_tail_serial": 1.2,
                    "ffn_batch": 1.2,
                },
            }
        },
    }
    ffn = {
        "analysis": "dspark_post_promotion_width_stratified_exact_ffn",
        "task": "humaneval_079",
        "widths": [2, 3, 4, 5],
        "width5_components": {
            "routed_moe": {"share": 0.4},
            "other": {"share": 0.6},
        },
    }
    tail = {
        "analysis": "dspark_post_promotion_width_stratified_attention_tail",
        "task": "humaneval_079",
        "layer": 42,
        "width_results": {
            "5": {
                "components": {
                    "attention": {"median_share": 0.2},
                    "other_a": {"median_share": 0.4},
                    "other_b": {"median_share": 0.4},
                }
            }
        },
    }
    return cost, layer, ffn, tail


class GapClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = gap.analyze_summaries(*evidence_summaries())

    def test_measured_gap_is_normalized_to_target_time(self):
        self.assertAlmostEqual(
            self.summary["cost"]["required_target_reduction"], 0.125
        )
        self.assertAlmostEqual(
            self.summary["cost"]["accounted_required_target_reduction"], 0.1
        )

    def test_width_five_alone_requires_partial_width_reduction(self):
        width5 = self.summary["width_closure"]["5"]
        self.assertTrue(width5["can_close_by_elimination"])
        self.assertAlmostEqual(width5["required_scope_reduction"], 0.125 / 0.55)

    def test_narrow_width_five_stage_cannot_be_overclaimed(self):
        prep = self.summary["stage_closure"]["attention_pre_batch"]
        self.assertAlmostEqual(prep["sampled_stage_share"], 0.2)
        self.assertAlmostEqual(prep["width5_only"]["target_time_share"], 0.11)
        self.assertFalse(prep["width5_only"]["can_close_by_elimination"])

    def test_inner_component_share_composes_stage_and_width(self):
        routed = self.summary["inner_component_closure"]["ffn/routed_moe"]
        self.assertAlmostEqual(routed["all_widths"]["target_time_share"], 0.16)
        self.assertAlmostEqual(routed["width5_only"]["target_time_share"], 0.088)
        self.assertFalse(routed["width5_only"]["can_close_by_elimination"])

    def test_decision_rejects_implausibly_large_inner_reduction(self):
        self.assertEqual(self.summary["credible_inner_components"], [])
        self.assertEqual(
            self.summary["decision"],
            "REQUIRE_CROSS_WIDTH_STAGE_OR_VERIFIER_REDESIGN",
        )

    def test_report_states_scope_and_model_free_limit(self):
        report = gap.render_report(self.summary)
        self.assertIn("REQUIRE_CROSS_WIDTH_STAGE_OR_VERIFIER_REDESIGN", report)
        self.assertIn("no model process or throughput benchmark was run", report)
        self.assertIn("impossible", report)

    def test_rejects_mismatched_protocol(self):
        cost, layer, ffn, tail = evidence_summaries()
        tail["layer"] = 41
        with self.assertRaisesRegex(ValueError, "serial-tail protocol"):
            gap.analyze_summaries(cost, layer, ffn, tail)


if __name__ == "__main__":
    unittest.main()
