#!/usr/bin/env python3
"""Model-free tests for the batched-verifier feasibility audit."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_dspark_batched_verifier_feasibility as feasibility  # noqa: E402


def evidence_summaries():
    cost = {
        "analysis": "dspark_humaneval_cumulative_exact_verifier_cost",
        "threshold": "0.75",
        "task_count": 32,
        "aggregate": {
            "baseline_ms_per_emitted": 40.174527873931396,
            "runtime_ms_per_emitted": 44.69408572931622,
            "runtime_deficit_ms_per_emitted": 4.519557855384825,
            "target_ms_per_emitted": 36.35562538962879,
        },
        "verifier_widths": {
            "1": {"target_time_share": 0.18820855675868783},
            "2": {"target_time_share": 0.07687460421554955},
            "3": {"target_time_share": 0.09146711834905777},
            "4": {"target_time_share": 0.09612157388410551},
            "5": {"target_time_share": 0.5473281467925993},
        },
    }
    layer = {
        "analysis": "dspark_post_promotion_width_stratified_exact_layer",
        "task": "humaneval_079",
        "widths": [2, 3, 4, 5],
        "stage_amortization": {
            "attention_pre_batch": {"width5_ms_per_row": 0.6558},
            "attention_tail_serial": {"width5_ms_per_row": 1.083},
            "ffn_batch": {"width5_ms_per_row": 1.070},
        },
    }
    tail = {
        "analysis": "dspark_post_promotion_width_stratified_attention_tail",
        "task": "humaneval_079",
        "layer": 42,
        "width_results": {
            "5": {
                "components": {
                    "kv_cache_update": {"median_ms_per_row": 0.3098},
                    "compressor_indexer": {"median_ms_per_row": 0.3280},
                    "attention": {"median_ms_per_row": 0.3820},
                    "inverse_rope": {"median_ms_per_row": 0.2868},
                    "projection_a": {"median_ms_per_row": 0.3520},
                    "projection_b_hc": {"median_ms_per_row": 0.3592},
                }
            }
        },
    }
    suffix = {
        "layers": {
            "42": {
                "candidate_projection_a_ms_per_row": 0.1160,
                "candidate_projection_b_hc_ms_per_row": 0.1112,
            }
        }
    }
    return cost, layer, tail, suffix


class BatchedVerifierDependencyTests(unittest.TestCase):
    def test_dependency_graph_places_state_before_attention(self):
        order = feasibility.topological_order()
        self.assertLess(
            order.index("proposal_state_slab"),
            order.index("causal_attention_batch"),
        )
        self.assertLess(
            order.index("causal_attention_batch"),
            order.index("projection_a_batch"),
        )
        self.assertLess(
            order.index("projection_b_hc_batch"), order.index("ffn_batch")
        )
        self.assertLess(
            order.index("output_head_batch"), order.index("acceptance_decision")
        )
        self.assertLess(
            order.index("acceptance_decision"),
            order.index("accepted_prefix_publish"),
        )

    def test_graph_has_no_previous_attention_output_dependency(self):
        for stage in feasibility.STAGES:
            self.assertNotIn("previous_attention_output", stage.dependencies)

    def test_all_ratio_phases_and_exact_widths_are_causal(self):
        audit = feasibility.audit_all_schedules()
        self.assertEqual(audit["scenarios"], (4 + 128) * 4)
        self.assertEqual(audit["rows"], (4 + 128) * (2 + 3 + 4 + 5))
        self.assertGreater(audit["boundary_visibility_checks"], 0)

    def test_each_row_sees_only_its_proposal_prefix(self):
        rows = feasibility.causal_schedule(11, 5, 4)
        self.assertEqual(rows[0]["visible_proposal_positions"], (11,))
        self.assertEqual(rows[0]["hidden_future_positions"], (12, 13, 14, 15))
        self.assertEqual(
            rows[-1]["visible_proposal_positions"], (11, 12, 13, 14, 15)
        )
        self.assertEqual(rows[-1]["new_compressed_boundaries"], (11, 15))
        self.assertTrue(feasibility.validate_causal_schedule(rows, 4))

    def test_rejects_unsupported_width_or_ratio(self):
        with self.assertRaisesRegex(ValueError, "width"):
            feasibility.causal_schedule(0, 1, 4)
        with self.assertRaisesRegex(ValueError, "ratio"):
            feasibility.causal_schedule(0, 2, 8)

    def test_gate_stops_on_any_missing_requirement(self):
        gate = feasibility.SAVINGS_GATE_MS_PER_EMITTED
        self.assertEqual(
            feasibility.feasibility_gate(True, True, gate),
            "PROCEED_SHADOW_PROTOTYPE",
        )
        self.assertEqual(
            feasibility.feasibility_gate(False, True, 10.0), "STOP"
        )
        self.assertEqual(
            feasibility.feasibility_gate(True, False, 10.0), "STOP"
        )
        self.assertEqual(
            feasibility.feasibility_gate(True, True, gate - 0.001), "STOP"
        )


class BatchedVerifierCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = feasibility.analyze_summaries(*evidence_summaries())

    def test_pinned_artifacts_produce_a_prototype_gate(self):
        self.assertEqual(
            self.summary["feasibility_gate"], "PROCEED_SHADOW_PROTOTYPE"
        )
        self.assertEqual(self.summary["exact_widths"], [2, 3, 4, 5])
        self.assertEqual(len(self.summary["production_prerequisites"]), 5)

    def test_cost_model_excludes_width_one_from_addressable_budget(self):
        model = self.summary["cost_model"]
        self.assertAlmostEqual(model["multi_width_target_share"], 0.8117914432)
        self.assertGreater(
            model["target_ms_per_emitted"],
            model["multi_width_tail_budget_ms_per_emitted"],
        )

    def test_directional_projection_evidence_clears_gate(self):
        model = self.summary["cost_model"]
        self.assertGreater(
            model["projection_directional_savings_ms_per_emitted"],
            feasibility.SAVINGS_GATE_MS_PER_EMITTED,
        )
        self.assertLess(
            model["projection_directional_savings_ms_per_emitted"],
            model["deficit_ms_per_emitted"],
        )
        self.assertGreater(model["additional_tail_reduction_for_parity"], 0.0)
        self.assertGreater(
            model["additional_non_projection_reduction_for_parity"],
            model["additional_tail_reduction_for_parity"],
        )

    def test_ratio4_equivalence_is_reused_as_a_hard_gate(self):
        self.assertEqual(
            self.summary["ratio4_equivalence_digest"],
            "b96415b03cbffdf044aa2d4a3b172c22326b516efc0638f6f901546e9b628793",
        )

    def test_report_distinguishes_old_suffix_candidate(self):
        report = feasibility.render_report(self.summary)
        self.assertIn("PROCEED_SHADOW_PROTOTYPE", report)
        self.assertIn("old suffix candidate is not this design", report)
        self.assertIn("no model process or benchmark was run", report)

    def test_source_still_has_serial_attention_and_batched_suffix_boundaries(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        self.assertIn(
            "for (uint32_t row = 0; ok && row < n_tokens; row++)", source
        )
        self.assertIn("Batch only the row-independent output projection", source)
        self.assertIn("metal_graph_exact_attention_suffix_batch", source)
        self.assertIn("metal_graph_exact_attention_pre_batch_prepare", source)


if __name__ == "__main__":
    unittest.main()
