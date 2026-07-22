#!/usr/bin/env python3
"""Model-free tests for ratio-4 materialized compressor state."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_dspark_ratio4_materialized_equivalence as equivalence  # noqa: E402


class Ratio4MaterializedEquivalenceTests(unittest.TestCase):
    def test_all_exact_proposal_widths_and_frontier_phases_pass(self):
        summary = equivalence.run_equivalence()
        self.assertEqual(summary["proposal_widths"], [2, 3, 4, 5])
        self.assertEqual(summary["frontier_phases"], [0, 1, 2, 3])
        self.assertEqual(summary["scenario_count"], 16)
        self.assertEqual(summary["frontier_bitwise_checks"], 56)
        self.assertEqual(summary["boundary_row_and_reduction_checks"], 14)
        self.assertEqual(summary["representation_gate"], "PASS")
        self.assertEqual(
            summary["operation_order_gate"], "PASS_DS4_SCALAR_ORDER"
        )

    def test_materialized_boundary_uses_two_chronological_lanes(self):
        head_dim = 2
        ape = equivalence.synthetic_ape(head_dim)
        state = equivalence.MaterializedRatio4(head_dim)
        for position in range(12):
            state.update(equivalence.synthetic_partial(position, head_dim), ape)
        rows = state.compression_rows(11)
        expected = []
        for position in range(4, 8):
            kv, score = state.cache[position]
            expected.append((kv[:head_dim], score[:head_dim]))
        for position in range(8, 12):
            kv, score = state.cache[position]
            expected.append((kv[head_dim:], score[head_dim:]))
        self.assertEqual(
            equivalence.rows_bits(rows), equivalence.rows_bits(expected)
        )

    def test_independent_pool_implementations_are_bitwise_equal(self):
        head_dim = 5
        ape = equivalence.synthetic_ape(head_dim)
        state = equivalence.MaterializedRatio4(head_dim)
        for position in range(8):
            state.update(equivalence.synthetic_partial(position, head_dim), ape)
        rows = state.compression_rows(7)
        self.assertEqual(
            equivalence.values_bits(equivalence.stream_pool(rows, head_dim)),
            equivalence.values_bits(
                equivalence.materialized_pool(rows, head_dim)
            ),
        )

    def test_missing_previous_block_matches_ds4_empty_frontier(self):
        head_dim = 2
        ape = equivalence.synthetic_ape(head_dim)
        stream = equivalence.StreamingRatio4(head_dim)
        materialized = equivalence.MaterializedRatio4(head_dim)
        for position in range(4):
            partial = equivalence.synthetic_partial(position, head_dim)
            stream_emission = stream.update(partial, ape)
            materialized_emission = materialized.update(partial, ape)
        self.assertEqual(
            equivalence.rows_bits(stream_emission.rows),
            equivalence.rows_bits(materialized_emission.rows),
        )
        self.assertEqual(
            equivalence.values_bits(stream_emission.pooled),
            equivalence.values_bits(materialized_emission.pooled),
        )

    def test_report_stops_a_standalone_state_rewrite(self):
        summary = equivalence.run_equivalence()
        self.assertEqual(summary["standalone_runtime_gate"], "STOP")
        report = equivalence.render_report(summary)
        self.assertIn("Representation gate: **PASS**", report)
        self.assertIn("Standalone runtime gate: **STOP**", report)
        self.assertIn("serial exact attention", report)

    def test_local_metal_pack_has_the_same_lane_mapping(self):
        source = (ROOT / "metal" / "dsv4_kv.metal").read_text(encoding="utf-8")
        self.assertIn(
            "four previous-half rows followed by four current-half rows", source
        )
        self.assertIn("args.head_dim + col", source)
        self.assertIn("(plane - 1u) * 4u + row", source)


if __name__ == "__main__":
    unittest.main()
