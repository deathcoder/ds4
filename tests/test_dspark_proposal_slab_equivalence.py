#!/usr/bin/env python3
"""Model-free tests for ratio-128 and staged proposal ownership."""

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_dspark_proposal_slab_equivalence as equivalence  # noqa: E402


class ProposalSlabEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = equivalence.run_analysis()

    def test_ratio128_all_widths_and_phases_pass(self):
        summary = self.summary["ratio128"]
        self.assertEqual(summary["scenario_count"], 512)
        self.assertEqual(summary["frontier_bitwise_checks"], 1792)
        self.assertEqual(summary["boundary_row_and_reduction_checks"], 14)
        self.assertEqual(summary["gate"], "PASS_SOURCE_ORDER")

    def test_ratio128_boundary_uses_chronological_physical_rows(self):
        head_dim = 2
        ape = equivalence.synthetic_ape128(head_dim)
        state = equivalence.MaterializedRatio128(head_dim)
        for position in range(256):
            state.update(
                equivalence.synthetic_partial128(position, head_dim), ape
            )
        rows = state.compression_rows(255)
        expected = tuple(state.cache[position] for position in range(128, 256))
        self.assertEqual(
            equivalence.ratio4.rows_bits(rows),
            equivalence.ratio4.rows_bits(expected),
        )

    def test_ratio128_independent_reductions_are_bitwise_equal(self):
        head_dim = 4
        ape = equivalence.synthetic_ape128(head_dim)
        state = equivalence.MaterializedRatio128(head_dim)
        for position in range(128):
            state.update(
                equivalence.synthetic_partial128(position, head_dim), ape
            )
        rows = state.compression_rows(127)
        self.assertEqual(
            equivalence.ratio4.values_bits(
                equivalence.stream_pool128(rows, head_dim)
            ),
            equivalence.ratio4.values_bits(
                equivalence.materialized_pool128(rows, head_dim)
            ),
        )

    def test_proposal_slab_exhaustive_contract_passes(self):
        summary = self.summary["proposal_slab"]
        self.assertEqual(summary["slab_scenario_count"], 528)
        self.assertEqual(summary["causal_row_view_checks"], 1848)
        self.assertEqual(summary["accepted_prefix_commit_cases"], 2376)
        self.assertEqual(summary["post_rejection_continuation_checks"], 4752)
        self.assertGreater(summary["rejected_boundary_isolation_checks"], 0)
        self.assertEqual(summary["visibility_gate"], "PASS_CAUSAL_PREFIX")
        self.assertEqual(
            summary["publication_gate"], "PASS_ACCEPTED_PREFIX"
        )
        self.assertEqual(
            summary["continuation_gate"],
            "PASS_REJECTED_STATE_ISOLATION",
        )

    def test_future_staged_rows_are_invisible_to_an_earlier_view(self):
        base, start = equivalence.seed_streaming(128, 126, 2)
        proposals = tuple(
            equivalence.synthetic_partial(position, 128, 2)
            for position in range(start, start + 5)
        )
        slab = equivalence.ProposalSlab(base, proposals)
        expected = copy.deepcopy(base)
        expected.apply(proposals[0])
        self.assertEqual(slab.view(1), expected.snapshot())
        self.assertEqual(slab.view(1).logical_position, start)
        self.assertNotIn(start + 1, slab.view(1).raw_visible)

    def test_rejected_boundary_can_be_replaced_without_state_leak(self):
        base, start = equivalence.seed_streaming(4, 2, 2)
        proposals = tuple(
            equivalence.synthetic_partial(position, 4, 2)
            for position in range(start, start + 5)
        )
        slab = equivalence.ProposalSlab(base, proposals)
        committed = slab.commit(1)
        expected = copy.deepcopy(base)
        expected.apply(proposals[0])
        for offset in range(2):
            position = start + 1 + offset
            replacement = equivalence.synthetic_partial(
                position, 4, 2, salt=123 + offset
            )
            committed.apply(replacement)
            expected.apply(replacement)
            self.assertEqual(committed.snapshot(), expected.snapshot())

    def test_report_authorizes_only_the_shadow_observer(self):
        self.assertEqual(
            self.summary["contract_gate"], "PASS_MODEL_FREE_CONTRACT"
        )
        self.assertEqual(
            self.summary["next_gate"],
            "PROCEED_SINGLE_LAYER_SHADOW_OBSERVER",
        )
        report = equivalence.render_report(self.summary)
        self.assertIn("no model process or timing benchmark", report)
        self.assertIn("single-layer shadow observer only", report)

    def test_local_source_uses_ratio128_modulo_rows_and_ordered_pool(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        metal = (ROOT / "metal" / "dsv4_misc.metal").read_text(
            encoding="utf-8"
        )
        self.assertIn("const uint32_t row = compress_ratio == 4", source)
        self.assertIn("pos_mod : pos_mod", source)
        self.assertGreaterEqual(
            metal.count("for (int64_t ir = 0; ir < args.ne00; ++ir)"), 2
        )
        self.assertIn("acc += v*w", metal)


if __name__ == "__main__":
    unittest.main()
