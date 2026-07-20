#!/usr/bin/env python3
"""Model-free tests for the threshold-0.75 width-tail profile."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_exact_attention_tail_profile as tail_profile  # noqa: E402
import run_dspark_threshold075_width_tail_profile as profile  # noqa: E402


class DSparkThreshold075WidthTailProfileTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(profile.TASK, "humaneval_079")
        self.assertEqual(profile.LAYER, 42)
        self.assertEqual(profile.WIDTHS, (2, 3, 4, 5))
        self.assertEqual(
            profile.POST_PROMOTION_LAYER_SOURCE_COMMIT,
            "83f3e803e7baa4097cc8c5ff490f72b29aced06c",
        )

    def test_profile_environment_enables_tail_boundaries(self):
        env = profile.profile_env()
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_STATS"], "1")
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"], "42")
        self.assertEqual(env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"], "42")
        self.assertEqual(env["DS4_DSPARK_EXACT_TAIL_PROFILE"], "1")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)

    @staticmethod
    def records():
        signature = ((100, 2), (102, 3), (105, 4), (109, 5))
        rows = []
        for stage in tail_profile.CONTROL_STAGES:
            for start, width in signature:
                rows.append({
                    "part": "exact",
                    "layer": 42,
                    "pos": start,
                    "tokens": width,
                    "stage": stage,
                    "ms": float(width),
                })
        weights = {
            "kv_cache_update": 1.0,
            "compressor_indexer": 2.0,
            "attention": 3.0,
            "inverse_rope": 1.0,
            "projection_a": 2.0,
            "projection_b_hc": 2.0,
        }
        for start, width in signature:
            for row in range(width):
                for stage in tail_profile.TAIL_STAGES:
                    rows.append({
                        "part": "tail",
                        "layer": 42,
                        "pos": start + row,
                        "tokens": 1,
                        "stage": stage,
                        "ms": weights[stage],
                    })
        return rows

    @staticmethod
    def stats():
        return {"verify_width_evals": [0, 0, 1, 1, 1, 1]}

    def test_assignment_uses_batch_sequence(self):
        signature, assigned = profile.assign_tail_batches(self.records())
        self.assertEqual(signature[1], (102, 3))
        selected = [
            row for row in assigned
            if row["batch"] == 2 and row["stage"] == "attention"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["width"], 3)
        self.assertAlmostEqual(selected[0]["ms_per_row"], 3.0)

    def test_summary_reports_largest_component(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        self.assertEqual(summary["largest_width5_component"], "attention")
        self.assertAlmostEqual(
            summary["width_results"]["5"]["median_tail_ms_per_row"], 11.0
        )
        self.assertAlmostEqual(
            summary["width_results"]["5"]["components"]["attention"][
                "median_share"
            ],
            3.0 / 11.0,
        )

    def test_assignment_rejects_row_order_mismatch(self):
        rows = self.records()
        for row in rows:
            if (
                row["part"] == "tail" and
                row["stage"] == "attention" and
                row["pos"] == 103
            ):
                row["pos"] = 999
                break
        with self.assertRaisesRegex(RuntimeError, "positions"):
            profile.assign_tail_batches(rows)

    def test_report_warns_about_sparse_widths(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        report = profile.render_report(summary)
        self.assertIn("Synchronized diagnostic only", report)
        self.assertIn("Widths 2 and 3 have one batch each", report)
        self.assertIn("No fresh throughput benchmark", report)

    def test_post_promotion_report_has_distinct_identity(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        summary["reference_kind"] = "post_promotion_cumulative"
        report = profile.render_report(summary)
        self.assertIn("Post-Promotion Width-Stratified", report)
        self.assertIn("frozen cumulative HumanEval artifact", report)


if __name__ == "__main__":
    unittest.main()
