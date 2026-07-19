#!/usr/bin/env python3
"""Model-free tests for the threshold-0.75 width-layer profile."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_exact_layer_profile as layer_profile  # noqa: E402
import run_dspark_threshold075_width_layer_profile as profile  # noqa: E402


class DSparkThreshold075WidthLayerProfileTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(profile.TASK, "humaneval_079")
        self.assertEqual(profile.LAYERS, (0, 21, 42))
        self.assertEqual(profile.WIDTHS, (2, 3, 4, 5))
        self.assertEqual(
            profile.CUMULATIVE_COST_SOURCE_COMMIT,
            "80ce67c00fb43e69a38893db03e589c67193ec50",
        )

    def test_cost_reference_contracts_distinguish_post_promotion(self):
        legacy = {
            "experiment": profile.LEGACY_COST_CONTRACT[0],
        }
        cumulative = {
            "experiment": profile.CUMULATIVE_COST_CONTRACT[0],
        }
        self.assertEqual(
            profile.cost_reference_kind(
                {"analysis": profile.LEGACY_COST_CONTRACT[1]}, legacy
            ),
            "legacy_threshold075",
        )
        self.assertEqual(
            profile.cost_reference_kind(
                {"analysis": profile.CUMULATIVE_COST_CONTRACT[1]}, cumulative
            ),
            "post_promotion_cumulative",
        )

    def test_profile_environment_is_exact_and_instrumented(self):
        env = profile.profile_env(21)
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_STATS"], "1")
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE"], "1")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"], "21")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", env)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_GATHERED_LEGACY", env)

    @staticmethod
    def records():
        rows = []
        counts = {2: 1, 3: 1, 4: 2, 5: 3}
        base = {
            "attention_pre_batch": 0.2,
            "attention_tail_serial": 0.5,
            "ffn_batch": 0.4,
        }
        for layer in profile.LAYERS:
            for width, count in counts.items():
                for batch in range(count):
                    for stage in layer_profile.EXACT_STAGES:
                        rows.append({
                            "part": "exact",
                            "layer": layer,
                            "pos": batch * 10,
                            "tokens": width,
                            "stage": stage,
                            "ms": base[stage] * width,
                        })
        return rows

    @staticmethod
    def stats():
        evals = [0, 0, 1, 1, 2, 3]
        return {"verify_width_evals": evals}

    def test_summary_groups_stages_by_actual_width(self):
        summary = profile.summarize(self.records(), self.stats())
        layer = summary["layer_widths"]["21"]["5"]
        self.assertEqual(layer["batches"], 3)
        self.assertAlmostEqual(layer["total_ms_per_row"], 1.1)
        self.assertAlmostEqual(layer["per_row_vs_width2"], 1.0)
        self.assertEqual(
            summary["largest_width5_stage"], "attention_tail_serial"
        )

    def test_summary_detects_missing_width_records(self):
        rows = [
            row for row in self.records()
            if not (
                row["layer"] == 42 and
                row["tokens"] == 5 and
                row["stage"] == "ffn_batch"
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "expected 3"):
            profile.summarize(rows, self.stats())

    def test_report_warns_about_sparse_widths(self):
        summary = profile.summarize(self.records(), self.stats())
        summary["reference_kind"] = "post_promotion_cumulative"
        report = profile.render_report(summary)
        self.assertIn("Post-Promotion Width-Stratified", report)
        self.assertIn("frozen cumulative HumanEval artifact", report)
        self.assertIn("Synchronized diagnostic only", report)
        self.assertIn("Widths 2 and 3 have one observation each", report)
        self.assertIn("No fresh throughput benchmark", report)


if __name__ == "__main__":
    unittest.main()
