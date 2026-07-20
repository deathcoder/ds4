#!/usr/bin/env python3
"""Model-free tests for the width-stratified attention-route profile."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_exact_attention_tail_profile as tail_components  # noqa: E402
import run_dspark_threshold075_width_attention_profile as profile  # noqa: E402


class DSparkThreshold075WidthAttentionProfileTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(profile.TASK, "humaneval_079")
        self.assertEqual(profile.LAYER, 42)
        self.assertEqual(profile.WIDTHS, (2, 3, 4, 5))
        self.assertEqual(
            profile.TAIL_SOURCE_COMMIT,
            "a31f69b91545d82d2d881fc05128904ce37424c4",
        )

    def test_profile_environment_is_attention_only(self):
        env = profile.profile_env()
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_STATS"], "1")
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"], "42")
        self.assertEqual(env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"], "42")
        self.assertEqual(env["DS4_DSPARK_EXACT_ATTENTION_PROFILE"], "1")
        self.assertNotIn("DS4_DSPARK_EXACT_TAIL_PROFILE", env)
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)

    @staticmethod
    def records():
        signature = ((100, 2), (102, 3), (105, 4), (109, 5))
        rows = []
        for stage in tail_components.CONTROL_STAGES:
            for start, width in signature:
                rows.append({
                    "variant": "default_rb16_direct",
                    "part": "exact",
                    "layer": 42,
                    "pos": start,
                    "tokens": width,
                    "stage": stage,
                    "ms": float(width),
                })
        modes = (
            ("raw", "dense_mixed"),
            ("dense_mixed", "dense_mixed", "sparse_indexed"),
            ("dense_mixed",) * 4,
            ("dense_mixed",) * 4 + ("sparse_indexed",),
        )
        costs = {"raw": 1.0, "dense_mixed": 2.0, "sparse_indexed": 4.0}
        for (start, width), batch_modes in zip(signature, modes):
            for offset, mode in enumerate(batch_modes):
                rows.append({
                    "variant": "default_rb16_direct",
                    "part": "attention",
                    "layer": 42,
                    "pos": start + offset,
                    "tokens": 1,
                    "stage": mode,
                    "ms": costs[mode],
                })
        return rows

    @staticmethod
    def stats():
        return {"verify_width_evals": [0, 0, 1, 1, 1, 1]}

    def test_assignment_preserves_batch_width_and_route(self):
        signature, assigned = profile.assign_attention_batches(self.records())
        self.assertEqual(signature[-1], (109, 5))
        last = [row for row in assigned if row["batch"] == 4]
        self.assertEqual([row["pos"] for row in last], [109, 110, 111, 112, 113])
        self.assertEqual(last[-1]["mode"], "sparse_indexed")

    def test_summary_reports_width5_cost_owner(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        width5 = summary["width_results"]["5"]
        self.assertEqual(width5["modes"]["dense_mixed"]["rows"], 4)
        self.assertEqual(width5["modes"]["sparse_indexed"]["rows"], 1)
        self.assertEqual(summary["dominant_width5_cost_mode"], "dense_mixed")
        self.assertEqual(summary["slowest_width5_mode"], "sparse_indexed")

    def test_assignment_rejects_row_order_mismatch(self):
        rows = self.records()
        for row in rows:
            if row["part"] == "attention" and row["pos"] == 103:
                row["pos"] = 999
                break
        with self.assertRaisesRegex(RuntimeError, "positions"):
            profile.assign_attention_batches(rows)

    def test_report_keeps_attribution_boundary(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        report = profile.render_report(summary)
        self.assertIn("Width-Stratified Attention Route Profile", report)
        self.assertIn("dense mixed", report)
        self.assertIn("Synchronized absolute timings", report)
        self.assertIn("No runtime candidate", report)


if __name__ == "__main__":
    unittest.main()
