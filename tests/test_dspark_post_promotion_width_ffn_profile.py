#!/usr/bin/env python3
"""Model-free tests for the post-promotion exact-FFN width profile."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_post_promotion_width_ffn_profile as profile  # noqa: E402


class DSparkPostPromotionWidthFFNProfileTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(profile.TASK, "humaneval_079")
        self.assertEqual(profile.LAYERS, (0, 21, 42))
        self.assertEqual(profile.WIDTHS, (2, 3, 4, 5))
        self.assertEqual(len(profile.FFN_STAGES), 7)
        self.assertEqual(
            profile.WIDTH_LAYER_SOURCE_COMMIT,
            "0286603f84183dacebbacad72f86745a7baa3935",
        )

    def test_environment_enables_only_required_profiles(self):
        env = profile.profile_env(21)
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_STATS"], "1")
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"], "21")
        self.assertEqual(env["DS4_METAL_LAYER_STAGE_PROFILE_LAYER"], "21")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", env)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_GATHERED_LEGACY", env)

    @staticmethod
    def records(layer=21):
        records = []
        sequence = 0
        counts = {2: 1, 3: 1, 4: 2, 5: 3}
        for width, count in counts.items():
            for batch in range(count):
                pos = width * 100 + batch * 10
                records.append({
                    "sequence": sequence,
                    "part": "exact",
                    "layer": layer,
                    "pos": pos,
                    "tokens": width,
                    "stage": "attention_tail_serial",
                    "ms": 1.0,
                })
                sequence += 1
                for index, stage in enumerate(profile.FFN_STAGES, start=1):
                    records.append({
                        "sequence": sequence,
                        "part": "ffn",
                        "layer": layer,
                        "pos": pos,
                        "tokens": width,
                        "stage": stage,
                        "ms": float(index * width),
                    })
                    sequence += 1
                records.append({
                    "sequence": sequence,
                    "part": "exact",
                    "layer": layer,
                    "pos": pos,
                    "tokens": width,
                    "stage": "ffn_batch",
                    "ms": 30.0 * width,
                })
                sequence += 1
        return records

    @staticmethod
    def stats():
        return {"verify_width_evals": [0, 0, 1, 1, 2, 3]}

    def test_mapping_uses_enclosing_exact_ffn_control(self):
        assigned = profile.assign_exact_ffn_batches(
            self.records(), self.stats(), 21
        )
        width5 = [row for row in assigned if row["width"] == 5]
        self.assertEqual(len(width5), 3 * 8)
        self.assertEqual(
            {row["stage"] for row in width5},
            set(profile.FFN_STAGES) | {"ffn_batch_control"},
        )
        routed = next(row for row in width5 if row["stage"] == "routed_moe")
        self.assertEqual(routed["ms_per_row"], 6.0)

    def test_mapping_ignores_same_key_stages_before_exact_tail(self):
        records = self.records()
        tail_index = next(
            index for index, row in enumerate(records)
            if row["part"] == "exact" and row["stage"] == "attention_tail_serial"
            and row["tokens"] == 5
        )
        sequence = records[tail_index]["sequence"]
        duplicates = []
        for offset, stage in enumerate(profile.FFN_STAGES):
            duplicates.append({
                "sequence": sequence - 0.9 + offset * 0.1,
                "part": "ffn",
                "layer": 21,
                "pos": records[tail_index]["pos"],
                "tokens": 5,
                "stage": stage,
                "ms": 999.0,
            })
        records.extend(duplicates)
        records.sort(key=lambda row: row["sequence"])
        assigned = profile.assign_exact_ffn_batches(records, self.stats(), 21)
        first_width5 = [
            row for row in assigned
            if row["width"] == 5 and row["batch"] == 5
            and row["stage"] != "ffn_batch_control"
        ]
        self.assertTrue(first_width5)
        self.assertTrue(all(row["ms"] != 999.0 for row in first_width5))

    def test_mapping_rejects_missing_internal_stage(self):
        records = [
            row for row in self.records()
            if not (
                row["tokens"] == 5 and row["pos"] == 500
                and row["stage"] == "router"
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "stages mismatch"):
            profile.assign_exact_ffn_batches(records, self.stats(), 21)

    def test_summary_reports_largest_and_weakest_stage(self):
        assigned = []
        for layer in profile.LAYERS:
            assigned.extend(
                profile.assign_exact_ffn_batches(
                    self.records(layer), self.stats(), layer
                )
            )
        summary = profile.summarize(assigned, self.stats())
        self.assertEqual(summary["largest_width5_stage"], "hc_post")
        self.assertEqual(summary["weakest_amortization_stage"], "hc_pre")
        self.assertAlmostEqual(
            summary["sampled_width_totals"]["5"]["substage_vs_control"],
            28.0 / 30.0,
        )
        report = profile.render_report(summary)
        self.assertIn("Width-Stratified Exact FFN Profile", report)
        self.assertIn("mapped to the enclosing exact-verifier", report)
        self.assertIn("No throughput benchmark", report)


if __name__ == "__main__":
    unittest.main()
