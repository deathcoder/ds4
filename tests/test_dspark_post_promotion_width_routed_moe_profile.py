#!/usr/bin/env python3
"""Model-free tests for the exact routed-MoE width profile."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_post_promotion_width_routed_moe_profile as profile  # noqa: E402


class DSparkPostPromotionWidthRoutedMoEProfileTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(profile.TASK, "humaneval_079")
        self.assertEqual(profile.LAYERS, (0, 21, 42))
        self.assertEqual(profile.WIDTHS, (2, 3, 4, 5))
        self.assertEqual(
            profile.MOE_STAGES,
            ("gate_up", "activation_weight", "down", "sum"),
        )

    def test_environment_enables_only_required_profiles(self):
        env = profile.profile_env(21)
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"], "21")
        self.assertEqual(env["DS4_METAL_LAYER_STAGE_PROFILE_LAYER"], "21")
        self.assertEqual(env["DS4_METAL_MOE_ONE_STAGE_PROFILE"], "1")
        self.assertEqual(env["DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER"], "21")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)

    def test_parser_reads_exact_and_moe_records_in_sequence(self):
        data = b"\n".join((
            b"ds4: metal layer stage part=exact layer=21 pos=9 tokens=5 "
            b"attention_tail_serial=1.000 ms",
            b"ds4: Metal routed MoE one stage layer=21 pairs=6 experts=6 "
            b"gate=iq2_xxs down=q2_k path=iq2_slots6_pair_swiglu "
            b"gate_up=0.250 ms",
        ))
        records = profile.parse_profile(data, 21, Path("synthetic.stderr"))
        self.assertEqual([row["part"] for row in records], ["exact", "moe_one"])
        self.assertEqual(records[1]["stage"], "gate_up")
        self.assertEqual(records[1]["path"], "iq2_slots6_pair_swiglu")

    @staticmethod
    def stats():
        return {"verify_width_evals": [0, 0, 1, 1, 2, 3]}

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
                records.append({
                    "sequence": sequence,
                    "part": "ffn",
                    "layer": layer,
                    "pos": pos,
                    "tokens": width,
                    "stage": "router",
                    "ms": 1.0,
                })
                sequence += 1
                for row in range(width):
                    for index, stage in enumerate(profile.MOE_STAGES, start=1):
                        records.append({
                            "sequence": sequence,
                            "part": "moe_one",
                            "layer": layer,
                            "pairs": 6,
                            "experts": 6,
                            "gate": "iq2_xxs",
                            "down": "q2_k",
                            "path": "iq2_slots6_pair_swiglu",
                            "stage": stage,
                            "ms": float(index + row),
                        })
                        sequence += 1
                records.append({
                    "sequence": sequence,
                    "part": "ffn",
                    "layer": layer,
                    "pos": pos,
                    "tokens": width,
                    "stage": "routed_moe",
                    "ms": 10.0 * width,
                })
                sequence += 1
                for stage in ("shared_gate_up", "shared_down", "hc_post"):
                    records.append({
                        "sequence": sequence,
                        "part": "ffn",
                        "layer": layer,
                        "pos": pos,
                        "tokens": width,
                        "stage": stage,
                        "ms": 1.0,
                    })
                    sequence += 1
                records.append({
                    "sequence": sequence,
                    "part": "exact",
                    "layer": layer,
                    "pos": pos,
                    "tokens": width,
                    "stage": "ffn_batch",
                    "ms": 20.0 * width,
                })
                sequence += 1
        return records

    def test_mapping_assigns_one_sequence_per_exact_row(self):
        assigned = profile.assign_exact_moe_batches(
            self.records(), self.stats(), 21
        )
        width5 = [row for row in assigned if row["width"] == 5]
        self.assertEqual(len(width5), 3 * (5 * 4 + 1))
        first_batch = [row for row in width5 if row["batch"] == 5]
        self.assertEqual(
            sum(row["stage"] == "gate_up" for row in first_batch), 5
        )
        control = next(
            row for row in first_batch if row["stage"] == "routed_moe_control"
        )
        self.assertEqual(control["ms"], 10.0)

    def test_mapping_ignores_moe_records_before_exact_tail(self):
        records = self.records()
        tail = next(
            row for row in records
            if row["part"] == "exact" and row["tokens"] == 5
            and row["stage"] == "attention_tail_serial"
        )
        for offset, stage in enumerate(profile.MOE_STAGES):
            records.append({
                "sequence": tail["sequence"] - 0.8 + offset * 0.1,
                "part": "moe_one",
                "layer": 21,
                "pairs": 6,
                "experts": 6,
                "gate": "iq2_xxs",
                "down": "q2_k",
                "path": "iq2_slots6_pair_swiglu",
                "stage": stage,
                "ms": 999.0,
            })
        records.sort(key=lambda row: row["sequence"])
        assigned = profile.assign_exact_moe_batches(records, self.stats(), 21)
        self.assertFalse(any(row["ms"] == 999.0 for row in assigned))

    def test_mapping_rejects_missing_inner_stage(self):
        records = self.records()
        removed = False
        filtered = []
        for row in records:
            if (
                not removed and row["part"] == "moe_one"
                and row["stage"] == "down"
            ):
                removed = True
                continue
            filtered.append(row)
        with self.assertRaisesRegex(RuntimeError, "stage sequence mismatch"):
            profile.assign_exact_moe_batches(filtered, self.stats(), 21)

    def test_summary_reports_width5_stage_share(self):
        assigned = []
        for layer in profile.LAYERS:
            assigned.extend(
                profile.assign_exact_moe_batches(
                    self.records(layer), self.stats(), layer
                )
            )
        summary = profile.summarize(assigned, self.stats())
        self.assertEqual(summary["largest_width5_stage"], "sum")
        self.assertEqual(summary["sampled_width_totals"]["5"]["rows"], 15)
        self.assertAlmostEqual(
            sum(
                item["share"]
                for item in summary["width5_components"].values()
            ),
            1.0,
        )
        report = profile.render_report(summary)
        self.assertIn("Exact Routed-MoE Stage Profile", report)
        self.assertIn("No throughput benchmark", report)


if __name__ == "__main__":
    unittest.main()
