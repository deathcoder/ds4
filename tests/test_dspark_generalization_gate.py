#!/usr/bin/env python3
"""Model-free tests for the frozen DSpark math/chat gate."""

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_generalization_gate as gate  # noqa: E402


class DSparkGeneralizationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = ROOT / "speed-bench/dspark-generalization"
        cls.samples_data = (corpus / "samples.jsonl").read_bytes()
        cls.records = [
            json.loads(line) for line in cls.samples_data.splitlines()
        ]
        cls.provenance = json.loads(
            (corpus / "provenance.json").read_text(encoding="utf-8")
        )

    def test_corpus_hash_and_domain_balance_are_frozen(self):
        self.assertEqual(len(self.records), 12)
        self.assertEqual(
            hashlib.sha256(self.samples_data).hexdigest(),
            self.provenance["samples_file_sha256"],
        )
        domains = [record["domain"] for record in self.records]
        self.assertEqual(domains.count("math"), 6)
        self.assertEqual(domains.count("chat"), 6)
        self.assertEqual(
            {record["dataset"] for record in self.records},
            {"gsm8k", "math500", "aime25", "alpaca", "mt-bench"},
        )
        metadata = {
            item["dataset"]: item for item in self.provenance["datasets"]
        }
        for record in self.records:
            self.assertLess(
                record["source_index"],
                metadata[record["dataset"]]["evaluator_rows"],
            )

    def test_mode_order_is_balanced_across_tasks(self):
        orders = [gate.rotated_modes(index) for index in range(12)]
        for mode in gate.MODES:
            positions = [order.index(mode) for order in orders]
            self.assertEqual(positions.count(0), 4)
            self.assertEqual(positions.count(1), 4)
            self.assertEqual(positions.count(2), 4)

    def test_threshold_is_predeclared_only_for_scheduled_mode(self):
        self.assertIsNone(gate.THRESHOLDS["baseline"])
        self.assertIsNone(gate.THRESHOLDS["fixed_k5"])
        self.assertEqual(gate.THRESHOLDS["threshold_0455"], "0.455")
        self.assertEqual(gate.target_mode("baseline"), "baseline")
        self.assertEqual(gate.target_mode("fixed_k5"), "runtime")
        self.assertEqual(gate.target_mode("threshold_0455"), "runtime")

    def test_summary_reports_all_three_controlled_ratios(self):
        rows = []
        for record in self.records:
            for mode, throughput in (
                ("baseline", 10.0),
                ("fixed_k5", 8.0),
                ("threshold_0455", 9.0),
            ):
                rows.append({
                    "task": record["label"],
                    "mode": mode,
                    "generation_tps": throughput,
                    "order": "baseline-fixed_k5-threshold_0455",
                })
        summary = gate.summarize(rows, self.records)
        for domain in ("math", "chat", "overall"):
            aggregate = summary["aggregates"][domain]
            self.assertAlmostEqual(
                aggregate["fixed_vs_baseline"]["median"], 0.8
            )
            self.assertAlmostEqual(
                aggregate["scheduled_vs_baseline"]["median"], 0.9
            )
            self.assertAlmostEqual(
                aggregate["scheduled_vs_fixed"]["median"], 1.125
            )
        self.assertTrue(summary["promotion_gate"]["passed"])
        report = gate.render_report(summary)
        self.assertIn("## Promotion Gate", report)
        self.assertIn("**PASS**", report)


if __name__ == "__main__":
    unittest.main()
