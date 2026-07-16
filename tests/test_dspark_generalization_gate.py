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
import run_dspark_generalization_attribution as attribution  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


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
        self.assertEqual(gate.THRESHOLDS["fixed_k5"], "0")
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


class DSparkGeneralizationAttributionTests(unittest.TestCase):
    def test_frozen_tasks_are_domain_extremes(self):
        summary = {
            "tasks": {
                "math500_00166": {
                    "domain": "math", "scheduled_vs_baseline": 0.61,
                },
                "gsm8k_00333": {
                    "domain": "math", "scheduled_vs_baseline": 0.80,
                },
                "math_other_1": {
                    "domain": "math", "scheduled_vs_baseline": 0.70,
                },
                "math_other_2": {
                    "domain": "math", "scheduled_vs_baseline": 0.71,
                },
                "math_other_3": {
                    "domain": "math", "scheduled_vs_baseline": 0.72,
                },
                "math_other_4": {
                    "domain": "math", "scheduled_vs_baseline": 0.73,
                },
                "mt_bench_00075": {
                    "domain": "chat", "scheduled_vs_baseline": 0.56,
                },
                "alpaca_00115": {
                    "domain": "chat", "scheduled_vs_baseline": 0.76,
                },
                "chat_other_1": {
                    "domain": "chat", "scheduled_vs_baseline": 0.60,
                },
                "chat_other_2": {
                    "domain": "chat", "scheduled_vs_baseline": 0.61,
                },
                "chat_other_3": {
                    "domain": "chat", "scheduled_vs_baseline": 0.62,
                },
                "chat_other_4": {
                    "domain": "chat", "scheduled_vs_baseline": 0.63,
                },
            }
        }
        self.assertEqual(
            attribution.frozen_extremes(summary),
            dict(attribution.TASK_ROLES),
        )

    def test_summary_separates_invocation_and_component_cost(self):
        def row(task, depth, rounds, target_ms, sidecar_ms):
            values = {
                field: 0 for field in common.STATS_FIELDS
            }
            values.update({
                "prompt": task,
                "emitted": 100,
                "multi_attempts": rounds,
                "avg_depth": depth,
                "target_evals": 25,
                "target_eval_tokens": 100,
                "target_evals_avoided": 75,
                "target_eval_ms": target_ms,
                "generation_sidecar_ms": sidecar_ms,
                "batch_attempts": 20,
                "batch_full": 10,
                "batch_partial": 10,
            })
            return values

        rows = [
            row("math500_00166", 2.0, 50, 8000.0, 1000.0),
            row("gsm8k_00333", 4.0, 25, 4000.0, 500.0),
            row("mt_bench_00075", 2.5, 40, 6000.0, 800.0),
            row("alpaca_00115", 5.0, 20, 3000.0, 400.0),
        ]
        reference = {
            "tasks": {
                task: {
                    "role": role,
                    "prior": {
                        "domain": role.split("_")[0],
                        "dataset": "synthetic",
                        "source_index": index,
                        "scheduled_vs_baseline": 0.5,
                        "scheduled_vs_fixed": 1.1,
                        "baseline_generation_tps": 20.0,
                        "scheduled_generation_tps": 10.0,
                    },
                }
                for index, (role, task) in enumerate(
                    attribution.TASK_ROLES
                )
            }
        }
        summary = attribution.summarize(rows, reference)
        self.assertEqual(
            summary["domains"]["math"]["low_high_ratios"][
                "proposal_rounds_per_emitted"
            ],
            2.0,
        )
        self.assertEqual(
            summary["domains"]["chat"]["low_high_ratios"][
                "target_ms_per_emitted"
            ],
            2.0,
        )
        report = attribution.render_report(summary)
        self.assertIn("Throughput values from these runs are intentionally omitted", report)
        self.assertIn("## Low/High Contrast", report)

    def test_promoted_command_uses_runtime_default(self):
        args = type("Args", (), {
            "binary": Path("/tmp/ds4"),
            "model": Path("/tmp/base.gguf"),
            "dspark_model": Path("/tmp/dspark.gguf"),
            "ctx": 16384,
            "tokens": 128,
            "nothink": True,
            "fast_verifier": False,
            "exact_head_batch": False,
            "confidence_threshold": None,
        })()
        command = common.command_text(
            args, Path("/tmp/prompt.txt"), "runtime", stats=True
        )
        self.assertIn("DS4_DSPARK_GPU_RUNTIME_STATS=1", command)
        self.assertNotIn("DS4_DSPARK_CONFIDENCE_THRESHOLD", command)


if __name__ == "__main__":
    unittest.main()
