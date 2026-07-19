#!/usr/bin/env python3
"""Model-free tests for the HumanEval compressor-prebatch gate."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_compressor_pre_batch as gate  # noqa: E402


def args():
    return SimpleNamespace(
        binary=Path("/tmp/ds4"),
        model=Path("/tmp/base.gguf"),
        dspark_model=Path("/tmp/dspark.gguf"),
        ctx=16384,
        tokens=128,
        fast_verifier=False,
        exact_head_batch=False,
    )


class HumanEvalCompressorPreBatchTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(gate.THRESHOLD, "0.75")
        self.assertEqual(gate.SAMPLE_COUNT, 32)
        self.assertEqual(
            gate.MODES, ("legacy_compressor_pre_batch", "default_exact")
        )
        self.assertEqual(gate.MIN_GEOMEAN, 1.005)
        self.assertEqual(gate.MIN_WINS, 20)
        self.assertEqual(gate.MIN_TASK_RATIO, 0.95)
        self.assertEqual(gate.MIN_LOW_ACCEPTANCE_GEOMEAN, 1.00)

    def test_order_and_warmups_are_balanced(self):
        self.assertEqual(
            gate.mode_order(1),
            ("legacy_compressor_pre_batch", "default_exact"),
        )
        self.assertEqual(
            gate.mode_order(2),
            ("default_exact", "legacy_compressor_pre_batch"),
        )
        records = [{"label": "first"}, {"label": "last"}]
        warmups = gate.warmup_schedule(records)
        self.assertEqual(warmups[0][1], gate.MODES)
        self.assertEqual(warmups[1][1], tuple(reversed(gate.MODES)))

    def test_only_legacy_reference_disables_compressor_prebatch(self):
        legacy = gate.mode_env("legacy_compressor_pre_batch")
        promoted = gate.mode_env("default_exact")
        self.assertEqual(
            legacy["DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH"], "0"
        )
        self.assertNotIn("DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH", promoted)
        self.assertEqual(
            promoted["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75"
        )
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", promoted)
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", promoted)
        self.assertNotIn(
            "DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH_TRACE", promoted
        )

    def test_commands_are_metal_and_uninstrumented(self):
        legacy = gate.command_text(
            args(), Path("/tmp/prompt.txt"), "legacy_compressor_pre_batch"
        )
        promoted = gate.command_text(
            args(), Path("/tmp/prompt.txt"), "default_exact"
        )
        self.assertIn("--backend metal", promoted)
        self.assertIn(
            "DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH=0", legacy
        )
        self.assertNotIn("DS4_DSPARK_EXACT_COMPRESSOR_PRE_BATCH", promoted)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", promoted)
        self.assertNotIn("TRACE", promoted)

    @staticmethod
    def records(count=32):
        return [
            {"label": f"task_{index:02d}", "source_index": index}
            for index in range(count)
        ]

    @staticmethod
    def reference(records):
        return {
            "tasks": {
                record["label"]: {"acceptance_verify_rate": 0.60}
                for record in records
            }
        }

    @staticmethod
    def rows(records, ratio):
        rows = []
        for index, record in enumerate(records):
            order = (
                "legacy_compressor_pre_batch-default_exact"
                if index % 2 == 0
                else "default_exact-legacy_compressor_pre_batch"
            )
            rows.extend((
                {
                    "prompt": record["label"],
                    "mode": "legacy_compressor_pre_batch",
                    "generation_tps": 10.0,
                    "pair_order": order,
                },
                {
                    "prompt": record["label"],
                    "mode": "default_exact",
                    "generation_tps": 10.0 * ratio,
                    "pair_order": order,
                },
            ))
        return rows

    def test_summary_passes_predeclared_gate(self):
        records = self.records()
        summary = gate.summarize(
            self.rows(records, 1.01), records, self.reference(records)
        )
        self.assertTrue(summary["promotion_gate"]["pass"])
        self.assertEqual(summary["candidate_faster_tasks"], 32)
        report = gate.render_report(summary)
        self.assertIn("Compressor Projection Prebatch Confirmation", report)
        self.assertIn("**PASS**", report)
        self.assertIn("No DSpark stats", report)

    def test_summary_rejects_subthreshold_geomean(self):
        records = self.records()
        summary = gate.summarize(
            self.rows(records, 1.004), records, self.reference(records)
        )
        self.assertFalse(summary["promotion_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
