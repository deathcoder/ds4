#!/usr/bin/env python3
"""Model-free tests for the upstream/current baseline comparison."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_ds4_master_baseline_comparison as comparison  # noqa: E402


def args():
    return SimpleNamespace(
        current_binary=Path("/tmp/current-ds4"),
        upstream_binary=Path("/tmp/upstream-ds4"),
        upstream_source=Path("/tmp/upstream-source"),
        model=Path("/tmp/model.gguf"),
        ctx=16384,
        tokens=128,
    )


class Ds4MasterBaselineComparisonTests(unittest.TestCase):
    def test_protocol_and_gate_are_frozen(self):
        self.assertEqual(comparison.SAMPLE_COUNT, 32)
        self.assertEqual(
            comparison.MODES, ("upstream_main", "current_branch")
        )
        self.assertEqual(
            comparison.EXPECTED_UPSTREAM_COMMIT,
            "80ebbc396aee40eedc1d829222f3362d10fa4c6c",
        )
        self.assertEqual(comparison.MIN_GEOMEAN, 1.01)
        self.assertEqual(comparison.MIN_WINS, 24)
        self.assertEqual(comparison.MIN_TASK_RATIO, 0.95)

    def test_order_and_warmups_are_balanced(self):
        self.assertEqual(
            comparison.mode_order(1), ("upstream_main", "current_branch")
        )
        self.assertEqual(
            comparison.mode_order(2), ("current_branch", "upstream_main")
        )
        records = [{"label": "first"}, {"label": "last"}]
        warmups = comparison.warmup_schedule(records)
        self.assertEqual(warmups[0][1], comparison.MODES)
        self.assertEqual(warmups[1][1], tuple(reversed(comparison.MODES)))

    def test_commands_use_distinct_binaries_without_dspark(self):
        prompt = Path("/tmp/prompt.txt")
        upstream = comparison.command_text(args(), prompt, "upstream_main")
        current = comparison.command_text(args(), prompt, "current_branch")
        self.assertTrue(upstream.startswith("/tmp/upstream-ds4 "))
        self.assertTrue(current.startswith("/tmp/current-ds4 "))
        self.assertIn("--backend metal", upstream)
        self.assertNotIn("--dspark", upstream)
        self.assertNotIn("--dspark", current)

    def test_each_binary_uses_its_own_source_working_directory(self):
        root = Path("/tmp/current-source")
        self.assertEqual(
            comparison.working_directory_for_mode(
                args(), root, "upstream_main"
            ),
            Path("/tmp/upstream-source"),
        )
        self.assertEqual(
            comparison.working_directory_for_mode(
                args(), root, "current_branch"
            ),
            root,
        )

    def test_environment_clears_ds4_instrumentation(self):
        env = comparison.mode_env()
        self.assertFalse(any(key.startswith("DS4_") for key in env))

    @staticmethod
    def records():
        return [
            {"label": f"task_{index:02d}", "source_index": index}
            for index in range(comparison.SAMPLE_COUNT)
        ]

    @staticmethod
    def rows(records, ratio):
        rows = []
        for index, record in enumerate(records, start=1):
            order = "-".join(comparison.mode_order(index))
            rows.extend((
                {
                    "prompt": record["label"],
                    "mode": "upstream_main",
                    "generation_tps": 20.0,
                    "pair_order": order,
                },
                {
                    "prompt": record["label"],
                    "mode": "current_branch",
                    "generation_tps": 20.0 * ratio,
                    "pair_order": order,
                },
            ))
        return rows

    def test_summary_passes_meaningful_progress_gate(self):
        records = self.records()
        summary = comparison.summarize(self.rows(records, 1.02), records)
        self.assertTrue(summary["progress_gate"]["pass"])
        self.assertEqual(summary["current_faster_tasks"], 32)
        report = comparison.render_report(summary, "upstream", "current")
        self.assertIn("ordinary Metal decoding", report)
        self.assertIn("**PASS**", report)

    def test_summary_rejects_neutral_movement(self):
        records = self.records()
        summary = comparison.summarize(self.rows(records, 1.005), records)
        self.assertFalse(summary["progress_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
