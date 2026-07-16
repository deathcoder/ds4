#!/usr/bin/env python3
"""Model-free tests for exact prefix-checkpoint attribution."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_checkpoint_attribution as attribution  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class DSparkCheckpointAttributionTests(unittest.TestCase):
    @staticmethod
    def blank_stats():
        values = {field: 0 for field in common.INT_STATS | common.FLOAT_STATS}
        for field in common.INT_ARRAY_STATS:
            values[field] = [0] * 6
        for field in common.FLOAT_ARRAY_STATS:
            values[field] = [0.0] * 6
        return values

    def test_task_roles_are_frozen(self):
        self.assertEqual(
            attribution.TASK_ROLES,
            (
                ("low_acceptance", "humaneval_152"),
                ("best_current_ratio", "humaneval_047"),
                ("large_checkpoint_gain", "humaneval_131"),
                ("large_gain_low_acceptance", "humaneval_137"),
            ),
        )

    def test_shared_stats_parser_requires_checkpoint_fields(self):
        for field in (
            "prefix_checkpoint_attempts",
            "prefix_checkpoint_successes",
            "prefix_checkpoint_fallbacks",
            "prefix_checkpoint_rows_avoided",
        ):
            self.assertIn(field, common.INT_STATS)

    def test_shared_stats_parser_reads_checkpoint_fields(self):
        values = {}
        for field in common.INT_STATS:
            values[field] = 1
        for field in common.FLOAT_STATS:
            values[field] = 1.0
        for field in common.INT_ARRAY_STATS:
            values[field] = [0] * 6
        for field in common.FLOAT_ARRAY_STATS:
            values[field] = [0.0] * 6
        values.update({
            "emitted": 10,
            "target_evals": 2,
            "target_eval_tokens": 4,
            "multi_attempts": 3,
            "sidecar_ms": 1.25,
            "sidecar_outside_scheduler_ms": 0.25,
            "prefix_checkpoint_attempts": 3,
            "prefix_checkpoint_successes": 2,
            "prefix_checkpoint_fallbacks": 1,
            "prefix_checkpoint_rows_avoided": 5,
            "scheduler_width_rounds": [0, 0, 1, 1, 1, 0],
            "scheduler_width_committed": [0, 0, 2, 3, 5, 0],
            "scheduler_width_sidecar_ms": [0, 0, 0.2, 0.3, 0.5, 0],
            "verify_width_evals": [0, 1, 1, 0, 0, 0],
            "verify_width_positions": [0, 1, 3, 0, 0, 0],
            "verify_width_target_ms": [0, 0.4, 0.6, 0, 0, 0],
        })
        fields = []
        for field in common.STATS_FIELDS:
            value = values[field]
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            fields.append(f"{field}={value}")
        record = " ".join(fields)
        parsed = common.parse_stats(
            common.STATS_PREFIX + record.encode("ascii") + b"\n",
            Path("synthetic.stderr"),
        )
        self.assertEqual(parsed["prefix_checkpoint_attempts"], 3)
        self.assertEqual(parsed["prefix_checkpoint_rows_avoided"], 5)

    def test_task_metrics_expose_avoided_replay_positions(self):
        row = self.blank_stats()
        row.update({
            "emitted": 100,
            "multi_attempts": 30,
            "avg_depth": 3.333,
            "batch_full": 10,
            "batch_partial": 20,
            "prefix_checkpoint_attempts": 20,
            "prefix_checkpoint_successes": 19,
            "prefix_checkpoint_fallbacks": 1,
            "prefix_checkpoint_rows_avoided": 57,
            "target_evals": 30,
            "target_eval_tokens": 120,
            "target_eval_ms": 5000.0,
            "generation_sidecar_ms": 600.0,
            "sidecar_outside_scheduler_ms": 50.0,
            "scheduler_width_rounds": [0, 0, 0, 10, 10, 10],
            "scheduler_width_committed": [0, 0, 0, 20, 30, 50],
            "scheduler_width_sidecar_ms": [0, 0, 0, 150, 200, 250],
            "verify_width_evals": [0, 0, 0, 10, 10, 10],
            "verify_width_positions": [0, 0, 0, 30, 40, 50],
            "verify_width_target_ms": [0, 0, 0, 1000, 1500, 2500],
        })
        context = {
            "role": "synthetic",
            "record": {"source_index": 1},
            "prior": {
                "acceptance_verify_rate": 0.7,
                "paired_ratio": 0.8,
                "paired_ratio_vs_historical": 1.1,
            },
        }
        item = attribution.task_metrics(row, context)
        self.assertEqual(item["checkpoint_fallbacks"], 1)
        self.assertAlmostEqual(item["checkpoint_success_rate"], 0.95)
        self.assertAlmostEqual(item["replay_rows_avoided_per_emitted"], 0.57)
        self.assertAlmostEqual(
            item["legacy_target_positions_per_emitted_proxy"], 1.77
        )
        self.assertAlmostEqual(
            item["scheduler_widths"]["5"]["progress_per_round"], 5.0
        )
        self.assertAlmostEqual(
            item["verifier_widths"]["5"]["target_ms_per_position"], 50.0
        )
        self.assertAlmostEqual(
            item["sidecar_outside_scheduler_ms_per_emitted"], 0.5
        )

    def test_task_metrics_reject_inconsistent_accounting(self):
        row = self.blank_stats()
        row.update({
            "emitted": 1,
            "prefix_checkpoint_attempts": 2,
            "prefix_checkpoint_successes": 2,
            "prefix_checkpoint_fallbacks": 1,
        })
        context = {
            "role": "synthetic",
            "record": {"source_index": 1},
            "prior": {
                "acceptance_verify_rate": 0.7,
                "paired_ratio": 0.8,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "attempt accounting"):
            attribution.task_metrics(row, context)

    def test_report_omits_throughput_and_shows_checkpoint_coverage(self):
        rows = []
        tasks = {}
        for index, (role, task) in enumerate(attribution.TASK_ROLES):
            row = self.blank_stats()
            row.update({
                "prompt": task,
                "emitted": 100,
                "multi_attempts": 30,
                "avg_depth": 3.333,
                "batch_full": 10,
                "batch_partial": 20,
                "prefix_checkpoint_attempts": 20,
                "prefix_checkpoint_successes": 20,
                "prefix_checkpoint_rows_avoided": 60,
                "target_evals": 30,
                "target_eval_tokens": 120,
                "target_eval_ms": 5000.0,
                "generation_sidecar_ms": 600.0,
                "sidecar_outside_scheduler_ms": 50.0,
                "scheduler_width_rounds": [0, 0, 0, 10, 10, 10],
                "scheduler_width_committed": [0, 0, 0, 20, 30, 50],
                "scheduler_width_sidecar_ms": [0, 0, 0, 150, 200, 250],
                "verify_width_evals": [0, 0, 0, 10, 10, 10],
                "verify_width_positions": [0, 0, 0, 30, 40, 50],
                "verify_width_target_ms": [0, 0, 0, 1000, 1500, 2500],
            })
            rows.append(row)
            tasks[task] = {
                "role": role,
                "record": {"source_index": index},
                "prior": {
                    "acceptance_verify_rate": 0.7,
                    "paired_ratio": 0.8,
                },
            }
        summary = attribution.summarize(rows, {"tasks": tasks})
        report = attribution.render_report(summary)
        self.assertIn("20/20 (100.0%)", report)
        self.assertIn("legacy positions/emitted proxy", report)
        self.assertIn("## Scheduler Width Economics", report)
        self.assertIn("## Verifier Width Economics", report)
        self.assertIn("Sidecar outside the multi-commit scheduler", report)
        self.assertIn("Throughput values are intentionally omitted", report)

    def test_command_enables_stats_but_not_checkpoint_override(self):
        args = type("Args", (), {
            "binary": Path("/tmp/ds4"),
            "model": Path("/tmp/base.gguf"),
            "dspark_model": Path("/tmp/dspark.gguf"),
            "ctx": 16384,
            "tokens": 128,
            "nothink": True,
            "fast_verifier": False,
            "exact_head_batch": False,
            "confidence_threshold": "0.455",
        })()
        command = common.command_text(
            args, Path("/tmp/prompt.txt"), "runtime", stats=True
        )
        self.assertIn("DS4_DSPARK_GPU_RUNTIME_STATS=1", command)
        self.assertIn("DS4_DSPARK_CONFIDENCE_THRESHOLD=0.455", command)
        self.assertNotIn("DS4_DSPARK_EXACT_PREFIX_CHECKPOINT", command)


if __name__ == "__main__":
    unittest.main()
