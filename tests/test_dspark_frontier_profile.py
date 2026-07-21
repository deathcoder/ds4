#!/usr/bin/env python3
"""Model-free tests for exact frontier bookkeeping attribution."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_frontier_profile as profile  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class DSparkFrontierProfileTests(unittest.TestCase):
    def test_policy_and_tasks_are_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(
            profile.TASK_ROLES,
            (
                ("low_acceptance", "humaneval_152"),
                ("high_acceptance", "humaneval_079"),
            ),
        )

    def test_command_enables_profile_and_stats_only(self):
        args = type("Args", (), {
            "binary": Path("/tmp/ds4"),
            "model": Path("/tmp/base.gguf"),
            "dspark_model": Path("/tmp/dspark.gguf"),
            "ctx": 16384,
            "tokens": 128,
            "nothink": True,
            "fast_verifier": False,
            "exact_head_batch": False,
            "confidence_threshold": profile.THRESHOLD,
        })()
        command = common.command_text(
            args,
            Path("/tmp/prompt.txt"),
            "runtime",
            stats=True,
            confidence_threshold=profile.THRESHOLD,
            frontier_profile=True,
        )
        self.assertIn("DS4_DSPARK_GPU_RUNTIME_STATS=1", command)
        self.assertIn("DS4_DSPARK_FRONTIER_PROFILE=1", command)
        self.assertIn("DS4_DSPARK_CONFIDENCE_THRESHOLD=0.75", command)
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", command)
        self.assertNotIn("DS4_DSPARK_ORACLE_TRACE", command)

    def test_frontier_profile_requires_runtime_stats(self):
        with self.assertRaisesRegex(ValueError, "requires runtime stats"):
            common.benchmark_env(
                "runtime", False, stats=False, frontier_profile=True
            )
        with self.assertRaisesRegex(ValueError, "requires runtime stats"):
            common.benchmark_env(
                "baseline", False, stats=True, frontier_profile=True
            )

    def test_parser_reads_and_validates_profile_fields(self):
        values = {
            "frontier_snapshot_calls": 10,
            "frontier_snapshot_successes": 10,
            "frontier_restore_calls": 2,
            "frontier_restore_successes": 2,
            "frontier_prefix_commit_calls": 3,
            "frontier_prefix_commit_successes": 3,
            "target_capture_finish_calls": 10,
            "target_capture_finish_successes": 10,
            "bookkeeping_sync_failures": 0,
            "frontier_snapshot_ms": 12.5,
            "frontier_restore_ms": 1.5,
            "frontier_prefix_commit_ms": 2.0,
            "target_capture_finish_ms": 9.0,
        }
        record = " ".join(f"{key}={value}" for key, value in values.items())
        parsed = profile.parse_profile_stats(
            common.STATS_PREFIX + record.encode("ascii") + b"\n",
            Path("synthetic.stderr"),
        )
        self.assertEqual(parsed, values)

        failed = record.replace(
            "bookkeeping_sync_failures=0", "bookkeeping_sync_failures=1"
        )
        with self.assertRaisesRegex(RuntimeError, "synchronization failed"):
            profile.parse_profile_stats(
                common.STATS_PREFIX + failed.encode("ascii") + b"\n",
                Path("synthetic.stderr"),
            )

    @staticmethod
    def synthetic_row(task, frontier_ms):
        row = {
            "prompt": task,
            "emitted": 100,
            "batch_attempts": 20,
            "batch_partial": 4,
            "prefix_checkpoint_attempts": 4,
            "frontier_snapshot_calls": 20,
            "frontier_snapshot_successes": 20,
            "frontier_restore_calls": 0,
            "frontier_restore_successes": 0,
            "frontier_prefix_commit_calls": 4,
            "frontier_prefix_commit_successes": 4,
            "target_capture_finish_calls": 20,
            "target_capture_finish_successes": 20,
            "bookkeeping_sync_failures": 0,
            "frontier_snapshot_ms": frontier_ms * 0.75,
            "frontier_restore_ms": 0.0,
            "frontier_prefix_commit_ms": frontier_ms * 0.25,
            "target_capture_finish_ms": 50.0,
        }
        return row

    @staticmethod
    def synthetic_reference():
        return {
            "tasks": {
                task: {
                    "role": role,
                    "prior": {
                        "acceptance_verify_rate": 0.5 if "152" in task else 0.84,
                        "paired_ratio": 0.9 if "152" in task else 1.0,
                    },
                }
                for role, task in profile.TASK_ROLES
            }
        }

    def test_summary_proceeds_only_on_frontier_cost(self):
        rows = [
            self.synthetic_row(task, 120.0)
            for task in profile.TASKS
        ]
        summary = profile.summarize(rows, self.synthetic_reference())
        aggregate = summary["aggregate"]
        self.assertAlmostEqual(aggregate["frontier_ms_per_emitted"], 1.2)
        self.assertAlmostEqual(aggregate["capture_finish_ms_per_emitted"], 0.5)
        self.assertEqual(
            aggregate["position_indexed_shadow_state_gate"], "PROCEED"
        )

        rows = [
            self.synthetic_row(task, 80.0)
            for task in profile.TASKS
        ]
        summary = profile.summarize(rows, self.synthetic_reference())
        self.assertEqual(
            summary["aggregate"]["position_indexed_shadow_state_gate"],
            "STOP_ROLLBACK_ONLY",
        )

    def test_report_keeps_capture_separate_and_warns_about_sync(self):
        rows = [
            self.synthetic_row(task, 120.0)
            for task in profile.TASKS
        ]
        report = profile.render_report(
            profile.summarize(rows, self.synthetic_reference())
        )
        self.assertIn("Position-indexed shadow-state gate: **PROCEED**", report)
        self.assertIn("Capture-finalization time is not added", report)
        self.assertIn("explicit synchronization", report)
        self.assertIn("No fresh baseline", report)

    def test_runtime_stats_line_exposes_profile_fields(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        self.assertIn('getenv("DS4_DSPARK_FRONTIER_PROFILE")', source)
        for field in profile.PROFILE_FIELDS:
            self.assertIn(f"{field}=", source)


if __name__ == "__main__":
    unittest.main()
