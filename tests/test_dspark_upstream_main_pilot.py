#!/usr/bin/env python3
"""Model-free tests for the upstream DSpark comparison pilot."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_upstream_main_pilot as pilot  # noqa: E402


def args():
    return SimpleNamespace(
        current_binary=Path("/current/ds4"),
        upstream_binary=Path("/upstream/ds4"),
        upstream_source=Path("/upstream"),
        model=Path("/models/target.gguf"),
        current_dspark_model=Path("/models/current-dspark.gguf"),
        upstream_dspark_model=Path("/models/upstream-support.gguf"),
        ctx=16384,
        tokens=128,
        upstream_confidence=None,
        disable_upstream_scheduler=False,
    )


class UpstreamPilotTests(unittest.TestCase):
    def test_latin_rotation_balances_every_mode_position(self):
        positions = {
            mode: [0 for _ in pilot.MODES] for mode in pilot.MODES
        }
        for task in range(1, 9):
            for position, mode in enumerate(pilot.mode_order(task)):
                positions[mode][position] += 1
        for counts in positions.values():
            self.assertEqual(counts, [2, 2, 2, 2])

    def test_upstream_uses_integrated_cli(self):
        cmd = pilot.command(args(), Path("/prompt.txt"), "upstream_dspark")
        self.assertIn("--mtp", cmd)
        self.assertIn("/models/upstream-support.gguf", cmd)
        self.assertIn("--dspark", cmd)
        self.assertNotIn("--dspark-confidence", cmd)

    def test_current_uses_research_sidecar_cli_and_promoted_env(self):
        config = args()
        cmd = pilot.command(config, Path("/prompt.txt"), "current_dspark")
        self.assertEqual(
            cmd[-2:], ["--dspark", "/models/current-dspark.gguf"]
        )
        env = pilot.mode_env(config, "current_dspark")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertEqual(
            env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], pilot.CURRENT_THRESHOLD
        )

    def test_controlled_upstream_policy_is_explicit(self):
        config = args()
        config.upstream_confidence = 0.75
        config.disable_upstream_scheduler = True
        cmd = pilot.command(
            config, Path("/prompt.txt"), "upstream_dspark"
        )
        self.assertEqual(cmd[-2:], ["--dspark-confidence", "0.75"])
        env = pilot.mode_env(config, "upstream_dspark")
        self.assertEqual(env["DS4_DSPARK_SCHEDULER"], "0")

    def test_upstream_inspect_uses_official_support_artifact(self):
        cmd = pilot.upstream_inspect_command(args())
        self.assertEqual(cmd[-3:], [
            "--mtp",
            "/models/upstream-support.gguf",
            "--inspect",
        ])

    def test_upstream_binding_requires_complete_compatible_layout(self):
        parsed = pilot.parse_upstream_support_binding(
            b"support binding: tensors=81 missing=0 invalid=0 "
            b"metadata_errors=0\n"
        )
        self.assertEqual(parsed["tensors"], 81)
        with self.assertRaisesRegex(ValueError, "not runtime-compatible"):
            pilot.parse_upstream_support_binding(
                b"support binding: tensors=81 missing=0 invalid=3 "
                b"metadata_errors=0\n"
            )

    def test_activation_requires_real_draft_proposals(self):
        line = (
            b"ds4: DSpark stats cycles=12 first_tokens=12 proposed=44 "
            b"accepted_draft=30 verifier_unavailable=0 errors=0\n"
        )
        parsed = pilot.parse_upstream_activation_stats(line)
        self.assertEqual(parsed["proposed"], 44)
        with self.assertRaisesRegex(ValueError, "no draft proposals"):
            pilot.parse_upstream_activation_stats(
                line.replace(b"proposed=44", b"proposed=0")
            )

    def test_summary_normalizes_each_runtime_to_its_own_plain_mode(self):
        records = [
            {"label": "task_a", "source_index": 0},
            {"label": "task_b", "source_index": 1},
        ]
        rows = []
        values = {
            "task_a": (20.0, 22.0, 25.0, 24.0),
            "task_b": (10.0, 12.0, 20.0, 22.0),
        }
        for task, task_values in values.items():
            for mode, value in zip(pilot.MODES, task_values):
                rows.append({
                    "prompt": task,
                    "mode": mode,
                    "generation_tps": value,
                    "group_order": "-".join(pilot.MODES),
                })
        summary = pilot.summarize(rows, records)
        self.assertAlmostEqual(
            summary["samples"]["task_a"]["upstream_dspark_vs_plain"], 1.1
        )
        self.assertAlmostEqual(
            summary["samples"]["task_a"]["current_dspark_vs_plain"], 0.96
        )
        self.assertAlmostEqual(
            summary["samples"]["task_b"][
                "current_dspark_vs_upstream_dspark"
            ],
            22.0 / 12.0,
        )

    def test_report_marks_policy_comparison_limit(self):
        records = [{"label": "task", "source_index": 0}]
        rows = []
        for mode, value in zip(pilot.MODES, (20.0, 21.0, 22.0, 23.0)):
            rows.append({
                "prompt": "task",
                "mode": mode,
                "generation_tps": value,
                "group_order": "-".join(pilot.MODES),
            })
        summary = pilot.summarize(rows, records)
        report = pilot.render_report(
            summary, "upstream", "current", None, False
        )
        self.assertIn("documented confidence default 0.9", report)
        self.assertIn("intended policy, not isolated verifier", report)


if __name__ == "__main__":
    unittest.main()
