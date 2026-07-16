#!/usr/bin/env python3
"""Model-free tests for dense-mixed FlashAttention stage attribution."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_dense_mixed_flash_profile as profile  # noqa: E402


class DenseMixedFlashProfileTests(unittest.TestCase):
    def test_profile_environment_is_gathered_only(self):
        reference = profile.profile_env()
        env = profile.profile_env(profile.LAYER)
        self.assertNotIn("DS4_METAL_FLASH_ATTN_GATHERED_PROFILE", reference)
        self.assertEqual(env["DS4_METAL_FLASH_ATTN_GATHERED_PROFILE"], "1")
        self.assertNotIn("DS4_METAL_FLASH_ATTN_STAGE_PROFILE", env)
        self.assertEqual(env["DS4_DSPARK_EXACT_ATTENTION_PROFILE"], "1")

    @staticmethod
    def flash_log():
        prefix = (
            b"ds4: Metal FlashAttention prefill stage "
            b"mode=gathered_decode tokens=1 comp=100 keys=228 "
            b"heads=8 dim=512 window=128 ratio=4 "
        )
        return b"\n".join([
            prefix + b"linearize_raw=1.000 ms",
            prefix + b"copy_raw=2.000 ms",
            prefix + b"copy_comp=3.000 ms",
            prefix + b"mask_fill=1.000 ms",
            prefix + b"pad=1.000 ms",
            prefix + b"attention_vec=5.000 ms",
            prefix + b"attention_reduce=2.000 ms",
        ])

    def test_parse_groups_one_complete_call(self):
        rows = profile.parse_flash(self.flash_log(), Path("test.stderr"))
        self.assertEqual({row["call"] for row in rows}, {1})
        self.assertEqual(rows[-1]["stage"], "attention_reduce")

    def test_parse_rejects_incomplete_call(self):
        data = self.flash_log().replace(
            b"\n" + (
                b"ds4: Metal FlashAttention prefill stage "
                b"mode=gathered_decode tokens=1 comp=100 keys=228 "
                b"heads=8 dim=512 window=128 ratio=4 "
                b"attention_reduce=2.000 ms"
            ),
            b"",
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            profile.parse_flash(data, Path("test.stderr"))

    def test_summary_attributes_stages(self):
        exact = [
            {
                "part": "attention",
                "layer": 42,
                "pos": 1,
                "tokens": 1,
                "stage": "dense_mixed",
                "ms": 1.0,
            }
        ]
        rows = profile.parse_flash(self.flash_log(), Path("test.stderr"))
        summary = profile.summarize(exact, rows)
        self.assertEqual(summary["dense_rows"], 1)
        self.assertAlmostEqual(summary["call_total"]["mean_ms"], 15.0)
        self.assertAlmostEqual(
            summary["stages"]["attention_vec"]["mean_contribution_share"],
            1.0 / 3.0,
        )

    def test_report_is_diagnostic_only(self):
        exact = [
            {
                "part": "attention",
                "layer": 42,
                "pos": 1,
                "tokens": 1,
                "stage": "dense_mixed",
                "ms": 1.0,
            }
        ]
        rows = profile.parse_flash(self.flash_log(), Path("test.stderr"))
        report = profile.render_report(profile.summarize(exact, rows))
        self.assertIn("Synchronized diagnostic only", report)
        self.assertIn("No timed throughput benchmark", report)


if __name__ == "__main__":
    unittest.main()
