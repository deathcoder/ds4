#!/usr/bin/env python3
"""Model-free tests for the width-stratified attention-route profile."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_exact_attention_tail_profile as tail_components  # noqa: E402
import run_dspark_threshold075_width_attention_profile as profile  # noqa: E402


class DSparkThreshold075WidthAttentionProfileTests(unittest.TestCase):
    def test_contract_is_frozen(self):
        self.assertEqual(profile.THRESHOLD, "0.75")
        self.assertEqual(profile.TASK, "humaneval_079")
        self.assertEqual(profile.LAYER, 42)
        self.assertEqual(profile.WIDTHS, (2, 3, 4, 5))
        self.assertEqual(
            profile.TAIL_SOURCE_COMMIT,
            "a31f69b91545d82d2d881fc05128904ce37424c4",
        )
        self.assertEqual(
            profile.ATTENTION_SOURCE_COMMIT,
            "c981252dafeb1253f14cc2761d6d56a53c6d5375",
        )

    def test_profile_environment_is_attention_only(self):
        env = profile.profile_env()
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_STATS"], "1")
        self.assertEqual(env["DS4_DSPARK_CONFIDENCE_THRESHOLD"], "0.75")
        self.assertEqual(env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"], "42")
        self.assertEqual(env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"], "42")
        self.assertEqual(env["DS4_DSPARK_EXACT_ATTENTION_PROFILE"], "1")
        self.assertNotIn("DS4_DSPARK_EXACT_TAIL_PROFILE", env)
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)

    def test_fused_gather_environment_adds_only_profile_switch(self):
        env = profile.profile_env(True)
        self.assertEqual(env["DS4_METAL_FLASH_ATTN_GATHERED_PROFILE"], "1")
        self.assertNotIn("DS4_METAL_DENSE_MIXED_GATHERED_LEGACY", env)
        self.assertNotIn("DS4_METAL_DENSE_MIXED_DIRECT", env)

    @staticmethod
    def records():
        signature = ((100, 2), (102, 3), (105, 4), (109, 5))
        rows = []
        for stage in tail_components.CONTROL_STAGES:
            for start, width in signature:
                rows.append({
                    "variant": "default_rb16_direct",
                    "part": "exact",
                    "layer": 42,
                    "pos": start,
                    "tokens": width,
                    "stage": stage,
                    "ms": float(width),
                })
        modes = (
            ("raw", "dense_mixed"),
            ("dense_mixed", "dense_mixed", "sparse_indexed"),
            ("dense_mixed",) * 4,
            ("dense_mixed",) * 4 + ("sparse_indexed",),
        )
        costs = {"raw": 1.0, "dense_mixed": 2.0, "sparse_indexed": 4.0}
        for (start, width), batch_modes in zip(signature, modes):
            for offset, mode in enumerate(batch_modes):
                rows.append({
                    "variant": "default_rb16_direct",
                    "part": "attention",
                    "layer": 42,
                    "pos": start + offset,
                    "tokens": 1,
                    "stage": mode,
                    "ms": costs[mode],
                })
        return rows

    @staticmethod
    def stats():
        return {"verify_width_evals": [0, 0, 1, 1, 1, 1]}

    def test_assignment_preserves_batch_width_and_route(self):
        signature, assigned = profile.assign_attention_batches(self.records())
        self.assertEqual(signature[-1], (109, 5))
        last = [row for row in assigned if row["batch"] == 4]
        self.assertEqual([row["pos"] for row in last], [109, 110, 111, 112, 113])
        self.assertEqual(last[-1]["mode"], "sparse_indexed")

    def test_summary_reports_width5_cost_owner(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        width5 = summary["width_results"]["5"]
        self.assertEqual(width5["modes"]["dense_mixed"]["rows"], 4)
        self.assertEqual(width5["modes"]["sparse_indexed"]["rows"], 1)
        self.assertEqual(summary["dominant_width5_cost_mode"], "dense_mixed")
        self.assertEqual(summary["slowest_width5_mode"], "sparse_indexed")

    def test_assignment_rejects_row_order_mismatch(self):
        rows = self.records()
        for row in rows:
            if row["part"] == "attention" and row["pos"] == 103:
                row["pos"] = 999
                break
        with self.assertRaisesRegex(RuntimeError, "positions"):
            profile.assign_attention_batches(rows)

    def test_report_keeps_attribution_boundary(self):
        summary, _ = profile.summarize(self.records(), self.stats())
        report = profile.render_report(summary)
        self.assertIn("Width-Stratified Attention Route Profile", report)
        self.assertIn("dense mixed", report)
        self.assertIn("Synchronized absolute timings", report)
        self.assertIn("No runtime candidate", report)

    def test_fused_gather_parser_and_width_summary(self):
        lines = []
        calls = sum(width for _, width in ((100, 2), (102, 3), (105, 4), (109, 5)))
        for call in range(calls):
            for stage, elapsed in (
                ("prepare", 1.0),
                ("attention_vec", 2.0),
                ("attention_reduce", 3.0),
            ):
                lines.append(
                    "ds4: Metal FlashAttention prefill stage "
                    "mode=fused_gather_decode tokens=1 comp=900 keys=1028 "
                    "heads=32 dim=512 window=0 ratio=0 "
                    f"{stage}={elapsed:.3f} ms"
                )
        flash = profile.parse_fused_gather_profile(
            "\n".join(lines).encode("ascii"), Path("synthetic.stderr")
        )
        records = self.records()
        for row in records:
            if row["part"] == "attention":
                row["stage"] = "dense_mixed"
        summary, assigned = profile.summarize_fused_gather(
            records, flash, self.stats()
        )
        self.assertEqual(len(assigned), calls * 3)
        self.assertEqual(summary["width_results"]["5"]["rows"], 5)
        self.assertEqual(summary["dominant_width5_stage"], "attention_reduce")
        self.assertAlmostEqual(
            summary["width_results"]["5"]["stages"]["attention_vec"][
                "cost_share"
            ],
            1.0 / 3.0,
        )

    def test_fused_gather_parser_rejects_stage_order(self):
        data = (
            "ds4: Metal FlashAttention prefill stage "
            "mode=fused_gather_decode tokens=1 comp=1 keys=2 heads=32 dim=512 "
            "window=0 ratio=0 attention_vec=1.000 ms\n"
            "ds4: Metal FlashAttention prefill stage "
            "mode=fused_gather_decode tokens=1 comp=1 keys=2 heads=32 dim=512 "
            "window=0 ratio=0 prepare=1.000 ms\n"
            "ds4: Metal FlashAttention prefill stage "
            "mode=fused_gather_decode tokens=1 comp=1 keys=2 heads=32 dim=512 "
            "window=0 ratio=0 attention_reduce=1.000 ms\n"
        ).encode("ascii")
        with self.assertRaisesRegex(RuntimeError, "sequence mismatch"):
            profile.parse_fused_gather_profile(data, Path("synthetic.stderr"))

    def test_source_has_diagnostic_only_fused_gather_boundaries(self):
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        self.assertIn('"fused_gather_decode", (name)', host)
        self.assertIn(
            'getenv("DS4_METAL_FLASH_ATTN_GATHERED_PROFILE")', host
        )
        self.assertIn(
            'DS4_METAL_PROFILE_FUSED_GATHER_STAGE("prepare")', host
        )
        self.assertIn(
            'DS4_METAL_PROFILE_FUSED_GATHER_STAGE("attention_vec")', host
        )
        self.assertIn(
            'DS4_METAL_PROFILE_FUSED_GATHER_STAGE("attention_reduce")', host
        )


if __name__ == "__main__":
    unittest.main()
