#!/usr/bin/env python3
"""Model-free tests for the DSpark round break-even oracle."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_humaneval_oracle_audit as oracle  # noqa: E402
import run_dspark_issue468_comparison as common  # noqa: E402


class DSparkOracleAuditTests(unittest.TestCase):
    def test_runtime_source_has_default_off_round_trace(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        self.assertIn('getenv("DS4_DSPARK_ORACLE_TRACE")', source)
        self.assertIn("ds4: DSpark oracle trace round=", source)
        self.assertIn(
            "enabled = dspark_session_runtime_stats_enabled() &&",
            source,
        )
        self.assertIn(
            "dspark_session_oracle_round_finish(d, verified, n_tokens);",
            source,
        )

    def test_command_requires_stats_and_enables_oracle_trace(self):
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
            args, Path("/tmp/prompt.txt"), "runtime",
            stats=True, oracle_trace=True,
        )
        self.assertIn("DS4_DSPARK_GPU_RUNTIME_STATS=1", command)
        self.assertIn("DS4_DSPARK_ORACLE_TRACE=1", command)
        with self.assertRaisesRegex(ValueError, "requires runtime stats"):
            common.command_text(
                args, Path("/tmp/prompt.txt"), "runtime",
                oracle_trace=True,
            )

    def test_parse_oracle_trace(self):
        data = (
            common.ORACLE_TRACE_PREFIX +
            b"round=1 proposed=5 selected=3 verified=3 accepted=2 "
            b"committed=2 sidecar_ms=10.25 target_ms=80.5 "
            b"target_evals=1 target_positions=3 "
            b"confidences=0.9,0.8,0.7,0.6,0.5\n"
        )
        rows = common.parse_oracle_trace(data, Path("synthetic.stderr"))
        self.assertEqual(rows[0]["selected"], 3)
        self.assertEqual(rows[0]["accepted"], 2)
        self.assertEqual(rows[0]["committed"], 2)
        self.assertEqual(len(rows[0]["confidences"]), 5)

    def test_parse_oracle_trace_rejects_invalid_widths(self):
        data = (
            common.ORACLE_TRACE_PREFIX +
            b"round=1 proposed=5 selected=3 verified=4 accepted=2 "
            b"committed=2 sidecar_ms=10 target_ms=80 "
            b"target_evals=1 target_positions=4 "
            b"confidences=0.9,0.8,0.7,0.6,0.5\n"
        )
        with self.assertRaisesRegex(RuntimeError, "invalid DSpark oracle widths"):
            common.parse_oracle_trace(data, Path("synthetic.stderr"))

    def test_oracle_separates_profitable_and_unprofitable_rounds(self):
        row = {
            "emitted": 3,
            "target_evals": 2,
            "target_eval_tokens": 3,
            "target_eval_ms": 130.0,
            "scheduler_width_sidecar_ms": [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
            "generation_sidecar_ms": 20.0,
            "prefill_sidecar_ms": 0.0,
            "sidecar_outside_scheduler_ms": 0.0,
            "oracle_trace": [
                {
                    "round": 1,
                    "proposed": 5,
                    "selected": 2,
                    "verified": 2,
                    "accepted": 2,
                    "committed": 2,
                    "sidecar_ms": 0.0,
                    "target_ms": 120.0,
                    "target_evals": 1,
                    "target_positions": 2,
                    "confidences": [0.9] * 5,
                },
                {
                    "round": 2,
                    "proposed": 5,
                    "selected": 1,
                    "verified": 1,
                    "accepted": 0,
                    "committed": 1,
                    "sidecar_ms": 20.0,
                    "target_ms": 10.0,
                    "target_evals": 1,
                    "target_positions": 1,
                    "confidences": [0.8] * 5,
                },
            ],
        }
        context = {
            "record": {"source_index": 7},
            "prior": {
                "acceptance_verify_rate": 0.7,
                "paired_ratio": 0.8,
            },
            "baseline_tps": 20.0,
        }
        result = oracle.analyze_task(row, context)
        self.assertAlmostEqual(result["accounted_ratio"], 1.0)
        self.assertAlmostEqual(result["route_oracle_ratio"], 150.0 / 130.0)
        self.assertEqual(result["profitable_rounds"], 1)
        self.assertAlmostEqual(result["profitable_token_share"], 1.0 / 3.0)
        self.assertAlmostEqual(
            result["zero_sidecar_oracle_ratio"], 150.0 / 110.0
        )
        self.assertAlmostEqual(result["target_scale_for_parity"], 1.0)

    def test_report_states_that_it_is_not_throughput(self):
        task = {
            "acceptance_verify_rate": 0.7,
            "prior_paired_ratio": 0.8,
            "accounted_ratio": 0.9,
            "route_oracle_ratio": 1.05,
            "profitable_round_share": 0.25,
            "profitable_token_share": 0.4,
        }
        summary = {
            "tasks": {"humaneval_000": task},
            "aggregate": {
                "accounted_ratio": 0.9,
                "route_oracle_ratio": 1.05,
                "zero_sidecar_oracle_ratio": 1.2,
                "zero_target_oracle_ratio": 2.0,
                "profitable_rounds": 1,
                "rounds": 4,
                "profitable_round_share": 0.25,
                "profitable_tokens": 4,
                "emitted": 10,
                "profitable_token_share": 0.4,
                "target_scale_for_parity": 0.8,
                "target_scale_for_10pct": 0.6,
            },
        }
        report = oracle.render_report(summary)
        self.assertIn("no fresh throughput benchmark was run", report)
        self.assertIn("Perfect round router", report)
        self.assertIn("local counterfactual", report)


if __name__ == "__main__":
    unittest.main()
