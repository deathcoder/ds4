#!/usr/bin/env python3
"""Model-free tests for the pinned oMLX comparison tools."""

from pathlib import Path
import json
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import analyze_omlx_humaneval_sweep as analysis  # noqa: E402
import run_omlx_humaneval_mode as runner  # noqa: E402


class OMLXHumanEvalToolTests(unittest.TestCase):
    def test_mode_matrix_is_explicit(self):
        self.assertEqual(
            runner.mode_settings("baseline"),
            {"mtp_enabled": False, "mtp_num_draft_tokens": None},
        )
        self.assertEqual(
            runner.mode_settings("mtp1"),
            {"mtp_enabled": True, "mtp_num_draft_tokens": 1},
        )
        self.assertEqual(runner.mode_settings("mtp3")["mtp_num_draft_tokens"], 3)
        with self.assertRaises(ValueError):
            runner.mode_settings("dspark")

    def test_generation_metrics_report_both_tps_conventions(self):
        metrics = runner.generation_metrics(
            prompt_tokens=100,
            completion_tokens=11,
            wall_start=10.0,
            first_token_at=12.0,
            last_token_at=14.0,
            wall_end=14.5,
        )
        self.assertEqual(metrics["prefill_tps"], 50.0)
        self.assertEqual(metrics["generation_tps"], 5.5)
        self.assertEqual(metrics["interval_generation_tps"], 5.0)
        self.assertEqual(metrics["wall_seconds"], 4.5)

    def test_mtp_log_parser_extracts_acceptance(self):
        parsed = runner.parse_mtp_stats(
            [
                "MTP[abc] finish=length tokens=128 cycles=40 tok/cycle=3.20 "
                "accept=88/120 (73.3%) depth[d1=40/40] emits[init=2,draft=88,bonus=38,verify=0]"
            ]
        )
        self.assertEqual(parsed["tokens"], 128)
        self.assertEqual(parsed["cycles"], 40)
        self.assertAlmostEqual(parsed["accept_rate"], 0.733)
        self.assertIsNone(runner.parse_mtp_stats(["ordinary decode finished"]))

    def test_checkpoint_validation_requires_all_mtp_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v4",
                        "num_nextn_predict_layers": 1,
                    }
                ),
                encoding="utf-8",
            )
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "model-1.safetensors",
                            "mtp.0.block.input_layernorm.weight": "model-2.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (model / "model-1.safetensors").write_bytes(b"target")
            with self.assertRaisesRegex(SystemExit, "missing 1 shards"):
                runner.validate_checkpoint(model)
            (model / "model-2.safetensors").write_bytes(b"mtp")
            result = runner.validate_checkpoint(model)
            self.assertEqual(result["shard_count"], 2)
            self.assertEqual(result["shard_bytes"], 9)
            self.assertEqual(result["mtp_tensor_count"], 1)

    @staticmethod
    def synthetic_runs():
        metadata = {
            "experiment": "omlx_humaneval_mode",
            "omlx_commit": "pinned",
            "model_config_sha256": "config",
            "model_index_sha256": "index",
            "selection": {"indices_zero_based": [0, 1]},
            "protocol": {"tokens": 128},
        }
        speeds = {
            "baseline": (20.0, 20.0),
            "mtp1": (22.0, 21.0),
            "mtp2": (24.0, 23.0),
            "mtp3": (23.0, 22.0),
        }
        runs = {}
        for mode, values in speeds.items():
            tasks = {}
            for index, value in enumerate(values):
                tasks[f"task_{index}"] = {
                    "prompt": f"task_{index}",
                    "generation_tps": value,
                    "interval_generation_tps": value * 0.99,
                    "completion_tokens": 128,
                    "output_sha256": f"hash-{index}",
                    "mtp_stats": None,
                }
            runs[mode] = {
                "metadata": {**metadata, "mode": mode},
                "tasks": tasks,
            }
        return runs

    def test_sweep_selects_winner_and_checks_output_identity(self):
        runs = self.synthetic_runs()
        tasks = analysis.validate_runs(runs)
        summary = analysis.summarize(runs, tasks)
        self.assertEqual(summary["winner"], "mtp2")
        self.assertTrue(summary["correctness_gate"])
        self.assertEqual(summary["modes"]["mtp2"]["faster_tasks"], 2)
        self.assertIn("Exact-output gate: PASS", analysis.render_report(summary))

    def test_sweep_rejects_output_drift(self):
        runs = self.synthetic_runs()
        runs["mtp3"]["tasks"]["task_1"]["output_sha256"] = "drift"
        summary = analysis.summarize(runs, ["task_0", "task_1"])
        self.assertFalse(summary["correctness_gate"])
        self.assertEqual(
            summary["output_mismatches"],
            [{"task": "task_1", "mode": "mtp3"}],
        )

    def test_sweep_rejects_completion_token_drift(self):
        runs = self.synthetic_runs()
        runs["mtp2"]["tasks"]["task_0"]["completion_tokens"] = 127
        summary = analysis.summarize(runs, ["task_0", "task_1"])
        self.assertFalse(summary["correctness_gate"])
        self.assertEqual(
            summary["token_count_mismatches"],
            [{"task": "task_0", "mode": "mtp2"}],
        )


if __name__ == "__main__":
    unittest.main()
