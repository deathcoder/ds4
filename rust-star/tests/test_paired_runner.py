from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paired_runner_lib import (  # noqa: E402
    PairedRunnerError,
    initialize_or_load_state,
    load_paired_plan,
    retry_pair,
    run_remaining,
)


FAKE_ADAPTER = """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--context", type=int, required=True)
parser.add_argument("--gen-tokens", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--role", required=True)
parser.add_argument("--control", type=Path, required=True)
args = parser.parse_args()

control = json.loads(args.control.read_text(encoding="utf-8"))
calls = control.setdefault("calls", {}).get(args.role, 0) + 1
control["calls"][args.role] = calls
args.control.write_text(json.dumps(control), encoding="utf-8")
args.output.mkdir(parents=True)
if calls in control.get("fail_on_calls", {}).get(args.role, []):
    payload = {
        "schema": "rust-star-engine-measurement-v1",
        "engine": args.role,
        "status": "failed",
        "context": args.context,
        "gen_tokens": args.gen_tokens,
        "validation_error": "synthetic external event",
    }
    (args.output / "measurement.json").write_text(json.dumps(payload), encoding="utf-8")
    raise SystemExit(1)

speed = 10.0 if args.role == "oracle" else 15.0
steady_tokens = args.gen_tokens - 1
metrics = {
    "ctx_tokens": args.context,
    "prefill_tokens": args.context,
    "gen_tokens": args.gen_tokens,
    "gen_steady_tokens": steady_tokens,
    "kvcache_bytes": None,
    "process_peak_bytes": 1024,
    "prefill_tps": 1000.0,
    "prefill_ms": args.context,
    "gen_tps": speed,
    "gen_ms": args.gen_tokens / speed * 1000.0,
    "gen_first_ms": 1000.0 / speed,
    "gen_steady_tps": speed,
    "gen_steady_ms": steady_tokens / speed * 1000.0,
    "process_wall_ms": 20000.0,
    "process_overhead_ms": 100.0,
}
payload = {
    "schema": "rust-star-engine-measurement-v1",
    "engine": args.role,
    "status": "passed",
    "context": args.context,
    "gen_tokens": args.gen_tokens,
    "metrics": metrics,
}
(args.output / "measurement.json").write_text(json.dumps(payload), encoding="utf-8")
"""


class PairedRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "fake_adapter.py"
        self.adapter.write_text(FAKE_ADAPTER, encoding="utf-8")
        self.adapter.chmod(0o755)
        self.control = self.root / "control.json"
        self.control.write_text(
            json.dumps({"calls": {}, "fail_on_calls": {}}), encoding="utf-8"
        )
        self.plan_path = self.root / "plan.json"
        self.write_plan()

    def identity(self, prefix: str) -> dict:
        return {
            "source_commit": prefix * 40,
            "source_tree": ("1" if prefix == "a" else "2") * 40,
            "executable_sha256": ("3" if prefix == "a" else "4") * 64,
            "model_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "backend": "metal",
            "build_configuration": {"profile": "synthetic"},
            "runtime_configuration": {"mode": "test"},
        }

    def adapter_config(self, role: str, identity: dict) -> dict:
        return {
            "identity": identity,
            "working_directory": str(self.root),
            "adapter_command": [
                sys.executable,
                str(self.adapter),
                "--context",
                "{context}",
                "--gen-tokens",
                "{gen_tokens}",
                "--output",
                "{output}",
                "--role",
                role,
                "--control",
                str(self.control),
            ],
        }

    def write_plan(self) -> None:
        plan = {
            "schema": "rust-star-paired-plan-v1",
            "protocol": "rust-star-paired-benchmark-v1",
            "correctness_class": "C0",
            "host_manifest_sha256": "d" * 64,
            "correctness_manifest_sha256": "e" * 64,
            "configuration": {
                "contexts": [2048, 32768],
                "repetitions": 2,
                "gen_tokens": 128,
                "sampling": "greedy argmax excluding EOS",
                "primary_metric": "gen_steady_tps",
            },
            "oracle": self.adapter_config("oracle", self.identity("a")),
            "candidate": self.adapter_config("candidate", self.identity("d")),
        }
        self.plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def load(self, output_name: str = "run"):
        plan = load_paired_plan(self.plan_path)
        output, state = initialize_or_load_state(plan, self.root / output_name)
        return plan, output, state

    def test_checkpoint_resume_and_final_export(self) -> None:
        plan, output, state = self.load()
        self.assertTrue(run_remaining(plan, output, state, max_new_pairs=1))
        self.assertEqual(state["phase"], "paused")
        self.assertEqual(len(state["attempts"]), 1)
        for label in ("oracle", "candidate"):
            warmup = state["warmups"][label][0]
            measurement = json.loads(
                (output / warmup["output"] / "measurement.json").read_text(encoding="utf-8")
            )
            self.assertEqual(measurement["gen_tokens"], 128)

        plan = load_paired_plan(self.plan_path)
        output, state = initialize_or_load_state(plan, output)
        self.assertTrue(run_remaining(plan, output, state))
        self.assertEqual(state["phase"], "complete")
        raw = json.loads((output / "paired-raw.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [(pair["context"], pair["repetition"]) for pair in raw["pairs"]],
            [(2048, 1), (32768, 1), (32768, 2), (2048, 2)],
        )
        summary = json.loads((output / "paired-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["source_path"], "paired-raw.json")
        self.assertEqual(
            summary["contexts"][0]["candidate_over_oracle"]["gen_steady_tps"]["median"],
            1.5,
        )

    def test_failed_pair_requires_explicit_reason_before_retry(self) -> None:
        self.control.write_text(
            json.dumps(
                {"calls": {}, "fail_on_calls": {"candidate": [2]}},
            ),
            encoding="utf-8",
        )
        plan, output, state = self.load("retry-run")
        self.assertFalse(run_remaining(plan, output, state))
        self.assertEqual(state["phase"], "blocked_pair")
        self.assertEqual(state["attempts"][0]["status"], "failed")
        self.assertFalse((output / "paired-raw.json").exists())

        self.assertTrue(
            retry_pair(
                plan,
                output,
                state,
                context=2048,
                repetition=1,
                reason="documented unrelated background process",
            )
        )
        self.assertEqual(state["attempts"][0]["status"], "invalid")
        self.assertEqual(state["attempts"][1]["attempt"], 2)
        self.assertTrue(run_remaining(plan, output, state))
        raw = json.loads((output / "paired-raw.json").read_text(encoding="utf-8"))
        self.assertFalse(raw["pairs"][0]["valid"])
        self.assertEqual(raw["pairs"][0]["attempt"], 1)
        self.assertTrue(raw["pairs"][1]["valid"])
        self.assertEqual(raw["pairs"][1]["attempt"], 2)

    def test_retry_after_completion_preserves_superseded_outputs(self) -> None:
        plan, output, state = self.load("completed-retry")
        self.assertTrue(run_remaining(plan, output, state))
        self.assertTrue(
            retry_pair(
                plan,
                output,
                state,
                context=2048,
                repetition=1,
                reason="thermal event noticed during review",
            )
        )
        self.assertFalse((output / "paired-raw.json").exists())
        self.assertTrue((output / "superseded/revision-01/paired-raw.json").is_file())
        self.assertTrue(run_remaining(plan, output, state))
        self.assertEqual(state["phase"], "complete")

    def test_plan_is_immutable_after_checkpoint_creation(self) -> None:
        plan, output, _ = self.load("immutable")
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload["configuration"]["gen_tokens"] = 64
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")
        changed = load_paired_plan(self.plan_path)
        with self.assertRaisesRegex(PairedRunnerError, "plan changed"):
            initialize_or_load_state(changed, output)
        self.assertNotEqual(plan.sha256, changed.sha256)

    def test_finalization_rejects_changed_measurement_evidence(self) -> None:
        plan, output, state = self.load("tampered-evidence")
        self.assertTrue(run_remaining(plan, output, state, max_new_pairs=1))
        record = state["attempts"][0]["engines"]["oracle"]
        measurement = output / record["output"] / "measurement.json"
        payload = json.loads(measurement.read_text(encoding="utf-8"))
        payload["metrics"]["gen_steady_tps"] = 999.0
        measurement.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(PairedRunnerError, "missing or changed"):
            run_remaining(plan, output, state)
        self.assertEqual(state["phase"], "blocked_evidence")
        self.assertFalse((output / "paired-raw.json").exists())


if __name__ == "__main__":
    unittest.main()
