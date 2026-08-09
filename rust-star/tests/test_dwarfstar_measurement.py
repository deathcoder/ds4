from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwarfstar_measurement_lib import (  # noqa: E402
    MeasurementError,
    parse_dwarfstar_csv,
    run_dwarfstar_measurement,
)


CSV_HEADER = (
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,"
    "gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n"
)


class DwarfStarMeasurementTests(unittest.TestCase):
    def temporary_path(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, Path(temporary.name)

    def test_csv_is_normalized_without_inventing_kv_bytes(self) -> None:
        _, root = self.temporary_path()
        csv_path = root / "result.csv"
        csv_path.write_text(
            CSV_HEADER + "2048,2048,4096.0,128,16.0,50.0,127,16.1,0\n",
            encoding="utf-8",
        )
        metrics = parse_dwarfstar_csv(csv_path, context=2048, gen_tokens=128)
        self.assertIsNone(metrics["kvcache_bytes"])
        self.assertEqual(metrics["prefill_ms"], 500.0)
        self.assertEqual(metrics["gen_ms"], 8000.0)

    def test_csv_rejects_partial_generation(self) -> None:
        _, root = self.temporary_path()
        csv_path = root / "result.csv"
        csv_path.write_text(
            CSV_HEADER + "2048,2048,4096.0,127,16.0,50.0,126,16.1,0\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(MeasurementError, "generated 127 tokens"):
            parse_dwarfstar_csv(csv_path, context=2048, gen_tokens=128)

    def test_isolated_process_measurement_and_log_redaction(self) -> None:
        _, root = self.temporary_path()
        executable = root / "fake-ds4-bench"
        executable.write_text(
            """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
csv_path = pathlib.Path(args[args.index("--csv") + 1])
context = int(args[args.index("--ctx-start") + 1])
gen_tokens = int(args[args.index("--gen-tokens") + 1])
print("model=" + args[args.index("-m") + 1])
csv_path.write_text(
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,"
    "gen_steady_tokens,gen_steady_tps,kvcache_bytes\\n"
    f"{context},{context},1000000000,{gen_tokens},1000000000,0.001,"
    f"{gen_tokens - 1},1000000000,0\\n",
    encoding="utf-8",
)
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        model = root / "private-model.gguf"
        prompt = root / "private-prompt.txt"
        model.write_bytes(b"model")
        prompt.write_text("prompt", encoding="utf-8")
        output = root / "measurement"

        result = run_dwarfstar_measurement(
            executable=executable,
            model=model,
            prompt=prompt,
            context=2048,
            gen_tokens=128,
            output_dir=output,
        )

        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["metrics"]["process_peak_bytes"], 0)
        self.assertGreaterEqual(result["metrics"]["process_overhead_ms"], 0)
        self.assertEqual(result["metrics"]["kvcache_bytes"], None)
        self.assertIn("model=$MODEL", (output / "stdout.log").read_text(encoding="utf-8"))
        self.assertNotIn(str(root), (output / "stdout.log").read_text(encoding="utf-8"))
        persisted = json.loads((output / "measurement.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["schema"], "rust-star-engine-measurement-v1")

    def test_invalid_success_csv_is_persisted_as_failure(self) -> None:
        _, root = self.temporary_path()
        executable = root / "fake-ds4-bench"
        executable.write_text(
            """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
pathlib.Path(args[args.index("--csv") + 1]).write_text("wrong\\nvalue\\n", encoding="utf-8")
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        model = root / "model.gguf"
        prompt = root / "prompt.txt"
        model.write_bytes(b"model")
        prompt.write_text("prompt", encoding="utf-8")
        output = root / "measurement"

        result = run_dwarfstar_measurement(
            executable=executable,
            model=model,
            prompt=prompt,
            context=2048,
            gen_tokens=128,
            output_dir=output,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("missing fields", result["validation_error"])
        persisted = json.loads((output / "measurement.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "failed")


if __name__ == "__main__":
    unittest.main()
