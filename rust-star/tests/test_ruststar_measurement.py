#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


RUST_STAR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUST_STAR_DIR))

from ruststar_measurement_lib import (  # noqa: E402
    MeasurementError,
    parse_engine_run,
    run_ruststar_measurement,
)


def engine_run(*, eligible: bool = True) -> dict[str, object]:
    prefill_ms = 20_000.0
    gen_first_ms = 60.0
    gen_steady_ms = 6_940.0
    gen_ms = gen_first_ms + gen_steady_ms
    return {
        "schema": "rust-star-engine-run-v1",
        "engine": "rust-star",
        "context": 2048,
        "gen_tokens": 128,
        "metrics": {
            "ctx_tokens": 2048,
            "prefill_tokens": 2048,
            "gen_tokens": 128,
            "gen_steady_tokens": 127,
            "prefill_tps": 2048 * 1000.0 / prefill_ms,
            "prefill_ms": prefill_ms,
            "gen_tps": 128 * 1000.0 / gen_ms,
            "gen_ms": gen_ms,
            "gen_first_ms": gen_first_ms,
            "gen_steady_tps": 127 * 1000.0 / gen_steady_ms,
            "gen_steady_ms": gen_steady_ms,
        },
        "selection": {"oracle_transcript_match": True},
        "timing": {
            "model_view_bytes": 86_370_050_944,
            "model_view_warm_touches": 83_334,
            "model_view_count": 1_136,
            "model_residency_allocations": 1_136,
            "model_residency_queue_attached": True,
            "model_view_warm_wall_ms": 8.0,
            "model_view_warm_gpu_ms": 7.0,
            "prefill_context_setup_ms": 100.0,
            "prefill_bootstrap_setup_ms": 150.0,
            "prefill_tile_wall_ms": 3_000.0,
            "prefill_tile_gpu_ms": 2_900.0,
            "prefill_transformer_setup_ms": 650.0,
            "prefill_transformer_wall_ms": 15_000.0,
            "prefill_transformer_gpu_ms": 14_900.0,
            "prefill_output_head_wall_ms": 20.0,
            "prefill_output_head_gpu_ms": 2.0,
            "prefill_output_head_setup_ms": 15.0,
            "prefill_handoff_wall_ms": 5.0,
            "prefill_handoff_gpu_ms": 1.0,
            "prefill_host_overhead_ms": 1_967.0,
            "generation_command_buffers_per_token": 44,
            "generation_host_waits_per_token": 2,
            "generation_correctness_collection": False,
            "prefill_correctness_collection": not eligible,
        },
        "paired_protocol_eligible": eligible,
        "paired_protocol_blocker": None if eligible else "prefill collection remains enabled",
    }


class RustStarMeasurementTests(unittest.TestCase):
    def temporary_path(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, Path(temporary.name)

    def test_eligible_engine_run_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(engine_run()), encoding="utf-8")
            metrics = parse_engine_run(path, context=2048, gen_tokens=128)
        self.assertEqual(metrics["ctx_tokens"], 2048)
        self.assertEqual(metrics["gen_steady_tokens"], 127)
        self.assertIsNone(metrics["kvcache_bytes"])
        self.assertAlmostEqual(metrics["gen_steady_tps"], 127_000 / 6940)

    def test_ineligible_engine_run_is_rejected_with_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(engine_run(eligible=False)), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "prefill collection"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_inconsistent_rate_is_rejected(self) -> None:
        payload = engine_run()
        payload["metrics"]["gen_steady_tps"] = 999.0  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "steady rate"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_cannot_hide_prefill_collection(self) -> None:
        payload = engine_run()
        payload["timing"]["prefill_correctness_collection"] = True  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "prefill_correctness_collection"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_requires_consistent_model_residency(self) -> None:
        payload = engine_run()
        payload["timing"]["model_residency_allocations"] = 1_135  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "allocation count"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_requires_model_residency_queue_attachment(self) -> None:
        payload = engine_run()
        payload["timing"]["model_residency_queue_attached"] = False  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "queue attachment"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_requires_complete_prefill_attribution(self) -> None:
        payload = engine_run()
        del payload["timing"]["prefill_transformer_gpu_ms"]  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "prefill_transformer_gpu_ms"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_rejects_inconsistent_prefill_attribution(self) -> None:
        payload = engine_run()
        payload["timing"]["prefill_host_overhead_ms"] = 1_000.0  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "attribution is inconsistent"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_rejects_setup_above_host_overhead(self) -> None:
        payload = engine_run()
        payload["timing"]["prefill_transformer_setup_ms"] = 2_000.0  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "setup attribution"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_eligibility_rejects_prefill_gpu_time_above_wall(self) -> None:
        payload = engine_run()
        payload["timing"]["prefill_tile_gpu_ms"] = 3_001.0  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "engine-run.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MeasurementError, "GPU time exceeds wall time"):
                parse_engine_run(path, context=2048, gen_tokens=128)

    def test_isolated_process_measurement_and_log_redaction(self) -> None:
        _, root = self.temporary_path()
        payload = engine_run()
        payload["metrics"].update(  # type: ignore[union-attr]
            {
                "prefill_tps": 2_048_000_000.0,
                "prefill_ms": 0.001,
                "gen_tps": 128_000_000.0,
                "gen_ms": 0.001,
                "gen_first_ms": 0.0005,
                "gen_steady_tps": 254_000_000.0,
                "gen_steady_ms": 0.0005,
            }
        )
        payload["timing"].update(  # type: ignore[union-attr]
            {
                "model_view_warm_wall_ms": 0.0001,
                "model_view_warm_gpu_ms": 0.00009,
                "prefill_context_setup_ms": 0.00004,
                "prefill_bootstrap_setup_ms": 0.00004,
                "prefill_tile_wall_ms": 0.0002,
                "prefill_tile_gpu_ms": 0.00018,
                "prefill_transformer_setup_ms": 0.00004,
                "prefill_transformer_wall_ms": 0.0003,
                "prefill_transformer_gpu_ms": 0.00028,
                "prefill_output_head_wall_ms": 0.0001,
                "prefill_output_head_gpu_ms": 0.00009,
                "prefill_output_head_setup_ms": 0.00003,
                "prefill_handoff_wall_ms": 0.0001,
                "prefill_handoff_gpu_ms": 0.00009,
                "prefill_host_overhead_ms": 0.0002,
            }
        )
        (root / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        executable = root / "fake-rust-star"
        executable.write_text(
            """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
output = pathlib.Path(args[args.index("--json") + 1])
print("model=" + args[1])
output.write_bytes((pathlib.Path(__file__).parent / "payload.json").read_bytes())
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        model = root / "private-model.gguf"
        model.write_bytes(b"model")
        output = root / "measurement"

        result = run_ruststar_measurement(
            executable=executable,
            model=model,
            context=2048,
            gen_tokens=128,
            output_dir=output,
        )

        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["metrics"]["process_peak_bytes"], 0)
        self.assertGreaterEqual(result["metrics"]["process_overhead_ms"], 0)
        log = (output / "stdout.log").read_text(encoding="utf-8")
        self.assertIn("model=$MODEL", log)
        self.assertNotIn(str(root), log)
        persisted = json.loads((output / "measurement.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["schema"], "rust-star-engine-measurement-v1")
        self.assertIn("engine_run", persisted["artifacts"])


if __name__ == "__main__":
    unittest.main()
