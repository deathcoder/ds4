"""One-process Rust Star measurement adapter for paired benchmarks."""

from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from dwarfstar_measurement_lib import artifact, atomic_json


MEASUREMENT_SCHEMA = "rust-star-engine-measurement-v1"
ENGINE_RUN_SCHEMA = "rust-star-engine-run-v1"


class MeasurementError(RuntimeError):
    """A Rust Star measurement could not satisfy the adapter contract."""


def _peak_child_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def _sanitize(data: bytes, replacements: Sequence[tuple[str, str]]) -> bytes:
    text = data.decode("utf-8", errors="replace")
    for private, public in sorted(
        replacements, key=lambda item: len(item[0]), reverse=True
    ):
        text = text.replace(private, public)
    return text.encode("utf-8")


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MeasurementError(f"{name} must be positive and finite")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise MeasurementError(f"{name} must be nonnegative and finite")
    return number


def parse_engine_run(path: Path, *, context: int, gen_tokens: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read Rust Star engine run: {exc}") from exc
    if payload.get("schema") != ENGINE_RUN_SCHEMA or payload.get("engine") != "rust-star":
        raise MeasurementError("Rust Star engine run has an unexpected identity")
    if payload.get("context") != context or payload.get("gen_tokens") != gen_tokens:
        raise MeasurementError("Rust Star engine run does not describe the requested workload")
    selection = payload.get("selection")
    if not isinstance(selection, dict) or selection.get("oracle_transcript_match") is not True:
        raise MeasurementError("Rust Star engine run did not preserve the oracle transcript")
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        raise MeasurementError("Rust Star engine run is missing timing metadata")
    if payload.get("paired_protocol_eligible") is not True:
        blocker = payload.get("paired_protocol_blocker")
        if not isinstance(blocker, str) or not blocker:
            blocker = "engine run is not paired-protocol eligible"
        raise MeasurementError(blocker)
    timing_expectations = {
        "generation_command_buffers_per_token": 44,
        "generation_host_waits_per_token": 2,
        "generation_correctness_collection": False,
        "prefill_correctness_collection": False,
    }
    for name, expected in timing_expectations.items():
        if timing.get(name) != expected:
            raise MeasurementError(f"Rust Star timing field {name} is inconsistent")
    residency_integer_expectations = {
        "model_view_bytes": timing.get("model_view_bytes"),
        "model_view_warm_touches": timing.get("model_view_warm_touches"),
        "model_view_count": timing.get("model_view_count"),
        "model_residency_allocations": timing.get("model_residency_allocations"),
    }
    for name, value in residency_integer_expectations.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MeasurementError(f"Rust Star timing field {name} is inconsistent")
    if timing.get("model_residency_allocations") != timing.get("model_view_count"):
        raise MeasurementError("Rust Star model residency allocation count is inconsistent")
    if timing.get("model_residency_queue_attached") is not True:
        raise MeasurementError("Rust Star model residency queue attachment is inconsistent")
    _positive_float(timing.get("model_view_warm_wall_ms"), "model_view_warm_wall_ms")
    _positive_float(timing.get("model_view_warm_gpu_ms"), "model_view_warm_gpu_ms")
    prefill_stages: dict[str, tuple[float, float]] = {}
    for stage in ("tile", "transformer", "output_head", "handoff"):
        wall = _positive_float(
            timing.get(f"prefill_{stage}_wall_ms"),
            f"prefill_{stage}_wall_ms",
        )
        gpu = _positive_float(
            timing.get(f"prefill_{stage}_gpu_ms"),
            f"prefill_{stage}_gpu_ms",
        )
        if gpu > wall + 1.0e-6:
            raise MeasurementError(f"Rust Star prefill {stage} GPU time exceeds wall time")
        prefill_stages[stage] = (wall, gpu)
    prefill_host_overhead_ms = _nonnegative_float(
        timing.get("prefill_host_overhead_ms"), "prefill_host_overhead_ms"
    )
    if payload.get("paired_protocol_blocker") is not None:
        raise MeasurementError("eligible Rust Star engine run retains a protocol blocker")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise MeasurementError("Rust Star engine run is missing metrics")
    integer_expectations = {
        "ctx_tokens": context,
        "prefill_tokens": context,
        "gen_tokens": gen_tokens,
        "gen_steady_tokens": gen_tokens - 1,
    }
    for name, expected in integer_expectations.items():
        if metrics.get(name) != expected:
            raise MeasurementError(f"Rust Star metric {name} is inconsistent")
    parsed = dict(integer_expectations)
    parsed["kvcache_bytes"] = None
    for name in (
        "prefill_tps",
        "prefill_ms",
        "gen_tps",
        "gen_ms",
        "gen_first_ms",
        "gen_steady_tps",
        "gen_steady_ms",
    ):
        parsed[name] = _positive_float(metrics.get(name), name)
    accounted_prefill_ms = (
        _positive_float(timing.get("model_view_warm_wall_ms"), "model_view_warm_wall_ms")
        + sum(wall for wall, _ in prefill_stages.values())
        + prefill_host_overhead_ms
    )
    if abs(accounted_prefill_ms - parsed["prefill_ms"]) > 1.0e-3:
        raise MeasurementError("Rust Star prefill timing attribution is inconsistent")
    tolerance = 1.0e-6
    if abs(parsed["prefill_tps"] - context * 1000.0 / parsed["prefill_ms"]) > tolerance:
        raise MeasurementError("Rust Star prefill rate and interval disagree")
    if abs(parsed["gen_tps"] - gen_tokens * 1000.0 / parsed["gen_ms"]) > tolerance:
        raise MeasurementError("Rust Star generation rate and interval disagree")
    if (
        abs(
            parsed["gen_steady_tps"]
            - (gen_tokens - 1) * 1000.0 / parsed["gen_steady_ms"]
        )
        > tolerance
    ):
        raise MeasurementError("Rust Star steady rate and interval disagree")
    if abs(parsed["gen_first_ms"] + parsed["gen_steady_ms"] - parsed["gen_ms"]) > tolerance:
        raise MeasurementError("Rust Star generation intervals do not sum")
    return parsed


def run_ruststar_measurement(
    *,
    executable: Path,
    model: Path,
    context: int,
    gen_tokens: int,
    output_dir: Path,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if context <= 0 or gen_tokens < 2:
        raise MeasurementError("context must be positive and gen_tokens must be at least two")
    executable = executable.expanduser().resolve()
    model = model.expanduser().resolve()
    for path, label in ((executable, "executable"), (model, "model")):
        if not path.is_file():
            raise MeasurementError(f"{label} is not a file: {path}")
    if not os.access(executable, os.X_OK):
        raise MeasurementError(f"executable is not executable: {executable}")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise MeasurementError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise MeasurementError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    engine_path = output_dir / "engine-run.json"
    result_path = output_dir / "measurement.json"

    with tempfile.TemporaryDirectory(prefix="rust-star-candidate-") as temporary:
        private_engine_path = Path(temporary) / "engine-run.json"
        command = [
            str(executable),
            "engine-measure",
            str(model),
            "--context",
            str(context),
            "--gen-tokens",
            str(gen_tokens),
            "--json",
            str(private_engine_path),
        ]
        replacements = (
            (temporary, "$TEMP"),
            (str(executable), "$ENGINE"),
            (str(executable.parent), "$ENGINE_DIR"),
            (str(model), "$MODEL"),
        )
        started_ns = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                command,
                cwd=executable.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(
                command,
                124,
                stdout=exc.stdout or b"",
                stderr=(exc.stderr or b"") + b"\nmeasurement adapter: timed out\n",
            )
            timed_out = True
        wall_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        peak_bytes = _peak_child_bytes()
        stdout_path.write_bytes(_sanitize(completed.stdout, replacements))
        stderr_path.write_bytes(_sanitize(completed.stderr, replacements))

        result: dict[str, Any] = {
            "schema": MEASUREMENT_SCHEMA,
            "engine": "rust-star",
            "status": "failed" if completed.returncode else "passed",
            "context": context,
            "gen_tokens": gen_tokens,
            "command": [
                "$ENGINE",
                "engine-measure",
                "$MODEL",
                "--context",
                str(context),
                "--gen-tokens",
                str(gen_tokens),
                "--json",
                "$TEMP/engine-run.json",
            ],
            "returncode": completed.returncode,
            "timed_out": timed_out,
            "process_wall_ms": wall_ms,
            "process_peak_bytes": peak_bytes,
            "artifacts": {
                "stdout": artifact(stdout_path, output_dir),
                "stderr": artifact(stderr_path, output_dir),
            },
        }
        if completed.returncode == 0:
            if not private_engine_path.is_file():
                result["status"] = "failed"
                result["validation_error"] = "Rust Star succeeded without its engine-run JSON"
            else:
                engine_path.write_bytes(private_engine_path.read_bytes())
                result["artifacts"]["engine_run"] = artifact(engine_path, output_dir)
                try:
                    metrics = parse_engine_run(
                        engine_path, context=context, gen_tokens=gen_tokens
                    )
                    metrics["process_peak_bytes"] = peak_bytes
                    metrics["process_wall_ms"] = wall_ms
                    residual = wall_ms - metrics["prefill_ms"] - metrics["gen_ms"]
                    tolerance_ms = max(5.0, wall_ms * 0.01)
                    if residual < -tolerance_ms:
                        raise MeasurementError(
                            "reported Rust Star intervals exceed process wall time"
                        )
                    metrics["process_overhead_ms"] = max(0.0, residual)
                    result["metrics"] = metrics
                except MeasurementError as exc:
                    result["status"] = "failed"
                    result["validation_error"] = str(exc)
        atomic_json(result_path, result)
        return result
