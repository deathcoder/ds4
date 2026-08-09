"""One-process DwarfStar measurement adapter for paired Rust Star benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


MEASUREMENT_SCHEMA = "rust-star-engine-measurement-v1"
INTEGER_FIELDS = (
    "ctx_tokens",
    "prefill_tokens",
    "gen_tokens",
    "gen_steady_tokens",
    "kvcache_bytes",
)
FLOAT_FIELDS = ("prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps")


class MeasurementError(RuntimeError):
    """A DwarfStar measurement could not satisfy the adapter contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _positive_float(value: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise MeasurementError(f"{name} is not a number: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise MeasurementError(f"{name} must be positive and finite")
    return number


def _nonnegative_int(value: str, name: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise MeasurementError(f"{name} is not an integer: {value!r}") from exc
    if number < 0:
        raise MeasurementError(f"{name} must be nonnegative")
    return number


def parse_dwarfstar_csv(path: Path, *, context: int, gen_tokens: int) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise MeasurementError(f"cannot read DwarfStar CSV {path}: {exc}") from exc
    if len(rows) != 1:
        raise MeasurementError(f"expected one DwarfStar CSV row, found {len(rows)}")
    row = rows[0]
    missing = [field for field in (*INTEGER_FIELDS, *FLOAT_FIELDS) if field not in row]
    if missing:
        raise MeasurementError(f"DwarfStar CSV is missing fields: {', '.join(missing)}")

    parsed: dict[str, Any] = {
        field: _nonnegative_int(row[field], field) for field in INTEGER_FIELDS
    }
    parsed.update({field: _positive_float(row[field], field) for field in FLOAT_FIELDS})
    if parsed["ctx_tokens"] != context or parsed["prefill_tokens"] != context:
        raise MeasurementError("DwarfStar CSV does not describe the requested single frontier")
    if parsed["gen_tokens"] != gen_tokens:
        raise MeasurementError(
            f"DwarfStar generated {parsed['gen_tokens']} tokens, expected {gen_tokens}"
        )
    if parsed["gen_steady_tokens"] != gen_tokens - 1:
        raise MeasurementError("DwarfStar steady token count must be generation length minus one")

    parsed["prefill_ms"] = parsed["prefill_tokens"] / parsed["prefill_tps"] * 1000.0
    parsed["gen_ms"] = parsed["gen_tokens"] / parsed["gen_tps"] * 1000.0
    parsed["gen_steady_ms"] = (
        parsed["gen_steady_tokens"] / parsed["gen_steady_tps"] * 1000.0
    )
    # At a terminal single frontier ds4-bench does not create a snapshot. Its
    # zero is therefore unavailable snapshot size, not zero live KV memory.
    if parsed["kvcache_bytes"] == 0:
        parsed["kvcache_bytes"] = None
    return parsed


def _peak_child_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # macOS reports bytes; Linux reports KiB. The production target is macOS,
    # while the Linux branch keeps synthetic CI tests meaningful.
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def _sanitize(data: bytes, replacements: Sequence[tuple[str, str]]) -> bytes:
    text = data.decode("utf-8", errors="replace")
    for private, public in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(private, public)
    return text.encode("utf-8")


def run_dwarfstar_measurement(
    *,
    executable: Path,
    model: Path,
    prompt: Path,
    context: int,
    gen_tokens: int,
    output_dir: Path,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if context <= 0 or gen_tokens < 2:
        raise MeasurementError("context must be positive and gen_tokens must be at least two")
    executable = executable.expanduser().resolve()
    model = model.expanduser().resolve()
    prompt = prompt.expanduser().resolve()
    for path, label in ((executable, "executable"), (model, "model"), (prompt, "prompt")):
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
    csv_artifact_path = output_dir / "benchmark.csv"
    result_path = output_dir / "measurement.json"

    with tempfile.TemporaryDirectory(prefix="rust-star-dwarfstar-") as temporary:
        temporary_root = Path(temporary)
        csv_path = temporary_root / "benchmark.csv"
        command = [
            str(executable),
            "--metal",
            "-m",
            str(model),
            "--prompt-file",
            str(prompt),
            "--ctx-start",
            str(context),
            "--ctx-max",
            str(context),
            "--gen-tokens",
            str(gen_tokens),
            "--warm-weights",
            "--csv",
            str(csv_path),
        ]
        replacements = (
            (str(temporary_root), "$TEMP"),
            (str(executable), "$ENGINE"),
            (str(model), "$MODEL"),
            (str(prompt), "$PROMPT"),
            (str(executable.parent), "$ENGINE_DIR"),
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

        public_command = [
            "$ENGINE",
            "--metal",
            "-m",
            "$MODEL",
            "--prompt-file",
            "$PROMPT",
            "--ctx-start",
            str(context),
            "--ctx-max",
            str(context),
            "--gen-tokens",
            str(gen_tokens),
            "--warm-weights",
            "--csv",
            "$TEMP/benchmark.csv",
        ]
        result: dict[str, Any] = {
            "schema": MEASUREMENT_SCHEMA,
            "engine": "dwarfstar",
            "status": "failed" if completed.returncode else "passed",
            "context": context,
            "gen_tokens": gen_tokens,
            "command": public_command,
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
            if not csv_path.is_file():
                result["status"] = "failed"
                result["validation_error"] = "DwarfStar succeeded without writing its CSV"
            else:
                shutil.copyfile(csv_path, csv_artifact_path)
                result["artifacts"]["csv"] = artifact(csv_artifact_path, output_dir)
                try:
                    metrics = parse_dwarfstar_csv(
                        csv_artifact_path, context=context, gen_tokens=gen_tokens
                    )
                    metrics["process_peak_bytes"] = peak_bytes
                    metrics["process_wall_ms"] = wall_ms
                    residual = wall_ms - metrics["prefill_ms"] - metrics["gen_ms"]
                    tolerance_ms = max(5.0, wall_ms * 0.01)
                    if residual < -tolerance_ms:
                        raise MeasurementError(
                            "reported engine intervals exceed externally measured process wall time"
                        )
                    metrics["process_overhead_ms"] = max(0.0, residual)
                    result["metrics"] = metrics
                except MeasurementError as exc:
                    result["status"] = "failed"
                    result["validation_error"] = str(exc)
        atomic_json(result_path, result)
        return result
