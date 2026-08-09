"""Checkpointed paired benchmark orchestration for Rust Star."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from paired_benchmark_lib import (
    PROTOCOL,
    RAW_SCHEMA,
    EngineIdentity,
    PairedBenchmarkError,
    load_paired_run,
    parse_engine_identity,
    summarize_paired_run,
    validate_metric_values,
)


PLAN_SCHEMA = "rust-star-paired-plan-v1"
STATE_SCHEMA = "rust-star-paired-state-v1"
MEASUREMENT_SCHEMA = "rust-star-engine-measurement-v1"
SUMMARY_NAME = "paired-summary.json"
RAW_NAME = "paired-raw.json"
PLACEHOLDERS = ("{context}", "{gen_tokens}", "{output}")


class PairedRunnerError(RuntimeError):
    """The plan, checkpoint, or adapter result violates the runner contract."""


def _reject_json_constant(value: str) -> None:
    raise PairedRunnerError(f"non-standard JSON numeric constant is forbidden: {value}")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_reject_json_constant)
    except PairedRunnerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedRunnerError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PairedRunnerError(f"{name} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairedRunnerError(f"{name} must be a positive integer")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PairedRunnerError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _required_string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PairedRunnerError(f"{name} must be 64 lowercase hexadecimal characters")
    return text


@dataclass(frozen=True)
class AdapterSpec:
    identity: EngineIdentity
    command: tuple[str, ...]
    working_directory: Path
    timeout_seconds: float | None


@dataclass(frozen=True)
class PairedPlan:
    path: Path
    sha256: str
    correctness_class: str
    host_manifest_sha256: str
    correctness_manifest_sha256: str
    contexts: tuple[int, ...]
    repetitions: int
    gen_tokens: int
    sampling: str
    primary_metric: str
    oracle: AdapterSpec
    candidate: AdapterSpec


def _parse_adapter(value: Any, name: str, plan_dir: Path) -> AdapterSpec:
    if not isinstance(value, dict):
        raise PairedRunnerError(f"{name} must be a JSON object")
    try:
        identity = parse_engine_identity(value.get("identity"), f"{name}.identity")
    except PairedBenchmarkError as exc:
        raise PairedRunnerError(str(exc)) from exc
    raw_command = value.get("adapter_command")
    if not isinstance(raw_command, list) or not raw_command:
        raise PairedRunnerError(f"{name}.adapter_command must be a non-empty string array")
    if any(not isinstance(argument, str) or not argument for argument in raw_command):
        raise PairedRunnerError(f"{name}.adapter_command must contain only non-empty strings")
    command = tuple(raw_command)
    for placeholder in PLACEHOLDERS:
        if command.count(placeholder) != 1:
            raise PairedRunnerError(
                f"{name}.adapter_command must contain {placeholder!r} exactly once as a full argument"
            )
    for argument in command:
        if "{" in argument or "}" in argument:
            if argument not in PLACEHOLDERS:
                raise PairedRunnerError(
                    f"{name}.adapter_command has unsupported embedded placeholder: {argument!r}"
                )
    raw_working_directory = value.get("working_directory", ".")
    working_text = _required_string(raw_working_directory, f"{name}.working_directory")
    working_directory = Path(working_text).expanduser()
    if not working_directory.is_absolute():
        working_directory = plan_dir / working_directory
    working_directory = working_directory.resolve()
    if not working_directory.is_dir():
        raise PairedRunnerError(f"{name}.working_directory is not a directory: {working_directory}")
    timeout = value.get("timeout_seconds")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise PairedRunnerError(f"{name}.timeout_seconds must be a positive number")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise PairedRunnerError(f"{name}.timeout_seconds must be a positive finite number")
    return AdapterSpec(identity, command, working_directory, timeout)


def load_paired_plan(path: Path) -> PairedPlan:
    path = path.expanduser().resolve()
    payload = _load_json(path, "paired plan")
    if payload.get("schema") != PLAN_SCHEMA:
        raise PairedRunnerError(f"plan schema must be {PLAN_SCHEMA!r}")
    if payload.get("protocol") != PROTOCOL:
        raise PairedRunnerError(f"plan protocol must be {PROTOCOL!r}")
    correctness_class = _required_string(payload.get("correctness_class"), "correctness_class")
    if correctness_class not in {"C0", "C1", "C2", "C3"}:
        raise PairedRunnerError(f"unknown correctness_class {correctness_class!r}")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise PairedRunnerError("configuration must be a JSON object")
    raw_contexts = configuration.get("contexts")
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise PairedRunnerError("configuration.contexts must be a non-empty array")
    contexts = tuple(_positive_int(value, "configuration.contexts[]") for value in raw_contexts)
    if tuple(sorted(set(contexts))) != contexts:
        raise PairedRunnerError("configuration.contexts must be sorted and unique")
    repetitions = _positive_int(configuration.get("repetitions"), "configuration.repetitions")
    gen_tokens = _positive_int(configuration.get("gen_tokens"), "configuration.gen_tokens")
    if gen_tokens < 2:
        raise PairedRunnerError("configuration.gen_tokens must be at least two")
    sampling = _required_string(configuration.get("sampling"), "configuration.sampling")
    primary_metric = _required_string(
        configuration.get("primary_metric"), "configuration.primary_metric"
    )
    if primary_metric != "gen_steady_tps":
        raise PairedRunnerError("configuration.primary_metric must be 'gen_steady_tps'")
    oracle = _parse_adapter(payload.get("oracle"), "oracle", path.parent)
    candidate = _parse_adapter(payload.get("candidate"), "candidate", path.parent)
    if oracle.identity.model_sha256 != candidate.identity.model_sha256:
        raise PairedRunnerError("oracle and candidate model SHA-256 differ")
    if oracle.identity.prompt_sha256 != candidate.identity.prompt_sha256:
        raise PairedRunnerError("oracle and candidate prompt SHA-256 differ")
    if oracle.identity.backend != candidate.identity.backend:
        raise PairedRunnerError("oracle and candidate backends differ")
    return PairedPlan(
        path=path,
        sha256=_sha256_file(path),
        correctness_class=correctness_class,
        host_manifest_sha256=_sha256(
            payload.get("host_manifest_sha256"), "host_manifest_sha256"
        ),
        correctness_manifest_sha256=_sha256(
            payload.get("correctness_manifest_sha256"), "correctness_manifest_sha256"
        ),
        contexts=contexts,
        repetitions=repetitions,
        gen_tokens=gen_tokens,
        sampling=sampling,
        primary_metric=primary_metric,
        oracle=oracle,
        candidate=candidate,
    )


def _state_path(output_dir: Path) -> Path:
    return output_dir / "state.json"


def initialize_or_load_state(plan: PairedPlan, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    output_dir = output_dir.expanduser().resolve()
    state_path = _state_path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise PairedRunnerError(f"output path is not a directory: {output_dir}")
    if state_path.exists():
        state = _load_json(state_path, "runner state")
        if state.get("schema") != STATE_SCHEMA:
            raise PairedRunnerError(f"state schema must be {STATE_SCHEMA!r}")
        if state.get("plan_sha256") != plan.sha256:
            raise PairedRunnerError("plan changed after this checkpoint was created")
        _validate_state(plan, state)
        return output_dir, state
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PairedRunnerError(f"new runner output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": STATE_SCHEMA,
        "protocol": PROTOCOL,
        "plan_sha256": plan.sha256,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "phase": "initialized",
        "warmups": {"oracle": [], "candidate": []},
        "attempts": [],
        "superseded_outputs": [],
    }
    _atomic_json(state_path, state)
    return output_dir, state


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(_state_path(output_dir), state)


def _validate_record_shape(record: Any, name: str, *, require_passed: bool) -> None:
    if not isinstance(record, dict):
        raise PairedRunnerError(f"{name} must be a JSON object")
    status = record.get("status")
    if status not in {"running", "passed", "failed"}:
        raise PairedRunnerError(f"{name} has invalid status {status!r}")
    relative = Path(_required_string(record.get("output"), f"{name}.output"))
    if relative.is_absolute() or ".." in relative.parts:
        raise PairedRunnerError(f"{name}.output must be a safe relative path")
    if require_passed:
        if status != "passed":
            raise PairedRunnerError(f"{name} must be passed")
        _sha256(record.get("measurement_sha256"), f"{name}.measurement_sha256")


def _validate_state(plan: PairedPlan, state: dict[str, Any]) -> None:
    if state.get("protocol") != PROTOCOL:
        raise PairedRunnerError("state protocol differs from plan")
    if state.get("phase") not in {
        "initialized",
        "warmup",
        "blocked_warmup",
        "timed",
        "blocked_pair",
        "blocked_evidence",
        "paused",
        "retrying",
        "complete",
    }:
        raise PairedRunnerError(f"state has unknown phase {state.get('phase')!r}")
    warmups = state.get("warmups")
    if not isinstance(warmups, dict) or set(warmups) != {"oracle", "candidate"}:
        raise PairedRunnerError("state warmups must contain oracle and candidate")
    for label in ("oracle", "candidate"):
        if not isinstance(warmups[label], list):
            raise PairedRunnerError(f"state warmups.{label} must be an array")
        for index, record in enumerate(warmups[label]):
            _validate_record_shape(record, f"warmups.{label}[{index}]", require_passed=False)
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise PairedRunnerError("state attempts must be an array")
    if not isinstance(state.get("superseded_outputs"), list):
        raise PairedRunnerError("state superseded_outputs must be an array")

    seen_counts: dict[tuple[int, int], int] = {}
    latest_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    initial_order: list[tuple[int, int]] = []
    for index, attempt in enumerate(attempts):
        name = f"attempts[{index}]"
        if not isinstance(attempt, dict):
            raise PairedRunnerError(f"{name} must be a JSON object")
        context = _positive_int(attempt.get("context"), f"{name}.context")
        repetition = _positive_int(attempt.get("repetition"), f"{name}.repetition")
        number = _positive_int(attempt.get("attempt"), f"{name}.attempt")
        if context not in plan.contexts or repetition > plan.repetitions:
            raise PairedRunnerError(f"{name} is outside the predeclared plan")
        key = (context, repetition)
        expected_number = seen_counts.get(key, 0) + 1
        if number != expected_number:
            raise PairedRunnerError(f"{name} attempt number must be {expected_number}")
        previous = latest_by_pair.get(key)
        if previous is not None and previous.get("status") != "invalid":
            raise PairedRunnerError(f"{name} follows an attempt that was not invalidated")
        seen_counts[key] = number
        latest_by_pair[key] = attempt
        if number == 1:
            initial_order.append(key)
        expected_order = "AB" if repetition % 2 else "BA"
        if attempt.get("order") != expected_order:
            raise PairedRunnerError(f"{name}.order must be {expected_order}")
        status = attempt.get("status")
        if status not in {"running", "passed", "failed", "invalid"}:
            raise PairedRunnerError(f"{name} has invalid status {status!r}")
        engines = attempt.get("engines")
        if not isinstance(engines, dict) or any(
            label not in {"oracle", "candidate"} for label in engines
        ):
            raise PairedRunnerError(f"{name}.engines is invalid")
        for label, record in engines.items():
            _validate_record_shape(
                record,
                f"{name}.engines.{label}",
                require_passed=status == "passed",
            )
        if status == "passed" and set(engines) != {"oracle", "candidate"}:
            raise PairedRunnerError(f"{name} passed without both engine measurements")
        if status == "invalid":
            _required_string(attempt.get("invalid_reason"), f"{name}.invalid_reason")

    expected_initial_order = _schedule(plan)
    if initial_order != expected_initial_order[: len(initial_order)]:
        raise PairedRunnerError("state initial attempts do not follow the predeclared schedule")


def _render_command(spec: AdapterSpec, *, context: int, gen_tokens: int, output: Path) -> list[str]:
    values = {
        "{context}": str(context),
        "{gen_tokens}": str(gen_tokens),
        "{output}": str(output),
    }
    return [values.get(argument, argument) for argument in spec.command]


def _measurement_error(payload: dict[str, Any]) -> str:
    if payload.get("timed_out") is True:
        return "engine measurement timed out"
    if payload.get("validation_error"):
        return "engine measurement failed adapter validation"
    return "engine measurement reported failure"


def _run_adapter(
    spec: AdapterSpec,
    *,
    label: str,
    context: int,
    gen_tokens: int,
    output_dir: Path,
    relative_output: Path,
) -> dict[str, Any]:
    measurement_dir = output_dir / relative_output
    if measurement_dir.exists():
        raise PairedRunnerError(f"adapter output already exists: {measurement_dir}")
    command = _render_command(
        spec, context=context, gen_tokens=gen_tokens, output=measurement_dir
    )
    process: subprocess.Popen[bytes] | None = None
    start_failed = False
    try:
        process = subprocess.Popen(
            command,
            cwd=spec.working_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=spec.timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 124
            else:
                returncode = 124
    except OSError:
        returncode = 127
        timed_out = False
        start_failed = True
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    measurement_dir.mkdir(parents=True, exist_ok=True)
    process_record = {
        "schema": "rust-star-adapter-process-v1",
        "engine": label,
        "returncode": returncode,
        "timed_out": timed_out,
        "start_failed": start_failed,
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(measurement_dir / "adapter-process.json", process_record)
    record: dict[str, Any] = {
        "status": "failed",
        "output": str(relative_output),
        "adapter_returncode": returncode,
        "timed_out": timed_out,
    }
    measurement_path = measurement_dir / "measurement.json"
    if returncode != 0:
        record["error"] = (
            "adapter process could not start"
            if start_failed
            else "adapter process failed; inspect its local evidence directory"
        )
        return record
    if not measurement_path.is_file():
        record["error"] = "adapter succeeded without measurement.json"
        return record
    try:
        payload = _load_json(measurement_path, "engine measurement")
        if payload.get("schema") != MEASUREMENT_SCHEMA:
            raise PairedRunnerError(f"measurement schema must be {MEASUREMENT_SCHEMA!r}")
        if payload.get("status") != "passed":
            record["error"] = _measurement_error(payload)
            return record
        if payload.get("context") != context or payload.get("gen_tokens") != gen_tokens:
            raise PairedRunnerError("measurement context or generation length differs from plan")
        try:
            row = validate_metric_values(
                payload.get("metrics"),
                f"{label}.metrics",
                context=context,
                gen_tokens=gen_tokens,
            )
        except PairedBenchmarkError as exc:
            raise PairedRunnerError(str(exc)) from exc
    except PairedRunnerError as exc:
        record["error"] = f"invalid measurement artifact: {exc}"
        return record
    record["status"] = "passed"
    record["metrics"] = row.values
    record["measurement_sha256"] = _sha256_file(measurement_path)
    record.pop("error", None)
    return record


def _adapter(plan: PairedPlan, label: str) -> AdapterSpec:
    return plan.oracle if label == "oracle" else plan.candidate


def ensure_warmups(plan: PairedPlan, output_dir: Path, state: dict[str, Any]) -> bool:
    state["phase"] = "warmup"
    save_state(output_dir, state)
    context = min(plan.contexts)
    for label in ("oracle", "candidate"):
        history = state["warmups"][label]
        if any(item["status"] == "passed" for item in history):
            continue
        if history and history[-1]["status"] == "failed":
            state["phase"] = "blocked_warmup"
            save_state(output_dir, state)
            return False
        number = len(history) + 1
        relative = Path("warmups") / f"{label}-{number:02d}"
        record = _run_adapter(
            _adapter(plan, label),
            label=label,
            context=context,
            gen_tokens=min(8, plan.gen_tokens),
            output_dir=output_dir,
            relative_output=relative,
        )
        record["attempt"] = number
        history.append(record)
        save_state(output_dir, state)
        if record["status"] != "passed":
            state["phase"] = "blocked_warmup"
            save_state(output_dir, state)
            return False
    return True


def _schedule(plan: PairedPlan) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    for repetition in range(1, plan.repetitions + 1):
        contexts: Sequence[int] = (
            plan.contexts if repetition % 2 else tuple(reversed(plan.contexts))
        )
        schedule.extend((context, repetition) for context in contexts)
    return schedule


def _attempts_for(
    state: dict[str, Any], context: int, repetition: int
) -> list[dict[str, Any]]:
    return [
        attempt
        for attempt in state["attempts"]
        if attempt["context"] == context and attempt["repetition"] == repetition
    ]


def _execute_pair(
    plan: PairedPlan,
    output_dir: Path,
    state: dict[str, Any],
    *,
    context: int,
    repetition: int,
) -> bool:
    prior = _attempts_for(state, context, repetition)
    attempt_number = len(prior) + 1
    order = "AB" if repetition % 2 else "BA"
    attempt: dict[str, Any] = {
        "context": context,
        "repetition": repetition,
        "attempt": attempt_number,
        "order": order,
        "status": "running",
        "started_at_utc": _utc_now(),
        "engines": {},
    }
    state["attempts"].append(attempt)
    state["phase"] = "timed"
    save_state(output_dir, state)
    labels = ("oracle", "candidate") if order == "AB" else ("candidate", "oracle")
    for label in labels:
        relative = (
            Path("measurements")
            / f"ctx-{context:07d}"
            / f"rep-{repetition:02d}"
            / f"attempt-{attempt_number:02d}"
            / label
        )
        attempt["engines"][label] = {
            "status": "running",
            "output": str(relative),
        }
        save_state(output_dir, state)
        record = _run_adapter(
            _adapter(plan, label),
            label=label,
            context=context,
            gen_tokens=plan.gen_tokens,
            output_dir=output_dir,
            relative_output=relative,
        )
        attempt["engines"][label] = record
        save_state(output_dir, state)
        if record["status"] != "passed":
            attempt["status"] = "failed"
            attempt["completed_at_utc"] = _utc_now()
            state["phase"] = "blocked_pair"
            save_state(output_dir, state)
            return False
    attempt["status"] = "passed"
    attempt["completed_at_utc"] = _utc_now()
    save_state(output_dir, state)
    return True


def _identity_payload(identity: EngineIdentity) -> dict[str, Any]:
    return {
        "source_commit": identity.source_commit,
        "source_tree": identity.source_tree,
        "executable_sha256": identity.executable_sha256,
        "model_sha256": identity.model_sha256,
        "prompt_sha256": identity.prompt_sha256,
        "backend": identity.backend,
        "build_configuration": identity.build_configuration,
        "runtime_configuration": identity.runtime_configuration,
    }


def _safe_evidence_path(output_dir: Path, relative_value: Any) -> Path:
    relative_text = _required_string(relative_value, "measurement evidence path")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise PairedRunnerError("measurement evidence path must stay inside runner output")
    path = (output_dir / relative).resolve()
    try:
        path.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise PairedRunnerError("measurement evidence path escapes runner output") from exc
    return path


def _revalidate_passed_record(
    output_dir: Path,
    record: dict[str, Any],
    *,
    label: str,
    context: int,
    gen_tokens: int,
) -> dict[str, int | float | None]:
    measurement_dir = _safe_evidence_path(output_dir, record.get("output"))
    measurement_path = measurement_dir / "measurement.json"
    expected_sha = _sha256(record.get("measurement_sha256"), "measurement_sha256")
    if not measurement_path.is_file() or _sha256_file(measurement_path) != expected_sha:
        raise PairedRunnerError(f"{label} measurement evidence is missing or changed")
    payload = _load_json(measurement_path, "engine measurement")
    if payload.get("schema") != MEASUREMENT_SCHEMA or payload.get("status") != "passed":
        raise PairedRunnerError(f"{label} measurement evidence is no longer a passing artifact")
    if payload.get("context") != context or payload.get("gen_tokens") != gen_tokens:
        raise PairedRunnerError(f"{label} measurement evidence identity changed")
    try:
        row = validate_metric_values(
            payload.get("metrics"),
            f"{label}.metrics",
            context=context,
            gen_tokens=gen_tokens,
        )
    except PairedBenchmarkError as exc:
        raise PairedRunnerError(str(exc)) from exc
    return row.values


def _pair_payload(attempt: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "context": attempt["context"],
        "repetition": attempt["repetition"],
        "attempt": attempt["attempt"],
        "order": attempt["order"],
        "valid": attempt["status"] == "passed",
        "evidence": {
            label: {
                "output": record.get("output"),
                "measurement_sha256": record.get("measurement_sha256"),
            }
            for label, record in attempt["engines"].items()
        },
    }
    if payload["valid"]:
        payload["oracle"] = attempt["engines"]["oracle"]["metrics"]
        payload["candidate"] = attempt["engines"]["candidate"]["metrics"]
    else:
        payload["invalid_reason"] = attempt["invalid_reason"]
    return payload


def finalize(plan: PairedPlan, output_dir: Path, state: dict[str, Any]) -> None:
    for context, repetition in _schedule(plan):
        attempts = _attempts_for(state, context, repetition)
        if not attempts or attempts[-1]["status"] != "passed":
            raise PairedRunnerError(
                f"cannot finalize: context {context} repetition {repetition} lacks a final valid pair"
            )
    try:
        for attempt in state["attempts"]:
            if attempt["status"] != "passed":
                continue
            for label in ("oracle", "candidate"):
                record = attempt["engines"][label]
                record["metrics"] = _revalidate_passed_record(
                    output_dir,
                    record,
                    label=label,
                    context=attempt["context"],
                    gen_tokens=plan.gen_tokens,
                )
    except PairedRunnerError:
        state["phase"] = "blocked_evidence"
        save_state(output_dir, state)
        raise
    raw = {
        "schema": RAW_SCHEMA,
        "protocol": PROTOCOL,
        "correctness_class": plan.correctness_class,
        "host_manifest_sha256": plan.host_manifest_sha256,
        "correctness_manifest_sha256": plan.correctness_manifest_sha256,
        "configuration": {
            "contexts": list(plan.contexts),
            "repetitions": plan.repetitions,
            "gen_tokens": plan.gen_tokens,
            "sampling": plan.sampling,
            "primary_metric": plan.primary_metric,
        },
        "oracle": _identity_payload(plan.oracle.identity),
        "candidate": _identity_payload(plan.candidate.identity),
        "pairs": [_pair_payload(attempt) for attempt in state["attempts"]],
    }
    raw_path = output_dir / RAW_NAME
    _atomic_json(raw_path, raw)
    try:
        summary = summarize_paired_run(load_paired_run(raw_path))
    except PairedBenchmarkError as exc:
        raise PairedRunnerError(f"generated paired artifact failed validation: {exc}") from exc
    _atomic_json(output_dir / SUMMARY_NAME, summary)
    state["phase"] = "complete"
    state["completed_at_utc"] = _utc_now()
    state["outputs"] = {"raw": RAW_NAME, "summary": SUMMARY_NAME}
    save_state(output_dir, state)


def run_remaining(
    plan: PairedPlan,
    output_dir: Path,
    state: dict[str, Any],
    *,
    max_new_pairs: int | None = None,
) -> bool:
    if max_new_pairs is not None and max_new_pairs <= 0:
        raise PairedRunnerError("max_new_pairs must be positive")
    if not ensure_warmups(plan, output_dir, state):
        return False
    executed = 0
    for context, repetition in _schedule(plan):
        attempts = _attempts_for(state, context, repetition)
        if attempts:
            latest = attempts[-1]
            if latest["status"] == "passed":
                continue
            if latest["status"] in {"failed", "running"}:
                state["phase"] = "blocked_pair"
                save_state(output_dir, state)
                return False
            if latest["status"] != "invalid":
                raise PairedRunnerError(f"unknown attempt status {latest['status']!r}")
        if max_new_pairs is not None and executed >= max_new_pairs:
            state["phase"] = "paused"
            save_state(output_dir, state)
            return True
        succeeded = _execute_pair(
            plan,
            output_dir,
            state,
            context=context,
            repetition=repetition,
        )
        executed += 1
        if not succeeded:
            return False
    finalize(plan, output_dir, state)
    return True


def _supersede_final_outputs(output_dir: Path, state: dict[str, Any]) -> None:
    existing = [output_dir / RAW_NAME, output_dir / SUMMARY_NAME]
    if not any(path.exists() for path in existing):
        return
    number = len(state["superseded_outputs"]) + 1
    relative = Path("superseded") / f"revision-{number:02d}"
    destination = output_dir / relative
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in existing:
        if path.exists():
            target = destination / path.name
            shutil.move(str(path), target)
            moved.append(str(target.relative_to(output_dir)))
    state["superseded_outputs"].append(
        {"created_at_utc": _utc_now(), "files": moved}
    )
    state.pop("outputs", None)
    state.pop("completed_at_utc", None)


def retry_pair(
    plan: PairedPlan,
    output_dir: Path,
    state: dict[str, Any],
    *,
    context: int,
    repetition: int,
    reason: str,
) -> bool:
    reason = reason.strip()
    if not reason:
        raise PairedRunnerError("retry reason must be non-empty")
    if context not in plan.contexts or not 1 <= repetition <= plan.repetitions:
        raise PairedRunnerError("retry target is outside the predeclared plan")
    attempts = _attempts_for(state, context, repetition)
    if not attempts:
        raise PairedRunnerError("cannot retry a pair that has not been attempted")
    latest = attempts[-1]
    if latest["status"] not in {"passed", "failed", "running"}:
        raise PairedRunnerError(f"pair cannot be retried from status {latest['status']!r}")
    _supersede_final_outputs(output_dir, state)
    latest["status"] = "invalid"
    latest["invalid_reason"] = reason
    latest["invalidated_at_utc"] = _utc_now()
    state["phase"] = "retrying"
    save_state(output_dir, state)
    succeeded = _execute_pair(
        plan,
        output_dir,
        state,
        context=context,
        repetition=repetition,
    )
    if not succeeded:
        return False
    state["phase"] = "paused"
    save_state(output_dir, state)
    return True


def state_summary(plan: PairedPlan, state: dict[str, Any]) -> dict[str, Any]:
    expected = len(plan.contexts) * plan.repetitions
    latest = []
    for context, repetition in _schedule(plan):
        attempts = _attempts_for(state, context, repetition)
        if attempts:
            latest.append(attempts[-1]["status"])
    return {
        "phase": state["phase"],
        "expected_pairs": expected,
        "valid_final_pairs": latest.count("passed"),
        "blocked_pairs": latest.count("failed") + latest.count("running"),
        "invalid_attempts_retained": sum(
            attempt["status"] == "invalid" for attempt in state["attempts"]
        ),
        "warmups_passed": {
            label: any(item["status"] == "passed" for item in state["warmups"][label])
            for label in ("oracle", "candidate")
        },
    }
