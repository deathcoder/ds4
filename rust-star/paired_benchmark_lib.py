"""Validation and aggregation for Rust Star paired benchmark artifacts."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RAW_SCHEMA = "rust-star-paired-raw-v1"
SUMMARY_SCHEMA = "rust-star-paired-summary-v1"
PROTOCOL = "rust-star-paired-benchmark-v1"
CORRECTNESS_CLASSES = frozenset({"C0", "C1", "C2", "C3"})
SHA256_LENGTH = 64
COMMIT_LENGTH = 40
INTEGER_METRICS = (
    "ctx_tokens",
    "prefill_tokens",
    "gen_tokens",
    "gen_steady_tokens",
    "kvcache_bytes",
    "process_peak_bytes",
)
FLOAT_METRICS = (
    "prefill_tps",
    "prefill_ms",
    "gen_tps",
    "gen_ms",
    "gen_first_ms",
    "gen_steady_tps",
    "gen_steady_ms",
    "process_wall_ms",
)
NONNEGATIVE_FLOAT_METRICS = ("model_load_ms",)
THROUGHPUT_METRICS = ("prefill_tps", "gen_tps", "gen_steady_tps")
RATIO_METRICS = (*THROUGHPUT_METRICS, "gen_first_ms")


class PairedBenchmarkError(RuntimeError):
    """The paired benchmark artifact violates its declared contract."""


def _reject_json_constant(value: str) -> None:
    raise PairedBenchmarkError(f"non-standard JSON numeric constant is forbidden: {value}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PairedBenchmarkError(f"{name} must be a JSON object")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PairedBenchmarkError(f"{name} must be a non-empty string")
    return value


def _hex(value: Any, name: str, length: int) -> str:
    text = _required_string(value, name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise PairedBenchmarkError(f"{name} must be {length} lowercase hexadecimal characters")
    return text


def _positive_int(value: Any, name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise PairedBenchmarkError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if not _is_int(value) or value < 0:
        raise PairedBenchmarkError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedBenchmarkError(f"{name} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        requirement = "positive finite" if positive else "nonnegative finite"
        raise PairedBenchmarkError(f"{name} must be {requirement}")
    return number


@dataclass(frozen=True)
class EngineIdentity:
    source_commit: str
    source_tree: str
    executable_sha256: str
    model_sha256: str
    prompt_sha256: str
    backend: str
    build_configuration: dict[str, Any]
    runtime_configuration: dict[str, Any]


@dataclass(frozen=True)
class MetricRow:
    values: dict[str, int | float]


@dataclass(frozen=True)
class Pair:
    context: int
    repetition: int
    attempt: int
    order: str
    valid: bool
    invalid_reason: str | None
    oracle: MetricRow | None
    candidate: MetricRow | None


@dataclass(frozen=True)
class PairedRun:
    path: Path
    host_manifest_sha256: str
    correctness_manifest_sha256: str
    correctness_class: str
    contexts: tuple[int, ...]
    repetitions: int
    gen_tokens: int
    sampling: str
    primary_metric: str
    oracle: EngineIdentity
    candidate: EngineIdentity
    pairs: tuple[Pair, ...]


def _parse_identity(value: Any, name: str) -> EngineIdentity:
    payload = _required_dict(value, name)
    return EngineIdentity(
        source_commit=_hex(payload.get("source_commit"), f"{name}.source_commit", COMMIT_LENGTH),
        source_tree=_hex(payload.get("source_tree"), f"{name}.source_tree", COMMIT_LENGTH),
        executable_sha256=_hex(
            payload.get("executable_sha256"), f"{name}.executable_sha256", SHA256_LENGTH
        ),
        model_sha256=_hex(payload.get("model_sha256"), f"{name}.model_sha256", SHA256_LENGTH),
        prompt_sha256=_hex(payload.get("prompt_sha256"), f"{name}.prompt_sha256", SHA256_LENGTH),
        backend=_required_string(payload.get("backend"), f"{name}.backend"),
        build_configuration=_required_dict(
            payload.get("build_configuration"), f"{name}.build_configuration"
        ),
        runtime_configuration=_required_dict(
            payload.get("runtime_configuration"), f"{name}.runtime_configuration"
        ),
    )


def _parse_metric_row(
    value: Any,
    name: str,
    *,
    context: int,
    gen_tokens: int,
) -> MetricRow:
    payload = _required_dict(value, name)
    parsed: dict[str, int | float] = {}
    for metric in INTEGER_METRICS:
        parsed[metric] = _nonnegative_int(payload.get(metric), f"{name}.{metric}")
    for metric in FLOAT_METRICS:
        parsed[metric] = _finite_number(payload.get(metric), f"{name}.{metric}", positive=True)
    for metric in NONNEGATIVE_FLOAT_METRICS:
        parsed[metric] = _finite_number(payload.get(metric), f"{name}.{metric}", positive=False)
    if parsed["ctx_tokens"] != context:
        raise PairedBenchmarkError(
            f"{name}.ctx_tokens is {parsed['ctx_tokens']}, expected context {context}"
        )
    if parsed["prefill_tokens"] != context:
        raise PairedBenchmarkError(
            f"{name}.prefill_tokens is {parsed['prefill_tokens']}, expected context {context}"
        )
    if parsed["gen_tokens"] != gen_tokens:
        raise PairedBenchmarkError(
            f"{name}.gen_tokens is {parsed['gen_tokens']}, expected {gen_tokens}"
        )
    if parsed["gen_steady_tokens"] != parsed["gen_tokens"] - 1:
        raise PairedBenchmarkError(f"{name}.gen_steady_tokens must be gen_tokens minus one")
    if parsed["prefill_tokens"] == 0 or parsed["gen_steady_tokens"] == 0:
        raise PairedBenchmarkError(f"{name} timed token counts must be positive")
    return MetricRow(parsed)


def load_paired_run(path: Path) -> PairedRun:
    try:
        with path.open(encoding="utf-8") as stream:
            root = json.load(stream, parse_constant=_reject_json_constant)
    except PairedBenchmarkError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedBenchmarkError(f"cannot read paired benchmark {path}: {exc}") from exc

    payload = _required_dict(root, "root")
    if payload.get("schema") != RAW_SCHEMA:
        raise PairedBenchmarkError(f"schema must be {RAW_SCHEMA!r}")
    if payload.get("protocol") != PROTOCOL:
        raise PairedBenchmarkError(f"protocol must be {PROTOCOL!r}")
    correctness_class = _required_string(payload.get("correctness_class"), "correctness_class")
    if correctness_class not in CORRECTNESS_CLASSES:
        raise PairedBenchmarkError(f"unknown correctness_class {correctness_class!r}")
    host_manifest_sha256 = _hex(
        payload.get("host_manifest_sha256"), "host_manifest_sha256", SHA256_LENGTH
    )
    correctness_manifest_sha256 = _hex(
        payload.get("correctness_manifest_sha256"),
        "correctness_manifest_sha256",
        SHA256_LENGTH,
    )

    configuration = _required_dict(payload.get("configuration"), "configuration")
    raw_contexts = configuration.get("contexts")
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise PairedBenchmarkError("configuration.contexts must be a non-empty array")
    contexts = tuple(_positive_int(value, "configuration.contexts[]") for value in raw_contexts)
    if tuple(sorted(set(contexts))) != contexts:
        raise PairedBenchmarkError("configuration.contexts must be sorted and unique")
    repetitions = _positive_int(configuration.get("repetitions"), "configuration.repetitions")
    gen_tokens = _positive_int(configuration.get("gen_tokens"), "configuration.gen_tokens")
    sampling = _required_string(configuration.get("sampling"), "configuration.sampling")
    primary_metric = _required_string(
        configuration.get("primary_metric"), "configuration.primary_metric"
    )
    if primary_metric != "gen_steady_tps":
        raise PairedBenchmarkError("configuration.primary_metric must be 'gen_steady_tps'")

    oracle = _parse_identity(payload.get("oracle"), "oracle")
    candidate = _parse_identity(payload.get("candidate"), "candidate")
    if candidate.model_sha256 != oracle.model_sha256:
        raise PairedBenchmarkError("candidate and oracle model SHA-256 differ")
    if candidate.prompt_sha256 != oracle.prompt_sha256:
        raise PairedBenchmarkError("candidate and oracle prompt SHA-256 differ")
    if candidate.backend != oracle.backend:
        raise PairedBenchmarkError("candidate and oracle backends differ")

    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise PairedBenchmarkError("pairs must be a JSON array")
    expected_keys = {(context, repetition) for context in contexts for repetition in range(1, repetitions + 1)}
    observed_attempts: set[tuple[int, int, int]] = set()
    pairs: list[Pair] = []
    for index, raw_pair in enumerate(raw_pairs):
        name = f"pairs[{index}]"
        pair = _required_dict(raw_pair, name)
        context = _positive_int(pair.get("context"), f"{name}.context")
        repetition = _positive_int(pair.get("repetition"), f"{name}.repetition")
        attempt = _positive_int(pair.get("attempt"), f"{name}.attempt")
        key = (context, repetition)
        if key not in expected_keys:
            raise PairedBenchmarkError(f"{name} has undeclared context/repetition {key}")
        attempt_key = (context, repetition, attempt)
        if attempt_key in observed_attempts:
            raise PairedBenchmarkError(f"duplicate context/repetition/attempt {attempt_key}")
        observed_attempts.add(attempt_key)
        order = _required_string(pair.get("order"), f"{name}.order")
        expected_order = "AB" if repetition % 2 else "BA"
        if order != expected_order:
            raise PairedBenchmarkError(
                f"{name}.order is {order!r}, expected {expected_order!r} for repetition {repetition}"
            )
        valid = pair.get("valid")
        if not isinstance(valid, bool):
            raise PairedBenchmarkError(f"{name}.valid must be a boolean")
        invalid_reason = pair.get("invalid_reason")
        if valid:
            if invalid_reason not in (None, ""):
                raise PairedBenchmarkError(f"{name} is valid but has invalid_reason")
            oracle_row = _parse_metric_row(
                pair.get("oracle"), f"{name}.oracle", context=context, gen_tokens=gen_tokens
            )
            candidate_row = _parse_metric_row(
                pair.get("candidate"),
                f"{name}.candidate",
                context=context,
                gen_tokens=gen_tokens,
            )
            reason = None
        else:
            reason = _required_string(invalid_reason, f"{name}.invalid_reason")
            oracle_row = None
            candidate_row = None
        pairs.append(Pair(context, repetition, attempt, order, valid, reason, oracle_row, candidate_row))

    for key in sorted(expected_keys):
        attempts = sorted(
            (pair for pair in pairs if (pair.context, pair.repetition) == key),
            key=lambda pair: pair.attempt,
        )
        if not attempts:
            raise PairedBenchmarkError(f"missing predeclared pair {key}")
        numbers = [pair.attempt for pair in attempts]
        if numbers != list(range(1, len(attempts) + 1)):
            raise PairedBenchmarkError(f"attempts for {key} are not contiguous from one: {numbers}")
        if any(pair.valid for pair in attempts[:-1]) or not attempts[-1].valid:
            raise PairedBenchmarkError(
                f"attempts for {key} must end in the only valid pair; earlier attempts must be invalid"
            )
    observed_initial_order = [
        (pair.context, pair.repetition) for pair in pairs if pair.attempt == 1
    ]
    expected_initial_order = []
    for repetition in range(1, repetitions + 1):
        scheduled_contexts = contexts if repetition % 2 else tuple(reversed(contexts))
        expected_initial_order.extend((context, repetition) for context in scheduled_contexts)
    if observed_initial_order != expected_initial_order:
        raise PairedBenchmarkError(
            "initial attempts do not follow repetition-major alternating context order"
        )
    return PairedRun(
        path=path,
        host_manifest_sha256=host_manifest_sha256,
        correctness_manifest_sha256=correctness_manifest_sha256,
        correctness_class=correctness_class,
        contexts=contexts,
        repetitions=repetitions,
        gen_tokens=gen_tokens,
        sampling=sampling,
        primary_metric=primary_metric,
        oracle=oracle,
        candidate=candidate,
        pairs=tuple(pairs),
    )


def _mad(values: Iterable[float]) -> float:
    sequence = list(values)
    center = statistics.median(sequence)
    return statistics.median(abs(value - center) for value in sequence)


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    sequence = list(values)
    if not sequence:
        return {"count": 0}
    return {
        "count": len(sequence),
        "median": statistics.median(sequence),
        "mad": _mad(sequence),
        "min": min(sequence),
        "max": max(sequence),
    }


def summarize_paired_run(run: PairedRun) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    for context in run.contexts:
        selected = [pair for pair in run.pairs if pair.context == context]
        valid = [pair for pair in selected if pair.valid]
        invalid = [pair for pair in selected if not pair.valid]
        oracle_metrics = {
            metric: _distribution(pair.oracle.values[metric] for pair in valid if pair.oracle)
            for metric in FLOAT_METRICS
        }
        candidate_metrics = {
            metric: _distribution(pair.candidate.values[metric] for pair in valid if pair.candidate)
            for metric in FLOAT_METRICS
        }
        ratios = {
            metric: _distribution(
                pair.candidate.values[metric] / pair.oracle.values[metric]
                for pair in valid
                if pair.oracle and pair.candidate
            )
            for metric in RATIO_METRICS
        }
        contexts.append(
            {
                "context": context,
                "predeclared_pairs": run.repetitions,
                "valid_pairs": len(valid),
                "invalid_pairs": [
                    {
                        "repetition": pair.repetition,
                        "attempt": pair.attempt,
                        "order": pair.order,
                        "reason": pair.invalid_reason,
                    }
                    for pair in invalid
                ],
                "oracle": oracle_metrics,
                "candidate": candidate_metrics,
                "candidate_over_oracle": ratios,
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "source_path": str(run.path),
        "correctness_class": run.correctness_class,
        "headline_eligible": run.correctness_class == "C0",
        "host_manifest_sha256": run.host_manifest_sha256,
        "correctness_manifest_sha256": run.correctness_manifest_sha256,
        "configuration": {
            "contexts": list(run.contexts),
            "repetitions": run.repetitions,
            "gen_tokens": run.gen_tokens,
            "sampling": run.sampling,
            "primary_metric": run.primary_metric,
        },
        "oracle": run.oracle.__dict__,
        "candidate": run.candidate.__dict__,
        "contexts": contexts,
    }
