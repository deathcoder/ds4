"""Shared validation and comparison primitives for Rust Star oracle artifacts."""

from __future__ import annotations

import contextlib
import hashlib
import heapq
import json
import math
import os
import re
import struct
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


ORACLE_SCHEMA = "rust-star-oracle-manifest-v1"
DIFFERENTIAL_FIXTURE_SCHEMA = "rust-star-differential-fixture-v1"
ORACLE_ID = "oracle-v1"
SOURCE_COMMIT = "b0309611041655f4e45671cfd9c9886aff161406"
SOURCE_TREE = "20c11af22f90a0bdf25da860da5ef06de4064060"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
METADATA_KEYS = (
    "vocab",
    "prompt_tokens",
    "frontier_tokens",
    "ctx",
    "quant_bits",
    "quality",
)


class ArtifactError(RuntimeError):
    """An artifact is invalid or incompatible with the requested operation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ArtifactError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_reject_json_constant)
    except ArtifactError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc


def float32_value_and_bits(value: Any) -> tuple[float, int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"logit is not a JSON number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactError(f"non-finite logit is forbidden: {number!r}")
    try:
        packed = struct.pack("<f", number)
    except OverflowError as exc:
        raise ArtifactError(f"logit is outside finite FP32 range: {number!r}") from exc
    rounded = struct.unpack("<f", packed)[0]
    if not math.isfinite(rounded):
        raise ArtifactError(f"logit rounds outside finite FP32 range: {number!r}")
    bits = struct.unpack("<I", packed)[0]
    return rounded, bits


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise ArtifactError("logit vector is empty")
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


@dataclass(frozen=True)
class LogitArtifact:
    path: Path
    metadata: dict[str, Any]
    values: tuple[float, ...]
    bits: tuple[int, ...]


def load_logit_artifact(path: Path) -> LogitArtifact:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ArtifactError(f"logit artifact must be a JSON object: {path}")
    vocab = payload.get("vocab")
    raw_logits = payload.get("logits")
    if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab <= 0:
        raise ArtifactError(f"invalid vocab in {path}: {vocab!r}")
    if not isinstance(raw_logits, list) or len(raw_logits) != vocab:
        actual = len(raw_logits) if isinstance(raw_logits, list) else None
        raise ArtifactError(f"logit length mismatch in {path}: vocab={vocab}, logits={actual}")
    for key in ("prompt_tokens", "frontier_tokens", "ctx", "quant_bits"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactError(f"missing or invalid {key} in {path}: {value!r}")
    if not isinstance(payload.get("quality"), bool):
        raise ArtifactError(f"missing or invalid quality flag in {path}: {payload.get('quality')!r}")
    recorded_argmax = payload.get("argmax_id")
    if (
        isinstance(recorded_argmax, bool)
        or not isinstance(recorded_argmax, int)
        or recorded_argmax < 0
        or recorded_argmax >= vocab
    ):
        raise ArtifactError(f"missing or invalid argmax_id in {path}: {recorded_argmax!r}")

    values: list[float] = []
    bits: list[int] = []
    for index, raw in enumerate(raw_logits):
        try:
            value, bit_pattern = float32_value_and_bits(raw)
        except ArtifactError as exc:
            raise ArtifactError(f"invalid logit at index {index} in {path}: {exc}") from exc
        values.append(value)
        bits.append(bit_pattern)

    computed_argmax = _argmax(values)
    if recorded_argmax != computed_argmax:
        raise ArtifactError(
            f"argmax metadata mismatch in {path}: recorded={recorded_argmax}, computed={computed_argmax}"
        )
    metadata = {key: payload.get(key) for key in METADATA_KEYS}
    metadata.update({
        "source": payload.get("source"),
        "backend": payload.get("backend"),
        "model": payload.get("model"),
        "argmax_id": computed_argmax,
    })
    return LogitArtifact(path=path, metadata=metadata, values=tuple(values), bits=tuple(bits))


def _ordered_float_bits(bits: int) -> int:
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return bits | 0x80000000


def _top_indices(values: Sequence[float], count: int) -> set[int]:
    count = min(max(0, count), len(values))
    return set(heapq.nlargest(count, range(len(values)), key=lambda index: (values[index], -index)))


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def compare_logit_artifacts(
    reference: LogitArtifact,
    candidate: LogitArtifact,
    *,
    mismatch_limit: int = 20,
) -> dict[str, Any]:
    metadata_mismatches = [
        {
            "field": key,
            "reference": reference.metadata.get(key),
            "candidate": candidate.metadata.get(key),
        }
        for key in METADATA_KEYS
        if reference.metadata.get(key) != candidate.metadata.get(key)
    ]
    if len(reference.values) != len(candidate.values):
        return {
            "schema": "rust-star-logit-comparison-v1",
            "classification": "not-C0",
            "c0_exact": False,
            "metadata_mismatches": metadata_mismatches,
            "shape_mismatch": {
                "reference": len(reference.values),
                "candidate": len(candidate.values),
            },
        }

    mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    squared_errors: list[float] = []
    max_abs_error = -1.0
    max_abs_index = 0
    max_relative_error = -1.0
    max_relative_index = 0
    max_ulp = 0
    max_ulp_index = 0
    dot_terms: list[float] = []
    reference_square_terms: list[float] = []
    candidate_square_terms: list[float] = []

    for index, (ref_value, cand_value, ref_bits, cand_bits) in enumerate(
        zip(reference.values, candidate.values, reference.bits, candidate.bits)
    ):
        if ref_bits != cand_bits:
            mismatch_count += 1
            if len(mismatch_examples) < mismatch_limit:
                mismatch_examples.append({
                    "index": index,
                    "reference": ref_value,
                    "candidate": cand_value,
                    "reference_bits": f"0x{ref_bits:08x}",
                    "candidate_bits": f"0x{cand_bits:08x}",
                })
        absolute = abs(cand_value - ref_value)
        relative = absolute / max(abs(ref_value), 1e-30)
        ulp = abs(_ordered_float_bits(ref_bits) - _ordered_float_bits(cand_bits))
        abs_errors.append(absolute)
        squared_errors.append(absolute * absolute)
        dot_terms.append(ref_value * cand_value)
        reference_square_terms.append(ref_value * ref_value)
        candidate_square_terms.append(cand_value * cand_value)
        if absolute > max_abs_error:
            max_abs_error = absolute
            max_abs_index = index
        if relative > max_relative_error:
            max_relative_error = relative
            max_relative_index = index
        if ulp > max_ulp:
            max_ulp = ulp
            max_ulp_index = index

    vector_length = len(reference.values)
    reference_norm = math.sqrt(math.fsum(reference_square_terms))
    candidate_norm = math.sqrt(math.fsum(candidate_square_terms))
    if reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 1.0 if reference.bits == candidate.bits else 0.0
    else:
        cosine = math.fsum(dot_terms) / (reference_norm * candidate_norm)

    reference_log_z = _logsumexp(reference.values)
    candidate_log_z = _logsumexp(candidate.values)
    kl_terms = (
        math.exp(ref_value - reference_log_z)
        * ((ref_value - reference_log_z) - (cand_value - candidate_log_z))
        for ref_value, cand_value in zip(reference.values, candidate.values)
    )
    kl_reference_candidate = max(0.0, math.fsum(kl_terms))

    top_k: dict[str, Any] = {}
    for count in (1, 10, 50):
        if count > vector_length:
            continue
        ref_top = _top_indices(reference.values, count)
        cand_top = _top_indices(candidate.values, count)
        intersection = len(ref_top & cand_top)
        union = len(ref_top | cand_top)
        top_k[str(count)] = {
            "intersection": intersection,
            "jaccard": intersection / union if union else 1.0,
        }

    c0_exact = mismatch_count == 0 and not metadata_mismatches
    return {
        "schema": "rust-star-logit-comparison-v1",
        "classification": "C0" if c0_exact else "not-C0",
        "c0_exact": c0_exact,
        "metadata_mismatches": metadata_mismatches,
        "vocab": vector_length,
        "bit_mismatches": mismatch_count,
        "bit_mismatch_rate": mismatch_count / vector_length,
        "mismatch_examples": mismatch_examples,
        "argmax_match": reference.metadata["argmax_id"] == candidate.metadata["argmax_id"],
        "reference_argmax": reference.metadata["argmax_id"],
        "candidate_argmax": candidate.metadata["argmax_id"],
        "max_absolute_error": max_abs_error,
        "max_absolute_error_index": max_abs_index,
        "mean_absolute_error": math.fsum(abs_errors) / vector_length,
        "rmse": math.sqrt(math.fsum(squared_errors) / vector_length),
        "max_relative_error": max_relative_error,
        "max_relative_error_index": max_relative_index,
        "max_ulp_distance": max_ulp,
        "max_ulp_distance_index": max_ulp_index,
        "cosine_similarity": cosine,
        "kl_reference_candidate": kl_reference_candidate,
        "top_k": top_k,
    }


def _safe_relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ArtifactError(f"artifact path must be relative: {relative}")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        common = Path(os.path.commonpath([resolved_root, resolved]))
    except ValueError as exc:
        raise ArtifactError(f"artifact path escapes bundle: {relative}") from exc
    if common != resolved_root:
        raise ArtifactError(f"artifact path escapes bundle: {relative}")
    return resolved


def _artifact_descriptors(value: Any, location: str = "manifest") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield location, value
        for key, child in value.items():
            yield from _artifact_descriptors(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _artifact_descriptors(child, f"{location}[{index}]")


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{location} must be a nonempty string")
    return value


def _require_nonnegative_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{location} must be a nonnegative integer")
    return value


def validate_differential_fixture(root: Path) -> dict[str, Any]:
    """Validate a self-contained kernel, layer-segment, or decode-step fixture."""
    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ArtifactError("fixture manifest.json must contain an object")
    if manifest.get("schema") != DIFFERENTIAL_FIXTURE_SCHEMA:
        raise ArtifactError(f"unexpected fixture schema: {manifest.get('schema')!r}")
    fixture_id = _require_nonempty_string(manifest.get("fixture_id"), "fixture_id")

    oracle = manifest.get("oracle")
    if not isinstance(oracle, dict):
        raise ArtifactError("fixture oracle is missing")
    if oracle.get("commit") != SOURCE_COMMIT or oracle.get("tree") != SOURCE_TREE:
        raise ArtifactError("fixture oracle commit/tree does not match oracle-v1")
    executable_sha256 = oracle.get("capture_executable_sha256")
    if not isinstance(executable_sha256, str) or not SHA256_RE.fullmatch(executable_sha256):
        raise ArtifactError("fixture oracle capture executable SHA-256 is invalid")

    model = manifest.get("model")
    if not isinstance(model, dict) or not SHA256_RE.fullmatch(str(model.get("sha256", ""))):
        raise ArtifactError("fixture model SHA-256 is invalid")

    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ArtifactError("fixture scope is missing")
    kind = scope.get("kind")
    if kind not in {"kernel", "layer-segment", "decode-step"}:
        raise ArtifactError(f"invalid fixture scope kind: {kind!r}")
    if scope.get("phase") not in {"prefill", "decode"}:
        raise ArtifactError(f"invalid fixture phase: {scope.get('phase')!r}")
    _require_nonnegative_integer(scope.get("position"), "scope.position")
    layer = scope.get("layer")
    if layer is not None:
        _require_nonnegative_integer(layer, "scope.layer")

    operations = manifest.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ArtifactError("fixture operations must be a nonempty array")
    operation_names: set[str] = set()
    for index, operation in enumerate(operations):
        location = f"operations[{index}]"
        if not isinstance(operation, dict):
            raise ArtifactError(f"{location} must be an object")
        name = _require_nonempty_string(operation.get("name"), f"{location}.name")
        _require_nonempty_string(operation.get("kernel"), f"{location}.kernel")
        if name in operation_names:
            raise ArtifactError(f"duplicate operation name: {name}")
        operation_names.add(name)

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ArtifactError("fixture tensors must be a nonempty array")
    tensor_names: set[str] = set()
    tensor_paths: set[str] = set()
    verified_bytes = 0
    for index, descriptor in enumerate(tensors):
        location = f"tensors[{index}]"
        if not isinstance(descriptor, dict):
            raise ArtifactError(f"{location} must be an object")
        name = _require_nonempty_string(descriptor.get("name"), f"{location}.name")
        if name in tensor_names:
            raise ArtifactError(f"duplicate tensor name: {name}")
        tensor_names.add(name)
        if descriptor.get("role") not in {"input", "intermediate", "output"}:
            raise ArtifactError(f"invalid {location}.role: {descriptor.get('role')!r}")
        dtype = descriptor.get("dtype")
        encoding = descriptor.get("encoding")
        valid_encoding = (
            dtype == "f32" and encoding == "little-endian-ieee754-binary32"
        ) or (
            dtype == "i32" and encoding == "little-endian-signed-integer32"
        )
        if not valid_encoding:
            raise ArtifactError(f"{location} must use a supported little-endian 32-bit encoding")
        shape = descriptor.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ArtifactError(f"{location}.shape must be a nonempty array")
        elements = 1
        for dimension_index, dimension in enumerate(shape):
            value = _require_nonnegative_integer(dimension, f"{location}.shape[{dimension_index}]")
            if value == 0:
                raise ArtifactError(f"{location}.shape dimensions must be positive")
            elements *= value
        relative = _require_nonempty_string(descriptor.get("path"), f"{location}.path")
        if relative in tensor_paths:
            raise ArtifactError(f"duplicate tensor path: {relative}")
        tensor_paths.add(relative)
        expected_bytes = elements * 4
        if descriptor.get("bytes") != expected_bytes:
            raise ArtifactError(
                f"{location}.bytes does not match shape: expected={expected_bytes}, actual={descriptor.get('bytes')!r}"
            )
        expected_sha256 = descriptor.get("sha256")
        if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
            raise ArtifactError(f"{location}.sha256 is invalid")
        path = _safe_relative(root, relative)
        if not path.is_file():
            raise ArtifactError(f"fixture tensor is not a regular file: {relative}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ArtifactError(f"fixture tensor integrity mismatch: {relative}")
        with path.open("rb") as stream:
            for element_index, packed in enumerate(iter(lambda: stream.read(4), b"")):
                if len(packed) != 4:
                    raise ArtifactError(f"fixture tensor is truncated: {relative}")
                if dtype == "f32" and not math.isfinite(struct.unpack("<f", packed)[0]):
                    raise ArtifactError(
                        f"fixture tensor contains non-finite f32 at element {element_index}: {relative}"
                    )
        verified_bytes += actual_bytes

    return {
        "valid": True,
        "schema": DIFFERENTIAL_FIXTURE_SCHEMA,
        "fixture_id": fixture_id,
        "scope": kind,
        "operations": len(operations),
        "tensors": len(tensors),
        "verified_bytes": verified_bytes,
        "model_sha256": model["sha256"],
    }


def validate_oracle_bundle(root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest.json must contain an object")
    if manifest.get("schema") != ORACLE_SCHEMA:
        raise ArtifactError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    if manifest.get("oracle_id") != ORACLE_ID:
        raise ArtifactError(f"unexpected oracle id: {manifest.get('oracle_id')!r}")
    complete = manifest.get("status") == "complete"
    if not complete and not allow_partial:
        raise ArtifactError(f"bundle is not complete: status={manifest.get('status')!r}")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ArtifactError("manifest source is missing")
    if source.get("commit") != SOURCE_COMMIT or source.get("tree") != SOURCE_TREE:
        raise ArtifactError("oracle source commit/tree does not match oracle-v1")

    model = manifest.get("model")
    if not isinstance(model, dict) and complete:
        raise ArtifactError("model identity is missing")
    if isinstance(model, dict):
        if not isinstance(model.get("bytes"), int) or model["bytes"] <= 0:
            raise ArtifactError("model byte size is invalid")
        if not isinstance(model.get("sha256"), str) or not SHA256_RE.fullmatch(model["sha256"]):
            raise ArtifactError("model SHA-256 is invalid")
        if model.get("absolute_path_recorded") is not False:
            raise ArtifactError("manifest does not confirm absolute model path exclusion")

    capture_kit = manifest.get("capture_kit")
    if complete and (not isinstance(capture_kit, dict) or capture_kit.get("tracked_worktree") != "clean"):
        raise ArtifactError("capture-kit revision is missing or was not clean")
    if isinstance(capture_kit, dict) and capture_kit.get("tracked_worktree") != "clean":
        raise ArtifactError("capture-kit worktree was not clean")

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ArtifactError("capture configuration is missing")
    contexts = configuration.get("contexts")
    if not isinstance(contexts, list) or not contexts or any(not isinstance(v, int) for v in contexts):
        raise ArtifactError("capture context list is invalid")

    for section_name, enabled_key in (
        ("correctness", "correctness_enabled"),
        ("conformance", "conformance_enabled"),
        ("performance", "performance_enabled"),
    ):
        section = manifest.get(section_name)
        if not isinstance(section, dict) and complete:
            raise ArtifactError(f"{section_name} section is missing")
        if not isinstance(section, dict):
            continue
        expected_status = "passed" if configuration.get(enabled_key) else "skipped"
        if complete and section.get("status") != expected_status:
            raise ArtifactError(
                f"{section_name} status mismatch: expected {expected_status}, got {section.get('status')}"
            )

    descriptors: list[dict[str, Any]] = []
    seen_paths: dict[str, tuple[int, str]] = {}
    for location, descriptor in _artifact_descriptors(manifest):
        relative = descriptor.get("path")
        size = descriptor.get("bytes")
        digest = descriptor.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ArtifactError(f"invalid artifact path at {location}")
        if not isinstance(size, int) or size < 0:
            raise ArtifactError(f"invalid artifact size at {location}")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ArtifactError(f"invalid artifact SHA-256 at {location}")
        path = _safe_relative(root, relative)
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"artifact is missing, not regular, or a symlink: {relative}")
        actual = (path.stat().st_size, sha256_file(path))
        expected = (size, digest)
        if actual != expected:
            raise ArtifactError(
                f"artifact integrity mismatch for {relative}: expected={expected}, actual={actual}"
            )
        if relative in seen_paths and seen_paths[relative] != expected:
            raise ArtifactError(f"conflicting descriptors for artifact: {relative}")
        seen_paths[relative] = expected
        descriptors.append({"location": location, "path": relative, "bytes": size, "sha256": digest})

    conformance = manifest.get("conformance")
    if (
        isinstance(conformance, dict)
        and configuration.get("conformance_enabled")
        and conformance.get("status") == "passed"
    ):
        runs = conformance.get("runs")
        if not isinstance(runs, list) or {run.get("context") for run in runs} != set(contexts):
            raise ArtifactError("conformance contexts do not match capture configuration")
        for run in runs:
            logits_descriptor = run.get("logits")
            if not isinstance(logits_descriptor, dict) or not isinstance(logits_descriptor.get("path"), str):
                raise ArtifactError("conformance run is missing its logit artifact")
            logits = load_logit_artifact(_safe_relative(root, logits_descriptor["path"]))
            if logits.metadata["frontier_tokens"] != run.get("context"):
                raise ArtifactError("conformance logit frontier does not match its run")
            if logits.metadata["quant_bits"] != 2:
                raise ArtifactError("oracle-v1 conformance artifact is not routed Q2")
            if logits.metadata["model"] != "model.gguf":
                raise ArtifactError("oracle conformance artifact exposes an unexpected model path")

    performance = manifest.get("performance")
    if (
        isinstance(performance, dict)
        and configuration.get("performance_enabled")
        and performance.get("status") == "passed"
    ):
        summary = performance.get("summary", {}).get("summary")
        if not isinstance(summary, list) or {row.get("ctx_tokens") for row in summary} != set(contexts):
            raise ArtifactError("performance summary contexts do not match capture configuration")
        expected_repetitions = configuration.get("performance_repetitions")
        if any(row.get("repetitions") != expected_repetitions for row in summary):
            raise ArtifactError("performance summary repetition count is inconsistent")

    return {
        "schema": "rust-star-oracle-verification-v1",
        "valid": True,
        "oracle_id": manifest["oracle_id"],
        "status": manifest.get("status"),
        "source_commit": source["commit"],
        "model_sha256": model.get("sha256") if isinstance(model, dict) else None,
        "contexts": contexts,
        "verified_artifacts": len(descriptors),
        "verified_bytes": sum(item["bytes"] for item in descriptors),
    }


def _safe_extract_bundle(archive: Path, destination: Path) -> None:
    resolved_destination = destination.resolve()
    try:
        source = tarfile.open(archive, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactError(f"cannot open bundle archive {archive}: {exc}") from exc
    with source:
        for member in source.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ArtifactError(f"bundle contains forbidden link/device entry: {member.name}")
            target = (destination / member.name).resolve()
            try:
                common = Path(os.path.commonpath([resolved_destination, target]))
            except ValueError as exc:
                raise ArtifactError(f"bundle path escapes extraction root: {member.name}") from exc
            if common != resolved_destination:
                raise ArtifactError(f"bundle path escapes extraction root: {member.name}")
        if sys.version_info >= (3, 12):
            source.extractall(destination, filter="data")
        else:
            source.extractall(destination)


def _locate_manifest_root(extracted: Path) -> Path:
    manifests = list(extracted.rglob("manifest.json"))
    if len(manifests) != 1:
        raise ArtifactError(f"bundle archive must contain exactly one manifest.json; found {len(manifests)}")
    return manifests[0].parent


@contextlib.contextmanager
def open_bundle(path: Path) -> Iterator[tuple[Path, str | None]]:
    path = path.expanduser().resolve()
    if path.is_dir():
        if not (path / "manifest.json").is_file():
            raise ArtifactError(f"bundle directory has no manifest.json: {path}")
        yield path, None
        return
    if not path.is_file():
        raise ArtifactError(f"bundle does not exist: {path}")
    archive_sha256 = sha256_file(path)
    with tempfile.TemporaryDirectory(prefix="rust-star-verify-") as temporary:
        destination = Path(temporary)
        _safe_extract_bundle(path, destination)
        yield _locate_manifest_root(destination), archive_sha256
