#!/usr/bin/env python3
"""Capture a reproducible DwarfStar oracle and M1 Ultra performance baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ORACLE_ID = "oracle-v1"
SCHEMA = "rust-star-oracle-manifest-v1"
SOURCE_REPOSITORY = "https://github.com/antirez/ds4.git"
SOURCE_COMMIT = "b0309611041655f4e45671cfd9c9886aff161406"
SOURCE_TREE = "20c11af22f90a0bdf25da860da5ef06de4064060"
DEFAULT_CONTEXTS = (2048, 32768)
FULL_CONTEXTS = (2048, 32768, 131072, 262144, 524288, 1000000)
DEFAULT_REPETITIONS = 3
DEFAULT_GEN_TOKENS = 128
PROMPT_BYTES_PER_TARGET_TOKEN = 8
PROMPT_SEPARATOR = b"\n\n"
SAFE_ENV_KEYS = (
    "CFLAGS",
    "OBJCFLAGS",
    "NATIVE_CPU_FLAG",
    "DS4_METAL_PREFILL_CHUNK",
    "DS4_BENCH_SNAPSHOT_MAX_BYTES",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "rust-star" / "results"


class CaptureError(RuntimeError):
    """A user-actionable capture failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    read = 0
    next_report = time.monotonic() + 10.0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read += len(chunk)
            if progress and time.monotonic() >= next_report:
                percent = (100.0 * read / total) if total else 100.0
                print(f"  hashed {read / (1024**3):.1f}/{total / (1024**3):.1f} GiB ({percent:.1f}%)")
                next_report = time.monotonic() + 10.0
    return digest.hexdigest()


def command_text(command: Sequence[str]) -> str:
    def quote(value: str) -> str:
        if value and all(ch.isalnum() or ch in "-._/:=$" for ch in value):
            return value
        return "'" + value.replace("'", "'\\''") + "'"

    return " ".join(quote(str(part)) for part in command)


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    display_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = list(display_command or command)
    print(f"\n$ {command_text(shown)}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(shown)}\n")
        log.flush()
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()
    elapsed = time.monotonic() - started
    result = {
        "command": [str(part) for part in shown],
        "cwd": ".",
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": returncode,
        "log": f"logs/{log_path.name}",
    }
    if returncode != 0:
        raise CaptureError(
            f"command failed with exit code {returncode}: {command_text(shown)}; "
            f"see logs/{log_path.name}"
        )
    return result


def quiet_output(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
        return {
            "command": [str(part) for part in command],
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": [str(part) for part in command],
            "exit_code": None,
            "output": str(exc),
        }


def parse_contexts(value: str) -> tuple[int, ...]:
    contexts: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            context = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid context: {raw}") from exc
        if context < 2:
            raise argparse.ArgumentTypeError("contexts must be at least 2 tokens")
        if context > 1_000_000:
            raise argparse.ArgumentTypeError("contexts above the model's 1M limit are unsupported")
        contexts.append(context)
    if not contexts:
        raise argparse.ArgumentTypeError("provide at least one context")
    if len(contexts) != len(set(contexts)):
        raise argparse.ArgumentTypeError("contexts must not contain duplicates")
    return tuple(sorted(contexts))


def parse_sw_vers(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def filtered_system_profiler() -> dict[str, Any]:
    command = ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"]
    raw = quiet_output(command)
    if raw["exit_code"] != 0:
        return {"error": raw["output"]}
    try:
        parsed = json.loads(raw["output"])
    except json.JSONDecodeError as exc:
        return {"error": f"invalid system_profiler JSON: {exc}"}

    hardware_allowlist = {
        "machine_name",
        "machine_model",
        "model_name",
        "model_identifier",
        "chip_type",
        "number_processors",
        "number_cores",
        "physical_memory",
    }
    display_allowlist = {
        "_name",
        "sppci_model",
        "sppci_bus",
        "sppci_cores",
        "spdisplays_metal",
        "spdisplays_metalfamily",
    }

    def filter_rows(rows: Any, allowlist: set[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if not isinstance(rows, list):
            return result
        for row in rows:
            if isinstance(row, dict):
                result.append({key: row[key] for key in sorted(allowlist) if key in row})
        return result

    return {
        "hardware": filter_rows(parsed.get("SPHardwareDataType"), hardware_allowlist),
        "displays": filter_rows(parsed.get("SPDisplaysDataType"), display_allowlist),
    }


def collect_host_manifest() -> dict[str, Any]:
    sw_vers = quiet_output(["sw_vers"])
    commands = {
        "uname": quiet_output(["uname", "-srm"]),
        "xcodebuild": quiet_output(["xcodebuild", "-version"]),
        "clang": quiet_output(["xcrun", "clang", "--version"]),
        "metal_path": quiet_output(["xcrun", "-f", "metal"]),
        "make": quiet_output(["make", "--version"]),
        "pmset_thermal": quiet_output(["pmset", "-g", "therm"]),
        "pmset_custom": quiet_output(["pmset", "-g", "custom"]),
        "uptime": quiet_output(["uptime"]),
    }
    sysctls: dict[str, Any] = {}
    for key in (
        "hw.memsize",
        "hw.ncpu",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "machdep.cpu.brand_string",
        "kern.osrelease",
        "kern.osversion",
        "vm.loadavg",
    ):
        value = quiet_output(["sysctl", "-n", key])
        if value["exit_code"] == 0:
            sysctls[key] = value["output"]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos": parse_sw_vers(sw_vers["output"]) if sw_vers["exit_code"] == 0 else {},
        "system_profiler_filtered": filtered_system_profiler(),
        "sysctl_allowlist": sysctls,
        "commands": commands,
    }


def validate_source_commit() -> None:
    commit = quiet_output(["git", "rev-parse", f"{SOURCE_COMMIT}^{{commit}}"], cwd=REPO_ROOT)
    tree = quiet_output(["git", "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"], cwd=REPO_ROOT)
    if commit["exit_code"] != 0 or commit["output"] != SOURCE_COMMIT:
        raise CaptureError(f"pinned source commit {SOURCE_COMMIT} is not available in this clone")
    if tree["exit_code"] != 0 or tree["output"] != SOURCE_TREE:
        raise CaptureError(
            f"pinned source tree mismatch: expected {SOURCE_TREE}, got {tree['output'] or 'unavailable'}"
        )


def capture_kit_revision() -> dict[str, str]:
    tracked_status = quiet_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
    )
    if tracked_status["exit_code"] != 0:
        raise CaptureError("unable to inspect the capture-kit git checkout")
    if tracked_status["output"]:
        raise CaptureError(
            "tracked files in the capture-kit checkout are modified; run from a clean clone or commit them first"
        )

    values: dict[str, str] = {}
    for name, revision in (
        ("commit", "HEAD^{commit}"),
        ("tree", "HEAD^{tree}"),
        ("branch", "--abbrev-ref HEAD"),
    ):
        command = ["git", "rev-parse"] + revision.split()
        result = quiet_output(command, cwd=REPO_ROOT)
        if result["exit_code"] != 0 or not result["output"]:
            raise CaptureError(f"unable to resolve capture-kit {name}")
        values[name] = result["output"]
    values["tracked_worktree"] = "clean"
    return values


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r") as source:
        for member in source.getmembers():
            target = (destination / member.name).resolve()
            try:
                common = Path(os.path.commonpath([destination_resolved, target]))
            except ValueError as exc:
                raise CaptureError(f"unsafe archive path: {member.name}") from exc
            if common != destination_resolved:
                raise CaptureError(f"unsafe archive path: {member.name}")
        source.extractall(destination)


def export_source(destination: Path, log_dir: Path) -> dict[str, Any]:
    archive = destination.parent / "source.tar"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "source_export.log"
    with archive.open("wb") as output, log_path.open("w", encoding="utf-8") as log:
        command = ["git", "archive", "--format=tar", SOURCE_COMMIT]
        log.write(f"$ {command_text(command)}\n")
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=log,
            check=False,
        )
    if process.returncode != 0:
        raise CaptureError(f"git archive failed; see {log_path}")
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract_tar(archive, destination)
    archive.unlink()
    return {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "repository": SOURCE_REPOSITORY,
        "export_log": f"logs/{log_path.name}",
    }


def prepare_prompt(source_dir: Path, max_context: int) -> dict[str, Any]:
    base = source_dir / "speed-bench" / "promessi_sposi.txt"
    if not base.is_file():
        raise CaptureError(f"missing benchmark prompt in source snapshot: {base}")
    base_size = base.stat().st_size
    target_bytes = max(base_size, max_context * PROMPT_BYTES_PER_TARGET_TOKEN)
    repeats = max(1, math.ceil(target_bytes / (base_size + len(PROMPT_SEPARATOR))))
    expanded = source_dir / "rust_star_oracle_prompt.txt"
    with expanded.open("wb") as output:
        for index in range(repeats):
            if index:
                output.write(PROMPT_SEPARATOR)
            with base.open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    return {
        "base_path": "speed-bench/promessi_sposi.txt",
        "base_bytes": base_size,
        "base_sha256": sha256_file(base),
        "expansion": "base bytes repeated with two newline bytes between copies",
        "repeats": repeats,
        "expanded_bytes": expanded.stat().st_size,
        "expanded_sha256": sha256_file(expanded),
        "runtime_path": expanded.name,
    }


def artifact(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def benchmark_command(
    source_dir: Path,
    *,
    context: int,
    gen_tokens: int,
    csv_path: Path,
    logits_dir: Path | None = None,
) -> list[str]:
    command = [
        str(source_dir / "ds4-bench"),
        "--metal",
        "-m",
        "model.gguf",
        "--prompt-file",
        "rust_star_oracle_prompt.txt",
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
    if logits_dir is not None:
        command.extend(["--dump-frontier-logits-dir", str(logits_dir)])
    return command


def sanitized_benchmark_command(command: Sequence[str], source_dir: Path, run_dir: Path) -> list[str]:
    sanitized: list[str] = []
    for value in command:
        text = str(value)
        text = text.replace(str(source_dir), "$ORACLE_SOURCE")
        text = text.replace(str(run_dir), "$RESULTS")
        sanitized.append(text)
    return sanitized


def read_single_csv_row(path: Path, expected_context: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise CaptureError(f"expected one benchmark row in {path}, found {len(rows)}")
    row = rows[0]
    if int(row["ctx_tokens"]) != expected_context:
        raise CaptureError(
            f"benchmark context mismatch in {path}: expected {expected_context}, got {row['ctx_tokens']}"
        )
    integer_fields = {"ctx_tokens", "prefill_tokens", "gen_tokens", "gen_steady_tokens", "kvcache_bytes"}
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        parsed[key] = int(value) if key in integer_fields else float(value)
    return parsed


def median_absolute_deviation(values: Iterable[float]) -> float:
    sequence = list(values)
    center = statistics.median(sequence)
    return statistics.median(abs(value - center) for value in sequence)


def aggregate_performance(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["ctx_tokens"]), []).append(row)

    metrics = ("prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps")
    summaries: list[dict[str, Any]] = []
    for context in sorted(groups):
        group = groups[context]
        summary: dict[str, Any] = {
            "ctx_tokens": context,
            "repetitions": len(group),
            "gen_tokens": int(group[0]["gen_tokens"]),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_median"] = round(statistics.median(values), 6)
            summary[f"{metric}_mad"] = round(median_absolute_deviation(values), 6)
            summary[f"{metric}_min"] = round(min(values), 6)
            summary[f"{metric}_max"] = round(max(values), 6)
        summaries.append(summary)

    csv_path = run_dir / "performance" / "summary.csv"
    json_path = run_dir / "performance" / "summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    atomic_json(json_path, {"schema": "rust-star-performance-summary-v1", "contexts": summaries})
    return {
        "rows": rows,
        "summary": summaries,
        "summary_csv": artifact(csv_path, run_dir) if summaries else None,
        "summary_json": artifact(json_path, run_dir),
    }


def validate_logits(path: Path, expected_context: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"invalid full-logit artifact {path}: {exc}") from exc
    if payload.get("frontier_tokens") != expected_context:
        raise CaptureError(f"logit frontier mismatch in {path}")
    vocab = payload.get("vocab")
    logits = payload.get("logits")
    if not isinstance(vocab, int) or not isinstance(logits, list) or len(logits) != vocab:
        raise CaptureError(f"invalid vocabulary/logit shape in {path}")
    if any(value is None for value in logits):
        raise CaptureError(f"non-finite logit encoded as null in {path}")
    return {
        "backend": payload.get("backend"),
        "quality": payload.get("quality"),
        "quant_bits": payload.get("quant_bits"),
        "vocab": vocab,
        "argmax_id": payload.get("argmax_id"),
        "argmax_logit": payload.get("argmax_logit"),
    }


def make_archive(run_dir: Path) -> tuple[Path, str]:
    archive = Path(str(run_dir) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(run_dir, arcname=run_dir.name)
    digest = sha256_file(archive, progress=True)
    checksum = Path(str(archive) + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the pinned DwarfStar oracle and M1 Ultra baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=Path, help="DeepSeek V4 Flash 0731 Q2-imatrix GGUF")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--contexts", type=parse_contexts, help="comma-separated context frontiers")
    group.add_argument("--full", action="store_true", help="use the 2K through 1M frontier set")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN_TOKENS)
    parser.add_argument("--output", type=Path, help="empty result directory")
    parser.add_argument("--notes", default="", help="manual environment/thermal notes; included in the archive")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-conformance", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without reading the model or running commands")
    return parser


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.repetitions < 1:
        raise CaptureError("--repetitions must be at least 1")
    if args.gen_tokens < 1:
        raise CaptureError("--gen-tokens must be at least 1")
    contexts = FULL_CONTEXTS if args.full else (args.contexts or DEFAULT_CONTEXTS)
    if not args.dry_run:
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise CaptureError("oracle capture must run on Apple Silicon macOS; use --dry-run elsewhere")
        if not args.model.is_file():
            raise CaptureError(f"model does not exist or is not a file: {args.model}")
        if args.model.is_symlink():
            args.model = args.model.resolve()
    return tuple(contexts)


def dry_run_plan(args: argparse.Namespace, contexts: tuple[int, ...]) -> int:
    plan = {
        "oracle_id": ORACLE_ID,
        "source_commit": SOURCE_COMMIT,
        "model": args.model.name,
        "contexts": contexts,
        "repetitions": args.repetitions,
        "gen_tokens": args.gen_tokens,
        "correctness": not args.skip_correctness,
        "conformance": not args.skip_conformance,
        "performance": not args.skip_performance,
        "full_logit_scope": "post-prefill frontier logits",
    }
    print(json.dumps(plan, indent=2))
    return 0


def capture(args: argparse.Namespace, contexts: tuple[int, ...]) -> int:
    output = args.output or (DEFAULT_RESULTS_ROOT / f"{ORACLE_ID}-{timestamp_id()}")
    output = output.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise CaptureError(f"output path exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise CaptureError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "oracle_id": ORACLE_ID,
        "status": "running",
        "started_at_utc": utc_now(),
        "notes": args.notes,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
        },
        "configuration": {
            "backend": "metal",
            "model_family": "DeepSeek-V4-Flash-0731",
            "expected_quantization": "resident imatrix Q2",
            "contexts": list(contexts),
            "performance_repetitions": args.repetitions,
            "performance_gen_tokens": args.gen_tokens,
            "sampling": "greedy argmax excluding EOS (ds4-bench behavior)",
            "correctness_enabled": not args.skip_correctness,
            "conformance_enabled": not args.skip_conformance,
            "performance_enabled": not args.skip_performance,
            "conformance_scope": "full FP32 logits immediately after prefill at each selected frontier",
            "performance_instrumentation": "frontier-logit dumping disabled",
        },
        "environment_allowlist": {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ},
    }
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)

    try:
        validate_source_commit()
        manifest["capture_kit"] = capture_kit_revision()
        print("Collecting privacy-filtered host manifest...")
        manifest["host"] = collect_host_manifest()

        model = args.model.expanduser().resolve()
        print(f"Hashing model ({model.stat().st_size / (1024**3):.2f} GiB)...")
        manifest["model"] = {
            "filename": model.name,
            "bytes": model.stat().st_size,
            "sha256": sha256_file(model, progress=True),
            "absolute_path_recorded": False,
        }
        atomic_json(manifest_path, manifest)

        with tempfile.TemporaryDirectory(prefix="rust-star-oracle-v1-") as temporary:
            temporary_root = Path(temporary)
            source_dir = temporary_root / "source"
            logs = output / "logs"
            manifest["source"] = export_source(source_dir, logs)

            model_link = source_dir / "model.gguf"
            model_link.symlink_to(model)
            manifest["prompt"] = prepare_prompt(source_dir, max(contexts))

            jobs = max(1, min(os.cpu_count() or 1, 16))
            build_command = ["make", "-j", str(jobs), "ds4-bench", "ds4_test"]
            manifest["build"] = run_logged(
                build_command,
                cwd=source_dir,
                log_path=logs / "build.log",
            )
            manifest["build"]["executables"] = {
                "ds4-bench": {
                    "bytes": (source_dir / "ds4-bench").stat().st_size,
                    "sha256": sha256_file(source_dir / "ds4-bench"),
                },
                "ds4_test": {
                    "bytes": (source_dir / "ds4_test").stat().st_size,
                    "sha256": sha256_file(source_dir / "ds4_test"),
                },
            }
            atomic_json(manifest_path, manifest)

            if args.skip_correctness:
                manifest["correctness"] = {"status": "skipped", "runs": []}
            else:
                correctness_env = dict(os.environ)
                correctness_env["DS4_TEST_MODEL"] = "model.gguf"
                correctness_runs: list[dict[str, Any]] = []
                manifest["correctness"] = {"status": "running", "runs": correctness_runs}
                atomic_json(manifest_path, manifest)
                for name, selector in (
                    ("metal-kernels", "--metal-kernels"),
                    ("official-logprob-vectors", "--logprob-vectors"),
                ):
                    result = run_logged(
                        [str(source_dir / "ds4_test"), selector],
                        cwd=source_dir,
                        log_path=logs / f"correctness_{name}.log",
                        env=correctness_env,
                        display_command=["./ds4_test", selector],
                    )
                    result["name"] = name
                    correctness_runs.append(result)
                    atomic_json(manifest_path, manifest)
                manifest["correctness"] = {"status": "passed", "runs": correctness_runs}
                atomic_json(manifest_path, manifest)

            conformance_runs: list[dict[str, Any]] = []
            if args.skip_conformance:
                manifest["conformance"] = {"status": "skipped", "runs": []}
            else:
                manifest["conformance"] = {
                    "status": "running",
                    "encoding": "JSON decimal %.9g; finite FP32 values round-trip exactly",
                    "coverage": "one post-prefill full-vocabulary tensor per selected context",
                    "runs": conformance_runs,
                }
                atomic_json(manifest_path, manifest)
                for context in contexts:
                    context_dir = output / "conformance" / f"ctx_{context:07d}"
                    logits_dir = context_dir / "logits"
                    logits_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = context_dir / "prefill.csv"
                    command = benchmark_command(
                        source_dir,
                        context=context,
                        gen_tokens=0,
                        csv_path=csv_path,
                        logits_dir=logits_dir,
                    )
                    result = run_logged(
                        command,
                        cwd=source_dir,
                        log_path=logs / f"conformance_ctx_{context:07d}.log",
                        display_command=sanitized_benchmark_command(command, source_dir, output),
                    )
                    logits_path = logits_dir / f"frontier_{context:06d}.logits.json"
                    metadata = validate_logits(logits_path, context)
                    row = read_single_csv_row(csv_path, context)
                    conformance_runs.append({
                        "context": context,
                        "command": result,
                        "metadata": metadata,
                        "prefill_row": row,
                        "logits": artifact(logits_path, output),
                        "csv": artifact(csv_path, output),
                    })
                    atomic_json(manifest_path, manifest)
                quant_bits = {run["metadata"].get("quant_bits") for run in conformance_runs}
                if quant_bits != {2}:
                    observed = sorted(str(value) for value in quant_bits)
                    raise CaptureError(f"expected Q2 routed quantization, observed quant_bits={observed}")
                manifest["conformance"] = {
                    "status": "passed",
                    "encoding": "JSON decimal %.9g; finite FP32 values round-trip exactly",
                    "coverage": "one post-prefill full-vocabulary tensor per selected context",
                    "runs": conformance_runs,
                }
                atomic_json(manifest_path, manifest)

            performance_rows: list[dict[str, Any]] = []
            if args.skip_performance:
                manifest["performance"] = {"status": "skipped", "runs": []}
            else:
                performance_dir = output / "performance"
                performance_dir.mkdir(parents=True, exist_ok=True)
                warmup_context = min(contexts)
                warmup_csv = performance_dir / "warmup.csv"
                warmup_command = benchmark_command(
                    source_dir,
                    context=warmup_context,
                    gen_tokens=min(8, args.gen_tokens),
                    csv_path=warmup_csv,
                )
                warmup = run_logged(
                    warmup_command,
                    cwd=source_dir,
                    log_path=logs / "performance_warmup.log",
                    display_command=sanitized_benchmark_command(warmup_command, source_dir, output),
                )
                warmup_record = {
                    "command": warmup,
                    "csv": artifact(warmup_csv, output),
                    "row": read_single_csv_row(warmup_csv, warmup_context),
                }

                run_records: list[dict[str, Any]] = []
                manifest["performance"] = {
                    "status": "running",
                    "warmup": warmup_record,
                    "runs": run_records,
                }
                atomic_json(manifest_path, manifest)
                for repetition in range(1, args.repetitions + 1):
                    ordered_contexts = contexts if repetition % 2 else tuple(reversed(contexts))
                    for context in ordered_contexts:
                        csv_path = performance_dir / f"ctx_{context:07d}_run_{repetition:02d}.csv"
                        command = benchmark_command(
                            source_dir,
                            context=context,
                            gen_tokens=args.gen_tokens,
                            csv_path=csv_path,
                        )
                        result = run_logged(
                            command,
                            cwd=source_dir,
                            log_path=logs / f"performance_ctx_{context:07d}_run_{repetition:02d}.log",
                            display_command=sanitized_benchmark_command(command, source_dir, output),
                        )
                        row = read_single_csv_row(csv_path, context)
                        row["repetition"] = repetition
                        performance_rows.append(row)
                        run_records.append(
                            {
                                "context": context,
                                "repetition": repetition,
                                "command": result,
                                "csv": artifact(csv_path, output),
                                "row": row,
                            }
                        )
                        atomic_json(manifest_path, manifest)
                aggregate = aggregate_performance(performance_rows, output)
                manifest["performance"] = {
                    "status": "passed",
                    "warmup": warmup_record,
                    "runs": run_records,
                    "summary": aggregate,
                }

        manifest["host_after"] = {
            "captured_at_utc": utc_now(),
            "pmset_thermal": quiet_output(["pmset", "-g", "therm"]),
            "uptime": quiet_output(["uptime"]),
        }
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        archive, digest = make_archive(output)
        print("\nOracle capture complete.")
        print(f"Results: {output}")
        print(f"Archive: {archive}")
        print(f"SHA-256: {digest}")
        return 0
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["completed_at_utc"] = utc_now()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            print(f"\nCapture interrupted. Partial results: {output}", file=sys.stderr)
            return 130
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        contexts = validate_args(args)
        if args.dry_run:
            return dry_run_plan(args, contexts)
        return capture(args, contexts)
    except CaptureError as exc:
        print(f"capture_oracle_v1.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
