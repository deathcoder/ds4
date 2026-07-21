#!/usr/bin/env python3
"""Measure exact-verifier frontier bookkeeping on frozen HumanEval tasks."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform

import run_dspark_humaneval_acceptance as corpus
import run_dspark_issue468_comparison as common


TASK_ROLES = (
    ("low_acceptance", "humaneval_152"),
    ("high_acceptance", "humaneval_079"),
)
TASKS = tuple(task for _, task in TASK_ROLES)
THRESHOLD = "0.75"
PROFILE_INT_FIELDS = (
    "frontier_snapshot_calls",
    "frontier_snapshot_successes",
    "frontier_restore_calls",
    "frontier_restore_successes",
    "frontier_prefix_commit_calls",
    "frontier_prefix_commit_successes",
    "target_capture_finish_calls",
    "target_capture_finish_successes",
    "bookkeeping_sync_failures",
)
PROFILE_FLOAT_FIELDS = (
    "frontier_snapshot_ms",
    "frontier_restore_ms",
    "frontier_prefix_commit_ms",
    "target_capture_finish_ms",
)
PROFILE_FIELDS = PROFILE_INT_FIELDS + PROFILE_FLOAT_FIELDS


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect synchronized exact-verifier frontier and target-capture "
            "bookkeeping timings on frozen low/high-acceptance HumanEval tasks."
        )
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / (
            "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-"
            "SExpQ8-OutQ8-chat-v2-imatrix.gguf"
        ),
    )
    parser.add_argument(
        "--dspark-model",
        type=Path,
        default=root / "gguf/ds4flash-dspark.gguf",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "speed-bench/humaneval-acceptance",
    )
    parser.add_argument(
        "--throughput-reference",
        type=Path,
        default=root / (
            "speed-bench/local-runs/"
            "humaneval-cumulative-throughput-32-20260719-223901/summary.json"
        ),
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to profile without --confirm-ready")

    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = True
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = THRESHOLD
    args.pairs = 0
    args.warmups = 0
    return args, root


def load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label} {path}: {exc}") from exc


def validate_metadata_path(metadata, key, expected):
    actual = metadata.get(key, {}).get("path")
    if actual is None or Path(actual).resolve() != expected.resolve():
        raise SystemExit(f"throughput reference {key} path mismatch")


def load_reference(args, records):
    summary_path = args.throughput_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "throughput.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("throughput CSV", csv_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing throughput reference {label}: {path}")

    summary = load_json(summary_path, "throughput summary")
    metadata = load_json(metadata_path, "throughput metadata")
    if metadata.get("experiment") != "dspark_humaneval_cumulative_throughput":
        raise SystemExit("throughput reference has the wrong experiment kind")
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "threshold": THRESHOLD,
        "instrumented": False,
        "promoted_defaults": True,
    }
    config = metadata.get("config", {})
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"throughput reference config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        validate_metadata_path(metadata, key, expected)
    if summary.get("sample_count") != 32 or summary.get("threshold") != THRESHOLD:
        raise SystemExit("throughput reference is not the frozen threshold-0.75 study")

    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read throughput reference CSV: {exc}") from exc
    record_map = {record["label"]: record for record in records}
    tasks = {}
    for role, task in TASK_ROLES:
        if task not in record_map or task not in summary.get("samples", {}):
            raise SystemExit(f"frozen profile task is missing: {task}")
        task_rows = [row for row in rows if row.get("prompt") == task]
        by_mode = {row.get("mode"): row for row in task_rows}
        if set(by_mode) != {"baseline", "runtime"} or len(task_rows) != 2:
            raise SystemExit(f"throughput reference has incomplete pair for {task}")
        if by_mode["baseline"]["stdout_sha256"] != by_mode["runtime"]["stdout_sha256"]:
            raise SystemExit(f"throughput reference output mismatch for {task}")
        output_path = run_dir / by_mode["runtime"]["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing throughput runtime output for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != by_mode["runtime"]["stdout_sha256"]:
            raise SystemExit(f"throughput output hash mismatch for {task}")
        record = record_map[task]
        tasks[task] = {
            "role": role,
            "record": record,
            "prior": summary["samples"][task],
            "output_path": output_path,
            "output_data": output_data,
            "prompt_data": record["turns"][0].encode("utf-8"),
        }
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "tasks": tasks,
    }


def parse_profile_stats(stderr_data, path):
    records = [
        line[len(common.STATS_PREFIX):]
        for line in stderr_data.splitlines()
        if line.startswith(common.STATS_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one DSpark stats record in {path}, found {len(records)}"
        )
    values = {}
    try:
        for item in records[0].decode("ascii").split():
            key, value = item.split("=", 1)
            if key in PROFILE_INT_FIELDS:
                values[key] = int(value)
            elif key in PROFILE_FLOAT_FIELDS:
                values[key] = float(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid frontier profile stats in {path}: {exc}") from exc
    missing = [field for field in PROFILE_FIELDS if field not in values]
    if missing:
        raise RuntimeError(
            f"incomplete frontier profile stats in {path}: {', '.join(missing)}"
        )
    if values["bookkeeping_sync_failures"] != 0:
        raise RuntimeError(f"frontier profile synchronization failed in {path}")
    for stem in (
        "frontier_snapshot",
        "frontier_restore",
        "frontier_prefix_commit",
        "target_capture_finish",
    ):
        if values[f"{stem}_successes"] != values[f"{stem}_calls"]:
            raise RuntimeError(f"{stem} operation failed in {path}")
    return values


def task_metrics(row, context):
    emitted = row["emitted"]
    batch_attempts = row["batch_attempts"]
    partial = row["batch_partial"]
    profile = {field: row[field] for field in PROFILE_FIELDS}
    if profile["frontier_snapshot_calls"] != batch_attempts:
        raise RuntimeError("frontier snapshots do not reconcile with batch attempts")
    if profile["frontier_prefix_commit_calls"] != row["prefix_checkpoint_attempts"]:
        raise RuntimeError("frontier prefix commits do not reconcile with attempts")
    if profile["target_capture_finish_calls"] != batch_attempts:
        raise RuntimeError("target capture finishes do not reconcile with batch attempts")
    frontier_ms = (
        profile["frontier_snapshot_ms"]
        + profile["frontier_restore_ms"]
        + profile["frontier_prefix_commit_ms"]
    )
    return {
        "role": context["role"],
        "acceptance_verify_rate": context["prior"]["acceptance_verify_rate"],
        "prior_paired_ratio": context["prior"]["paired_ratio"],
        "emitted": emitted,
        "batch_attempts": batch_attempts,
        "batch_partial": partial,
        "frontier_ms": frontier_ms,
        "frontier_ms_per_emitted": frontier_ms / emitted,
        "frontier_ms_per_batch": frontier_ms / batch_attempts,
        "capture_finish_ms": profile["target_capture_finish_ms"],
        "capture_finish_ms_per_emitted":
            profile["target_capture_finish_ms"] / emitted,
        "capture_finish_ms_per_batch":
            profile["target_capture_finish_ms"] / batch_attempts,
        "profile": profile,
    }


def summarize(rows, reference):
    tasks = {
        row["prompt"]: task_metrics(row, reference["tasks"][row["prompt"]])
        for row in rows
    }
    emitted = sum(item["emitted"] for item in tasks.values())
    batches = sum(item["batch_attempts"] for item in tasks.values())
    frontier_ms = sum(item["frontier_ms"] for item in tasks.values())
    capture_ms = sum(item["capture_finish_ms"] for item in tasks.values())
    frontier_per_emitted = frontier_ms / emitted
    return {
        "analysis": "dspark_exact_frontier_bookkeeping_profile",
        "threshold": THRESHOLD,
        "tasks": tasks,
        "aggregate": {
            "emitted": emitted,
            "batch_attempts": batches,
            "frontier_ms": frontier_ms,
            "frontier_ms_per_emitted": frontier_per_emitted,
            "frontier_ms_per_batch": frontier_ms / batches,
            "capture_finish_ms": capture_ms,
            "capture_finish_ms_per_emitted": capture_ms / emitted,
            "capture_finish_ms_per_batch": capture_ms / batches,
            "position_indexed_shadow_state_gate": (
                "PROCEED" if frontier_per_emitted >= 1.0 else "STOP_ROLLBACK_ONLY"
            ),
        },
    }


def render_report(summary):
    lines = [
        "# DSpark Exact Frontier Bookkeeping Profile",
        "",
        "Synchronized diagnostic only. Boundaries alter Metal scheduling; do not use these values as throughput measurements.",
        "Every profiled output matched the frozen cumulative HumanEval artifact byte-for-byte.",
        "Frontier time includes snapshot, restore, and partial-prefix commit. Target-capture finalization is reported separately because it overlaps sidecar preparation conceptually.",
        "",
        "| task | role | acceptance | prior ratio | emitted | batches | partial | frontier ms/emitted | capture finish ms/emitted |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary["tasks"][task]
        lines.append(
            f"| {task} | {item['role']} | "
            f"{item['acceptance_verify_rate']:.3f} | "
            f"{item['prior_paired_ratio']:.4f}x | "
            f"{item['emitted']} | {item['batch_attempts']} | "
            f"{item['batch_partial']} | "
            f"{item['frontier_ms_per_emitted']:.3f} | "
            f"{item['capture_finish_ms_per_emitted']:.3f} |"
        )
    aggregate = summary["aggregate"]
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Frontier bookkeeping: {aggregate['frontier_ms']:.3f} ms total, "
        f"{aggregate['frontier_ms_per_emitted']:.3f} ms/emitted, "
        f"{aggregate['frontier_ms_per_batch']:.3f} ms/batch.",
        f"- Target-capture finalization: {aggregate['capture_finish_ms']:.3f} ms total, "
        f"{aggregate['capture_finish_ms_per_emitted']:.3f} ms/emitted, "
        f"{aggregate['capture_finish_ms_per_batch']:.3f} ms/batch.",
        f"- Position-indexed shadow-state gate: **{aggregate['position_indexed_shadow_state_gate']}**.",
        "",
        "## Operation Detail",
        "",
        "| task | snapshot calls/ms | restore calls/ms | prefix commit calls/ms | capture finish calls/ms |",
        "|:---|---:|---:|---:|---:|",
    ])
    for task in TASKS:
        p = summary["tasks"][task]["profile"]
        lines.append(
            f"| {task} | {p['frontier_snapshot_calls']}/{p['frontier_snapshot_ms']:.3f} | "
            f"{p['frontier_restore_calls']}/{p['frontier_restore_ms']:.3f} | "
            f"{p['frontier_prefix_commit_calls']}/{p['frontier_prefix_commit_ms']:.3f} | "
            f"{p['target_capture_finish_calls']}/{p['target_capture_finish_ms']:.3f} |"
        )
    lines.extend([
        "",
        "- The proceed gate requires pooled frontier bookkeeping of at least 1.000 ms/emitted.",
        "- This is an upper bound: explicit synchronization prevents overlap that production execution may retain.",
        "- Capture-finalization time is not added to frontier time and is not assumed removable by logical rollback.",
        "- No fresh baseline, throughput pass, acceptance audit, oracle trace, or layer profiler is run.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    fields = (
        "prompt", "mode", "wall_seconds", "stdout_sha256", "stdout_file",
        "stderr_file",
    ) + common.STATS_FIELDS + PROFILE_FIELDS
    common.write_csv(path, rows, fields)


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir", "throughput_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = common.git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" + dirty
        )

    records, provenance = corpus.load_corpus(args, root)
    reference = load_reference(args, records)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/humaneval-frontier-profile-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = {task: run_dir / "prompts" / f"{task}.txt" for task in TASKS}
    for task in TASKS:
        print(
            f"{task} frontier profile: "
            f"{common.command_text(args, prompts[task], 'runtime', stats=True, confidence_threshold=THRESHOLD, frontier_profile=True)}"
        )
    print(
        "Frontier profile: two synchronized stats-only exact-runtime processes; "
        "outputs must match frozen cumulative references."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompts").mkdir()
    for task in TASKS:
        prompts[task].write_bytes(reference["tasks"][task]["prompt_data"])
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": dirty,
        "experiment": "dspark_humaneval_frontier_bookkeeping_profile",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("DS4_")
        },
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "tasks": TASK_ROLES,
            "confidence_threshold": THRESHOLD,
            "frontier_profile": True,
            "synchronized": True,
            "runtime_processes": len(TASKS),
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "provenance_source_commit": provenance.get("source_commit"),
        "throughput_reference": {
            "summary": common.file_metadata(reference["summary_path"]),
            "metadata": common.file_metadata(reference["metadata_path"]),
            "csv": common.file_metadata(reference["csv_path"]),
        },
        "commands": {
            task: common.command_text(
                args, prompts[task], "runtime", stats=True,
                confidence_threshold=THRESHOLD, frontier_profile=True,
            )
            for task in TASKS
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for task in TASKS:
        context = reference["tasks"][task]
        row, _ = common.execute(
            args, root, run_dir, "profile", task, prompts[task],
            "runtime", context["output_data"], stats=True,
            confidence_threshold=THRESHOLD, frontier_profile=True,
        )
        stderr_path = run_dir / row["stderr_file"]
        row.update(parse_profile_stats(stderr_path.read_bytes(), stderr_path))
        rows.append(row)
        common.cooldown(args.cooldown)

    summary = summarize(rows, reference)
    report = render_report(summary)
    write_csv(run_dir / "stats.csv", rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw stats: {run_dir / 'stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
