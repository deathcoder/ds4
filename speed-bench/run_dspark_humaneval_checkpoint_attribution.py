#!/usr/bin/env python3
"""Attribute exact prefix-checkpoint behavior on frozen HumanEval tasks."""

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
    ("best_current_ratio", "humaneval_047"),
    ("large_checkpoint_gain", "humaneval_131"),
    ("large_gain_low_acceptance", "humaneval_137"),
)
TASKS = tuple(task for _, task in TASK_ROLES)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect stats-only exact prefix-checkpoint attribution on four "
            "frozen HumanEval tasks."
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
    parser.add_argument("--throughput-reference", type=Path, required=True)
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
        parser.error("refusing to run attribution without --confirm-ready")

    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = True
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD
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


def load_throughput_reference(args, records):
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
    if metadata.get("experiment") != (
        "deepspec_humaneval_confidence_scheduler_throughput"
    ):
        raise SystemExit("throughput reference has the wrong experiment kind")
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "fast_verifier": False,
        "execution_mode": "throughput",
        "throughput_instrumentation": False,
        "runtime_instrumentation": False,
        "acceptance_audit": False,
        "confidence_threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        "nothink": True,
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
    if summary.get("sample_count") != 32:
        raise SystemExit("throughput reference is not the frozen 32-task study")
    if summary.get("confidence_scheduler") is not True:
        raise SystemExit("throughput reference did not use confidence scheduling")
    if summary.get("confidence_threshold") != (
        common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD
    ):
        raise SystemExit("throughput reference threshold mismatch")

    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read throughput reference CSV: {exc}") from exc
    record_map = {record["label"]: record for record in records}
    tasks = {}
    for role, task in TASK_ROLES:
        if task not in record_map or task not in summary.get("samples", {}):
            raise SystemExit(f"frozen attribution task is missing: {task}")
        task_rows = [row for row in rows if row.get("prompt") == task]
        by_mode = {row.get("mode"): row for row in task_rows}
        if set(by_mode) != {"baseline", "runtime"} or len(task_rows) != 2:
            raise SystemExit(f"throughput reference has incomplete pair for {task}")
        if by_mode["baseline"]["stdout_sha256"] != (
            by_mode["runtime"]["stdout_sha256"]
        ):
            raise SystemExit(f"throughput reference output mismatch for {task}")
        output_path = run_dir / by_mode["runtime"]["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing throughput runtime output for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != by_mode["runtime"]["stdout_sha256"]:
            raise SystemExit(f"throughput output hash mismatch for {task}")
        record = record_map[task]
        prompt_data = record["turns"][0].encode("utf-8")
        prompt_path = Path(metadata.get("prompts", {}).get(task, {}).get("path", ""))
        if not prompt_path.is_file() or prompt_path.read_bytes() != prompt_data:
            raise SystemExit(f"throughput prompt drift for {task}")
        tasks[task] = {
            "role": role,
            "record": record,
            "prior": summary["samples"][task],
            "output_path": output_path,
            "output_data": output_data,
            "prompt_data": prompt_data,
        }
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "tasks": tasks,
    }


def task_metrics(row, context):
    attempts = row["prefix_checkpoint_attempts"]
    successes = row["prefix_checkpoint_successes"]
    fallbacks = row["prefix_checkpoint_fallbacks"]
    rows_avoided = row["prefix_checkpoint_rows_avoided"]
    partial = row["batch_partial"]
    if (
        successes > attempts or attempts > partial or
        fallbacks != attempts - successes
    ):
        raise RuntimeError("invalid prefix-checkpoint attempt accounting")
    if (
        (successes == 0 and rows_avoided != 0) or
        rows_avoided < successes
    ):
        raise RuntimeError("prefix-checkpoint rows avoided without a success")
    emitted = row["emitted"]
    return {
        "role": context["role"],
        "source_index": context["record"]["source_index"],
        "acceptance_verify_rate": context["prior"]["acceptance_verify_rate"],
        "prior_paired_ratio": context["prior"]["paired_ratio"],
        "historical_ratio_movement": context["prior"].get(
            "paired_ratio_vs_historical"
        ),
        "emitted": emitted,
        "proposal_rounds": row["multi_attempts"],
        "average_progress": row["avg_depth"],
        "batch_full": row["batch_full"],
        "batch_partial": partial,
        "batch_fallbacks": row["batch_fallbacks"],
        "checkpoint_attempts": attempts,
        "checkpoint_successes": successes,
        "checkpoint_fallbacks": fallbacks,
        "checkpoint_success_rate": successes / attempts if attempts else None,
        "checkpoint_partial_coverage": successes / partial if partial else None,
        "replay_rows_avoided": rows_avoided,
        "replay_rows_avoided_per_success":
            rows_avoided / successes if successes else None,
        "replay_rows_avoided_per_emitted": rows_avoided / emitted,
        "target_evals": row["target_evals"],
        "target_positions": row["target_eval_tokens"],
        "target_evals_per_emitted": row["target_evals"] / emitted,
        "target_positions_per_emitted": row["target_eval_tokens"] / emitted,
        "legacy_target_positions_proxy": (
            row["target_eval_tokens"] + rows_avoided
        ),
        "legacy_target_positions_per_emitted_proxy": (
            row["target_eval_tokens"] + rows_avoided
        ) / emitted,
        "target_ms_per_emitted": row["target_eval_ms"] / emitted,
        "sidecar_ms_per_emitted": row["generation_sidecar_ms"] / emitted,
    }


def summarize(rows, reference):
    return {
        "analysis": "dspark_exact_prefix_checkpoint_attribution",
        "threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        "tasks": {
            row["prompt"]: task_metrics(
                row, reference["tasks"][row["prompt"]]
            )
            for row in rows
        },
    }


def format_percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def render_report(summary):
    lines = [
        "# DSpark Exact Prefix-Checkpoint Attribution",
        "",
        "Instrumented stats-only diagnostic. Throughput values are intentionally omitted.",
        "Every runtime output matched the completed uninstrumented HumanEval artifact byte-for-byte.",
        "",
        "| task | role | acceptance | prior ratio | partial batches | "
        "checkpoint success | fallback | replay rows avoided | "
        "rows avoided/emitted |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary["tasks"][task]
        success_rate = format_percent(item["checkpoint_success_rate"])
        lines.append(
            f"| {task} | {item['role']} | "
            f"{item['acceptance_verify_rate']:.3f} | "
            f"{item['prior_paired_ratio']:.4f}x | "
            f"{item['batch_partial']} | "
            f"{item['checkpoint_successes']}/{item['checkpoint_attempts']} "
            f"({success_rate}) | "
            f"{item['checkpoint_fallbacks']} | "
            f"{item['replay_rows_avoided']} | "
            f"{item['replay_rows_avoided_per_emitted']:.3f} |"
        )
    lines.extend([
        "",
        "## Target Position Effect",
        "",
        "| task | target evals/emitted | measured positions/emitted | "
        "legacy positions/emitted proxy | position reduction | "
        "target ms/emitted | sidecar ms/emitted |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for task in TASKS:
        item = summary["tasks"][task]
        measured = item["target_positions_per_emitted"]
        legacy = item["legacy_target_positions_per_emitted_proxy"]
        reduction = 1.0 - measured / legacy if legacy else None
        lines.append(
            f"| {task} | {item['target_evals_per_emitted']:.4f} | "
            f"{measured:.4f} | {legacy:.4f} | "
            f"{format_percent(reduction)} | "
            f"{item['target_ms_per_emitted']:.3f} | "
            f"{item['sidecar_ms_per_emitted']:.3f} |"
        )
    lines.extend([
        "",
        "- Replay rows avoided counts exact target positions the legacy partial-accept path would have reevaluated.",
        "- The legacy position value is a structural proxy, not a timing prediction; Metal cost is not linear in positions.",
        "- Checkpoint counters are collected only under the existing runtime-stats gate.",
        "- No fresh baseline, throughput pass, acceptance audit, trace, or layer profiler is run.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    fields = (
        "prompt", "mode", "wall_seconds", "stdout_sha256", "stdout_file",
        "stderr_file",
    ) + common.STATS_FIELDS
    common.write_csv(path, rows, fields)


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    records, provenance = corpus.load_corpus(args, root)
    reference = load_throughput_reference(args, records)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-checkpoint-attribution-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = {
        task: run_dir / "prompts" / f"{task}.txt" for task in TASKS
    }
    for task in TASKS:
        print(
            f"{task} checkpoint stats runtime: "
            f"{common.command_text(args, prompts[task], 'runtime', stats=True)}"
        )
    print(
        "HumanEval checkpoint attribution: four stats-enabled exact-runtime "
        "processes; outputs reuse the completed throughput references."
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
        "git_status_tracked": common.git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "experiment": "dspark_humaneval_prefix_checkpoint_attribution",
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
            "confidence_threshold":
                common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
            "prefix_checkpoint_override": None,
            "instrumented": True,
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
                args, prompts[task], "runtime", stats=True
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
            args, root, run_dir, "stats", task, prompts[task],
            "runtime", context["output_data"], stats=True,
            confidence_threshold=common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        )
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
