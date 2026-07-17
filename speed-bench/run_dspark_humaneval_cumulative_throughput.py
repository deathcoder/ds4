#!/usr/bin/env python3
"""Reassess cumulative exact DSpark throughput on frozen HumanEval."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics

import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_threshold075_throughput as threshold075
import run_dspark_humaneval_throughput as throughput
import run_dspark_issue468_comparison as common


THRESHOLD = "0.75"
SAMPLE_COUNT = 32
HISTORICAL_SOURCE_COMMIT = "7b954be938db1cd7daf7a37237bc13eb553d27d6"
MIN_MOVEMENT_GEOMEAN = 1.05
MIN_IMPROVED_TASKS = 24
MIN_TASK_MOVEMENT = 0.90
NEAR_PARITY_GEOMEAN = 0.95
PARITY_GEOMEAN = 1.00


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare ordinary baseline with the accumulated exact DSpark "
            "runtime on the frozen threshold-0.75 HumanEval workload."
        )
    )
    parser.add_argument("--historical-reference", type=Path, required=True)
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
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-idle", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_idle:
        parser.error("refusing to benchmark without --confirm-idle")
    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = False
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = THRESHOLD
    args.pairs = 1
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
        raise SystemExit(f"historical reference {key} path mismatch")


def load_historical_reference(args, records, selection):
    summary_path = args.historical_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "throughput.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("throughput CSV", csv_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing historical {label}: {path}")

    summary = load_json(summary_path, "historical summary")
    metadata = load_json(metadata_path, "historical metadata")
    if metadata.get("experiment") != "dspark_humaneval_threshold075_throughput":
        raise SystemExit("historical reference has the wrong experiment kind")
    if metadata.get("git_commit") != HISTORICAL_SOURCE_COMMIT:
        raise SystemExit("historical reference source commit mismatch")
    if metadata.get("git_status_tracked"):
        raise SystemExit("historical reference was produced from a dirty tree")
    if summary.get("sample_count") != SAMPLE_COUNT:
        raise SystemExit("historical reference is not the frozen 32-task run")
    if summary.get("selection") != selection:
        raise SystemExit("historical reference selection mismatch")
    if summary.get("threshold") != THRESHOLD:
        raise SystemExit("historical reference threshold mismatch")
    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "threshold": THRESHOLD,
        "instrumented": False,
        "measured_pairs_per_task": 1,
        "alternating_order": True,
        "global_warmup_pairs": 2,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"historical config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        validate_metadata_path(metadata, key, expected)

    try:
        csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read historical CSV: {exc}") from exc
    rows_by_task = {}
    for row in csv_rows:
        rows_by_task.setdefault(row.get("prompt"), {})[row.get("mode")] = row
    if len(csv_rows) != SAMPLE_COUNT * 2:
        raise SystemExit("historical reference measured-row count mismatch")

    tasks = {}
    for record in records:
        task = record["label"]
        by_mode = rows_by_task.get(task, {})
        prior = summary.get("samples", {}).get(task)
        if set(by_mode) != {"baseline", "runtime"} or prior is None:
            raise SystemExit(f"historical reference has incomplete task {task}")
        if by_mode["baseline"]["stdout_sha256"] != by_mode["runtime"]["stdout_sha256"]:
            raise SystemExit(f"historical output mismatch for {task}")
        output_path = run_dir / by_mode["baseline"]["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing historical output for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != by_mode["baseline"]["stdout_sha256"]:
            raise SystemExit(f"historical output hash mismatch for {task}")
        baseline = float(by_mode["baseline"]["generation_tps"])
        runtime = float(by_mode["runtime"]["generation_tps"])
        ratio = runtime / baseline
        if abs(ratio - prior.get("paired_ratio", -1)) > 1e-12:
            raise SystemExit(f"historical ratio mismatch for {task}")
        tasks[task] = {
            "output_data": output_data,
            "acceptance_verify_rate": prior["acceptance_verify_rate"],
            "baseline_generation_tps": baseline,
            "runtime_generation_tps": runtime,
            "paired_ratio": ratio,
        }
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "summary": summary,
        "tasks": tasks,
    }


def runtime_env():
    return common.benchmark_env(
        "runtime", False, confidence_threshold=THRESHOLD
    )


def summarize(rows, records, historical):
    samples = {}
    ratios = []
    movements = []
    baseline_values = []
    runtime_values = []
    improved = 0
    equal = 0
    for record in records:
        task = record["label"]
        selected = {row["mode"]: row for row in rows if row["prompt"] == task}
        if set(selected) != {"baseline", "runtime"}:
            raise RuntimeError(f"incomplete cumulative pair for {task}")
        baseline = selected["baseline"]["generation_tps"]
        runtime = selected["runtime"]["generation_tps"]
        ratio = runtime / baseline
        prior = historical["tasks"][task]
        movement = ratio / prior["paired_ratio"]
        ratios.append(ratio)
        movements.append(movement)
        baseline_values.append(baseline)
        runtime_values.append(runtime)
        improved += movement > 1.0
        equal += movement == 1.0
        samples[task] = {
            "source_index": record["source_index"],
            "acceptance_verify_rate": prior["acceptance_verify_rate"],
            "order": selected["baseline"]["pair_order"],
            "baseline_generation_tps": baseline,
            "runtime_generation_tps": runtime,
            "paired_ratio": ratio,
            "delta_percent": (ratio - 1.0) * 100.0,
            "historical_paired_ratio": prior["paired_ratio"],
            "paired_ratio_movement": movement,
        }
    quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
    geomean = statistics.geometric_mean(ratios)
    movement_geomean = statistics.geometric_mean(movements)
    faster = sum(ratio > 1.0 for ratio in ratios)
    movement_pass = (
        movement_geomean >= MIN_MOVEMENT_GEOMEAN and
        improved >= MIN_IMPROVED_TASKS and
        min(movements) >= MIN_TASK_MOVEMENT
    )
    if geomean >= PARITY_GEOMEAN:
        outcome = "parity_or_speedup"
    elif geomean >= NEAR_PARITY_GEOMEAN:
        outcome = "near_parity"
    else:
        outcome = "below_near_parity"
    return {
        "sample_count": len(records),
        "threshold": THRESHOLD,
        "samples": samples,
        "baseline_generation_tps_median": statistics.median(baseline_values),
        "runtime_generation_tps_median": statistics.median(runtime_values),
        "ratio_of_medians": (
            statistics.median(runtime_values) /
            statistics.median(baseline_values)
        ),
        "paired_ratio_median": statistics.median(ratios),
        "paired_ratio_geometric_mean": geomean,
        "paired_ratio_q1": quartiles[0],
        "paired_ratio_q3": quartiles[2],
        "paired_ratio_minimum": min(ratios),
        "paired_ratio_maximum": max(ratios),
        "runtime_faster_tasks": faster,
        "runtime_equal_tasks": sum(ratio == 1.0 for ratio in ratios),
        "runtime_slower_tasks": sum(ratio < 1.0 for ratio in ratios),
        "historical_paired_ratio_geometric_mean":
            historical["summary"]["paired_ratio_geometric_mean"],
        "paired_ratio_movement_median": statistics.median(movements),
        "paired_ratio_movement_geometric_mean": movement_geomean,
        "paired_ratio_improved_tasks": improved,
        "paired_ratio_equal_tasks": equal,
        "paired_ratio_regressed_tasks": len(records) - improved - equal,
        "paired_ratio_movement_minimum": min(movements),
        "movement_gate": {
            "minimum_geometric_mean": MIN_MOVEMENT_GEOMEAN,
            "minimum_improved_tasks": MIN_IMPROVED_TASKS,
            "minimum_task_movement": MIN_TASK_MOVEMENT,
            "pass": movement_pass,
        },
        "parity_gate": {
            "near_parity_geometric_mean": NEAR_PARITY_GEOMEAN,
            "parity_geometric_mean": PARITY_GEOMEAN,
            "outcome": outcome,
            "gap_to_parity_percent": max(0.0, (1.0 - geomean) * 100.0),
        },
    }


def render_report(summary):
    movement = summary["movement_gate"]
    parity = summary["parity_gate"]
    lines = [
        "# DSpark HumanEval Cumulative Throughput Reassessment",
        "",
        "All samples are uninstrumented and paired within the same frozen "
        "threshold-0.75 HumanEval task.",
        "Every current DSpark output matched ordinary baseline and the frozen "
        "historical artifact byte-for-byte.",
        "Generation t/s excludes process startup; current paired ratios are "
        "authoritative.",
        "",
        "| samples | baseline median | current DSpark median | ratio of medians "
        "| median paired | geometric mean | faster tasks |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | "
        f"{summary['baseline_generation_tps_median']:.2f} t/s | "
        f"{summary['runtime_generation_tps_median']:.2f} t/s | "
        f"{summary['ratio_of_medians']:.4f}x | "
        f"{summary['paired_ratio_median']:.4f}x | "
        f"{summary['paired_ratio_geometric_mean']:.4f}x | "
        f"{summary['runtime_faster_tasks']}/{summary['sample_count']} |",
        "",
        f"- Paired-ratio interquartile range: "
        f"{summary['paired_ratio_q1']:.4f}x-"
        f"{summary['paired_ratio_q3']:.4f}x.",
        f"- Paired-ratio range: {summary['paired_ratio_minimum']:.4f}x-"
        f"{summary['paired_ratio_maximum']:.4f}x.",
        f"- Tasks faster/equal/slower than baseline: "
        f"{summary['runtime_faster_tasks']}/"
        f"{summary['runtime_equal_tasks']}/"
        f"{summary['runtime_slower_tasks']}.",
        "",
        "## Historical Movement",
        "",
        "The historical ratios come from the frozen clean commit `7b954be`; "
        "movement is descriptive cross-run context.",
        f"- Historical geometric paired ratio: "
        f"{summary['historical_paired_ratio_geometric_mean']:.4f}x.",
        f"- Current/historical median task movement: "
        f"{summary['paired_ratio_movement_median']:.4f}x.",
        f"- Current/historical geometric task movement: "
        f"{summary['paired_ratio_movement_geometric_mean']:.4f}x.",
        f"- Tasks improved/equal/regressed: "
        f"{summary['paired_ratio_improved_tasks']}/"
        f"{summary['paired_ratio_equal_tasks']}/"
        f"{summary['paired_ratio_regressed_tasks']}.",
        f"- Worst task movement: "
        f"{summary['paired_ratio_movement_minimum']:.4f}x.",
        "",
        "## Tasks",
        "",
        "| task | acceptance | order | baseline | current DSpark | ratio | "
        "historical | movement |",
        "|:---|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for task, item in summary["samples"].items():
        lines.append(
            f"| {task} | {item['acceptance_verify_rate']:.3f} | "
            f"{item['order']} | "
            f"{item['baseline_generation_tps']:.2f} t/s | "
            f"{item['runtime_generation_tps']:.2f} t/s | "
            f"{item['paired_ratio']:.4f}x | "
            f"{item['historical_paired_ratio']:.4f}x | "
            f"{item['paired_ratio_movement']:.4f}x |"
        )
    lines.extend([
        "",
        "## Decision Gates",
        "",
        f"- Accumulated movement: {'PASS' if movement['pass'] else 'FAIL'}.",
        f"- Require movement geometric mean at least "
        f"`{movement['minimum_geometric_mean']:.2f}x`, at least "
        f"`{movement['minimum_improved_tasks']}/32` improved tasks, and no task "
        f"below `{movement['minimum_task_movement']:.2f}x` movement.",
        f"- End-to-end outcome: `{parity['outcome']}`.",
        f"- Near parity begins at `{parity['near_parity_geometric_mean']:.2f}x`; "
        f"parity begins at `{parity['parity_geometric_mean']:.2f}x`.",
        f"- Geometric gap to parity: {parity['gap_to_parity_percent']:.1f}%.",
        "",
        "- Two global warmup pairs are excluded from every reported value.",
        "- Measured task order alternates baseline-first and runtime-first.",
        "- Runtime pins confidence threshold 0.75 and otherwise uses promoted "
        "defaults.",
        "- No DSpark stats, acceptance audit, trace, diagnostics, profiler, or "
        "fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args, root = parse_args()
    for name in (
        "historical_reference", "binary", "model", "dspark_model",
        "corpus_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("corpus", args.corpus_dir),
    ):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")
    dirty = common.git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n"
            + dirty
        )

    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, SAMPLE_COUNT, provenance["selection_policy"]
    )
    historical = load_historical_reference(args, records, selection)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-cumulative-throughput-"
        f"{SAMPLE_COUNT}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for position, record in enumerate(records, start=1):
        order = throughput.measured_order(position)
        prompt = prompts[record["label"]]
        print(
            f"{record['label']} measured order: {' -> '.join(order)}\n"
            f"  baseline: {common.command_text(args, prompt, 'baseline')}\n"
            f"  current DSpark: "
            f"{common.command_text(args, prompt, 'runtime', confidence_threshold=THRESHOLD)}"
        )
    warmups = throughput.warmup_schedule(records)
    total = len(warmups) * 2 + len(records) * 2
    print(
        f"Cumulative throughput reassessment: {total} uninstrumented processes; "
        f"{len(warmups) * 2} excluded warmups and {len(records) * 2} measured."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts(prompts, records)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": dirty,
        "experiment": "dspark_humaneval_cumulative_throughput",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "selection": selection,
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "threshold": THRESHOLD,
            "instrumented": False,
            "measured_pairs_per_task": 1,
            "alternating_order": True,
            "global_warmup_pairs": len(warmups),
            "promoted_defaults": True,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "historical_reference": {
            "summary": common.file_metadata(historical["summary_path"]),
            "metadata": common.file_metadata(historical["metadata_path"]),
            "csv": common.file_metadata(historical["csv_path"]),
        },
        "commands": {
            record["label"]: {
                "baseline": common.command_text(
                    args, prompts[record["label"]], "baseline"
                ),
                "runtime": common.command_text(
                    args, prompts[record["label"]], "runtime",
                    confidence_threshold=THRESHOLD,
                ),
            }
            for record in records
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    warmup_rows = []
    for number, (record, order) in enumerate(warmups, start=1):
        warmup_rows.extend(threshold075.run_pair(
            args, root, run_dir, f"warmup-{number:02d}", record,
            prompts[record["label"]], order,
            historical["tasks"][record["label"]]["output_data"],
        ))
    measured_rows = []
    sequence = 0
    for position, record in enumerate(records, start=1):
        pair_rows = threshold075.run_pair(
            args, root, run_dir, f"measured-{position:02d}", record,
            prompts[record["label"]], throughput.measured_order(position),
            historical["tasks"][record["label"]]["output_data"],
        )
        for row in pair_rows:
            sequence += 1
            row["sequence"] = sequence
            measured_rows.append(row)

    fields = (
        "sequence", "prompt", "source_index", "pair_order", "pair_position",
        "mode", "prefill_tps", "generation_tps", "wall_seconds",
        "stdout_sha256", "stdout_file", "stderr_file",
    )
    common.write_csv(run_dir / "throughput.csv", measured_rows, fields)
    common.write_csv(
        run_dir / "warmups.csv", warmup_rows,
        tuple(field for field in fields if field != "sequence"),
    )
    summary = summarize(measured_rows, records, historical)
    summary["selection"] = selection
    summary["historical_reference"] = str(historical["summary_path"])
    report = render_report(summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw throughput: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
