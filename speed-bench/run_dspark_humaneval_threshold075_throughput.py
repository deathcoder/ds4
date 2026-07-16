#!/usr/bin/env python3
"""Confirm DSpark threshold 0.75 on the frozen 32-task HumanEval workload."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics

import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_aggressive_scheduler_gate as aggressive
import run_dspark_humaneval_oracle_audit as oracle
import run_dspark_humaneval_throughput as throughput
import run_dspark_issue468_comparison as common


THRESHOLD = "0.75"
SAMPLE_COUNT = 32
MIN_MOVEMENT_GEOMEAN = 1.05
MIN_IMPROVED_TASKS = 24
MIN_TASK_MOVEMENT = 0.80
NEAR_PARITY_GEOMEAN = 0.95
NEAR_PARITY_FASTER_TASKS = 8


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 32-task baseline-versus-threshold-0.75 "
            "HumanEval confirmation."
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
    parser.add_argument("--gate-reference", type=Path, required=True)
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
        raise SystemExit(f"aggressive gate {key} path mismatch")


def load_gate_reference(args):
    summary_path = args.gate_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "throughput.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("throughput CSV", csv_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing aggressive gate {label}: {path}")
    summary = load_json(summary_path, "aggressive gate summary")
    metadata = load_json(metadata_path, "aggressive gate metadata")
    if metadata.get("experiment") != (
        "dspark_humaneval_aggressive_scheduler_gate"
    ):
        raise SystemExit("aggressive gate has the wrong experiment kind")
    if summary.get("promotion_gate", {}).get("selected_candidate") != (
        "threshold_075"
    ):
        raise SystemExit("aggressive gate did not select threshold 0.75")
    if summary.get("thresholds_predeclared") != aggressive.THRESHOLDS:
        raise SystemExit("aggressive gate threshold contract mismatch")
    if tuple(summary.get("tasks_predeclared", ())) != aggressive.TASKS:
        raise SystemExit("aggressive gate task contract mismatch")
    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "instrumented": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"aggressive gate config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    if config.get("thresholds") != aggressive.THRESHOLDS:
        raise SystemExit("aggressive gate metadata threshold mismatch")
    for key, expected in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        # Early gate metadata predates shared file metadata fields; the frozen
        # commands still pin the paths when those fields are absent.
        if key in metadata:
            validate_metadata_path(metadata, key, expected)
    referenced = metadata.get("throughput_reference", {}).get(
        "summary", {}
    ).get("path")
    if (
        referenced is None or
        Path(referenced).resolve() != args.throughput_reference.resolve()
    ):
        raise SystemExit("aggressive gate throughput reference mismatch")
    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read aggressive gate CSV: {exc}") from exc
    if len(rows) != len(aggressive.TASKS) * len(aggressive.MODES):
        raise SystemExit("aggressive gate measured-row count mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "summary": summary,
        "metadata": metadata,
    }


def summarize(rows, records, reference):
    samples = {}
    ratios = []
    movement_ratios = []
    acceptance_rates = []
    baseline_values = []
    runtime_values = []
    improved = 0
    equal = 0
    for record in records:
        task = record["label"]
        selected = {row["mode"]: row for row in rows if row["prompt"] == task}
        if set(selected) != {"baseline", "runtime"}:
            raise RuntimeError(f"incomplete threshold-0.75 pair for {task}")
        baseline = selected["baseline"]["generation_tps"]
        runtime = selected["runtime"]["generation_tps"]
        ratio = runtime / baseline
        prior = reference["tasks"][task]["prior"]
        movement = ratio / prior["paired_ratio"]
        acceptance = prior["acceptance_verify_rate"]
        ratios.append(ratio)
        movement_ratios.append(movement)
        acceptance_rates.append(acceptance)
        baseline_values.append(baseline)
        runtime_values.append(runtime)
        improved += movement > 1.0
        equal += movement == 1.0
        samples[task] = {
            "source_index": record["source_index"],
            "order": selected["baseline"]["pair_order"],
            "acceptance_verify_rate": acceptance,
            "baseline_generation_tps": baseline,
            "runtime_generation_tps": runtime,
            "paired_ratio": ratio,
            "delta_percent": (ratio - 1.0) * 100.0,
            "historical_threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
            "historical_paired_ratio": prior["paired_ratio"],
            "paired_ratio_vs_historical": movement,
        }
    quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
    movement_geomean = statistics.geometric_mean(movement_ratios)
    faster_tasks = sum(ratio > 1.0 for ratio in ratios)
    confirmation_pass = (
        movement_geomean >= MIN_MOVEMENT_GEOMEAN and
        improved >= MIN_IMPROVED_TASKS and
        min(movement_ratios) >= MIN_TASK_MOVEMENT
    )
    near_parity = (
        statistics.geometric_mean(ratios) >= NEAR_PARITY_GEOMEAN or
        faster_tasks >= NEAR_PARITY_FASTER_TASKS
    )
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
        "paired_ratio_geometric_mean": statistics.geometric_mean(ratios),
        "paired_ratio_arithmetic_mean": statistics.mean(ratios),
        "paired_ratio_q1": quartiles[0],
        "paired_ratio_q3": quartiles[2],
        "paired_ratio_minimum": min(ratios),
        "paired_ratio_maximum": max(ratios),
        "runtime_faster_tasks": faster_tasks,
        "runtime_equal_tasks": sum(ratio == 1.0 for ratio in ratios),
        "runtime_slower_tasks": sum(ratio < 1.0 for ratio in ratios),
        "acceptance_speed_pearson": throughput.pearson_correlation(
            acceptance_rates, ratios
        ),
        "historical_threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        "historical_paired_ratio_median":
            statistics.median(
                reference["tasks"][task]["prior"]["paired_ratio"]
                for task in samples
            ),
        "paired_ratio_movement_median": statistics.median(movement_ratios),
        "paired_ratio_movement_geometric_mean": movement_geomean,
        "paired_ratio_improved_tasks": improved,
        "paired_ratio_equal_tasks": equal,
        "paired_ratio_regressed_tasks": len(records) - improved - equal,
        "paired_ratio_movement_minimum": min(movement_ratios),
        "confirmation_gate": {
            "minimum_movement_geometric_mean": MIN_MOVEMENT_GEOMEAN,
            "minimum_improved_tasks": MIN_IMPROVED_TASKS,
            "minimum_task_movement": MIN_TASK_MOVEMENT,
            "pass": confirmation_pass,
        },
        "next_path_gate": {
            "minimum_baseline_geometric_mean": NEAR_PARITY_GEOMEAN,
            "minimum_faster_tasks": NEAR_PARITY_FASTER_TASKS,
            "near_parity": near_parity,
            "next_path": (
                "audit_threshold_075_acceptance_and_costs"
                if near_parity else
                "freeze_scheduler_and_optimize_exact_verifier"
            ),
        },
    }


def render_report(summary):
    lines = [
        "# DSpark HumanEval Threshold 0.75 Confirmation",
        "",
        "All samples are uninstrumented and paired within the same HumanEval task.",
        "Every threshold-0.75 output matched its ordinary baseline byte-for-byte.",
        "Generation t/s excludes process startup; paired ratios are authoritative.",
        "",
        "| samples | baseline median | threshold 0.75 median | "
        "ratio of medians | median paired | geometric mean | faster tasks |",
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
        "## Historical Threshold 0.455 Movement",
        "",
        "The threshold-0.455 values are from the frozen prior run and are "
        "descriptive cross-run context, not within-run pairs.",
        f"- Prior median paired ratio: "
        f"{summary['historical_paired_ratio_median']:.4f}x.",
        f"- Threshold-0.75/prior median task movement: "
        f"{summary['paired_ratio_movement_median']:.4f}x.",
        f"- Threshold-0.75/prior geometric task movement: "
        f"{summary['paired_ratio_movement_geometric_mean']:.4f}x.",
        f"- Tasks improved/equal/regressed versus prior: "
        f"{summary['paired_ratio_improved_tasks']}/"
        f"{summary['paired_ratio_equal_tasks']}/"
        f"{summary['paired_ratio_regressed_tasks']}.",
        f"- Worst task movement versus prior: "
        f"{summary['paired_ratio_movement_minimum']:.4f}x.",
        "",
        "## Tasks",
        "",
        "| task | acceptance | order | baseline | threshold 0.75 | ratio | "
        "prior 0.455 | movement |",
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
            f"{item['paired_ratio_vs_historical']:.4f}x |"
        )
    confirmation = summary["confirmation_gate"]
    next_path = summary["next_path_gate"]
    lines.extend([
        "",
        "## Decision Gates",
        "",
        f"- Scheduler confirmation: "
        f"{'PASS' if confirmation['pass'] else 'FAIL'}.",
        f"- Require historical movement geometric mean at least "
        f"`{confirmation['minimum_movement_geometric_mean']:.2f}x`, "
        f"at least `{confirmation['minimum_improved_tasks']}/32` improved "
        f"tasks, and no task below "
        f"`{confirmation['minimum_task_movement']:.2f}x` movement.",
        f"- Near-parity gate: "
        f"{'PASS' if next_path['near_parity'] else 'FAIL'}.",
        f"- Near parity requires baseline geometric mean at least "
        f"`{next_path['minimum_baseline_geometric_mean']:.2f}x` or at least "
        f"`{next_path['minimum_faster_tasks']}` faster tasks.",
        f"- Next path: `{next_path['next_path']}`.",
        "",
        "- Two global warmup pairs are excluded from every reported value.",
        "- Measured task order alternates baseline-first and runtime-first.",
        "- No DSpark stats, acceptance audit, trace, diagnostics, profiler, or "
        "fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def run_pair(args, root, run_dir, label, record, prompt, order, expected):
    rows = []
    order_text = "-".join(order)
    for position, mode in enumerate(order, start=1):
        row, _ = common.execute(
            args, root, run_dir, label, record["label"], prompt,
            mode, expected,
            confidence_threshold=THRESHOLD if mode == "runtime" else None,
        )
        row.update(
            source_index=record["source_index"],
            pair_order=order_text,
            pair_position=position,
        )
        rows.append(row)
        common.cooldown(args.cooldown)
    return rows


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference", "gate_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, SAMPLE_COUNT, provenance["selection_policy"]
    )
    reference = oracle.load_throughput_reference(args, records, selection)
    gate_reference = load_gate_reference(args)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-threshold075-throughput-"
        f"{SAMPLE_COUNT}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for position, record in enumerate(records, start=1):
        order = throughput.measured_order(position)
        prompt = prompts[record["label"]]
        baseline_command = common.command_text(args, prompt, "baseline")
        runtime_command = common.command_text(
            args, prompt, "runtime", confidence_threshold=THRESHOLD
        )
        print(
            f"{record['label']} measured order: {' -> '.join(order)}\n"
            f"  baseline: {baseline_command}\n"
            f"  threshold 0.75: {runtime_command}"
        )
    warmups = throughput.warmup_schedule(records)
    total = len(warmups) * 2 + len(records) * 2
    print(
        f"Threshold-0.75 confirmation: {total} uninstrumented processes; "
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
        "git_status_tracked": common.git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "experiment": "dspark_humaneval_threshold075_throughput",
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
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "throughput_reference": {
            "summary": common.file_metadata(reference["summary_path"]),
            "metadata": common.file_metadata(reference["metadata_path"]),
            "csv": common.file_metadata(reference["csv_path"]),
        },
        "aggressive_gate_reference": {
            "summary": common.file_metadata(gate_reference["summary_path"]),
            "metadata": common.file_metadata(gate_reference["metadata_path"]),
            "csv": common.file_metadata(gate_reference["csv_path"]),
        },
        "commands": {
            record["label"]: {
                "baseline": common.command_text(
                    args, prompts[record["label"]], "baseline"
                ),
                "threshold_075": common.command_text(
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
        warmup_rows.extend(run_pair(
            args, root, run_dir, f"warmup-{number:02d}", record,
            prompts[record["label"]], order,
            reference["tasks"][record["label"]]["output_data"],
        ))

    measured_rows = []
    sequence = 0
    for position, record in enumerate(records, start=1):
        pair_rows = run_pair(
            args, root, run_dir, f"measured-{position:02d}", record,
            prompts[record["label"]], throughput.measured_order(position),
            reference["tasks"][record["label"]]["output_data"],
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
    summary = summarize(measured_rows, records, reference)
    summary["selection"] = selection
    summary["throughput_reference"] = str(reference["summary_path"])
    summary["gate_reference"] = str(gate_reference["summary_path"])
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
