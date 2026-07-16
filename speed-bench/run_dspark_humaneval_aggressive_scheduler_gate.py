#!/usr/bin/env python3
"""Compare aggressive DSpark scheduler thresholds on frozen HumanEval tasks."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import time

import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_oracle_audit as oracle
import run_dspark_issue468_comparison as common


TASKS = (
    "humaneval_152",
    "humaneval_032",
    "humaneval_000",
    "humaneval_121",
    "humaneval_131",
    "humaneval_137",
    "humaneval_011",
    "humaneval_079",
)
MODES = ("baseline", "threshold_0455", "threshold_075", "threshold_085")
THRESHOLDS = {
    "baseline": None,
    "threshold_0455": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
    "threshold_075": "0.75",
    "threshold_085": "0.85",
}
MODE_LABELS = {
    "baseline": "Baseline",
    "threshold_0455": "Threshold 0.455",
    "threshold_075": "Threshold 0.75",
    "threshold_085": "Threshold 0.85",
}
CANDIDATES = ("threshold_075", "threshold_085")
MIN_GEOMEAN_VS_CURRENT = 1.03
MIN_TASK_WINS = 6
MIN_TASK_RATIO = 0.90
HIGH_THRESHOLD_SELECTION_MARGIN = 1.01


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run a balanced HumanEval gate for aggressive DSpark confidence "
            "scheduler thresholds."
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
    args.confidence_threshold = None
    args.pairs = 0
    args.warmups = 0
    return args, root


def rotated_modes(offset):
    offset %= len(MODES)
    return MODES[offset:] + MODES[:offset]


def measured_order(task_index):
    return rotated_modes(task_index)


def target_mode(mode):
    return "baseline" if mode == "baseline" else "runtime"


def mode_env(mode):
    return common.benchmark_env(
        target_mode(mode),
        False,
        confidence_threshold=THRESHOLDS[mode],
    )


def command(args, prompt, mode):
    return common.mode_command(args, prompt, target_mode(mode))


def command_text(args, prompt, mode):
    env = mode_env(mode)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    rendered = shlex.join(command(args, prompt, mode))
    return (prefix + " " if prefix else "") + rendered


def execute(args, root, run_dir, name, task, prompt, mode, reference):
    stdout_path = run_dir / f"{name}.{task}.{mode}.stdout"
    stderr_path = run_dir / f"{name}.{task}.{mode}.stderr"
    print(
        f"[{name}/{task}] {MODE_LABELS[mode]}: "
        f"{command_text(args, prompt, mode)}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args, prompt, mode),
            cwd=root,
            env=mode_env(mode),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name}/{task}/{mode} failed with exit {completed.returncode}; "
            f"see {stderr_path}"
        )
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    if stdout_data != reference:
        raise RuntimeError(
            f"{name}/{task}/{mode} output differs from the frozen exact "
            f"artifact; see {stdout_path}"
        )
    forbidden = (
        common.STATS_PREFIX,
        common.ACCEPTANCE_PREFIX,
        common.ACCEPTANCE_TRACE_PREFIX,
        common.ORACLE_TRACE_PREFIX,
        b"ds4: DSpark confidence scheduler ",
    )
    if any(marker in stderr_data for marker in forbidden):
        raise RuntimeError(f"instrumentation leaked into {stderr_path}")
    prefill_tps, generation_tps = common.parse_timing(stderr_data, stderr_path)
    return {
        "task": task,
        "mode": mode,
        "threshold": "" if THRESHOLDS[mode] is None else THRESHOLDS[mode],
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": wall_seconds,
        "stdout_sha256": common.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def candidate_gate(ratios):
    geomean = statistics.geometric_mean(ratios)
    return {
        "geometric_mean_vs_current": geomean,
        "median_vs_current": statistics.median(ratios),
        "minimum_vs_current": min(ratios),
        "maximum_vs_current": max(ratios),
        "task_wins": sum(ratio > 1.0 for ratio in ratios),
        "task_count": len(ratios),
        "pass_geometric_mean": geomean >= MIN_GEOMEAN_VS_CURRENT,
        "pass_task_wins": sum(ratio > 1.0 for ratio in ratios) >= MIN_TASK_WINS,
        "pass_minimum": min(ratios) >= MIN_TASK_RATIO,
    }


def summarize(rows, reference):
    tasks = {}
    candidate_current_ratios = {mode: [] for mode in CANDIDATES}
    candidate_baseline_ratios = {mode: [] for mode in CANDIDATES}
    current_baseline_ratios = []
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        by_mode = {row["mode"]: row for row in selected}
        if set(by_mode) != set(MODES) or len(selected) != len(MODES):
            raise RuntimeError(f"incomplete aggressive scheduler task {task}")
        baseline_tps = by_mode["baseline"]["generation_tps"]
        current_tps = by_mode["threshold_0455"]["generation_tps"]
        item = {
            "acceptance_verify_rate":
                reference["tasks"][task]["prior"]["acceptance_verify_rate"],
            "prior_measured_ratio":
                reference["tasks"][task]["prior"]["paired_ratio"],
            "order": selected[0]["order"],
            "modes": {},
        }
        for mode in MODES:
            tps = by_mode[mode]["generation_tps"]
            mode_item = {"generation_tps": tps}
            if mode != "baseline":
                mode_item["ratio_vs_baseline"] = tps / baseline_tps
            if mode in CANDIDATES:
                mode_item["ratio_vs_current"] = tps / current_tps
                candidate_current_ratios[mode].append(
                    mode_item["ratio_vs_current"]
                )
                candidate_baseline_ratios[mode].append(
                    mode_item["ratio_vs_baseline"]
                )
            item["modes"][mode] = mode_item
        current_baseline_ratios.append(current_tps / baseline_tps)
        tasks[task] = item

    aggregate = {
        "threshold_0455": {
            "median_vs_baseline": statistics.median(current_baseline_ratios),
            "geometric_mean_vs_baseline":
                statistics.geometric_mean(current_baseline_ratios),
            "baseline_wins":
                sum(ratio >= 1.0 for ratio in current_baseline_ratios),
        }
    }
    passing = []
    for mode in CANDIDATES:
        gate = candidate_gate(candidate_current_ratios[mode])
        gate.update(
            median_vs_baseline=statistics.median(
                candidate_baseline_ratios[mode]
            ),
            geometric_mean_vs_baseline=statistics.geometric_mean(
                candidate_baseline_ratios[mode]
            ),
            baseline_wins=sum(
                ratio >= 1.0 for ratio in candidate_baseline_ratios[mode]
            ),
            pass_all=(
                gate["pass_geometric_mean"] and
                gate["pass_task_wins"] and
                gate["pass_minimum"]
            ),
        )
        aggregate[mode] = gate
        if gate["pass_all"]:
            passing.append(mode)

    selected = None
    if len(passing) == 1:
        selected = passing[0]
    elif len(passing) == 2:
        low = aggregate["threshold_075"]["geometric_mean_vs_current"]
        high = aggregate["threshold_085"]["geometric_mean_vs_current"]
        selected = (
            "threshold_085"
            if high >= low * HIGH_THRESHOLD_SELECTION_MARGIN
            else "threshold_075"
        )
    return {
        "tasks": tasks,
        "aggregate": aggregate,
        "promotion_gate": {
            "minimum_geometric_mean_vs_current": MIN_GEOMEAN_VS_CURRENT,
            "minimum_task_wins": MIN_TASK_WINS,
            "minimum_task_ratio": MIN_TASK_RATIO,
            "high_threshold_selection_margin":
                HIGH_THRESHOLD_SELECTION_MARGIN,
            "passing_candidates": passing,
            "selected_candidate": selected,
        },
    }


def pass_text(value):
    return "PASS" if value else "FAIL"


def render_report(summary):
    lines = [
        "# DSpark Aggressive Scheduler Gate",
        "",
        "All runs are uninstrumented and use exact target verification.",
        "Every mode matched the frozen HumanEval output byte-for-byte.",
        "Tasks, thresholds, ordering, and promotion rules were frozen before execution.",
        "",
        "| task | acceptance | baseline | current 0.455 | 0.75 | "
        "0.75/current | 0.85 | 0.85/current |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in summary["tasks"].items():
        modes = item["modes"]
        lines.append(
            f"| {task} | {item['acceptance_verify_rate']:.3f} | "
            f"{modes['baseline']['generation_tps']:.2f} t/s | "
            f"{modes['threshold_0455']['generation_tps']:.2f} t/s | "
            f"{modes['threshold_075']['generation_tps']:.2f} t/s | "
            f"{modes['threshold_075']['ratio_vs_current']:.4f}x | "
            f"{modes['threshold_085']['generation_tps']:.2f} t/s | "
            f"{modes['threshold_085']['ratio_vs_current']:.4f}x |"
        )
    lines.extend([
        "",
        "## Aggregate",
        "",
        "| policy | median vs baseline | geometric vs baseline | "
        "median vs current | geometric vs current | wins vs current | "
        "minimum vs current | gate |",
        "|:---|---:|---:|---:|---:|---:|---:|:---|",
    ])
    current = summary["aggregate"]["threshold_0455"]
    lines.append(
        f"| Threshold 0.455 | {current['median_vs_baseline']:.4f}x | "
        f"{current['geometric_mean_vs_baseline']:.4f}x | reference | "
        f"reference | n/a | n/a | reference |"
    )
    for mode in CANDIDATES:
        item = summary["aggregate"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {item['median_vs_baseline']:.4f}x | "
            f"{item['geometric_mean_vs_baseline']:.4f}x | "
            f"{item['median_vs_current']:.4f}x | "
            f"{item['geometric_mean_vs_current']:.4f}x | "
            f"{item['task_wins']}/{item['task_count']} | "
            f"{item['minimum_vs_current']:.4f}x | "
            f"{pass_text(item['pass_all'])} |"
        )
    gate = summary["promotion_gate"]
    selected = gate["selected_candidate"]
    lines.extend([
        "",
        "## Promotion Gate",
        "",
        f"- Require geometric mean versus current at least "
        f"`{gate['minimum_geometric_mean_vs_current']:.2f}x`.",
        f"- Require wins on at least `{gate['minimum_task_wins']}/"
        f"{len(TASKS)}` tasks.",
        f"- Require no task below `{gate['minimum_task_ratio']:.2f}x` "
        "versus current.",
        "- If both pass, select 0.85 only if its geometric mean is at least "
        f"`{gate['high_threshold_selection_margin']:.2f}x` the 0.75 result.",
        f"- Passing candidates: "
        f"{', '.join(gate['passing_candidates']) or 'none'}.",
        f"- Selected candidate: {selected or 'none'}.",
        "",
        "- The four mode positions are balanced exactly across the eight tasks.",
        "- Four global warmups are excluded from every reported value.",
        "- One measured run per mode and task is used; aggregate direction "
        "matters more than any single task.",
        "- No DSpark stats, acceptance audit, trace, diagnostics, or profiler "
        "is enabled.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    selected_records, selection = corpus.select_records(
        all_records, oracle.TASK_COUNT, provenance["selection_policy"]
    )
    reference = oracle.load_throughput_reference(
        args, selected_records, selection
    )
    records = [reference["tasks"][task]["record"] for task in TASKS]

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-aggressive-scheduler-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for task_index, task in enumerate(TASKS):
        order = measured_order(task_index)
        print(f"{task} order: {' -> '.join(order)}")
        for mode in order:
            print(
                f"  {MODE_LABELS[mode]}: "
                f"{command_text(args, prompts[task], mode)}"
            )
    measured_count = len(TASKS) * len(MODES)
    print(
        f"Aggressive scheduler gate: {measured_count + len(MODES)} processes; "
        f"{len(MODES)} excluded warmups and {measured_count} measured."
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
        "experiment": "dspark_humaneval_aggressive_scheduler_gate",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "tasks": TASKS,
            "modes": MODES,
            "thresholds": THRESHOLDS,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "instrumented": False,
            "promotion_gate": {
                "minimum_geometric_mean_vs_current":
                    MIN_GEOMEAN_VS_CURRENT,
                "minimum_task_wins": MIN_TASK_WINS,
                "minimum_task_ratio": MIN_TASK_RATIO,
                "high_threshold_selection_margin":
                    HIGH_THRESHOLD_SELECTION_MARGIN,
            },
        },
        "throughput_reference": {
            "summary": common.file_metadata(reference["summary_path"]),
            "metadata": common.file_metadata(reference["metadata_path"]),
            "csv": common.file_metadata(reference["csv_path"]),
        },
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "commands": {
            task: {
                mode: command_text(args, prompts[task], mode)
                for mode in MODES
            }
            for task in TASKS
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    warmup_rows = []
    warmup_task = TASKS[0]
    warmup_reference = reference["tasks"][warmup_task]["output_data"]
    for position, mode in enumerate(MODES, start=1):
        row = execute(
            args, root, run_dir, f"warmup-{position:02d}",
            warmup_task, prompts[warmup_task], mode, warmup_reference,
        )
        row.update(order_position=position)
        warmup_rows.append(row)
        common.cooldown(args.cooldown)

    measured_rows = []
    sequence = 0
    for task_index, task in enumerate(TASKS):
        expected = reference["tasks"][task]["output_data"]
        order = measured_order(task_index)
        order_text = "-".join(order)
        for position, mode in enumerate(order, start=1):
            sequence += 1
            row = execute(
                args, root, run_dir, f"measured-{sequence:02d}",
                task, prompts[task], mode, expected,
            )
            row.update(
                sequence=sequence,
                order=order_text,
                order_position=position,
            )
            measured_rows.append(row)
            common.cooldown(args.cooldown)

    write_csv(run_dir / "throughput.csv", measured_rows)
    write_csv(run_dir / "warmups.csv", warmup_rows)
    summary = summarize(measured_rows, reference)
    summary["tasks_predeclared"] = TASKS
    summary["thresholds_predeclared"] = THRESHOLDS
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
