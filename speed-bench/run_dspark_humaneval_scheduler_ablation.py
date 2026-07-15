#!/usr/bin/env python3
"""Compare fixed K=5 with two predeclared DSpark confidence policies."""

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
import run_dspark_humaneval_exact_profile as exact_profile
import run_dspark_issue468_comparison as common


TASKS = ("humaneval_152", "humaneval_079")
MODES = ("fixed_k5", "threshold_038", "threshold_0455")
PAIRS = 3
WARMUP_PERIODS = 1
THRESHOLDS = {
    "fixed_k5": common.DSPARK_FIXED_CONFIDENCE_THRESHOLD,
    "threshold_038": "0.38",
    "threshold_0455": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
}
MODE_LABELS = {
    "fixed_k5": "Fixed K=5",
    "threshold_038": "Threshold 0.38",
    "threshold_0455": "Threshold 0.455",
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared low/high-acceptance DSpark confidence "
            "scheduler ablation."
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

    # Attributes consumed by shared corpus and command helpers.
    args.tasks = TASKS
    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = False
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    return args, root


def rotated_modes(offset):
    offset %= len(MODES)
    return MODES[offset:] + MODES[:offset]


def measured_order(task_index, pair):
    return rotated_modes(task_index + pair - 1)


def mode_env(mode):
    env = common.benchmark_env("runtime", False)
    threshold = THRESHOLDS[mode]
    if threshold is not None:
        env["DS4_DSPARK_CONFIDENCE_THRESHOLD"] = threshold
    return env


def command(args, prompt):
    return common.mode_command(args, prompt, "runtime")


def command_text(args, prompt, mode):
    env = mode_env(mode)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args, prompt))


def execute(args, root, run_dir, name, task, prompt, mode, reference):
    stdout_path = run_dir / f"{name}.{task}.{mode}.stdout"
    stderr_path = run_dir / f"{name}.{task}.{mode}.stderr"
    print(f"[{name}/{task}] {MODE_LABELS[mode]}: "
          f"{command_text(args, prompt, mode)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args, prompt),
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
            f"{name}/{task}/{mode} output differs from the validated exact "
            f"runtime; see {stdout_path}"
        )
    forbidden = (
        common.STATS_PREFIX,
        common.ACCEPTANCE_PREFIX,
        common.ACCEPTANCE_TRACE_PREFIX,
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


def summarize(rows, reference):
    tasks = {}
    aggregate_ratios = {mode: [] for mode in MODES[1:]}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        by_mode = {
            mode: [row for row in selected if row["mode"] == mode]
            for mode in MODES
        }
        item = {
            "acceptance_verify_rate":
                reference["tasks"][task]["sample"]["acceptance_verify_rate"],
            "prior_baseline_ratio":
                reference["tasks"][task]["sample"]["paired_ratio"],
            "modes": {},
        }
        fixed_by_pair = {row["pair"]: row for row in by_mode["fixed_k5"]}
        for mode in MODES:
            values = [row["generation_tps"] for row in by_mode[mode]]
            mode_item = {"generation_tps_median": statistics.median(values)}
            if mode != "fixed_k5":
                ratios = [
                    row["generation_tps"] /
                    fixed_by_pair[row["pair"]]["generation_tps"]
                    for row in by_mode[mode]
                ]
                mode_item.update(
                    paired_ratio_median=statistics.median(ratios),
                    paired_ratio_geometric_mean=statistics.geometric_mean(ratios),
                    paired_ratio_values=ratios,
                    faster_pairs=sum(ratio > 1.0 for ratio in ratios),
                )
                aggregate_ratios[mode].extend(ratios)
            item["modes"][mode] = mode_item
        tasks[task] = item

    aggregate = {}
    for mode in MODES[1:]:
        ratios = aggregate_ratios[mode]
        aggregate[mode] = {
            "paired_ratio_median": statistics.median(ratios),
            "paired_ratio_geometric_mean": statistics.geometric_mean(ratios),
            "paired_ratio_values": ratios,
            "faster_pairs": sum(ratio > 1.0 for ratio in ratios),
            "measured_pairs": len(ratios),
        }
    return {"tasks": tasks, "aggregate": aggregate}


def render_report(summary):
    lines = [
        "# DSpark Confidence Scheduler Runtime Ablation",
        "",
        "All runs are paired, uninstrumented, and use exact target verification.",
        "Every output matched the prior validated exact DSpark artifact byte-for-byte.",
        "Thresholds and tasks were fixed before this throughput run.",
        "",
        "| task | acceptance | mode | median | paired vs fixed | faster pairs |",
        "|:---|---:|:---|---:|---:|---:|",
    ]
    for task, item in summary["tasks"].items():
        for mode in MODES:
            mode_item = item["modes"][mode]
            ratio = (
                "reference" if mode == "fixed_k5" else
                f"{mode_item['paired_ratio_median']:.4f}x"
            )
            faster = (
                "n/a" if mode == "fixed_k5" else
                f"{mode_item['faster_pairs']}/{len(mode_item['paired_ratio_values'])}"
            )
            lines.append(
                f"| {task} | {item['acceptance_verify_rate']:.3f} | "
                f"{MODE_LABELS[mode]} | "
                f"{mode_item['generation_tps_median']:.2f} t/s | "
                f"{ratio} | {faster} |"
            )
    lines.extend([
        "",
        "## Aggregate Paired Ratios",
        "",
        "| policy | median | geometric mean | faster pairs |",
        "|:---|---:|---:|---:|",
    ])
    for mode in MODES[1:]:
        item = summary["aggregate"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {item['paired_ratio_median']:.4f}x | "
            f"{item['paired_ratio_geometric_mean']:.4f}x | "
            f"{item['faster_pairs']}/{item['measured_pairs']} |"
        )
    lines.extend([
        "",
        "- Each task uses a three-period Latin rotation so every mode occupies "
        "every order position.",
        "- Warmups are excluded from all reported values.",
        "- No DSpark stats, acceptance audit, trace, diagnostics, or profiler is enabled.",
        "- This gate tests actual Metal scheduling cost; offline target-position "
        "proxies are not substituted for throughput.",
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
    reference = exact_profile.load_throughput_reference(args, all_records)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-scheduler-ablation-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    records = [reference["tasks"][task]["record"] for task in TASKS]
    prompts = corpus.prompt_paths(run_dir, records)
    for task_index, task in enumerate(TASKS):
        for pair in range(1, PAIRS + 1):
            order = measured_order(task_index, pair)
            print(f"{task} pair {pair} order: {' -> '.join(order)}")
            for mode in order:
                print(f"  {MODE_LABELS[mode]}: "
                      f"{command_text(args, prompts[task], mode)}")
    total = WARMUP_PERIODS * len(MODES) + PAIRS * len(TASKS) * len(MODES)
    print(
        f"Scheduler ablation: {total} exact DSpark processes; "
        f"{WARMUP_PERIODS * len(MODES)} warmup and "
        f"{PAIRS * len(TASKS) * len(MODES)} measured."
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
        "experiment": "deepspec_humaneval_confidence_scheduler_ablation",
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "pairs": PAIRS,
            "warmups": WARMUP_PERIODS,
            "cooldown": args.cooldown,
            "tasks": TASKS,
            "modes": MODES,
            "thresholds": THRESHOLDS,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "instrumented": False,
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
    low_reference = reference["tasks"][TASKS[0]]["prior_runtime_output"].read_bytes()
    for warmup in range(1, WARMUP_PERIODS + 1):
        for position, mode in enumerate(rotated_modes(warmup - 1), start=1):
            row = execute(
                args, root, run_dir, f"warmup-{warmup:02d}-{position}",
                TASKS[0], prompts[TASKS[0]], mode, low_reference,
            )
            row.update(warmup=warmup, order_position=position)
            warmup_rows.append(row)
            common.cooldown(args.cooldown)

    measured_rows = []
    sequence = 0
    for task_index, task in enumerate(TASKS):
        prior_output = reference["tasks"][task]["prior_runtime_output"].read_bytes()
        for pair in range(1, PAIRS + 1):
            order = measured_order(task_index, pair)
            order_text = "-".join(order)
            for position, mode in enumerate(order, start=1):
                sequence += 1
                row = execute(
                    args, root, run_dir,
                    f"measured-{sequence:02d}", task, prompts[task], mode,
                    prior_output,
                )
                row.update(
                    sequence=sequence,
                    pair=pair,
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
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["final_snapshot"] = common.machine_snapshot(root)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + report.rstrip())
    print(f"Raw throughput: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
