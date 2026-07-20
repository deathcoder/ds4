#!/usr/bin/env python3
"""Confirm exact routed gate/up SwiGLU width-5 execution on frozen HumanEval."""

import argparse
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
import run_dspark_humaneval_dense_mixed_direct as frozen
import run_dspark_humaneval_throughput as throughput
import run_dspark_issue468_comparison as common


THRESHOLD = "0.75"
SAMPLE_COUNT = 32
MODES = ("default_exact", "exact_routed_gate_up_swiglu_w5")
MIN_GEOMEAN = 1.005
MIN_WINS = 20
MIN_TASK_RATIO = 0.95
LOW_ACCEPTANCE_MAX = 0.65
MIN_LOW_ACCEPTANCE_GEOMEAN = 1.00


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare default exact DSpark with fused routed gate/up SwiGLU "
            "at verifier width 5 on frozen threshold-0.75 HumanEval."
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
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
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
    return args, root


def mode_order(position):
    return MODES if position % 2 == 1 else tuple(reversed(MODES))


def warmup_schedule(records):
    return (
        (records[0], MODES),
        (records[-1], tuple(reversed(MODES))),
    )


def command(args, prompt):
    return [
        str(args.binary),
        "--backend", "metal",
        "--model", str(args.model),
        "--prompt-file", str(prompt),
        "--ctx", str(args.ctx),
        "--nothink",
        "--temp", "0",
        "--seed", "1",
        "-n", str(args.tokens),
        "--dspark", str(args.dspark_model),
    ]


def mode_env(mode):
    if mode not in MODES:
        raise ValueError(f"unknown routed-SwiGLU mode: {mode}")
    env = common.benchmark_env(
        "runtime", False, confidence_threshold=THRESHOLD
    )
    env.pop("DS4_DSPARK_EXACT_ROUTED_GATE_UP_SWIGLU_W5", None)
    if mode == "exact_routed_gate_up_swiglu_w5":
        env["DS4_DSPARK_EXACT_ROUTED_GATE_UP_SWIGLU_W5"] = "1"
    return env


def command_text(args, prompt, mode):
    env = mode_env(mode)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_EXACT_ROUTED_GATE_UP_SWIGLU_W5",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args, prompt))


def execute(args, root, run_dir, label, record, prompt, mode, expected):
    stdout_path = run_dir / f"{label}.{record['label']}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{record['label']}.{mode}.stderr"
    print(
        f"[{label}/{record['label']}] {mode}: "
        f"{command_text(args, prompt, mode)}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            command(args, prompt),
            cwd=root,
            env=mode_env(mode),
            stdout=out,
            stderr=err,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} run failed with exit {completed.returncode}; see {stderr_path}"
        )
    output = stdout_path.read_bytes()
    if output != expected:
        raise RuntimeError(
            f"{mode} output differs from frozen exact output; see {stdout_path}"
        )
    stderr_data = stderr_path.read_bytes()
    prefill_tps, generation_tps = common.parse_timing(stderr_data, stderr_path)
    forbidden = (
        common.STATS_PREFIX,
        common.ACCEPTANCE_PREFIX,
        common.ACCEPTANCE_TRACE_PREFIX,
        b"DSpark exact routed gate/up SwiGLU width=5 layer=",
    )
    if any(marker in stderr_data for marker in forbidden):
        raise RuntimeError(f"instrumentation unexpectedly active in {stderr_path}")
    return {
        "prompt": record["label"],
        "source_index": record["source_index"],
        "mode": mode,
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": common.sha256(output),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def run_pair(args, root, run_dir, label, record, prompt, order, expected):
    rows = []
    order_text = "-".join(order)
    for position, mode in enumerate(order, start=1):
        row = execute(
            args, root, run_dir, label, record, prompt, mode, expected
        )
        row["pair_order"] = order_text
        row["pair_position"] = position
        rows.append(row)
        common.cooldown(args.cooldown)
    return rows


def summarize(rows, records, reference):
    samples = {}
    ratios = []
    low_acceptance_ratios = []
    default_values = []
    candidate_values = []
    wins = 0
    equals = 0
    for record in records:
        task = record["label"]
        selected = {row["mode"]: row for row in rows if row["prompt"] == task}
        if set(selected) != set(MODES):
            raise RuntimeError(f"incomplete routed-SwiGLU pair for {task}")
        default = selected["default_exact"]["generation_tps"]
        candidate = selected["exact_routed_gate_up_swiglu_w5"][
            "generation_tps"
        ]
        ratio = candidate / default
        acceptance = reference["tasks"][task]["acceptance_verify_rate"]
        ratios.append(ratio)
        default_values.append(default)
        candidate_values.append(candidate)
        if acceptance <= LOW_ACCEPTANCE_MAX:
            low_acceptance_ratios.append(ratio)
        wins += ratio > 1.0
        equals += ratio == 1.0
        samples[task] = {
            "source_index": record["source_index"],
            "order": selected["default_exact"]["pair_order"],
            "acceptance_verify_rate": acceptance,
            "default_generation_tps": default,
            "candidate_generation_tps": candidate,
            "paired_ratio": ratio,
            "delta_percent": (ratio - 1.0) * 100.0,
        }
    quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
    geomean = statistics.geometric_mean(ratios)
    low_geomean = statistics.geometric_mean(low_acceptance_ratios)
    gate = (
        geomean >= MIN_GEOMEAN and
        wins >= MIN_WINS and
        min(ratios) >= MIN_TASK_RATIO and
        low_geomean >= MIN_LOW_ACCEPTANCE_GEOMEAN
    )
    return {
        "analysis": "dspark_humaneval_exact_routed_gate_up_swiglu_w5",
        "sample_count": len(records),
        "threshold": THRESHOLD,
        "samples": samples,
        "default_generation_tps_median": statistics.median(default_values),
        "candidate_generation_tps_median": statistics.median(candidate_values),
        "ratio_of_medians": (
            statistics.median(candidate_values) /
            statistics.median(default_values)
        ),
        "paired_ratio_median": statistics.median(ratios),
        "paired_ratio_geometric_mean": geomean,
        "paired_ratio_q1": quartiles[0],
        "paired_ratio_q3": quartiles[2],
        "paired_ratio_minimum": min(ratios),
        "paired_ratio_maximum": max(ratios),
        "candidate_faster_tasks": wins,
        "candidate_equal_tasks": equals,
        "candidate_slower_tasks": len(records) - wins - equals,
        "acceptance_speed_pearson": throughput.pearson_correlation(
            [
                reference["tasks"][record["label"]]["acceptance_verify_rate"]
                for record in records
            ],
            ratios,
        ),
        "low_acceptance_task_count": len(low_acceptance_ratios),
        "low_acceptance_geometric_mean": low_geomean,
        "promotion_gate": {
            "minimum_geometric_mean": MIN_GEOMEAN,
            "minimum_wins": MIN_WINS,
            "minimum_task_ratio": MIN_TASK_RATIO,
            "low_acceptance_maximum": LOW_ACCEPTANCE_MAX,
            "minimum_low_acceptance_geometric_mean":
                MIN_LOW_ACCEPTANCE_GEOMEAN,
            "pass": gate,
        },
    }


def render_report(summary):
    correlation = summary["acceptance_speed_pearson"]
    correlation_text = "n/a" if correlation is None else f"{correlation:.3f}"
    lines = [
        "# DSpark HumanEval Exact Routed Gate/Up SwiGLU Width-5 Confirmation",
        "",
        "All samples are uninstrumented and paired within the same frozen "
        "threshold-0.75 HumanEval task.",
        "Every default and fused output matched the frozen exact artifact "
        "byte-for-byte.",
        "Generation t/s excludes process startup; paired ratios are authoritative.",
        "",
        "| samples | default median | fused median | ratio of medians | "
        "median paired | geometric mean | fused faster |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | "
        f"{summary['default_generation_tps_median']:.2f} t/s | "
        f"{summary['candidate_generation_tps_median']:.2f} t/s | "
        f"{summary['ratio_of_medians']:.4f}x | "
        f"{summary['paired_ratio_median']:.4f}x | "
        f"{summary['paired_ratio_geometric_mean']:.4f}x | "
        f"{summary['candidate_faster_tasks']}/{summary['sample_count']} |",
        "",
        f"- Paired-ratio interquartile range: "
        f"{summary['paired_ratio_q1']:.4f}x-"
        f"{summary['paired_ratio_q3']:.4f}x.",
        f"- Paired-ratio range: {summary['paired_ratio_minimum']:.4f}x-"
        f"{summary['paired_ratio_maximum']:.4f}x.",
        f"- Tasks faster/equal/slower with fused gate/up SwiGLU: "
        f"{summary['candidate_faster_tasks']}/"
        f"{summary['candidate_equal_tasks']}/"
        f"{summary['candidate_slower_tasks']}.",
        f"- Low-acceptance tasks (verify rate <= {LOW_ACCEPTANCE_MAX:.2f}): "
        f"{summary['low_acceptance_task_count']}; geometric candidate/default "
        f"ratio {summary['low_acceptance_geometric_mean']:.4f}x.",
        f"- Descriptive Pearson correlation, acceptance versus paired ratio: "
        f"{correlation_text}.",
        "",
        "## Tasks",
        "",
        "| task | acceptance | order | default | fused | ratio | delta |",
        "|:---|---:|:---|---:|---:|---:|---:|",
    ]
    for task, item in summary["samples"].items():
        lines.append(
            f"| {task} | {item['acceptance_verify_rate']:.3f} | "
            f"{item['order']} | "
            f"{item['default_generation_tps']:.2f} t/s | "
            f"{item['candidate_generation_tps']:.2f} t/s | "
            f"{item['paired_ratio']:.4f}x | "
            f"{item['delta_percent']:+.1f}% |"
        )
    gate = summary["promotion_gate"]
    lines.extend([
        "",
        "## Promotion Gate",
        "",
        f"**{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        f"- Require geometric mean at least "
        f"`{gate['minimum_geometric_mean']:.3f}x`.",
        f"- Require at least `{gate['minimum_wins']}/32` faster tasks.",
        f"- Require no task below `{gate['minimum_task_ratio']:.2f}x`.",
        f"- Require low-acceptance geometric mean at least "
        f"`{gate['minimum_low_acceptance_geometric_mean']:.2f}x`.",
        "",
        "- Two global warmup pairs are excluded from every reported value.",
        "- Measured order alternates default-first and fused-first.",
        "- No DSpark stats, trace, diagnostics, profiler, or fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference",
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
    dirty = common.git_output(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n"
            + dirty
        )

    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, SAMPLE_COUNT, provenance["selection_policy"]
    )
    reference = frozen.load_reference(args, records, selection)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-routed-gate-up-swiglu-w5-"
        f"{SAMPLE_COUNT}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for position, record in enumerate(records, start=1):
        order = mode_order(position)
        prompt = prompts[record["label"]]
        print(
            f"{record['label']} measured order: {' -> '.join(order)}\n"
            f"  default: {command_text(args, prompt, 'default_exact')}\n"
            f"  fused: {command_text(args, prompt, 'exact_routed_gate_up_swiglu_w5')}"
        )
    warmups = warmup_schedule(records)
    total = len(warmups) * 2 + len(records) * 2
    print(
        f"Routed-SwiGLU width-5 confirmation: {total} uninstrumented processes; "
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
        "experiment": "dspark_humaneval_exact_routed_gate_up_swiglu_w5",
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
            "reference_mode": "default_exact",
            "candidate_mode": "exact_routed_gate_up_swiglu_w5",
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
        "commands": {
            record["label"]: {
                mode: command_text(args, prompts[record["label"]], mode)
                for mode in MODES
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
            prompts[record["label"]], mode_order(position),
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
