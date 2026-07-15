#!/usr/bin/env python3
"""Reproduce the issue-468 8k workload with paired no-log DSpark runs."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time


TIMING_RE = re.compile(
    rb"ds4: prefill: ([0-9]+(?:\.[0-9]+)?) t/s, "
    rb"generation: ([0-9]+(?:\.[0-9]+)?) t/s"
)
STATS_PREFIX = b"ds4: DSpark runtime stats "
INT_STATS = {
    "proposals", "selected", "source_fallbacks", "multi_attempts", "emitted",
    "target_evals", "target_eval_tokens", "target_evals_avoided",
    "batch_attempts", "batch_full", "batch_partial", "batch_fallbacks",
    "fast_calls", "fast_failures", "fast_exact_fallbacks",
    "depth1", "depth2", "depth3", "depth4", "depth5",
}
FLOAT_STATS = {
    "avg_depth", "sidecar_ms", "bridge_ms", "stage0_ms", "stage1_ms",
    "stage2_ms", "head_ms", "chain_ms", "target_eval_ms",
    "prefill_sidecar_ms", "generation_sidecar_ms", "generation_bridge_ms",
    "generation_stage0_ms", "generation_stage1_ms", "generation_stage2_ms",
    "generation_head_ms", "generation_chain_ms",
}
STATS_FIELDS = tuple(sorted(INT_STATS | FLOAT_STATS))
PROMPT_ORDER = ("code_8k", "synthesis_8k", "grounded_8k")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def run_capture(command, cwd):
    try:
        return subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        ).stdout.decode("utf-8", "replace").strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def git_output(root, *args):
    return run_capture(["git", *args], root)


def cleared_env_keys(env):
    return sorted(key for key in env if key.startswith("DS4_"))


def benchmark_env(mode, fast_verifier, stats=False, exact_head_batch=False):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    if mode == "runtime":
        env["DS4_DSPARK_GPU_RUNTIME"] = "1"
        env["DS4_DSPARK_MULTI_COMMIT"] = "1"
        if fast_verifier:
            env["DS4_DSPARK_FAST_BATCH_VERIFY"] = "1"
        if exact_head_batch:
            env["DS4_DSPARK_EXACT_HEAD_BATCH"] = "1"
        if stats:
            env["DS4_DSPARK_GPU_RUNTIME_STATS"] = "1"
    return env


def parse_args():
    root = Path(__file__).resolve().parent.parent
    corpus = root / "speed-bench/issue468"
    parser = argparse.ArgumentParser(
        description="Run the issue-468 long-prompt baseline/DSpark comparison."
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model", type=Path,
        default=root / "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
    )
    parser.add_argument(
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
    )
    parser.add_argument("--corpus-dir", type=Path, default=corpus)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--stats-pass", action="store_true")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="run one baseline reference and one instrumented runtime per prompt",
    )
    parser.add_argument(
        "--fast-verifier", action="store_true",
        help="use the experimental compute-batched verifier (not correctness-safe on this corpus)",
    )
    parser.add_argument(
        "--exact-head-batch", action="store_true",
        help="batch intermediate output heads while retaining exact target state and final logits",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0 or args.pairs <= 0 or args.warmups < 0:
        parser.error("ctx, tokens, and pairs must be positive; warmups cannot be negative")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to benchmark without --confirm-ready")
    if args.fast_verifier and args.exact_head_batch:
        parser.error("--fast-verifier and --exact-head-batch are separate experiments")
    if args.stats_only and args.stats_pass:
        parser.error("--stats-only and --stats-pass are mutually exclusive")
    return args, root


def load_inputs(args, root):
    provenance_path = args.corpus_dir / "provenance.json"
    reference_path = args.corpus_dir / "mtp_reference.json"
    for label, path in (
        ("binary", args.binary), ("base model", args.model),
        ("DSpark model", args.dspark_model), ("provenance", provenance_path),
        ("MTP reference", reference_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" + dirty
        )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    prompts = {}
    for label in PROMPT_ORDER:
        item = provenance["prompts"][label]
        path = args.corpus_dir / item["file"]
        if not path.is_file():
            raise SystemExit(f"missing prompt {label}: {path}")
        data = path.read_bytes()
        if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
            raise SystemExit(f"provenance mismatch for {label}: {path}")
        prompts[label] = path
    return prompts, provenance, reference


def mode_command(args, prompt, mode):
    command = [
        str(args.binary), "--backend", "metal", "--model", str(args.model),
        "--ctx", str(args.ctx), "-n", str(args.tokens), "--temp", "0",
        "--seed", "1", "--prompt-file", str(prompt),
    ]
    if mode == "runtime":
        command.extend(("--dspark", str(args.dspark_model)))
    return command


def command_text(args, prompt, mode, stats=False):
    env = benchmark_env(mode, args.fast_verifier, stats, args.exact_head_batch)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME", "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_FAST_BATCH_VERIFY", "DS4_DSPARK_EXACT_HEAD_BATCH",
        "DS4_DSPARK_GPU_RUNTIME_STATS",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return (prefix + " " if prefix else "") + shlex.join(
        mode_command(args, prompt, mode)
    )


def parse_timing(stderr_data, path):
    matches = TIMING_RE.findall(stderr_data)
    if not matches:
        raise RuntimeError(f"timing line not found in {path}")
    prefill, generation = (float(value) for value in matches[-1])
    if prefill <= 0 or generation <= 0:
        raise RuntimeError(f"non-positive throughput in {path}")
    return prefill, generation


def parse_stats(stderr_data, path):
    records = [
        line[len(STATS_PREFIX):] for line in stderr_data.splitlines()
        if line.startswith(STATS_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(f"expected one DSpark stats record in {path}, found {len(records)}")
    values = {}
    try:
        for item in records[0].decode("ascii").split():
            key, value = item.split("=", 1)
            if key in INT_STATS:
                values[key] = int(value)
            elif key in FLOAT_STATS:
                values[key] = float(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid DSpark stats in {path}: {exc}") from exc
    missing = [key for key in STATS_FIELDS if key not in values]
    if missing:
        raise RuntimeError(f"incomplete DSpark stats in {path}: {', '.join(missing)}")
    if values["emitted"] <= 0 or values["target_evals"] <= 0:
        raise RuntimeError(f"empty DSpark stats in {path}")
    return values


def execute(args, root, run_dir, label, prompt_label, prompt, mode, reference, stats=False):
    stdout_path = run_dir / f"{label}.{prompt_label}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{prompt_label}.{mode}.stderr"
    command = mode_command(args, prompt, mode)
    print(f"[{label}/{prompt_label}] {mode}: {command_text(args, prompt, mode, stats)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command, cwd=root,
            env=benchmark_env(
                mode, args.fast_verifier, stats, args.exact_head_batch
            ),
            stdout=stdout_fp, stderr=stderr_fp, check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"run failed with exit {completed.returncode}; see {stderr_path}")
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    if reference is not None and stdout_data != reference:
        raise RuntimeError(f"output differs from {prompt_label} baseline; see {stdout_path}")
    prefill_tps, generation_tps = parse_timing(stderr_data, stderr_path)
    if not stats and STATS_PREFIX in stderr_data:
        raise RuntimeError(f"throughput run unexpectedly emitted DSpark stats: {stderr_path}")
    row = {
        "prompt": prompt_label, "mode": mode, "prefill_tps": prefill_tps,
        "generation_tps": generation_tps, "wall_seconds": wall_seconds,
        "stdout_sha256": sha256(stdout_data), "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    if stats:
        row.update(parse_stats(stderr_data, stderr_path))
    return row, stdout_data


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def file_metadata(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size,
            "mtime": dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()}


def machine_snapshot(root):
    return {
        "captured_at": dt.datetime.now().astimezone().isoformat(),
        "thermal_state": run_capture(["pmset", "-g", "therm"], root),
        "processes": run_capture(
            ["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"], root
        ).splitlines()[:25],
    }


def collect_metadata(args, root, prompts, provenance):
    commands = {}
    for label, prompt in prompts.items():
        if args.stats_only:
            commands[label] = {
                "baseline_reference": command_text(args, prompt, "baseline"),
                "stats_runtime": command_text(
                    args, prompt, "runtime", stats=True
                ),
            }
        else:
            commands[label] = {
                mode: command_text(args, prompt, mode)
                for mode in ("baseline", "runtime")
            }
            if args.stats_pass:
                commands[label]["stats_runtime"] = command_text(
                    args, prompt, "runtime", stats=True
                )
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": git_output(root, "status", "--porcelain", "--untracked-files=no"),
        "platform": platform.platform(), "uname": run_capture(["uname", "-a"], root),
        "cpu": run_capture(["sysctl", "-n", "machdep.cpu.brand_string"], root),
        "memory_bytes": run_capture(["sysctl", "-n", "hw.memsize"], root),
        "initial_snapshot": machine_snapshot(root),
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("DS4_")
        },
        "cleared_environment_keys": cleared_env_keys(os.environ),
        "child_ds4_environment_policy": {
            "clear_all_inherited_ds4_keys": True,
            "baseline_keys": [],
            "runtime_keys": [
                "DS4_DSPARK_GPU_RUNTIME",
                "DS4_DSPARK_MULTI_COMMIT",
            ],
            "optional_runtime_keys": [
                "DS4_DSPARK_FAST_BATCH_VERIFY",
                "DS4_DSPARK_EXACT_HEAD_BATCH",
                "DS4_DSPARK_GPU_RUNTIME_STATS",
            ],
        },
        "config": {
            "ctx": args.ctx, "tokens": args.tokens,
            "pairs": 0 if args.stats_only else args.pairs,
            "warmups_per_mode_per_prompt": 0 if args.stats_only else args.warmups,
            "cooldown_seconds": args.cooldown, "temperature": 0, "seed": 1,
            "fast_verifier": args.fast_verifier,
            "exact_head_batch": args.exact_head_batch,
            "execution_mode": "stats_only" if args.stats_only else "throughput",
            "throughput_instrumentation": False,
            "runtime_instrumentation": args.stats_only or args.stats_pass,
            "stats_pass": args.stats_pass,
            "stats_only": args.stats_only,
            "nothink": False,
        },
        "binary": file_metadata(args.binary), "base_model": file_metadata(args.model),
        "dspark_model": file_metadata(args.dspark_model), "provenance": provenance,
        "prompts": {label: file_metadata(path) for label, path in prompts.items()},
        "commands": commands,
    }


def summarize(rows, reference):
    prompts = {}
    all_paired = []
    for label in PROMPT_ORDER:
        selected = [row for row in rows if row["prompt"] == label]
        baseline = [row["generation_tps"] for row in selected if row["mode"] == "baseline"]
        runtime = [row["generation_tps"] for row in selected if row["mode"] == "runtime"]
        paired = []
        for pair in sorted({row["pair"] for row in selected}):
            pair_rows = {row["mode"]: row for row in selected if row["pair"] == pair}
            paired.append(pair_rows["runtime"]["generation_tps"] / pair_rows["baseline"]["generation_tps"])
        all_paired.extend(paired)
        ratio = statistics.median(runtime) / statistics.median(baseline)
        dspark_delta = (ratio - 1.0) * 100.0
        mtp = reference["results"][label]
        prompts[label] = {
            "baseline_generation_tps_median": statistics.median(baseline),
            "runtime_generation_tps_median": statistics.median(runtime),
            "ratio_of_medians": ratio,
            "paired_ratio_median": statistics.median(paired),
            "paired_ratio_values": paired,
            "dspark_delta_percent": dspark_delta,
            "improvement_over_mtp_k2_percentage_points": dspark_delta - mtp["k2"]["delta_percent"],
            "improvement_over_mtp_k5_percentage_points": dspark_delta - mtp["k5"]["delta_percent"],
        }
    aggregate_ratio = statistics.median(all_paired)
    aggregate_delta = (aggregate_ratio - 1.0) * 100.0
    k2_delta = statistics.median(reference["results"][label]["k2"]["delta_percent"] for label in PROMPT_ORDER)
    k5_delta = statistics.median(reference["results"][label]["k5"]["delta_percent"] for label in PROMPT_ORDER)
    return {
        "prompts": prompts,
        "aggregate_paired_ratio_median": aggregate_ratio,
        "aggregate_paired_ratio_values": all_paired,
        "aggregate_dspark_delta_percent": aggregate_delta,
        "reference_mtp_k2_delta_percent_median": k2_delta,
        "reference_mtp_k5_delta_percent_median": k5_delta,
        "aggregate_improvement_over_mtp_k2_percentage_points": aggregate_delta - k2_delta,
        "aggregate_improvement_over_mtp_k5_percentage_points": aggregate_delta - k5_delta,
    }


def summarize_stats(rows):
    summary = {}
    for row in rows:
        emitted = row["emitted"]
        target_evals = row["target_evals"]
        target_ms_per_emitted = row["target_eval_ms"] / emitted
        sidecar_ms_per_emitted = row["generation_sidecar_ms"] / emitted
        summary[row["prompt"]] = {
            "emitted": emitted,
            "average_accepted_depth": row["avg_depth"],
            "target_evals_avoided": row["target_evals_avoided"],
            "target_evals_per_emitted": target_evals / emitted,
            "target_positions_per_eval": row["target_eval_tokens"] / target_evals,
            "target_eval_ms_per_eval": row["target_eval_ms"] / target_evals,
            "target_eval_ms_per_emitted": target_ms_per_emitted,
            "generation_sidecar_ms_per_emitted": sidecar_ms_per_emitted,
            "accounted_generation_ms_per_emitted": (
                target_ms_per_emitted + sidecar_ms_per_emitted
            ),
            "generation_bridge_ms_per_emitted": row["generation_bridge_ms"] / emitted,
            "generation_stage0_ms_per_emitted": row["generation_stage0_ms"] / emitted,
            "generation_stage1_ms_per_emitted": row["generation_stage1_ms"] / emitted,
            "generation_stage2_ms_per_emitted": row["generation_stage2_ms"] / emitted,
            "generation_head_ms_per_emitted": row["generation_head_ms"] / emitted,
            "generation_chain_ms_per_emitted": row["generation_chain_ms"] / emitted,
            "prefill_sidecar_ms": row["prefill_sidecar_ms"],
            "batch_attempts": row["batch_attempts"],
            "batch_full": row["batch_full"],
            "batch_partial": row["batch_partial"],
            "fast_calls": row["fast_calls"], "fast_failures": row["fast_failures"],
            "fast_exact_fallbacks": row["fast_exact_fallbacks"],
            "batch_fallbacks": row["batch_fallbacks"],
            "source_fallbacks": row["source_fallbacks"],
            "depth_counts": {
                str(depth): row[f"depth{depth}"] for depth in range(1, 6)
            },
        }
    return summary


def render_stats_report(summary):
    lines = [
        "# DSpark Issue 468 Stats-Only Summary",
        "",
        "Instrumented diagnostic only. Throughput values are intentionally omitted.",
        "Each runtime output matched a fresh uninstrumented baseline reference.",
        "",
        "| prompt | depth | evals/emitted | positions/eval | target ms/emitted | sidecar ms/emitted | accounted ms/emitted | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in PROMPT_ORDER:
        item = summary[label]
        lines.append(
            f"| {label} | {item['average_accepted_depth']:.3f} | "
            f"{item['target_evals_per_emitted']:.4f} | "
            f"{item['target_positions_per_eval']:.3f} | "
            f"{item['target_eval_ms_per_emitted']:.3f} | "
            f"{item['generation_sidecar_ms_per_emitted']:.3f} | "
            f"{item['accounted_generation_ms_per_emitted']:.3f} | "
            f"{item['batch_fallbacks']} |"
        )
    lines.extend(["", "Sidecar breakdown per emitted token:"])
    for label in PROMPT_ORDER:
        item = summary[label]
        lines.append(
            f"- {label}: bridge {item['generation_bridge_ms_per_emitted']:.3f} ms, "
            f"stages {item['generation_stage0_ms_per_emitted']:.3f}/"
            f"{item['generation_stage1_ms_per_emitted']:.3f}/"
            f"{item['generation_stage2_ms_per_emitted']:.3f} ms, "
            f"head {item['generation_head_ms_per_emitted']:.3f} ms, "
            f"chain {item['generation_chain_ms_per_emitted']:.3f} ms"
        )
    lines.extend(["", "Verifier outcomes:"])
    for label in PROMPT_ORDER:
        item = summary[label]
        lines.append(
            f"- {label}: {item['target_evals_avoided']} target evals avoided; "
            f"target {item['target_eval_ms_per_eval']:.3f} ms/eval; "
            f"batches {item['batch_attempts']} attempts, {item['batch_full']} full, "
            f"{item['batch_partial']} partial, {item['batch_fallbacks']} fallbacks; "
            f"source fallbacks {item['source_fallbacks']}; "
            f"fast {item['fast_calls']} calls/{item['fast_failures']} failures/"
            f"{item['fast_exact_fallbacks']} exact fallbacks"
        )
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def render_report(summary, stats_summary=None):
    lines = [
        "# DSpark Issue 468 Comparison Summary", "",
        "Throughput samples are paired and uninstrumented. Published MTP values are",
        "single instrumented runs on another system; compare relative deltas only, not absolute t/s.", "",
        "| prompt | baseline | DSpark | ratio | DSpark delta | vs MTP K=2 | vs MTP K=5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in PROMPT_ORDER:
        item = summary["prompts"][label]
        lines.append(
            f"| {label} | {item['baseline_generation_tps_median']:.2f} t/s | "
            f"{item['runtime_generation_tps_median']:.2f} t/s | {item['paired_ratio_median']:.4f}x | "
            f"{item['dspark_delta_percent']:+.1f}% | "
            f"{item['improvement_over_mtp_k2_percentage_points']:+.1f} pp | "
            f"{item['improvement_over_mtp_k5_percentage_points']:+.1f} pp |"
        )
    lines.extend([
        "", f"- Aggregate median paired ratio: {summary['aggregate_paired_ratio_median']:.4f}x",
        f"- Aggregate DSpark delta: {summary['aggregate_dspark_delta_percent']:+.1f}%",
        f"- Published MTP K=2 median delta: {summary['reference_mtp_k2_delta_percent_median']:+.1f}%",
        f"- Improvement over published MTP K=2: {summary['aggregate_improvement_over_mtp_k2_percentage_points']:+.1f} percentage points",
        f"- Published MTP K=5 median delta: {summary['reference_mtp_k5_delta_percent_median']:+.1f}%",
        f"- Improvement over published MTP K=5: {summary['aggregate_improvement_over_mtp_k5_percentage_points']:+.1f} percentage points",
        f"- Measured pairs per prompt: {len(summary['prompts'][PROMPT_ORDER[0]]['paired_ratio_values'])}",
    ])
    if stats_summary:
        lines.extend(["", "## Separate Instrumentation Pass", ""])
        for label in PROMPT_ORDER:
            item = stats_summary[label]
            lines.append(
                f"- {label}: depth {item['average_accepted_depth']:.3f}, "
                f"target evals/emitted {item['target_evals_per_emitted']:.4f}, "
                f"positions/eval {item['target_positions_per_eval']:.3f}, "
                f"sidecar {item['generation_sidecar_ms_per_emitted']:.3f} ms/emitted, "
                f"fast {item['fast_calls']} calls/{item['fast_failures']} failures/"
                f"{item['fast_exact_fallbacks']} exact fallbacks"
            )
    return "\n".join(lines) + "\n"


def finish_metadata(metadata, root, run_dir):
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["final_snapshot"] = machine_snapshot(root)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def run_stats_only(args, root, run_dir, prompts, metadata):
    runs = []
    stats_rows = []
    for prompt_label, prompt in prompts.items():
        baseline_row, reference = execute(
            args, root, run_dir, "reference", prompt_label, prompt,
            "baseline", None,
        )
        runs.append(baseline_row)
        cooldown(args.cooldown)
        stats_row, _ = execute(
            args, root, run_dir, "stats", prompt_label, prompt,
            "runtime", reference, stats=True,
        )
        runs.append(stats_row)
        stats_rows.append(stats_row)
        cooldown(args.cooldown)

    fields = (
        "prompt", "mode", "prefill_tps", "generation_tps", "wall_seconds",
        "stdout_sha256", "stdout_file", "stderr_file",
    )
    write_csv(run_dir / "runs.csv", runs, fields + STATS_FIELDS)
    write_csv(run_dir / "stats.csv", stats_rows, fields + STATS_FIELDS)
    summary = summarize_stats(stats_rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = render_stats_report(summary)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw stats: {run_dir / 'stats.csv'}")
    return 0


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.corpus_dir = args.corpus_dir.resolve()
    prompts, provenance, reference = load_inputs(args, root)

    for label, prompt in prompts.items():
        if args.stats_only:
            print(
                f"{label} baseline reference: "
                f"{command_text(args, prompt, 'baseline')}"
            )
            print(
                f"{label} stats runtime:     "
                f"{command_text(args, prompt, 'runtime', stats=True)}"
            )
        else:
            print(f"{label} baseline: {command_text(args, prompt, 'baseline')}")
            print(f"{label} runtime:  {command_text(args, prompt, 'runtime')}")
    if args.stats_only:
        print(
            "Stats-only pass: one fresh baseline reference and one instrumented "
            "exact runtime per prompt; no throughput pairs."
        )
    else:
        print("Throughput pass: all DSpark stats and diagnostic instrumentation are disabled.")
    if args.fast_verifier:
        print(
            "WARNING: fast verification is known to diverge on code_8k; "
            "this mode is for correctness investigation, not performance reporting."
        )
    if args.exact_head_batch:
        print(
            "Exact-head batch mode: intermediate target heads are batched; "
            "target state and final continuation logits remain serial-exact."
        )
    if args.stats_pass:
        print("A separate one-run-per-prompt runtime stats pass will follow throughput.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = (
        f"issue468-stats-{stamp}" if args.stats_only else f"issue468-{stamp}"
    )
    run_dir = (
        args.output_dir or root / "speed-bench/local-runs" / default_dir
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = collect_metadata(args, root, prompts, provenance)
    (run_dir / "metadata.start.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.stats_only:
        return run_stats_only(args, root, run_dir, prompts, metadata)

    references = {}
    for prompt_label, prompt in prompts.items():
        for warmup in range(1, args.warmups + 1):
            for mode in ("baseline", "runtime"):
                _, output = execute(
                    args, root, run_dir, f"warmup-{warmup:02d}", prompt_label,
                    prompt, mode, references.get(prompt_label),
                )
                references.setdefault(prompt_label, output)
                cooldown(args.cooldown)

    rows = []
    sequence = 0
    for prompt_label, prompt in prompts.items():
        for pair in range(1, args.pairs + 1):
            order = ("baseline", "runtime") if pair % 2 else ("runtime", "baseline")
            for position, mode in enumerate(order, 1):
                sequence += 1
                row, output = execute(
                    args, root, run_dir, f"measured-{sequence:02d}", prompt_label,
                    prompt, mode, references.get(prompt_label),
                )
                references.setdefault(prompt_label, output)
                row.update(sequence=sequence, pair=pair, position=position)
                rows.append(row)
                cooldown(args.cooldown)

    summary = summarize(rows, reference)
    stats_rows = []
    stats_summary = None
    if args.stats_pass:
        for prompt_label, prompt in prompts.items():
            row, _ = execute(
                args, root, run_dir, "stats", prompt_label, prompt, "runtime",
                references[prompt_label], stats=True,
            )
            stats_rows.append(row)
            cooldown(args.cooldown)
        stats_summary = summarize_stats(stats_rows)

    throughput_fields = (
        "sequence", "prompt", "pair", "position", "mode", "prefill_tps",
        "generation_tps", "wall_seconds", "stdout_sha256", "stdout_file", "stderr_file",
    )
    write_csv(run_dir / "throughput.csv", rows, throughput_fields)
    if stats_rows:
        write_csv(
            run_dir / "stats.csv", stats_rows,
            ("prompt", "mode", "prefill_tps", "generation_tps", "wall_seconds",
             "stdout_sha256", "stdout_file", "stderr_file") + STATS_FIELDS,
        )
        (run_dir / "stats_summary.json").write_text(
            json.dumps(stats_summary, indent=2) + "\n", encoding="utf-8"
        )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render_report(summary, stats_summary)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw results: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial raw files were retained.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
