#!/usr/bin/env python3
"""Run an alternating, user-initiated DSpark throughput comparison."""

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
RUNTIME_STATS_PREFIX = b"ds4: DSpark runtime stats "
RUNTIME_STATS_INT_FIELDS = {
    "proposals",
    "selected",
    "source_fallbacks",
    "multi_attempts",
    "emitted",
    "target_evals",
    "target_eval_tokens",
    "target_evals_avoided",
    "batch_attempts",
    "batch_full",
    "batch_partial",
    "batch_fallbacks",
    "fast_calls",
    "fast_failures",
    "fast_exact_fallbacks",
    "exact_ffn_batch_attempts",
    "exact_ffn_batch_successes",
    "exact_attn_pre_batch_attempts",
    "exact_attn_pre_batch_successes",
    "exact_attn_suffix_batch_attempts",
    "exact_attn_suffix_batch_successes",
    "depth1",
    "depth2",
    "depth3",
    "depth4",
    "depth5",
}
RUNTIME_STATS_FLOAT_FIELDS = {
    "avg_depth",
    "sidecar_ms",
    "bridge_ms",
    "stage0_ms",
    "stage1_ms",
    "stage2_ms",
    "head_ms",
    "chain_ms",
    "target_eval_ms",
    "prefill_sidecar_ms",
    "generation_sidecar_ms",
    "generation_bridge_ms",
    "generation_stage0_ms",
    "generation_stage1_ms",
    "generation_stage2_ms",
    "generation_head_ms",
    "generation_chain_ms",
}
RUNTIME_STATS_FIELDS = tuple(
    f"stats_{name}"
    for name in sorted(RUNTIME_STATS_INT_FIELDS | RUNTIME_STATS_FLOAT_FIELDS)
)
INSTRUMENTATION_MARKERS = ("PROFILE", "TRACE", "DUMP", "TIMING")


def cleared_env_keys(env):
    keys = []
    for key in env:
        if key.startswith("DS4_DSPARK_") or (
            key.startswith("DS4_")
            and (any(marker in key for marker in INSTRUMENTATION_MARKERS)
                 or key.endswith("_LOG"))
        ):
            keys.append(key)
    return sorted(keys)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Compare deterministic ds4 and DSpark GPU runtime throughput."
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model",
        type=Path,
        default=root
        / "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
    )
    parser.add_argument(
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=root / "speed-bench/dspark_prompt.txt"
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-idle", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--fast-verifier",
        action="store_true",
        help="use the opt-in compute-batched DSpark target verifier",
    )
    parser.add_argument(
        "--serial-ffn-ablation",
        "--exact-ffn-batch-ablation",
        dest="serial_ffn_ablation",
        action="store_true",
        help="compare serial exact DSpark against default exact FFN batching",
    )
    parser.add_argument(
        "--attention-pre-ablation",
        action="store_true",
        help="compare serial attention preparation against default exact DSpark",
    )
    parser.add_argument(
        "--attention-suffix-ablation",
        action="store_true",
        help="compare default exact DSpark against exact attention-suffix batching",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.tokens <= 0 or args.ctx <= 0 or args.pairs <= 0 or args.warmups < 0:
        parser.error("tokens, ctx, and pairs must be positive; warmups cannot be negative")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    selected_modes = sum(
        bool(value)
        for value in (
            args.fast_verifier,
            args.serial_ffn_ablation,
            args.attention_pre_ablation,
            args.attention_suffix_ablation,
        )
    )
    if selected_modes > 1:
        parser.error(
            "--fast-verifier, --serial-ffn-ablation, --attention-pre-ablation, "
            "and --attention-suffix-ablation are mutually exclusive"
        )
    if not args.confirm_idle and not args.dry_run:
        parser.error("refusing to benchmark without --confirm-idle")
    return args, root


def run_capture(command, cwd):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.decode("utf-8", "replace").strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def git_output(root, *args):
    return run_capture(["git", *args], root)


def check_inputs(args, root):
    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("prompt", args.prompt_file),
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


def benchmark_modes(args):
    if args.attention_suffix_ablation:
        return ("default_exact", "batch_attention_suffix")
    if args.attention_pre_ablation:
        return ("serial_attention_pre", "default_exact")
    if args.serial_ffn_ablation:
        return ("serial_exact", "runtime")
    return ("baseline", "runtime")


def mode_label(mode, args):
    if mode == "runtime" and args.fast_verifier:
        return "Fast verifier DSpark"
    return {
        "baseline": "Baseline",
        "runtime": "Default exact DSpark",
        "serial_exact": "Serial exact DSpark",
        "serial_attention_pre": "Serial attention-pre DSpark",
        "default_exact": "Default exact DSpark",
        "batch_attention_suffix": "Exact attention-suffix batch DSpark",
    }[mode]


def throughput_runtime_stats_enabled(args):
    return not (
        args.serial_ffn_ablation
        or args.attention_pre_ablation
        or args.attention_suffix_ablation
    )


def clean_dspark_env(mode, fast_verifier=False, runtime_stats=True):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    if mode != "baseline":
        env["DS4_DSPARK_GPU_RUNTIME"] = "1"
        env["DS4_DSPARK_MULTI_COMMIT"] = "1"
        if runtime_stats:
            env["DS4_DSPARK_GPU_RUNTIME_STATS"] = "1"
        if fast_verifier:
            env["DS4_DSPARK_FAST_BATCH_VERIFY"] = "1"
        if mode == "serial_exact":
            env["DS4_DSPARK_EXACT_FFN_BATCH"] = "0"
        if mode == "serial_attention_pre":
            env["DS4_DSPARK_EXACT_ATTN_PRE_BATCH"] = "0"
        if mode == "batch_attention_suffix":
            env["DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH"] = "1"
    return env


def common_command(args):
    return [
        str(args.binary.resolve()),
        "--model",
        str(args.model.resolve()),
        "--prompt-file",
        str(args.prompt_file.resolve()),
        "--ctx",
        str(args.ctx),
        "--nothink",
        "--temp",
        "0",
        "--seed",
        "1",
        "-n",
        str(args.tokens),
    ]


def mode_command(args, mode):
    command = common_command(args)
    if args.attention_pre_ablation or args.attention_suffix_ablation:
        command[1:1] = ("--backend", "metal")
    if mode != "baseline":
        command.extend(("--dspark", str(args.dspark_model.resolve())))
    return command


def command_text(args, mode, runtime_stats=None):
    if runtime_stats is None:
        runtime_stats = throughput_runtime_stats_enabled(args)
    env = ""
    if mode != "baseline":
        env = "DS4_DSPARK_GPU_RUNTIME=1 DS4_DSPARK_MULTI_COMMIT=1 "
        if runtime_stats:
            env += "DS4_DSPARK_GPU_RUNTIME_STATS=1 "
        if args.fast_verifier:
            env += "DS4_DSPARK_FAST_BATCH_VERIFY=1 "
        if mode == "serial_exact":
            env += "DS4_DSPARK_EXACT_FFN_BATCH=0 "
        if mode == "serial_attention_pre":
            env += "DS4_DSPARK_EXACT_ATTN_PRE_BATCH=0 "
        if mode == "batch_attention_suffix":
            env += "DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH=1 "
    return env + shlex.join(mode_command(args, mode))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_timing(stderr_data, stderr_path):
    matches = TIMING_RE.findall(stderr_data)
    if not matches:
        raise RuntimeError(f"timing line not found in {stderr_path}")
    prefill, generation = matches[-1]
    prefill = float(prefill)
    generation = float(generation)
    if prefill <= 0.0 or generation <= 0.0:
        raise RuntimeError(f"non-positive throughput in {stderr_path}")
    return prefill, generation


def parse_runtime_stats(stderr_data, stderr_path, mode, expect_stats):
    records = [
        line[len(RUNTIME_STATS_PREFIX):]
        for line in stderr_data.splitlines()
        if line.startswith(RUNTIME_STATS_PREFIX)
    ]
    if mode == "baseline" or not expect_stats:
        if records:
            raise RuntimeError(f"unexpected DSpark runtime stats in {stderr_path}")
        return {field: None for field in RUNTIME_STATS_FIELDS}
    if len(records) != 1:
        raise RuntimeError(
            f"expected one DSpark runtime stats record in {stderr_path}, found {len(records)}"
        )

    values = {}
    try:
        for item in records[0].decode("ascii").split():
            key, value = item.split("=", 1)
            if key in RUNTIME_STATS_INT_FIELDS:
                values[f"stats_{key}"] = int(value)
            elif key in RUNTIME_STATS_FLOAT_FIELDS:
                values[f"stats_{key}"] = float(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid DSpark runtime stats in {stderr_path}: {exc}") from exc

    missing = [field for field in RUNTIME_STATS_FIELDS if field not in values]
    if missing:
        raise RuntimeError(
            f"incomplete DSpark runtime stats in {stderr_path}: missing {', '.join(missing)}"
        )
    if values["stats_emitted"] <= 0 or values["stats_target_evals"] <= 0:
        raise RuntimeError(f"empty DSpark runtime stats in {stderr_path}")
    return values


def execute_run(
        args, root, run_dir, label, mode, reference_output, runtime_stats):
    stdout_path = run_dir / f"{label}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{mode}.stderr"
    command = mode_command(args, mode)
    print(
        f"[{label}] {mode}: {command_text(args, mode, runtime_stats)}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command,
            cwd=root,
            env=clean_dspark_env(mode, args.fast_verifier, runtime_stats),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} run failed with exit {completed.returncode}; see {stderr_path}"
        )

    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    if reference_output is not None and stdout_data != reference_output:
        raise RuntimeError(
            f"{mode} output differs from the first-mode reference; see {stdout_path}"
        )
    prefill_tps, generation_tps = parse_timing(stderr_data, stderr_path)
    result = {
        "mode": mode,
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": wall_seconds,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "stdout_data": stdout_data,
    }
    result.update(parse_runtime_stats(stderr_data, stderr_path, mode, runtime_stats))
    return result


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def file_metadata(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }


def collect_metadata(args, root):
    modes = benchmark_modes(args)
    runtime_stats = throughput_runtime_stats_enabled(args)
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "platform": platform.platform(),
        "uname": run_capture(["uname", "-a"], root),
        "cpu": run_capture(["sysctl", "-n", "machdep.cpu.brand_string"], root),
        "memory_bytes": run_capture(["sysctl", "-n", "hw.memsize"], root),
        "thermal_state": run_capture(["pmset", "-g", "therm"], root),
        "process_snapshot": run_capture(
            ["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"], root
        ).splitlines()[:25],
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("DS4_")
        },
        "cleared_environment_keys": cleared_env_keys(os.environ),
        "config": {
            "tokens": args.tokens,
            "ctx": args.ctx,
            "pairs": args.pairs,
            "warmups_per_mode": args.warmups,
            "cooldown_seconds": args.cooldown,
            "runtime_diagnostics": False,
            "runtime_stats": runtime_stats,
            "fast_verifier": args.fast_verifier,
            "serial_ffn_ablation": args.serial_ffn_ablation,
            "attention_pre_ablation": args.attention_pre_ablation,
            "attention_suffix_ablation": args.attention_suffix_ablation,
            "temperature": 0,
            "seed": 1,
        },
        "binary": file_metadata(args.binary),
        "base_model": file_metadata(args.model),
        "dspark_model": file_metadata(args.dspark_model),
        "prompt": {
            **file_metadata(args.prompt_file),
            "sha256": sha256(args.prompt_file.read_bytes()),
        },
        "commands": {
            mode: command_text(args, mode, runtime_stats) for mode in modes
        },
    }


def summarize(rows, modes=("baseline", "runtime")):
    reference_mode, candidate_mode = modes
    reference = [
        row["generation_tps"] for row in rows if row["mode"] == reference_mode
    ]
    candidate = [
        row["generation_tps"] for row in rows if row["mode"] == candidate_mode
    ]
    paired = []
    for pair in sorted({row["pair"] for row in rows}):
        pair_rows = {row["mode"]: row for row in rows if row["pair"] == pair}
        paired.append(
            pair_rows[candidate_mode]["generation_tps"]
            / pair_rows[reference_mode]["generation_tps"]
        )
    reference_median = statistics.median(reference)
    candidate_median = statistics.median(candidate)
    summary = {
        "comparison": (
            "attention_suffix_ablation"
            if modes == ("default_exact", "batch_attention_suffix")
            else (
                "attention_pre_ablation"
                if modes == ("serial_attention_pre", "default_exact")
                else (
                    "serial_ffn_ablation"
                    if modes == ("serial_exact", "runtime")
                    else "baseline_runtime"
                )
            )
        ),
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "reference_generation_tps_median": reference_median,
        "candidate_generation_tps_median": candidate_median,
        "median_ratio_of_medians": candidate_median / reference_median,
        "paired_speedup_median": statistics.median(paired),
        "paired_speedup_values": paired,
    }

    if modes == ("serial_exact", "runtime"):
        summary.update({
            "serial_exact_generation_tps_median": reference_median,
            "default_exact_ffn_generation_tps_median": candidate_median,
            "default_exact_ffn_delta_percent":
                (candidate_median / reference_median - 1.0) * 100.0,
        })
        return summary

    if modes == ("serial_attention_pre", "default_exact"):
        summary.update({
            "serial_attention_pre_generation_tps_median": reference_median,
            "default_exact_generation_tps_median": candidate_median,
            "default_exact_attention_pre_generation_tps_median": candidate_median,
            "default_exact_attention_pre_delta_percent":
                (candidate_median / reference_median - 1.0) * 100.0,
        })
        return summary

    if modes == ("default_exact", "batch_attention_suffix"):
        summary.update({
            "default_exact_generation_tps_median": reference_median,
            "batch_attention_suffix_generation_tps_median": candidate_median,
            "batch_attention_suffix_delta_percent":
                (candidate_median / reference_median - 1.0) * 100.0,
        })
        return summary

    runtime_rows = [row for row in rows if row["mode"] == "runtime"]

    def runtime_median(field):
        return statistics.median(row[field] for row in runtime_rows)

    def runtime_ratio(numerator, denominator):
        return statistics.median(
            row[numerator] / row[denominator] for row in runtime_rows
        )

    summary.update({
        "baseline_generation_tps_median": reference_median,
        "runtime_generation_tps_median": candidate_median,
        "runtime_avg_depth_median": runtime_median("stats_avg_depth"),
        "runtime_target_evals_avoided_median": runtime_median(
            "stats_target_evals_avoided"
        ),
        "runtime_target_evals_per_emitted_median": runtime_ratio(
            "stats_target_evals", "stats_emitted"
        ),
        "runtime_target_eval_tokens_per_call_median": runtime_ratio(
            "stats_target_eval_tokens", "stats_target_evals"
        ),
        "runtime_generation_sidecar_ms_per_emitted_median": runtime_ratio(
            "stats_generation_sidecar_ms", "stats_emitted"
        ),
        "runtime_target_eval_ms_per_eval_median": runtime_ratio(
            "stats_target_eval_ms", "stats_target_evals"
        ),
        "runtime_target_eval_ms_per_emitted_median": runtime_ratio(
            "stats_target_eval_ms", "stats_emitted"
        ),
        "runtime_prefill_sidecar_ms_median": runtime_median(
            "stats_prefill_sidecar_ms"
        ),
        "runtime_batch_attempts_median": runtime_median("stats_batch_attempts"),
        "runtime_batch_full_median": runtime_median("stats_batch_full"),
        "runtime_batch_partial_median": runtime_median("stats_batch_partial"),
        "runtime_batch_fallbacks_median": runtime_median("stats_batch_fallbacks"),
        "runtime_fast_calls_median": runtime_median("stats_fast_calls"),
        "runtime_fast_failures_median": runtime_median("stats_fast_failures"),
        "runtime_fast_exact_fallbacks_median": runtime_median(
            "stats_fast_exact_fallbacks"
        ),
        "runtime_exact_attn_pre_batch_attempts_median": runtime_median(
            "stats_exact_attn_pre_batch_attempts"
        ),
        "runtime_exact_attn_pre_batch_successes_median": runtime_median(
            "stats_exact_attn_pre_batch_successes"
        ),
        "runtime_exact_attn_suffix_batch_attempts_median": runtime_median(
            "stats_exact_attn_suffix_batch_attempts"
        ),
        "runtime_exact_attn_suffix_batch_successes_median": runtime_median(
            "stats_exact_attn_suffix_batch_successes"
        ),
    })
    for component in ("bridge", "stage0", "stage1", "stage2", "head", "chain"):
        summary[f"runtime_{component}_ms_per_emitted_median"] = runtime_ratio(
            f"stats_generation_{component}_ms", "stats_emitted"
        )
    return summary


def format_report(summary):
    if summary["comparison"] == "attention_suffix_ablation":
        return (
            "# DSpark Exact Attention Suffix Batch Ablation\n\n"
            f"- Default exact median: "
            f"{summary['default_exact_generation_tps_median']:.2f} t/s\n"
            f"- Exact attention-suffix batch median: "
            f"{summary['batch_attention_suffix_generation_tps_median']:.2f} t/s\n"
            f"- Ratio of medians: {summary['median_ratio_of_medians']:.4f}x\n"
            f"- Median paired ratio: {summary['paired_speedup_median']:.4f}x\n"
            f"- Exact attention-suffix batch delta: "
            f"{summary['batch_attention_suffix_delta_percent']:+.1f}%\n"
            f"- Measured pairs: {len(summary['paired_speedup_values'])}\n"
        )

    if summary["comparison"] == "serial_ffn_ablation":
        return (
            "# DSpark Serial FFN Control Ablation\n\n"
            f"- Serial exact median: "
            f"{summary['serial_exact_generation_tps_median']:.2f} t/s\n"
            f"- Default exact FFN median: "
            f"{summary['default_exact_ffn_generation_tps_median']:.2f} t/s\n"
            f"- Ratio of medians: {summary['median_ratio_of_medians']:.4f}x\n"
            f"- Median paired ratio: {summary['paired_speedup_median']:.4f}x\n"
            f"- Default exact FFN delta: "
            f"{summary['default_exact_ffn_delta_percent']:+.1f}%\n"
            f"- Measured pairs: {len(summary['paired_speedup_values'])}\n"
        )

    if summary["comparison"] == "attention_pre_ablation":
        return (
            "# DSpark Serial Attention Preparation Control Ablation\n\n"
            f"- Serial attention-pre median: "
            f"{summary['serial_attention_pre_generation_tps_median']:.2f} t/s\n"
            f"- Default exact attention-pre median: "
            f"{summary['default_exact_attention_pre_generation_tps_median']:.2f} t/s\n"
            f"- Ratio of medians: {summary['median_ratio_of_medians']:.4f}x\n"
            f"- Median paired ratio: {summary['paired_speedup_median']:.4f}x\n"
            f"- Default exact attention-pre delta: "
            f"{summary['default_exact_attention_pre_delta_percent']:+.1f}%\n"
            f"- Measured pairs: {len(summary['paired_speedup_values'])}\n"
        )

    return (
        "# DSpark Benchmark Summary\n\n"
        f"- Baseline median: {summary['baseline_generation_tps_median']:.2f} t/s\n"
        f"- Runtime median: {summary['runtime_generation_tps_median']:.2f} t/s\n"
        f"- Ratio of medians: {summary['median_ratio_of_medians']:.4f}x\n"
        f"- Median paired speedup: {summary['paired_speedup_median']:.4f}x\n"
        f"- Measured pairs: {len(summary['paired_speedup_values'])}\n"
        f"- Runtime average accepted depth: {summary['runtime_avg_depth_median']:.3f}\n"
        f"- Runtime target evals avoided: {summary['runtime_target_evals_avoided_median']:.1f}\n"
        f"- Runtime target evals / emitted token: "
        f"{summary['runtime_target_evals_per_emitted_median']:.4f}\n"
        f"- Runtime token positions / target eval: "
        f"{summary['runtime_target_eval_tokens_per_call_median']:.3f}\n"
        f"- Runtime generation sidecar / emitted token: "
        f"{summary['runtime_generation_sidecar_ms_per_emitted_median']:.3f} ms\n"
        f"- Runtime prefill sidecar total: "
        f"{summary['runtime_prefill_sidecar_ms_median']:.3f} ms\n"
        f"- Runtime target time / target eval: "
        f"{summary['runtime_target_eval_ms_per_eval_median']:.3f} ms\n"
        f"- Runtime target time / emitted token: "
        f"{summary['runtime_target_eval_ms_per_emitted_median']:.3f} ms\n"
        f"- Runtime batch outcomes: "
        f"{summary['runtime_batch_attempts_median']:.1f} attempts, "
        f"{summary['runtime_batch_full_median']:.1f} full, "
        f"{summary['runtime_batch_partial_median']:.1f} partial, "
        f"{summary['runtime_batch_fallbacks_median']:.1f} fallbacks\n"
        f"- Runtime fast verifier: "
        f"{summary['runtime_fast_calls_median']:.1f} calls, "
        f"{summary['runtime_fast_failures_median']:.1f} failures, "
        f"{summary['runtime_fast_exact_fallbacks_median']:.1f} exact fallbacks\n"
        f"- Runtime exact attention-pre outcomes: "
        f"{summary['runtime_exact_attn_pre_batch_successes_median']:.1f}/"
        f"{summary['runtime_exact_attn_pre_batch_attempts_median']:.1f} successful\n"
        f"- Runtime exact attention-suffix outcomes: "
        f"{summary['runtime_exact_attn_suffix_batch_successes_median']:.1f}/"
        f"{summary['runtime_exact_attn_suffix_batch_attempts_median']:.1f} successful\n"
        f"- Generation sidecar breakdown / emitted token: "
        f"bridge {summary['runtime_bridge_ms_per_emitted_median']:.3f} ms, "
        f"stages {summary['runtime_stage0_ms_per_emitted_median']:.3f}/"
        f"{summary['runtime_stage1_ms_per_emitted_median']:.3f}/"
        f"{summary['runtime_stage2_ms_per_emitted_median']:.3f} ms, "
        f"head {summary['runtime_head_ms_per_emitted_median']:.3f} ms, "
        f"chain {summary['runtime_chain_ms_per_emitted_median']:.3f} ms\n"
    )


def write_results(run_dir, rows, summary, metadata):
    csv_path = run_dir / "results.csv"
    fields = (
        "sequence",
        "pair",
        "position",
        "mode",
        "prefill_tps",
        "generation_tps",
        "wall_seconds",
        "stdout_sha256",
        "stdout_file",
        "stderr_file",
    ) + RUNTIME_STATS_FIELDS
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["thermal_state_after"] = run_capture(
        ["pmset", "-g", "therm"], run_dir.parent.parent
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    report = format_report(summary)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    return csv_path, report


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.prompt_file = args.prompt_file.resolve()
    check_inputs(args, root)

    modes = benchmark_modes(args)
    runtime_stats = throughput_runtime_stats_enabled(args)
    for mode in modes:
        print(f"{mode_label(mode, args)} command:")
        print("  " + command_text(args, mode, runtime_stats))
    if runtime_stats:
        print("DSpark diagnostics are unset; one end-of-session runtime stats record is enabled.")
    else:
        print("DSpark diagnostics and runtime stats are unset in both ablation modes.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_dir or root / "speed-bench/local-runs" / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = collect_metadata(args, root)
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    reference_output = None
    for warmup in range(1, args.warmups + 1):
        for mode in modes:
            result = execute_run(
                args,
                root,
                run_dir,
                f"warmup-{warmup:02d}",
                mode,
                reference_output,
                runtime_stats,
            )
            if reference_output is None:
                reference_output = result["stdout_data"]
            cooldown(args.cooldown)

    rows = []
    sequence = 0
    for pair in range(1, args.pairs + 1):
        order = modes if pair % 2 else tuple(reversed(modes))
        for position, mode in enumerate(order, 1):
            sequence += 1
            result = execute_run(
                args,
                root,
                run_dir,
                f"measured-{sequence:02d}",
                mode,
                reference_output,
                runtime_stats,
            )
            if reference_output is None:
                reference_output = result["stdout_data"]
            result.update(sequence=sequence, pair=pair, position=position)
            result.pop("stdout_data")
            rows.append(result)
            cooldown(args.cooldown)

    summary = summarize(rows, modes)
    csv_path, report = write_results(run_dir, rows, summary, metadata)
    print()
    print(report.rstrip())
    print(f"Raw results: {csv_path}")
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
