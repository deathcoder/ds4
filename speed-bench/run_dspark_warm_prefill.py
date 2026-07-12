#!/usr/bin/env python3
"""Compare baseline and DSpark prefill after process-local GPU warmup."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time


CSV_FIELDS = (
    "kind",
    "run",
    "prompt_tokens",
    "prefill_seconds",
    "prefill_tps",
    "argmax",
    "logits_hash",
)
INSTRUMENTATION_MARKERS = ("PROFILE", "TRACE", "DUMP", "TIMING")


def cleared_env_keys(env):
    return sorted(
        key
        for key in env
        if key.startswith("DS4_DSPARK_")
        or (
            key.startswith("DS4_")
            and (
                any(marker in key for marker in INSTRUMENTATION_MARKERS)
                or key.endswith("_LOG")
            )
        )
    )


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Measure cold and process-warm fresh-session prefill."
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4-warm-prefill-bench")
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
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="fresh sessions before measured warm sessions; first is recorded as cold",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.pairs <= 0 or args.runs <= 0:
        parser.error("ctx, pairs, and runs must be positive")
    if args.warmups <= 0:
        parser.error("warmups must be positive so a cold conditioning session is recorded")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.confirm_ready and not args.dry_run:
        parser.error("refusing to benchmark without --confirm-ready")
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


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_metadata(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }


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


def clean_env(mode):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    if mode == "runtime":
        env["DS4_DSPARK_GPU_RUNTIME"] = "1"
        env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    return env


def mode_command(args, mode):
    command = [
        str(args.binary),
        "--model",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--ctx",
        str(args.ctx),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
    ]
    if mode == "runtime":
        command.extend(("--dspark", str(args.dspark_model)))
    return command


def command_text(args, mode):
    prefix = ""
    if mode == "runtime":
        prefix = "DS4_DSPARK_GPU_RUNTIME=1 DS4_DSPARK_MULTI_COMMIT=1 "
    return prefix + shlex.join(mode_command(args, mode))


def parse_output(data, source, warmups, runs):
    try:
        text = data.decode("ascii")
        reader = csv.DictReader(io.StringIO(text))
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"unexpected header {reader.fieldnames!r}")
        rows = []
        for raw in reader:
            row = {
                "kind": raw["kind"],
                "run": int(raw["run"]),
                "prompt_tokens": int(raw["prompt_tokens"]),
                "prefill_seconds": float(raw["prefill_seconds"]),
                "prefill_tps": float(raw["prefill_tps"]),
                "argmax": int(raw["argmax"]),
                "logits_hash": raw["logits_hash"],
            }
            if row["kind"] not in {"cold", "conditioning", "warm"}:
                raise ValueError(f"unexpected sample kind {row['kind']!r}")
            if (
                row["prompt_tokens"] <= 0
                or not math.isfinite(row["prefill_seconds"])
                or row["prefill_seconds"] <= 0
                or not math.isfinite(row["prefill_tps"])
                or row["prefill_tps"] <= 0
            ):
                raise ValueError("invalid prompt size, duration, or throughput")
            if len(row["logits_hash"]) != 16 or any(
                char not in "0123456789abcdef" for char in row["logits_hash"]
            ):
                raise ValueError(f"invalid logits hash {row['logits_hash']!r}")
            rows.append(row)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"invalid benchmark CSV in {source}: {exc}") from exc
    expected = warmups + runs
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} samples in {source}, found {len(rows)}")
    kinds = [row["kind"] for row in rows]
    expected_kinds = ["cold"] + ["conditioning"] * (warmups - 1) + ["warm"] * runs
    if kinds != expected_kinds:
        raise RuntimeError(f"unexpected sample order in {source}: {kinds}")
    return rows


def execute_child(args, root, run_dir, sequence, pair, position, mode, reference):
    label = f"measured-{sequence:02d}.{mode}"
    stdout_path = run_dir / f"{label}.stdout.csv"
    stderr_path = run_dir / f"{label}.stderr"
    command = mode_command(args, mode)
    print(f"[{pair}:{position}] {command_text(args, mode)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command,
            cwd=root,
            env=clean_env(mode),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} child failed with exit {completed.returncode}; see {stderr_path}"
        )
    samples = parse_output(
        stdout_path.read_bytes(), stdout_path, args.warmups, args.runs
    )
    for sample in samples:
        identity = (
            sample["prompt_tokens"],
            sample["argmax"],
            sample["logits_hash"],
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise RuntimeError(
                f"target logits differ from reference in {stdout_path}: "
                f"expected {reference}, got {identity}"
            )
        sample.update(
            sequence=sequence,
            pair=pair,
            position=position,
            mode=mode,
            child_wall_seconds=wall_seconds,
            stdout_file=stdout_path.name,
            stderr_file=stderr_path.name,
        )
    return samples, reference


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def collect_metadata(args, root):
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
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("DS4_")
        },
        "cleared_environment_keys": cleared_env_keys(os.environ),
        "config": {
            "ctx": args.ctx,
            "pairs": args.pairs,
            "conditioning_sessions_per_child": args.warmups,
            "warm_sessions_per_child": args.runs,
            "cooldown_seconds": args.cooldown,
            "runtime_diagnostics": False,
            "runtime_stats": False,
        },
        "binary": file_metadata(args.binary),
        "base_model": file_metadata(args.model),
        "dspark_model": file_metadata(args.dspark_model),
        "prompt": {
            **file_metadata(args.prompt_file),
            "sha256": sha256(args.prompt_file.read_bytes()),
        },
        "commands": {
            mode: command_text(args, mode) for mode in ("baseline", "runtime")
        },
    }


def summarize(rows):
    warm = [row for row in rows if row["kind"] == "warm"]
    cold = [row for row in rows if row["kind"] == "cold"]
    baseline = [row["prefill_tps"] for row in warm if row["mode"] == "baseline"]
    runtime = [row["prefill_tps"] for row in warm if row["mode"] == "runtime"]
    cold_baseline = [row["prefill_tps"] for row in cold if row["mode"] == "baseline"]
    cold_runtime = [row["prefill_tps"] for row in cold if row["mode"] == "runtime"]
    paired = []
    for pair in sorted({row["pair"] for row in warm}):
        by_mode = {}
        for mode in ("baseline", "runtime"):
            values = [
                row["prefill_tps"]
                for row in warm
                if row["pair"] == pair and row["mode"] == mode
            ]
            by_mode[mode] = statistics.median(values)
        paired.append(by_mode["runtime"] / by_mode["baseline"])
    children = []
    for sequence in sorted({row["sequence"] for row in rows}):
        child_rows = [row for row in rows if row["sequence"] == sequence]
        children.append(
            {
                "mode": child_rows[0]["mode"],
                "wall_seconds": child_rows[0]["child_wall_seconds"],
                "non_sync_seconds": child_rows[0]["child_wall_seconds"]
                - sum(row["prefill_seconds"] for row in child_rows),
            }
        )

    def child_median(mode, field):
        return statistics.median(
            child[field] for child in children if child["mode"] == mode
        )

    return {
        "baseline_warm_prefill_tps_median": statistics.median(baseline),
        "runtime_warm_prefill_tps_median": statistics.median(runtime),
        "warm_ratio_of_medians": statistics.median(runtime)
        / statistics.median(baseline),
        "warm_paired_speedup_median": statistics.median(paired),
        "warm_paired_speedup_values": paired,
        "baseline_cold_prefill_tps_median": statistics.median(cold_baseline),
        "runtime_cold_prefill_tps_median": statistics.median(cold_runtime),
        "cold_ratio_of_medians": statistics.median(cold_runtime)
        / statistics.median(cold_baseline),
        "baseline_child_wall_seconds_median": child_median(
            "baseline", "wall_seconds"
        ),
        "runtime_child_wall_seconds_median": child_median(
            "runtime", "wall_seconds"
        ),
        "child_wall_ratio_of_medians": child_median("runtime", "wall_seconds")
        / child_median("baseline", "wall_seconds"),
        "baseline_child_non_sync_seconds_median": child_median(
            "baseline", "non_sync_seconds"
        ),
        "runtime_child_non_sync_seconds_median": child_median(
            "runtime", "non_sync_seconds"
        ),
    }


def write_results(run_dir, rows, summary, metadata, root):
    fields = (
        "sequence",
        "pair",
        "position",
        "mode",
        "kind",
        "run",
        "prompt_tokens",
        "prefill_seconds",
        "prefill_tps",
        "argmax",
        "logits_hash",
        "child_wall_seconds",
        "stdout_file",
        "stderr_file",
    )
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["thermal_state_after"] = run_capture(["pmset", "-g", "therm"], root)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    report = (
        "# DSpark Warm Prefill Benchmark Summary\n\n"
        f"- Baseline warm median: {summary['baseline_warm_prefill_tps_median']:.2f} t/s\n"
        f"- Runtime warm median: {summary['runtime_warm_prefill_tps_median']:.2f} t/s\n"
        f"- Warm ratio of medians: {summary['warm_ratio_of_medians']:.4f}x\n"
        f"- Warm median paired ratio: {summary['warm_paired_speedup_median']:.4f}x\n"
        f"- Baseline cold median: {summary['baseline_cold_prefill_tps_median']:.2f} t/s\n"
        f"- Runtime cold median: {summary['runtime_cold_prefill_tps_median']:.2f} t/s\n"
        f"- Cold ratio of medians: {summary['cold_ratio_of_medians']:.4f}x\n"
        f"- Baseline child wall median: "
        f"{summary['baseline_child_wall_seconds_median']:.3f} s\n"
        f"- Runtime child wall median: "
        f"{summary['runtime_child_wall_seconds_median']:.3f} s\n"
        f"- Child wall ratio of medians: "
        f"{summary['child_wall_ratio_of_medians']:.4f}x\n"
        f"- Baseline non-sync child overhead median: "
        f"{summary['baseline_child_non_sync_seconds_median']:.3f} s\n"
        f"- Runtime non-sync child overhead median: "
        f"{summary['runtime_child_non_sync_seconds_median']:.3f} s\n"
        f"- Measured process pairs: {len(summary['warm_paired_speedup_values'])}\n"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    return csv_path, report


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.prompt_file = args.prompt_file.resolve()
    check_inputs(args, root)
    print("Baseline command:\n  " + command_text(args, "baseline"))
    print("Runtime command:\n  " + command_text(args, "runtime"))
    print("Diagnostics and runtime stats are unset in both modes.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("warm-prefill-%Y%m%d-%H%M%S")
    run_dir = (args.output_dir or root / "speed-bench/local-runs" / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = collect_metadata(args, root)
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    reference = None
    sequence = 0
    for pair in range(1, args.pairs + 1):
        order = ("baseline", "runtime") if pair % 2 else ("runtime", "baseline")
        for position, mode in enumerate(order, 1):
            sequence += 1
            child_rows, reference = execute_child(
                args, root, run_dir, sequence, pair, position, mode, reference
            )
            rows.extend(child_rows)
            cooldown(args.cooldown)

    summary = summarize(rows)
    csv_path, report = write_results(run_dir, rows, summary, metadata, root)
    print("\n" + report.rstrip())
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
