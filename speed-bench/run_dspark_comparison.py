#!/usr/bin/env python3
"""Run an alternating, user-initiated baseline/DSpark throughput comparison."""

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
        description="Compare deterministic ds4 baseline and DSpark GPU runtime throughput."
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.tokens <= 0 or args.ctx <= 0 or args.pairs <= 0 or args.warmups < 0:
        parser.error("tokens, ctx, and pairs must be positive; warmups cannot be negative")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
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


def clean_dspark_env(mode):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    if mode == "runtime":
        env["DS4_DSPARK_GPU_RUNTIME"] = "1"
        env["DS4_DSPARK_MULTI_COMMIT"] = "1"
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
    if mode == "runtime":
        command.extend(("--dspark", str(args.dspark_model.resolve())))
    return command


def command_text(args, mode):
    env = "" if mode == "baseline" else (
        "DS4_DSPARK_GPU_RUNTIME=1 DS4_DSPARK_MULTI_COMMIT=1 "
    )
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


def execute_run(args, root, run_dir, label, mode, reference_output):
    stdout_path = run_dir / f"{label}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{mode}.stderr"
    command = mode_command(args, mode)
    print(f"[{label}] {mode}: {command_text(args, mode)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command,
            cwd=root,
            env=clean_dspark_env(mode),
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
            f"{mode} output differs from the baseline reference; see {stdout_path}"
        )
    prefill_tps, generation_tps = parse_timing(stderr_data, stderr_path)
    return {
        "mode": mode,
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": wall_seconds,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "stdout_data": stdout_data,
    }


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
            mode: command_text(args, mode) for mode in ("baseline", "runtime")
        },
    }


def summarize(rows):
    baseline = [row["generation_tps"] for row in rows if row["mode"] == "baseline"]
    runtime = [row["generation_tps"] for row in rows if row["mode"] == "runtime"]
    paired = []
    for pair in sorted({row["pair"] for row in rows}):
        pair_rows = {row["mode"]: row for row in rows if row["pair"] == pair}
        paired.append(
            pair_rows["runtime"]["generation_tps"]
            / pair_rows["baseline"]["generation_tps"]
        )
    return {
        "baseline_generation_tps_median": statistics.median(baseline),
        "runtime_generation_tps_median": statistics.median(runtime),
        "median_ratio_of_medians": statistics.median(runtime)
        / statistics.median(baseline),
        "paired_speedup_median": statistics.median(paired),
        "paired_speedup_values": paired,
    }


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
    )
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

    report = (
        "# DSpark Benchmark Summary\n\n"
        f"- Baseline median: {summary['baseline_generation_tps_median']:.2f} t/s\n"
        f"- Runtime median: {summary['runtime_generation_tps_median']:.2f} t/s\n"
        f"- Ratio of medians: {summary['median_ratio_of_medians']:.4f}x\n"
        f"- Median paired speedup: {summary['paired_speedup_median']:.4f}x\n"
        f"- Measured pairs: {len(summary['paired_speedup_values'])}\n"
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

    print("Baseline command:")
    print("  " + command_text(args, "baseline"))
    print("Runtime command:")
    print("  " + command_text(args, "runtime"))
    print("DSpark and inherited instrumentation diagnostics are forcibly unset.")
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
        for mode in ("baseline", "runtime"):
            result = execute_run(
                args, root, run_dir, f"warmup-{warmup:02d}", mode, reference_output
            )
            if reference_output is None:
                reference_output = result["stdout_data"]
            cooldown(args.cooldown)

    rows = []
    sequence = 0
    for pair in range(1, args.pairs + 1):
        order = ("baseline", "runtime") if pair % 2 else ("runtime", "baseline")
        for position, mode in enumerate(order, 1):
            sequence += 1
            result = execute_run(
                args,
                root,
                run_dir,
                f"measured-{sequence:02d}",
                mode,
                reference_output,
            )
            if reference_output is None:
                reference_output = result["stdout_data"]
            result.update(sequence=sequence, pair=pair, position=position)
            result.pop("stdout_data")
            rows.append(result)
            cooldown(args.cooldown)

    summary = summarize(rows)
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
