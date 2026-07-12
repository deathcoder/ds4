#!/usr/bin/env python3
"""User-run stats ablation for serial-exact versus exact-head DSpark."""

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
INT_FIELDS = {
    "emitted", "target_evals", "target_eval_tokens", "target_evals_avoided",
    "exact_head_batch_attempts", "exact_head_batch_successes",
}
FLOAT_FIELDS = {
    "avg_depth", "target_eval_ms", "exact_layer_ms",
    "exact_head_batch_ms", "exact_head_serial_ms", "generation_sidecar_ms",
}
INSTRUMENTATION_MARKERS = ("PROFILE", "TRACE", "DUMP", "TIMING")


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


def cleared_env_keys(env):
    return sorted(
        key for key in env
        if key.startswith("DS4_DSPARK_")
        or (key.startswith("DS4_") and (
            any(marker in key for marker in INSTRUMENTATION_MARKERS)
            or key.endswith("_LOG")
        ))
    )


def mode_env(mode):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    env["DS4_DSPARK_GPU_RUNTIME"] = "1"
    env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    env["DS4_DSPARK_GPU_RUNTIME_STATS"] = "1"
    if mode == "exact_head":
        env["DS4_DSPARK_EXACT_HEAD_BATCH"] = "1"
    return env


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Compare instrumented exact and exact-head DSpark directly."
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model", type=Path,
        default=root / "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
    )
    parser.add_argument(
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
    )
    parser.add_argument(
        "--prompt-file", type=Path,
        default=root / "speed-bench/issue468/code_8k.txt",
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--cooldown", type=float, default=5.0)
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
        parser.error("refusing to run without --confirm-ready")
    return args, root


def command(args):
    return [
        str(args.binary), "--backend", "metal", "--model", str(args.model),
        "--ctx", str(args.ctx), "-n", str(args.tokens), "--temp", "0",
        "--seed", "1", "--prompt-file", str(args.prompt_file),
        "--dspark", str(args.dspark_model),
    ]


def command_text(args, mode):
    env = mode_env(mode)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME", "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_STATS", "DS4_DSPARK_EXACT_HEAD_BATCH",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args))


def check_inputs(args, root):
    for label, path in (
        ("binary", args.binary), ("base model", args.model),
        ("DSpark model", args.dspark_model), ("prompt", args.prompt_file),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" + dirty
        )


def parse_timing(data, path):
    matches = TIMING_RE.findall(data)
    if not matches:
        raise RuntimeError(f"timing line not found in {path}")
    return tuple(float(value) for value in matches[-1])


def parse_stats(data, path):
    records = [
        line[len(STATS_PREFIX):] for line in data.splitlines()
        if line.startswith(STATS_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(f"expected one DSpark stats record in {path}, found {len(records)}")
    values = {}
    try:
        for item in records[0].decode("ascii").split():
            key, value = item.split("=", 1)
            if key in INT_FIELDS:
                values[key] = int(value)
            elif key in FLOAT_FIELDS:
                values[key] = float(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid DSpark stats in {path}: {exc}") from exc
    missing = sorted((INT_FIELDS | FLOAT_FIELDS) - values.keys())
    if missing:
        raise RuntimeError(f"incomplete DSpark stats in {path}: {', '.join(missing)}")
    if values["emitted"] <= 0 or values["target_evals"] <= 0:
        raise RuntimeError(f"empty DSpark stats in {path}")
    return values


def execute(args, root, run_dir, label, mode, reference):
    stdout_path = run_dir / f"{label}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{mode}.stderr"
    print(f"[{label}] {mode}: {command_text(args, mode)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args), cwd=root, env=mode_env(mode),
            stdout=stdout_fp, stderr=stderr_fp, check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} failed with exit {completed.returncode}; see {stderr_path}")
    stdout_data = stdout_path.read_bytes()
    if reference is not None and stdout_data != reference:
        raise RuntimeError(f"{mode} output differs from exact reference; see {stdout_path}")
    stderr_data = stderr_path.read_bytes()
    prefill_tps, generation_tps = parse_timing(stderr_data, stderr_path)
    row = {
        "mode": mode, "prefill_tps": prefill_tps,
        "generation_tps": generation_tps, "wall_seconds": wall_seconds,
        "stdout_sha256": sha256(stdout_data), "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    row.update(parse_stats(stderr_data, stderr_path))
    return row, stdout_data


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def summarize(rows):
    result = {"modes": {}}
    for mode in ("exact", "exact_head"):
        selected = [row for row in rows if row["mode"] == mode]
        emitted = statistics.median(row["emitted"] for row in selected)
        target_ms = statistics.median(row["target_eval_ms"] for row in selected)
        layer_ms = statistics.median(row["exact_layer_ms"] for row in selected)
        batch_ms = statistics.median(row["exact_head_batch_ms"] for row in selected)
        serial_ms = statistics.median(row["exact_head_serial_ms"] for row in selected)
        result["modes"][mode] = {
            "samples": len(selected),
            "emitted_median": emitted,
            "target_eval_ms_per_emitted": target_ms / emitted,
            "exact_layer_ms_per_emitted": layer_ms / emitted,
            "exact_head_batch_ms_per_emitted": batch_ms / emitted,
            "exact_head_serial_ms_per_emitted": serial_ms / emitted,
            "target_residual_ms_per_emitted":
                (target_ms - layer_ms - batch_ms - serial_ms) / emitted,
            "generation_tps_median_instrumented": statistics.median(
                row["generation_tps"] for row in selected
            ),
            "head_batch_attempts_median": statistics.median(
                row["exact_head_batch_attempts"] for row in selected
            ),
            "head_batch_successes_median": statistics.median(
                row["exact_head_batch_successes"] for row in selected
            ),
        }
    exact = result["modes"]["exact"]
    head = result["modes"]["exact_head"]
    result["target_time_ratio_head_over_exact"] = (
        head["target_eval_ms_per_emitted"] / exact["target_eval_ms_per_emitted"]
    )
    result["target_time_saved_ms_per_emitted"] = (
        exact["target_eval_ms_per_emitted"] - head["target_eval_ms_per_emitted"]
    )
    return result


def report(summary):
    exact = summary["modes"]["exact"]
    head = summary["modes"]["exact_head"]
    return (
        "# DSpark Exact-Head Ablation\n\n"
        "Instrumented diagnostic only; do not compare these t/s values with uninstrumented runs.\n\n"
        "| mode | target ms/emitted | layers | batch head | serial head | residual |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| exact | {exact['target_eval_ms_per_emitted']:.3f} | "
        f"{exact['exact_layer_ms_per_emitted']:.3f} | "
        f"{exact['exact_head_batch_ms_per_emitted']:.3f} | "
        f"{exact['exact_head_serial_ms_per_emitted']:.3f} | "
        f"{exact['target_residual_ms_per_emitted']:.3f} |\n"
        f"| exact_head | {head['target_eval_ms_per_emitted']:.3f} | "
        f"{head['exact_layer_ms_per_emitted']:.3f} | "
        f"{head['exact_head_batch_ms_per_emitted']:.3f} | "
        f"{head['exact_head_serial_ms_per_emitted']:.3f} | "
        f"{head['target_residual_ms_per_emitted']:.3f} |\n\n"
        f"- Target-time ratio, exact-head / exact: "
        f"{summary['target_time_ratio_head_over_exact']:.4f}x\n"
        f"- Target time saved: {summary['target_time_saved_ms_per_emitted']:.3f} ms/emitted\n"
        f"- Exact-head batch outcomes: {head['head_batch_successes_median']:.0f}/"
        f"{head['head_batch_attempts_median']:.0f} successful\n"
        f"- Instrumented generation t/s (context only): exact "
        f"{exact['generation_tps_median_instrumented']:.2f}, exact-head "
        f"{head['generation_tps_median_instrumented']:.2f}\n"
    )


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.prompt_file = args.prompt_file.resolve()
    check_inputs(args, root)
    for mode in ("exact", "exact_head"):
        print(f"{mode}: {command_text(args, mode)}")
    print("Stats-only ablation: both modes are instrumented; output must match byte-for-byte.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_dir or root / "speed-bench/local-runs" / f"head-ablation-{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": run_capture(["git", "rev-parse", "HEAD"], root),
        "platform": platform.platform(),
        "config": {"ctx": args.ctx, "tokens": args.tokens, "pairs": args.pairs,
                   "warmups": args.warmups, "cooldown": args.cooldown,
                   "temperature": 0, "seed": 1, "instrumented": True},
        "commands": {mode: command_text(args, mode) for mode in ("exact", "exact_head")},
        "prompt": {"path": str(args.prompt_file),
                   "sha256": sha256(args.prompt_file.read_bytes())},
        "cleared_environment_keys": cleared_env_keys(os.environ),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    reference = None
    for warmup in range(1, args.warmups + 1):
        for mode in ("exact", "exact_head"):
            _, output = execute(args, root, run_dir, f"warmup-{warmup:02d}", mode, reference)
            if reference is None:
                reference = output
            cooldown(args.cooldown)

    rows = []
    sequence = 0
    for pair in range(1, args.pairs + 1):
        order = ("exact", "exact_head") if pair % 2 else ("exact_head", "exact")
        for position, mode in enumerate(order, 1):
            sequence += 1
            row, output = execute(
                args, root, run_dir, f"measured-{sequence:02d}", mode, reference
            )
            if reference is None:
                reference = output
            row.update(sequence=sequence, pair=pair, position=position)
            rows.append(row)
            cooldown(args.cooldown)

    fields = (
        "sequence", "pair", "position", "mode", "prefill_tps", "generation_tps",
        "wall_seconds", "stdout_sha256", "stdout_file", "stderr_file",
    ) + tuple(sorted(INT_FIELDS | FLOAT_FIELDS))
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    summary = summarize(rows)
    text = report(summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + text.rstrip())
    print(f"Raw results: {run_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial raw files were retained.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError) as exc:
        print(f"ablation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
