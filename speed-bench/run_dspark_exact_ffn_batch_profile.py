#!/usr/bin/env python3
"""User-run stats profile for default-exact versus exact-FFN DSpark."""

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
    "emitted",
    "target_evals",
    "target_eval_tokens",
    "target_evals_avoided",
    "batch_attempts",
    "batch_full",
    "batch_partial",
    "batch_fallbacks",
    "exact_ffn_batch_attempts",
    "exact_ffn_batch_successes",
}
FLOAT_FIELDS = {
    "avg_depth",
    "target_eval_ms",
    "exact_layer_ms",
    "exact_head_batch_ms",
    "exact_head_serial_ms",
    "generation_sidecar_ms",
}
INSTRUMENTATION_MARKERS = ("PROFILE", "TRACE", "DUMP", "TIMING")
MODES = ("exact", "exact_ffn")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


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


def mode_env(mode):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    env["DS4_DSPARK_GPU_RUNTIME"] = "1"
    env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    env["DS4_DSPARK_GPU_RUNTIME_STATS"] = "1"
    if mode == "exact_ffn":
        env["DS4_DSPARK_EXACT_FFN_BATCH"] = "1"
    return env


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Profile instrumented default-exact and exact-FFN DSpark."
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
    parser.add_argument("--ctx", type=int, default=256)
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
        str(args.binary),
        "--backend",
        "metal",
        "--model",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--ctx",
        str(args.ctx),
        "--nothink",
        "--temp",
        "0",
        "--seed",
        "1",
        "-n",
        str(args.tokens),
        "--dspark",
        str(args.dspark_model),
    ]


def command_text(args, mode):
    env = mode_env(mode)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_STATS",
        "DS4_DSPARK_EXACT_FFN_BATCH",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args))


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
    dirty = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n"
            + dirty
        )


def parse_timing(data, path):
    matches = TIMING_RE.findall(data)
    if not matches:
        raise RuntimeError(f"timing line not found in {path}")
    prefill, generation = (float(value) for value in matches[-1])
    if prefill <= 0.0 or generation <= 0.0:
        raise RuntimeError(f"non-positive throughput in {path}")
    return prefill, generation


def parse_stats(data, path):
    records = [
        line[len(STATS_PREFIX):]
        for line in data.splitlines()
        if line.startswith(STATS_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one DSpark stats record in {path}, found {len(records)}"
        )
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


def validate_mode_stats(row, path):
    attempts = row["exact_ffn_batch_attempts"]
    successes = row["exact_ffn_batch_successes"]
    if row["mode"] == "exact":
        if attempts != 0 or successes != 0:
            raise RuntimeError(f"default exact unexpectedly used FFN batching in {path}")
        return
    if attempts <= 0:
        raise RuntimeError(f"exact FFN candidate was never attempted in {path}")
    if successes > attempts:
        raise RuntimeError(f"invalid exact FFN outcomes in {path}")


def execute(args, root, run_dir, label, mode, reference):
    stdout_path = run_dir / f"{label}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{mode}.stderr"
    print(f"[{label}] {mode}: {command_text(args, mode)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args),
            cwd=root,
            env=mode_env(mode),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
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
        "mode": mode,
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": wall_seconds,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    row.update(parse_stats(stderr_data, stderr_path))
    validate_mode_stats(row, stderr_path)
    return row, stdout_data


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def summarize(rows):
    result = {"modes": {}}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]

        def median(field):
            return statistics.median(row[field] for row in selected)

        def per_emitted(field):
            return statistics.median(row[field] / row["emitted"] for row in selected)

        target = per_emitted("target_eval_ms")
        layers = per_emitted("exact_layer_ms")
        head = statistics.median(
            (row["exact_head_batch_ms"] + row["exact_head_serial_ms"])
            / row["emitted"]
            for row in selected
        )
        result["modes"][mode] = {
            "samples": len(selected),
            "target_eval_ms_per_emitted": target,
            "exact_layer_ms_per_emitted": layers,
            "exact_head_ms_per_emitted": head,
            "target_residual_ms_per_emitted": target - layers - head,
            "generation_sidecar_ms_per_emitted": per_emitted(
                "generation_sidecar_ms"
            ),
            "target_evals_per_emitted": statistics.median(
                row["target_evals"] / row["emitted"] for row in selected
            ),
            "target_tokens_per_eval": statistics.median(
                row["target_eval_tokens"] / row["target_evals"] for row in selected
            ),
            "average_depth_median": median("avg_depth"),
            "batch_fallbacks_median": median("batch_fallbacks"),
            "ffn_batch_attempts_median": median("exact_ffn_batch_attempts"),
            "ffn_batch_successes_median": median("exact_ffn_batch_successes"),
            "generation_tps_median_instrumented": median("generation_tps"),
        }
    exact = result["modes"]["exact"]
    candidate = result["modes"]["exact_ffn"]
    result["target_time_ratio_candidate_over_exact"] = (
        candidate["target_eval_ms_per_emitted"]
        / exact["target_eval_ms_per_emitted"]
    )
    result["target_time_saved_ms_per_emitted"] = (
        exact["target_eval_ms_per_emitted"]
        - candidate["target_eval_ms_per_emitted"]
    )
    result["layer_time_saved_ms_per_emitted"] = (
        exact["exact_layer_ms_per_emitted"]
        - candidate["exact_layer_ms_per_emitted"]
    )
    return result


def report(summary):
    exact = summary["modes"]["exact"]
    candidate = summary["modes"]["exact_ffn"]
    return (
        "# DSpark Exact FFN Batch Profile\n\n"
        "Instrumented synchronized diagnostic only; do not compare these t/s values "
        "with uninstrumented runs.\n\n"
        "| mode | target ms/emitted | layers | head | residual | sidecar |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| exact | {exact['target_eval_ms_per_emitted']:.3f} | "
        f"{exact['exact_layer_ms_per_emitted']:.3f} | "
        f"{exact['exact_head_ms_per_emitted']:.3f} | "
        f"{exact['target_residual_ms_per_emitted']:.3f} | "
        f"{exact['generation_sidecar_ms_per_emitted']:.3f} |\n"
        f"| exact_ffn | {candidate['target_eval_ms_per_emitted']:.3f} | "
        f"{candidate['exact_layer_ms_per_emitted']:.3f} | "
        f"{candidate['exact_head_ms_per_emitted']:.3f} | "
        f"{candidate['target_residual_ms_per_emitted']:.3f} | "
        f"{candidate['generation_sidecar_ms_per_emitted']:.3f} |\n\n"
        f"- Target-time ratio, exact FFN / exact: "
        f"{summary['target_time_ratio_candidate_over_exact']:.4f}x\n"
        f"- Target time saved: "
        f"{summary['target_time_saved_ms_per_emitted']:.3f} ms/emitted\n"
        f"- Layer time saved: "
        f"{summary['layer_time_saved_ms_per_emitted']:.3f} ms/emitted\n"
        f"- Exact FFN outcomes: {candidate['ffn_batch_successes_median']:.0f}/"
        f"{candidate['ffn_batch_attempts_median']:.0f} completed; "
        f"{candidate['batch_fallbacks_median']:.0f} verifier fallbacks\n"
        f"- Acceptance depth: exact {exact['average_depth_median']:.3f}, "
        f"exact FFN {candidate['average_depth_median']:.3f}\n"
        f"- Target positions / eval: exact {exact['target_tokens_per_eval']:.3f}, "
        f"exact FFN {candidate['target_tokens_per_eval']:.3f}\n"
        f"- Instrumented generation t/s (context only): exact "
        f"{exact['generation_tps_median_instrumented']:.2f}, exact FFN "
        f"{candidate['generation_tps_median_instrumented']:.2f}\n"
    )


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.prompt_file = args.prompt_file.resolve()
    check_inputs(args, root)
    for mode in MODES:
        print(f"{mode}: {command_text(args, mode)}")
    print(
        "Stats-only profile: both modes are instrumented; output must match "
        "byte-for-byte."
    )
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_dir
        or root / "speed-bench/local-runs" / f"ffn-profile-{stamp}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": run_capture(["git", "rev-parse", "HEAD"], root),
        "git_status_tracked": run_capture(
            ["git", "status", "--porcelain", "--untracked-files=no"], root
        ),
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "pairs": args.pairs,
            "warmups": args.warmups,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "instrumented": True,
        },
        "commands": {mode: command_text(args, mode) for mode in MODES},
        "prompt": {
            "path": str(args.prompt_file),
            "sha256": sha256(args.prompt_file.read_bytes()),
        },
        "cleared_environment_keys": cleared_env_keys(os.environ),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    reference = None
    for warmup in range(1, args.warmups + 1):
        for mode in MODES:
            _, output = execute(
                args, root, run_dir, f"warmup-{warmup:02d}", mode, reference
            )
            if reference is None:
                reference = output
            cooldown(args.cooldown)

    rows = []
    sequence = 0
    for pair in range(1, args.pairs + 1):
        order = MODES if pair % 2 else tuple(reversed(MODES))
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
        print(f"profile failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
