#!/usr/bin/env python3
"""Profile the retained one-row dense-mixed FlashAttention route."""

import argparse
from collections import Counter
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import time

import run_dspark_exact_attention_transition_profile as transition
import run_dspark_exact_layer_profile as layer_profile


LAYER = 42
STAGES = (
    "linearize_raw",
    "copy_raw",
    "copy_comp",
    "mask_fill",
    "mask_comp_copy",
    "pad",
    "attention_vec",
    "attention_reduce",
)
FLASH_RE = re.compile(
    rb"^ds4: Metal FlashAttention prefill stage "
    rb"mode=gathered_decode tokens=1 comp=(\d+) keys=(\d+) "
    rb"heads=(\d+) dim=(\d+) window=(\d+) ratio=(\d+) "
    rb"([a-z_]+)=([0-9]+(?:\.[0-9]+)?) ms$"
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile cache staging, FlashAttention, and reduction inside the "
            "retained layer-42 dense-mixed one-row route."
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
        "--prompt-file",
        type=Path,
        default=root / "speed-bench/issue468/code_8k.txt",
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to profile without --confirm-ready")
    return args, root


def profile_env(layer=None):
    env = transition.profile_env(layer)
    env.pop("DS4_METAL_FLASH_ATTN_STAGE_PROFILE", None)
    if layer is not None:
        env["DS4_METAL_DENSE_MIXED_GATHERED_LEGACY"] = "1"
        env["DS4_METAL_FLASH_ATTN_GATHERED_PROFILE"] = "1"
    return env


def command(args):
    return [
        str(args.binary),
        "--backend", "metal",
        "--model", str(args.model),
        "--ctx", str(args.ctx),
        "-n", str(args.tokens),
        "--temp", "0",
        "--seed", "1",
        "--prompt-file", str(args.prompt_file),
        "--dspark", str(args.dspark_model),
    ]


def command_text(args, layer=None):
    env = profile_env(layer)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
        "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
        "DS4_DSPARK_EXACT_ATTENTION_PROFILE",
        "DS4_METAL_FLASH_ATTN_GATHERED_PROFILE",
        "DS4_METAL_DENSE_MIXED_GATHERED_LEGACY",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args))


def parse_flash(data, path):
    rows = []
    for line in data.splitlines():
        match = FLASH_RE.match(line)
        if not match:
            continue
        comp, keys, heads, dim, window, ratio, stage, elapsed = match.groups()
        row = {
            "call": 0,
            "comp": int(comp),
            "keys": int(keys),
            "heads": int(heads),
            "dim": int(dim),
            "window": int(window),
            "ratio": int(ratio),
            "stage": stage.decode("ascii"),
            "ms": float(elapsed),
        }
        if row["stage"] not in STAGES:
            raise RuntimeError(
                f"unknown gathered FlashAttention stage {row['stage']} in {path}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no gathered FlashAttention records found in {path}")

    calls = []
    current = []
    for row in rows:
        if row["stage"] == "linearize_raw":
            if current:
                raise RuntimeError(f"new gathered call before prior completion in {path}")
            current = [row]
            continue
        if row["stage"] == "copy_raw":
            if current and current[-1]["stage"] != "linearize_raw":
                raise RuntimeError(f"duplicate gathered call start in {path}")
            if not current:
                current = []
            current.append(row)
            continue
        if not current:
            raise RuntimeError(f"gathered stage {row['stage']} precedes copy_raw in {path}")
        current.append(row)
        if row["stage"] == "attention_reduce":
            calls.append(current)
            current = []
    if current:
        raise RuntimeError(f"incomplete gathered FlashAttention call in {path}")

    assigned = []
    for call_index, call in enumerate(calls, start=1):
        stages = [row["stage"] for row in call]
        required = {
            "copy_raw", "copy_comp", "mask_fill",
            "attention_vec", "attention_reduce",
        }
        missing = sorted(required - set(stages))
        if missing:
            raise RuntimeError(
                f"gathered call {call_index} missing stages: {', '.join(missing)}"
            )
        if len(stages) != len(set(stages)):
            raise RuntimeError(f"gathered call {call_index} repeats a stage")
        signature = {
            (row["comp"], row["keys"], row["heads"], row["dim"],
             row["window"], row["ratio"])
            for row in call
        }
        if len(signature) != 1:
            raise RuntimeError(f"gathered call {call_index} changes shape")
        for row in call:
            row["call"] = call_index
            assigned.append(row)
    return assigned


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(exact_records, flash_records):
    attention = [
        row for row in exact_records
        if row["part"] == "attention"
    ]
    dense = [row for row in attention if row["stage"] == "dense_mixed"]
    if not dense:
        raise RuntimeError("profile contains no dense-mixed exact-attention rows")
    if len({row["call"] for row in flash_records}) != len(dense):
        raise RuntimeError(
            f"gathered calls {len({row['call'] for row in flash_records})} "
            f"do not match dense rows {len(dense)}"
        )

    call_totals = {}
    for row in flash_records:
        call_totals[row["call"]] = call_totals.get(row["call"], 0.0) + row["ms"]
    total_values = list(call_totals.values())
    stages = {}
    for stage in STAGES:
        values = [row["ms"] for row in flash_records if row["stage"] == stage]
        stages[stage] = {
            "occurrences": len(values),
            "median_ms": statistics.median(values) if values else 0.0,
            "mean_ms": statistics.mean(values) if values else 0.0,
            "p90_ms": percentile(values, 0.9) if values else 0.0,
            "max_ms": max(values) if values else 0.0,
            "mean_contribution_ms_per_call": (
                sum(values) / len(total_values) if total_values else 0.0
            ),
        }
    mean_total = statistics.mean(total_values)
    for item in stages.values():
        item["mean_contribution_share"] = (
            item["mean_contribution_ms_per_call"] / mean_total
            if mean_total else 0.0
        )
    return {
        "analysis": "dspark_dense_mixed_flash_attention_profile",
        "layer": LAYER,
        "dense_rows": len(dense),
        "flash_calls": len(total_values),
        "raw_rows": sum(row["stage"] == "raw" for row in attention),
        "sparse_rows": sum(
            row["stage"] == "sparse_indexed" for row in attention
        ),
        "call_total": {
            "median_ms": statistics.median(total_values),
            "mean_ms": mean_total,
            "p90_ms": percentile(total_values, 0.9),
            "max_ms": max(total_values),
        },
        "stages": stages,
        "key_range": {
            "min": min(row["keys"] for row in flash_records),
            "max": max(row["keys"] for row in flash_records),
        },
        "comp_range": {
            "min": min(row["comp"] for row in flash_records),
            "max": max(row["comp"] for row in flash_records),
        },
        "stage_occurrences": dict(Counter(row["stage"] for row in flash_records)),
    }


def render_report(summary):
    lines = [
        "# DSpark Dense-Mixed FlashAttention Stage Profile",
        "",
        "Synchronized diagnostic only. Stage boundaries change Metal scheduling; do not use these values as throughput measurements.",
        "The profiled output matched the uninstrumented exact reference byte-for-byte.",
        "",
        f"- Dense-mixed rows: {summary['dense_rows']}.",
        f"- Gathered FlashAttention calls: {summary['flash_calls']}.",
        f"- Key range: {summary['key_range']['min']}-{summary['key_range']['max']}.",
        f"- Compressed-row range: {summary['comp_range']['min']}-{summary['comp_range']['max']}.",
        f"- Mean synchronized call total: {summary['call_total']['mean_ms']:.3f} ms.",
        "",
        "| stage | occurrences | median ms | mean contribution/call | share |",
        "|:---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        item = summary["stages"][stage]
        lines.append(
            f"| {stage} | {item['occurrences']} | {item['median_ms']:.3f} | "
            f"{item['mean_contribution_ms_per_call']:.3f} | "
            f"{item['mean_contribution_share']:.1%} |"
        )
    lines.extend([
        "",
        "Interpretation limits:",
        "",
        "- Synchronization inflates absolute times and suppresses normal command overlap.",
        "- Optional raw-ring linearization and padding occur only on calls that require them.",
        "- The useful result is ownership across cache staging, attention, and reduction, not absolute latency.",
        "- No timed throughput benchmark or runtime candidate is enabled.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_metadata(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime": dt.datetime.fromtimestamp(
            stat.st_mtime
        ).astimezone().isoformat(),
    }


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "prompt_file"):
        setattr(args, name, getattr(args, name).resolve())
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/dense-mixed-flash-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    print(f"reference: {command_text(args)}")
    print(f"layer {LAYER}: {command_text(args, LAYER)}")
    print("One reference and one synchronized layer-42 profile process.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("prompt", args.prompt_file),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    dirty = layer_profile.run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n"
            + dirty
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": layer_profile.run_capture(
            ["git", "rev-parse", "HEAD"], root
        ),
        "git_status_tracked": dirty,
        "experiment": "dspark_dense_mixed_flash_attention_profile",
        "platform": platform.platform(),
        "config": {
            "layer": LAYER,
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "synchronized_profile": True,
            "timed_throughput": False,
        },
        "binary": file_metadata(args.binary),
        "base_model": file_metadata(args.model),
        "dspark_model": file_metadata(args.dspark_model),
        "prompt": file_metadata(args.prompt_file),
        "commands": {
            "reference": command_text(args),
            "profile": command_text(args, LAYER),
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    runs = []
    reference_path = run_dir / "reference.stdout"
    reference_err = run_dir / "reference.stderr"
    print(f"[reference] {command_text(args)}", flush=True)
    started = time.monotonic()
    with reference_path.open("wb") as out, reference_err.open("wb") as err:
        completed = subprocess.run(
            command(args),
            cwd=root,
            env=transition.profile_env(None),
            stdout=out,
            stderr=err,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"reference failed; see {reference_err}")
    reference = reference_path.read_bytes()
    runs.append({
        "name": "reference",
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": layer_profile.sha256(reference),
        "stdout_file": reference_path.name,
        "stderr_file": reference_err.name,
    })

    if args.cooldown:
        print(f"cooldown: {args.cooldown:g}s", flush=True)
        time.sleep(args.cooldown)

    stdout_path = run_dir / "layer-42.stdout"
    stderr_path = run_dir / "layer-42.stderr"
    print(f"[layer-42] {command_text(args, LAYER)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            command(args),
            cwd=root,
            env=profile_env(LAYER),
            stdout=out,
            stderr=err,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"profile failed; see {stderr_path}")
    output = stdout_path.read_bytes()
    if output != reference:
        raise RuntimeError("profile output differs from exact reference")
    stderr_data = stderr_path.read_bytes()
    exact_records = transition.parse_profile(
        stderr_data, LAYER, stderr_path
    )
    flash_records = parse_flash(stderr_data, stderr_path)
    summary = summarize(exact_records, flash_records)
    report = render_report(summary)
    runs.append({
        "name": "layer-42",
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": layer_profile.sha256(output),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    })

    write_csv(
        run_dir / "flash_stages.csv",
        ("call", "comp", "keys", "heads", "dim", "window", "ratio", "stage", "ms"),
        flash_records,
    )
    write_csv(
        run_dir / "runs.csv",
        ("name", "wall_seconds", "stdout_sha256", "stdout_file", "stderr_file"),
        runs,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["git_status_tracked_after"] = layer_profile.run_capture(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + report.rstrip())
    print(f"Raw stages: {run_dir / 'flash_stages.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
