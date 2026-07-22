#!/usr/bin/env python3
"""Validate the opt-in one-layer exact causal-attention runtime."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import time

import run_dspark_proposal_slab_observer as slab_observer


LAYER = 41
SCHEDULES = (("scheduled", "0.75"), ("fixed-k5", "0"))
RUNTIME_RE = re.compile(
    rb"^ds4: DSpark causal attention runtime proposed=(\d+) layer=(\d+) "
    rb"attempts=(\d+) successes=(\d+) result=(\w+)$",
    re.MULTILINE,
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Validate the one-layer exact causal-attention runtime and its "
            "accepted-prefix cache publication."
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
    parser.add_argument("--tokens", type=int, default=32)
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
        parser.error("refusing to run without --confirm-ready")
    return args, root


def git_output(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False, text=True,
    ).stdout.strip()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_env(runtime=False, threshold="0.75"):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DS4_")}
    if runtime:
        env.update({
            "DS4_DSPARK_GPU_RUNTIME": "1",
            "DS4_DSPARK_MULTI_COMMIT": "1",
            "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS": "1",
            "DS4_DSPARK_CONFIDENCE_THRESHOLD": threshold,
            "DS4_DSPARK_CAUSAL_ATTN_RUNTIME_LAYER": str(LAYER),
            "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER": str(LAYER),
        })
    return env


def command(args, runtime=False):
    cmd = [
        str(args.binary), "--backend", "metal", "--model", str(args.model),
        "--ctx", str(args.ctx), "-n", str(args.tokens), "--nothink",
        "--temp", "0", "--seed", "1", "--prompt-file",
        str(args.prompt_file),
    ]
    if runtime:
        cmd.extend(("--dspark", str(args.dspark_model)))
    return cmd


def command_text(args, runtime=False, threshold="0.75"):
    env = clean_env(runtime, threshold)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_CAUSAL_ATTN_RUNTIME_LAYER",
        "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return (prefix + " " if prefix else "") + shlex.join(
        command(args, runtime)
    )


def parse_runtime(stderr_data, layer=LAYER):
    records = []
    for match in RUNTIME_RE.finditer(stderr_data):
        proposed, record_layer, attempts, successes = (
            int(value) for value in match.groups()[:4]
        )
        result = match.group(5).decode("ascii")
        if record_layer != layer or proposed < 2 or proposed > 5:
            raise RuntimeError("unexpected causal-attention runtime identity")
        if attempts != 1 or successes != 1 or result != "pass":
            raise RuntimeError("causal-attention runtime fell back")
        records.append({"proposed": proposed})
    if not records:
        raise RuntimeError("no causal-attention runtime records")
    return records


def parse_diagnostics(stderr_data, layer=LAYER):
    runtime = parse_runtime(stderr_data, layer)
    slab = slab_observer.parse_observer(stderr_data, layer)
    if len(runtime) != len(slab["observer"]):
        raise RuntimeError("runtime/proposal-slab record count mismatch")
    runtime_widths = [item["proposed"] for item in runtime]
    slab_widths = [item["proposed"] for item in slab["observer"]]
    if runtime_widths != slab_widths:
        raise RuntimeError("runtime/proposal-slab proposal order mismatch")
    return {"runtime": runtime, **slab}


def execute(
        args, root, run_dir, label, runtime, reference=None,
        threshold="0.75"):
    stdout_path = run_dir / f"{label}.stdout"
    stderr_path = run_dir / f"{label}.stderr"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.run(
            command(args, runtime), cwd=root,
            env=clean_env(runtime, threshold), stdout=stdout_file,
            stderr=stderr_file, check=False,
        )
    wall = time.monotonic() - started
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {proc.returncode}; see {stderr_path}"
        )
    if reference is not None and stdout_data != reference:
        raise RuntimeError(
            f"{label} output differs from fresh baseline; see {stdout_path}"
        )
    parsed = parse_diagnostics(stderr_data) if runtime else None
    return stdout_data, {
        "label": label,
        "threshold": threshold if runtime else None,
        "wall_seconds": wall,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "runtime_records": 0 if parsed is None else len(parsed["runtime"]),
        "observer_records": 0 if parsed is None else len(parsed["observer"]),
        "publication_records": 0 if parsed is None else len(parsed["publication"]),
        "publications": [] if parsed is None else parsed["publication"],
    }


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "prompt_file"):
        setattr(args, name, getattr(args, name).resolve())
    for mode, threshold in SCHEDULES:
        print(f"{mode}: {command_text(args, True, threshold)}")
    print(f"fresh baseline: {command_text(args)}")
    print(
        "Correctness diagnostic only: the candidate replaces attention at "
        "layer 41; throughput reporting is disabled."
    )
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    for label, path in (
        ("binary", args.binary), ("base model", args.model),
        ("DSpark model", args.dspark_model), ("prompt", args.prompt_file),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" +
            dirty
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/causal-attn-runtime-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline, baseline_row = execute(
        args, root, run_dir, "baseline", False
    )
    rows = [baseline_row]
    for mode, threshold in SCHEDULES:
        if args.cooldown:
            time.sleep(args.cooldown)
        _, row = execute(
            args, root, run_dir, mode, True, baseline, threshold
        )
        row["mode"] = mode
        rows.append(row)

    fixed = next(row for row in rows if row.get("mode") == "fixed-k5")
    if not any(
            item["accepted"] < item["proposed"]
            for item in fixed["publications"]):
        raise RuntimeError(
            "fixed-K5 run did not exercise partial accepted-prefix publication"
        )

    summary = {
        "analysis": "dspark_causal_attention_runtime",
        "result": "PASS",
        "layer": LAYER,
        "runs": rows,
    }
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": dirty,
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx, "tokens": args.tokens,
            "cooldown": args.cooldown, "layer": LAYER,
            "schedules": list(SCHEDULES), "instrumented": True,
            "fast_verifier": False,
        },
        "paths": {
            "binary": str(args.binary), "model": str(args.model),
            "dspark_model": str(args.dspark_model),
            "prompt": str(args.prompt_file),
        },
        "commands": {
            "baseline": command_text(args),
            **{
                mode: command_text(args, True, threshold)
                for mode, threshold in SCHEDULES
            },
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n# DSpark One-Layer Causal Attention Runtime")
    print("\n- Result: PASS")
    print(f"- Runtime layer: {LAYER} (ratio 128).")
    for row in rows[1:]:
        partial = sum(
            item["accepted"] < item["proposed"]
            for item in row["publications"]
        )
        print(
            f"- {row['mode']}: {row['runtime_records']} runtime records; "
            f"{row['observer_records']} exact state records; "
            f"{row['publication_records']} exact publications "
            f"({partial} partial)."
        )
    print(f"Raw results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
