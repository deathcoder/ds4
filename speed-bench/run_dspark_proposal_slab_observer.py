#!/usr/bin/env python3
"""Validate isolated proposal-prefix state against the exact Metal verifier."""

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


LAYERS = (41, 42)
THRESHOLD = "0.75"
OBSERVER_RE = re.compile(
    rb"^ds4: DSpark proposal slab observer layer=(\d+) ratio=(\d+) "
    rb"proposed=(\d+) raw_rows=(\d+)/(\d+) attn_prefixes=(\d+)/(\d+) "
    rb"index_prefixes=(\d+)/(\d+) counters=(\d+)/(\d+) result=(\w+)$",
    re.MULTILINE,
)
PUBLICATION_RE = re.compile(
    rb"^ds4: DSpark proposal slab publication layer=(\d+) ratio=(\d+) "
    rb"proposed=(\d+) accepted=(\d+) prepared=(\w+) result=(\w+)$",
    re.MULTILINE,
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic-only proposal-slab observer at one ratio-128 "
            "and one ratio-4 target layer."
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
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    ).stdout.strip()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_env(layer=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DS4_")}
    if layer is not None:
        env.update({
            "DS4_DSPARK_GPU_RUNTIME": "1",
            "DS4_DSPARK_MULTI_COMMIT": "1",
            "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS": "1",
            "DS4_DSPARK_CONFIDENCE_THRESHOLD": THRESHOLD,
            "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER": str(layer),
        })
    return env


def command(args, layer=None):
    cmd = [
        str(args.binary),
        "--backend", "metal",
        "--model", str(args.model),
        "--ctx", str(args.ctx),
        "-n", str(args.tokens),
        "--nothink",
        "--temp", "0",
        "--seed", "1",
        "--prompt-file", str(args.prompt_file),
    ]
    if layer is not None:
        cmd.extend(("--dspark", str(args.dspark_model)))
    return cmd


def command_text(args, layer=None):
    env = clean_env(layer)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return (prefix + " " if prefix else "") + shlex.join(command(args, layer))


def parse_observer(stderr_data, layer):
    if b"DSpark proposal slab observer" not in stderr_data:
        raise RuntimeError(f"layer {layer}: no proposal-slab observer records")
    observer = []
    for match in OBSERVER_RE.finditer(stderr_data):
        values = tuple(int(value) for value in match.groups()[:-1])
        result = match.group(12).decode("ascii")
        (record_layer, ratio, proposed, raw_ok, raw_total, attn_ok,
         attn_total, index_ok, index_total, counter_ok,
         counter_total) = values
        if record_layer != layer or ratio not in (4, 128):
            raise RuntimeError(f"layer {layer}: unexpected observer identity")
        if proposed < 2 or proposed > 5:
            raise RuntimeError(f"layer {layer}: invalid proposal width {proposed}")
        if result != "exact" or raw_ok != raw_total or attn_ok != attn_total:
            raise RuntimeError(f"layer {layer}: observer reported state drift")
        if index_ok != index_total or counter_ok != counter_total:
            raise RuntimeError(f"layer {layer}: observer reported index/counter drift")
        observer.append({"ratio": ratio, "proposed": proposed})
    publication = []
    for match in PUBLICATION_RE.finditer(stderr_data):
        record_layer, ratio, proposed, accepted = (
            int(value) for value in match.groups()[:4]
        )
        prepared = match.group(5).decode("ascii")
        result = match.group(6).decode("ascii")
        if record_layer != layer or ratio not in (4, 128):
            raise RuntimeError(f"layer {layer}: unexpected publication identity")
        if not 1 <= accepted <= proposed:
            raise RuntimeError(f"layer {layer}: invalid publication width")
        if prepared != "exact" or result != "exact":
            raise RuntimeError(f"layer {layer}: publication reported state drift")
        publication.append({"proposed": proposed, "accepted": accepted})
    if not observer or not publication:
        raise RuntimeError(f"layer {layer}: incomplete observer/publication records")
    if any(
        line.startswith(b"ds4: DSpark proposal slab observer") and
        b"result=fallback" in line
        for line in stderr_data.splitlines()
    ):
        raise RuntimeError(f"layer {layer}: observer fell back")
    return {"observer": observer, "publication": publication}


def execute(args, root, run_dir, label, layer, reference=None):
    stdout_path = run_dir / f"{label}.stdout"
    stderr_path = run_dir / f"{label}.stderr"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.run(
            command(args, layer),
            cwd=root,
            env=clean_env(layer),
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
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
    parsed = None if layer is None else parse_observer(stderr_data, layer)
    return stdout_data, {
        "label": label,
        "layer": layer,
        "wall_seconds": wall,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "observer_records": 0 if parsed is None else len(parsed["observer"]),
        "publication_records": 0 if parsed is None else len(parsed["publication"]),
        "ratios": [] if parsed is None else sorted({
            item["ratio"] for item in parsed["observer"]
        }),
    }


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "prompt_file"):
        setattr(args, name, getattr(args, name).resolve())
    for layer in LAYERS:
        print(f"layer {layer}: {command_text(args, layer)}")
    print(f"fresh baseline: {command_text(args)}")
    print(
        "Diagnostic pass only: fast verification, stats, profilers, and "
        "throughput reporting are disabled."
    )
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
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" +
            dirty
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/proposal-slab-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline, baseline_row = execute(
        args, root, run_dir, "baseline", None
    )
    rows = [baseline_row]
    for layer in LAYERS:
        if args.cooldown:
            time.sleep(args.cooldown)
        _, row = execute(
            args, root, run_dir, f"layer-{layer}", layer, baseline
        )
        rows.append(row)

    summary = {
        "analysis": "dspark_proposal_slab_observer",
        "result": "PASS",
        "layers": list(LAYERS),
        "threshold": THRESHOLD,
        "runs": rows,
    }
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": dirty,
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "layers": list(LAYERS),
            "confidence_threshold": THRESHOLD,
            "fast_verifier": False,
            "instrumented": True,
        },
        "paths": {
            "binary": str(args.binary),
            "model": str(args.model),
            "dspark_model": str(args.dspark_model),
            "prompt": str(args.prompt_file),
        },
        "commands": {
            "baseline": command_text(args),
            **{f"layer-{layer}": command_text(args, layer) for layer in LAYERS},
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n# DSpark Proposal-Slab Observer")
    print("\n- Result: PASS")
    for row in rows[1:]:
        print(
            f"- Layer {row['layer']}: {row['observer_records']} preparation "
            f"records, {row['publication_records']} publication records, "
            f"ratios {row['ratios']}."
        )
    print(f"Raw results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
