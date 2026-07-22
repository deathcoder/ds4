#!/usr/bin/env python3
"""Validate ratio-128 causal multi-row attention heads against serial decode."""

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
HEAD_RE = re.compile(
    rb"^ds4: DSpark causal attention head observer layer=(\d+) ratio=(\d+) "
    rb"proposed=(\d+) row=(\d+) max=(\S+) rms=(\S+) rel_l2=(\S+) "
    rb"max_ulp=(\d+) first=(\d+) batch=(\S+) serial=(\S+) result=(\w+)$",
    re.MULTILINE,
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic-only layer-41 ratio-128 causal attention "
            "head shadow."
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
    parser.add_argument(
        "--serial-legacy",
        action="store_true",
        help=(
            "localize arithmetic drift by comparing against the historical "
            "one-row gathered FlashAttention route"
        ),
    )
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


def clean_env(layer=None, threshold="0.75", serial_legacy=False):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DS4_")}
    if layer is not None:
        env.update({
            "DS4_DSPARK_GPU_RUNTIME": "1",
            "DS4_DSPARK_MULTI_COMMIT": "1",
            "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS": "1",
            "DS4_DSPARK_CONFIDENCE_THRESHOLD": threshold,
            "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER": str(layer),
            "DS4_DSPARK_CAUSAL_ATTN_HEAD_OBSERVER_LAYER": str(layer),
        })
        if serial_legacy:
            env["DS4_METAL_DENSE_MIXED_GATHERED_LEGACY"] = "1"
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


def command_text(args, layer=None, threshold="0.75", serial_legacy=False):
    env = clean_env(layer, threshold, serial_legacy)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER",
        "DS4_DSPARK_CAUSAL_ATTN_HEAD_OBSERVER_LAYER",
        "DS4_METAL_DENSE_MIXED_GATHERED_LEGACY",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return (prefix + " " if prefix else "") + shlex.join(command(args, layer))


def parse_head_observer(stderr_data, layer=LAYER):
    if b"DSpark causal attention head observer" not in stderr_data:
        raise RuntimeError(f"layer {layer}: no causal attention head records")
    if any(
        line.startswith(b"ds4: DSpark causal attention head observer") and
        b"result=fallback" in line
        for line in stderr_data.splitlines()
    ):
        raise RuntimeError(f"layer {layer}: causal attention shadow fell back")

    proposals = []
    current = None
    for match in HEAD_RE.finditer(stderr_data):
        record_layer, ratio, proposed, row = (
            int(value) for value in match.groups()[:4]
        )
        result = match.group(12).decode("ascii")
        if record_layer != layer or ratio != 128:
            raise RuntimeError(f"layer {layer}: unexpected observer identity")
        if proposed < 2 or proposed > 5 or row >= proposed:
            raise RuntimeError(f"layer {layer}: invalid proposal row")
        if row == 0:
            if current is not None and current["rows"] != current["proposed"]:
                raise RuntimeError(f"layer {layer}: incomplete proposal rows")
            current = {"proposed": proposed, "rows": 0}
            proposals.append(current)
        if current is None or current["proposed"] != proposed or row != current["rows"]:
            raise RuntimeError(f"layer {layer}: non-contiguous proposal rows")
        if result != "exact":
            raise RuntimeError(
                f"layer {layer}: causal attention head {result} at row {row}"
            )
        current["rows"] += 1
    if not proposals or current["rows"] != current["proposed"]:
        raise RuntimeError(f"layer {layer}: incomplete proposal rows")
    return proposals


def parse_diagnostics(stderr_data, layer=LAYER):
    heads = parse_head_observer(stderr_data, layer)
    slab = slab_observer.parse_observer(stderr_data, layer)
    if len(heads) != len(slab["observer"]):
        raise RuntimeError(f"layer {layer}: head/slab record count mismatch")
    return {"heads": heads, "slab": slab}


def execute(
        args, root, run_dir, label, layer, reference=None, threshold="0.75",
        serial_legacy=False):
    stdout_path = run_dir / f"{label}.stdout"
    stderr_path = run_dir / f"{label}.stderr"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.run(
            command(args, layer),
            cwd=root,
            env=clean_env(layer, threshold, serial_legacy),
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
    parsed = None if layer is None else parse_diagnostics(stderr_data, layer)
    return stdout_data, {
        "label": label,
        "layer": layer,
        "threshold": None if layer is None else threshold,
        "wall_seconds": wall,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "proposal_records": 0 if parsed is None else len(parsed["heads"]),
        "head_rows": 0 if parsed is None else sum(
            item["rows"] for item in parsed["heads"]
        ),
    }


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "prompt_file"):
        setattr(args, name, getattr(args, name).resolve())
    for mode, threshold in SCHEDULES:
        print(
            f"{mode}: "
            f"{command_text(args, LAYER, threshold, args.serial_legacy)}"
        )
    print(f"fresh baseline: {command_text(args)}")
    print(
        "Diagnostic pass only: the serial verifier remains authoritative; "
        "throughput reporting is disabled."
    )
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    for name, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("prompt", args.prompt_file),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" +
            dirty
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/causal-attn-head-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline, baseline_row = execute(args, root, run_dir, "baseline", None)
    rows = [baseline_row]
    for mode, threshold in SCHEDULES:
        if args.cooldown:
            time.sleep(args.cooldown)
        _, row = execute(
            args,
            root,
            run_dir,
            mode,
            LAYER,
            baseline,
            threshold,
            args.serial_legacy,
        )
        row["mode"] = mode
        rows.append(row)

    fixed = next(row for row in rows if row.get("mode") == "fixed-k5")
    fixed_stderr = (run_dir / fixed["stderr_file"]).read_bytes()
    publications = slab_observer.parse_observer(
        fixed_stderr, LAYER
    )["publication"]
    if not any(item["accepted"] < item["proposed"] for item in publications):
        raise RuntimeError("fixed-K5 run did not exercise partial publication")

    summary = {
        "analysis": "dspark_causal_attention_head_observer",
        "result": "PASS",
        "layer": LAYER,
        "serial_route": (
            "legacy-gathered" if args.serial_legacy else "fused-gather"
        ),
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
            "layer": LAYER,
            "schedules": list(SCHEDULES),
            "fast_verifier": False,
            "instrumented": True,
            "serial_route": (
                "legacy-gathered" if args.serial_legacy else "fused-gather"
            ),
        },
        "paths": {
            "binary": str(args.binary),
            "model": str(args.model),
            "dspark_model": str(args.dspark_model),
            "prompt": str(args.prompt_file),
        },
        "commands": {
            "baseline": command_text(args),
            **{
                mode: command_text(
                    args, LAYER, threshold, args.serial_legacy
                )
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
    print("\n# DSpark Causal Attention Head Observer")
    print("\n- Result: PASS")
    print(
        "- Serial route: "
        + ("legacy gathered" if args.serial_legacy else "fused gather")
        + "."
    )
    for row in rows[1:]:
        print(
            f"- {row['mode']}: {row['proposal_records']} exact proposal "
            f"records and {row['head_rows']} exact head rows at layer 41."
        )
    print(f"Raw results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
