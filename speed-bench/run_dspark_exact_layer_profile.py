#!/usr/bin/env python3
"""User-run synchronized stage profile for the exact DSpark verifier."""

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


PROFILE_RE = re.compile(
    rb"^ds4: metal layer stage part=decode layer=([0-9]+) "
    rb"pos=([0-9]+) tokens=([0-9]+) ([a-z0-9_]+)=([0-9.]+) ms$"
)
ATTENTION_STAGES = {
    "attn_hc_pre", "attn_norm", "q_path", "kv_path",
    "compressor_indexer", "attention", "attn_output", "attn_hc_post",
}
FFN_STAGES = {
    "ffn_hc_pre", "ffn_norm", "router", "shared_gate_up",
    "routed_moe", "shared_down", "ffn_hc_post",
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


def profile_env(layer=None):
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    env["DS4_DSPARK_GPU_RUNTIME"] = "1"
    env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    if layer is not None:
        env["DS4_METAL_DECODE_STAGE_PROFILE"] = "1"
        env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"] = str(layer)
    return env


def parse_layers(value):
    try:
        layers = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers or any(layer < 0 or layer >= 61 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be in the range 0..60")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must not contain duplicates")
    return layers


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Profile synchronized stages in representative exact verifier layers."
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
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,30,60"))
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


def command(args):
    return [
        str(args.binary), "--backend", "metal", "--model", str(args.model),
        "--ctx", str(args.ctx), "-n", str(args.tokens), "--temp", "0",
        "--seed", "1", "--prompt-file", str(args.prompt_file),
        "--dspark", str(args.dspark_model),
    ]


def command_text(args, layer=None):
    env = profile_env(layer)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME", "DS4_DSPARK_MULTI_COMMIT",
        "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
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


def parse_profile(data, expected_layer, path):
    rows = []
    for line in data.splitlines():
        match = PROFILE_RE.match(line)
        if not match:
            continue
        layer, pos, tokens, stage, elapsed = match.groups()
        row = {
            "layer": int(layer), "pos": int(pos), "tokens": int(tokens),
            "stage": stage.decode("ascii"), "ms": float(elapsed),
        }
        if row["layer"] != expected_layer:
            raise RuntimeError(
                f"unexpected profiled layer {row['layer']} in {path}; expected {expected_layer}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no decode-stage profile records found in {path}")
    unknown = sorted({row["stage"] for row in rows} - ATTENTION_STAGES - FFN_STAGES)
    if unknown:
        raise RuntimeError(f"unknown decode stages in {path}: {', '.join(unknown)}")
    return rows


def execute(args, root, run_dir, name, layer, reference):
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    print(f"[{name}] {command_text(args, layer)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args), cwd=root, env=profile_env(layer),
            stdout=stdout_fp, stderr=stderr_fp, check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit {completed.returncode}; see {stderr_path}")
    stdout_data = stdout_path.read_bytes()
    if reference is not None and stdout_data != reference:
        raise RuntimeError(f"{name} output differs from exact reference; see {stdout_path}")
    records = [] if layer is None else parse_profile(
        stderr_path.read_bytes(), layer, stderr_path
    )
    return stdout_data, records, {
        "name": name, "layer": "" if layer is None else layer,
        "wall_seconds": wall_seconds, "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name, "stderr_file": stderr_path.name,
    }


def summarize(records):
    summary = {"layers": {}}
    for layer in sorted({row["layer"] for row in records}):
        selected = [row for row in records if row["layer"] == layer]
        by_stage = {}
        for stage in sorted({row["stage"] for row in selected}):
            values = [row["ms"] for row in selected if row["stage"] == stage]
            by_stage[stage] = {
                "records": len(values), "total_ms": sum(values),
                "median_ms": statistics.median(values),
            }
        row_count = by_stage.get("attn_hc_pre", {}).get("records", 0)
        if row_count == 0:
            raise RuntimeError(f"layer {layer} has no attn_hc_pre records")
        total_ms = sum(item["total_ms"] for item in by_stage.values())
        attention_ms = sum(
            item["total_ms"] for stage, item in by_stage.items()
            if stage in ATTENTION_STAGES
        )
        ffn_ms = sum(
            item["total_ms"] for stage, item in by_stage.items()
            if stage in FFN_STAGES
        )
        summary["layers"][str(layer)] = {
            "profiled_rows": row_count,
            "total_ms_per_row": total_ms / row_count,
            "attention_ms_per_row": attention_ms / row_count,
            "ffn_ms_per_row": ffn_ms / row_count,
            "attention_share": attention_ms / total_ms,
            "ffn_share": ffn_ms / total_ms,
            "stages": by_stage,
        }
    return summary


def report(summary):
    lines = [
        "# DSpark Exact Layer Stage Profile", "",
        "Synchronized diagnostic only. Stage boundaries change scheduling; do not use these values as throughput measurements.",
        "", "| layer | rows | total ms/row | attention | FFN | attention share | FFN share |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in summary["layers"].items():
        lines.append(
            f"| {layer} | {item['profiled_rows']} | {item['total_ms_per_row']:.3f} | "
            f"{item['attention_ms_per_row']:.3f} | {item['ffn_ms_per_row']:.3f} | "
            f"{item['attention_share'] * 100.0:.1f}% | {item['ffn_share'] * 100.0:.1f}% |"
        )
    lines.extend(["", "Largest stages by total synchronized time:"])
    for layer, item in summary["layers"].items():
        stages = sorted(
            item["stages"].items(), key=lambda pair: pair[1]["total_ms"], reverse=True
        )[:5]
        text = ", ".join(
            f"{stage} {values['total_ms'] / item['profiled_rows']:.3f} ms/row"
            for stage, values in stages
        )
        lines.append(f"- Layer {layer}: {text}")
    return "\n".join(lines) + "\n"


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.prompt_file = args.prompt_file.resolve()
    check_inputs(args, root)
    print(f"reference: {command_text(args)}")
    for layer in args.layers:
        print(f"layer {layer}: {command_text(args, layer)}")
    print("Profiled output must match the unprofiled exact reference byte-for-byte.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_dir or root / "speed-bench/local-runs" / f"layer-profile-{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": run_capture(["git", "rev-parse", "HEAD"], root),
        "platform": platform.platform(),
        "config": {"ctx": args.ctx, "tokens": args.tokens,
                   "layers": args.layers, "cooldown": args.cooldown,
                   "temperature": 0, "seed": 1, "synchronized_profile": True},
        "commands": {"reference": command_text(args)} | {
            f"layer_{layer}": command_text(args, layer) for layer in args.layers
        },
        "prompt": {"path": str(args.prompt_file),
                   "sha256": sha256(args.prompt_file.read_bytes())},
        "cleared_environment_keys": cleared_env_keys(os.environ),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    reference, _, reference_run = execute(
        args, root, run_dir, "reference", None, None
    )
    cooldown(args.cooldown)
    records = []
    runs = [reference_run]
    for layer in args.layers:
        _, layer_records, run = execute(
            args, root, run_dir, f"layer-{layer:02d}", layer, reference
        )
        records.extend(layer_records)
        runs.append(run)
        cooldown(args.cooldown)

    with (run_dir / "stages.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=("layer", "pos", "tokens", "stage", "ms"))
        writer.writeheader()
        writer.writerows(records)
    with (run_dir / "runs.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=("name", "layer", "wall_seconds", "stdout_sha256",
                            "stdout_file", "stderr_file")
        )
        writer.writeheader()
        writer.writerows(runs)
    summary = summarize(records)
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
    print(f"Raw stages: {run_dir / 'stages.csv'}")
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
