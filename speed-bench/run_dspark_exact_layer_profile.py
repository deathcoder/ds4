#!/usr/bin/env python3
"""User-run synchronized component profile for the exact DSpark verifier."""

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
    rb"^ds4: metal layer stage part=([a-z]+) layer=([0-9]+) "
    rb"pos=([0-9]+) tokens=([0-9]+) ([a-z0-9_]+)=([0-9.]+) ms$"
)
LAYER_COUNT_RE = re.compile(r"^layers:\s+([0-9]+)\s*$", re.MULTILINE)
EXACT_STAGES = (
    "attention_pre_batch",
    "attention_tail_serial",
    "ffn_batch",
)
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
        env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
        env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(layer)
    return env


def parse_layers(value):
    try:
        layers = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be non-negative")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must not contain duplicates")
    return layers


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Profile promoted exact-runtime components in representative layers."
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
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if args.output_dir and args.resume_dir:
        parser.error("--output-dir and --resume-dir are mutually exclusive")
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
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
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


def inspect_layer_count(args, root):
    completed = subprocess.run(
        [str(args.binary), "--inspect", "--model", str(args.model)],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"model inspection failed: {detail}")
    match = LAYER_COUNT_RE.search(completed.stdout)
    if not match:
        raise RuntimeError("model inspection did not report a layer count")
    count = int(match.group(1))
    if count <= 0:
        raise RuntimeError(f"invalid inspected layer count: {count}")
    return count


def parse_profile(data, expected_layer, path):
    rows = []
    for line in data.splitlines():
        match = PROFILE_RE.match(line)
        if not match:
            continue
        part, layer, pos, tokens, stage, elapsed = match.groups()
        row = {
            "part": part.decode("ascii"),
            "layer": int(layer), "pos": int(pos), "tokens": int(tokens),
            "stage": stage.decode("ascii"), "ms": float(elapsed),
        }
        if row["layer"] != expected_layer:
            raise RuntimeError(
                f"unexpected profiled layer {row['layer']} in {path}; expected {expected_layer}"
            )
        if row["part"] == "exact":
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no exact-layer profile records found in {path}")
    unknown = sorted({row["stage"] for row in rows} - set(EXACT_STAGES))
    if unknown:
        raise RuntimeError(f"unknown exact-layer stages in {path}: {', '.join(unknown)}")
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


def load_existing(run_dir, layer, reference):
    name = f"layer-{layer:02d}"
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    if not stdout_path.is_file() or not stderr_path.is_file():
        return None
    stdout_data = stdout_path.read_bytes()
    if stdout_data != reference:
        raise RuntimeError(f"retained {name} output differs from exact reference")
    records = parse_profile(stderr_path.read_bytes(), layer, stderr_path)
    run = {
        "name": name, "layer": layer, "wall_seconds": "",
        "stdout_sha256": sha256(stdout_data), "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    return records, run


def validate_resume(run_dir, args):
    metadata_path = run_dir / "metadata.start.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"resume directory has no start metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read resume metadata: {exc}") from exc
    config = metadata.get("config", {})
    if config.get("ctx") != args.ctx or config.get("tokens") != args.tokens:
        raise RuntimeError("resume ctx/tokens do not match the retained run")
    if tuple(config.get("components", ())) != EXACT_STAGES:
        raise RuntimeError("resume profile component contract is incompatible")
    prompt = metadata.get("prompt", {})
    if prompt.get("sha256") != sha256(args.prompt_file.read_bytes()):
        raise RuntimeError("resume prompt does not match the retained run")
    commands = metadata.get("commands", {})
    if commands.get("reference") != command_text(args):
        raise RuntimeError("resume reference command does not match the retained run")
    return metadata


def summarize(records):
    summary = {"layers": {}}
    for layer in sorted({row["layer"] for row in records}):
        selected = [row for row in records if row["layer"] == layer]
        by_stage = {}
        for stage in EXACT_STAGES:
            stage_rows = [row for row in selected if row["stage"] == stage]
            values = [row["ms"] / row["tokens"] for row in stage_rows]
            if not values:
                raise RuntimeError(f"layer {layer} has no {stage} records")
            ordered = sorted(values)
            by_stage[stage] = {
                "batches": len(values),
                "proposal_rows": sum(row["tokens"] for row in stage_rows),
                "total_ms": sum(row["ms"] for row in stage_rows),
                "mean_ms_per_row": statistics.mean(values),
                "median_ms_per_row": statistics.median(values),
                "p90_ms_per_row": ordered[int(0.9 * (len(ordered) - 1))],
                "max_ms_per_row": max(values),
            }
        batch_counts = {item["batches"] for item in by_stage.values()}
        row_counts = {item["proposal_rows"] for item in by_stage.values()}
        batch_signatures = {
            tuple(sorted(
                (row["pos"], row["tokens"])
                for row in selected if row["stage"] == stage
            ))
            for stage in EXACT_STAGES
        }
        if (len(batch_counts) != 1 or len(row_counts) != 1 or
                len(batch_signatures) != 1):
            raise RuntimeError(f"layer {layer} has incomplete exact-layer batches")
        typical_total_ms = sum(
            item["median_ms_per_row"] for item in by_stage.values()
        )
        outliers = [
            {
                "stage": stage,
                "median_ms_per_row": item["median_ms_per_row"],
                "max_ms_per_row": item["max_ms_per_row"],
            }
            for stage, item in by_stage.items()
            if item["median_ms_per_row"] > 0 and
               item["max_ms_per_row"] > 5.0 * item["median_ms_per_row"]
        ]
        batches = next(iter(batch_counts))
        proposal_rows = next(iter(row_counts))
        summary["layers"][str(layer)] = {
            "profiled_batches": batches,
            "profiled_rows": proposal_rows,
            "typical_total_ms_per_row": typical_total_ms,
            "typical_attention_pre_ms_per_row":
                by_stage["attention_pre_batch"]["median_ms_per_row"],
            "typical_attention_tail_ms_per_row":
                by_stage["attention_tail_serial"]["median_ms_per_row"],
            "typical_ffn_ms_per_row":
                by_stage["ffn_batch"]["median_ms_per_row"],
            "outliers": outliers,
            "stages": by_stage,
        }
    return summary


def report(summary):
    lines = [
        "# DSpark Exact Runtime Layer Profile", "",
        "Synchronized diagnostic only. Stage boundaries change scheduling; do not use these values as throughput measurements.",
        "Values are normalized by proposal rows before taking each component median.",
        "", "| layer | batches | rows | total ms/row | attention prep | serial attention tail | exact FFN |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in summary["layers"].items():
        lines.append(
            f"| {layer} | {item['profiled_batches']} | {item['profiled_rows']} | "
            f"{item['typical_total_ms_per_row']:.3f} | "
            f"{item['typical_attention_pre_ms_per_row']:.3f} | "
            f"{item['typical_attention_tail_ms_per_row']:.3f} | "
            f"{item['typical_ffn_ms_per_row']:.3f} |"
        )
    lines.extend(["", "Components by median synchronized time per proposal row:"])
    for layer, item in summary["layers"].items():
        stages = sorted(
            item["stages"].items(),
            key=lambda pair: pair[1]["median_ms_per_row"],
            reverse=True,
        )
        text = ", ".join(
            f"{stage} {values['median_ms_per_row']:.3f} ms/row"
            for stage, values in stages
        )
        lines.append(f"- Layer {layer}: {text}")
    outliers = [
        (layer, outlier)
        for layer, item in summary["layers"].items()
        for outlier in item["outliers"]
    ]
    if outliers:
        lines.extend(["", "Synchronization/residency outliers (>5x stage median):"])
        for layer, outlier in outliers:
            lines.append(
                f"- Layer {layer} {outlier['stage']}: median "
                f"{outlier['median_ms_per_row']:.3f} ms/row, max "
                f"{outlier['max_ms_per_row']:.3f} ms/row"
            )
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
    if args.resume_dir:
        args.resume_dir = args.resume_dir.resolve()
    check_inputs(args, root)
    n_layers = inspect_layer_count(args, root)
    if args.layers is None:
        args.layers = tuple(dict.fromkeys((0, (n_layers - 1) // 2, n_layers - 1)))
    invalid = [layer for layer in args.layers if layer >= n_layers]
    if invalid:
        raise RuntimeError(
            f"profile layers outside model range 0..{n_layers - 1}: "
            + ", ".join(str(layer) for layer in invalid)
        )
    print(f"reference: {command_text(args)}")
    for layer in args.layers:
        print(f"layer {layer}: {command_text(args, layer)}")
    print("Profiled output must match the unprofiled exact reference byte-for-byte.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if args.resume_dir:
        run_dir = args.resume_dir
        if not run_dir.is_dir():
            raise RuntimeError(f"resume directory does not exist: {run_dir}")
        retained_metadata = validate_resume(run_dir, args)
    else:
        run_dir = (args.output_dir or root / "speed-bench/local-runs" / f"layer-profile-{stamp}").resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": run_capture(["git", "rev-parse", "HEAD"], root),
        "platform": platform.platform(),
        "config": {"ctx": args.ctx, "tokens": args.tokens,
                   "layers": args.layers, "cooldown": args.cooldown,
                   "model_layer_count": n_layers,
                   "temperature": 0, "seed": 1, "synchronized_profile": True,
                   "components": EXACT_STAGES},
        "commands": {"reference": command_text(args)} | {
            f"layer_{layer}": command_text(args, layer) for layer in args.layers
        },
        "prompt": {"path": str(args.prompt_file),
                   "sha256": sha256(args.prompt_file.read_bytes())},
        "cleared_environment_keys": cleared_env_keys(os.environ),
    }
    records = []
    runs = []
    if args.resume_dir:
        reference_path = run_dir / "reference.stdout"
        if not reference_path.is_file():
            raise RuntimeError(f"resume directory has no reference output: {reference_path}")
        reference = reference_path.read_bytes()
        if not reference:
            raise RuntimeError(f"resume reference output is empty: {reference_path}")
        metadata["resumed_at"] = dt.datetime.now().astimezone().isoformat()
        metadata["original_created_at"] = retained_metadata.get("created_at")
        runs.append({
            "name": "reference", "layer": "", "wall_seconds": "",
            "stdout_sha256": sha256(reference), "stdout_file": "reference.stdout",
            "stderr_file": "reference.stderr",
        })
    else:
        (run_dir / "metadata.start.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        reference, _, reference_run = execute(
            args, root, run_dir, "reference", None, None
        )
        runs.append(reference_run)
        cooldown(args.cooldown)
    for layer in args.layers:
        existing = load_existing(run_dir, layer, reference)
        if existing is not None:
            layer_records, run = existing
            print(f"[layer-{layer:02d}] reusing retained matching profile", flush=True)
        else:
            _, layer_records, run = execute(
                args, root, run_dir, f"layer-{layer:02d}", layer, reference
            )
            cooldown(args.cooldown)
        records.extend(layer_records)
        runs.append(run)

    with (run_dir / "stages.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=("part", "layer", "pos", "tokens", "stage", "ms")
        )
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
