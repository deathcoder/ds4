#!/usr/bin/env python3
"""User-run synchronized profile of the retained exact serial attention tail."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

import run_dspark_exact_layer_profile as layer_profile


CONTROL_STAGES = ("attention_pre_batch", "ffn_batch")
TAIL_STAGES = (
    "kv_cache_update",
    "compressor_indexer",
    "attention",
    "inverse_rope",
    "projection_a",
    "projection_b_hc",
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Profile the retained row-interleaved exact attention tail."
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
        "--prompt-file",
        type=Path,
        default=root / "speed-bench/issue468/code_8k.txt",
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--layers", type=layer_profile.parse_layers)
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
    env = os.environ.copy()
    for key in layer_profile.cleared_env_keys(env):
        env.pop(key, None)
    env["DS4_DSPARK_GPU_RUNTIME"] = "1"
    env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    if layer is not None:
        env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
        env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(layer)
        env["DS4_METAL_DECODE_STAGE_PROFILE"] = "1"
        env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"] = str(layer)
        env["DS4_DSPARK_EXACT_TAIL_PROFILE"] = "1"
    return env


def command(args):
    return [
        str(args.binary),
        "--backend",
        "metal",
        "--model",
        str(args.model),
        "--ctx",
        str(args.ctx),
        "-n",
        str(args.tokens),
        "--temp",
        "0",
        "--seed",
        "1",
        "--prompt-file",
        str(args.prompt_file),
        "--dspark",
        str(args.dspark_model),
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
        "DS4_DSPARK_EXACT_TAIL_PROFILE",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args))


def parse_profile(data, expected_layer, path):
    rows = []
    for line in data.splitlines():
        match = layer_profile.PROFILE_RE.match(line)
        if not match:
            continue
        part, layer, pos, tokens, stage, elapsed = match.groups()
        part = part.decode("ascii")
        if part not in ("exact", "tail"):
            continue
        row = {
            "part": part,
            "layer": int(layer),
            "pos": int(pos),
            "tokens": int(tokens),
            "stage": stage.decode("ascii"),
            "ms": float(elapsed),
        }
        if row["layer"] != expected_layer:
            raise RuntimeError(
                f"unexpected profiled layer {row['layer']} in {path}; "
                f"expected {expected_layer}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no exact-tail profile records found in {path}")

    exact_rows = [row for row in rows if row["part"] == "exact"]
    tail_rows = [row for row in rows if row["part"] == "tail"]
    exact_unknown = sorted({row["stage"] for row in exact_rows} - set(CONTROL_STAGES))
    tail_unknown = sorted({row["stage"] for row in tail_rows} - set(TAIL_STAGES))
    if exact_unknown:
        raise RuntimeError(f"unknown exact control stages in {path}: {', '.join(exact_unknown)}")
    if tail_unknown:
        raise RuntimeError(f"unknown exact-tail stages in {path}: {', '.join(tail_unknown)}")
    for stage in CONTROL_STAGES:
        if not any(row["stage"] == stage for row in exact_rows):
            raise RuntimeError(f"missing exact control stage {stage} in {path}")
    for stage in TAIL_STAGES:
        if not any(row["stage"] == stage for row in tail_rows):
            raise RuntimeError(f"missing exact-tail stage {stage} in {path}")
    return rows


def execute(args, root, run_dir, name, layer, reference):
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    print(f"[{name}] {command_text(args, layer)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args),
            cwd=root,
            env=profile_env(layer),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
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
    run = {
        "name": name,
        "layer": "" if layer is None else layer,
        "wall_seconds": wall_seconds,
        "stdout_sha256": layer_profile.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    return stdout_data, records, run


def component_summary(rows, normalize_rows=False):
    values = [
        row["ms"] / row["tokens"] if normalize_rows else row["ms"]
        for row in rows
    ]
    ordered = sorted(values)
    return {
        "records": len(rows),
        "total_ms": sum(row["ms"] for row in rows),
        "mean_ms_per_row": statistics.mean(values),
        "median_ms_per_row": statistics.median(values),
        "p90_ms_per_row": ordered[int(0.9 * (len(ordered) - 1))],
        "max_ms_per_row": max(values),
    }


def summarize(records):
    summary = {"layers": {}}
    for layer in sorted({row["layer"] for row in records}):
        selected = [row for row in records if row["layer"] == layer]
        exact = [row for row in selected if row["part"] == "exact"]
        tail = [row for row in selected if row["part"] == "tail"]

        controls = {}
        control_signatures = set()
        for stage in CONTROL_STAGES:
            stage_rows = [row for row in exact if row["stage"] == stage]
            if not stage_rows:
                raise RuntimeError(f"layer {layer} has no {stage} records")
            controls[stage] = component_summary(stage_rows, normalize_rows=True)
            control_signatures.add(tuple((row["pos"], row["tokens"]) for row in stage_rows))
        if len(control_signatures) != 1:
            raise RuntimeError(f"layer {layer} has mismatched control batches")
        proposal_signature = next(iter(control_signatures))
        expected_positions = sorted(
            pos + row
            for pos, tokens in proposal_signature
            for row in range(tokens)
        )

        stages = {}
        for stage in TAIL_STAGES:
            stage_rows = [row for row in tail if row["stage"] == stage]
            positions = sorted(row["pos"] for row in stage_rows)
            if any(row["tokens"] != 1 for row in stage_rows):
                raise RuntimeError(f"layer {layer} {stage} contains non-row records")
            if positions != expected_positions:
                raise RuntimeError(
                    f"layer {layer} {stage} rows do not match proposal batches"
                )
            stages[stage] = component_summary(stage_rows)

        typical_total = sum(
            stages[stage]["median_ms_per_row"] for stage in TAIL_STAGES
        )
        for stage in TAIL_STAGES:
            stages[stage]["median_share_percent"] = (
                stages[stage]["median_ms_per_row"] / typical_total * 100.0
            )
        summary["layers"][str(layer)] = {
            "profiled_batches": len(proposal_signature),
            "profiled_rows": len(expected_positions),
            "typical_tail_total_ms_per_row": typical_total,
            "proposal_signature": [
                {"pos": pos, "tokens": tokens}
                for pos, tokens in proposal_signature
            ],
            "controls": controls,
            "stages": stages,
        }
    return summary


def report(summary):
    lines = [
        "# DSpark Exact Serial Attention Tail Profile",
        "",
        "Synchronized diagnostic only. Boundaries preserve operation order but change scheduling; do not use these values as throughput measurements.",
        "Tail components are one-row records; control medians are normalized by proposal rows.",
        "",
        "| layer | batches | rows | tail total | KV/cache | compressor/indexer | attention | inverse RoPE | projection A | projection B + HC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in summary["layers"].items():
        stages = item["stages"]
        lines.append(
            f"| {layer} | {item['profiled_batches']} | {item['profiled_rows']} | "
            f"{item['typical_tail_total_ms_per_row']:.3f} | "
            f"{stages['kv_cache_update']['median_ms_per_row']:.3f} | "
            f"{stages['compressor_indexer']['median_ms_per_row']:.3f} | "
            f"{stages['attention']['median_ms_per_row']:.3f} | "
            f"{stages['inverse_rope']['median_ms_per_row']:.3f} | "
            f"{stages['projection_a']['median_ms_per_row']:.3f} | "
            f"{stages['projection_b_hc']['median_ms_per_row']:.3f} |"
        )
    lines.extend(["", "Largest components by median synchronized time:"])
    for layer, item in summary["layers"].items():
        ranked = sorted(
            item["stages"].items(),
            key=lambda pair: pair[1]["median_ms_per_row"],
            reverse=True,
        )
        text = ", ".join(
            f"{stage} {values['median_ms_per_row']:.3f} ms/row "
            f"({values['median_share_percent']:.1f}%)"
            for stage, values in ranked
        )
        lines.append(f"- Layer {layer}: {text}")
    lines.extend(["", "Control-stage medians (attention prep / FFN):"])
    for layer, item in summary["layers"].items():
        controls = item["controls"]
        lines.append(
            f"- Layer {layer}: "
            f"{controls['attention_pre_batch']['median_ms_per_row']:.3f} / "
            f"{controls['ffn_batch']['median_ms_per_row']:.3f} ms/row"
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
    layer_profile.check_inputs(args, root)
    n_layers = layer_profile.inspect_layer_count(args, root)
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
    print("Every profiled output must match the unprofiled exact reference byte-for-byte.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_dir
        or root / "speed-bench/local-runs" / f"tail-profile-{stamp}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": layer_profile.run_capture(["git", "rev-parse", "HEAD"], root),
        "git_status_tracked": layer_profile.run_capture(
            ["git", "status", "--porcelain", "--untracked-files=no"], root
        ),
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "layers": args.layers,
            "cooldown": args.cooldown,
            "model_layer_count": n_layers,
            "temperature": 0,
            "seed": 1,
            "synchronized_profile": True,
            "control_components": CONTROL_STAGES,
            "tail_components": TAIL_STAGES,
        },
        "commands": {"reference": command_text(args)} | {
            f"layer_{layer}": command_text(args, layer) for layer in args.layers
        },
        "prompt": {
            "path": str(args.prompt_file),
            "sha256": layer_profile.sha256(args.prompt_file.read_bytes()),
        },
        "cleared_environment_keys": layer_profile.cleared_env_keys(os.environ),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    reference, _, reference_run = execute(
        args, root, run_dir, "reference", None, None
    )
    records = []
    runs = [reference_run]
    cooldown(args.cooldown)
    for layer in args.layers:
        _, stage_rows, run = execute(
            args, root, run_dir, f"layer-{layer:02d}", layer, reference
        )
        records.extend(stage_rows)
        runs.append(run)
        cooldown(args.cooldown)

    with (run_dir / "stages.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=("part", "layer", "pos", "tokens", "stage", "ms")
        )
        writer.writeheader()
        writer.writerows(records)
    with (run_dir / "runs.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=(
                "name", "layer", "wall_seconds", "stdout_sha256",
                "stdout_file", "stderr_file",
            ),
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
