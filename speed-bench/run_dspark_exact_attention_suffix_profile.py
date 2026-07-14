#!/usr/bin/env python3
"""User-run synchronized attribution profile for the exact attention suffix."""

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


DEFAULT_MODE = "default_exact"
CANDIDATE_MODE = "batch_attention_suffix"
MODES = (DEFAULT_MODE, CANDIDATE_MODE)
DEFAULT_STAGES = (
    "attention_pre_batch",
    "attention_tail_serial",
    "ffn_batch",
)
CANDIDATE_STAGES = (
    "attention_pre_batch",
    "attention_core_capture_serial",
    "attention_projection_a_batch",
    "attention_projection_b_hc_batch",
    "ffn_batch",
)
MODE_STAGES = {
    DEFAULT_MODE: DEFAULT_STAGES,
    CANDIDATE_MODE: CANDIDATE_STAGES,
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Attribute the exact attention-suffix batch regression by layer."
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


def profile_env(mode, layer=None):
    env = os.environ.copy()
    for key in layer_profile.cleared_env_keys(env):
        env.pop(key, None)
    env["DS4_DSPARK_GPU_RUNTIME"] = "1"
    env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    if mode == CANDIDATE_MODE:
        env["DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH"] = "1"
    if layer is not None:
        env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
        env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(layer)
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


def command_text(args, mode, layer=None):
    env = profile_env(mode, layer)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args))


def parse_profile(data, expected_layer, mode, path):
    rows = []
    for line in data.splitlines():
        match = layer_profile.PROFILE_RE.match(line)
        if not match:
            continue
        part, layer, pos, tokens, stage, elapsed = match.groups()
        if part != b"exact":
            continue
        row = {
            "mode": mode,
            "part": "exact",
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
        raise RuntimeError(f"no exact-layer profile records found in {path}")
    expected = set(MODE_STAGES[mode])
    unknown = sorted({row["stage"] for row in rows} - expected)
    if unknown:
        raise RuntimeError(f"unknown {mode} stages in {path}: {', '.join(unknown)}")
    missing = sorted(expected - {row["stage"] for row in rows})
    if missing:
        raise RuntimeError(f"missing {mode} stages in {path}: {', '.join(missing)}")
    return rows


def execute(args, root, run_dir, name, mode, layer, reference):
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    print(f"[{name}] {command_text(args, mode, layer)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args),
            cwd=root,
            env=profile_env(mode, layer),
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
        stderr_path.read_bytes(), layer, mode, stderr_path
    )
    run = {
        "name": name,
        "mode": mode,
        "layer": "" if layer is None else layer,
        "wall_seconds": wall_seconds,
        "stdout_sha256": layer_profile.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    return stdout_data, records, run


def stage_summary(rows):
    values = [row["ms"] / row["tokens"] for row in rows]
    ordered = sorted(values)
    return {
        "batches": len(rows),
        "proposal_rows": sum(row["tokens"] for row in rows),
        "total_ms": sum(row["ms"] for row in rows),
        "mean_ms_per_row": statistics.mean(values),
        "median_ms_per_row": statistics.median(values),
        "p90_ms_per_row": ordered[int(0.9 * (len(ordered) - 1))],
        "max_ms_per_row": max(values),
    }


def summarize(records):
    summary = {"layers": {}}
    layers = sorted({row["layer"] for row in records})
    for layer in layers:
        modes = {}
        for mode in MODES:
            selected = [
                row for row in records
                if row["layer"] == layer and row["mode"] == mode
            ]
            stages = {}
            signatures = set()
            for stage in MODE_STAGES[mode]:
                stage_rows = [row for row in selected if row["stage"] == stage]
                if not stage_rows:
                    raise RuntimeError(f"layer {layer} mode {mode} has no {stage} records")
                stages[stage] = stage_summary(stage_rows)
                signatures.add(tuple((row["pos"], row["tokens"]) for row in stage_rows))
            if len(signatures) != 1:
                raise RuntimeError(f"layer {layer} mode {mode} has incomplete batches")
            signature = next(iter(signatures))
            modes[mode] = {
                "proposal_signature": [
                    {"pos": pos, "tokens": tokens} for pos, tokens in signature
                ],
                "stages": stages,
            }

        if (modes[DEFAULT_MODE]["proposal_signature"] !=
                modes[CANDIDATE_MODE]["proposal_signature"]):
            raise RuntimeError(
                f"layer {layer} default and candidate proposal schedules differ"
            )

        default_tail = modes[DEFAULT_MODE]["stages"][
            "attention_tail_serial"
        ]["median_ms_per_row"]
        candidate_stages = modes[CANDIDATE_MODE]["stages"]
        candidate_core = candidate_stages[
            "attention_core_capture_serial"
        ]["median_ms_per_row"]
        candidate_a = candidate_stages[
            "attention_projection_a_batch"
        ]["median_ms_per_row"]
        candidate_b_hc = candidate_stages[
            "attention_projection_b_hc_batch"
        ]["median_ms_per_row"]
        candidate_total = candidate_core + candidate_a + candidate_b_hc
        batches = modes[DEFAULT_MODE]["stages"][DEFAULT_STAGES[0]]["batches"]
        rows = modes[DEFAULT_MODE]["stages"][DEFAULT_STAGES[0]]["proposal_rows"]
        for mode in MODES:
            for stage in modes[mode]["stages"].values():
                if stage["batches"] != batches or stage["proposal_rows"] != rows:
                    raise RuntimeError(
                        f"layer {layer} mode {mode} does not match the proposal schedule"
                    )
        summary["layers"][str(layer)] = {
            "profiled_batches": batches,
            "profiled_rows": rows,
            "default_attention_tail_ms_per_row": default_tail,
            "candidate_core_capture_ms_per_row": candidate_core,
            "candidate_projection_a_ms_per_row": candidate_a,
            "candidate_projection_b_hc_ms_per_row": candidate_b_hc,
            "candidate_attention_total_ms_per_row": candidate_total,
            "candidate_to_default_ratio": candidate_total / default_tail,
            "candidate_delta_ms_per_row": candidate_total - default_tail,
            "candidate_delta_percent": (candidate_total / default_tail - 1.0) * 100.0,
            "modes": modes,
        }
    return summary


def report(summary):
    lines = [
        "# DSpark Exact Attention Suffix Attribution",
        "",
        "Synchronized diagnostic only. Boundaries change scheduling; do not use these values as throughput measurements.",
        "Values are normalized by proposal rows before taking each component median.",
        "",
        "| layer | batches | rows | default serial tail | candidate core + capture | projection A | projection B + HC | candidate total | ratio | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in summary["layers"].items():
        lines.append(
            f"| {layer} | {item['profiled_batches']} | {item['profiled_rows']} | "
            f"{item['default_attention_tail_ms_per_row']:.3f} | "
            f"{item['candidate_core_capture_ms_per_row']:.3f} | "
            f"{item['candidate_projection_a_ms_per_row']:.3f} | "
            f"{item['candidate_projection_b_hc_ms_per_row']:.3f} | "
            f"{item['candidate_attention_total_ms_per_row']:.3f} | "
            f"{item['candidate_to_default_ratio']:.3f}x | "
            f"{item['candidate_delta_percent']:+.1f}% |"
        )
    lines.extend(["", "Control-stage medians (default/candidate):"])
    for layer, item in summary["layers"].items():
        default_stages = item["modes"][DEFAULT_MODE]["stages"]
        candidate_stages = item["modes"][CANDIDATE_MODE]["stages"]
        lines.append(
            f"- Layer {layer}: attention prep "
            f"{default_stages['attention_pre_batch']['median_ms_per_row']:.3f}/"
            f"{candidate_stages['attention_pre_batch']['median_ms_per_row']:.3f} ms/row; "
            f"FFN {default_stages['ffn_batch']['median_ms_per_row']:.3f}/"
            f"{candidate_stages['ffn_batch']['median_ms_per_row']:.3f} ms/row"
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

    print(f"reference: {command_text(args, DEFAULT_MODE)}")
    for index, layer in enumerate(args.layers):
        modes = MODES if index % 2 == 0 else tuple(reversed(MODES))
        for mode in modes:
            print(f"{mode} layer {layer}: {command_text(args, mode, layer)}")
    print("Every profiled output must match the unprofiled exact reference byte-for-byte.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_dir
        or root / "speed-bench/local-runs" / f"suffix-profile-{stamp}"
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
            "default_components": DEFAULT_STAGES,
            "candidate_components": CANDIDATE_STAGES,
        },
        "commands": {"reference": command_text(args, DEFAULT_MODE)} | {
            f"{mode}_layer_{layer}": command_text(args, mode, layer)
            for layer in args.layers
            for mode in MODES
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
        args, root, run_dir, "reference", DEFAULT_MODE, None, None
    )
    records = []
    runs = [reference_run]
    cooldown(args.cooldown)
    for index, layer in enumerate(args.layers):
        modes = MODES if index % 2 == 0 else tuple(reversed(MODES))
        for mode in modes:
            name = f"{mode}.layer-{layer:02d}"
            _, stage_rows, run = execute(
                args, root, run_dir, name, mode, layer, reference
            )
            records.extend(stage_rows)
            runs.append(run)
            cooldown(args.cooldown)

    with (run_dir / "stages.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=("mode", "part", "layer", "pos", "tokens", "stage", "ms"),
        )
        writer.writeheader()
        writer.writerows(records)
    with (run_dir / "runs.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=(
                "name", "mode", "layer", "wall_seconds", "stdout_sha256",
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
