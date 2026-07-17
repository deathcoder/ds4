#!/usr/bin/env python3
"""Profile exact routed-MoE one-row stages by verifier width."""

import argparse
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

import run_dspark_exact_layer_profile as layer_profile
import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_cumulative_cost_audit as cumulative_cost
import run_dspark_issue468_comparison as common
import run_dspark_post_promotion_width_ffn_profile as width_ffn
import run_dspark_threshold075_width_layer_profile as width_layer


THRESHOLD = width_layer.THRESHOLD
TASK = width_layer.TASK
LAYERS = width_layer.LAYERS
WIDTHS = width_layer.WIDTHS
MOE_STAGES = ("gate_up", "activation_weight", "down", "sum")
WIDTH_FFN_SOURCE_COMMIT = "862da2c3195df81db0a359a41718e039d1799700"
MOE_ONE_RE = re.compile(
    rb"^ds4: Metal routed MoE one stage layer=(\d+) pairs=(\d+) "
    rb"experts=(\d+) gate=(\S+) down=(\S+) path=(\S+) "
    rb"(\S+)=([0-9.]+) ms$"
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile exact routed-MoE gate/up, activation, down, and sum "
            "stages by verifier width on frozen HumanEval task 079."
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
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "speed-bench/humaneval-acceptance",
    )
    parser.add_argument("--throughput-reference", type=Path, required=True)
    parser.add_argument("--cost-reference", type=Path, required=True)
    parser.add_argument("--width-layer-reference", type=Path, required=True)
    parser.add_argument("--width-ffn-reference", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=1.0)
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

    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = True
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = THRESHOLD
    args.pairs = 0
    args.warmups = 0
    return args, root


def load_width_ffn_reference(args, cost_reference):
    summary_path = args.width_ffn_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    stages_path = run_dir / "stages.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("stages CSV", stages_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing width-FFN {label}: {path}")
    summary = width_ffn.load_json(summary_path, "width-FFN summary")
    metadata = width_ffn.load_json(metadata_path, "width-FFN metadata")
    expected_kind = "dspark_post_promotion_width_stratified_exact_ffn"
    if metadata.get("experiment") != expected_kind:
        raise SystemExit("width-FFN reference has the wrong experiment kind")
    if summary.get("analysis") != expected_kind:
        raise SystemExit("width-FFN reference has the wrong analysis kind")
    if metadata.get("git_commit") != WIDTH_FFN_SOURCE_COMMIT:
        raise SystemExit("width-FFN reference source commit mismatch")
    if metadata.get("git_status_tracked"):
        raise SystemExit("width-FFN reference used a dirty tree")
    expected_paths = {
        "throughput_reference": args.throughput_reference.resolve(),
        "cost_reference": args.cost_reference.resolve(),
        "width_layer_reference": args.width_layer_reference.resolve(),
    }
    for group, expected in expected_paths.items():
        value = metadata.get(group, {}).get("path")
        actual = Path(value).resolve() if value is not None else None
        if actual != expected:
            raise SystemExit(f"width-FFN {group} mismatch")
    expected_counts = {
        str(width): cost_reference["stats"]["verify_width_evals"][width]
        for width in WIDTHS
    }
    expected_summary = {
        "threshold": THRESHOLD,
        "task": TASK,
        "layers": list(LAYERS),
        "widths": list(WIDTHS),
        "expected_width_evals": expected_counts,
        "largest_width5_stage": "routed_moe",
        "weakest_amortization_stage": "routed_moe",
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise SystemExit(f"width-FFN reference {key} mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "stages_path": stages_path,
        "summary": summary,
        "metadata": metadata,
    }


def profile_env(layer):
    env = width_ffn.profile_env(layer)
    env["DS4_METAL_MOE_ONE_STAGE_PROFILE"] = "1"
    env["DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER"] = str(layer)
    return env


def command_text(args, prompt, layer):
    env = profile_env(layer)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_STATS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
        "DS4_METAL_LAYER_STAGE_PROFILE",
        "DS4_METAL_LAYER_STAGE_PROFILE_LAYER",
        "DS4_METAL_MOE_ONE_STAGE_PROFILE",
        "DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys)
    return prefix + " " + shlex.join(common.mode_command(args, prompt, "runtime"))


def parse_profile(data, expected_layer, path):
    records = []
    for sequence, line in enumerate(data.splitlines()):
        match = layer_profile.PROFILE_RE.match(line)
        if match:
            part, layer, pos, tokens, stage, elapsed = match.groups()
            row = {
                "sequence": sequence,
                "part": part.decode("ascii"),
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
            if row["part"] in {"exact", "ffn"}:
                records.append(row)
            continue
        match = MOE_ONE_RE.match(line)
        if not match:
            continue
        layer, pairs, experts, gate, down, moe_path, stage, elapsed = (
            match.groups()
        )
        layer = int(layer)
        if layer != expected_layer:
            raise RuntimeError(
                f"unexpected routed-MoE layer {layer} in {path}; "
                f"expected {expected_layer}"
            )
        records.append({
            "sequence": sequence,
            "part": "moe_one",
            "layer": layer,
            "pairs": int(pairs),
            "experts": int(experts),
            "gate": gate.decode("ascii"),
            "down": down.decode("ascii"),
            "path": moe_path.decode("ascii"),
            "stage": stage.decode("ascii"),
            "ms": float(elapsed),
        })
    if not any(row["part"] == "moe_one" for row in records):
        raise RuntimeError(f"no routed-MoE one-stage records found in {path}")
    return records


def assign_exact_moe_batches(records, expected_stats, layer):
    controls = [
        row for row in records
        if row["part"] == "exact" and row["stage"] == "ffn_batch"
        and row["tokens"] in WIDTHS
    ]
    for width in WIDTHS:
        actual = sum(row["tokens"] == width for row in controls)
        expected = expected_stats["verify_width_evals"][width]
        if actual != expected:
            raise RuntimeError(
                f"layer {layer} width {width} has {actual} controls, "
                f"expected {expected}"
            )

    assigned = []
    previous_control_sequence = -1
    for batch, control in enumerate(controls, start=1):
        interval = [
            row for row in records
            if previous_control_sequence < row["sequence"] < control["sequence"]
        ]
        tails = [
            row for row in interval
            if row["part"] == "exact"
            and row["stage"] == "attention_tail_serial"
            and row["pos"] == control["pos"]
            and row["tokens"] == control["tokens"]
        ]
        if len(tails) != 1:
            raise RuntimeError(
                f"layer {layer} batch {batch} has {len(tails)} exact tails"
            )
        exact_interval = [
            row for row in interval if row["sequence"] > tails[0]["sequence"]
        ]
        routers = [
            row for row in exact_interval
            if row["part"] == "ffn" and row["stage"] == "router"
            and row["pos"] == control["pos"]
            and row["tokens"] == control["tokens"]
        ]
        routed = [
            row for row in exact_interval
            if row["part"] == "ffn" and row["stage"] == "routed_moe"
            and row["pos"] == control["pos"]
            and row["tokens"] == control["tokens"]
        ]
        if len(routers) != 1 or len(routed) != 1:
            raise RuntimeError(
                f"layer {layer} batch {batch} has router/routed controls "
                f"{len(routers)}/{len(routed)}"
            )
        inner = [
            row for row in exact_interval
            if routers[0]["sequence"] < row["sequence"] < routed[0]["sequence"]
            and row["part"] == "moe_one"
        ]
        expected_sequence = list(MOE_STAGES) * control["tokens"]
        actual_sequence = [row["stage"] for row in inner]
        if actual_sequence != expected_sequence:
            raise RuntimeError(
                f"layer {layer} batch {batch} routed-MoE stage sequence "
                f"mismatch: {actual_sequence}"
            )
        paths = {row["path"] for row in inner}
        shapes = {(row["pairs"], row["experts"]) for row in inner}
        types = {(row["gate"], row["down"]) for row in inner}
        if len(paths) != 1 or shapes != {(6, 6)} or len(types) != 1:
            raise RuntimeError(
                f"layer {layer} batch {batch} routed-MoE identity mismatch"
            )
        for index, row in enumerate(inner):
            assigned.append({
                "layer": layer,
                "batch": batch,
                "pos": control["pos"],
                "width": control["tokens"],
                "row": index // len(MOE_STAGES),
                "stage": row["stage"],
                "path": row["path"],
                "gate": row["gate"],
                "down": row["down"],
                "ms": row["ms"],
            })
        assigned.append({
            "layer": layer,
            "batch": batch,
            "pos": control["pos"],
            "width": control["tokens"],
            "row": -1,
            "stage": "routed_moe_control",
            "path": next(iter(paths)),
            "gate": next(iter(types))[0],
            "down": next(iter(types))[1],
            "ms": routed[0]["ms"] / control["tokens"],
        })
        previous_control_sequence = control["sequence"]
    return assigned


def execute(args, root, run_dir, prompt, layer, reference):
    name = f"layer-{layer:02d}"
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    print(f"[{name}] {command_text(args, prompt, layer)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            common.mode_command(args, prompt, "runtime"),
            cwd=root,
            env=profile_env(layer),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit {completed.returncode}; see {stderr_path}"
        )
    stdout_data = stdout_path.read_bytes()
    if stdout_data != reference:
        raise RuntimeError(
            f"{name} output differs from frozen cumulative output; "
            f"see {stdout_path}"
        )
    stderr_data = stderr_path.read_bytes()
    return (
        parse_profile(stderr_data, layer, stderr_path),
        common.parse_stats(stderr_data, stderr_path),
        {
            "name": name,
            "layer": layer,
            "wall_seconds": wall_seconds,
            "stdout_sha256": common.sha256(stdout_data),
            "stdout_file": stdout_path.name,
            "stderr_file": stderr_path.name,
        },
    )


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(rows, expected_stats):
    layer_widths = {}
    for layer in LAYERS:
        layer_widths[str(layer)] = {}
        for width in WIDTHS:
            selected = [
                row for row in rows
                if row["layer"] == layer and row["width"] == width
            ]
            evals = expected_stats["verify_width_evals"][width]
            stages = {}
            for stage in MOE_STAGES:
                values = [row["ms"] for row in selected if row["stage"] == stage]
                expected = evals * width
                if len(values) != expected:
                    raise RuntimeError(
                        f"layer {layer} width {width} has {len(values)} "
                        f"{stage} rows, expected {expected}"
                    )
                stages[stage] = {
                    "rows": len(values),
                    "median_ms_per_row": statistics.median(values),
                    "mean_ms_per_row": statistics.mean(values),
                    "p90_ms_per_row": percentile(values, 0.9),
                    "max_ms_per_row": max(values),
                }
            controls = [
                row["ms"] for row in selected
                if row["stage"] == "routed_moe_control"
            ]
            if len(controls) != evals:
                raise RuntimeError(
                    f"layer {layer} width {width} has {len(controls)} controls, "
                    f"expected {evals}"
                )
            inner_total = sum(
                stages[stage]["median_ms_per_row"] for stage in MOE_STAGES
            )
            control = statistics.median(controls)
            layer_widths[str(layer)][str(width)] = {
                "evals": evals,
                "rows": evals * width,
                "inner_ms_per_row": inner_total,
                "control_ms_per_row": control,
                "inner_vs_control": inner_total / control,
                "stages": stages,
            }

    width_totals = {}
    for width in WIDTHS:
        components = {
            stage: sum(
                layer_widths[str(layer)][str(width)]["stages"][stage][
                    "median_ms_per_row"
                ]
                for layer in LAYERS
            )
            for stage in MOE_STAGES
        }
        inner_total = sum(components.values())
        control_total = sum(
            layer_widths[str(layer)][str(width)]["control_ms_per_row"]
            for layer in LAYERS
        )
        width_totals[str(width)] = {
            "evals": expected_stats["verify_width_evals"][width],
            "rows": expected_stats["verify_width_evals"][width] * width,
            "inner_ms_per_row": inner_total,
            "control_ms_per_row": control_total,
            "inner_vs_control": inner_total / control_total,
            "components": components,
        }

    width5 = width_totals["5"]["components"]
    width5_total = sum(width5.values())
    width5_components = {
        stage: {
            "ms_per_row": width5[stage],
            "share": width5[stage] / width5_total,
            "layers": {
                str(layer): layer_widths[str(layer)]["5"]["stages"][stage][
                    "median_ms_per_row"
                ]
                for layer in LAYERS
            },
        }
        for stage in MOE_STAGES
    }
    return {
        "analysis": "dspark_post_promotion_width_stratified_exact_routed_moe",
        "threshold": THRESHOLD,
        "task": TASK,
        "layers": list(LAYERS),
        "widths": list(WIDTHS),
        "moe_stages": list(MOE_STAGES),
        "expected_width_evals": {
            str(width): expected_stats["verify_width_evals"][width]
            for width in WIDTHS
        },
        "layer_widths": layer_widths,
        "sampled_width_totals": width_totals,
        "width5_components": width5_components,
        "largest_width5_stage": max(width5, key=width5.get),
    }


def render_report(summary):
    lines = [
        "# DSpark Post-Promotion Exact Routed-MoE Stage Profile",
        "",
        "Synchronized diagnostic only. Inner MoE boundaries change scheduling; do not use these values as throughput measurements.",
        "Every profiled output matched the frozen cumulative HumanEval artifact byte-for-byte.",
        "One-row MoE records are accepted only inside their enclosing exact-verifier routed-MoE interval.",
        "",
        "## Sampled Routed-MoE Totals",
        "",
        "| width | evals | rows | inner stages ms/row | outer MoE ms/row | inner/control |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["sampled_width_totals"][str(width)]
        lines.append(
            f"| {width} | {item['evals']} | {item['rows']} | "
            f"{item['inner_ms_per_row']:.3f} | "
            f"{item['control_ms_per_row']:.3f} | "
            f"{item['inner_vs_control']:.3f}x |"
        )
    lines.extend([
        "",
        "## Width-5 Components",
        "",
        "| stage | sampled ms/row | share | layer 0 | layer 21 | layer 42 |",
        "|:---|---:|---:|---:|---:|---:|",
    ])
    for stage in MOE_STAGES:
        item = summary["width5_components"][stage]
        layers = item["layers"]
        lines.append(
            f"| {stage} | {item['ms_per_row']:.3f} | "
            f"{item['share']:.1%} | {layers['0']:.3f} | "
            f"{layers['21']:.3f} | {layers['42']:.3f} |"
        )
    lines.extend([
        "",
        f"- Largest sampled width-5 routed-MoE stage: "
        f"`{summary['largest_width5_stage']}`.",
        "",
        "## Interpretation Limits",
        "",
        "- Widths 2 and 3 have one verifier evaluation each; width 5 supplies the stable stage-share sample.",
        "- Every inner record measures one routed-MoE row; width grouping describes scheduling context, not a batched inner kernel.",
        "- The sum of separately synchronized inner medians is reconciled against, but need not equal, the outer routed-MoE median.",
        "- Only layers 0, 21, and 42 are sampled; this identifies shared structure rather than a full-model time sum.",
        "- No throughput benchmark, acceptance audit, trace, fast verifier, or runtime candidate is enabled.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference", "cost_reference", "width_layer_reference",
        "width_ffn_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, cumulative_cost.TASK_COUNT, provenance["selection_policy"]
    )
    throughput_reference = cumulative_cost.load_throughput_reference(
        args, records, selection
    )
    cost_reference = width_layer.load_cost_reference(args)
    if cost_reference["reference_kind"] != "post_promotion_cumulative":
        raise SystemExit("routed-MoE profile requires post-promotion cost data")
    width_layer_reference = width_ffn.load_width_layer_reference(
        args, cost_reference
    )
    width_ffn_reference = load_width_ffn_reference(args, cost_reference)
    context = throughput_reference["tasks"][TASK]
    layer_count = layer_profile.inspect_layer_count(args, root)
    if any(layer >= layer_count for layer in LAYERS):
        raise SystemExit(f"profile layers outside model range 0..{layer_count - 1}")

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/post-promotion-width-moe-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    prompt = run_dir / "prompt.txt"
    for layer in LAYERS:
        print(f"layer {layer}: {command_text(args, prompt, layer)}")
    print(
        "Three synchronized exact routed-MoE profile processes; every output "
        "must match the frozen cumulative artifact."
    )
    if args.dry_run:
        print("Dry run only; no prompt materialized and no inference performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    prompt.write_bytes(context["prompt_data"])
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": common.git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "experiment": (
            "dspark_post_promotion_width_stratified_exact_routed_moe"
        ),
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "selection": selection,
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "confidence_threshold": THRESHOLD,
            "task": TASK,
            "layers": LAYERS,
            "widths": WIDTHS,
            "moe_stages": MOE_STAGES,
            "runtime_stats": True,
            "synchronized_profile": True,
            "timed_throughput": False,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "prompt_sha256": common.sha256(context["prompt_data"]),
        "throughput_reference": common.file_metadata(
            throughput_reference["summary_path"]
        ),
        "cost_reference": common.file_metadata(cost_reference["summary_path"]),
        "width_layer_reference": common.file_metadata(
            width_layer_reference["summary_path"]
        ),
        "width_ffn_reference": common.file_metadata(
            width_ffn_reference["summary_path"]
        ),
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "commands": {
            str(layer): command_text(args, prompt, layer) for layer in LAYERS
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    assigned = []
    stats_out = []
    runs = []
    for layer in LAYERS:
        profile_rows, stats, run = execute(
            args, root, run_dir, prompt, layer, context["output_data"]
        )
        width_layer.validate_counts(stats, cost_reference["stats"], layer)
        assigned.extend(
            assign_exact_moe_batches(profile_rows, cost_reference["stats"], layer)
        )
        stats_out.append({"layer": layer, **stats})
        runs.append(run)
        common.cooldown(args.cooldown)

    summary = summarize(assigned, cost_reference["stats"])
    report = render_report(summary)
    write_csv(
        run_dir / "stages.csv",
        (
            "layer", "batch", "pos", "width", "row", "stage", "path",
            "gate", "down", "ms",
        ),
        assigned,
    )
    write_csv(
        run_dir / "runs.csv",
        (
            "name", "layer", "wall_seconds", "stdout_sha256",
            "stdout_file", "stderr_file",
        ),
        runs,
    )
    common.write_csv(
        run_dir / "stats.csv", stats_out, ("layer",) + common.STATS_FIELDS
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw stages: {run_dir / 'stages.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
