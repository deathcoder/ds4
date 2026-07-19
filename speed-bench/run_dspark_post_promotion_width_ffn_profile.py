#!/usr/bin/env python3
"""Profile post-promotion exact-FFN sub-stages by verifier width."""

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
import time

import run_dspark_exact_layer_profile as layer_profile
import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_cumulative_cost_audit as cumulative_cost
import run_dspark_issue468_comparison as common
import run_dspark_threshold075_width_layer_profile as width_layer


THRESHOLD = width_layer.THRESHOLD
TASK = width_layer.TASK
LAYERS = width_layer.LAYERS
WIDTHS = width_layer.WIDTHS
FFN_STAGES = (
    "hc_pre",
    "norm",
    "router",
    "shared_gate_up",
    "shared_down",
    "routed_moe",
    "hc_post",
)
WIDTH_LAYER_SOURCE_COMMIT = "83f3e803e7baa4097cc8c5ff490f72b29aced06c"


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile exact-FFN HC, routing, shared-expert, and routed-MoE "
            "costs by verifier width on frozen HumanEval task 079."
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
        "--corpus-dir",
        type=Path,
        default=root / "speed-bench/humaneval-acceptance",
    )
    parser.add_argument("--throughput-reference", type=Path, required=True)
    parser.add_argument("--cost-reference", type=Path, required=True)
    parser.add_argument("--width-layer-reference", type=Path, required=True)
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


def load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label} {path}: {exc}") from exc


def referenced_path(metadata, group, name):
    value = metadata.get(group, {}).get(name, {}).get("path")
    return Path(value).resolve() if value is not None else None


def load_width_layer_reference(args, cost_reference):
    summary_path = args.width_layer_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    stages_path = run_dir / "stages.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("stages CSV", stages_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing post-promotion width-layer {label}: {path}")
    summary = load_json(summary_path, "post-promotion width-layer summary")
    metadata = load_json(metadata_path, "post-promotion width-layer metadata")
    if metadata.get("experiment") != (
        "dspark_post_promotion_width_stratified_exact_layer"
    ):
        raise SystemExit("width-layer reference has the wrong experiment kind")
    if summary.get("analysis") != (
        "dspark_post_promotion_width_stratified_exact_layer"
    ):
        raise SystemExit("width-layer reference has the wrong analysis kind")
    if summary.get("reference_kind") != "post_promotion_cumulative":
        raise SystemExit("width-layer reference is not post-promotion")
    if metadata.get("git_commit") != WIDTH_LAYER_SOURCE_COMMIT:
        raise SystemExit("width-layer reference source commit mismatch")
    if metadata.get("git_status_tracked"):
        raise SystemExit("width-layer reference used a dirty tree")
    if referenced_path(metadata, "throughput_reference", "summary") != (
        args.throughput_reference.resolve()
    ):
        raise SystemExit("width-layer throughput reference mismatch")
    if referenced_path(metadata, "cost_reference", "summary") != (
        args.cost_reference.resolve()
    ):
        raise SystemExit("width-layer cost reference mismatch")
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
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise SystemExit(f"width-layer reference {key} mismatch")
    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "confidence_threshold": THRESHOLD,
        "task": TASK,
        "layers": list(LAYERS),
        "widths": list(WIDTHS),
        "runtime_stats": True,
        "synchronized_profile": True,
        "timed_throughput": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"width-layer config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "stages_path": stages_path,
        "summary": summary,
        "metadata": metadata,
    }


def profile_env(layer):
    env = common.benchmark_env(
        "runtime", False, stats=True, confidence_threshold=THRESHOLD
    )
    env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
    env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(layer)
    env["DS4_METAL_LAYER_STAGE_PROFILE"] = "1"
    env["DS4_METAL_LAYER_STAGE_PROFILE_LAYER"] = str(layer)
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
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys)
    return prefix + " " + shlex.join(common.mode_command(args, prompt, "runtime"))


def parse_profile(data, expected_layer, path):
    rows = []
    for sequence, line in enumerate(data.splitlines()):
        match = layer_profile.PROFILE_RE.match(line)
        if not match:
            continue
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
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no exact/FFN profile records found in {path}")
    return rows


def assign_exact_ffn_batches(records, expected_stats, layer):
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
                f"layer {layer} width {width} has {actual} exact FFN controls, "
                f"expected {expected}"
            )

    assigned = []
    previous_control_sequence = -1
    for batch, control in enumerate(controls, start=1):
        tail_controls = [
            row for row in records
            if previous_control_sequence < row["sequence"] < control["sequence"]
            and row["part"] == "exact"
            and row["stage"] == "attention_tail_serial"
            and row["pos"] == control["pos"]
            and row["tokens"] == control["tokens"]
        ]
        if len(tail_controls) != 1:
            raise RuntimeError(
                f"layer {layer} batch {batch} has {len(tail_controls)} "
                "matching exact attention-tail controls"
            )
        tail_sequence = tail_controls[0]["sequence"]
        candidates = [
            row for row in records
            if tail_sequence < row["sequence"] < control["sequence"]
            and row["part"] == "ffn"
            and row["pos"] == control["pos"]
            and row["tokens"] == control["tokens"]
        ]
        by_stage = {}
        for row in candidates:
            if row["stage"] in by_stage:
                raise RuntimeError(
                    f"layer {layer} batch {batch} duplicates {row['stage']}"
                )
            by_stage[row["stage"]] = row
        if set(by_stage) != set(FFN_STAGES):
            missing = sorted(set(FFN_STAGES) - set(by_stage))
            extra = sorted(set(by_stage) - set(FFN_STAGES))
            raise RuntimeError(
                f"layer {layer} batch {batch} FFN stages mismatch; "
                f"missing={missing} extra={extra}"
            )
        for stage in FFN_STAGES:
            row = by_stage[stage]
            assigned.append({
                "layer": layer,
                "batch": batch,
                "pos": control["pos"],
                "width": control["tokens"],
                "stage": stage,
                "ms": row["ms"],
                "ms_per_row": row["ms"] / control["tokens"],
            })
        assigned.append({
            "layer": layer,
            "batch": batch,
            "pos": control["pos"],
            "width": control["tokens"],
            "stage": "ffn_batch_control",
            "ms": control["ms"],
            "ms_per_row": control["ms"] / control["tokens"],
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
    records = parse_profile(stderr_data, layer, stderr_path)
    stats = common.parse_stats(stderr_data, stderr_path)
    return records, stats, {
        "name": name,
        "layer": layer,
        "wall_seconds": wall_seconds,
        "stdout_sha256": common.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


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
            expected = expected_stats["verify_width_evals"][width]
            stages = {}
            for stage in FFN_STAGES + ("ffn_batch_control",):
                stage_rows = [row for row in selected if row["stage"] == stage]
                if len(stage_rows) != expected:
                    raise RuntimeError(
                        f"layer {layer} width {width} has {len(stage_rows)} "
                        f"{stage} rows, expected {expected}"
                    )
                values = [row["ms_per_row"] for row in stage_rows]
                stages[stage] = {
                    "batches": len(values),
                    "median_ms_per_row": statistics.median(values),
                    "mean_ms_per_row": statistics.mean(values),
                    "p90_ms_per_row": percentile(values, 0.9),
                    "max_ms_per_row": max(values),
                }
            substage_total = sum(
                stages[stage]["median_ms_per_row"] for stage in FFN_STAGES
            )
            control = stages["ffn_batch_control"]["median_ms_per_row"]
            layer_widths[str(layer)][str(width)] = {
                "batches": expected,
                "substage_ms_per_row": substage_total,
                "control_ms_per_row": control,
                "substage_vs_control": substage_total / control,
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
            for stage in FFN_STAGES
        }
        substage_total = sum(components.values())
        control_total = sum(
            layer_widths[str(layer)][str(width)]["control_ms_per_row"]
            for layer in LAYERS
        )
        width_totals[str(width)] = {
            "batches": expected_stats["verify_width_evals"][width],
            "substage_ms_per_row": substage_total,
            "control_ms_per_row": control_total,
            "substage_vs_control": substage_total / control_total,
            "components": components,
        }
    width2 = width_totals["2"]["substage_ms_per_row"]
    for width in WIDTHS:
        width_totals[str(width)]["per_row_vs_width2"] = (
            width_totals[str(width)]["substage_ms_per_row"] / width2
        )

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
        for stage in FFN_STAGES
    }
    stage_amortization = {}
    for stage in FFN_STAGES:
        width2_value = width_totals["2"]["components"][stage]
        width5_value = width5[stage]
        stage_amortization[stage] = {
            "width2_ms_per_row": width2_value,
            "width5_ms_per_row": width5_value,
            "width5_vs_width2": width5_value / width2_value,
        }
    return {
        "analysis": "dspark_post_promotion_width_stratified_exact_ffn",
        "threshold": THRESHOLD,
        "task": TASK,
        "layers": list(LAYERS),
        "widths": list(WIDTHS),
        "ffn_stages": list(FFN_STAGES),
        "expected_width_evals": {
            str(width): expected_stats["verify_width_evals"][width]
            for width in WIDTHS
        },
        "layer_widths": layer_widths,
        "sampled_width_totals": width_totals,
        "width5_components": width5_components,
        "stage_amortization": stage_amortization,
        "largest_width5_stage": max(width5, key=width5.get),
        "weakest_amortization_stage": max(
            stage_amortization,
            key=lambda stage: stage_amortization[stage]["width5_vs_width2"],
        ),
    }


def render_report(summary):
    lines = [
        "# DSpark Post-Promotion Width-Stratified Exact FFN Profile",
        "",
        "Synchronized diagnostic only. FFN boundaries change scheduling; do not use these values as throughput measurements.",
        "Every profiled output matched the frozen cumulative HumanEval artifact byte-for-byte.",
        "Internal FFN records are mapped to the enclosing exact-verifier proposal batch.",
        "",
        "## Sampled FFN Totals",
        "",
        "| width | evals | sub-stages ms/row | outer FFN ms/row | "
        "sub-stages/control | vs width 2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["sampled_width_totals"][str(width)]
        lines.append(
            f"| {width} | {item['batches']} | "
            f"{item['substage_ms_per_row']:.3f} | "
            f"{item['control_ms_per_row']:.3f} | "
            f"{item['substage_vs_control']:.3f}x | "
            f"{item['per_row_vs_width2']:.3f}x |"
        )
    lines.extend([
        "",
        "## Width-5 Components",
        "",
        "| stage | sampled ms/row | share | layer 0 | layer 21 | layer 42 |",
        "|:---|---:|---:|---:|---:|---:|",
    ])
    for stage in FFN_STAGES:
        item = summary["width5_components"][stage]
        layers = item["layers"]
        lines.append(
            f"| {stage} | {item['ms_per_row']:.3f} | "
            f"{item['share']:.1%} | {layers['0']:.3f} | "
            f"{layers['21']:.3f} | {layers['42']:.3f} |"
        )
    lines.extend([
        "",
        "## Stage Amortization",
        "",
        "| stage | width 2 ms/row | width 5 ms/row | width 5 / width 2 |",
        "|:---|---:|---:|---:|",
    ])
    for stage in FFN_STAGES:
        item = summary["stage_amortization"][stage]
        lines.append(
            f"| {stage} | {item['width2_ms_per_row']:.3f} | "
            f"{item['width5_ms_per_row']:.3f} | "
            f"{item['width5_vs_width2']:.3f}x |"
        )
    lines.extend([
        "",
        f"- Largest sampled width-5 FFN stage: "
        f"`{summary['largest_width5_stage']}`.",
        f"- Weakest width-2-to-width-5 FFN amortization: "
        f"`{summary['weakest_amortization_stage']}`.",
        "",
        "## Interpretation Limits",
        "",
        "- Widths 2 and 3 have one observation each; their amortization ratios are directional.",
        "- Width 4 has four observations and width 5 has twenty; width-5 medians are the stable optimization guide.",
        "- The sum of separately synchronized sub-stage medians is reconciled against, but need not equal, the outer FFN median.",
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
        raise SystemExit("FFN profile requires the post-promotion cost reference")
    width_reference = load_width_layer_reference(args, cost_reference)
    context = throughput_reference["tasks"][TASK]
    layer_count = layer_profile.inspect_layer_count(args, root)
    if any(layer >= layer_count for layer in LAYERS):
        raise SystemExit(f"profile layers outside model range 0..{layer_count - 1}")

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/post-promotion-width-ffn-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    prompt = run_dir / "prompt.txt"
    for layer in LAYERS:
        print(f"layer {layer}: {command_text(args, prompt, layer)}")
    print(
        "Three synchronized exact-FFN profile processes; every output must "
        "match the frozen cumulative artifact."
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
        "experiment": "dspark_post_promotion_width_stratified_exact_ffn",
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
            "ffn_stages": FFN_STAGES,
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
            width_reference["summary_path"]
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
            assign_exact_ffn_batches(profile_rows, cost_reference["stats"], layer)
        )
        stats_out.append({"layer": layer, **stats})
        runs.append(run)
        common.cooldown(args.cooldown)

    summary = summarize(assigned, cost_reference["stats"])
    report = render_report(summary)
    write_csv(
        run_dir / "stages.csv",
        ("layer", "batch", "pos", "width", "stage", "ms", "ms_per_row"),
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
