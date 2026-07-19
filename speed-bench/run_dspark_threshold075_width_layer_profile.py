#!/usr/bin/env python3
"""Profile exact-layer scaling by verifier width under threshold 0.75."""

import argparse
import ast
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
import run_dspark_humaneval_cumulative_cost_audit as cumulative_cost_audit
import run_dspark_humaneval_threshold075_cost_audit as cost_audit
import run_dspark_issue468_comparison as common


THRESHOLD = cost_audit.THRESHOLD
TASK = "humaneval_079"
LAYERS = (0, 21, 42)
WIDTHS = (2, 3, 4, 5)
CUMULATIVE_COST_SOURCE_COMMIT = "2863d7e26efcfd3e419691b2ae3d0547952eb886"
LEGACY_COST_CONTRACT = (
    "dspark_humaneval_threshold075_exact_verifier_cost",
    "dspark_humaneval_threshold075_exact_verifier_cost",
)
CUMULATIVE_COST_CONTRACT = (
    "dspark_humaneval_cumulative_exact_verifier_cost",
    "dspark_humaneval_cumulative_exact_verifier_cost",
)
COUNT_FIELDS = (
    "emitted",
    "target_evals",
    "target_eval_tokens",
    "batch_attempts",
    "batch_full",
    "batch_partial",
    "batch_fallbacks",
    "scheduler_width_rounds",
    "scheduler_width_committed",
    "verify_width_evals",
    "verify_width_positions",
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile attention-pre, serial-tail, and FFN exact-layer costs "
            "by verifier width on frozen HumanEval task 079."
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


def parse_csv_stats_row(row):
    values = {}
    for field in common.STATS_FIELDS:
        raw = row[field]
        if field in common.INT_STATS:
            values[field] = int(raw)
        elif field in common.FLOAT_STATS:
            values[field] = float(raw)
        elif field in common.INT_ARRAY_STATS:
            parsed = ast.literal_eval(raw)
            values[field] = [int(item) for item in parsed]
        elif field in common.FLOAT_ARRAY_STATS:
            parsed = ast.literal_eval(raw)
            values[field] = [float(item) for item in parsed]
    return values


def cost_reference_kind(summary, metadata):
    contract = (metadata.get("experiment"), summary.get("analysis"))
    if contract == LEGACY_COST_CONTRACT:
        return "legacy_threshold075"
    if contract == CUMULATIVE_COST_CONTRACT:
        return "post_promotion_cumulative"
    raise SystemExit("cost reference has the wrong experiment or analysis kind")


def load_cost_reference(args):
    summary_path = args.cost_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    stats_path = run_dir / "stats.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("stats CSV", stats_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing threshold-0.75 cost {label}: {path}")
    summary = load_json(summary_path, "exact-verifier cost summary")
    metadata = load_json(metadata_path, "exact-verifier cost metadata")
    reference_kind = cost_reference_kind(summary, metadata)
    if reference_kind == "post_promotion_cumulative":
        if metadata.get("git_commit") != CUMULATIVE_COST_SOURCE_COMMIT:
            raise SystemExit("cumulative cost reference source commit mismatch")
        if metadata.get("git_status_tracked"):
            raise SystemExit("cumulative cost reference used a dirty tree")
    if summary.get("threshold") != THRESHOLD:
        raise SystemExit("cost reference threshold mismatch")
    if summary.get("task_count") != cost_audit.TASK_COUNT:
        raise SystemExit("cost reference is not the frozen 32-task audit")
    referenced = metadata.get("throughput_reference", {}).get(
        "summary", {}
    ).get("path")
    if (
        referenced is None or
        Path(referenced).resolve() != args.throughput_reference.resolve()
    ):
        raise SystemExit("cost reference throughput reference mismatch")
    config = metadata.get("config", {})
    expected_config = [
        ("ctx", args.ctx),
        ("tokens", args.tokens),
        ("temperature", 0),
        ("seed", 1),
        ("nothink", True),
        ("confidence_threshold", THRESHOLD),
        ("runtime_stats", True),
        ("oracle_trace", False),
        ("fast_verifier", False),
    ]
    if reference_kind == "post_promotion_cumulative":
        expected_config.append(("promoted_defaults", True))
    for key, expected in expected_config:
        if config.get(key) != expected:
            raise SystemExit(
                f"cost reference config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    try:
        rows = list(csv.DictReader(stats_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read cost stats CSV: {exc}") from exc
    selected = [row for row in rows if row.get("prompt") == TASK]
    if len(selected) != 1:
        raise SystemExit(f"cost reference has {len(selected)} rows for {TASK}")
    stats = parse_csv_stats_row(selected[0])
    for width in WIDTHS:
        if stats["verify_width_evals"][width] <= 0:
            raise SystemExit(
                f"cost reference has no width-{width} eval for {TASK}"
            )
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "stats_path": stats_path,
        "summary": summary,
        "metadata": metadata,
        "stats": stats,
        "reference_kind": reference_kind,
    }


def profile_env(layer):
    env = common.benchmark_env(
        "runtime",
        False,
        stats=True,
        confidence_threshold=THRESHOLD,
    )
    env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
    env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(layer)
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
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys)
    return prefix + " " + shlex.join(common.mode_command(args, prompt, "runtime"))


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
            f"{name} output differs from frozen HumanEval output; "
            f"see {stdout_path}"
        )
    stderr_data = stderr_path.read_bytes()
    records = layer_profile.parse_profile(stderr_data, layer, stderr_path)
    stats = common.parse_stats(stderr_data, stderr_path)
    return records, stats, {
        "name": name,
        "layer": layer,
        "wall_seconds": wall_seconds,
        "stdout_sha256": common.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def validate_counts(stats, reference_stats, layer):
    for field in COUNT_FIELDS:
        if stats[field] != reference_stats[field]:
            raise RuntimeError(
                f"layer {layer} profile changed {field}: "
                f"{stats[field]!r} != {reference_stats[field]!r}"
            )


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(records, expected_stats):
    stages = {}
    for layer in LAYERS:
        layer_rows = [row for row in records if row["layer"] == layer]
        for width in WIDTHS:
            selected = [row for row in layer_rows if row["tokens"] == width]
            expected_batches = expected_stats["verify_width_evals"][width]
            by_stage = {}
            for stage in layer_profile.EXACT_STAGES:
                stage_rows = [row for row in selected if row["stage"] == stage]
                if len(stage_rows) != expected_batches:
                    raise RuntimeError(
                        f"layer {layer} width {width} has "
                        f"{len(stage_rows)} {stage} records, "
                        f"expected {expected_batches}"
                    )
                raw_ms = [row["ms"] for row in stage_rows]
                per_row = [row["ms"] / width for row in stage_rows]
                by_stage[stage] = {
                    "batches": len(stage_rows),
                    "median_ms_per_eval": statistics.median(raw_ms),
                    "median_ms_per_row": statistics.median(per_row),
                    "mean_ms_per_row": statistics.mean(per_row),
                    "p90_ms_per_row": percentile(per_row, 0.9),
                    "max_ms_per_row": max(per_row),
                }
            signatures = {
                tuple(
                    (row["pos"], row["tokens"])
                    for row in selected if row["stage"] == stage
                )
                for stage in layer_profile.EXACT_STAGES
            }
            if len(signatures) != 1:
                raise RuntimeError(
                    f"layer {layer} width {width} stage schedules differ"
                )
            total_ms_per_eval = sum(
                item["median_ms_per_eval"] for item in by_stage.values()
            )
            total_ms_per_row = sum(
                item["median_ms_per_row"] for item in by_stage.values()
            )
            stages[(layer, width)] = {
                "batches": expected_batches,
                "total_ms_per_eval": total_ms_per_eval,
                "total_ms_per_row": total_ms_per_row,
                "stages": by_stage,
            }

    for layer in LAYERS:
        width2 = stages[(layer, 2)]
        for width in WIDTHS:
            item = stages[(layer, width)]
            item["per_row_vs_width2"] = (
                item["total_ms_per_row"] / width2["total_ms_per_row"]
            )
            for stage in layer_profile.EXACT_STAGES:
                item["stages"][stage]["per_row_vs_width2"] = (
                    item["stages"][stage]["median_ms_per_row"] /
                    width2["stages"][stage]["median_ms_per_row"]
                )

    width_totals = {}
    for width in WIDTHS:
        total_by_stage = {}
        for stage in layer_profile.EXACT_STAGES:
            total_by_stage[stage] = sum(
                stages[(layer, width)]["stages"][stage]["median_ms_per_row"]
                for layer in LAYERS
            )
        total_ms_per_row = sum(total_by_stage.values())
        width_totals[str(width)] = {
            "batches": expected_stats["verify_width_evals"][width],
            "sampled_layer_ms_per_row": total_ms_per_row,
            "sampled_stage_ms_per_row": total_by_stage,
        }
    width2_total = width_totals["2"]["sampled_layer_ms_per_row"]
    for width in WIDTHS:
        width_totals[str(width)]["per_row_vs_width2"] = (
            width_totals[str(width)]["sampled_layer_ms_per_row"] / width2_total
        )

    width5_stages = width_totals["5"]["sampled_stage_ms_per_row"]
    largest_width5_stage = max(width5_stages, key=width5_stages.get)
    stage_amortization = {}
    for stage in layer_profile.EXACT_STAGES:
        width2_value = width_totals["2"]["sampled_stage_ms_per_row"][stage]
        width5_value = width5_stages[stage]
        stage_amortization[stage] = {
            "width2_ms_per_row": width2_value,
            "width5_ms_per_row": width5_value,
            "width5_vs_width2": width5_value / width2_value,
        }
    weakest_amortization_stage = max(
        stage_amortization,
        key=lambda stage: stage_amortization[stage]["width5_vs_width2"],
    )
    return {
        "analysis": "dspark_threshold075_width_stratified_exact_layer",
        "threshold": THRESHOLD,
        "task": TASK,
        "layers": list(LAYERS),
        "widths": list(WIDTHS),
        "expected_width_evals": {
            str(width): expected_stats["verify_width_evals"][width]
            for width in WIDTHS
        },
        "layer_widths": {
            str(layer): {
                str(width): stages[(layer, width)]
                for width in WIDTHS
            }
            for layer in LAYERS
        },
        "sampled_width_totals": width_totals,
        "stage_amortization": stage_amortization,
        "largest_width5_stage": largest_width5_stage,
        "weakest_amortization_stage": weakest_amortization_stage,
    }


def render_report(summary):
    post_promotion = (
        summary.get("reference_kind") == "post_promotion_cumulative"
    )
    title = (
        "# DSpark Post-Promotion Width-Stratified Exact Layer Profile"
        if post_promotion else
        "# DSpark Threshold 0.75 Width-Stratified Exact Layer Profile"
    )
    artifact_label = (
        "frozen cumulative HumanEval artifact"
        if post_promotion else
        "frozen threshold-0.75 HumanEval artifact"
    )
    lines = [
        title,
        "",
        "Synchronized diagnostic only. Stage boundaries change scheduling; do not use these values as throughput measurements.",
        f"Every profiled output matched the {artifact_label} byte-for-byte.",
        "Rows are grouped by the actual exact-verifier width recorded in each layer-stage event.",
        "",
        "## Sampled Layer Totals",
        "",
        "| width | evals | sampled layers ms/row | vs width 2 |",
        "|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["sampled_width_totals"][str(width)]
        lines.append(
            f"| {width} | {item['batches']} | "
            f"{item['sampled_layer_ms_per_row']:.3f} | "
            f"{item['per_row_vs_width2']:.3f}x |"
        )
    lines.extend([
        "",
        "## Layer Components",
        "",
        "| layer | width | evals | total ms/row | attention prep | "
        "serial tail | exact FFN | vs width 2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for layer in LAYERS:
        for width in WIDTHS:
            item = summary["layer_widths"][str(layer)][str(width)]
            stages = item["stages"]
            lines.append(
                f"| {layer} | {width} | {item['batches']} | "
                f"{item['total_ms_per_row']:.3f} | "
                f"{stages['attention_pre_batch']['median_ms_per_row']:.3f} | "
                f"{stages['attention_tail_serial']['median_ms_per_row']:.3f} | "
                f"{stages['ffn_batch']['median_ms_per_row']:.3f} | "
                f"{item['per_row_vs_width2']:.3f}x |"
            )
    lines.extend([
        "",
        "## Stage Amortization",
        "",
        "| stage | width 2 ms/row | width 5 ms/row | width 5 / width 2 |",
        "|:---|---:|---:|---:|",
    ])
    for stage in layer_profile.EXACT_STAGES:
        item = summary["stage_amortization"][stage]
        lines.append(
            f"| {stage} | {item['width2_ms_per_row']:.3f} | "
            f"{item['width5_ms_per_row']:.3f} | "
            f"{item['width5_vs_width2']:.3f}x |"
        )
    lines.extend([
        "",
        f"- Largest sampled width-5 stage: "
        f"`{summary['largest_width5_stage']}`.",
        f"- Weakest width-2-to-width-5 per-row amortization: "
        f"`{summary['weakest_amortization_stage']}`.",
        "",
        "## Interpretation Limits",
        "",
        "- Widths 2 and 3 have one observation each on this frozen task; treat their values as directional anchors.",
        "- Width 4 has four observations and width 5 has twenty, so the width-5 medians are the most stable.",
        "- Only layers 0, 21, and 42 are sampled; the report identifies scaling behavior, not a full-model time sum.",
        "- The synchronized boundaries alter Metal scheduling and inflate absolute stage times.",
        "- No fresh throughput benchmark, acceptance audit, oracle trace, fast verifier, or runtime candidate is enabled.",
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
        "throughput_reference", "cost_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records,
        cost_audit.TASK_COUNT,
        provenance["selection_policy"],
    )
    cost_reference = load_cost_reference(args)
    if cost_reference["reference_kind"] == "post_promotion_cumulative":
        throughput_reference = cumulative_cost_audit.load_throughput_reference(
            args, records, selection
        )
    else:
        throughput_reference = cost_audit.load_throughput_reference(
            args, records, selection
        )
    context = throughput_reference["tasks"][TASK]
    layer_count = layer_profile.inspect_layer_count(args, root)
    if any(layer >= layer_count for layer in LAYERS):
        raise SystemExit(
            f"profile layers outside model range 0..{layer_count - 1}"
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_prefix = (
        "post-promotion-width-layer"
        if cost_reference["reference_kind"] == "post_promotion_cumulative"
        else "threshold075-width-layer"
    )
    default_dir = root / f"speed-bench/local-runs/{run_prefix}-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    prompt = run_dir / "prompt.txt"
    for layer in LAYERS:
        print(f"layer {layer}: {command_text(args, prompt, layer)}")
    print(
        "Three synchronized exact-layer profile processes; every output "
        "must match the frozen HumanEval artifact."
    )
    if args.dry_run:
        print("Dry run only; no prompt materialized and no model execution performed.")
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
            "dspark_post_promotion_width_stratified_exact_layer"
            if cost_reference["reference_kind"] == "post_promotion_cumulative"
            else "dspark_threshold075_width_stratified_exact_layer"
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
            "components": layer_profile.EXACT_STAGES,
            "runtime_stats": True,
            "synchronized_profile": True,
            "timed_throughput": False,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "prompt_sha256": common.sha256(context["prompt_data"]),
        "throughput_reference": {
            "summary": common.file_metadata(
                throughput_reference["summary_path"]
            ),
            "metadata": common.file_metadata(
                throughput_reference["metadata_path"]
            ),
            "csv": common.file_metadata(throughput_reference["csv_path"]),
        },
        "cost_reference": {
            "summary": common.file_metadata(cost_reference["summary_path"]),
            "metadata": common.file_metadata(cost_reference["metadata_path"]),
            "stats": common.file_metadata(cost_reference["stats_path"]),
        },
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "commands": {
            str(layer): command_text(args, prompt, layer)
            for layer in LAYERS
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    records_out = []
    stats_out = []
    runs = []
    for layer in LAYERS:
        selected, stats, run = execute(
            args,
            root,
            run_dir,
            prompt,
            layer,
            context["output_data"],
        )
        validate_counts(stats, cost_reference["stats"], layer)
        records_out.extend(selected)
        stats_out.append({"layer": layer, **stats})
        runs.append(run)
        common.cooldown(args.cooldown)

    summary = summarize(records_out, cost_reference["stats"])
    summary["reference_kind"] = cost_reference["reference_kind"]
    if cost_reference["reference_kind"] == "post_promotion_cumulative":
        summary["analysis"] = (
            "dspark_post_promotion_width_stratified_exact_layer"
        )
    report = render_report(summary)
    write_csv(
        run_dir / "stages.csv",
        ("part", "layer", "pos", "tokens", "stage", "ms"),
        records_out,
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
        run_dir / "stats.csv",
        stats_out,
        ("layer",) + common.STATS_FIELDS,
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
