#!/usr/bin/env python3
"""Profile layer-42 serial-attention-tail components by verifier width."""

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

import run_dspark_exact_attention_tail_profile as tail_profile
import run_dspark_exact_layer_profile as layer_profile
import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_threshold075_cost_audit as cost_audit
import run_dspark_issue468_comparison as common
import run_dspark_threshold075_width_layer_profile as width_profile


THRESHOLD = width_profile.THRESHOLD
TASK = width_profile.TASK
LAYER = 42
WIDTHS = width_profile.WIDTHS


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile layer-42 serial attention-tail components by actual "
            "verifier width on frozen HumanEval task 079."
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
    parser.add_argument("--layer-reference", type=Path, required=True)
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


def load_layer_reference(args, cost_reference):
    summary_path = args.layer_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    stages_path = run_dir / "stages.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("stages CSV", stages_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing width-layer reference {label}: {path}")
    summary = load_json(summary_path, "width-layer summary")
    metadata = load_json(metadata_path, "width-layer metadata")
    if metadata.get("experiment") != (
        "dspark_threshold075_width_stratified_exact_layer"
    ):
        raise SystemExit("width-layer reference has the wrong experiment kind")
    if summary.get("analysis") != (
        "dspark_threshold075_width_stratified_exact_layer"
    ):
        raise SystemExit("width-layer reference has the wrong analysis kind")
    if summary.get("threshold") != THRESHOLD or summary.get("task") != TASK:
        raise SystemExit("width-layer reference policy mismatch")
    if tuple(summary.get("layers", ())) != width_profile.LAYERS:
        raise SystemExit("width-layer reference layer contract mismatch")
    if tuple(summary.get("widths", ())) != WIDTHS:
        raise SystemExit("width-layer reference width contract mismatch")
    referenced = metadata.get("cost_reference", {}).get("summary", {}).get("path")
    if (
        referenced is None or
        Path(referenced).resolve() != cost_reference["summary_path"].resolve()
    ):
        raise SystemExit("width-layer cost reference mismatch")
    expected_counts = {
        str(width): cost_reference["stats"]["verify_width_evals"][width]
        for width in WIDTHS
    }
    if summary.get("expected_width_evals") != expected_counts:
        raise SystemExit("width-layer reference count mismatch")
    if str(LAYER) not in summary.get("layer_widths", {}):
        raise SystemExit(f"width-layer reference omits layer {LAYER}")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "stages_path": stages_path,
        "summary": summary,
        "metadata": metadata,
    }


def profile_env():
    env = common.benchmark_env(
        "runtime",
        False,
        stats=True,
        confidence_threshold=THRESHOLD,
    )
    env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
    env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(LAYER)
    env["DS4_METAL_DECODE_STAGE_PROFILE"] = "1"
    env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"] = str(LAYER)
    env["DS4_DSPARK_EXACT_TAIL_PROFILE"] = "1"
    return env


def command_text(args, prompt):
    env = profile_env()
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_STATS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
        "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
        "DS4_DSPARK_EXACT_TAIL_PROFILE",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys)
    return prefix + " " + shlex.join(common.mode_command(args, prompt, "runtime"))


def execute(args, root, run_dir, prompt, reference):
    stdout_path = run_dir / "layer-42.stdout"
    stderr_path = run_dir / "layer-42.stderr"
    print(f"[layer-42] {command_text(args, prompt)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            common.mode_command(args, prompt, "runtime"),
            cwd=root,
            env=profile_env(),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"layer-42 failed with exit {completed.returncode}; see {stderr_path}"
        )
    stdout_data = stdout_path.read_bytes()
    if stdout_data != reference:
        raise RuntimeError(
            "layer-42 output differs from frozen threshold-0.75 output; "
            f"see {stdout_path}"
        )
    stderr_data = stderr_path.read_bytes()
    records = tail_profile.parse_profile(stderr_data, LAYER, stderr_path)
    stats = common.parse_stats(stderr_data, stderr_path)
    return records, stats, {
        "name": "layer-42",
        "layer": LAYER,
        "wall_seconds": wall_seconds,
        "stdout_sha256": common.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def proposal_signature(records):
    exact = [row for row in records if row["part"] == "exact"]
    signatures = []
    for stage in tail_profile.CONTROL_STAGES:
        stage_rows = [row for row in exact if row["stage"] == stage]
        if not stage_rows:
            raise RuntimeError(f"missing exact control stage {stage}")
        signatures.append(
            tuple((row["pos"], row["tokens"]) for row in stage_rows)
        )
    if len(set(signatures)) != 1:
        raise RuntimeError("exact control stages have different batch schedules")
    return signatures[0]


def assign_tail_batches(records):
    signature = proposal_signature(records)
    tail = [row for row in records if row["part"] == "tail"]
    assigned = []
    for stage in tail_profile.TAIL_STAGES:
        stage_rows = [row for row in tail if row["stage"] == stage]
        cursor = 0
        for batch_index, (start, width) in enumerate(signature, start=1):
            batch_rows = stage_rows[cursor:cursor + width]
            if len(batch_rows) != width:
                raise RuntimeError(
                    f"{stage} ends inside batch {batch_index} width {width}"
                )
            expected_positions = list(range(start, start + width))
            positions = [row["pos"] for row in batch_rows]
            if positions != expected_positions:
                raise RuntimeError(
                    f"{stage} batch {batch_index} positions {positions} "
                    f"do not match {expected_positions}"
                )
            if any(row["tokens"] != 1 for row in batch_rows):
                raise RuntimeError(f"{stage} batch {batch_index} has non-row data")
            assigned.append({
                "batch": batch_index,
                "start": start,
                "width": width,
                "stage": stage,
                "ms": sum(row["ms"] for row in batch_rows),
                "ms_per_row": sum(row["ms"] for row in batch_rows) / width,
            })
            cursor += width
        if cursor != len(stage_rows):
            raise RuntimeError(f"{stage} has unassigned tail rows")
    return signature, assigned


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(records, expected_stats):
    signature, assigned = assign_tail_batches(records)
    signature_counts = {
        width: sum(batch_width == width for _, batch_width in signature)
        for width in WIDTHS
    }
    for width in WIDTHS:
        expected = expected_stats["verify_width_evals"][width]
        if signature_counts[width] != expected:
            raise RuntimeError(
                f"width {width} profile has {signature_counts[width]} batches, "
                f"expected {expected}"
            )

    widths = {}
    for width in WIDTHS:
        selected = [row for row in assigned if row["width"] == width]
        components = {}
        for stage in tail_profile.TAIL_STAGES:
            stage_rows = [row for row in selected if row["stage"] == stage]
            expected = signature_counts[width]
            if len(stage_rows) != expected:
                raise RuntimeError(
                    f"width {width} has {len(stage_rows)} {stage} batches, "
                    f"expected {expected}"
                )
            values = [row["ms_per_row"] for row in stage_rows]
            components[stage] = {
                "batches": len(values),
                "median_ms_per_row": statistics.median(values),
                "mean_ms_per_row": statistics.mean(values),
                "p90_ms_per_row": percentile(values, 0.9),
                "max_ms_per_row": max(values),
            }
        batch_totals = {}
        for row in selected:
            batch_totals[row["batch"]] = (
                batch_totals.get(row["batch"], 0.0) + row["ms_per_row"]
            )
        total_values = list(batch_totals.values())
        total_median = statistics.median(total_values)
        for item in components.values():
            item["median_share"] = item["median_ms_per_row"] / sum(
                value["median_ms_per_row"] for value in components.values()
            )
        widths[str(width)] = {
            "batches": signature_counts[width],
            "median_tail_ms_per_row": total_median,
            "mean_tail_ms_per_row": statistics.mean(total_values),
            "p90_tail_ms_per_row": percentile(total_values, 0.9),
            "max_tail_ms_per_row": max(total_values),
            "components": components,
        }

    width2 = widths["2"]
    for width in WIDTHS:
        item = widths[str(width)]
        item["tail_vs_width2"] = (
            item["median_tail_ms_per_row"] /
            width2["median_tail_ms_per_row"]
        )
        for stage in tail_profile.TAIL_STAGES:
            item["components"][stage]["per_row_vs_width2"] = (
                item["components"][stage]["median_ms_per_row"] /
                width2["components"][stage]["median_ms_per_row"]
            )

    width5 = widths["5"]
    largest_width5_component = max(
        tail_profile.TAIL_STAGES,
        key=lambda stage:
            width5["components"][stage]["median_ms_per_row"],
    )
    weakest_amortization_component = max(
        tail_profile.TAIL_STAGES,
        key=lambda stage:
            width5["components"][stage]["per_row_vs_width2"],
    )
    return {
        "analysis": "dspark_threshold075_width_stratified_attention_tail",
        "threshold": THRESHOLD,
        "task": TASK,
        "layer": LAYER,
        "widths": list(WIDTHS),
        "profiled_batches": len(signature),
        "profiled_rows": sum(width for _, width in signature),
        "width_results": widths,
        "largest_width5_component": largest_width5_component,
        "weakest_amortization_component": weakest_amortization_component,
        "proposal_signature": [
            {"start": start, "width": width}
            for start, width in signature
        ],
    }, assigned


def render_report(summary):
    lines = [
        "# DSpark Threshold 0.75 Width-Stratified Serial Tail Profile",
        "",
        "Synchronized diagnostic only. Boundaries change Metal scheduling; do not use these values as throughput measurements.",
        "Every profiled output matched the frozen threshold-0.75 HumanEval artifact byte-for-byte.",
        "One-row tail events are assigned sequentially to their enclosing exact-verifier batch.",
        "",
        "## Tail Totals",
        "",
        "| width | evals | median tail ms/row | vs width 2 |",
        "|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["width_results"][str(width)]
        lines.append(
            f"| {width} | {item['batches']} | "
            f"{item['median_tail_ms_per_row']:.3f} | "
            f"{item['tail_vs_width2']:.3f}x |"
        )
    lines.extend([
        "",
        "## Components",
        "",
        "| width | KV/cache | compressor/indexer | attention | inverse RoPE | "
        "projection A | projection B + HC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for width in WIDTHS:
        components = summary["width_results"][str(width)]["components"]
        lines.append(
            f"| {width} | "
            f"{components['kv_cache_update']['median_ms_per_row']:.3f} | "
            f"{components['compressor_indexer']['median_ms_per_row']:.3f} | "
            f"{components['attention']['median_ms_per_row']:.3f} | "
            f"{components['inverse_rope']['median_ms_per_row']:.3f} | "
            f"{components['projection_a']['median_ms_per_row']:.3f} | "
            f"{components['projection_b_hc']['median_ms_per_row']:.3f} |"
        )
    width5 = summary["width_results"]["5"]
    lines.extend([
        "",
        "## Width 5 Distribution",
        "",
        "| component | median ms/row | share | width 5 / width 2 |",
        "|:---|---:|---:|---:|",
    ])
    for stage in tail_profile.TAIL_STAGES:
        item = width5["components"][stage]
        lines.append(
            f"| {stage} | {item['median_ms_per_row']:.3f} | "
            f"{item['median_share']:.1%} | "
            f"{item['per_row_vs_width2']:.3f}x |"
        )
    lines.extend([
        "",
        f"- Largest width-5 component: "
        f"`{summary['largest_width5_component']}`.",
        f"- Weakest width-2-to-width-5 amortization: "
        f"`{summary['weakest_amortization_component']}`.",
        "",
        "## Interpretation Limits",
        "",
        "- Widths 2 and 3 have one batch each; their component ratios are directional.",
        "- Width 4 has four batches and width 5 has twenty, making width 5 the stable optimization target.",
        "- Per-component medians do not necessarily sum to the median of per-batch tail totals.",
        "- The synchronized component boundaries inflate absolute times and may alter overlap.",
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
        "throughput_reference", "cost_reference", "layer_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records,
        cost_audit.TASK_COUNT,
        provenance["selection_policy"],
    )
    throughput_reference = cost_audit.load_throughput_reference(
        args, records, selection
    )
    cost_reference = width_profile.load_cost_reference(args)
    layer_reference = load_layer_reference(args, cost_reference)
    context = throughput_reference["tasks"][TASK]

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/threshold075-width-tail-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompt = run_dir / "prompt.txt"
    print(f"layer {LAYER}: {command_text(args, prompt)}")
    print(
        "One synchronized threshold-0.75 tail profile process; output and "
        "width counts must match the frozen artifacts."
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
        "experiment": "dspark_threshold075_width_stratified_attention_tail",
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
            "layer": LAYER,
            "widths": WIDTHS,
            "control_components": tail_profile.CONTROL_STAGES,
            "tail_components": tail_profile.TAIL_STAGES,
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
        "layer_reference": {
            "summary": common.file_metadata(layer_reference["summary_path"]),
            "metadata": common.file_metadata(layer_reference["metadata_path"]),
            "stages": common.file_metadata(layer_reference["stages_path"]),
        },
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "command": command_text(args, prompt),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    profile_records, stats, run = execute(
        args,
        root,
        run_dir,
        prompt,
        context["output_data"],
    )
    width_profile.validate_counts(stats, cost_reference["stats"], LAYER)
    summary, assigned = summarize(profile_records, cost_reference["stats"])
    report = render_report(summary)
    write_csv(
        run_dir / "stages.csv",
        ("part", "layer", "pos", "tokens", "stage", "ms"),
        profile_records,
    )
    write_csv(
        run_dir / "assigned.csv",
        ("batch", "start", "width", "stage", "ms", "ms_per_row"),
        assigned,
    )
    write_csv(
        run_dir / "runs.csv",
        (
            "name", "layer", "wall_seconds", "stdout_sha256",
            "stdout_file", "stderr_file",
        ),
        [run],
    )
    common.write_csv(
        run_dir / "stats.csv",
        [stats],
        common.STATS_FIELDS,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw assigned stages: {run_dir / 'assigned.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
