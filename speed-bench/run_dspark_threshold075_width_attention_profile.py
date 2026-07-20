#!/usr/bin/env python3
"""Profile exact attention routes by verifier width on frozen HumanEval 079."""

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

import run_dspark_exact_attention_transition_profile as attention_profile
import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_cumulative_cost_audit as cumulative_cost_audit
import run_dspark_humaneval_threshold075_cost_audit as cost_audit
import run_dspark_issue468_comparison as common
import run_dspark_threshold075_width_layer_profile as width_profile
import run_dspark_threshold075_width_tail_profile as tail_profile


THRESHOLD = width_profile.THRESHOLD
TASK = width_profile.TASK
LAYER = 42
WIDTHS = width_profile.WIDTHS
MODES = attention_profile.ATTENTION_MODES
TAIL_SOURCE_COMMIT = "a31f69b91545d82d2d881fc05128904ce37424c4"
TAIL_CONTRACT = (
    "dspark_post_promotion_width_stratified_attention_tail",
    "dspark_post_promotion_width_stratified_attention_tail",
)
ATTENTION_SOURCE_COMMIT = "c981252dafeb1253f14cc2761d6d56a53c6d5375"
ATTENTION_CONTRACT = (
    "dspark_post_promotion_width_stratified_attention_routes",
    "dspark_post_promotion_width_stratified_attention_routes",
)
FUSED_GATHER_STAGES = ("prepare", "attention_vec", "attention_reduce")
FUSED_GATHER_RE = re.compile(
    rb"^ds4: Metal FlashAttention prefill stage "
    rb"mode=fused_gather_decode tokens=1 comp=(\d+) keys=(\d+) "
    rb"heads=(\d+) dim=(\d+) window=0 ratio=0 "
    rb"([a-z_]+)=([0-9]+(?:\.[0-9]+)?) ms$"
)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile raw, dense-mixed, and sparse-indexed exact attention "
            "routes by actual verifier width on frozen HumanEval task 079."
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
    parser.add_argument("--tail-reference", type=Path, required=True)
    parser.add_argument("--attention-reference", type=Path)
    parser.add_argument("--fused-gather-stages", action="store_true")
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
    if args.fused_gather_stages and args.attention_reference is None:
        parser.error("--fused-gather-stages requires --attention-reference")
    if not args.fused_gather_stages and args.attention_reference is not None:
        parser.error("--attention-reference requires --fused-gather-stages")

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


def load_tail_reference(args, layer_reference, cost_reference):
    summary_path = args.tail_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    assigned_path = run_dir / "assigned.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("assigned CSV", assigned_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing width-tail reference {label}: {path}")

    summary = load_json(summary_path, "width-tail summary")
    metadata = load_json(metadata_path, "width-tail metadata")
    if (metadata.get("experiment"), summary.get("analysis")) != TAIL_CONTRACT:
        raise SystemExit("width-tail reference has the wrong contract")
    if metadata.get("git_commit") != TAIL_SOURCE_COMMIT:
        raise SystemExit("width-tail source commit mismatch")
    if metadata.get("git_status_tracked"):
        raise SystemExit("width-tail reference used a dirty tree")
    if (
        summary.get("threshold") != THRESHOLD or
        summary.get("task") != TASK or
        summary.get("layer") != LAYER or
        tuple(summary.get("widths", ())) != WIDTHS
    ):
        raise SystemExit("width-tail reference policy mismatch")
    for width in WIDTHS:
        expected = cost_reference["stats"]["verify_width_evals"][width]
        actual = summary.get("width_results", {}).get(str(width), {}).get("batches")
        if actual != expected:
            raise SystemExit(f"width-tail width {width} count mismatch")
    referenced = metadata.get("layer_reference", {}).get("summary", {}).get("path")
    if referenced is None or Path(referenced).resolve() != layer_reference["summary_path"]:
        raise SystemExit("width-tail layer reference mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "assigned_path": assigned_path,
        "summary": summary,
        "metadata": metadata,
    }


def load_attention_reference(args, tail_reference, cost_reference):
    summary_path = args.attention_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    assigned_path = run_dir / "assigned.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("assigned CSV", assigned_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing attention-route reference {label}: {path}")

    summary = load_json(summary_path, "attention-route summary")
    metadata = load_json(metadata_path, "attention-route metadata")
    if (metadata.get("experiment"), summary.get("analysis")) != ATTENTION_CONTRACT:
        raise SystemExit("attention-route reference has the wrong contract")
    if metadata.get("git_commit") != ATTENTION_SOURCE_COMMIT:
        raise SystemExit("attention-route source commit mismatch")
    if metadata.get("git_status_tracked"):
        raise SystemExit("attention-route reference used a dirty tree")
    if (
        summary.get("threshold") != THRESHOLD or
        summary.get("task") != TASK or
        summary.get("layer") != LAYER or
        tuple(summary.get("widths", ())) != WIDTHS
    ):
        raise SystemExit("attention-route reference policy mismatch")
    for width in WIDTHS:
        item = summary.get("width_results", {}).get(str(width), {})
        expected_batches = cost_reference["stats"]["verify_width_evals"][width]
        if item.get("batches") != expected_batches:
            raise SystemExit(f"attention-route width {width} count mismatch")
        dense = item.get("modes", {}).get("dense_mixed")
        if dense is None or dense.get("rows") != item.get("rows"):
            raise SystemExit(
                f"attention-route width {width} is not entirely dense-mixed"
            )
    referenced = metadata.get("tail_reference", {}).get("summary", {}).get("path")
    if referenced is None or Path(referenced).resolve() != tail_reference["summary_path"]:
        raise SystemExit("attention-route tail reference mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "assigned_path": assigned_path,
        "summary": summary,
        "metadata": metadata,
    }


def profile_env(fused_gather_stages=False):
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
    env["DS4_DSPARK_EXACT_ATTENTION_PROFILE"] = "1"
    if fused_gather_stages:
        env["DS4_METAL_FLASH_ATTN_GATHERED_PROFILE"] = "1"
    return env


def command_text(args, prompt, fused_gather_stages=False):
    env = profile_env(fused_gather_stages)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_STATS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
        "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
        "DS4_DSPARK_EXACT_ATTENTION_PROFILE",
        "DS4_METAL_FLASH_ATTN_GATHERED_PROFILE",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(common.mode_command(args, prompt, "runtime"))


def execute(args, root, run_dir, prompt, reference, fused_gather_stages=False):
    stdout_path = run_dir / "layer-42.stdout"
    stderr_path = run_dir / "layer-42.stderr"
    print(
        f"[layer-42] {command_text(args, prompt, fused_gather_stages)}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            common.mode_command(args, prompt, "runtime"),
            cwd=root,
            env=profile_env(fused_gather_stages),
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
            "layer-42 output differs from frozen cumulative output; "
            f"see {stdout_path}"
        )
    stderr_data = stderr_path.read_bytes()
    records = attention_profile.parse_profile(
        stderr_data,
        LAYER,
        stderr_path,
        attention_profile.DEFAULT_VARIANT,
    )
    flash_records = (
        parse_fused_gather_profile(stderr_data, stderr_path)
        if fused_gather_stages else []
    )
    stats = common.parse_stats(stderr_data, stderr_path)
    return records, flash_records, stats, {
        "name": "layer-42",
        "layer": LAYER,
        "wall_seconds": wall_seconds,
        "stdout_sha256": common.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def parse_fused_gather_profile(data, path):
    rows = []
    for line in data.splitlines():
        match = FUSED_GATHER_RE.match(line)
        if not match:
            continue
        comp, keys, heads, dim, stage, elapsed = match.groups()
        stage = stage.decode("ascii")
        if stage not in FUSED_GATHER_STAGES:
            raise RuntimeError(f"unknown fused-gather stage {stage} in {path}")
        rows.append({
            "call": 0,
            "comp": int(comp),
            "keys": int(keys),
            "heads": int(heads),
            "dim": int(dim),
            "stage": stage,
            "ms": float(elapsed),
        })
    if not rows:
        raise RuntimeError(f"no fused-gather stage records found in {path}")
    if len(rows) % len(FUSED_GATHER_STAGES) != 0:
        raise RuntimeError(f"incomplete fused-gather stage sequence in {path}")

    for offset in range(0, len(rows), len(FUSED_GATHER_STAGES)):
        call = rows[offset:offset + len(FUSED_GATHER_STAGES)]
        call_index = offset // len(FUSED_GATHER_STAGES) + 1
        if tuple(row["stage"] for row in call) != FUSED_GATHER_STAGES:
            raise RuntimeError(
                f"fused-gather call {call_index} stage sequence mismatch"
            )
        shape = {
            (row["comp"], row["keys"], row["heads"], row["dim"])
            for row in call
        }
        if len(shape) != 1:
            raise RuntimeError(f"fused-gather call {call_index} changes shape")
        for row in call:
            row["call"] = call_index
    return rows


def assign_attention_batches(records):
    signature = tail_profile.proposal_signature(records)
    attention = [row for row in records if row["part"] == "attention"]
    assigned = []
    cursor = 0
    for batch_index, (start, width) in enumerate(signature, start=1):
        batch_rows = attention[cursor:cursor + width]
        if len(batch_rows) != width:
            raise RuntimeError(
                f"attention records end inside batch {batch_index} width {width}"
            )
        expected_positions = list(range(start, start + width))
        positions = [row["pos"] for row in batch_rows]
        if positions != expected_positions:
            raise RuntimeError(
                f"attention batch {batch_index} positions {positions} "
                f"do not match {expected_positions}"
            )
        if any(row["tokens"] != 1 for row in batch_rows):
            raise RuntimeError(f"attention batch {batch_index} has non-row data")
        for row in batch_rows:
            assigned.append({
                "batch": batch_index,
                "start": start,
                "width": width,
                "pos": row["pos"],
                "mode": row["stage"],
                "ms": row["ms"],
            })
        cursor += width
    if cursor != len(attention):
        raise RuntimeError("attention profile has unassigned rows")
    return signature, assigned


def assign_fused_gather_batches(records, flash_records):
    signature, attention_rows = assign_attention_batches(records)
    if any(row["mode"] != "dense_mixed" for row in attention_rows):
        raise RuntimeError("fused-gather stage profile contains a non-dense row")
    calls = sorted({row["call"] for row in flash_records})
    if len(calls) != len(attention_rows):
        raise RuntimeError(
            f"fused-gather calls {len(calls)} do not match "
            f"dense attention rows {len(attention_rows)}"
        )
    assigned = []
    for attention_row, call_index in zip(attention_rows, calls):
        call_rows = [row for row in flash_records if row["call"] == call_index]
        for row in call_rows:
            assigned.append({
                "batch": attention_row["batch"],
                "start": attention_row["start"],
                "width": attention_row["width"],
                "pos": attention_row["pos"],
                "call": call_index,
                "comp": row["comp"],
                "keys": row["keys"],
                "heads": row["heads"],
                "dim": row["dim"],
                "stage": row["stage"],
                "ms": row["ms"],
            })
    return signature, assigned


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(records, expected_stats):
    signature, assigned = assign_attention_batches(records)
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
        values = [row["ms"] for row in selected]
        modes = {}
        for mode in MODES:
            mode_values = [row["ms"] for row in selected if row["mode"] == mode]
            modes[mode] = None if not mode_values else {
                "rows": len(mode_values),
                "row_share": len(mode_values) / len(values),
                "median_ms_per_row": statistics.median(mode_values),
                "mean_ms_per_row": statistics.mean(mode_values),
                "p90_ms_per_row": percentile(mode_values, 0.9),
                "total_ms": sum(mode_values),
                "cost_share": sum(mode_values) / sum(values),
            }
        widths[str(width)] = {
            "batches": signature_counts[width],
            "rows": len(values),
            "median_attention_ms_per_row": statistics.median(values),
            "mean_attention_ms_per_row": statistics.mean(values),
            "p90_attention_ms_per_row": percentile(values, 0.9),
            "modes": modes,
        }

    width5_modes = widths["5"]["modes"]
    present = {mode: item for mode, item in width5_modes.items() if item}
    dominant = max(present, key=lambda mode: present[mode]["total_ms"])
    slowest = max(present, key=lambda mode: present[mode]["median_ms_per_row"])
    return {
        "analysis": "dspark_post_promotion_width_stratified_attention_routes",
        "threshold": THRESHOLD,
        "task": TASK,
        "layer": LAYER,
        "widths": list(WIDTHS),
        "profiled_batches": len(signature),
        "profiled_rows": len(assigned),
        "width_results": widths,
        "dominant_width5_cost_mode": dominant,
        "slowest_width5_mode": slowest,
        "proposal_signature": [
            {"start": start, "width": width}
            for start, width in signature
        ],
    }, assigned


def summarize_fused_gather(records, flash_records, expected_stats):
    signature, assigned = assign_fused_gather_batches(records, flash_records)
    signature_counts = {
        width: sum(batch_width == width for _, batch_width in signature)
        for width in WIDTHS
    }
    widths = {}
    for width in WIDTHS:
        expected = expected_stats["verify_width_evals"][width]
        if signature_counts[width] != expected:
            raise RuntimeError(
                f"width {width} profile has {signature_counts[width]} batches, "
                f"expected {expected}"
            )
        selected = [row for row in assigned if row["width"] == width]
        calls = sorted({row["call"] for row in selected})
        stages = {}
        for stage in FUSED_GATHER_STAGES:
            values = [row["ms"] for row in selected if row["stage"] == stage]
            if len(values) != len(calls):
                raise RuntimeError(
                    f"width {width} has {len(values)} {stage} records, "
                    f"expected {len(calls)}"
                )
            stages[stage] = {
                "occurrences": len(values),
                "median_ms": statistics.median(values),
                "mean_ms": statistics.mean(values),
                "p90_ms": percentile(values, 0.9),
                "total_ms": sum(values),
            }
        total_ms = sum(item["total_ms"] for item in stages.values())
        call_totals = [
            sum(row["ms"] for row in selected if row["call"] == call)
            for call in calls
        ]
        for item in stages.values():
            item["cost_share"] = item["total_ms"] / total_ms
        widths[str(width)] = {
            "batches": signature_counts[width],
            "rows": len(calls),
            "median_call_ms": statistics.median(call_totals),
            "mean_call_ms": statistics.mean(call_totals),
            "p90_call_ms": percentile(call_totals, 0.9),
            "stages": stages,
        }
    width5 = widths["5"]["stages"]
    dominant = max(
        FUSED_GATHER_STAGES,
        key=lambda stage: width5[stage]["total_ms"],
    )
    return {
        "analysis": "dspark_post_promotion_width_stratified_fused_gather",
        "threshold": THRESHOLD,
        "task": TASK,
        "layer": LAYER,
        "widths": list(WIDTHS),
        "profiled_batches": len(signature),
        "profiled_rows": sum(width for _, width in signature),
        "width_results": widths,
        "dominant_width5_stage": dominant,
        "key_range": {
            "min": min(row["keys"] for row in assigned),
            "max": max(row["keys"] for row in assigned),
        },
        "comp_range": {
            "min": min(row["comp"] for row in assigned),
            "max": max(row["comp"] for row in assigned),
        },
        "proposal_signature": [
            {"start": start, "width": width}
            for start, width in signature
        ],
    }, assigned


def mode_value(item):
    return "n/a" if item is None else f"{item['median_ms_per_row']:.3f}"


def render_report(summary):
    lines = [
        "# DSpark Post-Promotion Width-Stratified Attention Route Profile",
        "",
        "Synchronized diagnostic only. Boundaries change Metal scheduling; do not use these values as throughput measurements.",
        "Every profiled output matched the frozen cumulative HumanEval artifact byte-for-byte.",
        "Each attention event is assigned sequentially to its enclosing exact-verifier batch.",
        "",
        "## Width Summary",
        "",
        "| width | evals | rows | attention ms/row | raw rows | raw | dense rows | dense mixed | sparse rows | sparse indexed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["width_results"][str(width)]
        modes = item["modes"]
        row = []
        for mode in MODES:
            mode_item = modes[mode]
            row.extend([
                "0" if mode_item is None else str(mode_item["rows"]),
                mode_value(mode_item),
            ])
        lines.append(
            f"| {width} | {item['batches']} | {item['rows']} | "
            f"{item['median_attention_ms_per_row']:.3f} | "
            + " | ".join(row) + " |"
        )

    lines.extend([
        "",
        "## Width 5 Cost Distribution",
        "",
        "| mode | rows | row share | median ms/row | total ms | cost share |",
        "|:---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        item = summary["width_results"]["5"]["modes"][mode]
        if item is None:
            lines.append(f"| {mode} | 0 | 0.0% | n/a | 0.000 | 0.0% |")
        else:
            lines.append(
                f"| {mode} | {item['rows']} | {item['row_share']:.1%} | "
                f"{item['median_ms_per_row']:.3f} | {item['total_ms']:.3f} | "
                f"{item['cost_share']:.1%} |"
            )
    lines.extend([
        "",
        f"- Dominant width-5 attention cost mode: `"
        f"{summary['dominant_width5_cost_mode']}`.",
        f"- Slowest width-5 attention mode per row: `"
        f"{summary['slowest_width5_mode']}`.",
        "",
        "## Interpretation Limits",
        "",
        "- Widths 2 and 3 have one verifier batch each; treat their route mix as directional.",
        "- Width 5 has twenty batches and is the stable optimization guide.",
        "- Route medians use identical immediate before/after attention-call boundaries.",
        "- Synchronized absolute timings are attribution data, not throughput measurements.",
        "- No runtime candidate, fresh throughput benchmark, acceptance audit, oracle trace, or fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def render_fused_gather_report(summary):
    lines = [
        "# DSpark Post-Promotion Width-Stratified Fused-Gather Profile",
        "",
        "Synchronized diagnostic only. Boundaries change Metal scheduling; do not use these values as throughput measurements.",
        "Every profiled output matched the frozen cumulative HumanEval artifact byte-for-byte.",
        "Each fused-gather dispatch sequence is assigned to its enclosing exact-verifier batch.",
        "",
        f"- Key range: {summary['key_range']['min']}-{summary['key_range']['max']}.",
        f"- Compressed-row range: {summary['comp_range']['min']}-{summary['comp_range']['max']}.",
        "",
        "## Width Summary",
        "",
        "| width | evals | rows | call ms | prepare | attention vec | reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["width_results"][str(width)]
        stages = item["stages"]
        lines.append(
            f"| {width} | {item['batches']} | {item['rows']} | "
            f"{item['median_call_ms']:.3f} | "
            f"{stages['prepare']['median_ms']:.3f} | "
            f"{stages['attention_vec']['median_ms']:.3f} | "
            f"{stages['attention_reduce']['median_ms']:.3f} |"
        )
    lines.extend([
        "",
        "## Width 5 Cost Distribution",
        "",
        "| stage | occurrences | median ms | total ms | cost share |",
        "|:---|---:|---:|---:|---:|",
    ])
    for stage in FUSED_GATHER_STAGES:
        item = summary["width_results"]["5"]["stages"][stage]
        lines.append(
            f"| {stage} | {item['occurrences']} | {item['median_ms']:.3f} | "
            f"{item['total_ms']:.3f} | {item['cost_share']:.1%} |"
        )
    lines.extend([
        "",
        f"- Dominant width-5 fused-gather stage: `"
        f"{summary['dominant_width5_stage']}`.",
        "",
        "## Interpretation Limits",
        "",
        "- Widths 2 and 3 have one verifier batch each; treat their stage medians as directional.",
        "- Width 5 has twenty batches and is the stable optimization guide.",
        "- Preparation, vector attention, and reduction retain their production operation order.",
        "- Synchronized absolute timings are attribution data, not throughput measurements.",
        "- No runtime candidate, fresh throughput benchmark, acceptance audit, oracle trace, or fast verifier is enabled.",
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
        "tail_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.attention_reference is not None:
        args.attention_reference = args.attention_reference.resolve()
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records,
        cost_audit.TASK_COUNT,
        provenance["selection_policy"],
    )
    cost_reference = width_profile.load_cost_reference(args)
    throughput_reference = cumulative_cost_audit.load_throughput_reference(
        args, records, selection
    )
    layer_reference = tail_profile.load_layer_reference(args, cost_reference)
    tail_reference = load_tail_reference(
        args, layer_reference, cost_reference
    )
    attention_reference = (
        load_attention_reference(args, tail_reference, cost_reference)
        if args.fused_gather_stages else None
    )
    context = throughput_reference["tasks"][TASK]

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_prefix = (
        "post-promotion-width-fused-gather"
        if args.fused_gather_stages else
        "post-promotion-width-attention"
    )
    default_dir = root / f"speed-bench/local-runs/{run_prefix}-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    prompt = run_dir / "prompt.txt"
    print(
        f"layer {LAYER}: "
        f"{command_text(args, prompt, args.fused_gather_stages)}"
    )
    print(
        "One synchronized "
        + ("fused-gather stage" if args.fused_gather_stages else "attention-route")
        + " profile process; output and width counts must match the frozen artifacts."
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
            "dspark_post_promotion_width_stratified_fused_gather"
            if args.fused_gather_stages else
            "dspark_post_promotion_width_stratified_attention_routes"
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
            "layer": LAYER,
            "widths": WIDTHS,
            "attention_modes": MODES,
            "fused_gather_stages": (
                FUSED_GATHER_STAGES if args.fused_gather_stages else ()
            ),
            "runtime_stats": True,
            "synchronized_profile": True,
            "timed_throughput": False,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "prompt_sha256": common.sha256(context["prompt_data"]),
        "throughput_reference": {
            "summary": common.file_metadata(throughput_reference["summary_path"]),
            "metadata": common.file_metadata(throughput_reference["metadata_path"]),
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
        "tail_reference": {
            "summary": common.file_metadata(tail_reference["summary_path"]),
            "metadata": common.file_metadata(tail_reference["metadata_path"]),
            "assigned": common.file_metadata(tail_reference["assigned_path"]),
        },
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "command": command_text(args, prompt, args.fused_gather_stages),
    }
    if attention_reference is not None:
        metadata["attention_reference"] = {
            "summary": common.file_metadata(attention_reference["summary_path"]),
            "metadata": common.file_metadata(attention_reference["metadata_path"]),
            "assigned": common.file_metadata(attention_reference["assigned_path"]),
        }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    profile_records, flash_records, stats, run = execute(
        args,
        root,
        run_dir,
        prompt,
        context["output_data"],
        args.fused_gather_stages,
    )
    width_profile.validate_counts(stats, cost_reference["stats"], LAYER)
    if args.fused_gather_stages:
        summary, assigned = summarize_fused_gather(
            profile_records, flash_records, cost_reference["stats"]
        )
        report = render_fused_gather_report(summary)
        assigned_fields = (
            "batch", "start", "width", "pos", "call", "comp", "keys",
            "heads", "dim", "stage", "ms",
        )
    else:
        summary, assigned = summarize(profile_records, cost_reference["stats"])
        report = render_report(summary)
        assigned_fields = ("batch", "start", "width", "pos", "mode", "ms")
    write_csv(
        run_dir / "stages.csv",
        ("variant", "part", "layer", "pos", "tokens", "stage", "ms"),
        profile_records,
    )
    write_csv(
        run_dir / "assigned.csv",
        assigned_fields,
        assigned,
    )
    if flash_records:
        write_csv(
            run_dir / "fused_gather_stages.csv",
            ("call", "comp", "keys", "heads", "dim", "stage", "ms"),
            flash_records,
        )
    write_csv(
        run_dir / "runs.csv",
        (
            "name", "layer", "wall_seconds", "stdout_sha256",
            "stdout_file", "stderr_file",
        ),
        [run],
    )
    common.write_csv(run_dir / "stats.csv", [stats], common.STATS_FIELDS)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw assigned profile rows: {run_dir / 'assigned.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
