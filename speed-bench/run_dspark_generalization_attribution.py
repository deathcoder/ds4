#!/usr/bin/env python3
"""Attribute promoted DSpark runtime cost on frozen math/chat extremes."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform

import run_dspark_generalization_gate as gate
import run_dspark_issue468_comparison as common


TASK_ROLES = (
    ("math_low", "math500_00166"),
    ("math_high", "gsm8k_00333"),
    ("chat_low", "mt_bench_00075"),
    ("chat_high", "alpaca_00115"),
)
TASKS = tuple(task for _, task in TASK_ROLES)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect one promoted-default DSpark stats record on the frozen "
            "low/high scheduled-throughput task in each non-code domain."
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
        default=root / "speed-bench/dspark-generalization",
    )
    parser.add_argument("--throughput-reference", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
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
        parser.error("refusing to run attribution without --confirm-ready")

    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = True
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = None
    args.pairs = 0
    args.warmups = 0
    return args, root


def load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label} {path}: {exc}") from exc


def frozen_extremes(summary):
    tasks = summary.get("tasks", {})
    result = {}
    for domain in ("math", "chat"):
        selected = [
            (label, item["scheduled_vs_baseline"])
            for label, item in tasks.items()
            if item.get("domain") == domain
        ]
        if len(selected) != 6:
            raise ValueError(f"reference must contain six {domain} tasks")
        result[f"{domain}_low"] = min(selected, key=lambda item: item[1])[0]
        result[f"{domain}_high"] = max(selected, key=lambda item: item[1])[0]
    return result


def validate_reference_config(args, metadata):
    if metadata.get("experiment") != "dspark_math_chat_scheduler_generalization":
        raise SystemExit("throughput reference has the wrong experiment kind")
    config = metadata.get("config", {})
    expected = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "instrumented": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f"throughput reference config {key} mismatch: "
                f"{config.get(key)!r} != {value!r}"
            )
    thresholds = config.get("thresholds", {})
    if thresholds.get("threshold_0455") != (
            common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD):
        raise SystemExit("throughput reference scheduled threshold mismatch")
    for key, expected_path in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        actual = metadata.get(key, {}).get("path")
        if actual is None or Path(actual).resolve() != expected_path.resolve():
            raise SystemExit(f"throughput reference {key} path mismatch")


def load_throughput_reference(args, records, provenance):
    summary_path = args.throughput_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "throughput.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("throughput CSV", csv_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing throughput reference {label}: {path}")

    summary = load_json(summary_path, "throughput summary")
    metadata = load_json(metadata_path, "throughput metadata")
    validate_reference_config(args, metadata)
    if summary.get("sample_count") != 12:
        raise SystemExit("throughput reference must contain 12 tasks")
    if summary.get("threshold") != common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD:
        raise SystemExit("throughput summary scheduled threshold mismatch")
    if metadata.get("corpus", {}).get("source_commit") != (
            provenance.get("source_commit")):
        raise SystemExit("throughput reference corpus commit mismatch")

    extremes = frozen_extremes(summary)
    if extremes != dict(TASK_ROLES):
        raise SystemExit(
            f"throughput extrema drifted: {extremes!r} != {dict(TASK_ROLES)!r}"
        )
    record_map = {record["label"]: record for record in records}
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"cannot read throughput reference CSV: {exc}") from exc

    tasks = {}
    for role, task in TASK_ROLES:
        if task not in record_map or task not in summary["tasks"]:
            raise SystemExit(f"frozen attribution task is missing: {task}")
        task_rows = [row for row in rows if row.get("task") == task]
        by_mode = {row.get("mode"): row for row in task_rows}
        if set(by_mode) != set(gate.MODES) or len(task_rows) != len(gate.MODES):
            raise SystemExit(f"throughput reference has incomplete rows for {task}")
        hashes = {row.get("stdout_sha256") for row in task_rows}
        if len(hashes) != 1:
            raise SystemExit(f"throughput reference output mismatch for {task}")
        scheduled = by_mode["threshold_0455"]
        if scheduled.get("threshold") != (
                common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD):
            raise SystemExit(f"scheduled threshold mismatch for {task}")
        output_path = run_dir / scheduled["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing scheduled output for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != scheduled["stdout_sha256"]:
            raise SystemExit(f"scheduled output hash mismatch for {task}")
        record = record_map[task]
        summary_task = summary["tasks"][task]
        for key in ("domain", "dataset", "source_index"):
            if record.get(key) != summary_task.get(key):
                raise SystemExit(f"throughput reference {key} mismatch for {task}")
        tasks[task] = {
            "role": role,
            "record": record,
            "prior": summary_task,
            "output_path": output_path,
            "output_data": output_data,
        }
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "tasks": tasks,
    }


def task_metrics(row, context):
    item = common.summarize_stats([row])[row["prompt"]]
    emitted = row["emitted"]
    proposal_rounds = row["multi_attempts"]
    target_ms = item["target_eval_ms_per_emitted"]
    sidecar_ms = item["generation_sidecar_ms_per_emitted"]
    accounted_ms = item["accounted_generation_ms_per_emitted"]
    prior = context["prior"]
    return {
        "role": context["role"],
        "domain": prior["domain"],
        "dataset": prior["dataset"],
        "source_index": prior["source_index"],
        "prior_scheduled_vs_baseline": prior["scheduled_vs_baseline"],
        "prior_scheduled_vs_fixed": prior["scheduled_vs_fixed"],
        "prior_baseline_ms_per_token": (
            1000.0 / prior["baseline_generation_tps"]
        ),
        "prior_scheduled_ms_per_token": (
            1000.0 / prior["scheduled_generation_tps"]
        ),
        "prior_overhead_ms_per_token": (
            1000.0 / prior["scheduled_generation_tps"] -
            1000.0 / prior["baseline_generation_tps"]
        ),
        "emitted": emitted,
        "proposal_rounds": proposal_rounds,
        "progress_per_proposal": row["avg_depth"],
        "proposal_rounds_per_emitted": proposal_rounds / emitted,
        "target_evals_per_emitted": item["target_evals_per_emitted"],
        "target_positions_per_eval": item["target_positions_per_eval"],
        "target_ms_per_eval": item["target_eval_ms_per_eval"],
        "target_ms_per_emitted": target_ms,
        "sidecar_ms_per_proposal": (
            row["generation_sidecar_ms"] / proposal_rounds
            if proposal_rounds else None
        ),
        "sidecar_ms_per_emitted": sidecar_ms,
        "accounted_ms_per_emitted": accounted_ms,
        "target_accounted_share": (
            target_ms / accounted_ms if accounted_ms else None
        ),
        "target_evals_avoided": item["target_evals_avoided"],
        "batch_attempts": item["batch_attempts"],
        "batch_full": item["batch_full"],
        "batch_partial": item["batch_partial"],
        "batch_fallbacks": item["batch_fallbacks"],
        "source_fallbacks": item["source_fallbacks"],
        "generation_bridge_ms_per_emitted":
            item["generation_bridge_ms_per_emitted"],
        "generation_stage0_ms_per_emitted":
            item["generation_stage0_ms_per_emitted"],
        "generation_stage1_ms_per_emitted":
            item["generation_stage1_ms_per_emitted"],
        "generation_stage2_ms_per_emitted":
            item["generation_stage2_ms_per_emitted"],
        "generation_head_ms_per_emitted":
            item["generation_head_ms_per_emitted"],
        "generation_chain_ms_per_emitted":
            item["generation_chain_ms_per_emitted"],
    }


def ratio(low, high):
    return low / high if high else None


def summarize(rows, reference):
    tasks = {
        row["prompt"]: task_metrics(row, reference["tasks"][row["prompt"]])
        for row in rows
    }
    domains = {}
    for domain in ("math", "chat"):
        low_name = dict(TASK_ROLES)[f"{domain}_low"]
        high_name = dict(TASK_ROLES)[f"{domain}_high"]
        low = tasks[low_name]
        high = tasks[high_name]
        domains[domain] = {
            "low_task": low_name,
            "high_task": high_name,
            "low_high_ratios": {
                "proposal_rounds_per_emitted": ratio(
                    low["proposal_rounds_per_emitted"],
                    high["proposal_rounds_per_emitted"],
                ),
                "target_ms_per_emitted": ratio(
                    low["target_ms_per_emitted"],
                    high["target_ms_per_emitted"],
                ),
                "sidecar_ms_per_emitted": ratio(
                    low["sidecar_ms_per_emitted"],
                    high["sidecar_ms_per_emitted"],
                ),
                "accounted_ms_per_emitted": ratio(
                    low["accounted_ms_per_emitted"],
                    high["accounted_ms_per_emitted"],
                ),
            },
        }
    return {
        "analysis": "promoted_dspark_cross_domain_runtime_attribution",
        "threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        "tasks": tasks,
        "domains": domains,
    }


def render_report(summary):
    lines = [
        "# DSpark Promoted Runtime Cross-Domain Attribution",
        "",
        "Instrumented diagnostic only. Throughput values from these runs are intentionally omitted.",
        "Each promoted-default runtime output matched the previously validated scheduled output byte-for-byte.",
        "The four tasks are the frozen low/high scheduled-versus-baseline extremes in math and chat.",
        "",
        "## Runtime Costs",
        "",
        "| task | role | prior DSpark/base | progress/round | rounds/emitted | "
        "target evals/emitted | positions/eval | target ms/emitted | "
        "sidecar ms/emitted | target share |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary["tasks"][task]
        lines.append(
            f"| {task} | {item['role']} | "
            f"{item['prior_scheduled_vs_baseline']:.4f}x | "
            f"{item['progress_per_proposal']:.3f} | "
            f"{item['proposal_rounds_per_emitted']:.4f} | "
            f"{item['target_evals_per_emitted']:.4f} | "
            f"{item['target_positions_per_eval']:.3f} | "
            f"{item['target_ms_per_emitted']:.3f} | "
            f"{item['sidecar_ms_per_emitted']:.3f} | "
            f"{item['target_accounted_share']:.1%} |"
        )
    lines.extend([
        "",
        "## Sidecar Breakdown",
        "",
    ])
    for task in TASKS:
        item = summary["tasks"][task]
        lines.append(
            f"- {task}: {item['sidecar_ms_per_proposal']:.3f} ms/proposal; "
            f"bridge {item['generation_bridge_ms_per_emitted']:.3f} ms, "
            f"stages {item['generation_stage0_ms_per_emitted']:.3f}/"
            f"{item['generation_stage1_ms_per_emitted']:.3f}/"
            f"{item['generation_stage2_ms_per_emitted']:.3f} ms, "
            f"head {item['generation_head_ms_per_emitted']:.3f} ms, "
            f"chain {item['generation_chain_ms_per_emitted']:.3f} ms per "
            "emitted token."
        )
    lines.extend([
        "",
        "## Low/High Contrast",
        "",
        "| domain | low task | high task | rounds/emitted | target cost | "
        "sidecar cost | accounted cost |",
        "|:---|:---|:---|---:|---:|---:|---:|",
    ])
    for domain in ("math", "chat"):
        item = summary["domains"][domain]
        values = item["low_high_ratios"]
        lines.append(
            f"| {domain} | {item['low_task']} | {item['high_task']} | "
            f"{values['proposal_rounds_per_emitted']:.3f}x | "
            f"{values['target_ms_per_emitted']:.3f}x | "
            f"{values['sidecar_ms_per_emitted']:.3f}x | "
            f"{values['accounted_ms_per_emitted']:.3f}x |"
        )
    lines.extend([
        "",
        "Historical uninstrumented latency context:",
    ])
    for task in TASKS:
        item = summary["tasks"][task]
        lines.append(
            f"- {task}: baseline {item['prior_baseline_ms_per_token']:.3f} "
            f"ms/token, scheduled {item['prior_scheduled_ms_per_token']:.3f} "
            f"ms/token, overhead {item['prior_overhead_ms_per_token']:.3f} "
            "ms/token."
        )
    lines.extend([
        "",
        "- Component timings are synchronized attribution and are not additive with the historical uninstrumented latency.",
        "- The promoted runtime inherits threshold `0.455`; the command intentionally carries no threshold override.",
        "- No fixed-K process, fresh baseline process, acceptance audit, trace, or layer profiler is run.",
        "- Use target-versus-sidecar share and low/high amplification to choose the next optimization; do not report diagnostic t/s.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    fields = (
        "prompt", "mode", "wall_seconds", "stdout_sha256", "stdout_file",
        "stderr_file",
    ) + common.STATS_FIELDS
    common.write_csv(path, rows, fields)


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    records, provenance, samples_path, provenance_path = gate.load_corpus(
        args, root
    )
    reference = load_throughput_reference(args, records, provenance)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/dspark-generalization-attribution-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = gate.prompt_paths(run_dir, [
        reference["tasks"][task]["record"] for task in TASKS
    ])
    for task in TASKS:
        print(
            f"{task} promoted stats runtime: "
            f"{common.command_text(args, prompts[task], 'runtime', stats=True)}"
        )
    print(
        "Cross-domain attribution: four instrumented promoted-default runtime "
        "processes; outputs reuse the validated generalization references."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    gate.materialize_prompts(
        prompts, [reference["tasks"][task]["record"] for task in TASKS]
    )
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": common.git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "experiment": "dspark_promoted_cross_domain_attribution",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("DS4_")
        },
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "tasks": TASK_ROLES,
            "runtime_threshold_override": None,
            "effective_threshold":
                common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
            "instrumented": True,
            "runtime_processes": len(TASKS),
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "corpus": {
            "samples": common.file_metadata(samples_path),
            "provenance": common.file_metadata(provenance_path),
            "source_commit": provenance["source_commit"],
        },
        "throughput_reference": {
            "summary": common.file_metadata(reference["summary_path"]),
            "metadata": common.file_metadata(reference["metadata_path"]),
            "csv": common.file_metadata(reference["csv_path"]),
        },
        "commands": {
            task: common.command_text(
                args, prompts[task], "runtime", stats=True
            )
            for task in TASKS
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for task in TASKS:
        context = reference["tasks"][task]
        row, _ = common.execute(
            args, root, run_dir, "stats", task, prompts[task],
            "runtime", context["output_data"], stats=True,
        )
        rows.append(row)
        common.cooldown(args.cooldown)

    summary = summarize(rows, reference)
    report = render_report(summary)
    write_csv(run_dir / "stats.csv", rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw stats: {run_dir / 'stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
