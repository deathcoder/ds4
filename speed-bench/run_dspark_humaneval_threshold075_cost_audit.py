#!/usr/bin/env python3
"""Audit exact-verifier costs under the frozen threshold-0.75 schedule."""

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform

import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_threshold075_throughput as confirmation
import run_dspark_issue468_comparison as common


THRESHOLD = confirmation.THRESHOLD
TASK_COUNT = confirmation.SAMPLE_COUNT


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect stats-only exact-runtime costs for the frozen "
            "threshold-0.75 HumanEval workload."
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
        parser.error("refusing to run cost audit without --confirm-ready")

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


def validate_metadata_path(metadata, key, expected):
    actual = metadata.get(key, {}).get("path")
    if actual is None or Path(actual).resolve() != expected.resolve():
        raise SystemExit(f"threshold-0.75 reference {key} path mismatch")


def load_throughput_reference(args, records, selection):
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
            raise SystemExit(f"missing threshold-0.75 reference {label}: {path}")

    summary = load_json(summary_path, "threshold-0.75 summary")
    metadata = load_json(metadata_path, "threshold-0.75 metadata")
    if metadata.get("experiment") != "dspark_humaneval_threshold075_throughput":
        raise SystemExit("throughput reference has the wrong experiment kind")
    if summary.get("sample_count") != TASK_COUNT:
        raise SystemExit("throughput reference is not the frozen 32-task study")
    if summary.get("threshold") != THRESHOLD:
        raise SystemExit("throughput reference threshold mismatch")
    if summary.get("selection") != selection:
        raise SystemExit("throughput reference selection mismatch")
    if not summary.get("confirmation_gate", {}).get("pass"):
        raise SystemExit("throughput reference did not pass scheduler confirmation")
    if summary.get("next_path_gate", {}).get("next_path") != (
        "freeze_scheduler_and_optimize_exact_verifier"
    ):
        raise SystemExit("throughput reference did not select verifier optimization")

    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "threshold": THRESHOLD,
        "instrumented": False,
        "measured_pairs_per_task": 1,
        "alternating_order": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"throughput reference config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        validate_metadata_path(metadata, key, expected)

    try:
        csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read threshold-0.75 CSV: {exc}") from exc
    if len(csv_rows) != TASK_COUNT * 2:
        raise SystemExit("threshold-0.75 reference measured-row count mismatch")
    rows_by_task = {}
    for row in csv_rows:
        task_rows = rows_by_task.setdefault(row["prompt"], {})
        if row["mode"] in task_rows:
            raise SystemExit(f"duplicate threshold-0.75 row for {row['prompt']}")
        task_rows[row["mode"]] = row

    tasks = {}
    for record in records:
        task = record["label"]
        sample = summary.get("samples", {}).get(task)
        by_mode = rows_by_task.get(task, {})
        if sample is None or set(by_mode) != {"baseline", "runtime"}:
            raise SystemExit(f"throughput reference has incomplete task {task}")
        baseline_row = by_mode["baseline"]
        runtime_row = by_mode["runtime"]
        if baseline_row["stdout_sha256"] != runtime_row["stdout_sha256"]:
            raise SystemExit(f"throughput reference output mismatch for {task}")
        output_path = run_dir / runtime_row["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing threshold-0.75 output for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != runtime_row["stdout_sha256"]:
            raise SystemExit(f"throughput output hash mismatch for {task}")
        prompt_data = record["turns"][0].encode("utf-8")
        prompt_path = run_dir / "prompts" / f"{task}.txt"
        if not prompt_path.is_file() or prompt_path.read_bytes() != prompt_data:
            raise SystemExit(f"throughput prompt drift for {task}")

        baseline_tps = float(baseline_row["generation_tps"])
        runtime_tps = float(runtime_row["generation_tps"])
        paired_ratio = runtime_tps / baseline_tps
        expected_values = (
            ("baseline_generation_tps", baseline_tps),
            ("runtime_generation_tps", runtime_tps),
            ("paired_ratio", paired_ratio),
        )
        for key, expected in expected_values:
            if not math.isclose(
                sample[key], expected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise SystemExit(f"throughput reference {key} mismatch for {task}")
        tasks[task] = {
            "record": record,
            "sample": sample,
            "output_data": output_data,
            "prompt_data": prompt_data,
            "baseline_tps": baseline_tps,
            "runtime_tps": runtime_tps,
        }
    if set(rows_by_task) != set(tasks):
        raise SystemExit("threshold-0.75 reference task set mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "summary": summary,
        "metadata": metadata,
        "tasks": tasks,
    }


def target_scale_for_parity(baseline_ms, runtime_ms, target_ms):
    if target_ms <= 0:
        return None
    return 1.0 - (runtime_ms - baseline_ms) / target_ms


def accounted_target_scale_for_parity(baseline_ms, sidecar_ms, target_ms):
    if target_ms <= 0:
        return None
    return (baseline_ms - sidecar_ms) / target_ms


def task_metrics(row, context):
    emitted = row["emitted"]
    if emitted <= 0:
        raise RuntimeError("empty threshold-0.75 runtime stats")
    baseline_ms = emitted * 1000.0 / context["baseline_tps"]
    runtime_ms = emitted * 1000.0 / context["runtime_tps"]
    target_ms = row["target_eval_ms"]
    sidecar_ms = row["generation_sidecar_ms"]
    residual_ms = runtime_ms - target_ms - sidecar_ms
    batch_total = row["batch_full"] + row["batch_partial"] + row["batch_fallbacks"]
    if batch_total != row["batch_attempts"]:
        raise RuntimeError("verifier batch outcomes do not reconcile")
    if row["fast_calls"] or row["fast_failures"] or row["fast_exact_fallbacks"]:
        raise RuntimeError("threshold-0.75 audit unexpectedly used the fast verifier")
    return {
        "source_index": context["record"]["source_index"],
        "acceptance_verify_rate": context["sample"]["acceptance_verify_rate"],
        "paired_ratio": context["sample"]["paired_ratio"],
        "emitted": emitted,
        "baseline_ms": baseline_ms,
        "runtime_ms": runtime_ms,
        "runtime_deficit_ms": runtime_ms - baseline_ms,
        "target_ms": target_ms,
        "sidecar_ms": sidecar_ms,
        "residual_ms": residual_ms,
        "target_ms_per_emitted": target_ms / emitted,
        "sidecar_ms_per_emitted": sidecar_ms / emitted,
        "residual_ms_per_emitted": residual_ms / emitted,
        "target_evals": row["target_evals"],
        "target_evals_per_emitted": row["target_evals"] / emitted,
        "target_positions": row["target_eval_tokens"],
        "target_positions_per_eval":
            row["target_eval_tokens"] / row["target_evals"],
        "batch_attempts": row["batch_attempts"],
        "batch_full": row["batch_full"],
        "batch_partial": row["batch_partial"],
        "batch_fallbacks": row["batch_fallbacks"],
        "target_scale_for_parity": target_scale_for_parity(
            baseline_ms, runtime_ms, target_ms
        ),
        "accounted_target_scale_for_parity":
            accounted_target_scale_for_parity(
                baseline_ms, sidecar_ms, target_ms
            ),
    }


def aggregate_widths(rows, count_field, value_fields):
    lengths = {len(row[count_field]) for row in rows}
    if len(lengths) != 1:
        raise RuntimeError(f"inconsistent {count_field} histogram widths")
    width_count = lengths.pop()
    result = {}
    for width in range(width_count):
        counts = sum(row[count_field][width] for row in rows)
        values = {
            field: sum(row[field][width] for row in rows)
            for field in value_fields
        }
        if counts or any(values.values()):
            result[str(width)] = {"count": counts, **values}
    return result


def summarize(rows, reference):
    tasks = {
        row["prompt"]: task_metrics(row, reference["tasks"][row["prompt"]])
        for row in rows
    }
    if set(tasks) != set(reference["tasks"]):
        raise RuntimeError("cost audit task set mismatch")
    emitted = sum(item["emitted"] for item in tasks.values())
    baseline_ms = sum(item["baseline_ms"] for item in tasks.values())
    runtime_ms = sum(item["runtime_ms"] for item in tasks.values())
    target_ms = sum(item["target_ms"] for item in tasks.values())
    sidecar_ms = sum(item["sidecar_ms"] for item in tasks.values())
    residual_ms = runtime_ms - target_ms - sidecar_ms
    target_evals = sum(item["target_evals"] for item in tasks.values())
    target_positions = sum(item["target_positions"] for item in tasks.values())
    aggregate = {
        "emitted": emitted,
        "baseline_ms": baseline_ms,
        "runtime_ms": runtime_ms,
        "runtime_deficit_ms": runtime_ms - baseline_ms,
        "pooled_runtime_ratio": baseline_ms / runtime_ms,
        "target_ms": target_ms,
        "sidecar_ms": sidecar_ms,
        "residual_ms": residual_ms,
        "baseline_ms_per_emitted": baseline_ms / emitted,
        "runtime_ms_per_emitted": runtime_ms / emitted,
        "runtime_deficit_ms_per_emitted": (runtime_ms - baseline_ms) / emitted,
        "target_ms_per_emitted": target_ms / emitted,
        "sidecar_ms_per_emitted": sidecar_ms / emitted,
        "residual_ms_per_emitted": residual_ms / emitted,
        "target_share_of_runtime": target_ms / runtime_ms,
        "sidecar_share_of_runtime": sidecar_ms / runtime_ms,
        "target_evals": target_evals,
        "target_evals_per_emitted": target_evals / emitted,
        "target_positions": target_positions,
        "target_positions_per_eval": target_positions / target_evals,
        "batch_attempts": sum(item["batch_attempts"] for item in tasks.values()),
        "batch_full": sum(item["batch_full"] for item in tasks.values()),
        "batch_partial": sum(item["batch_partial"] for item in tasks.values()),
        "batch_fallbacks":
            sum(item["batch_fallbacks"] for item in tasks.values()),
        "target_scale_for_parity": target_scale_for_parity(
            baseline_ms, runtime_ms, target_ms
        ),
        "accounted_target_scale_for_parity":
            accounted_target_scale_for_parity(
                baseline_ms, sidecar_ms, target_ms
            ),
        "prefill_sidecar_ms": sum(row["prefill_sidecar_ms"] for row in rows),
        "sidecar_outside_scheduler_ms":
            sum(row["sidecar_outside_scheduler_ms"] for row in rows),
    }
    verifier_widths = aggregate_widths(
        rows,
        "verify_width_evals",
        ("verify_width_positions", "verify_width_target_ms"),
    )
    for item in verifier_widths.values():
        if item["count"] <= 0 or item["verify_width_positions"] <= 0:
            raise RuntimeError("invalid nonempty verifier-width bucket")
        item["positions_per_eval"] = item["verify_width_positions"] / item["count"]
        item["target_ms_per_eval"] = item["verify_width_target_ms"] / item["count"]
        item["target_ms_per_position"] = (
            item["verify_width_target_ms"] / item["verify_width_positions"]
        )
        item["target_time_share"] = item["verify_width_target_ms"] / target_ms

    scheduler_widths = aggregate_widths(
        rows,
        "scheduler_width_rounds",
        ("scheduler_width_committed", "scheduler_width_sidecar_ms"),
    )
    for item in scheduler_widths.values():
        if item["count"] <= 0:
            raise RuntimeError("invalid nonempty scheduler-width bucket")
        item["progress_per_round"] = (
            item["scheduler_width_committed"] / item["count"]
        )
        item["sidecar_ms_per_round"] = (
            item["scheduler_width_sidecar_ms"] / item["count"]
        )

    return {
        "analysis": "dspark_humaneval_threshold075_exact_verifier_cost",
        "threshold": THRESHOLD,
        "task_count": len(tasks),
        "reference_paired_ratio_geometric_mean":
            reference["summary"]["paired_ratio_geometric_mean"],
        "tasks": tasks,
        "aggregate": aggregate,
        "verifier_widths": verifier_widths,
        "scheduler_widths": scheduler_widths,
    }


def format_scale(value):
    if value is None:
        return "n/a"
    if value < 0:
        return "impossible even with free target verification"
    if value <= 1.0:
        return f"{value:.3f}x ({1.0 - value:.1%} reduction)"
    return f"{value:.3f}x ({value - 1.0:.1%} headroom)"


def render_report(summary):
    aggregate = summary["aggregate"]
    lines = [
        "# DSpark Threshold 0.75 Exact Verifier Cost Audit",
        "",
        "Instrumented stats-only diagnostic; throughput values are intentionally omitted.",
        "Every runtime output matched the frozen uninstrumented threshold-0.75 artifact byte-for-byte.",
        "Frozen generation times provide the end-to-end budget; fresh stats provide component costs.",
        "",
        "## Aggregate Budget",
        "",
        "| baseline | threshold 0.75 | deficit | target | sidecar | residual |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {aggregate['baseline_ms_per_emitted']:.3f} ms/emitted | "
        f"{aggregate['runtime_ms_per_emitted']:.3f} ms/emitted | "
        f"{aggregate['runtime_deficit_ms_per_emitted']:.3f} ms/emitted | "
        f"{aggregate['target_ms_per_emitted']:.3f} ms/emitted | "
        f"{aggregate['sidecar_ms_per_emitted']:.3f} ms/emitted | "
        f"{aggregate['residual_ms_per_emitted']:.3f} ms/emitted |",
        "",
        f"- Frozen threshold-0.75 geometric mean: "
        f"{summary['reference_paired_ratio_geometric_mean']:.4f}x.",
        f"- Pooled frozen generation-time ratio: "
        f"{aggregate['pooled_runtime_ratio']:.4f}x.",
        f"- Target verification is {aggregate['target_share_of_runtime']:.1%} "
        "of frozen runtime time under cross-run component accounting.",
        f"- Sidecar generation is {aggregate['sidecar_share_of_runtime']:.1%} "
        "of frozen runtime time under cross-run component accounting.",
        f"- End-to-end-calibrated target scale for parity: "
        f"{format_scale(aggregate['target_scale_for_parity'])}.",
        f"- Component-accounted target scale for parity, excluding residual: "
        f"{format_scale(aggregate['accounted_target_scale_for_parity'])}.",
        f"- Prefill sidecar total across processes: "
        f"{aggregate['prefill_sidecar_ms']:.3f} ms.",
        f"- Sidecar outside the scheduler: "
        f"{aggregate['sidecar_outside_scheduler_ms']:.3f} ms.",
        "",
        "## Verifier Workload",
        "",
        f"- Target evals per emitted token: "
        f"{aggregate['target_evals_per_emitted']:.4f}.",
        f"- Target positions per eval: "
        f"{aggregate['target_positions_per_eval']:.3f}.",
        f"- Batch outcomes: {aggregate['batch_attempts']} attempts, "
        f"{aggregate['batch_full']} full, "
        f"{aggregate['batch_partial']} partial, "
        f"{aggregate['batch_fallbacks']} fallbacks.",
        "",
        "| width | evals | positions/eval | target ms/eval | "
        "target ms/position | target-time share |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width, item in summary["verifier_widths"].items():
        lines.append(
            f"| {width} | {item['count']} | "
            f"{item['positions_per_eval']:.3f} | "
            f"{item['target_ms_per_eval']:.3f} | "
            f"{item['target_ms_per_position']:.3f} | "
            f"{item['target_time_share']:.1%} |"
        )
    lines.extend([
        "",
        "## Scheduler Widths",
        "",
        "| selected width | rounds | committed progress | progress/round | "
        "sidecar ms/round |",
        "|---:|---:|---:|---:|---:|",
    ])
    for width, item in summary["scheduler_widths"].items():
        lines.append(
            f"| {width} | {item['count']} | "
            f"{item['scheduler_width_committed']} | "
            f"{item['progress_per_round']:.3f} | "
            f"{item['sidecar_ms_per_round']:.3f} |"
        )
    lines.extend([
        "",
        "## Tasks",
        "",
        "| task | frozen ratio | target ms/emitted | sidecar ms/emitted | "
        "residual ms/emitted | target scale for parity |",
        "|:---|---:|---:|---:|---:|---:|",
    ])
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['paired_ratio']:.4f}x | "
            f"{item['target_ms_per_emitted']:.3f} | "
            f"{item['sidecar_ms_per_emitted']:.3f} | "
            f"{item['residual_ms_per_emitted']:.3f} | "
            f"{format_scale(item['target_scale_for_parity'])} |"
        )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        "- The frozen baseline and runtime times come from the uninstrumented paired confirmation; component timings come from fresh stats runs.",
        "- Residual is the cross-run difference between frozen runtime time and fresh target plus sidecar accounting; it includes host work and measurement mismatch.",
        "- The calibrated target scale assumes every required end-to-end saving comes from target verification while sidecar and residual costs stay fixed.",
        "- Width timings are synchronized diagnostic values, not throughput measurements.",
        "- No fresh baseline, timed throughput pass, acceptance audit, oracle trace, layer profiler, or fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def write_stats(path, rows):
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
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, TASK_COUNT, provenance["selection_policy"]
    )
    reference = load_throughput_reference(args, records, selection)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-threshold075-cost-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for record in records:
        command = common.command_text(
            args, prompts[record["label"]], "runtime",
            stats=True, confidence_threshold=THRESHOLD,
        )
        print(f"{record['label']} threshold-0.75 stats runtime: {command}")
    print(
        f"Threshold-0.75 cost audit: {TASK_COUNT} stats-only exact-runtime "
        "processes; no fresh baseline or throughput pair."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts(prompts, records)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": common.git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "experiment": "dspark_humaneval_threshold075_exact_verifier_cost",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("DS4_")
        },
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "selection": selection,
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "confidence_threshold": THRESHOLD,
            "runtime_stats": True,
            "oracle_trace": False,
            "timed_throughput": False,
            "fast_verifier": False,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "provenance_source_commit": provenance.get("source_commit"),
        "throughput_reference": {
            "summary": common.file_metadata(reference["summary_path"]),
            "metadata": common.file_metadata(reference["metadata_path"]),
            "csv": common.file_metadata(reference["csv_path"]),
        },
        "commands": {
            record["label"]: common.command_text(
                args, prompts[record["label"]], "runtime",
                stats=True, confidence_threshold=THRESHOLD,
            )
            for record in records
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for record in records:
        task = record["label"]
        row, _ = common.execute(
            args, root, run_dir, "stats", task, prompts[task],
            "runtime", reference["tasks"][task]["output_data"],
            stats=True, confidence_threshold=THRESHOLD,
        )
        rows.append(row)
        common.cooldown(args.cooldown)

    summary = summarize(rows, reference)
    report = render_report(summary)
    write_stats(run_dir / "stats.csv", rows)
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
