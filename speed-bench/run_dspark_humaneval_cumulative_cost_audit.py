#!/usr/bin/env python3
"""Refresh exact-verifier costs against cumulative HumanEval throughput."""

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform

import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_cumulative_throughput as cumulative
import run_dspark_humaneval_threshold075_cost_audit as cost
import run_dspark_issue468_comparison as common


THRESHOLD = cumulative.THRESHOLD
TASK_COUNT = cumulative.SAMPLE_COUNT
CUMULATIVE_SOURCE_COMMIT = "8ee89c2ccb8e3d4269fa3f01f1109b1e1878c37d"


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect stats-only exact-runtime costs against the frozen "
            "post-promotion cumulative HumanEval result."
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
        raise SystemExit(f"cumulative reference {key} path mismatch")


def validate_reference_identity(args, summary, metadata, selection):
    if metadata.get("experiment") != "dspark_humaneval_cumulative_throughput":
        raise SystemExit("throughput reference has the wrong experiment kind")
    if metadata.get("git_commit") != CUMULATIVE_SOURCE_COMMIT:
        raise SystemExit("cumulative reference source commit mismatch")
    if metadata.get("git_status_tracked"):
        raise SystemExit("cumulative reference was produced from a dirty tree")
    if summary.get("sample_count") != TASK_COUNT:
        raise SystemExit("cumulative reference is not the frozen 32-task run")
    if summary.get("threshold") != THRESHOLD:
        raise SystemExit("cumulative reference threshold mismatch")
    if summary.get("selection") != selection:
        raise SystemExit("cumulative reference selection mismatch")

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
        "global_warmup_pairs": 2,
        "promoted_defaults": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"cumulative config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        validate_metadata_path(metadata, key, expected)


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
            raise SystemExit(f"missing cumulative reference {label}: {path}")

    summary = load_json(summary_path, "cumulative summary")
    metadata = load_json(metadata_path, "cumulative metadata")
    validate_reference_identity(args, summary, metadata, selection)

    try:
        csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read cumulative CSV: {exc}") from exc
    if len(csv_rows) != TASK_COUNT * 2:
        raise SystemExit("cumulative reference measured-row count mismatch")
    rows_by_task = {}
    for row in csv_rows:
        task_rows = rows_by_task.setdefault(row.get("prompt"), {})
        mode = row.get("mode")
        if mode in task_rows:
            raise SystemExit(f"duplicate cumulative row for {row.get('prompt')}")
        task_rows[mode] = row

    tasks = {}
    ratios = []
    for position, record in enumerate(records, start=1):
        task = record["label"]
        sample = summary.get("samples", {}).get(task)
        by_mode = rows_by_task.get(task, {})
        if sample is None or set(by_mode) != {"baseline", "runtime"}:
            raise SystemExit(f"cumulative reference has incomplete task {task}")
        baseline_row = by_mode["baseline"]
        runtime_row = by_mode["runtime"]
        expected_order = "-".join(cumulative.throughput.measured_order(position))
        if any(row.get("pair_order") != expected_order for row in by_mode.values()):
            raise SystemExit(f"cumulative pair order mismatch for {task}")
        if baseline_row["stdout_sha256"] != runtime_row["stdout_sha256"]:
            raise SystemExit(f"cumulative output mismatch for {task}")
        output_data = None
        for row in (baseline_row, runtime_row):
            output_path = run_dir / row["stdout_file"]
            if not output_path.is_file():
                raise SystemExit(f"missing cumulative output for {task}")
            candidate = output_path.read_bytes()
            if common.sha256(candidate) != row["stdout_sha256"]:
                raise SystemExit(f"cumulative output hash mismatch for {task}")
            if output_data is not None and candidate != output_data:
                raise SystemExit(f"cumulative output bytes differ for {task}")
            output_data = candidate
        prompt_data = record["turns"][0].encode("utf-8")
        prompt_path = run_dir / "prompts" / f"{task}.txt"
        if not prompt_path.is_file() or prompt_path.read_bytes() != prompt_data:
            raise SystemExit(f"cumulative prompt drift for {task}")

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
                sample.get(key, -1), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise SystemExit(f"cumulative reference {key} mismatch for {task}")
        ratios.append(paired_ratio)
        tasks[task] = {
            "record": record,
            "sample": sample,
            "output_data": output_data,
            "prompt_data": prompt_data,
            "baseline_tps": baseline_tps,
            "runtime_tps": runtime_tps,
        }
    if set(rows_by_task) != set(tasks):
        raise SystemExit("cumulative reference task set mismatch")
    geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    if not math.isclose(
        summary.get("paired_ratio_geometric_mean", -1),
        geometric_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit("cumulative aggregate ratio mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "summary": summary,
        "metadata": metadata,
        "tasks": tasks,
    }


def summarize(rows, reference):
    summary = cost.summarize(rows, reference)
    summary["analysis"] = "dspark_humaneval_cumulative_exact_verifier_cost"
    return summary


def render_report(summary):
    report = cost.render_report(summary)
    replacements = (
        (
            "# DSpark Threshold 0.75 Exact Verifier Cost Audit",
            "# DSpark Post-Promotion Exact Verifier Cost Audit",
        ),
        (
            "frozen uninstrumented threshold-0.75 artifact",
            "frozen uninstrumented cumulative artifact",
        ),
        ("| baseline | threshold 0.75 |", "| baseline | current DSpark |"),
        ("Frozen threshold-0.75 geometric mean", "Frozen cumulative geometric mean"),
        (
            "the uninstrumented paired confirmation",
            "the uninstrumented cumulative reassessment",
        ),
    )
    for old, new in replacements:
        report = report.replace(old, new)
    return report


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
    default_dir = root / f"speed-bench/local-runs/humaneval-cumulative-cost-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for record in records:
        command = common.command_text(
            args, prompts[record["label"]], "runtime",
            stats=True, confidence_threshold=THRESHOLD,
        )
        print(f"{record['label']} cumulative stats runtime: {command}")
    print(
        f"Post-promotion cost audit: {TASK_COUNT} stats-only exact-runtime "
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
        "experiment": "dspark_humaneval_cumulative_exact_verifier_cost",
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
            "promoted_defaults": True,
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
    cost.write_stats(run_dir / "stats.csv", rows)
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
