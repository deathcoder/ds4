#!/usr/bin/env python3
"""Estimate DSpark's break-even ceiling from round-level exact-runtime costs."""

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform

import run_dspark_humaneval_acceptance as corpus
import run_dspark_issue468_comparison as common


TASK_COUNT = 32


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect stats-only HumanEval round costs and estimate perfect "
            "routing and component-cost break-even ceilings."
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
        parser.error("refusing to run oracle audit without --confirm-ready")

    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = True
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD
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
        raise SystemExit(f"throughput reference {key} path mismatch")


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
            raise SystemExit(f"missing throughput reference {label}: {path}")

    summary = load_json(summary_path, "throughput summary")
    metadata = load_json(metadata_path, "throughput metadata")
    if metadata.get("experiment") != (
        "deepspec_humaneval_confidence_scheduler_throughput"
    ):
        raise SystemExit("throughput reference has the wrong experiment kind")
    if summary.get("sample_count") != TASK_COUNT:
        raise SystemExit("throughput reference is not the frozen 32-task study")
    if summary.get("selection") != selection:
        raise SystemExit("throughput reference selection mismatch")
    if summary.get("confidence_threshold") != (
        common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD
    ):
        raise SystemExit("throughput reference threshold mismatch")

    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "fast_verifier": False,
        "execution_mode": "throughput",
        "throughput_instrumentation": False,
        "runtime_instrumentation": False,
        "acceptance_audit": False,
        "confidence_threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        "nothink": True,
    }
    config = metadata.get("config", {})
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
        raise SystemExit(f"cannot read throughput reference CSV: {exc}") from exc
    rows_by_task = {}
    for row in csv_rows:
        rows_by_task.setdefault(row["prompt"], {})[row["mode"]] = row

    tasks = {}
    for record in records:
        task = record["label"]
        prior = summary.get("samples", {}).get(task)
        by_mode = rows_by_task.get(task, {})
        if prior is None or set(by_mode) != {"baseline", "runtime"}:
            raise SystemExit(f"throughput reference has incomplete task {task}")
        if by_mode["baseline"]["stdout_sha256"] != (
            by_mode["runtime"]["stdout_sha256"]
        ):
            raise SystemExit(f"throughput reference output mismatch for {task}")
        output_path = run_dir / by_mode["runtime"]["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing throughput runtime output for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != by_mode["runtime"]["stdout_sha256"]:
            raise SystemExit(f"throughput output hash mismatch for {task}")
        prompt_data = record["turns"][0].encode("utf-8")
        prompt_path = Path(metadata.get("prompts", {}).get(task, {}).get("path", ""))
        if not prompt_path.is_file() or prompt_path.read_bytes() != prompt_data:
            raise SystemExit(f"throughput prompt drift for {task}")
        baseline_tps = float(by_mode["baseline"]["generation_tps"])
        if not math.isclose(
            baseline_tps, prior["baseline_generation_tps"],
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise SystemExit(f"throughput baseline mismatch for {task}")
        tasks[task] = {
            "record": record,
            "prior": prior,
            "output_data": output_data,
            "prompt_data": prompt_data,
            "baseline_tps": baseline_tps,
        }
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "tasks": tasks,
    }


def _target_scale_for_ratio(baseline_ms, sidecar_ms, target_ms, ratio):
    budget = baseline_ms / ratio - sidecar_ms
    if target_ms <= 0:
        return None
    return budget / target_ms


def analyze_task(row, context):
    trace = row["oracle_trace"]
    emitted = row["emitted"]
    if sum(item["committed"] for item in trace) != emitted:
        raise RuntimeError("oracle committed progress does not match runtime stats")
    if sum(item["target_evals"] for item in trace) != row["target_evals"]:
        raise RuntimeError("oracle target evals do not match runtime stats")
    if sum(item["target_positions"] for item in trace) != row["target_eval_tokens"]:
        raise RuntimeError("oracle target positions do not match runtime stats")
    trace_target_ms = sum(item["target_ms"] for item in trace)
    if abs(trace_target_ms - row["target_eval_ms"]) > 0.02:
        raise RuntimeError("oracle target time does not match runtime stats")
    scheduled_sidecar_ms = sum(item["sidecar_ms"] for item in trace)
    expected_scheduled_ms = sum(row["scheduler_width_sidecar_ms"])
    if abs(scheduled_sidecar_ms - expected_scheduled_ms) > 0.02:
        raise RuntimeError("oracle sidecar time does not match runtime stats")

    prefetched_sidecar_ms = trace[0]["sidecar_ms"]
    if abs(prefetched_sidecar_ms - row["prefill_sidecar_ms"]) > 0.02:
        raise RuntimeError("first oracle round does not match prefetched sidecar time")
    generation_scheduled_ms = scheduled_sidecar_ms - prefetched_sidecar_ms
    outside_ms = row["generation_sidecar_ms"] - generation_scheduled_ms
    if outside_ms < -0.02:
        raise RuntimeError("scheduled sidecar exceeds generation sidecar time")
    outside_ms = max(0.0, outside_ms)
    baseline_ms_per_token = 1000.0 / context["baseline_tps"]
    rounds = []
    for index, item in enumerate(trace):
        outside_share = outside_ms * item["committed"] / emitted
        baseline_ms = item["committed"] * baseline_ms_per_token
        scheduled_generation_ms = 0.0 if index == 0 else item["sidecar_ms"]
        sidecar_ms = scheduled_generation_ms + outside_share
        target_ms = item["target_ms"]
        current_ms = sidecar_ms + target_ms
        rounds.append({
            **item,
            "trace_sidecar_ms": item["sidecar_ms"],
            "sidecar_ms": scheduled_generation_ms,
            "outside_sidecar_ms": outside_share,
            "baseline_ms": baseline_ms,
            "accounted_dspark_ms": current_ms,
            "current_profitable": current_ms < baseline_ms,
            "route_oracle_ms": min(current_ms, baseline_ms),
            "zero_sidecar_oracle_ms": min(target_ms, baseline_ms),
            "zero_target_oracle_ms": min(sidecar_ms, baseline_ms),
        })

    baseline_ms = sum(item["baseline_ms"] for item in rounds)
    sidecar_ms = sum(
        item["sidecar_ms"] + item["outside_sidecar_ms"] for item in rounds
    )
    target_ms = sum(item["target_ms"] for item in rounds)
    accounted_ms = sidecar_ms + target_ms
    route_oracle_ms = sum(item["route_oracle_ms"] for item in rounds)
    zero_sidecar_oracle_ms = sum(
        item["zero_sidecar_oracle_ms"] for item in rounds
    )
    zero_target_oracle_ms = sum(
        item["zero_target_oracle_ms"] for item in rounds
    )
    profitable = [item for item in rounds if item["current_profitable"]]
    profitable_tokens = sum(item["committed"] for item in profitable)
    return {
        "source_index": context["record"]["source_index"],
        "acceptance_verify_rate": context["prior"]["acceptance_verify_rate"],
        "prior_paired_ratio": context["prior"]["paired_ratio"],
        "baseline_tps": context["baseline_tps"],
        "emitted": emitted,
        "round_count": len(rounds),
        "baseline_ms": baseline_ms,
        "sidecar_ms": sidecar_ms,
        "prefetched_sidecar_ms": prefetched_sidecar_ms,
        "terminal_sidecar_ms": outside_ms,
        "target_ms": target_ms,
        "accounted_dspark_ms": accounted_ms,
        "accounted_ratio": baseline_ms / accounted_ms,
        "route_oracle_ms": route_oracle_ms,
        "route_oracle_ratio": baseline_ms / route_oracle_ms,
        "zero_sidecar_oracle_ratio": baseline_ms / zero_sidecar_oracle_ms,
        "zero_target_oracle_ratio": baseline_ms / zero_target_oracle_ms,
        "profitable_rounds": len(profitable),
        "profitable_round_share": len(profitable) / len(rounds),
        "profitable_tokens": profitable_tokens,
        "profitable_token_share": profitable_tokens / emitted,
        "target_scale_for_parity": _target_scale_for_ratio(
            baseline_ms, sidecar_ms, target_ms, 1.0
        ),
        "target_scale_for_10pct": _target_scale_for_ratio(
            baseline_ms, sidecar_ms, target_ms, 1.1
        ),
        "rounds": rounds,
    }


def summarize(rows, reference):
    tasks = {
        row["prompt"]: analyze_task(row, reference["tasks"][row["prompt"]])
        for row in rows
    }
    baseline_ms = sum(item["baseline_ms"] for item in tasks.values())
    sidecar_ms = sum(item["sidecar_ms"] for item in tasks.values())
    target_ms = sum(item["target_ms"] for item in tasks.values())
    accounted_ms = sidecar_ms + target_ms
    route_oracle_ms = sum(item["route_oracle_ms"] for item in tasks.values())
    zero_sidecar_ms = sum(
        item["baseline_ms"] / item["zero_sidecar_oracle_ratio"]
        for item in tasks.values()
    )
    zero_target_ms = sum(
        item["baseline_ms"] / item["zero_target_oracle_ratio"]
        for item in tasks.values()
    )
    rounds = [round_item for item in tasks.values() for round_item in item["rounds"]]
    profitable = [item for item in rounds if item["current_profitable"]]
    emitted = sum(item["emitted"] for item in tasks.values())
    return {
        "analysis": "dspark_humaneval_round_break_even_oracle",
        "threshold": common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
        "task_count": len(tasks),
        "tasks": tasks,
        "aggregate": {
            "emitted": emitted,
            "rounds": len(rounds),
            "baseline_ms": baseline_ms,
            "sidecar_ms": sidecar_ms,
            "target_ms": target_ms,
            "accounted_dspark_ms": accounted_ms,
            "accounted_ratio": baseline_ms / accounted_ms,
            "route_oracle_ratio": baseline_ms / route_oracle_ms,
            "zero_sidecar_oracle_ratio": baseline_ms / zero_sidecar_ms,
            "zero_target_oracle_ratio": baseline_ms / zero_target_ms,
            "profitable_rounds": len(profitable),
            "profitable_round_share": len(profitable) / len(rounds),
            "profitable_tokens": sum(item["committed"] for item in profitable),
            "profitable_token_share":
                sum(item["committed"] for item in profitable) / emitted,
            "target_scale_for_parity": _target_scale_for_ratio(
                baseline_ms, sidecar_ms, target_ms, 1.0
            ),
            "target_scale_for_10pct": _target_scale_for_ratio(
                baseline_ms, sidecar_ms, target_ms, 1.1
            ),
        },
    }


def format_scale(value):
    if value is None:
        return "n/a"
    if value < 0:
        return "impossible even with free target verification"
    if value <= 1.0:
        return f"{value:.3f}x current target time ({1.0 - value:.1%} reduction)"
    return f"{value:.3f}x current target time ({value - 1.0:.1%} headroom)"


def render_report(summary):
    aggregate = summary["aggregate"]
    lines = [
        "# DSpark HumanEval Break-Even Oracle",
        "",
        "Stats-only diagnostic; no fresh throughput benchmark was run.",
        "Each traced runtime output matched its frozen uninstrumented HumanEval artifact byte-for-byte.",
        "",
        "## Aggregate Ceiling",
        "",
        "| model | baseline-equivalent ratio |",
        "|:---|---:|",
        f"| Accounted current DSpark cost | {aggregate['accounted_ratio']:.4f}x |",
        f"| Perfect round router at current costs | {aggregate['route_oracle_ratio']:.4f}x |",
        f"| Perfect router with free sidecar | {aggregate['zero_sidecar_oracle_ratio']:.4f}x |",
        f"| Perfect router with free target verification | {aggregate['zero_target_oracle_ratio']:.4f}x |",
        "",
        f"- Profitable rounds at current accounted cost: "
        f"{aggregate['profitable_rounds']}/{aggregate['rounds']} "
        f"({aggregate['profitable_round_share']:.1%}).",
        f"- Tokens emitted by those rounds: {aggregate['profitable_tokens']}/"
        f"{aggregate['emitted']} ({aggregate['profitable_token_share']:.1%}).",
        f"- Target-time scale required for all-DSpark parity under the same "
        f"schedule: {format_scale(aggregate['target_scale_for_parity'])}.",
        f"- Target-time scale required for an accounted 1.10x all-DSpark result: "
        f"{format_scale(aggregate['target_scale_for_10pct'])}.",
        "",
        "## Tasks",
        "",
        "| task | acceptance | prior measured | accounted | route oracle | "
        "profitable rounds | profitable tokens |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['acceptance_verify_rate']:.3f} | "
            f"{item['prior_paired_ratio']:.4f}x | "
            f"{item['accounted_ratio']:.4f}x | "
            f"{item['route_oracle_ratio']:.4f}x | "
            f"{item['profitable_round_share']:.1%} | "
            f"{item['profitable_token_share']:.1%} |"
        )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        "- The router has future knowledge and is therefore an optimistic ceiling, not an implementable policy.",
        "- Routing a round to baseline would change later proposal boundaries, so this is a local counterfactual.",
        "- Accounted cost includes measured target and sidecar time, but not every host/runtime overhead.",
        "- The first traced proposal block is charged to prefill, while terminal unused sidecar work remains a generation cost.",
        "- Baseline cost uses each task's frozen uninstrumented generation t/s; trace timings come from fresh instrumented runs.",
        "- Free-sidecar and free-verifier rows are diagnostic upper bounds, not optimization predictions.",
        "- No timed benchmark, acceptance audit, layer profiler, or fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def write_rounds(path, summary):
    fields = (
        "task", "round", "proposed", "selected", "verified", "accepted",
        "committed", "trace_sidecar_ms", "sidecar_ms",
        "outside_sidecar_ms", "target_ms",
        "target_evals", "target_positions", "baseline_ms",
        "accounted_dspark_ms", "current_profitable", "route_oracle_ms",
        "zero_sidecar_oracle_ms", "zero_target_oracle_ms", "confidences",
    )
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for task, task_item in summary["tasks"].items():
            for item in task_item["rounds"]:
                row = {field: item.get(field) for field in fields}
                row["task"] = task
                row["confidences"] = ",".join(
                    f"{value:.9g}" for value in item["confidences"]
                )
                writer.writerow(row)


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
        f"speed-bench/local-runs/humaneval-oracle-audit-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for record in records:
        command = common.command_text(
            args, prompts[record["label"]], "runtime",
            stats=True, oracle_trace=True,
        )
        print(
            f"{record['label']} oracle stats runtime: "
            f"{command}"
        )
    print(
        f"HumanEval oracle audit: {TASK_COUNT} stats-only exact-runtime "
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
        "experiment": "dspark_humaneval_round_break_even_oracle",
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
            "confidence_threshold":
                common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
            "runtime_stats": True,
            "oracle_trace": True,
            "timed_throughput": False,
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
                stats=True, oracle_trace=True,
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
            args, root, run_dir, "oracle", task, prompts[task],
            "runtime", reference["tasks"][task]["output_data"],
            stats=True,
            confidence_threshold=common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD,
            oracle_trace=True,
        )
        rows.append(row)
        common.cooldown(args.cooldown)

    summary = summarize(rows, reference)
    report = render_report(summary)
    write_stats(run_dir / "stats.csv", rows)
    write_rounds(run_dir / "rounds.csv", summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw rounds: {run_dir / 'rounds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
