#!/usr/bin/env python3
"""Benchmark exact DSpark throughput on the pinned HumanEval corpus."""

import argparse
import datetime as dt
import json
from pathlib import Path
import statistics

import run_dspark_humaneval_acceptance as corpus
import run_dspark_issue468_comparison as common


SCHEDULER_THRESHOLD = common.DSPARK_DEFAULT_CONFIDENCE_THRESHOLD
SCHEDULER_RETENTION_FLOOR = "0.975"


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run paired uninstrumented HumanEval DSpark throughput."
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
    parser.add_argument("--acceptance-reference", type=Path)
    parser.add_argument(
        "--confidence-scheduler",
        action="store_true",
        help="run the frozen confidence-prefix gate at threshold 0.455",
    )
    parser.add_argument("--scheduler-reference", type=Path)
    parser.add_argument(
        "--historical-throughput-reference",
        type=Path,
        help=(
            "prior frozen scheduled-throughput summary for descriptive "
            "task-level movement"
        ),
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.sample_count < 2:
        parser.error("sample-count must be at least two")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to benchmark without --confirm-ready")
    if not args.dry_run and args.acceptance_reference is None:
        parser.error("--acceptance-reference is required for a measured run")
    if args.confidence_scheduler:
        if args.sample_count != 32:
            parser.error("the confidence-scheduler gate requires sample-count 32")
        if args.acceptance_reference is None:
            parser.error(
                "--acceptance-reference is required with --confidence-scheduler"
            )
        if args.scheduler_reference is None:
            parser.error(
                "--scheduler-reference is required with --confidence-scheduler"
            )
    elif args.scheduler_reference is not None:
        parser.error("--scheduler-reference requires --confidence-scheduler")
    if (
        args.historical_throughput_reference is not None and
        not args.confidence_scheduler
    ):
        parser.error(
            "--historical-throughput-reference requires --confidence-scheduler"
        )

    # Attributes consumed by the shared command, execution, and metadata paths.
    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = False
    args.stats_pass = False
    args.acceptance_audit = False
    args.confidence_threshold = (
        SCHEDULER_THRESHOLD if args.confidence_scheduler else
        common.DSPARK_FIXED_CONFIDENCE_THRESHOLD
    )
    args.pairs = 1
    args.warmups = 0
    return args, root


def _validate_path(metadata, key, expected):
    actual = metadata.get(key, {}).get("path")
    if actual is None or Path(actual).resolve() != expected.resolve():
        raise SystemExit(f"acceptance reference {key} path mismatch")


def load_acceptance_reference(args, selection, provenance):
    if args.acceptance_reference is None:
        return None
    summary_path = args.acceptance_reference.resolve()
    metadata_path = summary_path.parent / "metadata.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing acceptance reference: {summary_path}")
    if not metadata_path.is_file():
        raise SystemExit(f"missing acceptance reference metadata: {metadata_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid acceptance reference: {exc}") from exc

    if summary.get("dataset") != "HumanEval":
        raise SystemExit("acceptance reference is not a HumanEval study")
    if summary.get("sample_count") != selection["sample_count"]:
        raise SystemExit("acceptance reference sample-count mismatch")
    protocol = summary.get("protocol", {})
    if protocol.get("source_commit") != provenance.get("source_commit"):
        raise SystemExit("acceptance reference source-commit mismatch")
    if protocol.get("selection") != selection:
        raise SystemExit("acceptance reference selection mismatch")
    expected_protocol = {
        "non_thinking": True,
        "confidence_scheduler": False,
        "temperature": 0.0,
        "max_new_tokens": args.tokens,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise SystemExit(
                f"acceptance reference protocol {key} mismatch: "
                f"{protocol.get(key)!r} != {expected!r}"
            )

    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "execution_mode": "acceptance_audit",
        "nothink": True,
        "fast_verifier": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"acceptance reference config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    _validate_path(metadata, "binary", args.binary)
    _validate_path(metadata, "base_model", args.model)
    _validate_path(metadata, "dspark_model", args.dspark_model)

    aggregate = summary.get("aggregate", {})
    if aggregate.get("block_size") != 5 or aggregate.get("proposals", 0) <= 0:
        raise SystemExit("acceptance reference has invalid aggregate metrics")
    expected_labels = {
        f"humaneval_{index:03d}" for index in selection["indices_zero_based"]
    }
    samples = summary.get("samples", {})
    if set(samples) != expected_labels:
        raise SystemExit("acceptance reference sample labels mismatch")
    if any(samples[label].get("paper_verify_rate") is None for label in samples):
        raise SystemExit("acceptance reference is missing per-sample verify rates")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "summary": summary,
        "metadata": metadata,
    }


def validate_scheduler_summary(summary):
    if summary.get("analysis") != "deepspec_confidence_prefix_local_counterfactual":
        raise ValueError("scheduler reference has the wrong analysis kind")
    if summary.get("samples") != 32 or summary.get("block_size") != 5:
        raise ValueError(
            "scheduler reference does not describe the frozen 32-task K=5 study"
        )
    in_sample = summary.get("in_sample_policies", {}).get(
        SCHEDULER_RETENTION_FLOOR, {}
    )
    leave_one_out = summary.get("leave_one_task_out", {}).get(
        SCHEDULER_RETENTION_FLOOR, {}
    )
    expected = {
        "in-sample threshold": (in_sample.get("threshold"), 0.455891937),
        "held-out threshold median": (
            leave_one_out.get("threshold_median"), 0.455
        ),
        "held-out threshold minimum": (
            leave_one_out.get("threshold_minimum"), 0.455
        ),
        "held-out threshold maximum": (
            leave_one_out.get("threshold_maximum"), 0.460
        ),
    }
    for label, (actual, wanted) in expected.items():
        if not isinstance(actual, (int, float)) or abs(actual - wanted) > 1e-9:
            raise ValueError(f"scheduler reference {label} mismatch")
    if leave_one_out.get("retention_floor") != 0.975:
        raise ValueError("scheduler reference retention floor mismatch")
    return {
        "threshold": SCHEDULER_THRESHOLD,
        "retention_floor": 0.975,
        "in_sample": in_sample,
        "leave_one_task_out": leave_one_out,
    }


def load_scheduler_reference(args, acceptance_reference, selection):
    if not args.confidence_scheduler:
        return None
    summary_path = args.scheduler_reference.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"missing scheduler reference: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        policy = validate_scheduler_summary(summary)
        source_run = Path(summary["source_run"]).resolve()
        trace_summary_path = source_run / "summary.json"
        trace_metadata_path = source_run / "metadata.json"
        trace_summary = json.loads(
            trace_summary_path.read_text(encoding="utf-8")
        )
        trace_metadata = json.loads(
            trace_metadata_path.read_text(encoding="utf-8")
        )
    except (KeyError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid scheduler reference: {exc}") from exc

    expected_acceptance = acceptance_reference["summary_path"].resolve()
    trace_reference = trace_summary.get("trace_reference")
    if (
        trace_reference is None or
        Path(trace_reference).resolve() != expected_acceptance
    ):
        raise SystemExit("scheduler trace does not derive from the acceptance reference")
    protocol = trace_summary.get("protocol", {})
    if protocol.get("selection") != selection:
        raise SystemExit("scheduler trace selection mismatch")
    if protocol.get("confidence_scheduler") is not False:
        raise SystemExit("scheduler trace must come from fixed K=5 proposals")
    if trace_metadata.get("experiment") != (
        "deepspec_humaneval_confidence_scheduler_trace"
    ):
        raise SystemExit("scheduler trace metadata experiment mismatch")
    config = trace_metadata.get("config", {})
    if not config.get("acceptance_trace") or not config.get("acceptance_audit"):
        raise SystemExit("scheduler trace metadata lacks acceptance tracing")
    return {
        "summary_path": summary_path,
        "summary": summary,
        "trace_summary_path": trace_summary_path,
        "trace_metadata_path": trace_metadata_path,
        "policy": policy,
    }


def validate_historical_throughput_summary(summary, selection):
    if summary.get("sample_count") != 32:
        raise ValueError("historical throughput reference is not a 32-task study")
    if summary.get("selection") != selection:
        raise ValueError("historical throughput selection mismatch")
    if summary.get("confidence_scheduler") is not True:
        raise ValueError("historical throughput did not use confidence scheduling")
    if summary.get("confidence_threshold") != SCHEDULER_THRESHOLD:
        raise ValueError("historical throughput confidence threshold mismatch")

    samples = summary.get("samples", {})
    expected = {
        f"humaneval_{index:03d}": index
        for index in selection["indices_zero_based"]
    }
    if set(samples) != set(expected):
        raise ValueError("historical throughput sample labels mismatch")
    for label, source_index in expected.items():
        item = samples[label]
        if item.get("source_index") != source_index:
            raise ValueError(
                f"historical throughput source index mismatch for {label}"
            )
        for key in ("runtime_generation_tps", "paired_ratio"):
            value = item.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"historical throughput {key} is invalid for {label}"
                )


def load_historical_throughput_reference(
    args, selection, acceptance_reference, scheduler_reference
):
    if args.historical_throughput_reference is None:
        return None
    summary_path = args.historical_throughput_reference.resolve()
    metadata_path = summary_path.parent / "metadata.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"missing historical throughput reference: {summary_path}"
        )
    if not metadata_path.is_file():
        raise SystemExit(
            f"missing historical throughput metadata: {metadata_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_historical_throughput_summary(summary, selection)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid historical throughput reference: {exc}") from exc

    if metadata.get("experiment") != (
        "deepspec_humaneval_confidence_scheduler_throughput"
    ):
        raise SystemExit("historical throughput metadata experiment mismatch")
    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "execution_mode": "throughput",
        "throughput_instrumentation": False,
        "runtime_instrumentation": False,
        "fast_verifier": False,
        "confidence_threshold": SCHEDULER_THRESHOLD,
        "nothink": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"historical throughput config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    _validate_path(metadata, "binary", args.binary)
    _validate_path(metadata, "base_model", args.model)
    _validate_path(metadata, "dspark_model", args.dspark_model)

    historical_acceptance = summary.get("acceptance_reference", {}).get(
        "summary_path"
    )
    if (
        historical_acceptance is None or
        Path(historical_acceptance).resolve() !=
        acceptance_reference["summary_path"].resolve()
    ):
        raise SystemExit("historical throughput acceptance reference mismatch")
    historical_scheduler = summary.get("scheduler_reference", {}).get(
        "summary_path"
    )
    if (
        historical_scheduler is None or
        Path(historical_scheduler).resolve() !=
        scheduler_reference["summary_path"].resolve()
    ):
        raise SystemExit("historical throughput scheduler reference mismatch")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "summary": summary,
        "metadata": metadata,
    }


def warmup_schedule(records):
    if len(records) == 1:
        return ((records[0], ("baseline", "runtime")),)
    return (
        (records[0], ("baseline", "runtime")),
        (records[-1], ("runtime", "baseline")),
    )


def measured_order(position):
    if position % 2 == 0:
        return ("runtime", "baseline")
    return ("baseline", "runtime")


def pearson_correlation(xs, ys):
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = (
        sum(value * value for value in centered_x) *
        sum(value * value for value in centered_y)
    ) ** 0.5
    if denominator == 0:
        return None
    return sum(
        left * right for left, right in zip(centered_x, centered_y)
    ) / denominator


def summarize(
    rows, records, acceptance_reference, scheduler_reference=None,
    historical_reference=None,
):
    samples = {}
    ratios = []
    acceptance_rates = []
    baseline_values = []
    runtime_values = []
    acceptance_samples = acceptance_reference["summary"]["samples"]
    for record in records:
        label = record["label"]
        selected = {row["mode"]: row for row in rows if row["prompt"] == label}
        baseline = selected["baseline"]["generation_tps"]
        runtime = selected["runtime"]["generation_tps"]
        ratio = runtime / baseline
        ratios.append(ratio)
        acceptance_rate = acceptance_samples[label]["paper_verify_rate"]
        acceptance_rates.append(acceptance_rate)
        baseline_values.append(baseline)
        runtime_values.append(runtime)
        samples[label] = {
            "source_index": record["source_index"],
            "order": selected["baseline"]["pair_order"],
            "baseline_generation_tps": baseline,
            "runtime_generation_tps": runtime,
            "paired_ratio": ratio,
            "delta_percent": (ratio - 1.0) * 100.0,
            "acceptance_verify_rate": acceptance_rate,
        }
    quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
    median_ratio = statistics.median(ratios)
    result = {
        "sample_count": len(records),
        "samples": samples,
        "baseline_generation_tps_median": statistics.median(baseline_values),
        "runtime_generation_tps_median": statistics.median(runtime_values),
        "ratio_of_medians": (
            statistics.median(runtime_values) /
            statistics.median(baseline_values)
        ),
        "paired_ratio_median": median_ratio,
        "paired_ratio_geometric_mean": statistics.geometric_mean(ratios),
        "paired_ratio_arithmetic_mean": statistics.mean(ratios),
        "paired_ratio_q1": quartiles[0],
        "paired_ratio_q3": quartiles[2],
        "paired_ratio_minimum": min(ratios),
        "paired_ratio_maximum": max(ratios),
        "median_delta_percent": (median_ratio - 1.0) * 100.0,
        "runtime_faster_tasks": sum(ratio > 1.0 for ratio in ratios),
        "runtime_equal_tasks": sum(ratio == 1.0 for ratio in ratios),
        "runtime_slower_tasks": sum(ratio < 1.0 for ratio in ratios),
        "acceptance_speed_pearson": pearson_correlation(
            acceptance_rates, ratios
        ),
        "paired_ratio_values": ratios,
    }
    result["confidence_scheduler"] = scheduler_reference is not None
    result["confidence_threshold"] = (
        SCHEDULER_THRESHOLD if scheduler_reference else None
    )
    if scheduler_reference:
        result["scheduler_reference"] = {
            "summary_path": str(scheduler_reference["summary_path"]),
            "retention_floor": scheduler_reference["policy"]["retention_floor"],
            "in_sample_progress_retention": (
                scheduler_reference["policy"]["in_sample"]
                ["progress_retention"]
            ),
            "held_out_progress_retention": (
                scheduler_reference["policy"]["leave_one_task_out"]
                ["progress_retention"]
            ),
        }
    if historical_reference:
        historical = historical_reference["summary"]
        runtime_movement = []
        paired_ratio_movement = []
        improved = 0
        equal = 0
        for label, item in samples.items():
            prior = historical["samples"][label]
            runtime_ratio = (
                item["runtime_generation_tps"] /
                prior["runtime_generation_tps"]
            )
            paired_movement = item["paired_ratio"] / prior["paired_ratio"]
            runtime_movement.append(runtime_ratio)
            paired_ratio_movement.append(paired_movement)
            improved += paired_movement > 1.0
            equal += paired_movement == 1.0
            item["historical_runtime_generation_tps"] = (
                prior["runtime_generation_tps"]
            )
            item["historical_paired_ratio"] = prior["paired_ratio"]
            item["runtime_tps_vs_historical"] = runtime_ratio
            item["paired_ratio_vs_historical"] = paired_movement
        result["historical_throughput_reference"] = {
            "summary_path": str(historical_reference["summary_path"]),
            "historical_paired_ratio_median": historical["paired_ratio_median"],
            "historical_runtime_generation_tps_median": (
                historical["runtime_generation_tps_median"]
            ),
            "runtime_tps_movement_median": statistics.median(runtime_movement),
            "runtime_tps_movement_geometric_mean": statistics.geometric_mean(
                runtime_movement
            ),
            "paired_ratio_movement_median": statistics.median(
                paired_ratio_movement
            ),
            "paired_ratio_movement_geometric_mean": statistics.geometric_mean(
                paired_ratio_movement
            ),
            "paired_ratio_improved_tasks": improved,
            "paired_ratio_equal_tasks": equal,
            "paired_ratio_regressed_tasks": len(samples) - improved - equal,
        }
    return result


def render_report(
    summary, acceptance_reference, scheduler_reference=None,
    historical_reference=None,
):
    runtime_label = (
        f"DSpark threshold {SCHEDULER_THRESHOLD}"
        if scheduler_reference else "DSpark"
    )
    lines = [
        (
            "# DSpark HumanEval Scheduled Throughput"
            if scheduler_reference else "# DSpark HumanEval Paired Throughput"
        ),
        "",
        "All samples are uninstrumented and paired within the same HumanEval task.",
        "Every exact DSpark output matched its baseline byte-for-byte.",
        "Generation t/s excludes process startup; paired ratios are the primary metric.",
        "",
        f"| samples | baseline median | {runtime_label} median | "
        "ratio of medians | median paired ratio | geometric mean | median delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | "
        f"{summary['baseline_generation_tps_median']:.2f} t/s | "
        f"{summary['runtime_generation_tps_median']:.2f} t/s | "
        f"{summary['ratio_of_medians']:.4f}x | "
        f"{summary['paired_ratio_median']:.4f}x | "
        f"{summary['paired_ratio_geometric_mean']:.4f}x | "
        f"{summary['median_delta_percent']:+.1f}% |",
        "",
        f"- Paired-ratio interquartile range: "
        f"{summary['paired_ratio_q1']:.4f}x-"
        f"{summary['paired_ratio_q3']:.4f}x",
        f"- Paired-ratio range: {summary['paired_ratio_minimum']:.4f}x-"
        f"{summary['paired_ratio_maximum']:.4f}x",
        f"- Tasks faster/equal/slower with {runtime_label}: "
        f"{summary['runtime_faster_tasks']}/"
        f"{summary['runtime_equal_tasks']}/"
        f"{summary['runtime_slower_tasks']}",
        (
            "- Descriptive Pearson correlation, prior acceptance verify rate vs "
            f"paired speed ratio: {summary['acceptance_speed_pearson']:.3f}"
            if summary["acceptance_speed_pearson"] is not None else
            "- Acceptance/speed correlation is undefined because one input is constant."
        ),
        "",
        "## Tasks",
        "",
        f"| sample | source index | order | acceptance | baseline | "
        f"{runtime_label} | ratio | delta |",
        "|---|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for label, item in summary["samples"].items():
        lines.append(
            f"| {label} | {item['source_index']} | {item['order']} | "
            f"{item['acceptance_verify_rate']:.3f} | "
            f"{item['baseline_generation_tps']:.2f} t/s | "
            f"{item['runtime_generation_tps']:.2f} t/s | "
            f"{item['paired_ratio']:.4f}x | "
            f"{item['delta_percent']:+.1f}% |"
        )
    if acceptance_reference:
        aggregate = acceptance_reference["summary"]["aggregate"]
        lines.extend(["", "## Separate Acceptance Gate", ""])
        if scheduler_reference:
            lines.extend([
                f"The prior fixed-K {acceptance_reference['summary']['sample_count']}"
                f"-sample audit freezes task selection and supplies acceptance "
                f"context: {aggregate['proposals']} proposal rounds, accepted "
                f"length {aggregate['paper_acceptance_length']:.3f}, verify rate "
                f"{aggregate['paper_verify_rate']:.3f}, and full acceptance "
                f"{aggregate['full_accept_rate']:.1%}.",
                "Scheduling changes later proposal boundaries, so those fixed-K "
                "counts are not claimed as scheduled-runtime acceptance results.",
            ])
        else:
            lines.append(
                f"The validated uninstrumented workload matches the prior "
                f"{acceptance_reference['summary']['sample_count']}-sample audit: "
                f"{aggregate['proposals']} proposal rounds, accepted length "
                f"{aggregate['paper_acceptance_length']:.3f}, verify rate "
                f"{aggregate['paper_verify_rate']:.3f}, and full acceptance "
                f"{aggregate['full_accept_rate']:.1%}."
            )
        lines.append(
            "Acceptance instrumentation was not enabled during this throughput run."
        )
    if scheduler_reference:
        policy = scheduler_reference["policy"]
        lines.extend([
            "",
            "## Frozen Scheduler Gate",
            "",
            f"The runtime uses the predeclared threshold `{SCHEDULER_THRESHOLD}` "
            f"from the separate {policy['retention_floor']:.1%}-retention study. "
            f"Its offline in-sample progress retention was "
            f"{policy['in_sample']['progress_retention']:.3f}; this throughput "
            "run does not reuse that proxy as a speed result.",
            "The offline study was a local counterfactual, not an exact replay; "
            "this run measures the policy's actual changed proposal sequence.",
            "The complete five-row sidecar block is still computed; only the "
            "exact target-verification prefix is scheduled.",
        ])
    if historical_reference:
        historical = summary["historical_throughput_reference"]
        lines.extend([
            "",
            "## Historical Movement",
            "",
            "The prior scheduled run is descriptive context only; it was "
            "collected in a different process sequence and is not a paired "
            "ablation.",
            f"- Prior median paired DSpark/baseline ratio: "
            f"{historical['historical_paired_ratio_median']:.4f}x",
            f"- Current/prior median task-level paired-ratio movement: "
            f"{historical['paired_ratio_movement_median']:.4f}x",
            f"- Current/prior geometric task-level paired-ratio movement: "
            f"{historical['paired_ratio_movement_geometric_mean']:.4f}x",
            f"- Tasks with improved/equal/regressed paired ratio: "
            f"{historical['paired_ratio_improved_tasks']}/"
            f"{historical['paired_ratio_equal_tasks']}/"
            f"{historical['paired_ratio_regressed_tasks']}",
            f"- Current/prior median task-level DSpark t/s movement: "
            f"{historical['runtime_tps_movement_median']:.4f}x",
        ])
    lines.extend([
        "",
        "- The two global warmup tasks are excluded from every reported value.",
        "- Measured task order alternates baseline-first and DSpark-first.",
        "- No DSpark stats, acceptance audit, diagnostic route, or profiler is enabled.",
        "- Cross-task medians summarize the workload; absolute t/s is not averaged across unlike generations.",
    ])
    return "\n".join(lines) + "\n"


def run_pair(args, root, run_dir, label, record, prompt, order):
    rows = []
    reference = None
    order_text = "-".join(order)
    for position, mode in enumerate(order, start=1):
        row, output = common.execute(
            args, root, run_dir, label, record["label"], prompt,
            mode, reference,
            confidence_threshold=(
                args.confidence_threshold if mode == "runtime" else None
            ),
        )
        reference = output
        row.update(
            source_index=record["source_index"],
            pair_order=order_text,
            pair_position=position,
        )
        rows.append(row)
        common.cooldown(args.cooldown)
    return rows


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "corpus_dir"):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, args.sample_count, provenance["selection_policy"]
    )
    acceptance_reference = load_acceptance_reference(
        args, selection, provenance
    )
    scheduler_reference = load_scheduler_reference(
        args, acceptance_reference, selection
    )
    historical_reference = load_historical_throughput_reference(
        args, selection, acceptance_reference, scheduler_reference
    )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_prefix = (
        "humaneval-scheduler-throughput"
        if scheduler_reference else "humaneval-throughput"
    )
    default_dir = root / (
        f"speed-bench/local-runs/{run_prefix}-"
        f"{args.sample_count}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for position, record in enumerate(records, start=1):
        order = measured_order(position)
        prompt = prompts[record["label"]]
        baseline_command = common.command_text(args, prompt, "baseline")
        runtime_command = common.command_text(
            args, prompt, "runtime",
            confidence_threshold=args.confidence_threshold,
        )
        print(
            f"{record['label']} measured order: {' -> '.join(order)}\n"
            f"  baseline: {baseline_command}\n"
            f"  runtime:  {runtime_command}"
        )
    print(
        f"HumanEval throughput: {args.sample_count} uninstrumented paired tasks, "
        f"two global warmup pairs, {args.cooldown:g}s cooldown."
    )
    if acceptance_reference:
        print(
            "Acceptance reference: "
            f"{acceptance_reference['summary_path']}"
        )
    if scheduler_reference:
        print(
            f"Scheduler reference: {scheduler_reference['summary_path']}\n"
            f"Frozen confidence threshold: {SCHEDULER_THRESHOLD}"
        )
    if historical_reference:
        print(
            "Historical throughput reference: "
            f"{historical_reference['summary_path']}"
        )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts(prompts, records)
    metadata = common.collect_metadata(
        args, root, prompts, provenance, acceptance_reference=None
    )
    metadata["experiment"] = (
        "deepspec_humaneval_confidence_scheduler_throughput"
        if scheduler_reference else "deepspec_humaneval_paired_throughput"
    )
    metadata["experiment_selection"] = selection
    metadata["throughput_schedule"] = {
        "global_warmup_samples": [
            record["label"] for record, _ in warmup_schedule(records)
        ],
        "global_warmup_orders": [
            list(order) for _, order in warmup_schedule(records)
        ],
        "measured_pairs_per_sample": 1,
        "alternating_measured_order": True,
        "cooldown_seconds": args.cooldown,
    }
    metadata["acceptance_reference"] = {
        "summary": common.file_metadata(acceptance_reference["summary_path"]),
        "metadata": common.file_metadata(acceptance_reference["metadata_path"]),
    }
    if scheduler_reference:
        metadata["scheduler_reference"] = {
            "summary": common.file_metadata(scheduler_reference["summary_path"]),
            "trace_summary": common.file_metadata(
                scheduler_reference["trace_summary_path"]
            ),
            "trace_metadata": common.file_metadata(
                scheduler_reference["trace_metadata_path"]
            ),
            "threshold": SCHEDULER_THRESHOLD,
            "retention_floor": scheduler_reference["policy"]["retention_floor"],
        }
    if historical_reference:
        metadata["historical_throughput_reference"] = {
            "summary": common.file_metadata(historical_reference["summary_path"]),
            "metadata": common.file_metadata(
                historical_reference["metadata_path"]
            ),
            "comparison_semantics": "descriptive_cross_run_only",
        }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    warmup_rows = []
    for number, (record, order) in enumerate(warmup_schedule(records), start=1):
        warmup_rows.extend(run_pair(
            args, root, run_dir, f"warmup-{number:02d}", record,
            prompts[record["label"]], order,
        ))

    measured_rows = []
    sequence = 0
    for position, record in enumerate(records, start=1):
        pair_rows = run_pair(
            args, root, run_dir, f"measured-{position:02d}", record,
            prompts[record["label"]], measured_order(position),
        )
        for row in pair_rows:
            sequence += 1
            row["sequence"] = sequence
            measured_rows.append(row)

    fields = (
        "sequence", "prompt", "source_index", "pair_order", "pair_position",
        "mode", "prefill_tps", "generation_tps", "wall_seconds",
        "stdout_sha256", "stdout_file", "stderr_file",
    )
    common.write_csv(run_dir / "throughput.csv", measured_rows, fields)
    common.write_csv(
        run_dir / "warmups.csv", warmup_rows, fields[1:]
    )
    summary = summarize(
        measured_rows, records, acceptance_reference, scheduler_reference,
        historical_reference,
    )
    summary["selection"] = selection
    summary["acceptance_reference"] = {
        "summary_path": str(acceptance_reference["summary_path"]),
        "aggregate": acceptance_reference["summary"]["aggregate"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(
        summary, acceptance_reference, scheduler_reference,
        historical_reference,
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw throughput: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
