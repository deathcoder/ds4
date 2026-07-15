#!/usr/bin/env python3
"""Analyze DeepSpec confidence-prefix policies from DSpark proposal traces."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import run_dspark_issue468_comparison as common


RETENTION_FLOORS = (0.99, 0.975, 0.95)


def parse_trace(stderr_data, path, sample):
    records = []
    for line in stderr_data.splitlines():
        if not line.startswith(common.ACCEPTANCE_TRACE_PREFIX):
            continue
        values = {}
        try:
            payload = line[len(common.ACCEPTANCE_TRACE_PREFIX):].decode("ascii")
            for item in payload.split():
                key, value = item.split("=", 1)
                values[key] = value
            proposed = int(values["proposed"])
            accepted = int(values["accepted"])
            truncated = int(values["truncated"])
            confidences = (
                [] if values["confidences"] == "none" else
                [float(value) for value in values["confidences"].split(",")]
            )
            round_number = int(values["round"])
        except (KeyError, UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid acceptance trace in {path}: {exc}") from exc
        if round_number != len(records) + 1:
            raise RuntimeError(f"non-sequential acceptance trace in {path}")
        if proposed < 0 or accepted < 0 or accepted > proposed:
            raise RuntimeError(f"invalid proposal counts in {path}")
        if truncated not in (0, 1):
            raise RuntimeError(f"invalid truncated flag in {path}")
        if len(confidences) != proposed:
            raise RuntimeError(f"confidence count mismatch in {path}")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in confidences):
            raise RuntimeError(f"invalid confidence value in {path}")
        records.append({
            "sample": sample,
            "round": round_number,
            "proposed": proposed,
            "accepted": accepted,
            "truncated": bool(truncated),
            "confidences": confidences,
        })
    if not records:
        raise RuntimeError(f"no acceptance trace records found in {path}")
    return records


def selected_prefix(confidences, threshold):
    if threshold <= 0.0:
        return len(confidences)
    for position, confidence in enumerate(confidences):
        if confidence < threshold:
            return position
    return len(confidences)


def policy_metrics(records, threshold=None, oracle=False):
    if (threshold is None) == (not oracle):
        raise ValueError("choose exactly one of threshold or oracle")
    selected = []
    accepted = []
    for record in records:
        accepted_count = record["accepted"]
        selected_count = (
            accepted_count if oracle else
            selected_prefix(record["confidences"], threshold)
        )
        selected.append(selected_count)
        accepted.append(accepted_count)

    progress = sum(min(a, k) + 1 for a, k in zip(accepted, selected))
    fixed_progress = sum(value + 1 for value in accepted)
    target_positions = sum(max(1, value) for value in selected)
    fixed_target_positions = sum(record["proposed"] for record in records)
    rounds = len(records)
    evals_per_progress = rounds / progress
    fixed_evals_per_progress = rounds / fixed_progress
    positions_per_progress = target_positions / progress
    fixed_positions_per_progress = fixed_target_positions / fixed_progress
    return {
        "threshold": threshold,
        "rounds": rounds,
        "mean_selected_drafts": statistics.mean(selected),
        "zero_draft_rate": sum(value == 0 for value in selected) / rounds,
        "full_block_rate": sum(
            value == record["proposed"]
            for value, record in zip(selected, records)
        ) / rounds,
        "premature_cut_rate": sum(
            value < accepted_count
            for value, accepted_count in zip(selected, accepted)
        ) / rounds,
        "lost_accepted_drafts": sum(
            max(0, accepted_count - value)
            for value, accepted_count in zip(selected, accepted)
        ),
        "wasted_verified_positions": sum(
            max(0, value - accepted_count)
            for value, accepted_count in zip(selected, accepted)
        ),
        "local_progress": progress,
        "progress_retention": progress / fixed_progress,
        "target_positions": target_positions,
        "target_positions_per_progress": positions_per_progress,
        "target_position_proxy_ratio": (
            positions_per_progress / fixed_positions_per_progress
        ),
        "target_evals_per_progress": evals_per_progress,
        "target_eval_amplification": (
            evals_per_progress / fixed_evals_per_progress
        ),
    }


def candidate_thresholds(records):
    values = {0.0}
    for record in records:
        values.update(
            math.nextafter(value, math.inf)
            for value in record["confidences"]
        )
    return sorted(values)


def choose_from_candidates(candidates, retention_floor):
    eligible = [
        item for item in candidates
        if item["progress_retention"] >= retention_floor
    ]
    if not eligible:
        raise RuntimeError(f"no policy satisfies retention floor {retention_floor}")
    return min(
        eligible,
        key=lambda item: (
            item["target_positions_per_progress"],
            item["target_eval_amplification"],
            -item["progress_retention"],
            item["threshold"],
        ),
    )


def choose_policy(records, retention_floor, thresholds=None):
    candidates = [
        policy_metrics(records, threshold=threshold)
        for threshold in (
            candidate_thresholds(records) if thresholds is None else thresholds
        )
    ]
    return choose_from_candidates(candidates, retention_floor), candidates


def leave_one_task_out(records, labels, retention_floors):
    # A fixed grid avoids fitting thresholds to single floating-point confidence
    # values and keeps the 32-fold validation inexpensive. The pooled in-sample
    # frontier still uses every exact confidence breakpoint.
    validation_thresholds = [value / 200.0 for value in range(201)]
    policies = {str(floor): {} for floor in retention_floors}
    for label in labels:
        training = [record for record in records if record["sample"] != label]
        candidates = [
            policy_metrics(training, threshold=threshold)
            for threshold in validation_thresholds
        ]
        for floor in retention_floors:
            policies[str(floor)][label] = choose_from_candidates(
                candidates, floor
            )["threshold"]

    results = {}
    for floor in retention_floors:
        selected_by_label = policies[str(floor)]
        selected = [
            selected_prefix(
                record["confidences"], selected_by_label[record["sample"]]
            )
            for record in records
        ]
        accepted = [record["accepted"] for record in records]
        progress = sum(min(a, k) + 1 for a, k in zip(accepted, selected))
        fixed_progress = sum(value + 1 for value in accepted)
        target_positions = sum(max(1, value) for value in selected)
        fixed_target_positions = sum(record["proposed"] for record in records)
        rounds = len(records)
        positions_per_progress = target_positions / progress
        fixed_positions_per_progress = fixed_target_positions / fixed_progress
        evals_per_progress = rounds / progress
        fixed_evals_per_progress = rounds / fixed_progress
        thresholds = list(selected_by_label.values())
        results[str(floor)] = {
            "retention_floor": floor,
            "threshold_median": statistics.median(thresholds),
            "threshold_minimum": min(thresholds),
            "threshold_maximum": max(thresholds),
            "mean_selected_drafts": statistics.mean(selected),
            "progress_retention": progress / fixed_progress,
            "target_position_proxy_ratio": (
                positions_per_progress / fixed_positions_per_progress
            ),
            "target_eval_amplification": (
                evals_per_progress / fixed_evals_per_progress
            ),
            "premature_cut_rate": sum(
                value < accepted_count
                for value, accepted_count in zip(selected, accepted)
            ) / rounds,
            "selected_thresholds": selected_by_label,
        }
    return results


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc


def load_run(run_dir):
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    runs_path = run_dir / "runs.csv"
    summary = _load_json(summary_path)
    metadata = _load_json(metadata_path)
    if metadata.get("config", {}).get("acceptance_trace") is not True:
        raise RuntimeError("run metadata does not declare acceptance tracing")
    labels = list(summary.get("samples", {}))
    if not labels:
        raise RuntimeError("run summary has no samples")
    with runs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    runtime_rows = {row["prompt"]: row for row in rows if row["mode"] == "runtime"}
    if set(runtime_rows) != set(labels):
        raise RuntimeError("runtime trace rows do not match summary samples")

    records = []
    for label in labels:
        stderr_path = (run_dir / runtime_rows[label]["stderr_file"]).resolve()
        if stderr_path.parent != run_dir.resolve() or not stderr_path.is_file():
            raise RuntimeError(f"invalid trace stderr path for {label}")
        sample_records = parse_trace(stderr_path.read_bytes(), stderr_path, label)
        audit = common.parse_acceptance_audit(stderr_path.read_bytes(), stderr_path)
        full = [record for record in sample_records if not record["truncated"]]
        truncated = [record for record in sample_records if record["truncated"]]
        expected = summary["samples"][label]
        if len(full) != audit["proposals"] or len(full) != expected["proposals"]:
            raise RuntimeError(f"full proposal count mismatch for {label}")
        if len(truncated) != audit["truncated_proposals"]:
            raise RuntimeError(f"truncated proposal count mismatch for {label}")
        if any(record["proposed"] != audit["block_size"] for record in full):
            raise RuntimeError(f"non-full proposal included for {label}")
        if sum(record["accepted"] for record in full) != audit["accepted_drafts"]:
            raise RuntimeError(f"accepted draft count mismatch for {label}")
        records.extend(full)
    return records, labels, summary, metadata


def _metric_row(name, item):
    return {
        "policy": name,
        "threshold": item.get("threshold"),
        "mean_selected_drafts": item["mean_selected_drafts"],
        "progress_retention": item["progress_retention"],
        "target_position_proxy_ratio": item["target_position_proxy_ratio"],
        "target_eval_amplification": item["target_eval_amplification"],
        "premature_cut_rate": item["premature_cut_rate"],
        "lost_accepted_drafts": item.get("lost_accepted_drafts"),
        "wasted_verified_positions": item.get("wasted_verified_positions"),
    }


def analyze_run(run_dir):
    records, labels, trace_summary, metadata = load_run(run_dir)
    fixed = policy_metrics(records, threshold=0.0)
    oracle = policy_metrics(records, oracle=True)
    threshold_rows = [
        policy_metrics(records, threshold=threshold)
        for threshold in candidate_thresholds(records)
    ]
    in_sample = {
        str(floor): choose_from_candidates(threshold_rows, floor)
        for floor in RETENTION_FLOORS
    }
    cross_validated = leave_one_task_out(records, labels, RETENTION_FLOORS)
    summary = {
        "analysis": "deepspec_confidence_prefix_local_counterfactual",
        "samples": len(labels),
        "proposals": len(records),
        "block_size": trace_summary["aggregate"]["block_size"],
        "confidence_source": "raw sigmoid output; no STS calibration",
        "fixed_block": fixed,
        "oracle_accepted_prefix": oracle,
        "in_sample_policies": in_sample,
        "leave_one_task_out": cross_validated,
        "source_run": str(run_dir),
        "deepspec_reference": metadata.get("deepspec_scheduler_reference"),
        "caveats": [
            "Prefix truncation changes future proposal boundaries, so this is not an exact session replay.",
            "The released DeepSpec path computes the full sidecar block before applying the threshold.",
            "Target cost is not linear in verified positions; proxy ratios are not speed predictions.",
            "No throughput conclusion follows from this diagnostic.",
        ],
    }

    rows = [_metric_row("fixed_k5", fixed), _metric_row("oracle", oracle)]
    rows.extend(
        _metric_row(f"in_sample_retention_{floor:g}", in_sample[str(floor)])
        for floor in RETENTION_FLOORS
    )
    common.write_csv(run_dir / "scheduler_policies.csv", rows, tuple(rows[0]))
    common.write_csv(
        run_dir / "scheduler_thresholds.csv",
        threshold_rows,
        tuple(threshold_rows[0]),
    )
    trace_rows = []
    for record in records:
        trace_rows.append({
            "sample": record["sample"],
            "round": record["round"],
            "accepted": record["accepted"],
            **{
                f"confidence_{position + 1}": value
                for position, value in enumerate(record["confidences"])
            },
        })
    common.write_csv(
        run_dir / "scheduler_trace.csv", trace_rows, tuple(trace_rows[0])
    )
    (run_dir / "scheduler_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(summary)
    (run_dir / "scheduler_summary.md").write_text(report, encoding="utf-8")
    return summary, report


def _fmt_threshold(value):
    return f"{value:.6f}"


def render_report(summary):
    fixed = summary["fixed_block"]
    oracle = summary["oracle_accepted_prefix"]
    lines = [
        "# DSpark Confidence Scheduler Study",
        "",
        "Offline correctness diagnostic only. Throughput values are intentionally omitted.",
        "The policy matches DeepSpec's first-raw-confidence-below-threshold prefix rule.",
        "",
        f"- Samples: {summary['samples']}",
        f"- Full proposals: {summary['proposals']}",
        f"- Fixed block: K={summary['block_size']}",
        "- Confidence: raw sigmoid output; no STS calibration",
        "",
        "## In-Sample Frontier",
        "",
        "| policy | threshold | mean K | progress retained | target-position proxy | eval/round amplification | premature cuts |",
        "|:---|---:|---:|---:|---:|---:|---:|",
        f"| fixed K=5 | 0 | {fixed['mean_selected_drafts']:.3f} | "
        f"{fixed['progress_retention']:.3f} | 1.000x | 1.000x | 0.0% |",
    ]
    for floor in RETENTION_FLOORS:
        item = summary["in_sample_policies"][str(floor)]
        lines.append(
            f"| >= {floor:.1%} retention | {_fmt_threshold(item['threshold'])} | "
            f"{item['mean_selected_drafts']:.3f} | "
            f"{item['progress_retention']:.3f} | "
            f"{item['target_position_proxy_ratio']:.3f}x | "
            f"{item['target_eval_amplification']:.3f}x | "
            f"{item['premature_cut_rate']:.1%} |"
        )
    lines.append(
        f"| oracle K=accepted | n/a | {oracle['mean_selected_drafts']:.3f} | "
        f"1.000 | {oracle['target_position_proxy_ratio']:.3f}x | "
        "1.000x | 0.0% |"
    )
    lines.extend([
        "",
        "## Leave-One-Task-Out Check",
        "",
        "Each task uses a threshold selected from the other tasks.",
        "",
        "| training floor | threshold median (range) | mean K | held-out progress | target-position proxy | eval/round amplification | premature cuts |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for floor in RETENTION_FLOORS:
        item = summary["leave_one_task_out"][str(floor)]
        lines.append(
            f"| {floor:.1%} | {_fmt_threshold(item['threshold_median'])} "
            f"({_fmt_threshold(item['threshold_minimum'])}-"
            f"{_fmt_threshold(item['threshold_maximum'])}) | "
            f"{item['mean_selected_drafts']:.3f} | "
            f"{item['progress_retention']:.3f} | "
            f"{item['target_position_proxy_ratio']:.3f}x | "
            f"{item['target_eval_amplification']:.3f}x | "
            f"{item['premature_cut_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        "- This is a local counterfactual, not an exact replay: changing K changes later proposal boundaries.",
        "- DeepSpec computes the complete sidecar block before selecting the confidence prefix, so this policy does not reduce sidecar rounds by itself.",
        "- Target cost is not linear in verified positions; the target-position proxy is not a speed prediction.",
        "- The oracle uses future acceptance and is only a non-implementable lower bound.",
        "- No throughput conclusion follows from this diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a traced DSpark HumanEval acceptance run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    _, report = analyze_run(run_dir)
    print(report.rstrip())
    print(f"Raw trace: {run_dir / 'scheduler_trace.csv'}")
    print(f"Threshold sweep: {run_dir / 'scheduler_thresholds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
