#!/usr/bin/env python3
"""Audit a measured-cost DSpark width policy from frozen proposal traces."""

import argparse
import csv
import json
import math
from pathlib import Path


CANDIDATE_WIDTHS = (0, 2, 3, 4, 5)
CALIBRATION_POWERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
STATIC_THRESHOLD = 0.75


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc


def load_trace(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError(f"cannot read trace {path}: {exc}") from exc
    required = {"sample", "round", "accepted"} | {
        f"confidence_{position}" for position in range(1, 6)
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("trace does not contain five-position proposal records")

    records = []
    next_round = {}
    for row in rows:
        try:
            sample = row["sample"]
            round_number = int(row["round"])
            accepted = int(row["accepted"])
            confidences = tuple(
                float(row[f"confidence_{position}"])
                for position in range(1, 6)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid proposal trace row: {exc}") from exc
        if not sample or round_number != next_round.get(sample, 1):
            raise RuntimeError(f"non-sequential proposal trace for {sample!r}")
        if not 0 <= accepted <= 5:
            raise RuntimeError(f"invalid accepted prefix for {sample} round {round_number}")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in confidences):
            raise RuntimeError(f"invalid confidence for {sample} round {round_number}")
        next_round[sample] = round_number + 1
        records.append({
            "sample": sample,
            "round": round_number,
            "accepted": accepted,
            "confidences": confidences,
        })
    return records


def load_costs(path):
    summary = load_json(path)
    if summary.get("analysis") != "dspark_humaneval_cumulative_exact_verifier_cost":
        raise RuntimeError("cost reference is not the cumulative exact-verifier audit")
    aggregate = summary.get("aggregate", {})
    evals = aggregate.get("target_evals", 0)
    if not isinstance(evals, int) or evals <= 0:
        raise RuntimeError("cost reference has no target evaluations")
    fixed_ms = (
        float(aggregate["sidecar_ms"]) + float(aggregate["residual_ms"])
    ) / evals
    target_ms = {}
    for width in range(1, 6):
        try:
            value = float(summary["verifier_widths"][str(width)]["target_ms_per_eval"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"cost reference lacks verifier width {width}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"invalid verifier width-{width} cost")
        target_ms[width] = value
    return {
        "summary": summary,
        "fixed_ms_per_round": fixed_ms,
        "target_ms_per_eval": target_ms,
        "cycle_ms": {
            width: fixed_ms + target_ms[max(1, width)]
            for width in CANDIDATE_WIDTHS
        },
    }


def selected_prefix(confidences, threshold):
    for position, confidence in enumerate(confidences):
        if confidence < threshold:
            return position
    return len(confidences)


def runtime_progress(accepted, width):
    # ds4's K=0 and K=1 routes both perform ordinary one-token target eval.
    if width < 2:
        return 1
    return min(accepted, width) + 1


def expected_progress(confidences, width, calibration_power=1.0):
    if width < 2:
        return 1.0
    expected = 1.0
    survival = 1.0
    for confidence in confidences[:width]:
        survival *= confidence ** calibration_power
        expected += survival
    return expected


def cost_aware_width(confidences, cycle_ms, calibration_power=1.0):
    def score(width):
        return expected_progress(confidences, width, calibration_power) / cycle_ms[width]

    # Prefer the shallower width on exact ties.
    return max(CANDIDATE_WIDTHS, key=lambda width: (score(width), -width))


def policy_metrics(records, cycle_ms, selector):
    task_totals = {}
    widths = {width: 0 for width in range(6)}
    progress = 0
    cost_ms = 0.0
    for record in records:
        width = selector(record)
        if width not in range(6):
            raise RuntimeError(f"policy selected invalid width {width}")
        local_progress = runtime_progress(record["accepted"], width)
        local_cost = cycle_ms[0 if width < 2 else width]
        progress += local_progress
        cost_ms += local_cost
        widths[width] += 1
        task = task_totals.setdefault(record["sample"], {"progress": 0, "cost_ms": 0.0})
        task["progress"] += local_progress
        task["cost_ms"] += local_cost
    return {
        "rounds": len(records),
        "progress": progress,
        "cost_ms": cost_ms,
        "progress_per_second_proxy": 1000.0 * progress / cost_ms,
        "mean_selected_width": sum(width * count for width, count in widths.items()) / len(records),
        "width_counts": {str(width): count for width, count in widths.items()},
        "tasks": {
            task: {
                **values,
                "progress_per_second_proxy": 1000.0 * values["progress"] / values["cost_ms"],
            }
            for task, values in task_totals.items()
        },
    }


def compare(candidate, reference):
    candidate["ratio_vs_static"] = (
        candidate["progress_per_second_proxy"] /
        reference["progress_per_second_proxy"]
    )
    task_ratios = []
    for task, values in candidate["tasks"].items():
        ratio = (
            values["progress_per_second_proxy"] /
            reference["tasks"][task]["progress_per_second_proxy"]
        )
        values["ratio_vs_static"] = ratio
        task_ratios.append(ratio)
    candidate["task_ratio_geometric_mean"] = math.exp(
        sum(math.log(value) for value in task_ratios) / len(task_ratios)
    )
    candidate["tasks_better_equal_worse"] = [
        sum(value > 1.0 for value in task_ratios),
        sum(value == 1.0 for value in task_ratios),
        sum(value < 1.0 for value in task_ratios),
    ]
    return candidate


def analyze(records, costs, trace_path=None, cost_path=None):
    cycle_ms = costs["cycle_ms"]
    static = policy_metrics(
        records,
        cycle_ms,
        lambda record: selected_prefix(record["confidences"], STATIC_THRESHOLD),
    )
    candidate = compare(policy_metrics(
        records,
        cycle_ms,
        lambda record: cost_aware_width(record["confidences"], cycle_ms),
    ), static)
    fixed_k2 = compare(policy_metrics(records, cycle_ms, lambda _record: 2), static)
    oracle = compare(policy_metrics(
        records,
        cycle_ms,
        lambda record: max(
            CANDIDATE_WIDTHS,
            key=lambda width: (
                runtime_progress(record["accepted"], width) / cycle_ms[width],
                -width,
            ),
        ),
    ), static)
    calibration = {}
    for power in CALIBRATION_POWERS:
        item = compare(policy_metrics(
            records,
            cycle_ms,
            lambda record, power=power: cost_aware_width(
                record["confidences"], cycle_ms, power
            ),
        ), static)
        calibration[str(power)] = item
    return {
        "analysis": "dspark_measured_cost_scheduler_local_counterfactual",
        "samples": len({record["sample"] for record in records}),
        "proposals": len(records),
        "static_threshold": STATIC_THRESHOLD,
        "candidate_widths": list(CANDIDATE_WIDTHS),
        "cost_model": {
            "fixed_ms_per_round": costs["fixed_ms_per_round"],
            "target_ms_per_eval": {
                str(key): value for key, value in costs["target_ms_per_eval"].items()
            },
            "cycle_ms": {str(key): value for key, value in cycle_ms.items()},
        },
        "static_threshold_policy": static,
        "cost_aware_policy": candidate,
        "fixed_k2_control": fixed_k2,
        "realized_round_oracle": oracle,
        "calibration_stress": calibration,
        "source_trace": str(trace_path) if trace_path else None,
        "source_cost": str(cost_path) if cost_path else None,
        "caveats": [
            "Changing width changes later proposal boundaries, so this is not an exact session replay.",
            "The proposal trace predates the promoted Metal kernels; acceptance is reused because promoted routes remained byte-exact.",
            "Width costs are pooled synchronized stats from another run, not per-round timings.",
            "Raw confidence values are not STS-calibrated probabilities.",
            "The full five-token sidecar cost is charged to every round because DSpark selects width after drafting.",
            "Proxy improvement is a screening result, not a throughput prediction.",
        ],
    }


def render_report(summary):
    static = summary["static_threshold_policy"]
    candidate = summary["cost_aware_policy"]
    fixed = summary["fixed_k2_control"]
    oracle = summary["realized_round_oracle"]
    lines = [
        "# DSpark Measured-Cost Scheduler Audit",
        "",
        "Model-free local counterfactual only; throughput values are intentionally omitted.",
        "Each policy is charged the post-promotion fixed sidecar/residual cost plus the measured exact-verifier cost for its selected width.",
        "",
        "| policy | mean K | progress/s proxy | vs static | task geometric | better/equal/worse |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| static threshold 0.75 | {static['mean_selected_width']:.3f} | {static['progress_per_second_proxy']:.3f} | 1.0000x | 1.0000x | reference |",
        f"| measured-cost confidence | {candidate['mean_selected_width']:.3f} | {candidate['progress_per_second_proxy']:.3f} | {candidate['ratio_vs_static']:.4f}x | {candidate['task_ratio_geometric_mean']:.4f}x | {'/'.join(map(str, candidate['tasks_better_equal_worse']))} |",
        f"| fixed K=2 control | {fixed['mean_selected_width']:.3f} | {fixed['progress_per_second_proxy']:.3f} | {fixed['ratio_vs_static']:.4f}x | {fixed['task_ratio_geometric_mean']:.4f}x | {'/'.join(map(str, fixed['tasks_better_equal_worse']))} |",
        f"| realized round oracle | {oracle['mean_selected_width']:.3f} | {oracle['progress_per_second_proxy']:.3f} | {oracle['ratio_vs_static']:.4f}x | {oracle['task_ratio_geometric_mean']:.4f}x | {'/'.join(map(str, oracle['tasks_better_equal_worse']))} |",
        "",
        "## Width Selection",
        "",
        "| policy | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (
        ("static threshold 0.75", static),
        ("measured-cost confidence", candidate),
        ("realized round oracle", oracle),
    ):
        counts = item["width_counts"]
        lines.append(
            f"| {label} | {counts['0']} | {counts['1']} | {counts['2']} | {counts['3']} | {counts['4']} | {counts['5']} |"
        )
    lines.extend([
        "",
        "## Calibration Stress",
        "",
        "Confidence values are raised to each power before scoring expected progress.",
        "",
        "| power | mean K | vs static | task geometric | better/equal/worse |",
        "|---:|---:|---:|---:|---:|",
    ])
    for power in CALIBRATION_POWERS:
        item = summary["calibration_stress"][str(power)]
        lines.append(
            f"| {power:g} | {item['mean_selected_width']:.3f} | {item['ratio_vs_static']:.4f}x | {item['task_ratio_geometric_mean']:.4f}x | {'/'.join(map(str, item['tasks_better_equal_worse']))} |"
        )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        *[f"- {item}" for item in summary["caveats"]],
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Screen a measured-cost DSpark verifier-width policy."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cost-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace_path = args.trace.resolve()
    cost_path = args.cost_summary.resolve()
    output_dir = args.output_dir.resolve()
    records = load_trace(trace_path)
    costs = load_costs(cost_path)
    summary = analyze(records, costs, trace_path, cost_path)
    report = render_report(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report.rstrip())
    print(f"Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
