#!/usr/bin/env python3
"""Screen causal pre-sidecar routing policies from a DSpark oracle trace."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


ANALYSIS = "dspark_pre_sidecar_router_predictor_study"
ORACLE_ANALYSIS = "dspark_humaneval_round_break_even_oracle"
THRESHOLD = "0.75"
MIN_ROUTE_SHARE = 0.02
MIN_POOLED_RATIO = 1.01
MIN_GEOMETRIC_RATIO = 1.01
MIN_TASK_RATIO = 0.95


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc


def _parse_bool(value, path, row_number):
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise RuntimeError(
            f"invalid boolean in {path} row {row_number}: {value}"
        )
    return normalized == "true"


def load_oracle_run(run_dir):
    summary_path = run_dir / "summary.json"
    rounds_path = run_dir / "rounds.csv"
    summary = _load_json(summary_path)
    if summary.get("analysis") != ORACLE_ANALYSIS:
        raise RuntimeError("input is not a DSpark round break-even oracle")
    if summary.get("threshold") != THRESHOLD:
        raise RuntimeError(
            f"oracle threshold must be {THRESHOLD}, got {summary.get('threshold')}"
        )

    try:
        with rounds_path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError(f"cannot read {rounds_path}: {exc}") from exc
    if not source_rows:
        raise RuntimeError("oracle trace has no rounds")

    rows = []
    for row_number, source in enumerate(source_rows, 2):
        try:
            row = {
                "task": source["task"],
                "round": int(source["round"]),
                "proposed": int(source["proposed"]),
                "selected": int(source["selected"]),
                "verified": int(source["verified"]),
                "accepted": int(source["accepted"]),
                "committed": int(source["committed"]),
                "baseline_ms": float(source["baseline_ms"]),
                "accounted_dspark_ms": float(source["accounted_dspark_ms"]),
                "current_profitable": _parse_bool(
                    source["current_profitable"], rounds_path, row_number
                ),
                "confidences": tuple(
                    float(value) for value in source["confidences"].split(",")
                    if value
                ),
            }
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"invalid oracle row in {rounds_path} row {row_number}: {exc}"
            ) from exc
        counts = (
            row["proposed"], row["selected"], row["verified"],
            row["accepted"], row["committed"],
        )
        if any(value < 0 for value in counts):
            raise RuntimeError(f"negative count in {rounds_path} row {row_number}")
        # The final capacity-limited round may verify only a prefix of the
        # confidence-selected block.
        if not (
            0 <= row["accepted"] <= row["verified"] <=
            row["selected"] <= row["proposed"]
        ):
            raise RuntimeError(f"invalid widths in {rounds_path} row {row_number}")
        if row["committed"] != max(1, row["accepted"]):
            raise RuntimeError(
                f"invalid committed progress in {rounds_path} row {row_number}"
            )
        if len(row["confidences"]) != row["proposed"]:
            raise RuntimeError(
                f"confidence count mismatch in {rounds_path} row {row_number}"
            )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in row["confidences"]
        ):
            raise RuntimeError(
                f"invalid confidence in {rounds_path} row {row_number}"
            )
        if row["baseline_ms"] <= 0.0 or row["accounted_dspark_ms"] <= 0.0:
            raise RuntimeError(f"invalid cost in {rounds_path} row {row_number}")
        profitable = row["accounted_dspark_ms"] < row["baseline_ms"]
        if row["current_profitable"] != profitable:
            raise RuntimeError(
                f"profitability mismatch in {rounds_path} row {row_number}"
            )
        rows.append(row)

    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for task, task_rows in by_task.items():
        task_rows.sort(key=lambda item: item["round"])
        expected = list(range(1, len(task_rows) + 1))
        actual = [item["round"] for item in task_rows]
        if actual != expected:
            raise RuntimeError(f"non-sequential rounds for {task}")

    if set(by_task) != set(summary.get("tasks", {})):
        raise RuntimeError("oracle CSV tasks do not match its summary")
    aggregate = summary.get("aggregate", {})
    checks = {
        "rounds": len(rows),
        "baseline_ms": sum(row["baseline_ms"] for row in rows),
        "accounted_dspark_ms": sum(
            row["accounted_dspark_ms"] for row in rows
        ),
        "profitable_rounds": sum(row["current_profitable"] for row in rows),
    }
    for key, actual in checks.items():
        expected = aggregate.get(key)
        if expected is None or not math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6
        ):
            raise RuntimeError(f"oracle aggregate mismatch for {key}")
    return summary, by_task, summary_path, rounds_path


def candidate_policies():
    policies = []
    for feature in ("accepted", "committed", "selected"):
        for threshold in range(1, 6):
            policies.append({
                "name": f"previous {feature} >= {threshold}",
                "kind": "previous_count",
                "feature": feature,
                "threshold": float(threshold),
            })
    for feature in ("accepted", "committed", "selected"):
        for window in (2, 3, 4, 5, 8, 12):
            for doubled in range(4, 11):
                threshold = doubled / 2.0
                policies.append({
                    "name": (
                        f"last {window} mean {feature} >= {threshold:g}"
                    ),
                    "kind": "rolling_mean",
                    "feature": feature,
                    "window": window,
                    "threshold": threshold,
                })
    confidence_thresholds = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
    for feature in ("first_confidence", "selected_min_confidence"):
        for threshold in confidence_thresholds:
            policies.append({
                "name": f"previous {feature} >= {threshold:g}",
                "kind": "previous_confidence",
                "feature": feature,
                "threshold": threshold,
            })
    for accepted in range(1, 6):
        for confidence in confidence_thresholds:
            policies.append({
                "name": (
                    f"previous accepted >= {accepted} and selected minimum "
                    f"confidence >= {confidence:g}"
                ),
                "kind": "accepted_and_confidence",
                "accepted": accepted,
                "threshold": confidence,
            })
    return policies


def policy_routes(policy, history):
    previous = history[-1]
    kind = policy["kind"]
    if kind == "previous_count":
        return previous[policy["feature"]] >= policy["threshold"]
    if kind == "rolling_mean":
        values = [
            row[policy["feature"]] for row in history[-policy["window"]:]
        ]
        return statistics.mean(values) >= policy["threshold"]
    if kind == "previous_confidence":
        if policy["feature"] == "first_confidence":
            value = previous["confidences"][0]
        else:
            width = max(1, previous["selected"])
            value = min(previous["confidences"][:width])
        return value >= policy["threshold"]
    if kind == "accepted_and_confidence":
        width = max(1, previous["selected"])
        confidence = min(previous["confidences"][:width])
        return (
            previous["accepted"] >= policy["accepted"] and
            confidence >= policy["threshold"]
        )
    raise ValueError(f"unknown policy kind {kind}")


def evaluate_policy(by_task, tasks, policy_by_task):
    baseline_ms = 0.0
    policy_ms = 0.0
    routed_rounds = 0
    task_ratios = {}
    for task in tasks:
        task_baseline = 0.0
        task_policy = 0.0
        rows = by_task[task]
        policy = policy_by_task[task]
        for index, row in enumerate(rows):
            # The first round has no causal lagged observation and therefore
            # defaults to ordinary target decoding.
            route = index > 0 and policy_routes(policy, rows[:index])
            routed_rounds += int(route)
            task_baseline += row["baseline_ms"]
            task_policy += (
                row["accounted_dspark_ms"] if route else row["baseline_ms"]
            )
        baseline_ms += task_baseline
        policy_ms += task_policy
        task_ratios[task] = task_baseline / task_policy
    ratios = list(task_ratios.values())
    return {
        "baseline_ms": baseline_ms,
        "policy_ms": policy_ms,
        "pooled_ratio": baseline_ms / policy_ms,
        "task_median_ratio": statistics.median(ratios),
        "task_geometric_ratio": math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)
        ),
        "minimum_task_ratio": min(ratios),
        "tasks_faster": sum(value > 1.0 for value in ratios),
        "tasks_equal": sum(math.isclose(value, 1.0) for value in ratios),
        "tasks_slower": sum(value < 1.0 for value in ratios),
        "routed_rounds": routed_rounds,
        "total_rounds": sum(len(by_task[task]) for task in tasks),
        "route_share": routed_rounds / sum(len(by_task[task]) for task in tasks),
        "task_ratios": task_ratios,
    }


def _policy_for_all(tasks, policy):
    return {task: policy for task in tasks}


def rank_policies(by_task, tasks, policies):
    minimum_routes = math.ceil(
        MIN_ROUTE_SHARE * sum(len(by_task[task]) for task in tasks)
    )
    ranked = []
    for policy in policies:
        metrics = evaluate_policy(
            by_task, tasks, _policy_for_all(tasks, policy)
        )
        if metrics["routed_rounds"] >= minimum_routes:
            ranked.append((policy, metrics))
    if not ranked:
        raise RuntimeError("no candidate satisfies minimum routing coverage")
    ranked.sort(
        key=lambda item: (
            item[1]["pooled_ratio"],
            item[1]["task_geometric_ratio"],
            -item[1]["routed_rounds"],
            item[0]["name"],
        ),
        reverse=True,
    )
    return ranked


def leave_one_task_out(by_task, tasks, policies):
    selected = {}
    folds = []
    for held_out in tasks:
        training = [task for task in tasks if task != held_out]
        policy, training_metrics = rank_policies(
            by_task, training, policies
        )[0]
        selected[held_out] = policy
        held_metrics = evaluate_policy(
            by_task, [held_out], {held_out: policy}
        )
        folds.append({
            "task": held_out,
            "policy": policy["name"],
            "training_ratio": training_metrics["pooled_ratio"],
            "held_out_ratio": held_metrics["pooled_ratio"],
            "held_out_routed_rounds": held_metrics["routed_rounds"],
        })
    return evaluate_policy(by_task, tasks, selected), folds


def transition_summary(by_task):
    counts = {
        "unprofitable_to_unprofitable": 0,
        "unprofitable_to_profitable": 0,
        "profitable_to_unprofitable": 0,
        "profitable_to_profitable": 0,
    }
    for rows in by_task.values():
        for previous, current in zip(rows, rows[1:]):
            left = "profitable" if previous["current_profitable"] else "unprofitable"
            right = "profitable" if current["current_profitable"] else "unprofitable"
            counts[f"{left}_to_{right}"] += 1
    unprofitable_total = (
        counts["unprofitable_to_unprofitable"] +
        counts["unprofitable_to_profitable"]
    )
    profitable_total = (
        counts["profitable_to_unprofitable"] +
        counts["profitable_to_profitable"]
    )
    return {
        **counts,
        "profitable_after_unprofitable": (
            counts["unprofitable_to_profitable"] / unprofitable_total
        ),
        "profitable_after_profitable": (
            counts["profitable_to_profitable"] / profitable_total
        ),
    }


def analyze(summary, by_task, summary_path, rounds_path):
    tasks = sorted(by_task)
    policies = candidate_policies()
    ranked = rank_policies(by_task, tasks, policies)
    best_policy, best_metrics = ranked[0]
    loto_metrics, folds = leave_one_task_out(by_task, tasks, policies)
    gate = {
        "pooled_ratio_at_least_1_01": (
            loto_metrics["pooled_ratio"] >= MIN_POOLED_RATIO
        ),
        "geometric_ratio_at_least_1_01": (
            loto_metrics["task_geometric_ratio"] >= MIN_GEOMETRIC_RATIO
        ),
        "minimum_task_ratio_at_least_0_95": (
            loto_metrics["minimum_task_ratio"] >= MIN_TASK_RATIO
        ),
        "route_share_at_least_0_02": (
            loto_metrics["route_share"] >= MIN_ROUTE_SHARE
        ),
    }
    aggregate = summary["aggregate"]
    result = {
        "analysis": ANALYSIS,
        "source_summary": str(summary_path),
        "source_rounds": str(rounds_path),
        "threshold": THRESHOLD,
        "task_count": len(tasks),
        "round_count": sum(len(rows) for rows in by_task.values()),
        "reference": {
            "current_accounted_ratio": aggregate["accounted_ratio"],
            "perfect_route_oracle_ratio": aggregate["route_oracle_ratio"],
            "profitable_round_share": aggregate["profitable_round_share"],
            "profitable_token_share": aggregate["profitable_token_share"],
        },
        "transitions": transition_summary(by_task),
        "candidate_count": len(policies),
        "best_in_sample": {
            "policy": best_policy,
            **best_metrics,
        },
        "leave_one_task_out": loto_metrics,
        "folds": folds,
        "promotion_gate": {
            "passed": all(gate.values()),
            "checks": gate,
        },
        "candidate_ranking": [
            {
                "rank": index,
                "policy": policy,
                **{key: value for key, value in metrics.items()
                   if key != "task_ratios"},
            }
            for index, (policy, metrics) in enumerate(ranked, 1)
        ],
        "caveats": [
            "All results are local counterfactuals over the frozen oracle trace, not throughput measurements.",
            "Only previous-round or rolling-history features are used; current-round confidence, acceptance, and cost are excluded.",
            "The study optimistically exposes lagged proposal observations even after a policy-routed baseline round; a real pre-sidecar router would need probing and would have less information.",
            "Routing changes later token and proposal boundaries, so even held-out ratios are ceilings rather than runtime predictions.",
            "The first round always uses ordinary target decoding because no causal lagged observation exists.",
        ],
    }
    return result


def render_report(summary):
    reference = summary["reference"]
    best = summary["best_in_sample"]
    loto = summary["leave_one_task_out"]
    transition = summary["transitions"]
    gate = summary["promotion_gate"]
    policy_counts = {}
    for fold in summary["folds"]:
        policy_counts[fold["policy"]] = policy_counts.get(fold["policy"], 0) + 1
    lines = [
        "# DSpark Pre-Sidecar Router Predictor Study",
        "",
        "Offline diagnostic only; no model execution or throughput benchmark was run.",
        "Every candidate uses only lagged features, but the skipped-round observation assumption still makes this an optimistic ceiling.",
        "",
        "## Aggregate",
        "",
        "| model | baseline-equivalent ratio | geometric task ratio | minimum task | routed rounds |",
        "|:---|---:|---:|---:|---:|",
        f"| Current all-DSpark accounting | {reference['current_accounted_ratio']:.4f}x | n/a | n/a | {summary['round_count']} |",
        f"| Perfect future-known router | {reference['perfect_route_oracle_ratio']:.4f}x | n/a | n/a | n/a |",
        f"| Best optimistic in-sample lagged policy | {best['pooled_ratio']:.4f}x | {best['task_geometric_ratio']:.4f}x | {best['minimum_task_ratio']:.4f}x | {best['routed_rounds']}/{best['total_rounds']} |",
        f"| Leave-one-task-out lagged policy | {loto['pooled_ratio']:.4f}x | {loto['task_geometric_ratio']:.4f}x | {loto['minimum_task_ratio']:.4f}x | {loto['routed_rounds']}/{loto['total_rounds']} |",
        f"| Always baseline | 1.0000x | 1.0000x | 1.0000x | 0/{summary['round_count']} |",
        "",
        f"- Best in-sample policy: `{best['policy']['name']}`.",
        f"- Held-out tasks faster/equal/slower than baseline: {loto['tasks_faster']}/{loto['tasks_equal']}/{loto['tasks_slower']}.",
        f"- Candidate policies screened: {summary['candidate_count']}.",
        "",
        "## Lag Persistence",
        "",
        "| previous round | next profitable | transitions |",
        "|:---|---:|---:|",
        f"| unprofitable | {transition['profitable_after_unprofitable']:.1%} | {transition['unprofitable_to_profitable']}/{transition['unprofitable_to_unprofitable'] + transition['unprofitable_to_profitable']} |",
        f"| profitable | {transition['profitable_after_profitable']:.1%} | {transition['profitable_to_profitable']}/{transition['profitable_to_unprofitable'] + transition['profitable_to_profitable']} |",
        "",
        "## Held-Out Selections",
        "",
        "| policy selected on 31 tasks | folds |",
        "|:---|---:|",
    ]
    for name, count in sorted(
        policy_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Promotion Gate",
        "",
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
    ])
    labels = {
        "pooled_ratio_at_least_1_01": "held-out pooled ratio at least 1.01x",
        "geometric_ratio_at_least_1_01": "held-out geometric ratio at least 1.01x",
        "minimum_task_ratio_at_least_0_95": "no held-out task below 0.95x",
        "route_share_at_least_0_02": "held-out routed-round share at least 2%",
    }
    for key, passed in gate["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {labels[key]}.")
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        *[f"- {item}" for item in summary["caveats"]],
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Screen causal lagged DSpark pre-sidecar routing policies."
    )
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.oracle_run.resolve()
    output_dir = args.output_dir.resolve()
    oracle, by_task, summary_path, rounds_path = load_oracle_run(run_dir)
    summary = analyze(oracle, by_task, summary_path, rounds_path)
    report = render_report(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    write_csv(
        output_dir / "folds.csv",
        ("task", "policy", "training_ratio", "held_out_ratio",
         "held_out_routed_rounds"),
        summary["folds"],
    )
    write_csv(
        output_dir / "candidate_ranking.csv",
        ("rank", "policy", "pooled_ratio", "task_geometric_ratio",
         "minimum_task_ratio", "routed_rounds", "total_rounds", "route_share"),
        ({
            "rank": row["rank"],
            "policy": row["policy"]["name"],
            "pooled_ratio": row["pooled_ratio"],
            "task_geometric_ratio": row["task_geometric_ratio"],
            "minimum_task_ratio": row["minimum_task_ratio"],
            "routed_rounds": row["routed_rounds"],
            "total_rounds": row["total_rounds"],
            "route_share": row["route_share"],
        } for row in summary["candidate_ranking"]),
    )
    print(report.rstrip())
    print(f"Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
