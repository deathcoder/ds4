#!/usr/bin/env python3
"""Compare oMLX depth-two MTP with custom and stock verify QMM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import analyze_omlx_humaneval_sweep as sweep


MODES = ("baseline", "mtp2_stock_qmm", "mtp2")


def validate_modes(runs: dict[str, dict]) -> list[str]:
    tasks = sweep.validate_runs(runs)
    expected = {
        "baseline": (False, None, True),
        "mtp2_stock_qmm": (True, 2, False),
        "mtp2": (True, 2, True),
    }
    for mode, values in expected.items():
        settings = runs[mode]["metadata"].get("mode_settings") or {}
        actual = (
            settings.get("mtp_enabled"),
            settings.get("mtp_num_draft_tokens"),
            settings.get("custom_verify_qmm", True),
        )
        if actual != values:
            raise SystemExit(f"{mode} has unexpected mode settings: {settings}")
    return tasks


def comparison(numerator: dict, denominator: dict, tasks: list[str]) -> dict:
    ratios = [
        float(numerator[task]["generation_tps"])
        / float(denominator[task]["generation_tps"])
        for task in tasks
    ]
    numerator_tps = [float(numerator[task]["generation_tps"]) for task in tasks]
    denominator_tps = [float(denominator[task]["generation_tps"]) for task in tasks]
    return {
        "ratio_of_medians": statistics.median(numerator_tps)
        / statistics.median(denominator_tps),
        "paired_ratio_median": statistics.median(ratios),
        "paired_ratio_geometric_mean": statistics.geometric_mean(ratios),
        "faster_tasks": sum(ratio > 1.0 for ratio in ratios),
        "minimum_ratio": min(ratios),
        "maximum_ratio": max(ratios),
    }


def summarize(runs: dict[str, dict], tasks: list[str]) -> dict:
    baseline = runs["baseline"]["tasks"]
    stock = runs["mtp2_stock_qmm"]["tasks"]
    custom = runs["mtp2"]["tasks"]
    task_rows = []
    for task in tasks:
        task_rows.append(
            {
                "task": task,
                "baseline_tps": baseline[task]["generation_tps"],
                "stock_tps": stock[task]["generation_tps"],
                "custom_tps": custom[task]["generation_tps"],
                "custom_stock_ratio": float(custom[task]["generation_tps"])
                / float(stock[task]["generation_tps"]),
                "stock_baseline_exact": stock[task]["output_sha256"]
                == baseline[task]["output_sha256"],
                "custom_baseline_exact": custom[task]["output_sha256"]
                == baseline[task]["output_sha256"],
                "custom_stock_exact": custom[task]["output_sha256"]
                == stock[task]["output_sha256"],
                "token_counts_equal": len(
                    {
                        baseline[task]["completion_tokens"],
                        stock[task]["completion_tokens"],
                        custom[task]["completion_tokens"],
                    }
                )
                == 1,
            }
        )
    return {
        "sample_count": len(tasks),
        "baseline_median": statistics.median(
            float(baseline[task]["generation_tps"]) for task in tasks
        ),
        "stock_median": statistics.median(
            float(stock[task]["generation_tps"]) for task in tasks
        ),
        "custom_median": statistics.median(
            float(custom[task]["generation_tps"]) for task in tasks
        ),
        "stock_vs_baseline": comparison(stock, baseline, tasks),
        "custom_vs_baseline": comparison(custom, baseline, tasks),
        "custom_vs_stock": comparison(custom, stock, tasks),
        "tasks": task_rows,
        "stock_baseline_exact_tasks": sum(
            item["stock_baseline_exact"] for item in task_rows
        ),
        "custom_baseline_exact_tasks": sum(
            item["custom_baseline_exact"] for item in task_rows
        ),
        "custom_stock_exact_tasks": sum(
            item["custom_stock_exact"] for item in task_rows
        ),
        "token_count_gate": all(item["token_counts_equal"] for item in task_rows),
    }


def render_report(summary: dict) -> str:
    custom_stock = summary["custom_vs_stock"]
    lines = [
        "# oMLX Depth-Two Verify-QMM Ablation",
        "",
        "Cross-process diagnostic only. Native MTP depth, controller, checkpoint, and protocol are identical; only custom verify-QMM eligibility differs.",
        "",
        "| mode | median | paired geometric vs baseline |",
        "|:---|---:|---:|",
        f"| baseline | {summary['baseline_median']:.2f} t/s | 1.0000x |",
        f"| stock verify QMM | {summary['stock_median']:.2f} t/s | {summary['stock_vs_baseline']['paired_ratio_geometric_mean']:.4f}x |",
        f"| custom verify QMM | {summary['custom_median']:.2f} t/s | {summary['custom_vs_baseline']['paired_ratio_geometric_mean']:.4f}x |",
        "",
        "## Custom Versus Stock",
        "",
        f"- Ratio of medians: {custom_stock['ratio_of_medians']:.4f}x.",
        f"- Median paired ratio: {custom_stock['paired_ratio_median']:.4f}x.",
        f"- Geometric paired ratio: {custom_stock['paired_ratio_geometric_mean']:.4f}x.",
        f"- Custom faster on {custom_stock['faster_tasks']}/{summary['sample_count']} tasks; range {custom_stock['minimum_ratio']:.4f}x-{custom_stock['maximum_ratio']:.4f}x.",
        f"- Exact outputs versus baseline: stock {summary['stock_baseline_exact_tasks']}/{summary['sample_count']}, custom {summary['custom_baseline_exact_tasks']}/{summary['sample_count']}.",
        f"- Custom/stock exact agreement: {summary['custom_stock_exact_tasks']}/{summary['sample_count']} tasks.",
        f"- Completion-token-count gate: {'PASS' if summary['token_count_gate'] else 'FAIL'}.",
        "",
        "This isolates mechanism direction, not a final speed claim. A material custom-kernel win needs a balanced-order confirmation before any ds4 port.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--custom", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    runs = {
        "baseline": sweep.load_run(args.baseline, "baseline"),
        "mtp2_stock_qmm": sweep.load_run(args.stock, "mtp2_stock_qmm"),
        "mtp2": sweep.load_run(args.custom, "mtp2"),
    }
    tasks = validate_modes(runs)
    summary = summarize(runs, tasks)
    report = render_report(summary)
    if args.output_dir:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if summary["token_count_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
