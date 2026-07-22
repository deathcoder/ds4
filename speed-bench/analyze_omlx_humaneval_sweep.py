#!/usr/bin/env python3
"""Compare separately captured oMLX HumanEval mode runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


MODES = ("baseline", "mtp1", "mtp2", "mtp3")


def load_run(path: Path, expected_mode: str) -> dict:
    path = path.resolve()
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        metadata = json.loads(
            (path.parent / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {expected_mode} run {path}: {exc}") from exc
    if summary.get("mode") != expected_mode or metadata.get("mode") != expected_mode:
        raise SystemExit(f"{path} is not an {expected_mode} run")
    if metadata.get("experiment") != "omlx_humaneval_mode":
        raise SystemExit(f"{path} has the wrong experiment kind")
    rows = summary.get("samples") or []
    if len(rows) != summary.get("sample_count"):
        raise SystemExit(f"{path} sample count mismatch")
    by_task = {row.get("prompt"): row for row in rows}
    if len(by_task) != len(rows) or None in by_task:
        raise SystemExit(f"{path} has duplicate or unlabeled tasks")
    return {"path": path, "summary": summary, "metadata": metadata, "tasks": by_task}


def validate_runs(runs: dict[str, dict]) -> list[str]:
    baseline = runs["baseline"]
    tasks = list(baseline["tasks"])
    reference = baseline["metadata"]
    for mode, run in runs.items():
        metadata = run["metadata"]
        if list(run["tasks"]) != tasks:
            raise SystemExit(f"{mode} task order differs from baseline")
        for key in (
            "omlx_commit",
            "model_revision",
            "model_config_sha256",
            "model_index_sha256",
            "native_deepseek_kernels",
            "selection",
        ):
            if metadata.get(key) != reference.get(key):
                raise SystemExit(f"{mode} metadata {key} differs from baseline")
        if metadata.get("protocol") != reference.get("protocol"):
            raise SystemExit(f"{mode} protocol differs from baseline")
    return tasks


def summarize(runs: dict[str, dict], tasks: list[str]) -> dict:
    modes = {}
    baseline = runs["baseline"]["tasks"]
    for mode in MODES:
        rows = runs[mode]["tasks"]
        tps = [float(rows[task]["generation_tps"]) for task in tasks]
        intervals = [float(rows[task]["interval_generation_tps"]) for task in tasks]
        if mode == "baseline":
            ratios = [1.0] * len(tasks)
        else:
            ratios = [
                float(rows[task]["generation_tps"])
                / float(baseline[task]["generation_tps"])
                for task in tasks
            ]
        modes[mode] = {
            "generation_tps_median": statistics.median(tps),
            "interval_generation_tps_median": statistics.median(intervals),
            "ratio_of_medians": (
                statistics.median(tps)
                / statistics.median(
                    float(baseline[task]["generation_tps"]) for task in tasks
                )
            ),
            "paired_ratio_median": statistics.median(ratios),
            "paired_ratio_geometric_mean": statistics.geometric_mean(ratios),
            "faster_tasks": sum(ratio > 1.0 for ratio in ratios),
            "minimum_ratio": min(ratios),
        }

    output_mismatches = []
    token_count_mismatches = []
    task_rows = []
    for task in tasks:
        base = baseline[task]
        item = {"task": task, "modes": {}}
        for mode in MODES:
            row = runs[mode]["tasks"][task]
            exact = row["output_sha256"] == base["output_sha256"]
            same_tokens = row["completion_tokens"] == base["completion_tokens"]
            if not exact:
                output_mismatches.append({"task": task, "mode": mode})
            if not same_tokens:
                token_count_mismatches.append({"task": task, "mode": mode})
            item["modes"][mode] = {
                "generation_tps": row["generation_tps"],
                "ratio": (float(row["generation_tps"]) / float(base["generation_tps"])),
                "output_exact": exact,
                "completion_tokens": row["completion_tokens"],
                "mtp_stats": row.get("mtp_stats"),
            }
        task_rows.append(item)
    candidates = [mode for mode in MODES if mode != "baseline"]
    winner = max(
        candidates, key=lambda mode: modes[mode]["paired_ratio_geometric_mean"]
    )
    return {
        "sample_count": len(tasks),
        "modes": modes,
        "tasks": task_rows,
        "output_mismatches": output_mismatches,
        "token_count_mismatches": token_count_mismatches,
        "winner": winner,
        "correctness_gate": not output_mismatches and not token_count_mismatches,
    }


def render_report(summary: dict) -> str:
    lines = [
        "# oMLX DeepSeek V4 Lightning-MTP Sweep",
        "",
        "Each mode was captured in a separate process with the same pinned oMLX source, oQ2e-MTP checkpoint, frozen HumanEval tasks, and cache-disabled greedy protocol.",
        "Cross-process ratios are diagnostic; confirm the selected mode with balanced ordering before promotion.",
        "",
        "| mode | median | interval median | ratio of medians | paired median | geometric mean | wins | minimum |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary["modes"][mode]
        lines.append(
            f"| {mode} | {item['generation_tps_median']:.2f} t/s | "
            f"{item['interval_generation_tps_median']:.2f} t/s | "
            f"{item['ratio_of_medians']:.4f}x | "
            f"{item['paired_ratio_median']:.4f}x | "
            f"{item['paired_ratio_geometric_mean']:.4f}x | "
            f"{item['faster_tasks']}/{summary['sample_count']} | "
            f"{item['minimum_ratio']:.4f}x |"
        )
    lines.extend(
        [
            "",
            f"- Best MTP candidate: `{summary['winner']}`.",
            f"- Exact-output gate: {'PASS' if summary['correctness_gate'] else 'FAIL'}.",
            f"- Output mismatches: {len(summary['output_mismatches'])}.",
            f"- Completion-token mismatches: {len(summary['token_count_mismatches'])}.",
            "",
            "## Tasks",
            "",
            "| task | baseline | mtp1 | mtp2 | mtp3 | exact |",
            "|:---|---:|---:|---:|---:|:---|",
        ]
    )
    for item in summary["tasks"]:
        modes = item["modes"]
        exact = all(mode["output_exact"] for mode in modes.values())
        lines.append(
            f"| {item['task']} | {modes['baseline']['generation_tps']:.2f} t/s | "
            f"{modes['mtp1']['ratio']:.4f}x | {modes['mtp2']['ratio']:.4f}x | "
            f"{modes['mtp3']['ratio']:.4f}x | {'yes' if exact else 'no'} |"
        )
    lines.extend(
        [
            "",
            "- oMLX's primary generation TPS convention is completion tokens divided by the producer interval; interval median uses N-1 token gaps as a cross-check.",
            "- This sweep selects an oMLX configuration. It does not yet compare oMLX against ds4 because their target quantizations differ.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    runs = {mode: load_run(getattr(args, mode), mode) for mode in MODES}
    tasks = validate_runs(runs)
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
    return 0 if summary["correctness_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
