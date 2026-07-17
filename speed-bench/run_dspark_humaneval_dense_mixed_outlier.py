#!/usr/bin/env python3
"""Adjudicate the sole fused-gather HumanEval confirmation outlier."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics

import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_dense_mixed_direct as confirmation
import run_dspark_issue468_comparison as common


TASK = "humaneval_121"
MEASURED_PAIRS = 6
WARMUP_PAIRS = 2
MIN_MEDIAN_RATIO = 1.02
MIN_GEOMEAN_RATIO = 1.02
MIN_WINS = 4
ORIGINAL_TASK_FLOOR = confirmation.MIN_TASK_RATIO
MIN_PAIRS_ABOVE_FLOOR = 5


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the sole sub-0.95 HumanEval fused-gather task from the "
            "frozen 32-task confirmation."
        )
    )
    parser.add_argument("--confirmation-reference", type=Path, required=True)
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
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-idle", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_idle:
        parser.error("refusing to benchmark without --confirm-idle")
    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    return args, root


def load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label} {path}: {exc}") from exc


def load_confirmation_reference(args):
    summary_path = args.confirmation_reference.resolve()
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "throughput.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("throughput CSV", csv_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing confirmation {label}: {path}")

    summary = load_json(summary_path, "confirmation summary")
    metadata = load_json(metadata_path, "confirmation metadata")
    if metadata.get("experiment") != "dspark_humaneval_dense_mixed_direct":
        raise SystemExit("confirmation reference has the wrong experiment kind")
    if metadata.get("git_status_tracked"):
        raise SystemExit("confirmation reference was produced from a dirty tree")
    if summary.get("sample_count") != confirmation.SAMPLE_COUNT:
        raise SystemExit("confirmation reference is not the frozen 32-task gate")
    if summary.get("threshold") != confirmation.THRESHOLD:
        raise SystemExit("confirmation reference threshold mismatch")
    gate = summary.get("promotion_gate", {})
    if gate.get("pass") is not False:
        raise SystemExit("confirmation reference is not the failed gate")
    expected_gate = {
        "minimum_geometric_mean": confirmation.MIN_GEOMEAN,
        "minimum_wins": confirmation.MIN_WINS,
        "minimum_task_ratio": confirmation.MIN_TASK_RATIO,
        "low_acceptance_maximum": confirmation.LOW_ACCEPTANCE_MAX,
        "minimum_low_acceptance_geometric_mean":
            confirmation.MIN_LOW_ACCEPTANCE_GEOMEAN,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise SystemExit(f"confirmation gate {key} mismatch")
    if summary.get("paired_ratio_geometric_mean", 0) < confirmation.MIN_GEOMEAN:
        raise SystemExit("confirmation failed more than the per-task floor")
    if summary.get("direct_faster_tasks", 0) < confirmation.MIN_WINS:
        raise SystemExit("confirmation failed more than the per-task floor")
    if (summary.get("low_acceptance_geometric_mean", 0) <
            confirmation.MIN_LOW_ACCEPTANCE_GEOMEAN):
        raise SystemExit("confirmation failed more than the per-task floor")

    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read confirmation CSV: {exc}") from exc
    rows_by_task = {}
    for row in rows:
        rows_by_task.setdefault(row.get("prompt"), {})[row.get("mode")] = row
    samples = summary.get("samples", {})
    if len(samples) != confirmation.SAMPLE_COUNT or set(rows_by_task) != set(samples):
        raise SystemExit("confirmation task set mismatch")
    below_floor = []
    for task, sample in samples.items():
        by_mode = rows_by_task[task]
        if set(by_mode) != set(confirmation.MODES):
            raise SystemExit(f"confirmation has an incomplete {task} pair")
        hashes = {row["stdout_sha256"] for row in by_mode.values()}
        if len(hashes) != 1:
            raise SystemExit(f"confirmation output mismatch for {task}")
        try:
            ratio = (
                float(by_mode["fused_gather"]["generation_tps"]) /
                float(by_mode["gathered"]["generation_tps"])
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise SystemExit(f"invalid confirmation timing for {task}") from exc
        if abs(ratio - sample.get("paired_ratio", -1)) > 1e-12:
            raise SystemExit(f"confirmation ratio mismatch for {task}")
        if ratio < confirmation.MIN_TASK_RATIO:
            below_floor.append(task)
    if below_floor != [TASK]:
        raise SystemExit(
            f"expected {TASK} as the sole confirmation outlier, got {below_floor}"
        )

    throughput_path = Path(summary.get("throughput_reference", "")).resolve()
    if not throughput_path.is_file():
        raise SystemExit("confirmation throughput reference is missing")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "summary": summary,
        "throughput_path": throughput_path,
    }


def measured_order(pair_number):
    return confirmation.mode_order(pair_number)


def summarize(rows):
    pairs = []
    ratios = []
    gathered_values = []
    fused_values = []
    wins = 0
    above_floor = 0
    for pair_number in range(1, MEASURED_PAIRS + 1):
        selected = {
            row["mode"]: row for row in rows
            if row["pair_number"] == pair_number
        }
        if set(selected) != set(confirmation.MODES):
            raise RuntimeError(f"incomplete adjudication pair {pair_number}")
        gathered = selected["gathered"]["generation_tps"]
        fused = selected["fused_gather"]["generation_tps"]
        ratio = fused / gathered
        gathered_values.append(gathered)
        fused_values.append(fused)
        ratios.append(ratio)
        wins += ratio > 1.0
        above_floor += ratio >= ORIGINAL_TASK_FLOOR
        pairs.append({
            "pair": pair_number,
            "order": selected["gathered"]["pair_order"],
            "gathered_generation_tps": gathered,
            "fused_generation_tps": fused,
            "paired_ratio": ratio,
            "delta_percent": (ratio - 1.0) * 100.0,
        })
    median_ratio = statistics.median(ratios)
    geomean_ratio = statistics.geometric_mean(ratios)
    gate = (
        median_ratio >= MIN_MEDIAN_RATIO and
        geomean_ratio >= MIN_GEOMEAN_RATIO and
        wins >= MIN_WINS and
        above_floor >= MIN_PAIRS_ABOVE_FLOOR
    )
    return {
        "task": TASK,
        "measured_pairs": MEASURED_PAIRS,
        "pairs": pairs,
        "gathered_generation_tps_median": statistics.median(gathered_values),
        "fused_generation_tps_median": statistics.median(fused_values),
        "ratio_of_medians": (
            statistics.median(fused_values) /
            statistics.median(gathered_values)
        ),
        "paired_ratio_median": median_ratio,
        "paired_ratio_geometric_mean": geomean_ratio,
        "paired_ratio_minimum": min(ratios),
        "paired_ratio_maximum": max(ratios),
        "fused_wins": wins,
        "pairs_above_original_floor": above_floor,
        "adjudication_gate": {
            "minimum_median_ratio": MIN_MEDIAN_RATIO,
            "minimum_geometric_mean_ratio": MIN_GEOMEAN_RATIO,
            "minimum_wins": MIN_WINS,
            "original_task_floor": ORIGINAL_TASK_FLOOR,
            "minimum_pairs_above_original_floor": MIN_PAIRS_ABOVE_FLOOR,
            "pass": gate,
        },
    }


def render_report(summary):
    gate = summary["adjudication_gate"]
    lines = [
        "# DSpark Dense-Mixed Fused-Gather Outlier Adjudication",
        "",
        f"The frozen 32-task gate remains a formal failure. This run repeats its "
        f"sole sub-{ORIGINAL_TASK_FLOOR:.2f} task with balanced order.",
        "Every gathered and fused-gather output matched the frozen exact artifact "
        "byte-for-byte.",
        "Generation t/s excludes process startup; paired ratios are authoritative.",
        "",
        "| task | pairs | gathered median | fused-gather median | ratio of medians "
        "| median paired | geometric mean | wins |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['task']} | {summary['measured_pairs']} | "
        f"{summary['gathered_generation_tps_median']:.2f} t/s | "
        f"{summary['fused_generation_tps_median']:.2f} t/s | "
        f"{summary['ratio_of_medians']:.4f}x | "
        f"{summary['paired_ratio_median']:.4f}x | "
        f"{summary['paired_ratio_geometric_mean']:.4f}x | "
        f"{summary['fused_wins']}/{summary['measured_pairs']} |",
        "",
        "## Pairs",
        "",
        "| pair | order | gathered | fused gather | ratio | delta |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for item in summary["pairs"]:
        lines.append(
            f"| {item['pair']} | {item['order']} | "
            f"{item['gathered_generation_tps']:.2f} t/s | "
            f"{item['fused_generation_tps']:.2f} t/s | "
            f"{item['paired_ratio']:.4f}x | {item['delta_percent']:+.1f}% |"
        )
    lines.extend([
        "",
        "## Adjudication Gate",
        "",
        f"**{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        f"- Require median paired ratio at least "
        f"`{gate['minimum_median_ratio']:.2f}x`.",
        f"- Require geometric mean at least "
        f"`{gate['minimum_geometric_mean_ratio']:.2f}x`.",
        f"- Require at least `{gate['minimum_wins']}/{MEASURED_PAIRS}` wins.",
        f"- Require at least "
        f"`{gate['minimum_pairs_above_original_floor']}/{MEASURED_PAIRS}` pairs "
        f"at or above the original `{gate['original_task_floor']:.2f}x` floor.",
        "",
        f"- Paired-ratio range: {summary['paired_ratio_minimum']:.4f}x-"
        f"{summary['paired_ratio_maximum']:.4f}x.",
        f"- Pairs at or above the original floor: "
        f"{summary['pairs_above_original_floor']}/{MEASURED_PAIRS}.",
        f"- {WARMUP_PAIRS} balanced warmup pairs are excluded.",
        "- No DSpark stats, trace, diagnostics, profiler, or fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args, root = parse_args()
    for name in (
        "confirmation_reference", "binary", "model", "dspark_model",
        "corpus_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("corpus", args.corpus_dir),
    ):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")
    dirty = common.git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n"
            + dirty
        )

    gate_reference = load_confirmation_reference(args)
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, confirmation.SAMPLE_COUNT, provenance["selection_policy"]
    )
    if selection != gate_reference["summary"].get("selection"):
        raise SystemExit("confirmation task selection mismatch")
    record = next((item for item in records if item["label"] == TASK), None)
    if record is None:
        raise SystemExit(f"missing adjudication task {TASK}")

    args.throughput_reference = gate_reference["throughput_path"]
    frozen = confirmation.load_reference(args, records, selection)
    expected = frozen["tasks"][TASK]["output_data"]

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-dense-mixed-outlier-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompt = run_dir / "prompts" / f"{TASK}.txt"
    print(f"Adjudication task: {TASK}")
    for pair_number in range(1, MEASURED_PAIRS + 1):
        order = measured_order(pair_number)
        print(f"measured pair {pair_number}: {' -> '.join(order)}")
    total = (WARMUP_PAIRS + MEASURED_PAIRS) * len(confirmation.MODES)
    print(
        f"Outlier adjudication: {total} uninstrumented processes; "
        f"{WARMUP_PAIRS * 2} excluded warmups and {MEASURED_PAIRS * 2} measured."
    )
    print(f"  gathered: {confirmation.command_text(args, prompt, 'gathered')}")
    print(
        "  fused gather: "
        f"{confirmation.command_text(args, prompt, 'fused_gather')}"
    )
    if args.dry_run:
        print("Dry run only; no prompt materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts({TASK: prompt}, [record])
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": dirty,
        "experiment": "dspark_humaneval_dense_mixed_outlier",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "task": {"label": TASK, "source_index": record["source_index"]},
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "threshold": confirmation.THRESHOLD,
            "instrumented": False,
            "measured_pairs": MEASURED_PAIRS,
            "warmup_pairs": WARMUP_PAIRS,
            "balanced_order": True,
        },
        "gate": {
            "minimum_median_ratio": MIN_MEDIAN_RATIO,
            "minimum_geometric_mean_ratio": MIN_GEOMEAN_RATIO,
            "minimum_wins": MIN_WINS,
            "original_task_floor": ORIGINAL_TASK_FLOOR,
            "minimum_pairs_above_original_floor": MIN_PAIRS_ABOVE_FLOOR,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "confirmation_reference": {
            "summary": common.file_metadata(gate_reference["summary_path"]),
            "metadata": common.file_metadata(gate_reference["metadata_path"]),
            "csv": common.file_metadata(gate_reference["csv_path"]),
        },
        "throughput_reference": common.file_metadata(
            gate_reference["throughput_path"]
        ),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    warmup_rows = []
    for pair_number in range(1, WARMUP_PAIRS + 1):
        warmup_rows.extend(confirmation.run_pair(
            args, root, run_dir, f"warmup-{pair_number:02d}", record, prompt,
            measured_order(pair_number), expected,
        ))

    measured_rows = []
    sequence = 0
    for pair_number in range(1, MEASURED_PAIRS + 1):
        pair_rows = confirmation.run_pair(
            args, root, run_dir, f"measured-{pair_number:02d}", record, prompt,
            measured_order(pair_number), expected,
        )
        for row in pair_rows:
            sequence += 1
            row["sequence"] = sequence
            row["pair_number"] = pair_number
            measured_rows.append(row)

    fields = (
        "sequence", "pair_number", "prompt", "source_index", "pair_order",
        "pair_position", "mode", "prefill_tps", "generation_tps",
        "wall_seconds", "stdout_sha256", "stdout_file", "stderr_file",
    )
    common.write_csv(run_dir / "throughput.csv", measured_rows, fields)
    common.write_csv(
        run_dir / "warmups.csv", warmup_rows,
        tuple(field for field in fields if field not in {"sequence", "pair_number"}),
    )
    summary = summarize(measured_rows)
    summary["confirmation_reference"] = str(gate_reference["summary_path"])
    report = render_report(summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw throughput: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
