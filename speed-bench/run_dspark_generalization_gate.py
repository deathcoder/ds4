#!/usr/bin/env python3
"""Run the frozen math/chat DSpark confidence-scheduler generalization gate."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics

import run_dspark_issue468_comparison as common


MODES = ("baseline", "fixed_k5", "threshold_0455")
THRESHOLDS = {
    "baseline": None,
    "fixed_k5": None,
    "threshold_0455": "0.455",
}
MODE_LABELS = {
    "baseline": "Ordinary baseline",
    "fixed_k5": "Fixed K=5",
    "threshold_0455": "Threshold 0.455",
}
WARMUP_PERIODS = 1
DOMAIN_MIN_WINS = 4
OVERALL_MIN_GEOMETRIC_MEAN = 1.03
TASK_MIN_RATIO = 0.90


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen non-code baseline/fixed/scheduled DSpark gate."
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
        default=root / "speed-bench/dspark-generalization",
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=5.0)
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

    # Attributes consumed by the shared command and execution helpers.
    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = False
    args.stats_pass = False
    args.acceptance_audit = False
    args.acceptance_trace = False
    args.confidence_threshold = None
    args.pairs = 1
    args.warmups = WARMUP_PERIODS
    return args, root


def load_corpus(args, root):
    samples_path = args.corpus_dir / "samples.jsonl"
    provenance_path = args.corpus_dir / "provenance.json"
    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("samples", samples_path),
        ("provenance", provenance_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = common.git_output(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass "
            f"--allow-dirty:\n{dirty}"
        )

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        samples_data = samples_path.read_bytes()
        records = [
            json.loads(line) for line in samples_data.splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid generalization corpus: {exc}") from exc
    if (
        len(samples_data) != provenance.get("samples_file_bytes") or
        common.sha256(samples_data) != provenance.get("samples_file_sha256")
    ):
        raise SystemExit("generalization samples-file provenance mismatch")
    expected = provenance.get("samples", [])
    if len(records) != 12 or len(records) != provenance.get("stored_rows"):
        raise SystemExit("generalization corpus must contain exactly 12 rows")
    if len(expected) != len(records):
        raise SystemExit("generalization provenance entry count mismatch")

    labels = set()
    for record, item in zip(records, expected):
        identity = (
            "label", "domain", "dataset", "source_index", "source_turn_index"
        )
        if any(record.get(key) != item.get(key) for key in identity):
            raise SystemExit("generalization sample identity mismatch")
        label = record["label"]
        if label in labels:
            raise SystemExit(f"duplicate generalization label: {label}")
        labels.add(label)
        turns = record.get("turns")
        if not (
            isinstance(turns, list) and len(turns) == 1 and
            isinstance(turns[0], str)
        ):
            raise SystemExit(f"invalid prompt turns for {label}")
        prompt_data = turns[0].encode("utf-8")
        if (
            len(prompt_data) != item.get("prompt_bytes") or
            common.sha256(prompt_data) != item.get("prompt_sha256")
        ):
            raise SystemExit(f"prompt provenance mismatch for {label}")

    domains = [record["domain"] for record in records]
    if domains.count("math") != 6 or domains.count("chat") != 6:
        raise SystemExit("generalization corpus must contain six math and six chat rows")
    datasets = {record["dataset"] for record in records}
    expected_datasets = {"gsm8k", "math500", "aime25", "alpaca", "mt-bench"}
    if datasets != expected_datasets:
        raise SystemExit("generalization corpus dataset set mismatch")
    if provenance.get("source_commit") != (
        "005e03b81cec38b7da6399833d609ee89a2587f2"
    ):
        raise SystemExit("generalization corpus source commit mismatch")
    dataset_metadata = {
        item.get("dataset"): item for item in provenance.get("datasets", [])
    }
    if set(dataset_metadata) != expected_datasets:
        raise SystemExit("generalization dataset provenance set mismatch")
    for dataset in expected_datasets:
        selected = [
            record["source_index"] for record in records
            if record["dataset"] == dataset
        ]
        metadata = dataset_metadata[dataset]
        if selected != metadata.get("selected_indices_zero_based"):
            raise SystemExit(f"generalization selection mismatch for {dataset}")
        evaluator_rows = metadata.get("evaluator_rows", 0)
        if (
            evaluator_rows <= 0 or
            any(index >= evaluator_rows for index in selected)
        ):
            raise SystemExit(f"selection exceeds evaluator rows for {dataset}")
    return records, provenance, samples_path, provenance_path


def prompt_paths(run_dir, records):
    return {
        record["label"]: run_dir / "prompts" / f"{record['label']}.txt"
        for record in records
    }


def materialize_prompts(prompts, records):
    next(iter(prompts.values())).parent.mkdir(parents=True, exist_ok=False)
    for record in records:
        prompts[record["label"]].write_text(
            record["turns"][0], encoding="utf-8"
        )


def rotated_modes(offset):
    offset %= len(MODES)
    return MODES[offset:] + MODES[:offset]


def target_mode(mode):
    return "baseline" if mode == "baseline" else "runtime"


def command_text(args, prompt, mode):
    return common.command_text(
        args, prompt, target_mode(mode),
        confidence_threshold=THRESHOLDS[mode],
    )


def execute(args, root, run_dir, name, record, prompt, mode, reference):
    shared_mode = target_mode(mode)
    row, output = common.execute(
        args, root, run_dir, f"{name}-{mode}", record["label"], prompt,
        shared_mode, reference,
        confidence_threshold=THRESHOLDS[mode],
    )
    stderr_data = (run_dir / row["stderr_file"]).read_bytes()
    if b"ds4: DSpark confidence scheduler " in stderr_data:
        raise RuntimeError(
            f"scheduler diagnostics leaked into {run_dir / row['stderr_file']}"
        )
    row.pop("prompt", None)
    row.update(
        task=record["label"],
        domain=record["domain"],
        dataset=record["dataset"],
        source_index=record["source_index"],
        mode=mode,
        threshold=THRESHOLDS[mode] or "",
    )
    return row, output


def ratio_metrics(values):
    return {
        "median": statistics.median(values),
        "geometric_mean": statistics.geometric_mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "values": values,
        "faster": sum(value > 1.0 for value in values),
        "equal": sum(value == 1.0 for value in values),
        "slower": sum(value < 1.0 for value in values),
    }


def summarize(rows, records):
    tasks = {}
    buckets = {
        domain: {
            "fixed_vs_baseline": [],
            "scheduled_vs_baseline": [],
            "scheduled_vs_fixed": [],
        }
        for domain in ("math", "chat", "overall")
    }
    for record in records:
        selected = {
            row["mode"]: row for row in rows if row["task"] == record["label"]
        }
        if set(selected) != set(MODES):
            raise RuntimeError(f"incomplete measured modes for {record['label']}")
        baseline = selected["baseline"]["generation_tps"]
        fixed = selected["fixed_k5"]["generation_tps"]
        scheduled = selected["threshold_0455"]["generation_tps"]
        ratios = {
            "fixed_vs_baseline": fixed / baseline,
            "scheduled_vs_baseline": scheduled / baseline,
            "scheduled_vs_fixed": scheduled / fixed,
        }
        for bucket in (record["domain"], "overall"):
            for key, value in ratios.items():
                buckets[bucket][key].append(value)
        tasks[record["label"]] = {
            "domain": record["domain"],
            "dataset": record["dataset"],
            "source_index": record["source_index"],
            "order": selected["baseline"]["order"],
            "baseline_generation_tps": baseline,
            "fixed_generation_tps": fixed,
            "scheduled_generation_tps": scheduled,
            **ratios,
        }
    aggregates = {
        domain: {
            key: ratio_metrics(values) for key, values in metrics.items()
        }
        for domain, metrics in buckets.items()
    }
    promotion_checks = {
        "math_median_above_one": (
            aggregates["math"]["scheduled_vs_fixed"]["median"] > 1.0
        ),
        "chat_median_above_one": (
            aggregates["chat"]["scheduled_vs_fixed"]["median"] > 1.0
        ),
        "math_wins_at_least_four": (
            aggregates["math"]["scheduled_vs_fixed"]["faster"] >=
            DOMAIN_MIN_WINS
        ),
        "chat_wins_at_least_four": (
            aggregates["chat"]["scheduled_vs_fixed"]["faster"] >=
            DOMAIN_MIN_WINS
        ),
        "overall_geometric_mean_at_least_1_03": (
            aggregates["overall"]["scheduled_vs_fixed"]["geometric_mean"] >=
            OVERALL_MIN_GEOMETRIC_MEAN
        ),
        "no_task_below_0_90": (
            aggregates["overall"]["scheduled_vs_fixed"]["minimum"] >=
            TASK_MIN_RATIO
        ),
    }
    return {
        "sample_count": len(records),
        "modes": MODES,
        "threshold": "0.455",
        "tasks": tasks,
        "aggregates": aggregates,
        "promotion_gate": {
            "passed": all(promotion_checks.values()),
            "checks": promotion_checks,
            "criteria": {
                "domain_minimum_wins": DOMAIN_MIN_WINS,
                "overall_minimum_geometric_mean": OVERALL_MIN_GEOMETRIC_MEAN,
                "task_minimum_ratio": TASK_MIN_RATIO,
            },
        },
    }


def render_report(summary):
    lines = [
        "# DSpark Math/Chat Scheduler Generalization Gate",
        "",
        "All runs are uninstrumented and use exact target verification.",
        "Every fixed and scheduled DSpark output matched its ordinary baseline byte-for-byte.",
        "Threshold `0.455`, source rows, prompts, and ordering were frozen before model execution.",
        "",
        "## Domain Summary",
        "",
        "| domain | tasks | fixed/baseline | scheduled/baseline | "
        "scheduled/fixed | scheduled faster than fixed |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for domain in ("math", "chat", "overall"):
        aggregates = summary["aggregates"][domain]
        count = 12 if domain == "overall" else 6
        fixed = aggregates["fixed_vs_baseline"]
        scheduled = aggregates["scheduled_vs_baseline"]
        improvement = aggregates["scheduled_vs_fixed"]
        lines.append(
            f"| {domain} | {count} | {fixed['median']:.4f}x | "
            f"{scheduled['median']:.4f}x | {improvement['median']:.4f}x | "
            f"{improvement['faster']}/{count} |"
        )
    lines.extend([
        "",
        "## Tasks",
        "",
        "| task | domain | dataset | order | baseline | fixed K=5 | "
        "threshold 0.455 | scheduled/fixed | scheduled/baseline |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|",
    ])
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['domain']} | {item['dataset']} | "
            f"{item['order']} | {item['baseline_generation_tps']:.2f} t/s | "
            f"{item['fixed_generation_tps']:.2f} t/s | "
            f"{item['scheduled_generation_tps']:.2f} t/s | "
            f"{item['scheduled_vs_fixed']:.4f}x | "
            f"{item['scheduled_vs_baseline']:.4f}x |"
        )
    overall = summary["aggregates"]["overall"]
    improvement = overall["scheduled_vs_fixed"]
    scheduled = overall["scheduled_vs_baseline"]
    promotion = summary["promotion_gate"]
    lines.extend([
        "",
        "## Promotion Gate",
        "",
        f"**{'PASS' if promotion['passed'] else 'FAIL'}**",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}"
            for name, passed in promotion["checks"].items()
        ],
        "",
        f"- Overall scheduled/fixed geometric mean: "
        f"{improvement['geometric_mean']:.4f}x; range "
        f"{improvement['minimum']:.4f}x-{improvement['maximum']:.4f}x.",
        f"- Scheduled DSpark faster/equal/slower than ordinary baseline: "
        f"{scheduled['faster']}/{scheduled['equal']}/{scheduled['slower']}.",
        "- One measured run per mode and task is used; domain and overall "
        "direction matter more than any single task.",
        "- The three mode positions are balanced exactly across the 12 tasks; "
        "three global warmups are excluded.",
        "- No DSpark stats, acceptance audit, trace, diagnostics, or profiler is enabled.",
        "- MT-Bench contributes only its self-contained first user turn; this "
        "is a scheduler generalization gate, not a matched benchmark-score "
        "reproduction.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "corpus_dir"):
        setattr(args, name, getattr(args, name).resolve())
    records, provenance, samples_path, provenance_path = load_corpus(args, root)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/dspark-generalization-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = prompt_paths(run_dir, records)
    for task_index, record in enumerate(records):
        order = rotated_modes(task_index)
        print(f"{record['label']} order: {' -> '.join(order)}")
        for mode in order:
            print(
                f"  {MODE_LABELS[mode]}: "
                f"{command_text(args, prompts[record['label']], mode)}"
            )
    warmups = WARMUP_PERIODS * len(MODES)
    measured = len(records) * len(MODES)
    print(
        f"Generalization gate: {warmups + measured} processes; "
        f"{warmups} warmup and {measured} measured."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    materialize_prompts(prompts, records)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": common.git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": common.git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "experiment": "dspark_math_chat_scheduler_generalization",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("DS4_")
        },
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "modes": MODES,
            "thresholds": THRESHOLDS,
            "warmup_processes": warmups,
            "measured_processes": measured,
            "instrumented": False,
        },
        "binary": common.file_metadata(args.binary),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "corpus": {
            "samples": common.file_metadata(samples_path),
            "provenance": common.file_metadata(provenance_path),
            "source_repository": provenance["source_repository"],
            "source_commit": provenance["source_commit"],
        },
        "commands": {
            record["label"]: {
                mode: command_text(args, prompts[record["label"]], mode)
                for mode in MODES
            }
            for record in records
        },
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    warmup_rows = []
    warmup_reference = None
    for position, mode in enumerate(MODES, start=1):
        row, warmup_reference = execute(
            args, root, run_dir, f"warmup-{position:02d}", records[0],
            prompts[records[0]["label"]], mode, warmup_reference,
        )
        row.update(warmup=1, order_position=position)
        warmup_rows.append(row)
        common.cooldown(args.cooldown)

    measured_rows = []
    sequence = 0
    for task_index, record in enumerate(records):
        order = rotated_modes(task_index)
        order_text = "-".join(order)
        reference = None
        for position, mode in enumerate(order, start=1):
            sequence += 1
            row, reference = execute(
                args, root, run_dir, f"measured-{sequence:02d}", record,
                prompts[record["label"]], mode, reference,
            )
            row.update(
                sequence=sequence,
                order=order_text,
                order_position=position,
            )
            measured_rows.append(row)
            common.cooldown(args.cooldown)

    fields = (
        "sequence", "task", "domain", "dataset", "source_index", "order",
        "order_position", "mode", "threshold", "prefill_tps",
        "generation_tps", "wall_seconds", "stdout_sha256", "stdout_file",
        "stderr_file",
    )
    write_csv(run_dir / "throughput.csv", measured_rows, fields)
    write_csv(
        run_dir / "warmups.csv", warmup_rows,
        ("warmup",) + fields[1:5] + ("order_position",) + fields[7:],
    )
    summary = summarize(measured_rows, records)
    summary["corpus"] = {
        "source_repository": provenance["source_repository"],
        "source_commit": provenance["source_commit"],
        "samples_path": str(samples_path),
    }
    report = render_report(summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["final_snapshot"] = common.machine_snapshot(root)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + report.rstrip())
    print(f"Raw throughput: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
