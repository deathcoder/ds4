#!/usr/bin/env python3
"""Profile exact DSpark costs on frozen low/high-acceptance HumanEval tasks."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re

import run_dspark_exact_layer_profile as layer_profile
import run_dspark_humaneval_acceptance as corpus


RUNTIME_STATS_RE = re.compile(
    rb"^ds4: DSpark runtime stats (.+)$", re.MULTILINE
)
STAT_FIELD_RE = re.compile(rb"([a-z0-9_]+)=([0-9]+(?:\.[0-9]+)?)")
DEFAULT_TASKS = ("humaneval_152", "humaneval_079")
REQUIRED_STATS = (
    "emitted",
    "target_evals",
    "target_eval_tokens",
    "target_evals_avoided",
    "multi_attempts",
    "avg_depth",
)


def parse_tasks(value):
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(tasks) < 2:
        raise argparse.ArgumentTypeError("tasks must contain at least two labels")
    if len(set(tasks)) != len(tasks):
        raise argparse.ArgumentTypeError("tasks must not contain duplicates")
    if any(not re.fullmatch(r"humaneval_[0-9]{3}", task) for task in tasks):
        raise argparse.ArgumentTypeError("tasks must use humaneval_NNN labels")
    return tasks


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Profile promoted exact-runtime layer costs on representative "
            "HumanEval tasks."
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
    parser.add_argument("--tasks", type=parse_tasks, default=DEFAULT_TASKS)
    parser.add_argument("--layers", type=layer_profile.parse_layers)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=5.0)
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
        parser.error("refusing to profile without --confirm-ready")
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


def load_throughput_reference(args, records):
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
    if metadata.get("experiment") != "deepspec_humaneval_paired_throughput":
        raise SystemExit("throughput reference has the wrong experiment kind")
    config = metadata.get("config", {})
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
        "nothink": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"throughput reference config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    validate_metadata_path(metadata, "binary", args.binary)
    validate_metadata_path(metadata, "base_model", args.model)
    validate_metadata_path(metadata, "dspark_model", args.dspark_model)

    record_map = {record["label"]: record for record in records}
    sample_map = summary.get("samples", {})
    selected = {}
    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read throughput reference CSV: {exc}") from exc
    for task in args.tasks:
        if task not in record_map or task not in sample_map:
            raise SystemExit(f"task {task} is absent from the frozen throughput workload")
        task_rows = [row for row in rows if row.get("prompt") == task]
        by_mode = {row.get("mode"): row for row in task_rows}
        if set(by_mode) != {"baseline", "runtime"} or len(task_rows) != 2:
            raise SystemExit(f"throughput reference has incomplete pair for {task}")
        if by_mode["baseline"].get("stdout_sha256") != by_mode["runtime"].get(
                "stdout_sha256"):
            raise SystemExit(f"throughput reference output mismatch for {task}")
        runtime_output = run_dir / by_mode["runtime"]["stdout_file"]
        prompt_path = Path(metadata.get("prompts", {}).get(task, {}).get("path", ""))
        if not runtime_output.is_file() or not prompt_path.is_file():
            raise SystemExit(f"throughput reference artifacts are missing for {task}")
        prompt_data = record_map[task]["turns"][0].encode("utf-8")
        if prompt_path.read_bytes() != prompt_data:
            raise SystemExit(f"throughput reference prompt drift for {task}")
        selected[task] = {
            "record": record_map[task],
            "sample": sample_map[task],
            "prior_runtime_output": runtime_output,
            "prompt_data": prompt_data,
        }
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "metadata": metadata,
        "tasks": selected,
    }


def profile_args(args, prompt_path, layers):
    return argparse.Namespace(
        binary=args.binary,
        model=args.model,
        dspark_model=args.dspark_model,
        prompt_file=prompt_path,
        ctx=args.ctx,
        tokens=args.tokens,
        layers=layers,
        cooldown=args.cooldown,
        nothink=True,
    )


def parse_runtime_stats(path):
    data = path.read_bytes()
    matches = RUNTIME_STATS_RE.findall(data)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one runtime stats record in {path}, found {len(matches)}"
        )
    fields = {
        key.decode("ascii"): value.decode("ascii")
        for key, value in STAT_FIELD_RE.findall(matches[0])
    }
    missing = [key for key in REQUIRED_STATS if key not in fields]
    if missing:
        raise RuntimeError(
            f"runtime stats in {path} omit: {', '.join(missing)}"
        )
    result = {key: int(fields[key]) for key in REQUIRED_STATS[:-1]}
    result["avg_depth"] = float(fields["avg_depth"])
    if result["emitted"] <= 0 or result["target_evals"] <= 0:
        raise RuntimeError(f"runtime stats in {path} have invalid counts")
    return result


def summarize_task(task, context, stats, records):
    layer_summary = layer_profile.summarize(records)
    emitted = stats["emitted"]
    target_evals = stats["target_evals"]
    target_eval_tokens = stats["target_eval_tokens"]
    row_counts = {
        item["profiled_rows"] for item in layer_summary["layers"].values()
    }
    batch_counts = {
        item["profiled_batches"] for item in layer_summary["layers"].values()
    }
    if len(row_counts) != 1 or len(batch_counts) != 1:
        raise RuntimeError(f"{task} layers used different proposal schedules")
    profiled_rows = next(iter(row_counts))
    profiled_batches = next(iter(batch_counts))
    for item in layer_summary["layers"].values():
        stage_totals = {
            stage: values["total_ms"] / emitted
            for stage, values in item["stages"].items()
        }
        item["synchronized_ms_per_emitted"] = sum(stage_totals.values())
        item["synchronized_stage_ms_per_emitted"] = stage_totals
    sample = context["sample"]
    return {
        "source_index": context["record"]["source_index"],
        "acceptance_verify_rate": sample["acceptance_verify_rate"],
        "prior_paired_speed_ratio": sample["paired_ratio"],
        "emitted": emitted,
        "target_evals": target_evals,
        "target_eval_tokens": target_eval_tokens,
        "target_evals_avoided": stats["target_evals_avoided"],
        "multi_attempts": stats["multi_attempts"],
        "average_accepted_depth": stats["avg_depth"],
        "target_evals_per_emitted": target_evals / emitted,
        "target_positions_per_eval": target_eval_tokens / target_evals,
        "profiled_batches": profiled_batches,
        "profiled_rows": profiled_rows,
        "profiled_rows_per_emitted": profiled_rows / emitted,
        "profiled_target_position_coverage": (
            profiled_rows / target_eval_tokens if target_eval_tokens else None
        ),
        "layers": layer_summary["layers"],
    }


def summarize(tasks):
    ordered = sorted(
        tasks,
        key=lambda pair: pair[1]["acceptance_verify_rate"],
    )
    low_name, low = ordered[0]
    high_name, high = ordered[-1]
    comparisons = {}
    for layer in low["layers"]:
        low_layer = low["layers"][layer]
        high_layer = high["layers"][layer]
        comparisons[layer] = {
            "low_high_ms_per_row_ratio": (
                low_layer["typical_total_ms_per_row"] /
                high_layer["typical_total_ms_per_row"]
            ),
            "low_high_rows_per_emitted_ratio": (
                low["profiled_rows_per_emitted"] /
                high["profiled_rows_per_emitted"]
            ),
            "low_high_ms_per_emitted_ratio": (
                low_layer["synchronized_ms_per_emitted"] /
                high_layer["synchronized_ms_per_emitted"]
            ),
        }
    return {
        "tasks": dict(tasks),
        "comparison": {
            "low_acceptance_task": low_name,
            "high_acceptance_task": high_name,
            "layers": comparisons,
        },
    }


def render_report(summary):
    lines = [
        "# DSpark HumanEval Exact Runtime Attribution",
        "",
        "Synchronized diagnostic only. Stage boundaries change scheduling; do not use these values as throughput measurements.",
        "Every reference and profiled output matched the prior uninstrumented HumanEval runtime artifact byte-for-byte.",
        "Per-row medians isolate component cost; per-emitted totals expose acceptance-driven invocation amplification.",
        "",
        "## Task Context",
        "",
        "| task | acceptance | prior speed ratio | emitted | target evals/emitted | positions/eval | profiled rows/emitted | profile coverage |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['acceptance_verify_rate']:.3f} | "
            f"{item['prior_paired_speed_ratio']:.4f}x | {item['emitted']} | "
            f"{item['target_evals_per_emitted']:.4f} | "
            f"{item['target_positions_per_eval']:.3f} | "
            f"{item['profiled_rows_per_emitted']:.3f} | "
            f"{item['profiled_target_position_coverage']:.1%} |"
        )
    lines.extend([
        "",
        "## Layer Components",
        "",
        "| task | layer | batches | rows | total ms/row | attention prep | serial tail | exact FFN | synchronized ms/emitted |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task, task_item in summary["tasks"].items():
        for layer, item in task_item["layers"].items():
            lines.append(
                f"| {task} | {layer} | {item['profiled_batches']} | "
                f"{item['profiled_rows']} | "
                f"{item['typical_total_ms_per_row']:.3f} | "
                f"{item['typical_attention_pre_ms_per_row']:.3f} | "
                f"{item['typical_attention_tail_ms_per_row']:.3f} | "
                f"{item['typical_ffn_ms_per_row']:.3f} | "
                f"{item['synchronized_ms_per_emitted']:.3f} |"
            )
    comparison = summary["comparison"]
    lines.extend([
        "",
        "## Low/High Acceptance Ratios",
        "",
        f"Low acceptance: `{comparison['low_acceptance_task']}`; high acceptance: `{comparison['high_acceptance_task']}`.",
        "",
        "| layer | cost/row | rows/emitted | cost/emitted |",
        "|---:|---:|---:|---:|",
    ])
    for layer, item in comparison["layers"].items():
        lines.append(
            f"| {layer} | {item['low_high_ms_per_row_ratio']:.3f}x | "
            f"{item['low_high_rows_per_emitted_ratio']:.3f}x | "
            f"{item['low_high_ms_per_emitted_ratio']:.3f}x |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "throughput_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    reference = load_throughput_reference(args, all_records)
    layer_count = layer_profile.inspect_layer_count(args, root)
    if args.layers is None:
        args.layers = tuple(dict.fromkeys((0, (layer_count - 1) // 2, layer_count - 1)))
    invalid = [layer for layer in args.layers if layer >= layer_count]
    if invalid:
        raise SystemExit(
            f"profile layers outside model range 0..{layer_count - 1}: "
            + ", ".join(str(layer) for layer in invalid)
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_dir or root / "speed-bench/local-runs" /
               f"humaneval-exact-profile-{stamp}").resolve()
    for task in args.tasks:
        prompt_path = run_dir / task / "prompt.txt"
        task_args = profile_args(args, prompt_path, args.layers)
        print(f"{task} reference: {layer_profile.command_text(task_args, runtime_stats=True)}")
        for layer in args.layers:
            print(f"{task} layer {layer}: {layer_profile.command_text(task_args, layer)}")
    print(
        "Every output must match the prior uninstrumented HumanEval runtime "
        "artifact byte-for-byte."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": layer_profile.run_capture(["git", "rev-parse", "HEAD"], root),
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "tasks": args.tasks,
            "layers": args.layers,
            "cooldown": args.cooldown,
            "model_layer_count": layer_count,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "synchronized_profile": True,
            "reference_stats_only": True,
            "components": layer_profile.EXACT_STAGES,
        },
        "throughput_reference": {
            "summary": str(reference["summary_path"]),
            "metadata": str(reference["metadata_path"]),
            "csv": str(reference["csv_path"]),
        },
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": layer_profile.cleared_env_keys(os.environ),
        "tasks": {},
        "commands": {},
    }
    for task in args.tasks:
        context = reference["tasks"][task]
        prompt_path = run_dir / task / "prompt.txt"
        task_args = profile_args(args, prompt_path, args.layers)
        metadata["tasks"][task] = {
            "source_index": context["record"]["source_index"],
            "acceptance_verify_rate":
                context["sample"]["acceptance_verify_rate"],
            "prior_paired_speed_ratio": context["sample"]["paired_ratio"],
            "prompt_sha256": layer_profile.sha256(context["prompt_data"]),
            "prior_runtime_output": str(context["prior_runtime_output"]),
        }
        metadata["commands"][task] = {
            "reference": layer_profile.command_text(
                task_args, runtime_stats=True
            ),
            "layers": {
                str(layer): layer_profile.command_text(task_args, layer)
                for layer in args.layers
            },
        }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    task_summaries = []
    stage_rows = []
    run_rows = []
    for task in args.tasks:
        context = reference["tasks"][task]
        task_dir = run_dir / task
        task_dir.mkdir()
        prompt_path = task_dir / "prompt.txt"
        prompt_path.write_bytes(context["prompt_data"])
        task_args = profile_args(args, prompt_path, args.layers)
        prior_output = context["prior_runtime_output"].read_bytes()
        output, _, run = layer_profile.execute(
            task_args, root, task_dir, "reference", None, prior_output,
            runtime_stats=True,
        )
        run_rows.append({"task": task, **run})
        stats = parse_runtime_stats(task_dir / "reference.stderr")
        layer_profile.cooldown(args.cooldown)
        records = []
        for layer in args.layers:
            _, selected, run = layer_profile.execute(
                task_args, root, task_dir, f"layer-{layer:02d}", layer, output
            )
            records.extend(selected)
            run_rows.append({"task": task, **run})
            stage_rows.extend({"task": task, **row} for row in selected)
            layer_profile.cooldown(args.cooldown)
        task_summaries.append((
            task, summarize_task(task, context, stats, records)
        ))

    summary = summarize(task_summaries)
    report = render_report(summary)
    write_csv(
        run_dir / "stages.csv",
        ("task", "part", "layer", "pos", "tokens", "stage", "ms"),
        stage_rows,
    )
    write_csv(
        run_dir / "runs.csv",
        ("task", "name", "layer", "wall_seconds", "stdout_sha256",
         "stdout_file", "stderr_file"),
        run_rows,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + report.rstrip())
    print(f"Raw stages: {run_dir / 'stages.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
