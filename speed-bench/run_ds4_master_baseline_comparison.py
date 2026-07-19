#!/usr/bin/env python3
"""Compare ordinary Metal decoding on current ds4 and pinned upstream main."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import time

import run_dspark_humaneval_acceptance as corpus
import run_dspark_issue468_comparison as common


SAMPLE_COUNT = 32
MODES = ("upstream_main", "current_branch")
EXPECTED_UPSTREAM_COMMIT = "80ebbc396aee40eedc1d829222f3362d10fa4c6c"
MIN_GEOMEAN = 1.01
MIN_WINS = 24
MIN_TASK_RATIO = 0.95


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare ordinary non-DSpark Metal decoding between a pinned "
            "upstream ds4 worktree and the current branch."
        )
    )
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--current-binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--upstream-source",
        type=Path,
        default=root.parent / "ds4-master-baseline",
    )
    parser.add_argument("--upstream-binary", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / (
            "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-"
            "SExpQ8-OutQ8-chat-v2-imatrix.gguf"
        ),
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
    if args.upstream_binary is None:
        args.upstream_binary = args.upstream_source / "ds4"
    args.nothink = True
    return args, root


def load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label} {path}: {exc}") from exc


def validate_metadata_path(metadata, key, expected):
    actual = metadata.get(key, {}).get("path")
    if actual is None or Path(actual).resolve() != expected.resolve():
        raise SystemExit(f"output reference {key} path mismatch")


def load_output_reference(args, records, selection):
    summary_path = args.output_reference
    run_dir = summary_path.parent
    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "throughput.csv"
    for label, path in (
        ("summary", summary_path),
        ("metadata", metadata_path),
        ("throughput CSV", csv_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing output reference {label}: {path}")

    summary = load_json(summary_path, "output reference summary")
    metadata = load_json(metadata_path, "output reference metadata")
    if metadata.get("experiment") != "dspark_humaneval_cumulative_throughput":
        raise SystemExit("output reference has the wrong experiment kind")
    if metadata.get("git_status_tracked"):
        raise SystemExit("output reference was produced from a dirty tree")
    if summary.get("sample_count") != SAMPLE_COUNT:
        raise SystemExit("output reference is not the frozen 32-task study")
    if summary.get("selection") != selection:
        raise SystemExit("output reference selection mismatch")
    config = metadata.get("config", {})
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
        "nothink": True,
        "instrumented": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"output reference config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    validate_metadata_path(metadata, "base_model", args.model)

    try:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    except OSError as exc:
        raise SystemExit(f"cannot read output reference CSV: {exc}") from exc
    rows_by_task = {}
    for row in rows:
        rows_by_task.setdefault(row.get("prompt"), {})[row.get("mode")] = row
    if len(rows) != SAMPLE_COUNT * 2:
        raise SystemExit("output reference measured-row count mismatch")

    tasks = {}
    for record in records:
        task = record["label"]
        by_mode = rows_by_task.get(task, {})
        if set(by_mode) != {"baseline", "runtime"}:
            raise SystemExit(f"output reference has incomplete task {task}")
        baseline = by_mode["baseline"]
        runtime = by_mode["runtime"]
        if baseline["stdout_sha256"] != runtime["stdout_sha256"]:
            raise SystemExit(f"output reference differs within task {task}")
        output_path = run_dir / baseline["stdout_file"]
        if not output_path.is_file():
            raise SystemExit(f"missing output reference for {task}")
        output_data = output_path.read_bytes()
        if common.sha256(output_data) != baseline["stdout_sha256"]:
            raise SystemExit(f"output reference hash mismatch for {task}")
        tasks[task] = output_data
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "csv_path": csv_path,
        "tasks": tasks,
    }


def mode_order(position):
    return MODES if position % 2 == 1 else tuple(reversed(MODES))


def warmup_schedule(records):
    return (
        (records[0], MODES),
        (records[-1], tuple(reversed(MODES))),
    )


def binary_for_mode(args, mode):
    if mode == "upstream_main":
        return args.upstream_binary
    if mode == "current_branch":
        return args.current_binary
    raise ValueError(f"unknown baseline comparison mode: {mode}")


def command(args, prompt, mode):
    return [
        str(binary_for_mode(args, mode)),
        "--backend", "metal",
        "--model", str(args.model),
        "--prompt-file", str(prompt),
        "--ctx", str(args.ctx),
        "--nothink",
        "--temp", "0",
        "--seed", "1",
        "-n", str(args.tokens),
    ]


def mode_env():
    return common.benchmark_env("baseline", False)


def command_text(args, prompt, mode):
    return shlex.join(command(args, prompt, mode))


def execute(args, root, run_dir, label, record, prompt, mode, expected):
    stdout_path = run_dir / f"{label}.{record['label']}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{record['label']}.{mode}.stderr"
    print(
        f"[{label}/{record['label']}] {mode}: "
        f"{command_text(args, prompt, mode)}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            command(args, prompt, mode),
            cwd=root,
            env=mode_env(),
            stdout=out,
            stderr=err,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} run failed with exit {completed.returncode}; "
            f"see {stderr_path}"
        )
    output = stdout_path.read_bytes()
    if output != expected:
        raise RuntimeError(
            f"{mode} output differs from frozen output; see {stdout_path}"
        )
    stderr_data = stderr_path.read_bytes()
    prefill_tps, generation_tps = common.parse_timing(stderr_data, stderr_path)
    forbidden = (
        common.STATS_PREFIX,
        common.ACCEPTANCE_PREFIX,
        common.ACCEPTANCE_TRACE_PREFIX,
    )
    if any(marker in stderr_data for marker in forbidden):
        raise RuntimeError(f"instrumentation unexpectedly active in {stderr_path}")
    return {
        "prompt": record["label"],
        "source_index": record["source_index"],
        "mode": mode,
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": common.sha256(output),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def run_pair(args, root, run_dir, label, record, prompt, order, expected):
    rows = []
    order_text = "-".join(order)
    for position, mode in enumerate(order, start=1):
        row = execute(
            args, root, run_dir, label, record, prompt, mode, expected
        )
        row["pair_order"] = order_text
        row["pair_position"] = position
        rows.append(row)
        common.cooldown(args.cooldown)
    return rows


def summarize(rows, records):
    samples = {}
    ratios = []
    upstream_values = []
    current_values = []
    wins = 0
    equals = 0
    for record in records:
        task = record["label"]
        selected = {row["mode"]: row for row in rows if row["prompt"] == task}
        if set(selected) != set(MODES):
            raise RuntimeError(f"incomplete baseline pair for {task}")
        upstream = selected["upstream_main"]["generation_tps"]
        current = selected["current_branch"]["generation_tps"]
        ratio = current / upstream
        ratios.append(ratio)
        upstream_values.append(upstream)
        current_values.append(current)
        wins += ratio > 1.0
        equals += ratio == 1.0
        samples[task] = {
            "source_index": record["source_index"],
            "order": selected["current_branch"]["pair_order"],
            "upstream_generation_tps": upstream,
            "current_generation_tps": current,
            "paired_ratio": ratio,
            "delta_percent": (ratio - 1.0) * 100.0,
        }
    quartiles = statistics.quantiles(ratios, n=4, method="inclusive")
    geomean = statistics.geometric_mean(ratios)
    gate = (
        geomean >= MIN_GEOMEAN and
        wins >= MIN_WINS and
        min(ratios) >= MIN_TASK_RATIO
    )
    return {
        "analysis": "ds4_upstream_main_baseline_comparison",
        "sample_count": len(records),
        "samples": samples,
        "upstream_generation_tps_median": statistics.median(upstream_values),
        "current_generation_tps_median": statistics.median(current_values),
        "ratio_of_medians": (
            statistics.median(current_values) /
            statistics.median(upstream_values)
        ),
        "paired_ratio_median": statistics.median(ratios),
        "paired_ratio_geometric_mean": geomean,
        "paired_ratio_q1": quartiles[0],
        "paired_ratio_q3": quartiles[2],
        "paired_ratio_minimum": min(ratios),
        "paired_ratio_maximum": max(ratios),
        "current_faster_tasks": wins,
        "current_equal_tasks": equals,
        "current_slower_tasks": len(records) - wins - equals,
        "progress_gate": {
            "minimum_geometric_mean": MIN_GEOMEAN,
            "minimum_wins": MIN_WINS,
            "minimum_task_ratio": MIN_TASK_RATIO,
            "pass": gate,
        },
    }


def render_report(summary, upstream_commit, current_commit):
    gate = summary["progress_gate"]
    lines = [
        "# ds4 Upstream/Main Baseline Comparison",
        "",
        "All samples are uninstrumented ordinary Metal decoding with no "
        "DSpark model or runtime enabled.",
        "Both binaries reproduced the frozen output byte-for-byte. Paired "
        "ratios are authoritative.",
        "",
        f"- Upstream commit: `{upstream_commit}`.",
        f"- Current commit: `{current_commit}`.",
        "",
        "| samples | upstream median | current median | ratio of medians | "
        "median paired | geometric mean | current faster |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | "
        f"{summary['upstream_generation_tps_median']:.2f} t/s | "
        f"{summary['current_generation_tps_median']:.2f} t/s | "
        f"{summary['ratio_of_medians']:.4f}x | "
        f"{summary['paired_ratio_median']:.4f}x | "
        f"{summary['paired_ratio_geometric_mean']:.4f}x | "
        f"{summary['current_faster_tasks']}/{summary['sample_count']} |",
        "",
        f"- Paired-ratio interquartile range: "
        f"{summary['paired_ratio_q1']:.4f}x-"
        f"{summary['paired_ratio_q3']:.4f}x.",
        f"- Paired-ratio range: {summary['paired_ratio_minimum']:.4f}x-"
        f"{summary['paired_ratio_maximum']:.4f}x.",
        f"- Current faster/equal/slower: {summary['current_faster_tasks']}/"
        f"{summary['current_equal_tasks']}/"
        f"{summary['current_slower_tasks']}.",
        "",
        "## Tasks",
        "",
        "| task | order | upstream | current | ratio | delta |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for task, item in summary["samples"].items():
        lines.append(
            f"| {task} | {item['order']} | "
            f"{item['upstream_generation_tps']:.2f} t/s | "
            f"{item['current_generation_tps']:.2f} t/s | "
            f"{item['paired_ratio']:.4f}x | "
            f"{item['delta_percent']:+.1f}% |"
        )
    lines.extend([
        "",
        "## Progress Gate",
        "",
        f"**{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        f"- Require geometric current/upstream ratio at least "
        f"`{gate['minimum_geometric_mean']:.2f}x`.",
        f"- Require at least `{gate['minimum_wins']}/{summary['sample_count']}` "
        "tasks faster.",
        f"- Require no task below `{gate['minimum_task_ratio']:.2f}x`.",
        "",
        "- Two global warmup pairs are excluded from every reported value.",
        "- Measured order alternates upstream-first and current-first.",
        "- Every `DS4_*` environment variable is cleared for both modes.",
        "- No DSpark sidecar, stats, trace, diagnostics, or profiler is enabled.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args, root = parse_args()
    for name in (
        "output_reference", "current_binary", "upstream_source",
        "upstream_binary", "model", "corpus_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    # The shared frozen-corpus loader validates these paths even though this
    # benchmark never enables DSpark.
    args.binary = args.current_binary
    args.dspark_model = (root / "gguf/ds4flash-dspark.gguf").resolve()
    for label, path in (
        ("current binary", args.current_binary),
        ("upstream source", args.upstream_source),
        ("upstream binary", args.upstream_binary),
        ("base model", args.model),
        ("corpus", args.corpus_dir),
    ):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    dirty = common.git_output(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass "
            "--allow-dirty:\n" + dirty
        )
    upstream_dirty = common.git_output(
        args.upstream_source, "status", "--porcelain", "--untracked-files=no"
    )
    if upstream_dirty:
        raise SystemExit("upstream worktree has tracked changes:\n" + upstream_dirty)
    current_commit = common.git_output(root, "rev-parse", "HEAD")
    upstream_commit = common.git_output(
        args.upstream_source, "rev-parse", "HEAD"
    )
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise SystemExit(
            "upstream worktree commit mismatch: "
            f"{upstream_commit} != {EXPECTED_UPSTREAM_COMMIT}"
        )

    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, SAMPLE_COUNT, provenance["selection_policy"]
    )
    reference = load_output_reference(args, records, selection)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/ds4-master-baseline-{SAMPLE_COUNT}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for position, record in enumerate(records, start=1):
        order = mode_order(position)
        prompt = prompts[record["label"]]
        print(
            f"{record['label']} measured order: {' -> '.join(order)}\n"
            f"  upstream: {command_text(args, prompt, 'upstream_main')}\n"
            f"  current: {command_text(args, prompt, 'current_branch')}"
        )
    warmups = warmup_schedule(records)
    total = len(warmups) * 2 + len(records) * 2
    print(
        f"Upstream baseline comparison: {total} uninstrumented processes; "
        f"{len(warmups) * 2} excluded warmups and "
        f"{len(records) * 2} measured."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts(prompts, records)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": current_commit,
        "git_status_tracked": dirty,
        "experiment": "ds4_upstream_main_baseline_comparison",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "selection": selection,
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "instrumented": False,
            "dspark": False,
            "measured_pairs_per_task": 1,
            "alternating_order": True,
            "global_warmup_pairs": len(warmups),
        },
        "current_binary": common.file_metadata(args.current_binary),
        "current_commit": current_commit,
        "upstream_binary": common.file_metadata(args.upstream_binary),
        "upstream_source": str(args.upstream_source),
        "upstream_commit": upstream_commit,
        "upstream_git_status_tracked": upstream_dirty,
        "base_model": common.file_metadata(args.model),
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "output_reference": {
            "summary": common.file_metadata(reference["summary_path"]),
            "metadata": common.file_metadata(reference["metadata_path"]),
            "csv": common.file_metadata(reference["csv_path"]),
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
    for number, (record, order) in enumerate(warmups, start=1):
        warmup_rows.extend(run_pair(
            args, root, run_dir, f"warmup-{number:02d}", record,
            prompts[record["label"]], order,
            reference["tasks"][record["label"]],
        ))
    measured_rows = []
    sequence = 0
    for position, record in enumerate(records, start=1):
        pair_rows = run_pair(
            args, root, run_dir, f"measured-{position:02d}", record,
            prompts[record["label"]], mode_order(position),
            reference["tasks"][record["label"]],
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
        run_dir / "warmups.csv", warmup_rows,
        tuple(field for field in fields if field != "sequence"),
    )
    summary = summarize(measured_rows, records)
    summary["selection"] = selection
    summary["upstream_commit"] = upstream_commit
    summary["current_commit"] = current_commit
    summary["output_reference"] = str(reference["summary_path"])
    report = render_report(summary, upstream_commit, current_commit)
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
