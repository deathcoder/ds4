#!/usr/bin/env python3
"""Compare upstream and research-branch DSpark against their own baselines."""

import argparse
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


MODES = (
    "upstream_plain",
    "upstream_dspark",
    "current_plain",
    "current_dspark",
)
DEFAULT_SAMPLES = 8
CURRENT_THRESHOLD = "0.75"
EXPECTED_UPSTREAM_COMMIT = "0a7ad776b9068348e6cb09df8cafa9cadd285298"
EXPECTED_UPSTREAM_TREE = "06259f4681d5d59305ef1c8bb5705cb98e05193e"
UPSTREAM_STATS_PREFIX = b"ds4: DSpark stats "


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--upstream-source",
        type=Path,
        default=root.parent / "ds4-upstream-0a7ad77",
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
        "--dspark-model",
        type=Path,
        default=root / "gguf/ds4flash-dspark.gguf",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "speed-bench/humaneval-acceptance",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument(
        "--upstream-confidence",
        type=float,
        help=(
            "override upstream's documented default confidence threshold; "
            "omit for the recommended-policy pilot"
        ),
    )
    parser.add_argument(
        "--disable-upstream-scheduler",
        action="store_true",
        help="disable upstream's adaptive scheduler for a controlled follow-up",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-idle", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples < 4 or args.samples % len(MODES) != 0:
        parser.error("samples must be at least four and divisible by four")
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if (
        args.upstream_confidence is not None
        and not 0.0 <= args.upstream_confidence <= 1.0
    ):
        parser.error("upstream confidence must be in [0,1]")
    if not args.dry_run and not args.confirm_idle:
        parser.error("refusing to benchmark without --confirm-idle")
    if args.upstream_binary is None:
        args.upstream_binary = args.upstream_source / "ds4"
    args.nothink = True
    args.binary = args.current_binary
    return args, root


def mode_order(position):
    offset = (position - 1) % len(MODES)
    return MODES[offset:] + MODES[:offset]


def warmup_schedule(records):
    return ((records[0], MODES),)


def binary_for_mode(args, mode):
    if mode.startswith("upstream_"):
        return args.upstream_binary
    if mode.startswith("current_"):
        return args.current_binary
    raise ValueError(f"unknown mode {mode}")


def working_directory_for_mode(args, root, mode):
    if mode.startswith("upstream_"):
        return args.upstream_source
    if mode.startswith("current_"):
        return root
    raise ValueError(f"unknown mode {mode}")


def command(args, prompt, mode):
    cmd = [
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
    if mode == "upstream_dspark":
        cmd.extend(("--mtp", str(args.dspark_model), "--dspark"))
        if args.upstream_confidence is not None:
            cmd.extend((
                "--dspark-confidence",
                f"{args.upstream_confidence:g}",
            ))
    elif mode == "current_dspark":
        cmd.extend(("--dspark", str(args.dspark_model)))
    return cmd


def mode_env(args, mode):
    env = os.environ.copy()
    for key in common.cleared_env_keys(env):
        env.pop(key, None)
    if mode == "upstream_dspark" and args.disable_upstream_scheduler:
        env["DS4_DSPARK_SCHEDULER"] = "0"
    if mode == "current_dspark":
        env["DS4_DSPARK_GPU_RUNTIME"] = "1"
        env["DS4_DSPARK_MULTI_COMMIT"] = "1"
        env["DS4_DSPARK_CONFIDENCE_THRESHOLD"] = CURRENT_THRESHOLD
    return env


def command_text(args, prompt, mode):
    env = mode_env(args, mode)
    keys = (
        "DS4_DSPARK_SCHEDULER",
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    rendered = shlex.join(command(args, prompt, mode))
    return f"{prefix} {rendered}" if prefix else rendered


def execute(args, root, run_dir, label, record, prompt, mode):
    stdout_path = run_dir / f"{label}.{record['label']}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{record['label']}.{mode}.stderr"
    print(
        f"[{label}/{record['label']}] {mode}: "
        f"(cwd {working_directory_for_mode(args, root, mode)}) "
        f"{command_text(args, prompt, mode)}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            command(args, prompt, mode),
            cwd=working_directory_for_mode(args, root, mode),
            env=mode_env(args, mode),
            stdout=out,
            stderr=err,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} run failed with exit {completed.returncode}; "
            f"see {stderr_path}"
        )
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    prefill_tps, generation_tps = common.parse_timing(stderr_data, stderr_path)
    forbidden = (
        common.STATS_PREFIX,
        common.ACCEPTANCE_PREFIX,
        common.ACCEPTANCE_TRACE_PREFIX,
        common.ORACLE_TRACE_PREFIX,
        UPSTREAM_STATS_PREFIX,
    )
    if any(marker in stderr_data for marker in forbidden):
        raise RuntimeError(
            f"instrumentation unexpectedly active in {stderr_path}"
        )
    return {
        "prompt": record["label"],
        "source_index": record["source_index"],
        "mode": mode,
        "prefill_tps": prefill_tps,
        "generation_tps": generation_tps,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": common.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }


def run_group(args, root, run_dir, label, record, prompt, order):
    rows = []
    order_text = "-".join(order)
    outputs = {}
    for position, mode in enumerate(order, start=1):
        row = execute(
            args, root, run_dir, label, record, prompt, mode
        )
        row["group_order"] = order_text
        row["group_position"] = position
        rows.append(row)
        outputs[mode] = (run_dir / row["stdout_file"]).read_bytes()
        common.cooldown(args.cooldown)
    hashes = {common.sha256(value) for value in outputs.values()}
    if len(hashes) != 1:
        reference_mode = order[0]
        mismatches = [
            mode for mode in order
            if outputs[mode] != outputs[reference_mode]
        ]
        raise RuntimeError(
            f"output mismatch for {record['label']}: "
            f"{reference_mode} differs from {', '.join(mismatches)}"
        )
    return rows


def _metric_summary(values):
    return {
        "median": statistics.median(values),
        "geometric_mean": statistics.geometric_mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "wins": sum(value > 1.0 for value in values),
        "equals": sum(value == 1.0 for value in values),
        "losses": sum(value < 1.0 for value in values),
    }


def summarize(rows, records):
    samples = {}
    metric_values = {
        "upstream_dspark_vs_plain": [],
        "current_dspark_vs_plain": [],
        "current_plain_vs_upstream_plain": [],
        "current_dspark_vs_upstream_dspark": [],
        "relative_dspark_uplift": [],
    }
    medians = {mode: [] for mode in MODES}
    for record in records:
        task = record["label"]
        selected = {row["mode"]: row for row in rows if row["prompt"] == task}
        if set(selected) != set(MODES):
            raise RuntimeError(f"incomplete upstream comparison for {task}")
        values = {
            mode: selected[mode]["generation_tps"] for mode in MODES
        }
        upstream_ratio = (
            values["upstream_dspark"] / values["upstream_plain"]
        )
        current_ratio = (
            values["current_dspark"] / values["current_plain"]
        )
        current_plain_ratio = (
            values["current_plain"] / values["upstream_plain"]
        )
        current_dspark_ratio = (
            values["current_dspark"] / values["upstream_dspark"]
        )
        relative_uplift = current_ratio / upstream_ratio
        metrics = {
            "upstream_dspark_vs_plain": upstream_ratio,
            "current_dspark_vs_plain": current_ratio,
            "current_plain_vs_upstream_plain": current_plain_ratio,
            "current_dspark_vs_upstream_dspark": current_dspark_ratio,
            "relative_dspark_uplift": relative_uplift,
        }
        for name, value in metrics.items():
            metric_values[name].append(value)
        for mode, value in values.items():
            medians[mode].append(value)
        samples[task] = {
            "source_index": record["source_index"],
            "order": selected[MODES[0]]["group_order"],
            **{f"{mode}_tps": value for mode, value in values.items()},
            **metrics,
        }
    return {
        "analysis": "dspark_upstream_main_recommended_policy_pilot",
        "sample_count": len(records),
        "mode_generation_tps_medians": {
            mode: statistics.median(values)
            for mode, values in medians.items()
        },
        "metrics": {
            name: _metric_summary(values)
            for name, values in metric_values.items()
        },
        "samples": samples,
    }


def _ratio(value):
    return f"{value:.4f}x"


def render_report(summary, upstream_commit, current_commit,
                  upstream_confidence, scheduler_disabled):
    metrics = summary["metrics"]
    medians = summary["mode_generation_tps_medians"]
    upstream_policy = (
        f"confidence {upstream_confidence:g}"
        if upstream_confidence is not None
        else "documented confidence default 0.9"
    )
    upstream_policy += (
        ", scheduler disabled"
        if scheduler_disabled
        else ", adaptive scheduler enabled"
    )
    lines = [
        "# DSpark Upstream/Main Recommended-Policy Pilot",
        "",
        "All modes are uninstrumented, single-session Metal generation.",
        "Every upstream/current plain/DSpark output matched byte-for-byte.",
        "Each DSpark engine is compared with its own ordinary baseline.",
        "",
        f"- Upstream commit: `{upstream_commit}`.",
        f"- Current commit: `{current_commit}`.",
        f"- Upstream policy: {upstream_policy}.",
        f"- Current policy: confidence {CURRENT_THRESHOLD}, promoted exact runtime.",
        "",
        "| samples | upstream plain | upstream DSpark | upstream uplift | "
        "current plain | current DSpark | current uplift |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | "
        f"{medians['upstream_plain']:.2f} t/s | "
        f"{medians['upstream_dspark']:.2f} t/s | "
        f"{_ratio(metrics['upstream_dspark_vs_plain']['geometric_mean'])} | "
        f"{medians['current_plain']:.2f} t/s | "
        f"{medians['current_dspark']:.2f} t/s | "
        f"{_ratio(metrics['current_dspark_vs_plain']['geometric_mean'])} |",
        "",
        "## Aggregate",
        "",
        "| comparison | median | geometric mean | wins | range |",
        "|:---|---:|---:|---:|---:|",
    ]
    labels = {
        "upstream_dspark_vs_plain": "upstream DSpark / upstream plain",
        "current_dspark_vs_plain": "current DSpark / current plain",
        "current_plain_vs_upstream_plain": "current plain / upstream plain",
        "current_dspark_vs_upstream_dspark":
            "current DSpark / upstream DSpark",
        "relative_dspark_uplift": "current uplift / upstream uplift",
    }
    for name, label in labels.items():
        item = metrics[name]
        lines.append(
            f"| {label} | {_ratio(item['median'])} | "
            f"{_ratio(item['geometric_mean'])} | "
            f"{item['wins']}/{summary['sample_count']} | "
            f"{_ratio(item['minimum'])}-{_ratio(item['maximum'])} |"
        )
    lines.extend([
        "",
        "## Tasks",
        "",
        "| task | upstream plain | upstream DSpark | uplift | current plain | "
        "current DSpark | uplift | current DSpark / upstream |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task, item in summary["samples"].items():
        lines.append(
            f"| {task} | {item['upstream_plain_tps']:.2f} | "
            f"{item['upstream_dspark_tps']:.2f} | "
            f"{_ratio(item['upstream_dspark_vs_plain'])} | "
            f"{item['current_plain_tps']:.2f} | "
            f"{item['current_dspark_tps']:.2f} | "
            f"{_ratio(item['current_dspark_vs_plain'])} | "
            f"{_ratio(item['current_dspark_vs_upstream_dspark'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This pilot compares each implementation's intended policy, not isolated "
        "verifier mechanisms.",
        "- If upstream is competitive, rerun with confidence 0.75 and its "
        "scheduler disabled before attributing the difference to kernels.",
        "- One four-mode warmup group is excluded from every reported value.",
        "- Four mode positions are balanced exactly across the selected tasks.",
        "- No DSpark stats, trace, diagnostics, profiler, or fast verifier is enabled.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args, root = parse_args()
    for name in (
        "current_binary", "upstream_source", "upstream_binary", "model",
        "dspark_model", "corpus_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.binary = args.current_binary
    for label, path in (
        ("current binary", args.current_binary),
        ("upstream source", args.upstream_source),
        ("upstream binary", args.upstream_binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("corpus", args.corpus_dir),
    ):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.current_binary, os.X_OK):
        raise SystemExit(f"current binary is not executable: {args.current_binary}")
    if not os.access(args.upstream_binary, os.X_OK):
        raise SystemExit(
            f"upstream binary is not executable: {args.upstream_binary}"
        )

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
        raise SystemExit(
            "upstream worktree has tracked changes:\n" + upstream_dirty
        )
    upstream_commit = common.git_output(
        args.upstream_source, "rev-parse", "HEAD"
    )
    upstream_tree = common.git_output(
        args.upstream_source, "rev-parse", "HEAD^{tree}"
    )
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise SystemExit(
            f"upstream commit mismatch: {upstream_commit} != "
            f"{EXPECTED_UPSTREAM_COMMIT}"
        )
    if upstream_tree != EXPECTED_UPSTREAM_TREE:
        raise SystemExit(
            f"upstream tree mismatch: {upstream_tree} != "
            f"{EXPECTED_UPSTREAM_TREE}"
        )
    current_commit = common.git_output(root, "rev-parse", "HEAD")

    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, args.samples, provenance["selection_policy"]
    )
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/dspark-upstream-main-pilot-"
        f"{args.samples}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for position, record in enumerate(records, start=1):
        order = mode_order(position)
        prompt = prompts[record["label"]]
        print(f"{record['label']} measured order: {' -> '.join(order)}")
        for mode in order:
            print(
                f"  {mode} (cwd "
                f"{working_directory_for_mode(args, root, mode)}): "
                f"{command_text(args, prompt, mode)}"
            )
    warmups = warmup_schedule(records)
    total = len(warmups) * len(MODES) + len(records) * len(MODES)
    print(
        f"Upstream DSpark pilot: {total} uninstrumented processes; "
        f"{len(warmups) * len(MODES)} excluded warmups and "
        f"{len(records) * len(MODES)} measured."
    )
    if args.dry_run:
        print(
            "Dry run only; no prompts materialized and no model execution "
            "performed."
        )
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts(prompts, records)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "experiment": "dspark_upstream_main_recommended_policy_pilot",
        "platform": platform.platform(),
        "initial_snapshot": common.machine_snapshot(root),
        "git_commit": current_commit,
        "git_status_tracked": dirty,
        "upstream_commit": upstream_commit,
        "upstream_tree": upstream_tree,
        "upstream_git_status_tracked": upstream_dirty,
        "selection": selection,
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "temperature": 0,
            "seed": 1,
            "nothink": True,
            "instrumented": False,
            "sample_count": args.samples,
            "current_confidence": CURRENT_THRESHOLD,
            "upstream_confidence": args.upstream_confidence,
            "upstream_default_confidence": args.upstream_confidence is None,
            "upstream_scheduler_disabled": args.disable_upstream_scheduler,
            "mode_order_balanced": True,
            "warmup_groups": len(warmups),
        },
        "current_binary": common.file_metadata(args.current_binary),
        "upstream_binary": common.file_metadata(args.upstream_binary),
        "upstream_source": str(args.upstream_source),
        "base_model": common.file_metadata(args.model),
        "dspark_model": common.file_metadata(args.dspark_model),
        "provenance_source_commit": provenance.get("source_commit"),
        "cleared_environment_keys": common.cleared_env_keys(os.environ),
        "commands": {
            record["label"]: {
                mode: {
                    "cwd": str(
                        working_directory_for_mode(args, root, mode)
                    ),
                    "command": command_text(
                        args, prompts[record["label"]], mode
                    ),
                }
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
        warmup_rows.extend(run_group(
            args, root, run_dir, f"warmup-{number:02d}", record,
            prompts[record["label"]], order,
        ))
    measured_rows = []
    sequence = 0
    for position, record in enumerate(records, start=1):
        group_rows = run_group(
            args, root, run_dir, f"measured-{position:02d}", record,
            prompts[record["label"]], mode_order(position),
        )
        for row in group_rows:
            sequence += 1
            row["sequence"] = sequence
            measured_rows.append(row)

    fields = (
        "sequence", "prompt", "source_index", "group_order",
        "group_position", "mode", "prefill_tps", "generation_tps",
        "wall_seconds", "stdout_sha256", "stdout_file", "stderr_file",
    )
    common.write_csv(run_dir / "throughput.csv", measured_rows, fields)
    common.write_csv(
        run_dir / "warmups.csv",
        warmup_rows,
        tuple(field for field in fields if field != "sequence"),
    )
    summary = summarize(measured_rows, records)
    summary["selection"] = selection
    summary["upstream_commit"] = upstream_commit
    summary["current_commit"] = current_commit
    report = render_report(
        summary,
        upstream_commit,
        current_commit,
        args.upstream_confidence,
        args.disable_upstream_scheduler,
    )
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
