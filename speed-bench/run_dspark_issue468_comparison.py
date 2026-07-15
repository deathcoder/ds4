#!/usr/bin/env python3
"""Reproduce the issue-468 8k workload with paired no-log DSpark runs."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time


TIMING_RE = re.compile(
    rb"ds4: prefill: ([0-9]+(?:\.[0-9]+)?) t/s, "
    rb"generation: ([0-9]+(?:\.[0-9]+)?) t/s"
)
STATS_PREFIX = b"ds4: DSpark runtime stats "
ACCEPTANCE_PREFIX = b"ds4: DSpark acceptance audit "
ACCEPTANCE_TRACE_PREFIX = b"ds4: DSpark acceptance trace "
INT_STATS = {
    "proposals", "selected", "source_fallbacks", "multi_attempts", "emitted",
    "target_evals", "target_eval_tokens", "target_evals_avoided",
    "batch_attempts", "batch_full", "batch_partial", "batch_fallbacks",
    "fast_calls", "fast_failures", "fast_exact_fallbacks",
    "depth1", "depth2", "depth3", "depth4", "depth5",
}
FLOAT_STATS = {
    "avg_depth", "sidecar_ms", "bridge_ms", "stage0_ms", "stage1_ms",
    "stage2_ms", "head_ms", "chain_ms", "target_eval_ms",
    "prefill_sidecar_ms", "generation_sidecar_ms", "generation_bridge_ms",
    "generation_stage0_ms", "generation_stage1_ms", "generation_stage2_ms",
    "generation_head_ms", "generation_chain_ms",
}
STATS_FIELDS = tuple(sorted(INT_STATS | FLOAT_STATS))
ACCEPTANCE_INT_FIELDS = (
    "block_size", "proposals", "proposed_drafts", "accepted_drafts",
    "paper_acceptance_sum", "full_accepts", "truncated_proposals",
)
ACCEPTANCE_INT_ARRAY_FIELDS = (
    "proposed_at", "reached_at", "accepted_at", "rejected_at",
    "confidence_valid", "prefix_confidence_valid", "confidence_nonfinite",
)
ACCEPTANCE_FLOAT_ARRAY_FIELDS = (
    "confidence_sum", "confidence_brier", "prefix_confidence_sum",
    "prefix_brier",
)
PROMPT_ORDER = ("code_8k", "synthesis_8k", "grounded_8k")
DSPARK_DEFAULT_CONFIDENCE_THRESHOLD = "0.455"
DSPARK_FIXED_CONFIDENCE_THRESHOLD = "0"

# DSpark rows from Table 1 of arXiv:2607.05147v1. The paper used non-thinking
# generation, temperature 1.0, seven draft tokens, and the named benchmark
# suites. These are reference figures, not matched V4-Flash thresholds.
PAPER_DSPARK_TABLE1 = {
    "Qwen3-4B": {
        "math": (6.11, 5.70, 4.89),
        "code": (5.13, 5.38, 4.86),
        "chat": (3.64, 3.54, 3.29),
    },
    "Qwen3-8B": {
        "math": (6.17, 5.78, 5.01),
        "code": (5.16, 5.52, 5.17),
        "chat": (3.72, 3.58, 3.21),
    },
    "Qwen3-14B": {
        "math": (6.21, 5.74, 4.94),
        "code": (5.26, 5.43, 5.02),
        "chat": (3.70, 3.58, 3.13),
    },
    "Gemma4-12B": {
        "math": (6.05, 5.78, 5.12),
        "code": (5.11, 5.64, 4.51),
        "chat": (3.49, 3.35, 2.92),
    },
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def run_capture(command, cwd):
    try:
        return subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        ).stdout.decode("utf-8", "replace").strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def git_output(root, *args):
    return run_capture(["git", *args], root)


def cleared_env_keys(env):
    return sorted(key for key in env if key.startswith("DS4_"))


def benchmark_env(
    mode, fast_verifier, stats=False, exact_head_batch=False,
    acceptance_audit=False, acceptance_trace=False, confidence_threshold=None,
):
    if acceptance_trace and not acceptance_audit:
        raise ValueError("acceptance trace requires acceptance audit")
    if confidence_threshold is not None:
        if mode != "runtime":
            raise ValueError("confidence threshold requires DSpark runtime mode")
        try:
            threshold = float(confidence_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence threshold must be a finite number") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0,1]")
    env = os.environ.copy()
    for key in cleared_env_keys(env):
        env.pop(key, None)
    if mode == "runtime":
        env["DS4_DSPARK_GPU_RUNTIME"] = "1"
        env["DS4_DSPARK_MULTI_COMMIT"] = "1"
        if fast_verifier:
            env["DS4_DSPARK_FAST_BATCH_VERIFY"] = "1"
        if exact_head_batch:
            env["DS4_DSPARK_EXACT_HEAD_BATCH"] = "1"
        if stats:
            env["DS4_DSPARK_GPU_RUNTIME_STATS"] = "1"
        if acceptance_audit:
            env["DS4_DSPARK_ACCEPTANCE_AUDIT"] = "1"
        if acceptance_trace:
            env["DS4_DSPARK_ACCEPTANCE_TRACE"] = "1"
        if confidence_threshold is not None:
            env["DS4_DSPARK_CONFIDENCE_THRESHOLD"] = str(confidence_threshold)
    return env


def parse_args():
    root = Path(__file__).resolve().parent.parent
    corpus = root / "speed-bench/issue468"
    parser = argparse.ArgumentParser(
        description="Run the issue-468 long-prompt baseline/DSpark comparison."
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model", type=Path,
        default=root / "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
    )
    parser.add_argument(
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
    )
    parser.add_argument("--corpus-dir", type=Path, default=corpus)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--stats-pass", action="store_true")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="run one baseline reference and one instrumented runtime per prompt",
    )
    parser.add_argument(
        "--acceptance-audit",
        action="store_true",
        help="collect paper-aligned acceptance and confidence metrics only",
    )
    parser.add_argument(
        "--acceptance-reference",
        type=Path,
        help="compare an acceptance audit with a prior summary.json",
    )
    parser.add_argument(
        "--nothink",
        action="store_true",
        help="render prompts in non-thinking mode instead of the CLI default",
    )
    parser.add_argument(
        "--fast-verifier", action="store_true",
        help="use the experimental compute-batched verifier (not correctness-safe on this corpus)",
    )
    parser.add_argument(
        "--exact-head-batch", action="store_true",
        help="batch intermediate output heads while retaining exact target state and final logits",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0 or args.pairs <= 0 or args.warmups < 0:
        parser.error("ctx, tokens, and pairs must be positive; warmups cannot be negative")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to benchmark without --confirm-ready")
    if args.fast_verifier and args.exact_head_batch:
        parser.error("--fast-verifier and --exact-head-batch are separate experiments")
    if args.acceptance_audit and args.fast_verifier:
        parser.error("--acceptance-audit requires the exact verifier")
    diagnostic_modes = args.stats_only + args.acceptance_audit
    if args.stats_pass and diagnostic_modes:
        parser.error("--stats-pass is mutually exclusive with diagnostic-only modes")
    if diagnostic_modes > 1:
        parser.error("--stats-only and --acceptance-audit are mutually exclusive")
    if args.acceptance_reference and not args.acceptance_audit:
        parser.error("--acceptance-reference requires --acceptance-audit")
    if args.nothink and not args.acceptance_audit:
        parser.error("--nothink requires --acceptance-audit")
    return args, root


def load_inputs(args, root):
    provenance_path = args.corpus_dir / "provenance.json"
    reference_path = args.corpus_dir / "mtp_reference.json"
    for label, path in (
        ("binary", args.binary), ("base model", args.model),
        ("DSpark model", args.dspark_model), ("provenance", provenance_path),
        ("MTP reference", reference_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" + dirty
        )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    prompts = {}
    for label in PROMPT_ORDER:
        item = provenance["prompts"][label]
        path = args.corpus_dir / item["file"]
        if not path.is_file():
            raise SystemExit(f"missing prompt {label}: {path}")
        data = path.read_bytes()
        if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
            raise SystemExit(f"provenance mismatch for {label}: {path}")
        prompts[label] = path
    return prompts, provenance, reference


def load_acceptance_reference(args, prompts, provenance):
    if args.acceptance_reference is None:
        return None
    summary_path = args.acceptance_reference.resolve()
    metadata_path = summary_path.parent / "metadata.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing acceptance reference: {summary_path}")
    if not metadata_path.is_file():
        raise SystemExit(f"missing acceptance reference metadata: {metadata_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid acceptance reference: {exc}") from exc

    config = metadata.get("config", {})
    if config.get("execution_mode") != "acceptance_audit":
        raise SystemExit("acceptance reference was not produced by --acceptance-audit")
    reference_nothink = bool(config.get("nothink"))
    if reference_nothink == args.nothink:
        raise SystemExit("acceptance reference must use the opposite thinking mode")
    expected_config = {
        "ctx": args.ctx,
        "tokens": args.tokens,
        "temperature": 0,
        "seed": 1,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"acceptance reference {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    for key, expected_path in (
        ("binary", args.binary),
        ("base_model", args.model),
        ("dspark_model", args.dspark_model),
    ):
        actual = metadata.get(key, {}).get("path")
        if actual is None or Path(actual).resolve() != expected_path.resolve():
            raise SystemExit(f"acceptance reference {key} path mismatch")
    reference_prompts = metadata.get("provenance", {}).get("prompts", {})
    for label, path in prompts.items():
        item = reference_prompts.get(label, {})
        expected = provenance["prompts"][label]
        if item.get("sha256") != expected["sha256"] or not path.is_file():
            raise SystemExit(f"acceptance reference prompt mismatch for {label}")
        if label not in summary.get("prompts", {}):
            raise SystemExit(f"acceptance reference summary omitted {label}")
    return {
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "summary": summary,
        "metadata": metadata,
        "nothink": reference_nothink,
    }


def mode_command(args, prompt, mode):
    command = [
        str(args.binary), "--backend", "metal", "--model", str(args.model),
        "--ctx", str(args.ctx), "-n", str(args.tokens), "--temp", "0",
        "--seed", "1", "--prompt-file", str(prompt),
    ]
    if args.nothink:
        command.append("--nothink")
    if mode == "runtime":
        command.extend(("--dspark", str(args.dspark_model)))
    return command


def command_text(
    args, prompt, mode, stats=False, acceptance_audit=False,
    acceptance_trace=False, confidence_threshold=None,
):
    if mode == "runtime" and confidence_threshold is None:
        confidence_threshold = getattr(args, "confidence_threshold", None)
    env = benchmark_env(
        mode, args.fast_verifier, stats, args.exact_head_batch,
        acceptance_audit, acceptance_trace, confidence_threshold,
    )
    keys = (
        "DS4_DSPARK_GPU_RUNTIME", "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_FAST_BATCH_VERIFY", "DS4_DSPARK_EXACT_HEAD_BATCH",
        "DS4_DSPARK_GPU_RUNTIME_STATS",
        "DS4_DSPARK_ACCEPTANCE_AUDIT", "DS4_DSPARK_ACCEPTANCE_TRACE",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return (prefix + " " if prefix else "") + shlex.join(
        mode_command(args, prompt, mode)
    )


def parse_timing(stderr_data, path):
    matches = TIMING_RE.findall(stderr_data)
    if not matches:
        raise RuntimeError(f"timing line not found in {path}")
    prefill, generation = (float(value) for value in matches[-1])
    if prefill <= 0 or generation <= 0:
        raise RuntimeError(f"non-positive throughput in {path}")
    return prefill, generation


def parse_stats(stderr_data, path):
    records = [
        line[len(STATS_PREFIX):] for line in stderr_data.splitlines()
        if line.startswith(STATS_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(f"expected one DSpark stats record in {path}, found {len(records)}")
    values = {}
    try:
        for item in records[0].decode("ascii").split():
            key, value = item.split("=", 1)
            if key in INT_STATS:
                values[key] = int(value)
            elif key in FLOAT_STATS:
                values[key] = float(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid DSpark stats in {path}: {exc}") from exc
    missing = [key for key in STATS_FIELDS if key not in values]
    if missing:
        raise RuntimeError(f"incomplete DSpark stats in {path}: {', '.join(missing)}")
    if values["emitted"] <= 0 or values["target_evals"] <= 0:
        raise RuntimeError(f"empty DSpark stats in {path}")
    return values


def _parse_audit_array(value, cast, field, path):
    try:
        return [cast(item) for item in value.split(",")]
    except ValueError as exc:
        raise RuntimeError(
            f"invalid DSpark acceptance array {field} in {path}: {exc}"
        ) from exc


def parse_acceptance_audit(stderr_data, path):
    records = [
        line[len(ACCEPTANCE_PREFIX):] for line in stderr_data.splitlines()
        if line.startswith(ACCEPTANCE_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one DSpark acceptance record in {path}, found {len(records)}"
        )
    values = {}
    try:
        for item in records[0].decode("ascii").split():
            key, value = item.split("=", 1)
            if key in ACCEPTANCE_INT_FIELDS:
                values[key] = int(value)
            elif key in ACCEPTANCE_INT_ARRAY_FIELDS:
                values[key] = _parse_audit_array(value, int, key, path)
            elif key in ACCEPTANCE_FLOAT_ARRAY_FIELDS:
                values[key] = _parse_audit_array(value, float, key, path)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid DSpark acceptance record in {path}: {exc}") from exc

    required = (
        ACCEPTANCE_INT_FIELDS + ACCEPTANCE_INT_ARRAY_FIELDS +
        ACCEPTANCE_FLOAT_ARRAY_FIELDS
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError(
            f"incomplete DSpark acceptance record in {path}: {', '.join(missing)}"
        )
    block_size = values["block_size"]
    if block_size <= 0 or values["proposals"] <= 0:
        raise RuntimeError(f"empty DSpark acceptance record in {path}")
    for field in ACCEPTANCE_INT_ARRAY_FIELDS + ACCEPTANCE_FLOAT_ARRAY_FIELDS:
        if len(values[field]) != block_size:
            raise RuntimeError(
                f"DSpark acceptance field {field} has {len(values[field])} "
                f"positions, expected {block_size}, in {path}"
            )
    if values["paper_acceptance_sum"] != (
        values["accepted_drafts"] + values["proposals"]
    ):
        raise RuntimeError(f"invalid paper acceptance sum in {path}")
    if sum(values["proposed_at"]) != values["proposed_drafts"]:
        raise RuntimeError(f"invalid proposed-position total in {path}")
    if sum(values["accepted_at"]) != values["accepted_drafts"]:
        raise RuntimeError(f"invalid accepted-position total in {path}")
    if sum(values["rejected_at"]) + values["full_accepts"] != values["proposals"]:
        raise RuntimeError(f"invalid rejection-position total in {path}")
    for pos in range(block_size):
        if not (
            values["accepted_at"][pos] <= values["reached_at"][pos]
            <= values["proposed_at"][pos]
        ):
            raise RuntimeError(f"invalid position {pos + 1} acceptance counts in {path}")
        if values["confidence_valid"][pos] > values["reached_at"][pos]:
            raise RuntimeError(f"invalid position {pos + 1} confidence count in {path}")
        if values["prefix_confidence_valid"][pos] > values["proposed_at"][pos]:
            raise RuntimeError(f"invalid position {pos + 1} prefix count in {path}")
    return values


def execute(
    args, root, run_dir, label, prompt_label, prompt, mode, reference,
    stats=False, acceptance_audit=False, acceptance_trace=False,
    confidence_threshold=None,
):
    if acceptance_trace and (mode != "runtime" or not acceptance_audit):
        raise ValueError("acceptance trace requires a runtime acceptance audit")
    stdout_path = run_dir / f"{label}.{prompt_label}.{mode}.stdout"
    stderr_path = run_dir / f"{label}.{prompt_label}.{mode}.stderr"
    if mode == "runtime" and confidence_threshold is None:
        confidence_threshold = getattr(args, "confidence_threshold", None)
    command = mode_command(args, prompt, mode)
    rendered_command = command_text(
        args, prompt, mode, stats, acceptance_audit, acceptance_trace,
        confidence_threshold,
    )
    print(
        f"[{label}/{prompt_label}] {mode}: {rendered_command}",
        flush=True,
    )
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command, cwd=root,
            env=benchmark_env(
                mode, args.fast_verifier, stats, args.exact_head_batch,
                acceptance_audit, acceptance_trace, confidence_threshold,
            ),
            stdout=stdout_fp, stderr=stderr_fp, check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"run failed with exit {completed.returncode}; see {stderr_path}")
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    if reference is not None and stdout_data != reference:
        raise RuntimeError(f"output differs from {prompt_label} baseline; see {stdout_path}")
    prefill_tps, generation_tps = parse_timing(stderr_data, stderr_path)
    if not stats and STATS_PREFIX in stderr_data:
        raise RuntimeError(f"throughput run unexpectedly emitted DSpark stats: {stderr_path}")
    if not acceptance_audit and ACCEPTANCE_PREFIX in stderr_data:
        raise RuntimeError(
            f"non-audit run unexpectedly emitted DSpark acceptance data: {stderr_path}"
        )
    has_acceptance_trace = ACCEPTANCE_TRACE_PREFIX in stderr_data
    if acceptance_trace and not has_acceptance_trace:
        raise RuntimeError(
            f"acceptance trace run emitted no proposal records: {stderr_path}"
        )
    if not acceptance_trace and has_acceptance_trace:
        raise RuntimeError(
            f"non-trace run unexpectedly emitted acceptance trace data: {stderr_path}"
        )
    row = {
        "prompt": prompt_label, "mode": mode, "prefill_tps": prefill_tps,
        "generation_tps": generation_tps, "wall_seconds": wall_seconds,
        "stdout_sha256": sha256(stdout_data), "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    if stats:
        row.update(parse_stats(stderr_data, stderr_path))
    if acceptance_audit:
        row["acceptance_audit"] = parse_acceptance_audit(stderr_data, stderr_path)
    return row, stdout_data


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def file_metadata(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size,
            "mtime": dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()}


def machine_snapshot(root):
    return {
        "captured_at": dt.datetime.now().astimezone().isoformat(),
        "thermal_state": run_capture(["pmset", "-g", "therm"], root),
        "processes": run_capture(
            ["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"], root
        ).splitlines()[:25],
    }


def collect_metadata(args, root, prompts, provenance, acceptance_reference=None):
    confidence_threshold = getattr(args, "confidence_threshold", None)
    commands = {}
    for label, prompt in prompts.items():
        if args.stats_only or args.acceptance_audit:
            commands[label] = {
                "baseline_reference": command_text(args, prompt, "baseline"),
            }
            if args.stats_only:
                commands[label]["stats_runtime"] = command_text(
                    args, prompt, "runtime", stats=True,
                    confidence_threshold=confidence_threshold,
                )
            else:
                commands[label]["acceptance_runtime"] = command_text(
                    args, prompt, "runtime", acceptance_audit=True,
                    acceptance_trace=bool(
                        getattr(args, "acceptance_trace", False)
                    ),
                    confidence_threshold=confidence_threshold,
                )
        else:
            commands[label] = {
                "baseline": command_text(args, prompt, "baseline"),
                "runtime": command_text(
                    args, prompt, "runtime",
                    confidence_threshold=confidence_threshold,
                ),
            }
            if args.stats_pass:
                commands[label]["stats_runtime"] = command_text(
                    args, prompt, "runtime", stats=True,
                    confidence_threshold=confidence_threshold,
                )
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": git_output(root, "status", "--porcelain", "--untracked-files=no"),
        "platform": platform.platform(), "uname": run_capture(["uname", "-a"], root),
        "cpu": run_capture(["sysctl", "-n", "machdep.cpu.brand_string"], root),
        "memory_bytes": run_capture(["sysctl", "-n", "hw.memsize"], root),
        "initial_snapshot": machine_snapshot(root),
        "inherited_ds4_environment": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("DS4_")
        },
        "cleared_environment_keys": cleared_env_keys(os.environ),
        "child_ds4_environment_policy": {
            "clear_all_inherited_ds4_keys": True,
            "baseline_keys": [],
            "runtime_keys": [
                "DS4_DSPARK_GPU_RUNTIME",
                "DS4_DSPARK_MULTI_COMMIT",
            ],
            "optional_runtime_keys": [
                "DS4_DSPARK_FAST_BATCH_VERIFY",
                "DS4_DSPARK_EXACT_HEAD_BATCH",
                "DS4_DSPARK_GPU_RUNTIME_STATS",
                "DS4_DSPARK_ACCEPTANCE_AUDIT",
                "DS4_DSPARK_ACCEPTANCE_TRACE",
                "DS4_DSPARK_CONFIDENCE_THRESHOLD",
            ],
        },
        "config": {
            "ctx": args.ctx, "tokens": args.tokens,
            "pairs": 0 if (args.stats_only or args.acceptance_audit) else args.pairs,
            "warmups_per_mode_per_prompt": 0 if (
                args.stats_only or args.acceptance_audit
            ) else args.warmups,
            "cooldown_seconds": args.cooldown, "temperature": 0, "seed": 1,
            "fast_verifier": args.fast_verifier,
            "exact_head_batch": args.exact_head_batch,
            "execution_mode": (
                "stats_only" if args.stats_only else
                "acceptance_audit" if args.acceptance_audit else "throughput"
            ),
            "throughput_instrumentation": False,
            "runtime_instrumentation": (
                args.stats_only or args.stats_pass or args.acceptance_audit
            ),
            "stats_pass": args.stats_pass,
            "stats_only": args.stats_only,
            "acceptance_audit": args.acceptance_audit,
            "acceptance_trace": bool(getattr(args, "acceptance_trace", False)),
            "confidence_threshold": confidence_threshold,
            "effective_confidence_threshold": (
                confidence_threshold if confidence_threshold is not None else
                DSPARK_DEFAULT_CONFIDENCE_THRESHOLD
            ),
            "nothink": args.nothink,
        },
        "binary": file_metadata(args.binary), "base_model": file_metadata(args.model),
        "dspark_model": file_metadata(args.dspark_model), "provenance": provenance,
        "prompts": {label: file_metadata(path) for label, path in prompts.items()},
        "commands": commands,
        "acceptance_reference": (
            {
                "summary": file_metadata(
                    acceptance_reference["summary_path"]
                ),
                "metadata": file_metadata(
                    acceptance_reference["metadata_path"]
                ),
                "nothink": acceptance_reference["nothink"],
            }
            if acceptance_reference else None
        ),
    }


def summarize(rows, reference):
    prompts = {}
    all_paired = []
    for label in PROMPT_ORDER:
        selected = [row for row in rows if row["prompt"] == label]
        baseline = [row["generation_tps"] for row in selected if row["mode"] == "baseline"]
        runtime = [row["generation_tps"] for row in selected if row["mode"] == "runtime"]
        paired = []
        for pair in sorted({row["pair"] for row in selected}):
            pair_rows = {row["mode"]: row for row in selected if row["pair"] == pair}
            paired.append(pair_rows["runtime"]["generation_tps"] / pair_rows["baseline"]["generation_tps"])
        all_paired.extend(paired)
        ratio = statistics.median(runtime) / statistics.median(baseline)
        dspark_delta = (ratio - 1.0) * 100.0
        mtp = reference["results"][label]
        prompts[label] = {
            "baseline_generation_tps_median": statistics.median(baseline),
            "runtime_generation_tps_median": statistics.median(runtime),
            "ratio_of_medians": ratio,
            "paired_ratio_median": statistics.median(paired),
            "paired_ratio_values": paired,
            "dspark_delta_percent": dspark_delta,
            "improvement_over_mtp_k2_percentage_points": dspark_delta - mtp["k2"]["delta_percent"],
            "improvement_over_mtp_k5_percentage_points": dspark_delta - mtp["k5"]["delta_percent"],
        }
    aggregate_ratio = statistics.median(all_paired)
    aggregate_delta = (aggregate_ratio - 1.0) * 100.0
    k2_delta = statistics.median(reference["results"][label]["k2"]["delta_percent"] for label in PROMPT_ORDER)
    k5_delta = statistics.median(reference["results"][label]["k5"]["delta_percent"] for label in PROMPT_ORDER)
    return {
        "prompts": prompts,
        "aggregate_paired_ratio_median": aggregate_ratio,
        "aggregate_paired_ratio_values": all_paired,
        "aggregate_dspark_delta_percent": aggregate_delta,
        "reference_mtp_k2_delta_percent_median": k2_delta,
        "reference_mtp_k5_delta_percent_median": k5_delta,
        "aggregate_improvement_over_mtp_k2_percentage_points": aggregate_delta - k2_delta,
        "aggregate_improvement_over_mtp_k5_percentage_points": aggregate_delta - k5_delta,
    }


def summarize_stats(rows):
    summary = {}
    for row in rows:
        emitted = row["emitted"]
        target_evals = row["target_evals"]
        target_ms_per_emitted = row["target_eval_ms"] / emitted
        sidecar_ms_per_emitted = row["generation_sidecar_ms"] / emitted
        summary[row["prompt"]] = {
            "emitted": emitted,
            "average_accepted_depth": row["avg_depth"],
            "target_evals_avoided": row["target_evals_avoided"],
            "target_evals_per_emitted": target_evals / emitted,
            "target_positions_per_eval": row["target_eval_tokens"] / target_evals,
            "target_eval_ms_per_eval": row["target_eval_ms"] / target_evals,
            "target_eval_ms_per_emitted": target_ms_per_emitted,
            "generation_sidecar_ms_per_emitted": sidecar_ms_per_emitted,
            "accounted_generation_ms_per_emitted": (
                target_ms_per_emitted + sidecar_ms_per_emitted
            ),
            "generation_bridge_ms_per_emitted": row["generation_bridge_ms"] / emitted,
            "generation_stage0_ms_per_emitted": row["generation_stage0_ms"] / emitted,
            "generation_stage1_ms_per_emitted": row["generation_stage1_ms"] / emitted,
            "generation_stage2_ms_per_emitted": row["generation_stage2_ms"] / emitted,
            "generation_head_ms_per_emitted": row["generation_head_ms"] / emitted,
            "generation_chain_ms_per_emitted": row["generation_chain_ms"] / emitted,
            "prefill_sidecar_ms": row["prefill_sidecar_ms"],
            "batch_attempts": row["batch_attempts"],
            "batch_full": row["batch_full"],
            "batch_partial": row["batch_partial"],
            "fast_calls": row["fast_calls"], "fast_failures": row["fast_failures"],
            "fast_exact_fallbacks": row["fast_exact_fallbacks"],
            "batch_fallbacks": row["batch_fallbacks"],
            "source_fallbacks": row["source_fallbacks"],
            "depth_counts": {
                str(depth): row[f"depth{depth}"] for depth in range(1, 6)
            },
        }
    return summary


def render_stats_report(summary):
    lines = [
        "# DSpark Issue 468 Stats-Only Summary",
        "",
        "Instrumented diagnostic only. Throughput values are intentionally omitted.",
        "Each runtime output matched a fresh uninstrumented baseline reference.",
        "",
        "| prompt | depth | evals/emitted | positions/eval | target ms/emitted | sidecar ms/emitted | accounted ms/emitted | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in PROMPT_ORDER:
        item = summary[label]
        lines.append(
            f"| {label} | {item['average_accepted_depth']:.3f} | "
            f"{item['target_evals_per_emitted']:.4f} | "
            f"{item['target_positions_per_eval']:.3f} | "
            f"{item['target_eval_ms_per_emitted']:.3f} | "
            f"{item['generation_sidecar_ms_per_emitted']:.3f} | "
            f"{item['accounted_generation_ms_per_emitted']:.3f} | "
            f"{item['batch_fallbacks']} |"
        )
    lines.extend(["", "Sidecar breakdown per emitted token:"])
    for label in PROMPT_ORDER:
        item = summary[label]
        lines.append(
            f"- {label}: bridge {item['generation_bridge_ms_per_emitted']:.3f} ms, "
            f"stages {item['generation_stage0_ms_per_emitted']:.3f}/"
            f"{item['generation_stage1_ms_per_emitted']:.3f}/"
            f"{item['generation_stage2_ms_per_emitted']:.3f} ms, "
            f"head {item['generation_head_ms_per_emitted']:.3f} ms, "
            f"chain {item['generation_chain_ms_per_emitted']:.3f} ms"
        )
    lines.extend(["", "Verifier outcomes:"])
    for label in PROMPT_ORDER:
        item = summary[label]
        lines.append(
            f"- {label}: {item['target_evals_avoided']} target evals avoided; "
            f"target {item['target_eval_ms_per_eval']:.3f} ms/eval; "
            f"batches {item['batch_attempts']} attempts, {item['batch_full']} full, "
            f"{item['batch_partial']} partial, {item['batch_fallbacks']} fallbacks; "
            f"source fallbacks {item['source_fallbacks']}; "
            f"fast {item['fast_calls']} calls/{item['fast_failures']} failures/"
            f"{item['fast_exact_fallbacks']} exact fallbacks"
        )
    return "\n".join(lines) + "\n"


def _ratio_or_none(numerator, denominator):
    return numerator / denominator if denominator else None


def _difference_or_none(current, reference):
    if current is None or reference is None:
        return None
    return current - reference


def paper_acceptance_reference():
    domains = {}
    for domain in ("math", "code", "chat"):
        model_means = [
            statistics.mean(model[domain])
            for model in PAPER_DSPARK_TABLE1.values()
        ]
        values = [
            value
            for model in PAPER_DSPARK_TABLE1.values()
            for value in model[domain]
        ]
        domains[domain] = {
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.mean(values),
            "verify_rate_minimum": min(values) / 8.0,
            "verify_rate_maximum": max(values) / 8.0,
            "verify_rate_mean": statistics.mean(values) / 8.0,
            "model_macro_minimum": min(model_means),
            "model_macro_maximum": max(model_means),
            "model_macro_mean": statistics.mean(model_means),
            "model_macro_verify_rate_minimum": min(model_means) / 8.0,
            "model_macro_verify_rate_maximum": max(model_means) / 8.0,
        }
    return {
        "source": "DSpark paper arXiv:2607.05147v1, Table 1",
        "metric": "accepted draft tokens plus one target bonus token per round",
        "protocol": {
            "target_models": list(PAPER_DSPARK_TABLE1),
            "draft_tokens": 7,
            "temperature": 1.0,
            "thinking": False,
            "confidence_scheduler": False,
            "domains": {
                "math": ["GSM8K", "MATH500", "AIME25"],
                "code": ["MBPP", "HumanEval", "LiveCodeBench"],
                "chat": ["MT-Bench", "Alpaca", "Arena-Hard"],
            },
        },
        "table1": PAPER_DSPARK_TABLE1,
        "domain_ranges": domains,
    }


def build_acceptance_mode_comparison(current_prompts, reference):
    reference_prompts = reference["summary"].get("prompts", {})
    prompt_comparisons = {}
    for label in PROMPT_ORDER:
        current = current_prompts[label]
        prior = reference_prompts[label]
        if current["block_size"] != prior.get("block_size"):
            raise SystemExit(
                f"acceptance reference block-size mismatch for {label}"
            )
        if len(current["positions"]) != len(prior.get("positions", [])):
            raise SystemExit(
                f"acceptance reference position-count mismatch for {label}"
            )
        positions = []
        for current_pos, prior_pos in zip(
            current["positions"], prior["positions"]
        ):
            if current_pos["position"] != prior_pos.get("position"):
                raise SystemExit(
                    f"acceptance reference position mismatch for {label}"
                )
            positions.append({
                "position": current_pos["position"],
                "conditional_acceptance_rate_reference": (
                    prior_pos["conditional_acceptance_rate"]
                ),
                "conditional_acceptance_rate_current": (
                    current_pos["conditional_acceptance_rate"]
                ),
                "conditional_acceptance_rate_delta": (
                    _difference_or_none(
                        current_pos["conditional_acceptance_rate"],
                        prior_pos["conditional_acceptance_rate"],
                    )
                ),
                "prefix_survival_rate_reference": (
                    prior_pos["prefix_survival_rate"]
                ),
                "prefix_survival_rate_current": (
                    current_pos["prefix_survival_rate"]
                ),
                "prefix_survival_rate_delta": (
                    _difference_or_none(
                        current_pos["prefix_survival_rate"],
                        prior_pos["prefix_survival_rate"],
                    )
                ),
            })
        prompt_comparisons[label] = {
            "paper_acceptance_length_reference": (
                prior["paper_acceptance_length"]
            ),
            "paper_acceptance_length_current": (
                current["paper_acceptance_length"]
            ),
            "paper_acceptance_length_delta": (
                current["paper_acceptance_length"] -
                prior["paper_acceptance_length"]
            ),
            "paper_verify_rate_reference": prior["paper_verify_rate"],
            "paper_verify_rate_current": current["paper_verify_rate"],
            "paper_verify_rate_delta": (
                current["paper_verify_rate"] - prior["paper_verify_rate"]
            ),
            "full_accept_rate_reference": prior["full_accept_rate"],
            "full_accept_rate_current": current["full_accept_rate"],
            "full_accept_rate_delta": (
                current["full_accept_rate"] - prior["full_accept_rate"]
            ),
            "positions": positions,
        }
    return {
        "reference_summary": str(reference["summary_path"]),
        "reference_generation_mode": (
            "non_thinking" if reference["nothink"] else "thinking_high"
        ),
        "prompts": prompt_comparisons,
    }


def summarize_acceptance(rows, nothink=False, acceptance_reference=None):
    prompts = {}
    for row in rows:
        audit = row["acceptance_audit"]
        proposals = audit["proposals"]
        proposed_drafts = audit["proposed_drafts"]
        positions = []
        for pos in range(audit["block_size"]):
            reached = audit["reached_at"][pos]
            proposed = audit["proposed_at"][pos]
            confidence_valid = audit["confidence_valid"][pos]
            prefix_valid = audit["prefix_confidence_valid"][pos]
            positions.append({
                "position": pos + 1,
                "proposed": proposed,
                "reached": reached,
                "accepted": audit["accepted_at"][pos],
                "rejected": audit["rejected_at"][pos],
                "prefix_survival_rate": _ratio_or_none(
                    audit["accepted_at"][pos], proposed
                ),
                "conditional_acceptance_rate": _ratio_or_none(
                    audit["accepted_at"][pos], reached
                ),
                "mean_conditional_confidence": _ratio_or_none(
                    audit["confidence_sum"][pos], confidence_valid
                ),
                "conditional_confidence_brier": _ratio_or_none(
                    audit["confidence_brier"][pos], confidence_valid
                ),
                "mean_prefix_confidence": _ratio_or_none(
                    audit["prefix_confidence_sum"][pos], prefix_valid
                ),
                "prefix_confidence_brier": _ratio_or_none(
                    audit["prefix_brier"][pos], prefix_valid
                ),
                "confidence_nonfinite": audit["confidence_nonfinite"][pos],
            })
        prompts[row["prompt"]] = {
            "block_size": audit["block_size"],
            "proposals": proposals,
            "draft_tokens_per_proposal": proposed_drafts / proposals,
            "accepted_draft_tokens_per_proposal": (
                audit["accepted_drafts"] / proposals
            ),
            "paper_acceptance_length": (
                audit["paper_acceptance_sum"] / proposals
            ),
            "paper_verify_rate": (
                audit["paper_acceptance_sum"] /
                (proposed_drafts + proposals)
            ),
            "full_accept_rate": audit["full_accepts"] / proposals,
            "full_accepts": audit["full_accepts"],
            "truncated_proposals": audit["truncated_proposals"],
            "positions": positions,
        }
    result = {
        "prompts": prompts,
        "paper_reference": paper_acceptance_reference(),
        "comparison_policy": {
            "generation_mode": (
                "non_thinking" if nothink else "thinking_high"
            ),
            "code_8k_reference_domain": "code",
            "synthesis_8k_reference_domain": None,
            "grounded_8k_reference_domain": None,
            "matched_reproduction": False,
            "reason": (
                "V4-Flash, greedy custom 8K prompts, and a five-token block do "
                "not match Table 1's target models, datasets, sampling, or block size"
            ),
        },
    }
    if acceptance_reference:
        result["mode_comparison"] = build_acceptance_mode_comparison(
            prompts, acceptance_reference
        )
    return result


def _fmt_rate(value):
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_rate_delta(value):
    return "n/a" if value is None else f"{value:+.3f}"


def render_acceptance_report(summary):
    generation_mode = summary["comparison_policy"]["generation_mode"]
    lines = [
        "# DSpark Issue 468 Acceptance Audit",
        "",
        "Correctness diagnostic only. Throughput values are intentionally omitted.",
        "Each audited runtime output matched a fresh uninstrumented baseline reference.",
        "Accepted length uses the paper's definition: accepted draft tokens plus one target bonus token.",
        f"Generation mode: {generation_mode.replace('_', '-')}.",
        "",
        "| prompt | proposals | drafts/proposal | accepted drafts/proposal | paper accept_len | verify rate | full accept |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in PROMPT_ORDER:
        item = summary["prompts"][label]
        lines.append(
            f"| {label} | {item['proposals']} | "
            f"{item['draft_tokens_per_proposal']:.3f} | "
            f"{item['accepted_draft_tokens_per_proposal']:.3f} | "
            f"{item['paper_acceptance_length']:.3f} | "
            f"{item['paper_verify_rate']:.3f} | "
            f"{item['full_accept_rate']:.1%} |"
        )
    for label in PROMPT_ORDER:
        lines.extend([
            "",
            f"## {label}",
            "",
            "| pos | reached | accepted | conditional | prefix survival | confidence | prefix confidence | rejected here |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in summary["prompts"][label]["positions"]:
            lines.append(
                f"| {item['position']} | {item['reached']} | {item['accepted']} | "
                f"{_fmt_rate(item['conditional_acceptance_rate'])} | "
                f"{_fmt_rate(item['prefix_survival_rate'])} | "
                f"{_fmt_rate(item['mean_conditional_confidence'])} | "
                f"{_fmt_rate(item['mean_prefix_confidence'])} | "
                f"{item['rejected']} |"
            )

    mode_comparison = summary.get("mode_comparison")
    if mode_comparison:
        lines.extend([
            "",
            "## Thinking-Mode Control",
            "",
            f"Reference mode: {mode_comparison['reference_generation_mode'].replace('_', '-')}; "
            f"current mode: {generation_mode.replace('_', '-')}.",
            "",
            "| prompt | reference accept_len | current accept_len | delta | reference verify | current verify | delta | reference full | current full | delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label in PROMPT_ORDER:
            item = mode_comparison["prompts"][label]
            lines.append(
                f"| {label} | "
                f"{item['paper_acceptance_length_reference']:.3f} | "
                f"{item['paper_acceptance_length_current']:.3f} | "
                f"{item['paper_acceptance_length_delta']:+.3f} | "
                f"{item['paper_verify_rate_reference']:.3f} | "
                f"{item['paper_verify_rate_current']:.3f} | "
                f"{item['paper_verify_rate_delta']:+.3f} | "
                f"{item['full_accept_rate_reference']:.1%} | "
                f"{item['full_accept_rate_current']:.1%} | "
                f"{item['full_accept_rate_delta']:+.1%} |"
            )
        lines.extend([
            "",
            "| prompt | pos | reference conditional | current conditional | delta | reference prefix | current prefix | delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label in PROMPT_ORDER:
            for item in mode_comparison["prompts"][label]["positions"]:
                lines.append(
                    f"| {label} | {item['position']} | "
                    f"{_fmt_rate(item['conditional_acceptance_rate_reference'])} | "
                    f"{_fmt_rate(item['conditional_acceptance_rate_current'])} | "
                    f"{_fmt_rate_delta(item['conditional_acceptance_rate_delta'])} | "
                    f"{_fmt_rate(item['prefix_survival_rate_reference'])} | "
                    f"{_fmt_rate(item['prefix_survival_rate_current'])} | "
                    f"{_fmt_rate_delta(item['prefix_survival_rate_delta'])} |"
                )

    reference = summary["paper_reference"]
    lines.extend([
        "",
        "## Official Reference",
        "",
        "DSpark paper Table 1 reports these accepted lengths across its released Qwen3 and Gemma4 checkpoints:",
        "",
        "| domain | benchmark-cell range | per-model macro range | macro mean | macro verify-rate range |",
        "|---|---:|---:|---:|---:|",
    ])
    for domain in ("math", "code", "chat"):
        item = reference["domain_ranges"][domain]
        lines.append(
            f"| {domain} | {item['minimum']:.2f}-{item['maximum']:.2f} | "
            f"{item['model_macro_minimum']:.2f}-{item['model_macro_maximum']:.2f} | "
            f"{item['model_macro_mean']:.2f} | "
            f"{item['model_macro_verify_rate_minimum']:.3f}-"
            f"{item['model_macro_verify_rate_maximum']:.3f} |"
        )
    code_item = summary["prompts"]["code_8k"]
    code_reference = reference["domain_ranges"]["code"]
    lines.extend([
        "",
        f"Directional code target: code_8k measured accept_len {code_item['paper_acceptance_length']:.3f} and verify rate {code_item['paper_verify_rate']:.3f}; the official Table 1 per-model code macro ranges are {code_reference['model_macro_minimum']:.2f}-{code_reference['model_macro_maximum']:.2f} and {code_reference['model_macro_verify_rate_minimum']:.3f}-{code_reference['model_macro_verify_rate_maximum']:.3f}, respectively.",
        "",
        "Protocol warning: Table 1 used Qwen3/Gemma4 targets, seven draft tokens, temperature 1.0 rejection sampling, non-thinking mode, the named public benchmark suites, and no confidence scheduler. This V4-Flash audit uses a five-token block, greedy decoding, and custom 8K prompts. " +
        (
            "The generation mode matches Table 1, but the remaining protocol differences still make this a directional comparison."
            if generation_mode == "non_thinking" else
            "The generation mode also differs from Table 1."
        ) +
        " Only code_8k has a declared nearest domain (code), and this is not a matched reproduction.",
    ])
    nonfinite = sum(
        position["confidence_nonfinite"]
        for prompt in summary["prompts"].values()
        for position in prompt["positions"]
    )
    lines.extend([
        "",
        f"- Non-finite confidence values: {nonfinite}",
        "- Capacity/EOS-truncated proposals are excluded from paper-aligned metrics: " +
        ", ".join(
            f"{label}={summary['prompts'][label]['truncated_proposals']}"
            for label in PROMPT_ORDER
        ),
        "- Conditional acceptance is P(position accepted | all earlier positions accepted).",
        "- Prefix survival is P(all positions through this one accepted), matching DeepSpec evaluator accept_rate@position.",
        "- Confidence columns use the checkpoint's raw sigmoid confidence-head outputs. No paper STS calibration parameters are present in the released V4 inference config or applied by ds4.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def render_report(summary, stats_summary=None):
    lines = [
        "# DSpark Issue 468 Comparison Summary", "",
        "Throughput samples are paired and uninstrumented. Published MTP values are",
        "single instrumented runs on another system; compare relative deltas only, not absolute t/s.", "",
        "| prompt | baseline | DSpark | ratio | DSpark delta | vs MTP K=2 | vs MTP K=5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in PROMPT_ORDER:
        item = summary["prompts"][label]
        lines.append(
            f"| {label} | {item['baseline_generation_tps_median']:.2f} t/s | "
            f"{item['runtime_generation_tps_median']:.2f} t/s | {item['paired_ratio_median']:.4f}x | "
            f"{item['dspark_delta_percent']:+.1f}% | "
            f"{item['improvement_over_mtp_k2_percentage_points']:+.1f} pp | "
            f"{item['improvement_over_mtp_k5_percentage_points']:+.1f} pp |"
        )
    lines.extend([
        "", f"- Aggregate median paired ratio: {summary['aggregate_paired_ratio_median']:.4f}x",
        f"- Aggregate DSpark delta: {summary['aggregate_dspark_delta_percent']:+.1f}%",
        f"- Published MTP K=2 median delta: {summary['reference_mtp_k2_delta_percent_median']:+.1f}%",
        f"- Improvement over published MTP K=2: {summary['aggregate_improvement_over_mtp_k2_percentage_points']:+.1f} percentage points",
        f"- Published MTP K=5 median delta: {summary['reference_mtp_k5_delta_percent_median']:+.1f}%",
        f"- Improvement over published MTP K=5: {summary['aggregate_improvement_over_mtp_k5_percentage_points']:+.1f} percentage points",
        f"- Measured pairs per prompt: {len(summary['prompts'][PROMPT_ORDER[0]]['paired_ratio_values'])}",
    ])
    if stats_summary:
        lines.extend(["", "## Separate Instrumentation Pass", ""])
        for label in PROMPT_ORDER:
            item = stats_summary[label]
            lines.append(
                f"- {label}: depth {item['average_accepted_depth']:.3f}, "
                f"target evals/emitted {item['target_evals_per_emitted']:.4f}, "
                f"positions/eval {item['target_positions_per_eval']:.3f}, "
                f"sidecar {item['generation_sidecar_ms_per_emitted']:.3f} ms/emitted, "
                f"fast {item['fast_calls']} calls/{item['fast_failures']} failures/"
                f"{item['fast_exact_fallbacks']} exact fallbacks"
            )
    return "\n".join(lines) + "\n"


def finish_metadata(metadata, root, run_dir):
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    metadata["final_snapshot"] = machine_snapshot(root)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def run_stats_only(args, root, run_dir, prompts, metadata):
    runs = []
    stats_rows = []
    for prompt_label, prompt in prompts.items():
        baseline_row, reference = execute(
            args, root, run_dir, "reference", prompt_label, prompt,
            "baseline", None,
        )
        runs.append(baseline_row)
        cooldown(args.cooldown)
        stats_row, _ = execute(
            args, root, run_dir, "stats", prompt_label, prompt,
            "runtime", reference, stats=True,
        )
        runs.append(stats_row)
        stats_rows.append(stats_row)
        cooldown(args.cooldown)

    fields = (
        "prompt", "mode", "prefill_tps", "generation_tps", "wall_seconds",
        "stdout_sha256", "stdout_file", "stderr_file",
    )
    write_csv(run_dir / "runs.csv", runs, fields + STATS_FIELDS)
    write_csv(run_dir / "stats.csv", stats_rows, fields + STATS_FIELDS)
    summary = summarize_stats(stats_rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = render_stats_report(summary)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw stats: {run_dir / 'stats.csv'}")
    return 0


def run_acceptance_audit(
    args, root, run_dir, prompts, metadata, acceptance_reference=None
):
    runs = []
    audit_rows = []
    for prompt_label, prompt in prompts.items():
        baseline_row, reference = execute(
            args, root, run_dir, "reference", prompt_label, prompt,
            "baseline", None,
        )
        runs.append(baseline_row)
        cooldown(args.cooldown)
        audit_row, _ = execute(
            args, root, run_dir, "acceptance", prompt_label, prompt,
            "runtime", reference, acceptance_audit=True,
        )
        runs.append(audit_row)
        audit_rows.append(audit_row)
        cooldown(args.cooldown)

    run_fields = (
        "prompt", "mode", "prefill_tps", "generation_tps", "wall_seconds",
        "stdout_sha256", "stdout_file", "stderr_file",
    )
    write_csv(run_dir / "runs.csv", runs, run_fields)
    summary = summarize_acceptance(
        audit_rows,
        nothink=args.nothink,
        acceptance_reference=acceptance_reference,
    )
    scalar_rows = []
    position_rows = []
    for label in PROMPT_ORDER:
        item = summary["prompts"][label]
        scalar_rows.append({
            "prompt": label,
            "block_size": item["block_size"],
            "proposals": item["proposals"],
            "draft_tokens_per_proposal": item["draft_tokens_per_proposal"],
            "accepted_draft_tokens_per_proposal": (
                item["accepted_draft_tokens_per_proposal"]
            ),
            "paper_acceptance_length": item["paper_acceptance_length"],
            "paper_verify_rate": item["paper_verify_rate"],
            "full_accept_rate": item["full_accept_rate"],
            "truncated_proposals": item["truncated_proposals"],
        })
        for position in item["positions"]:
            position_rows.append({"prompt": label, **position})
    write_csv(
        run_dir / "acceptance.csv",
        scalar_rows,
        tuple(scalar_rows[0]),
    )
    write_csv(
        run_dir / "positions.csv",
        position_rows,
        tuple(position_rows[0]),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = render_acceptance_report(summary)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw acceptance: {run_dir / 'acceptance.csv'}")
    print(f"Position details: {run_dir / 'positions.csv'}")
    return 0


def main():
    args, root = parse_args()
    args.confidence_threshold = (
        DSPARK_FIXED_CONFIDENCE_THRESHOLD if args.acceptance_audit else None
    )
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.corpus_dir = args.corpus_dir.resolve()
    prompts, provenance, reference = load_inputs(args, root)
    acceptance_reference = load_acceptance_reference(
        args, prompts, provenance
    )

    for label, prompt in prompts.items():
        if args.stats_only:
            print(
                f"{label} baseline reference: "
                f"{command_text(args, prompt, 'baseline')}"
            )
            print(
                f"{label} stats runtime:     "
                f"{command_text(args, prompt, 'runtime', stats=True)}"
            )
        elif args.acceptance_audit:
            print(
                f"{label} baseline reference: "
                f"{command_text(args, prompt, 'baseline')}"
            )
            print(
                f"{label} acceptance runtime: "
                f"{command_text(args, prompt, 'runtime', acceptance_audit=True)}"
            )
        else:
            print(f"{label} baseline: {command_text(args, prompt, 'baseline')}")
            print(f"{label} runtime:  {command_text(args, prompt, 'runtime')}")
    if args.stats_only:
        print(
            "Stats-only pass: one fresh baseline reference and one instrumented "
            "exact runtime per prompt; no throughput pairs."
        )
    elif args.acceptance_audit:
        print(
            "Acceptance audit: one fresh baseline reference and one exact "
            "paper-aligned acceptance runtime per prompt; no throughput pairs."
        )
        if acceptance_reference:
            print(
                "Thinking-mode control reference: "
                f"{acceptance_reference['summary_path']}"
            )
    else:
        print("Throughput pass: all DSpark stats and diagnostic instrumentation are disabled.")
    if args.fast_verifier:
        print(
            "WARNING: fast verification is known to diverge on code_8k; "
            "this mode is for correctness investigation, not performance reporting."
        )
    if args.exact_head_batch:
        print(
            "Exact-head batch mode: intermediate target heads are batched; "
            "target state and final continuation logits remain serial-exact."
        )
    if args.stats_pass:
        print("A separate one-run-per-prompt runtime stats pass will follow throughput.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = (
        f"issue468-stats-{stamp}" if args.stats_only else
        f"issue468-acceptance-nothink-{stamp}" if (
            args.acceptance_audit and args.nothink
        ) else
        f"issue468-acceptance-{stamp}" if args.acceptance_audit else
        f"issue468-{stamp}"
    )
    run_dir = (
        args.output_dir or root / "speed-bench/local-runs" / default_dir
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = collect_metadata(
        args, root, prompts, provenance, acceptance_reference
    )
    (run_dir / "metadata.start.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.stats_only:
        return run_stats_only(args, root, run_dir, prompts, metadata)
    if args.acceptance_audit:
        return run_acceptance_audit(
            args, root, run_dir, prompts, metadata, acceptance_reference
        )

    references = {}
    for prompt_label, prompt in prompts.items():
        for warmup in range(1, args.warmups + 1):
            for mode in ("baseline", "runtime"):
                _, output = execute(
                    args, root, run_dir, f"warmup-{warmup:02d}", prompt_label,
                    prompt, mode, references.get(prompt_label),
                )
                references.setdefault(prompt_label, output)
                cooldown(args.cooldown)

    rows = []
    sequence = 0
    for prompt_label, prompt in prompts.items():
        for pair in range(1, args.pairs + 1):
            order = ("baseline", "runtime") if pair % 2 else ("runtime", "baseline")
            for position, mode in enumerate(order, 1):
                sequence += 1
                row, output = execute(
                    args, root, run_dir, f"measured-{sequence:02d}", prompt_label,
                    prompt, mode, references.get(prompt_label),
                )
                references.setdefault(prompt_label, output)
                row.update(sequence=sequence, pair=pair, position=position)
                rows.append(row)
                cooldown(args.cooldown)

    summary = summarize(rows, reference)
    stats_rows = []
    stats_summary = None
    if args.stats_pass:
        for prompt_label, prompt in prompts.items():
            row, _ = execute(
                args, root, run_dir, "stats", prompt_label, prompt, "runtime",
                references[prompt_label], stats=True,
            )
            stats_rows.append(row)
            cooldown(args.cooldown)
        stats_summary = summarize_stats(stats_rows)

    throughput_fields = (
        "sequence", "prompt", "pair", "position", "mode", "prefill_tps",
        "generation_tps", "wall_seconds", "stdout_sha256", "stdout_file", "stderr_file",
    )
    write_csv(run_dir / "throughput.csv", rows, throughput_fields)
    if stats_rows:
        write_csv(
            run_dir / "stats.csv", stats_rows,
            ("prompt", "mode", "prefill_tps", "generation_tps", "wall_seconds",
             "stdout_sha256", "stdout_file", "stderr_file") + STATS_FIELDS,
        )
        (run_dir / "stats_summary.json").write_text(
            json.dumps(stats_summary, indent=2) + "\n", encoding="utf-8"
        )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render_report(summary, stats_summary)
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw results: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial raw files were retained.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
