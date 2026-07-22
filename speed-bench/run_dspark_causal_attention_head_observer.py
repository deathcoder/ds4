#!/usr/bin/env python3
"""Validate ratio-128 causal multi-row attention heads against serial decode."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import time

import run_dspark_proposal_slab_observer as slab_observer


LAYER = 41
SCHEDULES = (("scheduled", "0.75"), ("fixed-k5", "0"))
HEAD_RE = re.compile(
    rb"^ds4: DSpark causal attention head observer layer=(\d+) ratio=(\d+) "
    rb"proposed=(\d+) row=(\d+) max=(\S+) rms=(\S+) rel_l2=(\S+) "
    rb"max_ulp=(\d+) first=(\d+) batch=(\S+) serial=(\S+) result=(\w+)$",
    re.MULTILINE,
)
CONTROL_RE = re.compile(
    rb"^ds4: DSpark causal attention rowwise control layer=(\d+) ratio=(\d+) "
    rb"proposed=(\d+) row=(\d+) serial_max=(\S+) serial_rms=(\S+) "
    rb"serial_rel_l2=(\S+) batch_max=(\S+) batch_rms=(\S+) "
    rb"batch_rel_l2=(\S+) serial_result=(\w+) batch_result=(\w+)$",
    re.MULTILINE,
)
VEC_RE = re.compile(
    rb"^ds4: DSpark causal attention vec control layer=(\d+) ratio=(\d+) "
    rb"proposed=(\d+) row=(\d+) max=(\S+) rms=(\S+) rel_l2=(\S+) "
    rb"max_ulp=(\d+) result=(\w+)$",
    re.MULTILINE,
)
PROFILE_RE = re.compile(
    rb"^ds4: Metal FlashAttention prefill stage "
    rb"mode=(fused_gather_decode|causal_vec_query) tokens=(\d+) comp=(\d+) "
    rb"keys=(\d+) heads=(\d+) dim=(\d+) window=(\d+) ratio=(\d+) "
    rb"(prepare|attention_vec|attention_reduce)=([0-9]+(?:\.[0-9]+)?) ms$"
)
PROFILE_STAGES = ("prepare", "attention_vec", "attention_reduce")


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostic-only layer-41 ratio-128 causal attention "
            "head shadow."
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
        "--rowwise-control",
        action="store_true",
        help="report rowwise generic-attention localization without requiring exact heads",
    )
    parser.add_argument(
        "--vec-control",
        action="store_true",
        help=(
            "require the multi-query serial-arithmetic vec shadow to match "
            "the authoritative serial heads exactly"
        ),
    )
    parser.add_argument(
        "--vec-profile",
        action="store_true",
        help="synchronously attribute serial and exact vec-shadow attention cost",
    )
    parser.add_argument(
        "--dspark-model",
        type=Path,
        default=root / "gguf/ds4flash-dspark.gguf",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=root / "speed-bench/issue468/code_8k.txt",
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--serial-legacy",
        action="store_true",
        help=(
            "localize arithmetic drift by comparing against the historical "
            "one-row gathered FlashAttention route"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if args.rowwise_control and args.vec_control:
        parser.error("--rowwise-control and --vec-control are mutually exclusive")
    if args.vec_profile and not args.vec_control:
        parser.error("--vec-profile requires --vec-control")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to run without --confirm-ready")
    return args, root


def git_output(root, *args):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    ).stdout.strip()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_env(
        layer=None, threshold="0.75", serial_legacy=False, vec_profile=False):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("DS4_")}
    if layer is not None:
        env.update({
            "DS4_DSPARK_GPU_RUNTIME": "1",
            "DS4_DSPARK_MULTI_COMMIT": "1",
            "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS": "1",
            "DS4_DSPARK_CONFIDENCE_THRESHOLD": threshold,
            "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER": str(layer),
            "DS4_DSPARK_CAUSAL_ATTN_HEAD_OBSERVER_LAYER": str(layer),
        })
        if serial_legacy:
            env["DS4_METAL_DENSE_MIXED_GATHERED_LEGACY"] = "1"
        if vec_profile:
            env.update({
                "DS4_DSPARK_EXACT_LAYER_PROFILE": "1",
                "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER": str(layer),
                "DS4_METAL_DECODE_STAGE_PROFILE": "1",
                "DS4_METAL_DECODE_STAGE_PROFILE_LAYER": str(layer),
                "DS4_DSPARK_EXACT_ATTENTION_PROFILE": "1",
                "DS4_METAL_FLASH_ATTN_GATHERED_PROFILE": "1",
                "DS4_DSPARK_CAUSAL_ATTN_VEC_PROFILE": "1",
            })
    return env


def command(args, layer=None):
    cmd = [
        str(args.binary),
        "--backend", "metal",
        "--model", str(args.model),
        "--ctx", str(args.ctx),
        "-n", str(args.tokens),
        "--nothink",
        "--temp", "0",
        "--seed", "1",
        "--prompt-file", str(args.prompt_file),
    ]
    if layer is not None:
        cmd.extend(("--dspark", str(args.dspark_model)))
    return cmd


def command_text(
        args, layer=None, threshold="0.75", serial_legacy=False,
        vec_profile=False):
    env = clean_env(layer, threshold, serial_legacy, vec_profile)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS",
        "DS4_DSPARK_CONFIDENCE_THRESHOLD",
        "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER",
        "DS4_DSPARK_CAUSAL_ATTN_HEAD_OBSERVER_LAYER",
        "DS4_METAL_DENSE_MIXED_GATHERED_LEGACY",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
        "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
        "DS4_DSPARK_EXACT_ATTENTION_PROFILE",
        "DS4_METAL_FLASH_ATTN_GATHERED_PROFILE",
        "DS4_DSPARK_CAUSAL_ATTN_VEC_PROFILE",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return (prefix + " " if prefix else "") + shlex.join(command(args, layer))


def parse_head_observer(stderr_data, layer=LAYER, require_exact=True):
    if b"DSpark causal attention head observer" not in stderr_data:
        raise RuntimeError(f"layer {layer}: no causal attention head records")
    if any(
        line.startswith(b"ds4: DSpark causal attention head observer") and
        b"result=fallback" in line
        for line in stderr_data.splitlines()
    ):
        raise RuntimeError(f"layer {layer}: causal attention shadow fell back")

    proposals = []
    current = None
    for match in HEAD_RE.finditer(stderr_data):
        record_layer, ratio, proposed, row = (
            int(value) for value in match.groups()[:4]
        )
        result = match.group(12).decode("ascii")
        if record_layer != layer or ratio != 128:
            raise RuntimeError(f"layer {layer}: unexpected observer identity")
        if proposed < 2 or proposed > 5 or row >= proposed:
            raise RuntimeError(f"layer {layer}: invalid proposal row")
        if row == 0:
            if current is not None and current["rows"] != current["proposed"]:
                raise RuntimeError(f"layer {layer}: incomplete proposal rows")
            current = {"proposed": proposed, "rows": 0}
            proposals.append(current)
        if current is None or current["proposed"] != proposed or row != current["rows"]:
            raise RuntimeError(f"layer {layer}: non-contiguous proposal rows")
        if require_exact and result != "exact":
            raise RuntimeError(
                f"layer {layer}: causal attention head {result} at row {row}"
            )
        current.setdefault("results", []).append(result)
        current["rows"] += 1
    if not proposals or current["rows"] != current["proposed"]:
        raise RuntimeError(f"layer {layer}: incomplete proposal rows")
    return proposals


def parse_rowwise_control(stderr_data, layer=LAYER):
    controls = []
    for match in CONTROL_RE.finditer(stderr_data):
        record_layer, ratio, proposed, row = (
            int(value) for value in match.groups()[:4]
        )
        serial_result = match.group(11).decode("ascii")
        batch_result = match.group(12).decode("ascii")
        if record_layer != layer or ratio != 128 or not 2 <= proposed <= 5:
            raise RuntimeError(f"layer {layer}: unexpected rowwise control identity")
        if row >= proposed:
            raise RuntimeError(f"layer {layer}: invalid rowwise control row")
        controls.append({
            "proposed": proposed,
            "row": row,
            "serial_result": serial_result,
            "batch_result": batch_result,
        })
    if not controls:
        raise RuntimeError(f"layer {layer}: no rowwise control records")
    return controls


def parse_vec_control(stderr_data, layer=LAYER, require_exact=False):
    controls = []
    for match in VEC_RE.finditer(stderr_data):
        record_layer, ratio, proposed, row = (
            int(value) for value in match.groups()[:4]
        )
        result = match.group(9).decode("ascii")
        if record_layer != layer or ratio != 128 or not 2 <= proposed <= 5:
            raise RuntimeError(f"layer {layer}: unexpected vec control identity")
        if row >= proposed:
            raise RuntimeError(f"layer {layer}: invalid vec control row")
        if require_exact and result != "exact":
            raise RuntimeError(
                f"layer {layer}: causal vec attention {result} at row {row}"
            )
        controls.append({
            "proposed": proposed,
            "row": row,
            "result": result,
        })
    if not controls:
        raise RuntimeError(f"layer {layer}: no vec control records")
    return controls


def parse_vec_profile(stderr_data, layer=LAYER):
    events = []
    for line in stderr_data.splitlines():
        match = PROFILE_RE.match(line)
        if not match:
            continue
        mode, tokens, comp, keys, heads, dim, window, ratio, stage, elapsed = (
            match.groups()
        )
        event = {
            "mode": mode.decode("ascii"),
            "tokens": int(tokens),
            "comp": int(comp),
            "keys": int(keys),
            "heads": int(heads),
            "dim": int(dim),
            "window": int(window),
            "ratio": int(ratio),
            "stage": stage.decode("ascii"),
            "ms": float(elapsed),
        }
        if event["heads"] <= 0 or event["dim"] != 512:
            raise RuntimeError(f"layer {layer}: unexpected profile head shape")
        events.append(event)
    if not events:
        raise RuntimeError(f"layer {layer}: no causal vec profile records")

    serial_calls = []
    serial_current = []
    candidate_current = []
    comparisons = []
    for event in events:
        current = serial_current if event["mode"] == "fused_gather_decode" \
            else candidate_current
        expected = PROFILE_STAGES[len(current)] if len(current) < 3 else None
        if event["stage"] != expected:
            raise RuntimeError(
                f"layer {layer}: {event['mode']} stage sequence mismatch"
            )
        if current:
            signature = tuple(
                event[key] for key in
                ("tokens", "comp", "keys", "heads", "dim", "window", "ratio")
            )
            first_signature = tuple(
                current[0][key] for key in
                ("tokens", "comp", "keys", "heads", "dim", "window", "ratio")
            )
            if signature != first_signature:
                raise RuntimeError(
                    f"layer {layer}: {event['mode']} stage shape mismatch"
                )
        current.append(event)
        if event["stage"] != "attention_reduce":
            continue
        if event["mode"] == "fused_gather_decode":
            if event["tokens"] != 1:
                raise RuntimeError(
                    f"layer {layer}: serial profile width is not one"
                )
            serial_calls.append({
                "heads": event["heads"],
                "dim": event["dim"],
                "stages": {row["stage"]: row["ms"] for row in current},
            })
            serial_current = []
            continue

        width = event["tokens"]
        if len(serial_calls) != width:
            raise RuntimeError(
                f"layer {layer}: vec width {width} follows "
                f"{len(serial_calls)} serial calls"
            )
        candidate_shape = (event["heads"], event["dim"])
        if any(
                (call["heads"], call["dim"]) != candidate_shape
                for call in serial_calls):
            raise RuntimeError(
                f"layer {layer}: serial/candidate profile shape mismatch"
            )
        serial = {
            stage: sum(call["stages"][stage] for call in serial_calls)
            for stage in PROFILE_STAGES
        }
        candidate = {row["stage"]: row["ms"] for row in current}
        serial_total = sum(serial.values())
        candidate_total = sum(candidate.values())
        comparisons.append({
            "width": width,
            "serial": serial,
            "candidate": candidate,
            "serial_total_ms": serial_total,
            "candidate_total_ms": candidate_total,
            "candidate_serial_ratio": candidate_total / serial_total,
        })
        serial_calls = []
        candidate_current = []
    if serial_current or candidate_current or serial_calls:
        raise RuntimeError(f"layer {layer}: incomplete causal vec profile sequence")
    if not comparisons:
        raise RuntimeError(f"layer {layer}: no complete causal vec comparisons")
    return comparisons


def parse_diagnostics(
        stderr_data, layer=LAYER, rowwise_control=False, vec_control=False):
    heads = parse_head_observer(
        stderr_data, layer, require_exact=not (rowwise_control or vec_control)
    )
    slab = slab_observer.parse_observer(stderr_data, layer)
    if len(heads) != len(slab["observer"]):
        raise RuntimeError(f"layer {layer}: head/slab record count mismatch")
    controls = parse_rowwise_control(stderr_data, layer)
    expected_rows = sum(item["rows"] for item in heads)
    if len(controls) != expected_rows:
        raise RuntimeError(f"layer {layer}: incomplete rowwise control records")
    vec_controls = parse_vec_control(
        stderr_data, layer, require_exact=vec_control
    )
    if len(vec_controls) != expected_rows:
        raise RuntimeError(f"layer {layer}: incomplete vec control records")
    return {
        "heads": heads,
        "slab": slab,
        "controls": controls,
        "vec_controls": vec_controls,
    }


def execute(
        args, root, run_dir, label, layer, reference=None, threshold="0.75",
        serial_legacy=False, rowwise_control=False, vec_control=False,
        vec_profile=False):
    stdout_path = run_dir / f"{label}.stdout"
    stderr_path = run_dir / f"{label}.stderr"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.run(
            command(args, layer),
            cwd=root,
            env=clean_env(layer, threshold, serial_legacy, vec_profile),
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    wall = time.monotonic() - started
    stdout_data = stdout_path.read_bytes()
    stderr_data = stderr_path.read_bytes()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {proc.returncode}; see {stderr_path}"
        )
    if reference is not None and stdout_data != reference:
        raise RuntimeError(
            f"{label} output differs from fresh baseline; see {stdout_path}"
        )
    parsed = None if layer is None else parse_diagnostics(
        stderr_data, layer, rowwise_control, vec_control
    )
    profile = parse_vec_profile(stderr_data, layer) if vec_profile else []
    return stdout_data, {
        "label": label,
        "layer": layer,
        "threshold": None if layer is None else threshold,
        "wall_seconds": wall,
        "stdout_sha256": sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "proposal_records": 0 if parsed is None else len(parsed["heads"]),
        "head_rows": 0 if parsed is None else sum(
            item["rows"] for item in parsed["heads"]
        ),
        "head_results": [] if parsed is None else sorted({
            result
            for item in parsed["heads"]
            for result in item["results"]
        }),
        "rowwise_serial_results": [] if parsed is None else sorted({
            item["serial_result"] for item in parsed["controls"]
        }),
        "batch_rowwise_results": [] if parsed is None else sorted({
            item["batch_result"] for item in parsed["controls"]
        }),
        "vec_serial_results": [] if parsed is None else sorted({
            item["result"] for item in parsed["vec_controls"]
        }),
        "vec_profile": profile,
    }


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "prompt_file"):
        setattr(args, name, getattr(args, name).resolve())
    for mode, threshold in SCHEDULES:
        print(
            f"{mode}: "
            f"{command_text(args, LAYER, threshold, args.serial_legacy, args.vec_profile)}"
        )
    print(f"fresh baseline: {command_text(args)}")
    print(
        "Diagnostic pass only: the serial verifier remains authoritative; "
        "throughput reporting is disabled."
    )
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    for name, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("prompt", args.prompt_file),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass --allow-dirty:\n" +
            dirty
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / f"speed-bench/local-runs/causal-attn-head-{stamp}"
    run_dir = (args.output_dir or default_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline, baseline_row = execute(args, root, run_dir, "baseline", None)
    rows = [baseline_row]
    for mode, threshold in SCHEDULES:
        if args.cooldown:
            time.sleep(args.cooldown)
        _, row = execute(
            args,
            root,
            run_dir,
            mode,
            LAYER,
            baseline,
            threshold,
            args.serial_legacy,
            args.rowwise_control,
            args.vec_control,
            args.vec_profile,
        )
        row["mode"] = mode
        rows.append(row)

    fixed = next(row for row in rows if row.get("mode") == "fixed-k5")
    fixed_stderr = (run_dir / fixed["stderr_file"]).read_bytes()
    publications = slab_observer.parse_observer(
        fixed_stderr, LAYER
    )["publication"]
    if not any(item["accepted"] < item["proposed"] for item in publications):
        raise RuntimeError("fixed-K5 run did not exercise partial publication")

    summary = {
        "analysis": "dspark_causal_attention_head_observer",
        "result": "LOCALIZATION" if args.rowwise_control else "PASS",
        "layer": LAYER,
        "serial_route": (
            "legacy-gathered" if args.serial_legacy else "fused-gather"
        ),
        "runs": rows,
    }
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status_tracked": dirty,
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "cooldown": args.cooldown,
            "layer": LAYER,
            "schedules": list(SCHEDULES),
            "fast_verifier": False,
            "instrumented": True,
            "serial_route": (
                "legacy-gathered" if args.serial_legacy else "fused-gather"
            ),
            "rowwise_control": args.rowwise_control,
            "vec_control": args.vec_control,
            "vec_profile": args.vec_profile,
        },
        "paths": {
            "binary": str(args.binary),
            "model": str(args.model),
            "dspark_model": str(args.dspark_model),
            "prompt": str(args.prompt_file),
        },
        "commands": {
            "baseline": command_text(args),
            **{
                mode: command_text(
                    args, LAYER, threshold, args.serial_legacy,
                    args.vec_profile
                )
                for mode, threshold in SCHEDULES
            },
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n# DSpark Causal Attention Head Observer")
    print("\n- Result: " + ("LOCALIZATION" if args.rowwise_control else "PASS"))
    print(
        "- Serial route: "
        + ("legacy gathered" if args.serial_legacy else "fused gather")
        + "."
    )
    for row in rows[1:]:
        print(
            f"- {row['mode']}: {row['proposal_records']} proposal records, "
            f"{row['head_rows']} head rows; head {row['head_results']}, "
            f"rowwise/serial {row['rowwise_serial_results']}, "
            f"batch/rowwise {row['batch_rowwise_results']}, "
            f"vec/serial {row['vec_serial_results']}."
        )
    if args.vec_profile:
        by_width = {}
        for row in rows[1:]:
            for comparison in row["vec_profile"]:
                by_width.setdefault(comparison["width"], []).append(comparison)
        print("\n| width | calls | serial ms | vec shadow ms | ratio | prepare | vec | reduce |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for width in sorted(by_width):
            items = by_width[width]
            serial_ms = statistics.median(
                item["serial_total_ms"] for item in items
            )
            candidate_ms = statistics.median(
                item["candidate_total_ms"] for item in items
            )
            stages = {
                stage: statistics.median(
                    item["candidate"][stage] for item in items
                )
                for stage in PROFILE_STAGES
            }
            print(
                f"| {width} | {len(items)} | {serial_ms:.3f} | "
                f"{candidate_ms:.3f} | {candidate_ms / serial_ms:.3f}x | "
                f"{stages['prepare']:.3f} | {stages['attention_vec']:.3f} | "
                f"{stages['attention_reduce']:.3f} |"
            )
        print(
            "\nSynchronized diagnostic only; these values are not throughput measurements."
        )
    print(f"Raw results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
