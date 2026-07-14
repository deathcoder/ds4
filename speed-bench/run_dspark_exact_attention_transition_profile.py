#!/usr/bin/env python3
"""User-run synchronized profile of exact DSpark attention modes."""

import argparse
from collections import Counter
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

import run_dspark_exact_layer_profile as layer_profile


CONTROL_STAGES = ("attention_pre_batch", "ffn_batch")
ATTENTION_MODES = ("raw", "dense_mixed", "sparse_indexed")
EXTRA_CLEARED_ENV_KEYS = ("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD",)
DEFAULT_VARIANT = "default_rb16"
DIRECT_VARIANT = "rb16_direct"
COMPARISON_VARIANTS = (DEFAULT_VARIANT, DIRECT_VARIANT)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Profile retained exact attention across dense-to-sparse entry."
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model",
        type=Path,
        default=root
        / "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
    )
    parser.add_argument(
        "--dspark-model", type=Path, default=root / "gguf/ds4flash-dspark.gguf"
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=root / "speed-bench/issue468/code_8k.txt",
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--layers", type=layer_profile.parse_layers, default=(42,))
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-single-mode",
        action="store_true",
        help="allow a run that does not contain both dense and sparse attention",
    )
    parser.add_argument(
        "--rb16-direct-comparison",
        action="store_true",
        help="compare default RB16 against the opt-in RB16-direct kernel",
    )
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


def selected_variants(args):
    return COMPARISON_VARIANTS if args.rb16_direct_comparison else (DEFAULT_VARIANT,)


def profile_env(layer=None, variant=DEFAULT_VARIANT):
    env = os.environ.copy()
    for key in layer_profile.cleared_env_keys(env):
        env.pop(key, None)
    for key in EXTRA_CLEARED_ENV_KEYS:
        env.pop(key, None)
    env["DS4_DSPARK_GPU_RUNTIME"] = "1"
    env["DS4_DSPARK_MULTI_COMMIT"] = "1"
    if variant == DIRECT_VARIANT:
        env["DS4_METAL_INDEXED_ATTN_RB16_DIRECT"] = "1"
    if layer is not None:
        env["DS4_DSPARK_EXACT_LAYER_PROFILE"] = "1"
        env["DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER"] = str(layer)
        env["DS4_METAL_DECODE_STAGE_PROFILE"] = "1"
        env["DS4_METAL_DECODE_STAGE_PROFILE_LAYER"] = str(layer)
        env["DS4_DSPARK_EXACT_ATTENTION_PROFILE"] = "1"
    return env


def command(args):
    return [
        str(args.binary),
        "--backend", "metal",
        "--model", str(args.model),
        "--ctx", str(args.ctx),
        "-n", str(args.tokens),
        "--temp", "0",
        "--seed", "1",
        "--prompt-file", str(args.prompt_file),
        "--dspark", str(args.dspark_model),
    ]


def command_text(args, layer=None, variant=DEFAULT_VARIANT):
    env = profile_env(layer, variant)
    keys = (
        "DS4_DSPARK_GPU_RUNTIME",
        "DS4_DSPARK_MULTI_COMMIT",
        "DS4_METAL_INDEXED_ATTN_RB16_DIRECT",
        "DS4_DSPARK_EXACT_LAYER_PROFILE",
        "DS4_DSPARK_EXACT_LAYER_PROFILE_LAYER",
        "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
        "DS4_DSPARK_EXACT_ATTENTION_PROFILE",
    )
    prefix = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return prefix + " " + shlex.join(command(args))


def parse_profile(data, expected_layer, path, variant=DEFAULT_VARIANT):
    rows = []
    for line in data.splitlines():
        match = layer_profile.PROFILE_RE.match(line)
        if not match:
            continue
        part, layer, pos, tokens, stage, elapsed = match.groups()
        part = part.decode("ascii")
        if part not in ("exact", "attention"):
            continue
        row = {
            "variant": variant,
            "part": part,
            "layer": int(layer),
            "pos": int(pos),
            "tokens": int(tokens),
            "stage": stage.decode("ascii"),
            "ms": float(elapsed),
        }
        if row["layer"] != expected_layer:
            raise RuntimeError(
                f"unexpected profiled layer {row['layer']} in {path}; "
                f"expected {expected_layer}"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no exact-attention profile records found in {path}")

    exact = [row for row in rows if row["part"] == "exact"]
    attention = [row for row in rows if row["part"] == "attention"]
    unknown_exact = sorted({row["stage"] for row in exact} - set(CONTROL_STAGES))
    unknown_attention = sorted(
        {row["stage"] for row in attention} - set(ATTENTION_MODES)
    )
    if unknown_exact:
        raise RuntimeError(
            f"unknown exact control stages in {path}: {', '.join(unknown_exact)}"
        )
    if unknown_attention:
        raise RuntimeError(
            f"unknown exact-attention modes in {path}: "
            + ", ".join(unknown_attention)
        )
    for stage in CONTROL_STAGES:
        if not any(row["stage"] == stage for row in exact):
            raise RuntimeError(f"missing exact control stage {stage} in {path}")
    if not attention:
        raise RuntimeError(f"missing exact-attention mode records in {path}")
    return rows


def execute(
    args, root, run_dir, name, layer, reference, variant=DEFAULT_VARIANT
):
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    print(f"[{name}] {command_text(args, layer, variant)}", flush=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        completed = subprocess.run(
            command(args),
            cwd=root,
            env=profile_env(layer, variant),
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit {completed.returncode}; see {stderr_path}"
        )
    stdout_data = stdout_path.read_bytes()
    if reference is not None and stdout_data != reference:
        raise RuntimeError(
            f"{name} output differs from exact reference; see {stdout_path}"
        )
    records = [] if layer is None else parse_profile(
        stderr_path.read_bytes(), layer, stderr_path, variant
    )
    run = {
        "name": name,
        "variant": variant,
        "layer": "" if layer is None else layer,
        "wall_seconds": wall_seconds,
        "stdout_sha256": layer_profile.sha256(stdout_data),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    return stdout_data, records, run


def component_summary(rows, normalize_rows=False):
    values = [
        row["ms"] / row["tokens"] if normalize_rows else row["ms"]
        for row in rows
    ]
    ordered = sorted(values)
    return {
        "records": len(rows),
        "unique_positions": len({row["pos"] for row in rows}),
        "min_position": min(row["pos"] for row in rows),
        "max_position": max(row["pos"] for row in rows),
        "mean_ms_per_row": statistics.mean(values),
        "median_ms_per_row": statistics.median(values),
        "p90_ms_per_row": ordered[int(0.9 * (len(ordered) - 1))],
        "max_ms_per_row": max(values),
    }


def summarize(records, variant=DEFAULT_VARIANT):
    records = [row for row in records if row["variant"] == variant]
    summary = {"variant": variant, "layers": {}}
    for layer in sorted({row["layer"] for row in records}):
        selected = [row for row in records if row["layer"] == layer]
        exact = [row for row in selected if row["part"] == "exact"]
        attention = [row for row in selected if row["part"] == "attention"]

        controls = {}
        signatures = set()
        for stage in CONTROL_STAGES:
            stage_rows = [row for row in exact if row["stage"] == stage]
            controls[stage] = component_summary(stage_rows, normalize_rows=True)
            signatures.add(tuple((row["pos"], row["tokens"]) for row in stage_rows))
        if len(signatures) != 1:
            raise RuntimeError(f"layer {layer} has mismatched control batches")
        proposal_signature = next(iter(signatures))
        expected_positions = [
            pos + offset
            for pos, width in proposal_signature
            for offset in range(width)
        ]
        if any(row["tokens"] != 1 for row in attention):
            raise RuntimeError(f"layer {layer} attention contains non-row records")
        if Counter(row["pos"] for row in attention) != Counter(expected_positions):
            raise RuntimeError(
                f"layer {layer} attention rows do not match proposal batches"
            )

        modes = {}
        for mode in ATTENTION_MODES:
            mode_rows = [row for row in attention if row["stage"] == mode]
            modes[mode] = component_summary(mode_rows) if mode_rows else None
        dense = modes["dense_mixed"]
        sparse = modes["sparse_indexed"]
        ratio = None
        delta_percent = None
        if dense is not None and sparse is not None:
            ratio = (
                sparse["median_ms_per_row"] / dense["median_ms_per_row"]
                if dense["median_ms_per_row"] != 0
                else None
            )
            if ratio is not None:
                delta_percent = (ratio - 1.0) * 100.0
        summary["layers"][str(layer)] = {
            "profiled_batches": len(proposal_signature),
            "profiled_rows": len(expected_positions),
            "proposal_signature": [
                {"pos": pos, "tokens": width}
                for pos, width in proposal_signature
            ],
            "controls": controls,
            "modes": modes,
            "sparse_to_dense_median_ratio": ratio,
            "sparse_delta_percent": delta_percent,
        }
    return summary


def median_ratio(candidate, default):
    if candidate is None or default is None or default == 0:
        return None
    return candidate / default


def compare_variants(records, variants):
    summaries = {variant: summarize(records, variant) for variant in variants}
    comparison = {"variants": summaries, "layers": {}}
    default_summary = summaries[DEFAULT_VARIANT]
    direct_summary = summaries[DIRECT_VARIANT]
    if set(default_summary["layers"]) != set(direct_summary["layers"]):
        raise RuntimeError("default and RB16-direct profiled layer sets differ")

    for layer in default_summary["layers"]:
        default_item = default_summary["layers"][layer]
        direct_item = direct_summary["layers"][layer]
        if default_item["proposal_signature"] != direct_item["proposal_signature"]:
            raise RuntimeError(
                f"layer {layer} default and RB16-direct proposal schedules differ"
            )
        signatures = {}
        for variant in variants:
            signatures[variant] = Counter(
                (row["pos"], row["stage"])
                for row in records
                if row["variant"] == variant
                and str(row["layer"]) == layer
                and row["part"] == "attention"
            )
        if signatures[DEFAULT_VARIANT] != signatures[DIRECT_VARIANT]:
            raise RuntimeError(
                f"layer {layer} default and RB16-direct attention modes differ"
            )

        modes = {}
        for mode in ATTENTION_MODES:
            default_mode = default_item["modes"][mode]
            direct_mode = direct_item["modes"][mode]
            default_ms = (
                None if default_mode is None else default_mode["median_ms_per_row"]
            )
            direct_ms = (
                None if direct_mode is None else direct_mode["median_ms_per_row"]
            )
            ratio = median_ratio(direct_ms, default_ms)
            modes[mode] = {
                "default_ms_per_row": default_ms,
                "rb16_direct_ms_per_row": direct_ms,
                "rb16_direct_to_default_ratio": ratio,
                "rb16_direct_delta_percent": (
                    None if ratio is None else (ratio - 1.0) * 100.0
                ),
            }

        controls = {}
        for stage in CONTROL_STAGES:
            default_ms = default_item["controls"][stage]["median_ms_per_row"]
            direct_ms = direct_item["controls"][stage]["median_ms_per_row"]
            ratio = median_ratio(direct_ms, default_ms)
            controls[stage] = {
                "default_ms_per_row": default_ms,
                "rb16_direct_ms_per_row": direct_ms,
                "rb16_direct_to_default_ratio": ratio,
                "rb16_direct_delta_percent": (ratio - 1.0) * 100.0,
            }
        comparison["layers"][layer] = {
            "profiled_batches": default_item["profiled_batches"],
            "profiled_rows": default_item["profiled_rows"],
            "modes": modes,
            "controls": controls,
        }
    return comparison


def value(item):
    return "n/a" if item is None else f"{item['median_ms_per_row']:.3f}"


def report(summary):
    lines = [
        "# DSpark Exact Attention Mode Profile",
        "",
        "Synchronized diagnostic only. Boundaries preserve row order but change scheduling; do not use these values as throughput measurements.",
        "Each mode uses the same immediate before/after attention-call boundary; controls are normalized by proposal rows.",
        "",
        "| layer | batches | rows | raw rows | raw | dense rows | dense mixed | sparse rows | sparse indexed | sparse/dense | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in summary["layers"].items():
        modes = item["modes"]
        raw = modes["raw"]
        dense = modes["dense_mixed"]
        sparse = modes["sparse_indexed"]
        ratio = item["sparse_to_dense_median_ratio"]
        delta = item["sparse_delta_percent"]
        lines.append(
            f"| {layer} | {item['profiled_batches']} | {item['profiled_rows']} | "
            f"{0 if raw is None else raw['records']} | {value(raw)} | "
            f"{0 if dense is None else dense['records']} | {value(dense)} | "
            f"{0 if sparse is None else sparse['records']} | {value(sparse)} | "
            f"{'n/a' if ratio is None else f'{ratio:.3f}x'} | "
            f"{'n/a' if delta is None else f'{delta:+.1f}%'} |"
        )
    lines.extend(["", "Control-stage medians (attention prep / FFN):"])
    for layer, item in summary["layers"].items():
        controls = item["controls"]
        lines.append(
            f"- Layer {layer}: "
            f"{controls['attention_pre_batch']['median_ms_per_row']:.3f} / "
            f"{controls['ffn_batch']['median_ms_per_row']:.3f} ms/row"
        )
    return "\n".join(lines) + "\n"


def comparison_value(item, key):
    value = item[key]
    return "n/a" if value is None else f"{value:.3f}"


def comparison_ratio(item):
    ratio = item["rb16_direct_to_default_ratio"]
    delta = item["rb16_direct_delta_percent"]
    if ratio is None:
        return "n/a", "n/a"
    return f"{ratio:.3f}x", f"{delta:+.1f}%"


def report_comparison(summary):
    lines = [
        "# DSpark Indexed Attention RB16-Direct Attribution",
        "",
        "Synchronized diagnostic only. Boundaries preserve row order but change scheduling; do not use these values as throughput measurements.",
        "Default RB16 and RB16-direct use identical immediate attention-call boundaries and proposal schedules.",
        "",
        "| layer | batches | rows | default dense | direct dense | dense ratio | default sparse | direct sparse | sparse ratio | sparse delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in summary["layers"].items():
        dense = item["modes"]["dense_mixed"]
        sparse = item["modes"]["sparse_indexed"]
        dense_ratio, _ = comparison_ratio(dense)
        sparse_ratio, sparse_delta = comparison_ratio(sparse)
        lines.append(
            f"| {layer} | {item['profiled_batches']} | {item['profiled_rows']} | "
            f"{comparison_value(dense, 'default_ms_per_row')} | "
            f"{comparison_value(dense, 'rb16_direct_ms_per_row')} | "
            f"{dense_ratio} | "
            f"{comparison_value(sparse, 'default_ms_per_row')} | "
            f"{comparison_value(sparse, 'rb16_direct_ms_per_row')} | "
            f"{sparse_ratio} | {sparse_delta} |"
        )
    lines.extend(["", "Control-stage medians (default/direct):"])
    for layer, item in summary["layers"].items():
        prep = item["controls"]["attention_pre_batch"]
        ffn = item["controls"]["ffn_batch"]
        prep_ratio, prep_delta = comparison_ratio(prep)
        ffn_ratio, ffn_delta = comparison_ratio(ffn)
        lines.append(
            f"- Layer {layer}: attention prep "
            f"{prep['default_ms_per_row']:.3f}/{prep['rb16_direct_ms_per_row']:.3f} "
            f"ms/row ({prep_ratio}, {prep_delta}); FFN "
            f"{ffn['default_ms_per_row']:.3f}/{ffn['rb16_direct_ms_per_row']:.3f} "
            f"ms/row ({ffn_ratio}, {ffn_delta})"
        )
    return "\n".join(lines) + "\n"


def cooldown(seconds):
    if seconds > 0:
        print(f"cooldown: {seconds:g}s", flush=True)
        time.sleep(seconds)


def main():
    args, root = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.dspark_model = args.dspark_model.resolve()
    args.prompt_file = args.prompt_file.resolve()
    layer_profile.check_inputs(args, root)
    n_layers = layer_profile.inspect_layer_count(args, root)
    invalid = [layer for layer in args.layers if layer >= n_layers]
    if invalid:
        raise RuntimeError(
            f"profile layers outside model range 0..{n_layers - 1}: "
            + ", ".join(str(layer) for layer in invalid)
        )
    variants = selected_variants(args)

    print(f"reference: {command_text(args)}")
    for layer_index, layer in enumerate(args.layers):
        layer_variants = (
            variants if layer_index % 2 == 0 else tuple(reversed(variants))
        )
        for variant in layer_variants:
            print(
                f"{variant} layer {layer}: "
                f"{command_text(args, layer, variant)}"
            )
    print("Every profiled output must match the unprofiled exact reference byte-for-byte.")
    if args.dry_run:
        print("Dry run only; no model execution performed.")
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_dir
        or root / "speed-bench/local-runs" /
        (
            f"attention-transition-rb16-direct-{stamp}"
            if args.rb16_direct_comparison
            else f"attention-transition-{stamp}"
        )
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "git_commit": layer_profile.run_capture(["git", "rev-parse", "HEAD"], root),
        "git_status_tracked": layer_profile.run_capture(
            ["git", "status", "--porcelain", "--untracked-files=no"], root
        ),
        "platform": platform.platform(),
        "config": {
            "ctx": args.ctx,
            "tokens": args.tokens,
            "layers": args.layers,
            "cooldown": args.cooldown,
            "model_layer_count": n_layers,
            "temperature": 0,
            "seed": 1,
            "synchronized_profile": True,
            "attention_modes": ATTENTION_MODES,
            "control_components": CONTROL_STAGES,
            "variants": variants,
            "rb16_direct_comparison": args.rb16_direct_comparison,
            "require_dense_to_sparse_transition": not args.allow_single_mode,
        },
        "commands": {"reference": command_text(args)} | {
            f"{variant}_layer_{layer}": command_text(args, layer, variant)
            for layer in args.layers
            for variant in variants
        },
        "prompt": {
            "path": str(args.prompt_file),
            "sha256": layer_profile.sha256(args.prompt_file.read_bytes()),
        },
        "cleared_environment_keys": sorted(
            set(layer_profile.cleared_env_keys(os.environ))
            | (set(EXTRA_CLEARED_ENV_KEYS) & set(os.environ))
        ),
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    reference, _, reference_run = execute(
        args, root, run_dir, "reference", None, None
    )
    records = []
    runs = [reference_run]
    cooldown(args.cooldown)
    for layer_index, layer in enumerate(args.layers):
        layer_variants = (
            variants if layer_index % 2 == 0 else tuple(reversed(variants))
        )
        for variant in layer_variants:
            name = f"{variant}.layer-{layer:02d}"
            _, stage_rows, run = execute(
                args, root, run_dir, name, layer, reference, variant
            )
            records.extend(stage_rows)
            runs.append(run)
            cooldown(args.cooldown)

    with (run_dir / "stages.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=(
                "variant", "part", "layer", "pos", "tokens", "stage", "ms",
            ),
        )
        writer.writeheader()
        writer.writerows(records)
    with (run_dir / "runs.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=(
                "name", "variant", "layer", "wall_seconds", "stdout_sha256",
                "stdout_file", "stderr_file",
            ),
        )
        writer.writeheader()
        writer.writerows(runs)

    if args.rb16_direct_comparison:
        summary = compare_variants(records, variants)
        variant_summaries = summary["variants"]
        summary_text = report_comparison(summary)
    else:
        summary = summarize(records)
        variant_summaries = {DEFAULT_VARIANT: summary}
        summary_text = report(summary)
    if not args.allow_single_mode:
        for variant, variant_summary in variant_summaries.items():
            has_transition = any(
                item["modes"]["dense_mixed"] is not None
                and item["modes"]["sparse_indexed"] is not None
                for item in variant_summary["layers"].values()
            )
            if not has_transition:
                raise RuntimeError(
                    f"{variant} did not cross from dense to sparse attention; "
                    f"raw records were retained in {run_dir / 'stages.csv'}"
                )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    metadata["finished_at"] = dt.datetime.now().astimezone().isoformat()
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + summary_text.rstrip())
    print(f"Raw stages: {run_dir / 'stages.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial raw files were retained.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError) as exc:
        print(f"profile failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
