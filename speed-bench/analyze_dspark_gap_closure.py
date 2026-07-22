#!/usr/bin/env python3
"""Quantify how much exact-verifier work must change to reach parity."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COST = ROOT / (
    "speed-bench/local-runs/"
    "humaneval-cumulative-cost-20260722-195022/summary.json"
)
DEFAULT_LAYER = ROOT / (
    "speed-bench/local-runs/"
    "post-promotion-width-layer-20260722-202450/summary.json"
)
DEFAULT_FFN = ROOT / (
    "speed-bench/local-runs/"
    "post-promotion-width-ffn-20260722-202732/summary.json"
)
DEFAULT_TAIL = ROOT / (
    "speed-bench/local-runs/"
    "post-promotion-width-tail-20260722-203133/summary.json"
)
WIDTHS = (1, 2, 3, 4, 5)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc


def validate_inputs(cost, layer, ffn, tail):
    if cost.get("analysis") != "dspark_humaneval_cumulative_exact_verifier_cost":
        raise ValueError("unexpected cumulative-cost analysis")
    if cost.get("threshold") != "0.75" or cost.get("task_count") != 32:
        raise ValueError("cumulative-cost protocol mismatch")
    if set(cost.get("verifier_widths", {})) != {str(width) for width in WIDTHS}:
        raise ValueError("cumulative-cost verifier widths mismatch")
    if layer.get("analysis") != "dspark_post_promotion_width_stratified_exact_layer":
        raise ValueError("unexpected width-layer analysis")
    if layer.get("task") != "humaneval_079" or layer.get("widths") != [2, 3, 4, 5]:
        raise ValueError("width-layer protocol mismatch")
    if ffn.get("analysis") != "dspark_post_promotion_width_stratified_exact_ffn":
        raise ValueError("unexpected width-FFN analysis")
    if ffn.get("task") != "humaneval_079" or ffn.get("widths") != [2, 3, 4, 5]:
        raise ValueError("width-FFN protocol mismatch")
    if tail.get("analysis") != "dspark_post_promotion_width_stratified_attention_tail":
        raise ValueError("unexpected serial-tail analysis")
    if tail.get("task") != "humaneval_079" or tail.get("layer") != 42:
        raise ValueError("serial-tail protocol mismatch")


def _closure(required_target_reduction, target_share):
    required_scope_reduction = (
        required_target_reduction / target_share if target_share else None
    )
    return {
        "target_time_share": target_share,
        "elimination_parity_coverage": (
            target_share / required_target_reduction
            if required_target_reduction else None
        ),
        "required_scope_reduction": required_scope_reduction,
        "can_close_by_elimination": target_share >= required_target_reduction,
    }


def analyze_summaries(cost, layer, ffn, tail, artifact_paths=None):
    validate_inputs(cost, layer, ffn, tail)
    aggregate = cost["aggregate"]
    target_ms = aggregate["target_ms_per_emitted"]
    deficit_ms = aggregate["runtime_deficit_ms_per_emitted"]
    required_target_reduction = deficit_ms / target_ms
    accounted_required_target_reduction = (
        1.0 - aggregate["accounted_target_scale_for_parity"]
    )

    width_shares = {
        width: cost["verifier_widths"][str(width)]["target_time_share"]
        for width in WIDTHS
    }
    width_closure = {
        str(width): _closure(required_target_reduction, share)
        for width, share in width_shares.items()
    }

    width5 = layer["sampled_width_totals"]["5"]
    stage_ms = width5["sampled_stage_ms_per_row"]
    sampled_total = width5["sampled_layer_ms_per_row"]
    stage_shares = {
        name: value / sampled_total for name, value in stage_ms.items()
    }
    stage_closure = {}
    for name, share in stage_shares.items():
        stage_closure[name] = {
            "sampled_width5_ms_per_row": stage_ms[name],
            "sampled_stage_share": share,
            "all_widths": _closure(required_target_reduction, share),
            "width5_only": _closure(
                required_target_reduction, width_shares[5] * share
            ),
        }

    inner_components = {}
    groups = (
        ("ffn", "ffn_batch", ffn["width5_components"]),
        (
            "tail",
            "attention_tail_serial",
            tail["width_results"]["5"]["components"],
        ),
    )
    for group, stage_name, components in groups:
        for name, values in components.items():
            inner_share = values.get("share", values.get("median_share"))
            if inner_share is None:
                raise ValueError(f"missing component share for {group}/{name}")
            all_widths_share = stage_shares[stage_name] * inner_share
            width5_share = width_shares[5] * all_widths_share
            inner_components[f"{group}/{name}"] = {
                "group": group,
                "component": name,
                "within_stage_share": inner_share,
                "all_widths": _closure(
                    required_target_reduction, all_widths_share
                ),
                "width5_only": _closure(
                    required_target_reduction, width5_share
                ),
            }

    credible_inner = [
        name
        for name, values in inner_components.items()
        if values["all_widths"]["can_close_by_elimination"]
        and values["all_widths"]["required_scope_reduction"] <= 0.5
    ]
    decision = (
        "REQUIRE_CROSS_WIDTH_STAGE_OR_VERIFIER_REDESIGN"
        if not credible_inner
        else "INNER_COMPONENT_CANDIDATE_REMAINS"
    )

    return {
        "analysis": "dspark_exact_verifier_gap_closure",
        "artifact_paths": artifact_paths or {},
        "cost": {
            "baseline_ms_per_emitted": aggregate["baseline_ms_per_emitted"],
            "runtime_ms_per_emitted": aggregate["runtime_ms_per_emitted"],
            "deficit_ms_per_emitted": deficit_ms,
            "target_ms_per_emitted": target_ms,
            "required_target_reduction": required_target_reduction,
            "accounted_required_target_reduction": (
                accounted_required_target_reduction
            ),
        },
        "width_target_shares": {str(key): value for key, value in width_shares.items()},
        "width_closure": width_closure,
        "sampled_width5_stage_shares": stage_shares,
        "stage_closure": stage_closure,
        "inner_component_closure": inner_components,
        "credible_inner_components": credible_inner,
        "decision": decision,
        "interpretation_limits": [
            "The end-to-end deficit comes from the frozen 32-task cumulative cost audit.",
            "Stage and component shares come from synchronized task-079 samples and are directional, not full-model wall-time sums.",
            "All-width estimates assume the sampled width-five stage mix is representative of other verifier widths.",
            "Elimination ceilings are impossibly favorable bounds, not speed predictions.",
            "The report screens optimization scope; it does not prove that a cross-stage redesign will be fast or byte-exact.",
        ],
    }


def analyze(cost_path=DEFAULT_COST, layer_path=DEFAULT_LAYER,
            ffn_path=DEFAULT_FFN, tail_path=DEFAULT_TAIL):
    return analyze_summaries(
        _load_json(cost_path),
        _load_json(layer_path),
        _load_json(ffn_path),
        _load_json(tail_path),
        artifact_paths={
            "cost": str(cost_path),
            "layer": str(layer_path),
            "ffn": str(ffn_path),
            "tail": str(tail_path),
        },
    )


def _pct(value):
    return f"{100.0 * value:.1f}%"


def _required(value):
    if value["can_close_by_elimination"]:
        return _pct(value["required_scope_reduction"])
    return "impossible"


def render_report(summary):
    cost = summary["cost"]
    lines = [
        "# DSpark Exact Verifier Gap-Closure Ledger",
        "",
        "Model-free scope audit; no model process or throughput benchmark was run.",
        "",
        "## Measured Gap",
        "",
        f"- Baseline: {cost['baseline_ms_per_emitted']:.3f} ms/emitted.",
        f"- Current DSpark: {cost['runtime_ms_per_emitted']:.3f} ms/emitted.",
        f"- Deficit: {cost['deficit_ms_per_emitted']:.3f} ms/emitted.",
        f"- Target verification: {cost['target_ms_per_emitted']:.3f} ms/emitted.",
        "- Target reduction required for measured end-to-end parity: "
        f"{_pct(cost['required_target_reduction'])}.",
        "- Accounted-cost target reduction required for parity: "
        f"{_pct(cost['accounted_required_target_reduction'])}.",
        "",
        "## Verifier Widths",
        "",
        "| width | target-time share | reduction if this width acts alone |",
        "|---:|---:|---:|",
    ]
    for width in (1, 2, 3, 4, 5):
        closure = summary["width_closure"][str(width)]
        lines.append(
            f"| {width} | {_pct(closure['target_time_share'])} | "
            f"{_required(closure)} |"
        )

    lines.extend([
        "",
        "## Width-Five Stage Scope",
        "",
        "| stage | sampled share | all-width reduction | width-5-only share | width-5-only reduction |",
        "|:---|---:|---:|---:|---:|",
    ])
    for name in (
        "attention_pre_batch", "attention_tail_serial", "ffn_batch"
    ):
        stage = summary["stage_closure"][name]
        lines.append(
            f"| `{name}` | {_pct(stage['sampled_stage_share'])} | "
            f"{_required(stage['all_widths'])} | "
            f"{_pct(stage['width5_only']['target_time_share'])} | "
            f"{_required(stage['width5_only'])} |"
        )

    ranked = sorted(
        summary["inner_component_closure"].items(),
        key=lambda item: item[1]["all_widths"]["target_time_share"],
        reverse=True,
    )
    lines.extend([
        "",
        "## Inner-Component Ceilings",
        "",
        "| component | all-width target share | all-width reduction | width-5-only target share |",
        "|:---|---:|---:|---:|",
    ])
    for name, component in ranked:
        lines.append(
            f"| `{name}` | {_pct(component['all_widths']['target_time_share'])} | "
            f"{_required(component['all_widths'])} | "
            f"{_pct(component['width5_only']['target_time_share'])} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        "No remaining inner FFN or serial-tail component can credibly close the "
        "measured gap by itself. The next candidate must reduce a complete stage "
        "across verifier widths, combine savings across stages, or change the "
        "verifier architecture.",
        "",
        "## Limits",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, default=DEFAULT_COST)
    parser.add_argument("--layer", type=Path, default=DEFAULT_LAYER)
    parser.add_argument("--ffn", type=Path, default=DEFAULT_FFN)
    parser.add_argument("--tail", type=Path, default=DEFAULT_TAIL)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = analyze(args.cost, args.layer, args.ffn, args.tail)
    report = render_report(summary)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
