#!/usr/bin/env python3
"""Model-free dependency and cost audit for a truly batched DSpark verifier."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import analyze_dspark_ratio4_materialized_equivalence as ratio4_equivalence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COST = ROOT / (
    "speed-bench/local-runs/"
    "humaneval-cumulative-cost-20260719-225512/summary.json"
)
DEFAULT_LAYER = ROOT / (
    "speed-bench/local-runs/"
    "post-promotion-width-layer-20260719-232840/summary.json"
)
DEFAULT_TAIL = ROOT / (
    "speed-bench/local-runs/"
    "post-promotion-width-tail-20260720-123704/summary.json"
)
DEFAULT_SUFFIX = ROOT / (
    "speed-bench/local-runs/suffix-profile-20260714-131411/summary.json"
)
SAVINGS_GATE_MS_PER_EMITTED = 1.5
EXACT_WIDTHS = (2, 3, 4, 5)


@dataclass(frozen=True)
class Stage:
    name: str
    dependencies: tuple
    execution: str
    ownership: str


STAGES = (
    Stage(
        "input_hc_rows",
        (),
        "already_batched",
        "Exact FFN output from the preceding layer supplies every proposal row.",
    ),
    Stage(
        "attention_pre_batch",
        ("input_hc_rows",),
        "already_batched",
        "HC mixing, normalization, Q/KV projection, RoPE, and compressor projections.",
    ),
    Stage(
        "proposal_state_slab",
        ("attention_pre_batch",),
        "new_batched",
        "Position-indexed raw KV and compressor partial states for all proposal rows.",
    ),
    Stage(
        "compressor_boundary_rows",
        ("proposal_state_slab",),
        "new_batched",
        "Boundary reductions gather only absolute positions visible to that row.",
    ),
    Stage(
        "causal_attention_batch",
        (
            "attention_pre_batch",
            "proposal_state_slab",
            "compressor_boundary_rows",
        ),
        "new_batched",
        "One row grid with per-row raw/compressed key limits and sparse masks.",
    ),
    Stage(
        "inverse_rope_batch",
        ("causal_attention_batch",),
        "new_batched_or_fused",
        "Preserve the existing inverse-RoPE arithmetic while writing batch heads.",
    ),
    Stage(
        "projection_a_batch",
        ("inverse_rope_batch",),
        "existing_retired_component",
        "Reuse the byte-exact batched low projection without serial core capture.",
    ),
    Stage(
        "projection_b_hc_batch",
        ("projection_a_batch", "input_hc_rows"),
        "existing_retired_component",
        "Reuse the byte-exact batched output expansion and HC update.",
    ),
    Stage(
        "ffn_batch",
        ("projection_b_hc_batch",),
        "already_batched",
        "The promoted exact FFN consumes all proposal rows together.",
    ),
    Stage(
        "output_head_batch",
        ("ffn_batch",),
        "already_batched",
        "Intermediate top-1 tokens and final continuation logits for verification.",
    ),
    Stage(
        "acceptance_decision",
        ("output_head_batch",),
        "host_decision",
        "Compare target tokens with the draft and choose the accepted prefix.",
    ),
    Stage(
        "accepted_prefix_publish",
        (
            "proposal_state_slab",
            "compressor_boundary_rows",
            "acceptance_decision",
        ),
        "logical_or_prefix_commit",
        "Publish only the accepted state prefix after target verification.",
    ),
)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc


def topological_order(stages=STAGES):
    by_name = {stage.name: stage for stage in stages}
    if len(by_name) != len(stages):
        raise ValueError("duplicate dependency-graph stage")
    pending = set(by_name)
    resolved = []
    while pending:
        ready = sorted(
            name
            for name in pending
            if all(dep in resolved for dep in by_name[name].dependencies)
        )
        if not ready:
            missing = {
                dep
                for name in pending
                for dep in by_name[name].dependencies
                if dep not in by_name
            }
            if missing:
                raise ValueError(f"unknown graph dependencies: {sorted(missing)}")
            raise ValueError("cyclic dependency graph")
        for name in ready:
            pending.remove(name)
            resolved.append(name)
    return resolved


def causal_schedule(start_position, width, compress_ratio):
    if width not in EXACT_WIDTHS:
        raise ValueError("exact verifier width must be 2 through 5")
    if compress_ratio not in (4, 128):
        raise ValueError("compress ratio must be 4 or 128")
    positions = tuple(range(start_position, start_position + width))
    rows = []
    for row, position in enumerate(positions):
        visible = positions[:row + 1]
        hidden = positions[row + 1:]
        compressed = tuple(
            candidate
            for candidate in visible
            if (candidate + 1) % compress_ratio == 0
        )
        rows.append({
            "row": row,
            "position": position,
            "visible_proposal_positions": visible,
            "hidden_future_positions": hidden,
            "new_compressed_boundaries": compressed,
        })
    return rows


def validate_causal_schedule(rows, compress_ratio):
    positions = tuple(row["position"] for row in rows)
    for row in rows:
        position = row["position"]
        visible = row["visible_proposal_positions"]
        hidden = row["hidden_future_positions"]
        boundaries = row["new_compressed_boundaries"]
        if visible + hidden != positions:
            raise ValueError("proposal visibility does not partition the batch")
        if any(candidate > position for candidate in visible):
            raise ValueError("future proposal state is visible to attention")
        if any(candidate <= position for candidate in hidden):
            raise ValueError("causal proposal state is hidden from attention")
        if any(candidate not in visible for candidate in boundaries):
            raise ValueError("compressed boundary reads a future proposal")
        if any((candidate + 1) % compress_ratio for candidate in boundaries):
            raise ValueError("invalid compressed boundary")
    return True


def audit_all_schedules():
    scenarios = 0
    rows = 0
    boundaries = 0
    for ratio in (4, 128):
        for phase in range(ratio):
            start = 2 * ratio + phase
            for width in EXACT_WIDTHS:
                schedule = causal_schedule(start, width, ratio)
                validate_causal_schedule(schedule, ratio)
                scenarios += 1
                rows += len(schedule)
                boundaries += sum(
                    len(row["new_compressed_boundaries"])
                    for row in schedule
                )
    return {
        "scenarios": scenarios,
        "rows": rows,
        "boundary_visibility_checks": boundaries,
    }


def validate_inputs(cost, layer, tail, suffix):
    if cost.get("analysis") != "dspark_humaneval_cumulative_exact_verifier_cost":
        raise ValueError("unexpected cumulative-cost analysis")
    if cost.get("threshold") != "0.75" or cost.get("task_count") != 32:
        raise ValueError("cumulative-cost protocol mismatch")
    if layer.get("analysis") != "dspark_post_promotion_width_stratified_exact_layer":
        raise ValueError("unexpected width-layer analysis")
    if layer.get("task") != "humaneval_079" or layer.get("widths") != [2, 3, 4, 5]:
        raise ValueError("width-layer protocol mismatch")
    if tail.get("analysis") != "dspark_post_promotion_width_stratified_attention_tail":
        raise ValueError("unexpected serial-tail analysis")
    if tail.get("task") != "humaneval_079" or tail.get("layer") != 42:
        raise ValueError("serial-tail protocol mismatch")
    if "42" not in suffix.get("layers", {}):
        raise ValueError("suffix evidence has no layer 42")


def build_cost_model(cost, layer, tail, suffix):
    aggregate = cost["aggregate"]
    verifier_widths = cost["verifier_widths"]
    multi_width_target_share = sum(
        verifier_widths[str(width)]["target_time_share"]
        for width in EXACT_WIDTHS
    )

    amortization = layer["stage_amortization"]
    sampled_width5 = {
        name: values["width5_ms_per_row"]
        for name, values in amortization.items()
    }
    sampled_layer_total = sum(sampled_width5.values())
    tail_stage_share = (
        sampled_width5["attention_tail_serial"] / sampled_layer_total
    )
    multi_width_tail_budget = (
        aggregate["target_ms_per_emitted"]
        * multi_width_target_share
        * tail_stage_share
    )

    components = tail["width_results"]["5"]["components"]
    component_costs = {
        name: values["median_ms_per_row"]
        for name, values in components.items()
    }
    component_sum = sum(component_costs.values())
    current_projection = (
        component_costs["projection_a"] + component_costs["projection_b_hc"]
    )
    suffix_layer = suffix["layers"]["42"]
    measured_batch_projection = (
        suffix_layer["candidate_projection_a_ms_per_row"]
        + suffix_layer["candidate_projection_b_hc_ms_per_row"]
    )
    projection_reduction = max(
        0.0, (current_projection - measured_batch_projection) / component_sum
    )
    current_projection_share = current_projection / component_sum
    non_projection_share = 1.0 - current_projection_share
    projection_implied_savings = multi_width_tail_budget * projection_reduction

    deficit = aggregate["runtime_deficit_ms_per_emitted"]
    gate_tail_reduction = SAVINGS_GATE_MS_PER_EMITTED / multi_width_tail_budget
    parity_tail_reduction = deficit / multi_width_tail_budget
    additional_tail_reduction = max(
        0.0, parity_tail_reduction - projection_reduction
    )
    additional_non_projection_reduction = (
        additional_tail_reduction / non_projection_share
        if non_projection_share else 0.0
    )

    return {
        "baseline_ms_per_emitted": aggregate["baseline_ms_per_emitted"],
        "runtime_ms_per_emitted": aggregate["runtime_ms_per_emitted"],
        "deficit_ms_per_emitted": deficit,
        "target_ms_per_emitted": aggregate["target_ms_per_emitted"],
        "multi_width_target_share": multi_width_target_share,
        "sampled_width5_stage_ms_per_row": sampled_width5,
        "sampled_width5_tail_share": tail_stage_share,
        "multi_width_tail_budget_ms_per_emitted": multi_width_tail_budget,
        "tail_component_ms_per_row": component_costs,
        "tail_component_normalizer_ms_per_row": component_sum,
        "current_projection_ms_per_row": current_projection,
        "current_projection_tail_share": current_projection_share,
        "non_projection_tail_share": non_projection_share,
        "measured_batch_projection_ms_per_row": measured_batch_projection,
        "projection_directional_tail_reduction": projection_reduction,
        "projection_directional_savings_ms_per_emitted": (
            projection_implied_savings
        ),
        "gate_required_tail_reduction": gate_tail_reduction,
        "parity_required_tail_reduction": parity_tail_reduction,
        "additional_tail_reduction_for_parity": additional_tail_reduction,
        "additional_non_projection_reduction_for_parity": (
            additional_non_projection_reduction
        ),
    }


def feasibility_gate(graph_batchable, ratio4_pass, modeled_savings,
                     threshold=SAVINGS_GATE_MS_PER_EMITTED):
    return (
        "PROCEED_SHADOW_PROTOTYPE"
        if graph_batchable and ratio4_pass and modeled_savings >= threshold
        else "STOP"
    )


def analyze_summaries(cost, layer, tail, suffix, artifact_paths=None):
    validate_inputs(cost, layer, tail, suffix)

    graph_order = topological_order()
    schedule_audit = audit_all_schedules()
    ratio4_summary = ratio4_equivalence.run_equivalence()
    model = build_cost_model(cost, layer, tail, suffix)
    graph_batchable = "causal_attention_batch" in graph_order and not any(
        "previous_attention_output" in stage.dependencies
        for stage in STAGES
    )
    gate = feasibility_gate(
        graph_batchable,
        ratio4_summary["representation_gate"] == "PASS",
        model["projection_directional_savings_ms_per_emitted"],
    )

    return {
        "analysis": "dspark_batched_verifier_feasibility",
        "artifact_paths": artifact_paths or {},
        "exact_widths": list(EXACT_WIDTHS),
        "dependency_order": graph_order,
        "stages": [stage.__dict__ for stage in STAGES],
        "causal_schedule_audit": schedule_audit,
        "ratio4_equivalence_digest": ratio4_summary["digest"],
        "cost_model": model,
        "savings_gate_ms_per_emitted": SAVINGS_GATE_MS_PER_EMITTED,
        "feasibility_gate": gate,
        "production_prerequisites": [
            "Prove ratio-128 materialized-state source-order equivalence.",
            "Prove row-specific raw/compressed visibility against the exact path.",
            "Write attention output directly to batch_heads without core capture.",
            "Preserve prefix-checkpoint state for every possible accepted width.",
            "Pass a single-layer shadow observer before any throughput ablation.",
        ],
        "interpretation_limits": [
            "Synchronized profile medians are normalized into shares; they are not added as wall time.",
            "Width-5 layer shares are extrapolated only to multi-row target time, not width 1.",
            "Projection evidence comes from an older synchronized suffix profile and is directional.",
            "The retired suffix candidate kept attention serial and does not benchmark this design.",
            "The modeled saving is a feasibility signal, not a throughput prediction.",
        ],
    }


def analyze(cost_path=DEFAULT_COST, layer_path=DEFAULT_LAYER,
            tail_path=DEFAULT_TAIL, suffix_path=DEFAULT_SUFFIX):
    return analyze_summaries(
        _load_json(cost_path),
        _load_json(layer_path),
        _load_json(tail_path),
        _load_json(suffix_path),
        artifact_paths={
            "cost": str(cost_path),
            "layer": str(layer_path),
            "tail": str(tail_path),
            "suffix": str(suffix_path),
        },
    )


def _pct(value):
    return f"{100.0 * value:.1f}%"


def render_report(summary):
    model = summary["cost_model"]
    lines = [
        "# DSpark Batched-Verifier Feasibility",
        "",
        "Model-free dependency and cost audit; no model process or benchmark was run.",
        "",
        "## Gate",
        "",
        f"**{summary['feasibility_gate']}**",
        "",
        f"- Required credible saving: {summary['savings_gate_ms_per_emitted']:.3f} ms/emitted.",
        "- Multi-row serial-tail budget: "
        f"{model['multi_width_tail_budget_ms_per_emitted']:.3f} ms/emitted.",
        "- Directional projection evidence: "
        f"{model['projection_directional_savings_ms_per_emitted']:.3f} ms/emitted.",
        "- Tail reduction needed for the gate: "
        f"{_pct(model['gate_required_tail_reduction'])}.",
        "- Tail reduction needed for parity: "
        f"{_pct(model['parity_required_tail_reduction'])}.",
        "- Additional tail reduction beyond projection evidence: "
        f"{_pct(model['additional_tail_reduction_for_parity'])}.",
        "- Equivalent reduction required from the non-projection group: "
        f"{_pct(model['additional_non_projection_reduction_for_parity'])}.",
        "",
        "## Design",
        "",
        "1. Keep the current batched HC/Q/KV/compressor projections.",
        "2. Stage proposal raw KV and compressor partials by absolute position.",
        "3. Reduce ratio-4/128 boundary rows from the staged proposal slab.",
        "4. Run one causal attention row grid with per-row cache visibility.",
        "5. Fuse or batch inverse RoPE and write directly to `batch_heads`.",
        "6. Reuse the exact batched projections and FFN, then publish only the accepted prefix.",
        "",
        "The old suffix candidate is not this design: it retained serial attention, "
        "split the fused one-row core, and added capture/direct-write overhead before "
        "batching only the projections.",
        "",
        "## Prerequisites",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["production_prerequisites"])
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in summary["interpretation_limits"])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, default=DEFAULT_COST)
    parser.add_argument("--layer", type=Path, default=DEFAULT_LAYER)
    parser.add_argument("--tail", type=Path, default=DEFAULT_TAIL)
    parser.add_argument("--suffix", type=Path, default=DEFAULT_SUFFIX)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = analyze(args.cost, args.layer, args.tail, args.suffix)
    report = render_report(summary)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    if args.report_out:
        args.report_out.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
