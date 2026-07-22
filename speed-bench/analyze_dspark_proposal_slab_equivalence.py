#!/usr/bin/env python3
"""Model-free DSpark ratio-128 and proposal-slab equivalence proof.

The streaming references mirror ds4's mutable compressor frontiers. The
materialized references store partial states by absolute token position. A
proposal slab stages all rows without mutating persistent state, exposes only
the causal prefix for each proposal row, and publishes only the accepted
prefix. Rejected positions are then overwritten with different continuation
data to detect latent state leakage.
"""

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import analyze_dspark_ratio4_materialized_equivalence as ratio4


RATIO128 = 128
DEFAULT_HEAD_DIM = 3
RAW_CAPACITY = 7
PROPOSAL_WIDTHS = (2, 3, 4, 5)


def synthetic_ape128(head_dim=DEFAULT_HEAD_DIM):
    return tuple(
        tuple(
            ratio4.f32(((phase + 5) * (col + 7) % 37 - 18) / 23.0)
            for col in range(head_dim)
        )
        for phase in range(RATIO128)
    )


def synthetic_partial128(position, head_dim=DEFAULT_HEAD_DIM, salt=0):
    kv = tuple(
        ratio4.f32(
            ((position + 11 + salt * 3) * (col + 5) % 43 - 21) / 17.0
        )
        for col in range(head_dim)
    )
    score = tuple(
        ratio4.f32(
            ((position + 13 + salt * 5) * (col + 9) % 41 - 20) / 19.0
        )
        for col in range(head_dim)
    )
    return ratio4.PartialState(position, kv, score)


def synthetic_ape(compress_ratio, head_dim):
    if compress_ratio == ratio4.RATIO:
        return ratio4.synthetic_ape(head_dim)
    if compress_ratio == RATIO128:
        return synthetic_ape128(head_dim)
    raise ValueError(f"unsupported compressor ratio: {compress_ratio}")


def synthetic_partial(position, compress_ratio, head_dim, salt=0):
    if compress_ratio == ratio4.RATIO:
        if salt == 0:
            return ratio4.synthetic_partial(position, head_dim)
        width = ratio4.LANES * head_dim
        kv = tuple(
            ratio4.f32(
                ((position + 3 + salt * 7) * (col + 5) % 47 - 23) / 13.0
            )
            for col in range(width)
        )
        score = tuple(
            ratio4.f32(
                ((position + 7 + salt * 11) * (col + 2) % 43 - 21)
                / 17.0
            )
            for col in range(width)
        )
        return ratio4.PartialState(position, kv, score)
    if compress_ratio == RATIO128:
        return synthetic_partial128(position, head_dim, salt)
    raise ValueError(f"unsupported compressor ratio: {compress_ratio}")


def stored_partial128(partial, ape):
    phase = ape[partial.position % RATIO128]
    return (
        tuple(ratio4.f32(value) for value in partial.kv),
        tuple(
            ratio4.add_f32(value, phase[col])
            for col, value in enumerate(partial.score)
        ),
    )


def stream_pool128(rows, head_dim):
    """Reduce physical rows in ds4's two-pass ascending source order."""
    output = []
    for col in range(head_dim):
        maximum = ratio4.f32(ratio4.NEG_INF)
        for _kv, score in rows:
            maximum = max(maximum, score[col])
        denominator = ratio4.f32(0.0)
        total = ratio4.f32(0.0)
        for kv, score in rows:
            weight = ratio4.exp_f32(ratio4.f32(score[col] - maximum))
            denominator = ratio4.add_f32(denominator, weight)
            total = ratio4.add_f32(
                total, ratio4.mul_f32(kv[col], weight)
            )
        output.append(ratio4.f32(total / denominator))
    return tuple(output)


def materialized_pool128(rows, head_dim):
    """Independent two-pass reduction over chronological absolute rows."""
    result = []
    for dimension in range(head_dim):
        maximum = ratio4.f32(ratio4.NEG_INF)
        for index in range(len(rows)):
            candidate = rows[index][1][dimension]
            if candidate > maximum:
                maximum = candidate
        normalizer = ratio4.f32(0.0)
        weighted = ratio4.f32(0.0)
        for index in range(len(rows)):
            score = rows[index][1][dimension]
            weight = ratio4.exp_f32(ratio4.f32(score - maximum))
            normalizer = ratio4.f32(normalizer + weight)
            product = ratio4.f32(rows[index][0][dimension] * weight)
            weighted = ratio4.f32(weighted + product)
        result.append(ratio4.f32(weighted / normalizer))
    return tuple(result)


class StreamingRatio128:
    """ds4's mutable modulo-128 compressor frontier."""

    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.kv = []
        self.score = []
        for _ in range(RATIO128):
            kv, score = ratio4.empty_row(head_dim)
            self.kv.append(kv)
            self.score.append(score)

    def frontier_rows(self):
        return tuple(
            (tuple(self.kv[row]), tuple(self.score[row]))
            for row in range(RATIO128)
        )

    def update(self, partial, ape):
        kv, score = stored_partial128(partial, ape)
        row = partial.position % RATIO128
        self.kv[row][:] = kv
        self.score[row][:] = score
        if (partial.position + 1) % RATIO128:
            return None
        rows = self.frontier_rows()
        return ratio4.Emission(
            partial.position,
            rows,
            stream_pool128(rows, self.head_dim),
        )


class MaterializedRatio128:
    """Absolute-position ratio-128 partial-state cache."""

    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.cache = {}

    def _row(self, position):
        if position not in self.cache:
            kv, score = ratio4.empty_row(self.head_dim)
            return tuple(kv), tuple(score)
        return self.cache[position]

    def update(self, partial, ape):
        self.cache[partial.position] = stored_partial128(partial, ape)
        if (partial.position + 1) % RATIO128:
            return None
        rows = self.compression_rows(partial.position)
        return ratio4.Emission(
            partial.position,
            rows,
            materialized_pool128(rows, self.head_dim),
        )

    def compression_rows(self, boundary):
        start = boundary - RATIO128 + 1
        return tuple(self._row(position) for position in range(start, boundary + 1))

    def frontier_rows(self, position):
        rows = []
        for physical_row in range(RATIO128):
            absolute = position - ((position - physical_row) % RATIO128)
            rows.append(self._row(absolute))
        return tuple(rows)


def make_streaming(compress_ratio, head_dim):
    if compress_ratio == ratio4.RATIO:
        return ratio4.StreamingRatio4(head_dim)
    if compress_ratio == RATIO128:
        return StreamingRatio128(head_dim)
    raise ValueError(f"unsupported compressor ratio: {compress_ratio}")


def make_materialized(compress_ratio, head_dim):
    if compress_ratio == ratio4.RATIO:
        return ratio4.MaterializedRatio4(head_dim)
    if compress_ratio == RATIO128:
        return MaterializedRatio128(head_dim)
    raise ValueError(f"unsupported compressor ratio: {compress_ratio}")


def raw_row(partial, head_dim):
    return tuple(
        ratio4.f32(
            partial.kv[col] + partial.score[(col * 3) % len(partial.score)]
        )
        for col in range(head_dim)
    )


def raw_slots_bits(slots):
    result = bytearray()
    for entry in slots:
        if entry is None:
            result.extend((-1).to_bytes(8, "little", signed=True))
            continue
        position, values = entry
        result.extend(position.to_bytes(8, "little", signed=True))
        result.extend(ratio4.values_bits(values))
    return bytes(result)


def compressed_bits(emissions):
    result = bytearray()
    for emission in emissions:
        result.extend(emission.position.to_bytes(8, "little", signed=True))
        result.extend(ratio4.values_bits(emission.pooled))
    return bytes(result)


@dataclass(frozen=True)
class StateSnapshot:
    logical_position: int
    raw_physical: bytes
    raw_visible: tuple
    compressor_frontier: bytes
    compressed: bytes
    n_compressed: int


class StreamingPersistentState:
    def __init__(self, compress_ratio, head_dim, raw_capacity=RAW_CAPACITY):
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.raw_capacity = raw_capacity
        self.ape = synthetic_ape(compress_ratio, head_dim)
        self.compressor = make_streaming(compress_ratio, head_dim)
        self.raw_slots = [None] * raw_capacity
        self.compressed = []
        self.history = {}
        self.logical_position = -1

    def apply(self, partial):
        if partial.position != self.logical_position + 1:
            raise AssertionError("persistent updates must be contiguous")
        self.history[partial.position] = partial
        self.raw_slots[partial.position % self.raw_capacity] = (
            partial.position,
            raw_row(partial, self.head_dim),
        )
        emission = self.compressor.update(partial, self.ape)
        if emission is not None:
            self.compressed.append(emission)
        self.logical_position = partial.position

    def snapshot(self):
        minimum = self.logical_position - self.raw_capacity + 1
        visible = tuple(
            position
            for position, _values in sorted(
                entry for entry in self.raw_slots if entry is not None
            )
            if minimum <= position <= self.logical_position
        )
        return StateSnapshot(
            self.logical_position,
            raw_slots_bits(self.raw_slots),
            visible,
            ratio4.rows_bits(self.compressor.frontier_rows()),
            compressed_bits(self.compressed),
            len(self.compressed),
        )


class MaterializedPersistentState:
    def __init__(
        self,
        compress_ratio,
        head_dim,
        history,
        raw_slots,
        compressed,
        logical_position,
        raw_capacity=RAW_CAPACITY,
    ):
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.raw_capacity = raw_capacity
        self.ape = synthetic_ape(compress_ratio, head_dim)
        self.compressor = make_materialized(compress_ratio, head_dim)
        self.history = dict(history)
        for position in sorted(self.history):
            self.compressor.update(self.history[position], self.ape)
        self.raw_slots = copy.deepcopy(raw_slots)
        self.compressed = list(compressed)
        self.logical_position = logical_position

    def apply(self, partial):
        if partial.position != self.logical_position + 1:
            raise AssertionError("materialized updates must be contiguous")
        self.history[partial.position] = partial
        self.raw_slots[partial.position % self.raw_capacity] = (
            partial.position,
            raw_row(partial, self.head_dim),
        )
        emission = self.compressor.update(partial, self.ape)
        if emission is not None:
            self.compressed.append(emission)
        self.logical_position = partial.position

    def snapshot(self):
        minimum = self.logical_position - self.raw_capacity + 1
        visible = tuple(
            position
            for position, _values in sorted(
                entry for entry in self.raw_slots if entry is not None
            )
            if minimum <= position <= self.logical_position
        )
        return StateSnapshot(
            self.logical_position,
            raw_slots_bits(self.raw_slots),
            visible,
            ratio4.rows_bits(
                self.compressor.frontier_rows(self.logical_position)
            ),
            compressed_bits(self.compressed),
            len(self.compressed),
        )


class ProposalSlab:
    """Logical staged proposal state with prefix-scoped visibility."""

    def __init__(self, base, proposals):
        self.base = base
        self.proposals = tuple(proposals)
        self.materialized = make_materialized(
            base.compress_ratio, base.head_dim
        )
        for position in sorted(base.history):
            self.materialized.update(base.history[position], base.ape)
        self.staged_emissions = []
        for partial in self.proposals:
            emission = self.materialized.update(partial, base.ape)
            if emission is not None:
                self.staged_emissions.append(emission)

    def _prefix_parts(self, accepted):
        if not 0 <= accepted <= len(self.proposals):
            raise ValueError("accepted prefix is outside the proposal slab")
        history = dict(self.base.history)
        slots = copy.deepcopy(self.base.raw_slots)
        for partial in self.proposals[:accepted]:
            history[partial.position] = partial
            slots[partial.position % self.base.raw_capacity] = (
                partial.position,
                raw_row(partial, self.base.head_dim),
            )
        limit = self.base.logical_position + accepted
        compressed = list(self.base.compressed)
        compressed.extend(
            emission
            for emission in self.staged_emissions
            if emission.position <= limit
        )
        return history, slots, compressed, limit

    def view(self, prefix_length):
        history, slots, compressed, limit = self._prefix_parts(prefix_length)
        minimum = limit - self.base.raw_capacity + 1
        visible = tuple(
            position
            for position, _values in sorted(
                entry for entry in slots if entry is not None
            )
            if minimum <= position <= limit
        )
        return StateSnapshot(
            limit,
            raw_slots_bits(slots),
            visible,
            ratio4.rows_bits(self.materialized.frontier_rows(limit)),
            compressed_bits(compressed),
            len(compressed),
        )

    def commit(self, accepted):
        history, slots, compressed, limit = self._prefix_parts(accepted)
        return MaterializedPersistentState(
            self.base.compress_ratio,
            self.base.head_dim,
            history,
            slots,
            compressed,
            limit,
            self.base.raw_capacity,
        )


def seed_streaming(compress_ratio, phase, head_dim):
    state = StreamingPersistentState(compress_ratio, head_dim)
    start = 2 * compress_ratio + phase
    for position in range(start):
        state.apply(synthetic_partial(position, compress_ratio, head_dim))
    return state, start


def assert_same_snapshot(left, right, context):
    if left != right:
        raise AssertionError(f"persistent state mismatch: {context}")


def run_ratio128_equivalence(head_dim=DEFAULT_HEAD_DIM):
    ape = synthetic_ape128(head_dim)
    frontier_checks = 0
    boundary_checks = 0
    digest = hashlib.sha256()

    for width in PROPOSAL_WIDTHS:
        for phase in range(RATIO128):
            start = 2 * RATIO128 + phase
            stream = StreamingRatio128(head_dim)
            materialized = MaterializedRatio128(head_dim)
            for position in range(start):
                partial = synthetic_partial128(position, head_dim)
                stream.update(partial, ape)
                materialized.update(partial, ape)
            for position in range(start, start + width):
                partial = synthetic_partial128(position, head_dim)
                stream_emission = stream.update(partial, ape)
                materialized_emission = materialized.update(partial, ape)
                stream_rows = stream.frontier_rows()
                materialized_rows = materialized.frontier_rows(position)
                if ratio4.rows_bits(stream_rows) != ratio4.rows_bits(
                    materialized_rows
                ):
                    raise AssertionError(
                        f"ratio-128 frontier mismatch width={width} "
                        f"phase={phase} position={position}"
                    )
                frontier_checks += 1
                digest.update(ratio4.rows_bits(stream_rows))
                if (stream_emission is None) != (
                    materialized_emission is None
                ):
                    raise AssertionError("ratio-128 boundary mismatch")
                if stream_emission is None:
                    continue
                if ratio4.rows_bits(stream_emission.rows) != ratio4.rows_bits(
                    materialized_emission.rows
                ):
                    raise AssertionError("ratio-128 source rows mismatch")
                if ratio4.values_bits(
                    stream_emission.pooled
                ) != ratio4.values_bits(materialized_emission.pooled):
                    raise AssertionError("ratio-128 reduction mismatch")
                boundary_checks += 1
                digest.update(ratio4.values_bits(stream_emission.pooled))

    return {
        "scenario_count": RATIO128 * len(PROPOSAL_WIDTHS),
        "frontier_bitwise_checks": frontier_checks,
        "boundary_row_and_reduction_checks": boundary_checks,
        "digest": digest.hexdigest(),
        "gate": "PASS_SOURCE_ORDER",
    }


def run_proposal_slab_equivalence(head_dim=DEFAULT_HEAD_DIM):
    slab_scenarios = 0
    row_views = 0
    commit_cases = 0
    continuation_checks = 0
    rejected_boundary_checks = 0
    digest = hashlib.sha256()

    for compress_ratio in (ratio4.RATIO, RATIO128):
        for phase in range(compress_ratio):
            base, start = seed_streaming(compress_ratio, phase, head_dim)
            for width in PROPOSAL_WIDTHS:
                proposals = tuple(
                    synthetic_partial(
                        position, compress_ratio, head_dim
                    )
                    for position in range(start, start + width)
                )
                slab = ProposalSlab(base, proposals)
                serial_prefix = copy.deepcopy(base)
                for prefix_length, partial in enumerate(proposals, start=1):
                    serial_prefix.apply(partial)
                    candidate_view = slab.view(prefix_length)
                    assert_same_snapshot(
                        candidate_view,
                        serial_prefix.snapshot(),
                        f"view ratio={compress_ratio} phase={phase} "
                        f"width={width} prefix={prefix_length}",
                    )
                    if any(
                        position > candidate_view.logical_position
                        for position in candidate_view.raw_visible
                    ):
                        raise AssertionError("future raw position became visible")
                    row_views += 1
                    digest.update(candidate_view.compressor_frontier)

                for accepted in range(width + 1):
                    expected = copy.deepcopy(base)
                    for partial in proposals[:accepted]:
                        expected.apply(partial)
                    committed = slab.commit(accepted)
                    committed_snapshot = committed.snapshot()
                    assert_same_snapshot(
                        committed_snapshot,
                        expected.snapshot(),
                        f"commit ratio={compress_ratio} phase={phase} "
                        f"width={width} accepted={accepted}",
                    )
                    rejected_boundaries = [
                        emission.position
                        for emission in slab.staged_emissions
                        if emission.position
                        > committed_snapshot.logical_position
                    ]
                    if rejected_boundaries:
                        rejected_boundary_checks += 1
                        if any(
                            emission.position in rejected_boundaries
                            for emission in committed.compressed
                        ):
                            raise AssertionError(
                                "rejected compressed boundary was published"
                            )

                    for offset in range(2):
                        position = start + accepted + offset
                        replacement = synthetic_partial(
                            position,
                            compress_ratio,
                            head_dim,
                            salt=97 + offset,
                        )
                        expected.apply(replacement)
                        committed.apply(replacement)
                        assert_same_snapshot(
                            committed.snapshot(),
                            expected.snapshot(),
                            f"continuation ratio={compress_ratio} "
                            f"phase={phase} width={width} "
                            f"accepted={accepted} offset={offset}",
                        )
                        continuation_checks += 1
                    commit_cases += 1
                    digest.update(committed.snapshot().raw_physical)
                slab_scenarios += 1

    return {
        "ratios": [ratio4.RATIO, RATIO128],
        "proposal_widths": list(PROPOSAL_WIDTHS),
        "raw_capacity": RAW_CAPACITY,
        "slab_scenario_count": slab_scenarios,
        "causal_row_view_checks": row_views,
        "accepted_prefix_commit_cases": commit_cases,
        "post_rejection_continuation_checks": continuation_checks,
        "rejected_boundary_isolation_checks": rejected_boundary_checks,
        "digest": digest.hexdigest(),
        "visibility_gate": "PASS_CAUSAL_PREFIX",
        "publication_gate": "PASS_ACCEPTED_PREFIX",
        "continuation_gate": "PASS_REJECTED_STATE_ISOLATION",
    }


def run_analysis(head_dim=DEFAULT_HEAD_DIM):
    ratio128_summary = run_ratio128_equivalence(head_dim)
    slab_summary = run_proposal_slab_equivalence(head_dim)
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(ratio128_summary["digest"]))
    digest.update(bytes.fromhex(slab_summary["digest"]))
    return {
        "analysis": "dspark_proposal_slab_equivalence",
        "head_dim": head_dim,
        "ratio128": ratio128_summary,
        "proposal_slab": slab_summary,
        "contract_gate": "PASS_MODEL_FREE_CONTRACT",
        "next_gate": "PROCEED_SINGLE_LAYER_SHADOW_OBSERVER",
        "digest": digest.hexdigest(),
    }


def render_report(summary):
    ratio128_summary = summary["ratio128"]
    slab = summary["proposal_slab"]
    return "\n".join([
        "# DSpark Ratio-128 and Proposal-Slab Equivalence",
        "",
        "Model-free F32 diagnostic; no model process or timing benchmark was run.",
        "",
        "## Ratio-128 Materialized State",
        "",
        f"- Scenarios: {ratio128_summary['scenario_count']}",
        "- Bitwise frontier checks: "
        f"{ratio128_summary['frontier_bitwise_checks']}",
        "- Boundary row/reduction checks: "
        f"{ratio128_summary['boundary_row_and_reduction_checks']}",
        f"- Source-order gate: **{ratio128_summary['gate']}**",
        "",
        "The streaming modulo-128 frontier and absolute-position cache agree "
        "bitwise after every proposal row. Boundary pooling uses the same "
        "ascending 0..127 max pass followed by the same ascending weighted-sum "
        "pass as ds4's generic Metal softmax-pool kernel.",
        "",
        "## Proposal Slab",
        "",
        f"- Ratios: {', '.join(map(str, slab['ratios']))}",
        f"- Proposal widths: {', '.join(map(str, slab['proposal_widths']))}",
        f"- Slab scenarios: {slab['slab_scenario_count']}",
        f"- Causal row-view checks: {slab['causal_row_view_checks']}",
        "- Accepted-prefix commit cases: "
        f"{slab['accepted_prefix_commit_cases']}",
        "- Post-rejection continuation checks: "
        f"{slab['post_rejection_continuation_checks']}",
        "- Rejected-boundary isolation checks: "
        f"{slab['rejected_boundary_isolation_checks']}",
        f"- Visibility gate: **{slab['visibility_gate']}**",
        f"- Publication gate: **{slab['publication_gate']}**",
        f"- Continuation gate: **{slab['continuation_gate']}**",
        "",
        "Every staged row sees exactly its raw/compressed causal prefix. "
        "Publishing any prefix from zero through N reproduces serial raw-ring "
        "slots, compressor frontier, compressed outputs, and counters. Two "
        "different replacement rows are then executed at rejected positions; "
        "their states remain bitwise identical to clean serial execution.",
        "",
        f"- Contract gate: **{summary['contract_gate']}**",
        f"- Next gate: **{summary['next_gate']}**",
        "",
        "This proves the bounded logical ownership contract, not a GPU kernel. "
        "The next implementation may be a single-layer shadow observer only; "
        "it must compare staged Metal state with the production exact row loop "
        "and remain unavailable to normal runtime and throughput runners.",
        "",
        f"Deterministic digest: `{summary['digest']}`",
    ]) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_analysis()
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
