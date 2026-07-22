#!/usr/bin/env python3
"""Model-free equivalence probe for ratio-4 compressor state layouts.

The streaming reference mirrors ds4's duplicated eight-row frontier.  The
materialized reference mirrors vLLM's position-indexed partial-state cache:
at a compression boundary p it gathers lane 0 from p-7..p-4 and lane 1 from
p-3..p.  Both references retain ds4's scalar eight-row reduction order.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct


RATIO = 4
LANES = 2
NEG_INF = -1.0e30
DEFAULT_HEAD_DIM = 3
PROPOSAL_WIDTHS = (2, 3, 4, 5)
FRONTIER_PHASES = (0, 1, 2, 3)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value):
    return struct.pack("<f", value)


def add_f32(left, right):
    return f32(f32(left) + f32(right))


def mul_f32(left, right):
    return f32(f32(left) * f32(right))


def exp_f32(value):
    return f32(math.exp(f32(value)))


@dataclass(frozen=True)
class PartialState:
    position: int
    kv: tuple
    score: tuple


@dataclass(frozen=True)
class Emission:
    position: int
    rows: tuple
    pooled: tuple


def synthetic_ape(head_dim=DEFAULT_HEAD_DIM):
    width = LANES * head_dim
    return tuple(
        tuple(f32(((phase + 2) * (col + 3) % 13 - 6) / 17.0)
              for col in range(width))
        for phase in range(RATIO)
    )


def synthetic_partial(position, head_dim=DEFAULT_HEAD_DIM):
    width = LANES * head_dim
    kv = tuple(
        f32(((position + 3) * (col + 5) % 29 - 14) / 11.0)
        for col in range(width)
    )
    score = tuple(
        f32(((position + 7) * (col + 2) % 23 - 11) / 13.0)
        for col in range(width)
    )
    return PartialState(position, kv, score)


def stored_partial(partial, ape):
    phase_ape = ape[partial.position % RATIO]
    return (
        tuple(f32(value) for value in partial.kv),
        tuple(add_f32(value, phase_ape[col])
              for col, value in enumerate(partial.score)),
    )


def empty_row(width):
    return ([f32(0.0)] * width, [f32(NEG_INF)] * width)


def rows_bits(rows):
    return b"".join(
        f32_bits(value)
        for kv, score in rows
        for values in (kv, score)
        for value in values
    )


def values_bits(values):
    return b"".join(f32_bits(value) for value in values)


def stream_pool(rows, head_dim):
    """Reduce streaming rows in ds4's packed Metal source order."""
    output = []
    for col in range(head_dim):
        maximum = f32(NEG_INF)
        for _kv, score in rows:
            maximum = max(maximum, score[col])
        denominator = f32(0.0)
        total = f32(0.0)
        for kv, score in rows:
            weight = exp_f32(f32(score[col] - maximum))
            denominator = add_f32(denominator, weight)
            total = add_f32(total, mul_f32(kv[col], weight))
        output.append(f32(total / denominator))
    return tuple(output)


def materialized_pool(rows, head_dim):
    """Independent scalar reduction over a chronological materialized view."""
    result = []
    for dimension in range(head_dim):
        scores = [row[1][dimension] for row in rows]
        maximum = f32(NEG_INF)
        for score in scores:
            if score > maximum:
                maximum = score
        normalizer = f32(0.0)
        weighted = f32(0.0)
        for index in range(len(rows)):
            weight = exp_f32(f32(scores[index] - maximum))
            normalizer = f32(normalizer + weight)
            product = f32(rows[index][0][dimension] * weight)
            weighted = f32(weighted + product)
        result.append(f32(weighted / normalizer))
    return tuple(result)


class StreamingRatio4:
    """The mutable lower/upper frontier used by ds4."""

    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.width = LANES * head_dim
        self.kv = []
        self.score = []
        for _ in range(LANES * RATIO):
            kv, score = empty_row(self.width)
            self.kv.append(kv)
            self.score.append(score)

    def frontier_rows(self):
        return tuple(
            (tuple(self.kv[row]), tuple(self.score[row]))
            for row in range(LANES * RATIO)
        )

    def compression_rows(self):
        rows = []
        for row in range(RATIO):
            rows.append((
                tuple(self.kv[row][:self.head_dim]),
                tuple(self.score[row][:self.head_dim]),
            ))
        for row in range(RATIO, LANES * RATIO):
            rows.append((
                tuple(self.kv[row][self.head_dim:]),
                tuple(self.score[row][self.head_dim:]),
            ))
        return tuple(rows)

    def update(self, partial, ape):
        kv, score = stored_partial(partial, ape)
        row = RATIO + partial.position % RATIO
        self.kv[row][:] = kv
        self.score[row][:] = score
        if (partial.position + 1) % RATIO:
            return None

        rows = self.compression_rows()
        emission = Emission(
            partial.position,
            rows,
            stream_pool(rows, self.head_dim),
        )
        for row in range(RATIO):
            self.kv[row][:] = self.kv[RATIO + row]
            self.score[row][:] = self.score[RATIO + row]
        for row in range(RATIO):
            self.kv[RATIO + row][:] = self.kv[row]
            self.score[RATIO + row][:] = self.score[row]
        return emission


class MaterializedRatio4:
    """Position-indexed [kv, score+APE] partial states."""

    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.width = LANES * head_dim
        self.cache = {}

    def update(self, partial, ape):
        self.cache[partial.position] = stored_partial(partial, ape)
        if (partial.position + 1) % RATIO:
            return None
        rows = self.compression_rows(partial.position)
        return Emission(
            partial.position,
            rows,
            materialized_pool(rows, self.head_dim),
        )

    def _full_row(self, position):
        if position not in self.cache:
            kv, score = empty_row(self.width)
            return tuple(kv), tuple(score)
        return self.cache[position]

    def compression_rows(self, boundary):
        rows = []
        for offset in range(LANES * RATIO):
            position = boundary - (LANES * RATIO - 1) + offset
            kv, score = self._full_row(position)
            lane_start = 0 if offset < RATIO else self.head_dim
            lane_end = lane_start + self.head_dim
            rows.append((kv[lane_start:lane_end], score[lane_start:lane_end]))
        return tuple(rows)

    def frontier_rows(self, position):
        """Reconstruct ds4's duplicated frontier without replaying updates."""
        rows = [self._full_row(-1) for _ in range(LANES * RATIO)]
        completed_end = position - ((position + 1) % RATIO)
        if completed_end >= RATIO - 1:
            block = [
                self._full_row(pos)
                for pos in range(completed_end - RATIO + 1, completed_end + 1)
            ]
            rows[:RATIO] = block
            rows[RATIO:] = block
        for pos in range(max(0, completed_end + 1), position + 1):
            rows[RATIO + pos % RATIO] = self._full_row(pos)
        return tuple(rows)


def run_equivalence(head_dim=DEFAULT_HEAD_DIM):
    ape = synthetic_ape(head_dim)
    state_checks = 0
    emission_checks = 0
    scenarios = []
    digest = hashlib.sha256()

    for width in PROPOSAL_WIDTHS:
        for phase in FRONTIER_PHASES:
            start = 8 + phase
            stream = StreamingRatio4(head_dim)
            materialized = MaterializedRatio4(head_dim)

            for position in range(start):
                partial = synthetic_partial(position, head_dim)
                stream.update(partial, ape)
                materialized.update(partial, ape)

            scenario_emissions = 0
            for position in range(start, start + width):
                partial = synthetic_partial(position, head_dim)
                stream_emission = stream.update(partial, ape)
                materialized_emission = materialized.update(partial, ape)

                stream_frontier = stream.frontier_rows()
                materialized_frontier = materialized.frontier_rows(position)
                if rows_bits(stream_frontier) != rows_bits(materialized_frontier):
                    raise AssertionError(
                        f"frontier mismatch at width={width} phase={phase} "
                        f"position={position}"
                    )
                state_checks += 1
                digest.update(rows_bits(stream_frontier))

                if (stream_emission is None) != (materialized_emission is None):
                    raise AssertionError("compression boundary mismatch")
                if stream_emission is None:
                    continue
                if rows_bits(stream_emission.rows) != rows_bits(
                    materialized_emission.rows
                ):
                    raise AssertionError("compression row mapping mismatch")
                if values_bits(stream_emission.pooled) != values_bits(
                    materialized_emission.pooled
                ):
                    raise AssertionError("scalar reduction mismatch")
                emission_checks += 1
                scenario_emissions += 1
                digest.update(values_bits(stream_emission.pooled))

            scenarios.append({
                "proposal_width": width,
                "frontier_phase": phase,
                "start_position": start,
                "emissions": scenario_emissions,
            })

    return {
        "analysis": "dspark_ratio4_materialized_equivalence",
        "ratio": RATIO,
        "head_dim": head_dim,
        "proposal_widths": list(PROPOSAL_WIDTHS),
        "frontier_phases": list(FRONTIER_PHASES),
        "scenario_count": len(scenarios),
        "frontier_bitwise_checks": state_checks,
        "boundary_row_and_reduction_checks": emission_checks,
        "digest": digest.hexdigest(),
        "representation_gate": "PASS",
        "operation_order_gate": "PASS_DS4_SCALAR_ORDER",
        "standalone_runtime_gate": "STOP",
        "standalone_runtime_reason": (
            "Position-indexed partial states batch state writes and boundary "
            "gathers, but do not remove the serial exact attention, inverse-"
            "RoPE, or attention-projection tail."
        ),
        "scenarios": scenarios,
    }


def render_report(summary):
    return "\n".join([
        "# DSpark Ratio-4 Materialized-State Equivalence",
        "",
        "Model-free scalar diagnostic; no model process or timing benchmark "
        "was run.",
        "",
        f"- Proposal widths: {', '.join(map(str, summary['proposal_widths']))}",
        "- Starting frontier phases: "
        f"{', '.join(map(str, summary['frontier_phases']))}",
        f"- Scenarios: {summary['scenario_count']}",
        f"- Bitwise frontier checks: {summary['frontier_bitwise_checks']}",
        "- Boundary row/reduction checks: "
        f"{summary['boundary_row_and_reduction_checks']}",
        f"- Representation gate: **{summary['representation_gate']}**",
        "- ds4 scalar operation-order gate: "
        f"**{summary['operation_order_gate']}**",
        f"- Standalone runtime gate: **{summary['standalone_runtime_gate']}**",
        "",
        "At boundary `p`, both references consume lane 0 from `p-7..p-4` "
        "and lane 1 from `p-3..p`. The streaming reference duplicates each "
        "completed block into the lower and upper frontier halves; the "
        "materialized reference reconstructs the same frontier directly from "
        "absolute token positions.",
        "",
        summary["standalone_runtime_reason"],
        "The representation is therefore valid infrastructure for a larger "
        "fused compressor/cache kernel, but it is not a standalone candidate "
        "for closing the current parity deficit.",
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
    summary = run_equivalence()
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
