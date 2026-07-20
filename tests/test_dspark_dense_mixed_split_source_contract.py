#!/usr/bin/env python3
"""Synthetic contract tests for prepare-free dense-mixed attention."""

from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parent.parent
WIDTH = 512
MAX_HALF = 65504.0


def f16(value):
    """Apply one IEEE binary16 storage round-trip."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def make_row(base):
    return [base + col / 1024.0 for col in range(WIDTH)]


def prepared_view(raw, raw_start, n_raw, compressed, comp_f16):
    n_keys = n_raw + len(compressed)
    n_rows = (n_keys + 31) // 32 * 32
    rows = []
    masks = []
    for row in range(n_rows):
        if row < n_raw:
            source = raw[(raw_start + row) % len(raw)]
            values = [f16(value) for value in source]
            mask = f16(0.0)
        elif row < n_keys:
            source = compressed[row - n_raw]
            values = list(source) if comp_f16 else [f16(value) for value in source]
            mask = f16(0.0)
        else:
            values = [f16(0.0)] * WIDTH
            mask = f16(-MAX_HALF)
        rows.append(values)
        masks.append(mask)
    return rows, masks


def split_source_value(raw, raw_start, n_raw, compressed, comp_f16, row, col):
    n_keys = n_raw + len(compressed)
    if row < n_raw:
        return f16(raw[(raw_start + row) % len(raw)][col]), f16(0.0)
    if row < n_keys:
        value = compressed[row - n_raw][col]
        return (value if comp_f16 else f16(value)), f16(0.0)
    return f16(0.0), f16(-MAX_HALF)


class DenseMixedSplitSourceContractTests(unittest.TestCase):
    def assert_views_equal(
        self, raw, raw_start, n_raw, compressed, comp_f16
    ):
        rows, masks = prepared_view(
            raw, raw_start, n_raw, compressed, comp_f16
        )
        for row, values in enumerate(rows):
            for col, expected in enumerate(values):
                actual, mask = split_source_value(
                    raw,
                    raw_start,
                    n_raw,
                    compressed,
                    comp_f16,
                    row,
                    col,
                )
                self.assertEqual(actual, expected)
                self.assertEqual(mask, masks[row])

    def test_fp32_compressed_rows_and_padded_tail(self):
        raw = [make_row(10.0), make_row(20.0), make_row(30.0)]
        compressed = [make_row(100.0), make_row(200.0)]
        self.assert_views_equal(raw, 0, 3, compressed, False)
        rows, masks = prepared_view(raw, 0, 3, compressed, False)
        self.assertEqual(len(rows), 32)
        self.assertEqual(rows[0][7], f16(raw[0][7]))
        self.assertEqual(rows[3][7], f16(compressed[0][7]))
        self.assertEqual(rows[5], [0.0] * WIDTH)
        self.assertEqual(masks[:5], [0.0] * 5)
        self.assertEqual(masks[5:], [-MAX_HALF] * 27)

    def test_raw_ring_wraps_oldest_to_newest(self):
        raw = [make_row(float(index * 10)) for index in range(5)]
        compressed = [make_row(100.0)]
        self.assert_views_equal(raw, 3, 4, compressed, False)
        rows, _ = prepared_view(raw, 3, 4, compressed, False)
        self.assertEqual(
            [row[0] for row in rows[:4]],
            [f16(raw[index][0]) for index in (3, 4, 0, 1)],
        )

    def test_f16_compressed_rows_are_not_recomputed_from_fp32(self):
        raw = [make_row(1.0)]
        compressed = [
            [f16(0.333251953125 + col / 4096.0) for col in range(WIDTH)]
        ]
        self.assert_views_equal(raw, 0, 1, compressed, True)
        rows, _ = prepared_view(raw, 0, 1, compressed, True)
        self.assertEqual(rows[1], compressed[0])

    def test_exact_32_row_multiple_has_no_padding(self):
        raw = [make_row(float(index)) for index in range(16)]
        compressed = [make_row(float(100 + index)) for index in range(16)]
        self.assert_views_equal(raw, 0, 16, compressed, False)
        rows, masks = prepared_view(raw, 0, 16, compressed, False)
        self.assertEqual(len(rows), 32)
        self.assertEqual(masks, [0.0] * 32)

    def test_source_contract_matches_production_kernels(self):
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        misc = (ROOT / "metal/dsv4_misc.metal").read_text(encoding="utf-8")

        self.assertIn("(n_keys + ncpsg - 1u) / ncpsg * ncpsg", host)
        self.assertIn(".ne11 = (int32_t)n_rows", host)
        self.assertIn(".raw_row_stride = row_bytes", host)
        self.assertIn(
            ".comp_row_stride = comp_kv_f16 ? row_bytes_f16 : row_bytes",
            host,
        )
        self.assertIn("(args.raw_start + row) % args.raw_cap", misc)
        self.assertIn("value = (half)src[col]", misc)
        self.assertIn("value = args.comp_kv_f16 != 0u", misc)
        self.assertIn("dst_mask[row] = row < n_keys ? 0.0h", misc)

    def test_parity_view_is_independent_and_compares_storage_bits(self):
        misc = (ROOT / "metal/dsv4_misc.metal").read_text(encoding="utf-8")

        self.assertIn(
            "kernel_dsv4_dense_mixed_split_source_view_f16", misc
        )
        self.assertIn("device      half4 *dst_kv", misc)
        self.assertIn("uint2 gid [[thread_position_in_grid]]", misc)
        self.assertIn("half4((half)source.x, (half)source.y", misc)
        self.assertIn(
            "kernel_dsv4_dense_mixed_split_source_compare", misc
        )
        self.assertIn("device const ushort *prepared_kv", misc)
        self.assertIn("prepared_kv[i] != split_kv[i]", misc)

    def test_parity_route_preserves_production_attention_inputs(self):
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")

        self.assertIn(
            'getenv("DS4_METAL_DENSE_MIXED_SPLIT_SOURCE_PARITY")', host
        )
        self.assertIn(
            "g_dsv4_dense_mixed_split_source_compare_pipeline", host
        )
        self.assertIn(
            "[enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2]",
            host,
        )
        self.assertIn(
            "Metal dense mixed split-source parity ", host
        )
        self.assertIn('"ok" : "mismatch"', host)

    def test_correctness_matrix_rejects_parity_mismatch(self):
        harness = (
            ROOT / "tests/dspark_gpu_candidates_correctness.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "DS4_TEST_DSPARK_DENSE_MIXED_SPLIT_SOURCE_PARITY", harness
        )
        self.assertIn(
            "DS4_METAL_DENSE_MIXED_SPLIT_SOURCE_PARITY=1", harness
        )
        self.assertIn("result=(mismatch|command_error)", harness)
        self.assertIn("raw_start=[1-9][0-9]*", harness)

    def test_written_contract_pins_arithmetic_boundary(self):
        contract = (
            ROOT / "DSPARK_DENSE_MIXED_CONTRACT.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`(half)` cast before FlashAttention consumes it", contract)
        self.assertIn("`ne11 = n_rows`", contract)
        self.assertIn("same prepared F16 buffer is bound as both K and V", contract)
        self.assertIn("final reduction order", contract)


if __name__ == "__main__":
    unittest.main()
