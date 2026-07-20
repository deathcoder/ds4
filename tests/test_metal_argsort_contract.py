#!/usr/bin/env python3
"""Model-free checks for the Metal argsort threadgroup-memory contract."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


class MetalArgsortContractTests(unittest.TestCase):
    def test_shader_stages_indices_and_scores(self):
        shader = (ROOT / "metal/argsort.metal").read_text(encoding="utf-8")
        self.assertIn("shmem_i32 + ntg.x", shader)
        self.assertIn("shmem_f32[col] = src0_row[i00 + col]", shader)

    def test_host_allocates_both_scratch_arrays(self):
        host = (ROOT / "ds4_metal.m").read_text(encoding="utf-8")
        self.assertIn(
            "(sizeof(int32_t) + sizeof(float))",
            host,
        )


if __name__ == "__main__":
    unittest.main()
