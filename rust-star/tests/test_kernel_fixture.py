#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import struct
import unittest
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "q8-attn-q-a-v1"


class KernelFixtureTests(unittest.TestCase):
    def test_q8_projection_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "rust-star-kernel-fixture-v1")
        self.assertEqual(
            manifest["fixture_id"],
            "dwarfstar-oracle-v1-layer0-pos1-attn-q-a",
        )
        self.assertEqual(manifest["operation"]["tensor"], "blk.0.attn_q_a.weight")
        self.assertEqual(manifest["operation"]["tensor_type"], "Q8_0")

        expected_elements = {
            "activation.f32le.bin": manifest["operation"]["input_elements"],
            "output.f32le.bin": manifest["operation"]["output_elements"],
        }
        for name, descriptor in manifest["artifacts"].items():
            payload = (FIXTURE / name).read_bytes()
            self.assertEqual(len(payload), descriptor["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), descriptor["sha256"])
            self.assertEqual(len(payload), expected_elements[name] * 4)
            values = struct.iter_unpack("<f", payload)
            self.assertTrue(all(math.isfinite(value[0]) for value in values))


if __name__ == "__main__":
    unittest.main()
