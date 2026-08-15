#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


RUST_STAR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUST_STAR_DIR))

from artifact_lib import ArtifactError, validate_differential_fixture  # noqa: E402


FIXTURE = RUST_STAR_DIR / "fixtures" / "q8-attn-q-a-v1"
INGRESS_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-attention-ingress-v1"
SETUP_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-qkv-setup-v1"
ROPE_STORE_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-rope-kv-store-v1"
ATTENTION_READ_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-attention-read-v1"
ATTENTION_OUTPUT_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-attention-output-v1"
FFN_ROUTER_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-ffn-router-v1"
MOE_OUTPUT_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer0-moe-output-v1"
LAYER1_COMPLETE_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer1-complete-v1"


class KernelFixtureTests(unittest.TestCase):
    def test_q8_projection_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "rust-star-differential-fixture-v1")
        self.assertEqual(
            manifest["fixture_id"],
            "dwarfstar-oracle-v1-layer0-pos1-attn-q-a",
        )
        self.assertEqual(manifest["scope"]["kind"], "kernel")
        self.assertEqual(manifest["operations"][0]["weights"], ["blk.0.attn_q_a.weight"])
        report = validate_differential_fixture(FIXTURE)
        self.assertTrue(report["valid"])
        self.assertEqual(report["operations"], 1)
        self.assertEqual(report["tensors"], 2)
        self.assertEqual(report["verified_bytes"], 20_480)

    def test_layer0_ingress_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(INGRESS_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-attention-ingress")
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 6)
        self.assertEqual(report["tensors"], 7)
        self.assertEqual(report["verified_bytes"], 37_056)

    def test_layer0_qkv_setup_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(SETUP_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-qkv-setup")
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 3)
        self.assertEqual(report["tensors"], 6)
        self.assertEqual(report["verified_bytes"], 159_744)

    def test_layer0_rope_kv_store_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(ROPE_STORE_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-rope-kv-store")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 3)
        self.assertEqual(report["tensors"], 7)
        self.assertEqual(report["verified_bytes"], 401_408)

    def test_layer0_attention_read_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(ATTENTION_READ_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-attention-read")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 5)
        self.assertEqual(report["tensors"], 4)
        self.assertEqual(report["verified_bytes"], 266_240)

    def test_layer0_attention_output_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(ATTENTION_OUTPUT_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-attention-output")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 4)
        self.assertEqual(report["verified_bytes"], 245_760)

    def test_layer0_ffn_router_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(FFN_ROUTER_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-ffn-router")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 7)
        self.assertEqual(report["tensors"], 11)
        self.assertEqual(report["verified_bytes"], 100_592)

    def test_layer0_moe_output_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(MOE_OUTPUT_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer0-pos1-moe-output")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 4)
        self.assertEqual(report["tensors"], 11)
        self.assertEqual(report["verified_bytes"], 229_520)

    def test_layer1_complete_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(LAYER1_COMPLETE_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer1-pos1-complete")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 28)
        self.assertEqual(report["tensors"], 33)
        self.assertEqual(report["verified_bytes"], 741_808)

    def test_fixture_shape_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture"
            shutil.copytree(FIXTURE, fixture)
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tensors"][0]["shape"] = [4095]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "bytes does not match shape"):
                validate_differential_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
