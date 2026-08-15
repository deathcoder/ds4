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
LAYER2_COMPLETE_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer2-complete-v1"
LAYER3_COMPLETE_FIXTURE = RUST_STAR_DIR / "fixtures" / "layer3-complete-v1"
POSITION2_COMPLETE_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos2-complete-v1"
    for layer in range(4)
]
POSITION3_COMPLETE_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos3-complete-v1"
    for layer in range(4)
]
COMPRESSOR_PRIME_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos0-compressor-prime-v1"
    for layer in range(2, 8)
]
LAYER47_COMPLETE_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos{position}-complete-v1"
    for layer in range(4, 8)
    for position in (1, 2, 3)
]
RATIO128_COMPRESSOR_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos127-compressor-replay-v1"
    for layer in (3, 5)
]


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

    def test_layer2_complete_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(LAYER2_COMPLETE_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer2-pos1-complete")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 28)
        self.assertEqual(report["tensors"], 33)
        self.assertEqual(report["verified_bytes"], 741_808)

    def test_layer3_complete_fixture_manifest_and_payloads(self) -> None:
        report = validate_differential_fixture(LAYER3_COMPLETE_FIXTURE)
        self.assertEqual(report["fixture_id"], "dwarfstar-oracle-v1-layer3-pos1-complete")
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 28)
        self.assertEqual(report["tensors"], 33)
        self.assertEqual(report["verified_bytes"], 741_808)

    def test_position2_complete_fixtures_manifest_and_payloads(self) -> None:
        for layer, fixture in enumerate(POSITION2_COMPLETE_FIXTURES):
            with self.subTest(layer=layer):
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos2-complete",
                )
                self.assertEqual(report["scope"], "decode-step")
                self.assertEqual(report["operations"], 30 if layer == 0 else 28)
                self.assertEqual(report["tensors"], 32)
                self.assertEqual(report["verified_bytes"], 739_760)

    def test_position3_complete_fixtures_manifest_and_payloads(self) -> None:
        expected_operations = [30, 28, 34, 30]
        for layer, fixture in enumerate(POSITION3_COMPLETE_FIXTURES):
            with self.subTest(layer=layer):
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos3-complete",
                )
                self.assertEqual(report["scope"], "decode-step")
                self.assertEqual(report["operations"], expected_operations[layer])
                self.assertEqual(report["tensors"], 33 if layer == 2 else 32)
                self.assertEqual(
                    report["verified_bytes"], 741_808 if layer == 2 else 739_760
                )

    def test_compressor_prime_fixtures_manifest_and_payloads(self) -> None:
        for layer, fixture in zip(range(2, 8), COMPRESSOR_PRIME_FIXTURES):
            with self.subTest(layer=layer):
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos0-compressor-prime",
                )
                self.assertEqual(report["scope"], "layer-segment")
                self.assertEqual(report["operations"], 1)
                self.assertEqual(report["tensors"], 1)
                self.assertEqual(report["verified_bytes"], 16_384)

    def test_layer47_complete_fixtures_manifest_and_payloads(self) -> None:
        expected_operations = {
            layer: ({1: 32, 2: 32, 3: 34} if layer % 2 == 0 else {1: 30, 2: 30, 3: 30})
            for layer in range(4, 8)
        }
        for fixture in LAYER47_COMPLETE_FIXTURES:
            manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
            layer = manifest["scope"]["layer"]
            position = manifest["scope"]["position"]
            with self.subTest(layer=layer, position=position):
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos{position}-complete",
                )
                self.assertEqual(report["operations"], expected_operations[layer][position])
                self.assertEqual(
                    report["tensors"],
                    33 if position == 1 or (layer % 2 == 0 and position == 3) else 32,
                )
                self.assertEqual(report["verified_bytes"], 741_808 if report["tensors"] == 33 else 739_760)

    def test_ratio128_compressor_fixtures_manifest_and_payloads(self) -> None:
        for layer, fixture in zip((3, 5), RATIO128_COMPRESSOR_FIXTURES):
            with self.subTest(layer=layer):
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos127-compressor-replay",
                )
                self.assertEqual(report["scope"], "layer-segment")
                self.assertEqual(report["operations"], 4)
                self.assertEqual(report["tensors"], 2)
                self.assertEqual(report["verified_bytes"], 2_099_200)

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
