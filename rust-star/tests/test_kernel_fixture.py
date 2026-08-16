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
POSITION4_COMPLETE_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos4-complete-v1"
    for layer in range(43)
]
COMPRESSOR_PRIME_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos0-compressor-prime-v1"
    for layer in range(2, 43)
]
LATER_LAYER_COMPLETE_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos{position}-complete-v1"
    for layer in range(4, 43)
    for position in (1, 2, 3)
]
RATIO128_COMPRESSOR_FIXTURES = [
    RUST_STAR_DIR / "fixtures" / f"layer{layer}-pos127-compressor-replay-v1"
    for layer in (3, 5)
]
POSITION127_DECODER_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "decoder-frontier-pos127-v1"
)
COLD_PREFILL_FIXTURE = RUST_STAR_DIR / "fixtures" / "cold-prefill-pos0-v1"
PREFILL_FRONTIER_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-frontier-2048-v1"
)
PREFILL_Q8_BOUNDARY_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-q8-boundary-2048-v1"
)
PREFILL_QKV_BOUNDARY_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-qkv-boundary-2048-v1"
)
PREFILL_HC_INGRESS_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-hc-ingress-2048-v1"
)
PREFILL_ATTENTION_READ_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-attention-read-2048-v1"
)
PREFILL_ATTENTION_OUTPUT_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-attention-output-2048-v1"
)
PREFILL_FFN_OUTPUT_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-ffn-output-2048-v1"
)
PREFILL_LAYER1_INGRESS_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-layer1-ingress-2048-v1"
)
PREFILL_LAYER1_COMPLETE_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-layer1-complete-2048-v1"
)
PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-layers01-previous-tile-2048-v1"
)
PREFILL_LAYER2_KVNORM_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-layer2-kvnorm-2048-v1"
)
PREFILL_LAYER2_KV_STATE_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-layer2-kv-state-2048-v1"
)
PREFILL_LAYER2_ATTENTION_FIXTURE = (
    RUST_STAR_DIR / "fixtures" / "prefill-layer2-attention-2048-v1"
)


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

    def test_position4_complete_fixtures_manifest_and_payloads(self) -> None:
        for layer, fixture in enumerate(POSITION4_COMPLETE_FIXTURES):
            with self.subTest(layer=layer):
                manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos4-complete",
                )
                self.assertEqual(report["scope"], "decode-step")
                expected_operations = (
                    30 if layer == 0 else 28 if layer < 4 else 32 if layer % 2 == 0 else 30
                )
                self.assertEqual(report["operations"], expected_operations)
                self.assertEqual(report["tensors"], 32)
                self.assertEqual(report["verified_bytes"], 739_760)
                self.assertEqual(manifest["capture"]["token_id"], 262)
                self.assertEqual(manifest["capture"]["fresh_process_captures"], 2)
                self.assertEqual(manifest["capture"]["batch_capture_layers"], [0, 42])

    def test_compressor_prime_fixtures_manifest_and_payloads(self) -> None:
        for layer, fixture in zip(range(2, 43), COMPRESSOR_PRIME_FIXTURES):
            with self.subTest(layer=layer):
                manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
                report = validate_differential_fixture(fixture)
                self.assertEqual(
                    report["fixture_id"],
                    f"dwarfstar-oracle-v1-layer{layer}-pos0-compressor-prime",
                )
                self.assertEqual(report["scope"], "layer-segment")
                self.assertEqual(report["operations"], 1)
                self.assertEqual(report["tensors"], 1)
                self.assertEqual(report["verified_bytes"], 16_384)
                if layer >= 8:
                    self.assertEqual(manifest["captured_at_utc"], "2026-08-15T10:38:14Z")
                    self.assertEqual(manifest["capture"]["batch_capture_layers"], [0, 42])
                    self.assertEqual(
                        manifest["capture"]["environment"]["DS4_METAL_GRAPH_DUMP_LAYER"],
                        "all",
                    )

    def test_later_layer_complete_fixtures_manifest_and_payloads(self) -> None:
        expected_operations = {
            layer: ({1: 32, 2: 32, 3: 34} if layer % 2 == 0 else {1: 30, 2: 30, 3: 30})
            for layer in range(4, 43)
        }
        for fixture in LATER_LAYER_COMPLETE_FIXTURES:
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
                if layer >= 8:
                    self.assertEqual(manifest["captured_at_utc"], "2026-08-15T10:38:14Z")
                    self.assertEqual(manifest["capture"]["batch_capture_layers"], [0, 42])
                    self.assertEqual(
                        manifest["capture"]["environment"]["DS4_METAL_GRAPH_DUMP_LAYER"],
                        "all",
                    )

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

    def test_position127_decoder_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (POSITION127_DECODER_FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        report = validate_differential_fixture(POSITION127_DECODER_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-decoder-frontier-pos127",
        )
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 2)
        self.assertEqual(report["verified_bytes"], 517_632)
        self.assertEqual(manifest["capture"]["committed_tokens"], 128)
        self.assertEqual(manifest["selection"]["token_id"], 33148)

    def test_cold_prefill_fixture_manifest_and_payload(self) -> None:
        manifest = json.loads(
            (COLD_PREFILL_FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        report = validate_differential_fixture(COLD_PREFILL_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-cold-prefill-pos0",
        )
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 1)
        self.assertEqual(report["tensors"], 1)
        self.assertEqual(report["verified_bytes"], 517_120)
        self.assertEqual(manifest["capture"]["prompt_token_ids"], [36662])
        self.assertEqual(manifest["selection"]["token_id"], 201)

    def test_prefill_frontier_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_FRONTIER_FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        report = validate_differential_fixture(PREFILL_FRONTIER_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-frontier-2048",
        )
        self.assertEqual(report["scope"], "decode-step")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 3)
        self.assertEqual(report["verified_bytes"], 1_042_432)
        self.assertEqual(manifest["capture"]["prefill_tokens"], 2048)
        self.assertFalse(manifest["capture"]["decode_replay"]["equals_batched_prefill"])
        self.assertEqual(manifest["selection"]["token_id"], 15342)

    def test_prefill_q8_boundary_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_Q8_BOUNDARY_FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        report = validate_differential_fixture(PREFILL_Q8_BOUNDARY_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-q8-boundary-2048",
        )
        self.assertEqual(report["scope"], "kernel")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 3)
        self.assertEqual(report["verified_bytes"], 2_625_536)
        self.assertEqual(
            manifest["operations"][0]["kernel"], "kernel_mul_mm_q8_0_f32"
        )
        self.assertEqual(
            manifest["operations"][0]["dispatch"]["threadgroups"], [4, 16, 1]
        )
        self.assertEqual(manifest["arithmetic_boundary"]["final_row_mismatches"], 1024)

    def test_prefill_qkv_boundary_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_QKV_BOUNDARY_FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        report = validate_differential_fixture(PREFILL_QKV_BOUNDARY_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-qkv-boundary-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 5)
        self.assertEqual(report["tensors"], 7)
        self.assertEqual(report["verified_bytes"], 9_306_112)
        self.assertEqual(manifest["scope"]["captured_position_range"], [2016, 2047])
        self.assertEqual(
            [operation["kernel"] for operation in manifest["operations"]],
            [
                "kernel_mul_mm_q8_0_f32",
                "kernel_mul_mm_q8_0_f32",
                "kernel_dsv4_qkv_rms_norm_f32_4",
                "kernel_mul_mm_q8_0_f32",
                "kernel_dsv4_head_rms_norm_rope_tail_f32",
            ],
        )

    def test_prefill_hc_ingress_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_HC_INGRESS_FIXTURE / "manifest.json").read_text(encoding="utf-8")
        )
        report = validate_differential_fixture(PREFILL_HC_INGRESS_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-hc-ingress-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 5)
        self.assertEqual(report["tensors"], 3)
        self.assertEqual(report["verified_bytes"], 1_048_704)
        self.assertEqual(manifest["scope"]["captured_position_range"], [2016, 2047])
        self.assertEqual(
            [operation["kernel"] for operation in manifest["operations"]],
            [
                "kernel_get_rows_f16",
                "kernel_repeat_f32",
                "kernel_rms_norm_f32_4",
                "kernel_mul_mm_f16_f32",
                "kernel_dsv4_hc_split_weighted_sum_norm4",
            ],
        )

    def test_prefill_attention_read_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_ATTENTION_READ_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_ATTENTION_READ_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-attention-read-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 5)
        self.assertEqual(report["tensors"], 3)
        self.assertEqual(report["verified_bytes"], 12_517_376)
        self.assertEqual(manifest["scope"]["query_position_range"], [2016, 2047])
        self.assertEqual(manifest["scope"]["kv_position_range"], [0, 2047])
        self.assertEqual(
            [operation["kernel"] for operation in manifest["operations"]],
            [
                "MTLBlitCommandEncoder.copyFromBuffer",
                "kernel_cpy_contig_f32_f16_4",
                "kernel_flash_attn_ext_blk",
                "kernel_flash_attn_ext_f16_dk512_dv512",
                "kernel_dsv4_rope_tail_f32",
            ],
        )

    def test_prefill_attention_output_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_ATTENTION_OUTPUT_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_ATTENTION_OUTPUT_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-attention-output-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 3)
        self.assertEqual(report["tensors"], 3)
        self.assertEqual(report["verified_bytes"], 3_670_016)
        self.assertEqual(manifest["scope"]["captured_position_range"], [2016, 2047])
        self.assertEqual(
            [operation["kernel"] for operation in manifest["operations"]],
            [
                "kernel_mul_mm_id_q8_0_f32",
                "kernel_mul_mm_q8_0_f32",
                "kernel_dsv4_hc_expand4",
            ],
        )

    def test_prefill_ffn_output_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_FFN_OUTPUT_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_FFN_OUTPUT_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-ffn-output-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 5)
        self.assertEqual(report["tensors"], 10)
        self.assertEqual(report["verified_bytes"], 5_834_240)
        self.assertEqual(manifest["scope"]["captured_position_range"], [2016, 2047])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])
        self.assertEqual(
            [operation["kernel"] for operation in manifest["operations"]],
            [
                "kernel_dsv4_hc_split_weighted_sum_norm4",
                "kernel_mul_mm_f16_f32 plus legacy batch router kernels",
                "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16",
                "kernel_mul_mm_q8_0_f32 plus kernel_swiglu_flat_f32",
                "kernel_dsv4_hc_expand4",
            ],
        )

    def test_prefill_layer1_ingress_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_LAYER1_INGRESS_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_LAYER1_INGRESS_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-layer1-ingress-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 3)
        self.assertEqual(report["verified_bytes"], 1_179_648)
        self.assertEqual(manifest["scope"]["layer"], 1)
        self.assertEqual(manifest["scope"]["captured_position_range"], [2016, 2047])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])

    def test_prefill_layer1_complete_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_LAYER1_COMPLETE_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_LAYER1_COMPLETE_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-layer1-complete-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 9)
        self.assertEqual(report["tensors"], 21)
        self.assertEqual(report["verified_bytes"], 26_543_616)
        self.assertEqual(manifest["scope"]["layer"], 1)
        self.assertEqual(manifest["scope"]["captured_position_range"], [2016, 2047])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])

    def test_prefill_layers01_previous_tile_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-layers01-previous-tile-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 6)
        self.assertEqual(report["verified_bytes"], 4_326_912)
        self.assertEqual(manifest["scope"]["layers"], [0, 1])
        self.assertEqual(manifest["scope"]["captured_position_range"], [1984, 2015])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])

    def test_prefill_layer2_kvnorm_fixture_manifest_and_payload(self) -> None:
        manifest = json.loads(
            (PREFILL_LAYER2_KVNORM_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_LAYER2_KVNORM_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-layer2-kvnorm-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 1)
        self.assertEqual(report["tensors"], 1)
        self.assertEqual(report["verified_bytes"], 4_194_304)
        self.assertEqual(manifest["scope"]["layer"], 2)
        self.assertEqual(manifest["scope"]["captured_position_range"], [0, 2047])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])

    def test_prefill_layer2_kv_state_fixture_manifest_and_payloads(self) -> None:
        manifest = json.loads(
            (PREFILL_LAYER2_KV_STATE_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_LAYER2_KV_STATE_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-layer2-kv-state-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 2)
        self.assertEqual(report["tensors"], 2)
        self.assertEqual(report["verified_bytes"], 8_388_608)
        self.assertEqual(manifest["scope"]["layer"], 2)
        self.assertEqual(manifest["scope"]["captured_position_range"], [0, 2047])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])

    def test_prefill_layer2_attention_fixture_manifest_and_payload(self) -> None:
        manifest = json.loads(
            (PREFILL_LAYER2_ATTENTION_FIXTURE / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report = validate_differential_fixture(PREFILL_LAYER2_ATTENTION_FIXTURE)
        self.assertEqual(
            report["fixture_id"],
            "dwarfstar-oracle-v1-prefill-layer2-attention-2048",
        )
        self.assertEqual(report["scope"], "layer-segment")
        self.assertEqual(report["operations"], 4)
        self.assertEqual(report["tensors"], 1)
        self.assertEqual(report["verified_bytes"], 33_554_432)
        self.assertEqual(manifest["scope"]["layer"], 2)
        self.assertEqual(manifest["scope"]["captured_position_range"], [0, 2047])
        self.assertTrue(manifest["capture"]["fresh_process_bitwise_match"])
        self.assertIn("remain dense", manifest["capture"]["indexer_policy"])

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
