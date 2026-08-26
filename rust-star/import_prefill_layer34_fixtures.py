#!/usr/bin/env python3
"""Import repeated full-2K layer-34 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer30_fixtures as importer


importer.LAYER = 34
importer.TEMPLATE_LAYER = 33
importer.CAPTURED_AT_UTC = "2026-08-26T18:09:09Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "e42c9434bc1dc2ed08357a2b4780806fc9e6da13a3d559a142e81d7bf7c5f4b5",
    "attn_norm": "ffdeecaefcd91a8c1da23508bf40f4f5620442feef9597daed5213b8bc62d810",
    "q_lora": "431351f0b3e3ef879a4b898ba69d9230210d5fa63aff4f8ecf3f09f242ead356",
    "q_lora_norm": "f9dedda00f474d6d365a86bd45cb0556288620168fe9a005de872bad902af5f4",
    "KVraw": "6051d25a0c705ffbe413bb6b4ac9bbde5cdf3a75978f79d9355507da8b5ac7e7",
    "KVnorm": "ca2382ec1141e937e815a137cf26a5f35ea1797326d84225bcfb5cedcc27ddfa",
    "Qraw": "07af85aba6cb0d262f20dab31d5a0f59f34ca00045f4f48b151b560044dac68c",
    "Qcur": "23d55f867116d9da941e69af4ce169f62feca7d45682fb4719098a15a230424f",
    "KVrope": "0e0a55f97df8133030caa62bd1f20388b7c3371d966acb774634744487bf2ae0",
    "KVcur": "ef230b5b7d67f99a7393ddb7bcb877d8ffd246d9e022546c948f6323a6b345bd",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [512, 512],
        1_048_576,
        "9f718931c9b3fcacf4cfe6e30d8e6913ce80bb9eb07600da945e71c90f2d7980",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [8, 1024],
        32_768,
        "498a333a1d058fe002b4392dc6db69df36919464e911e2673d9a57aea3116ab2",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [8, 1024],
        32_768,
        "fa268c764160719533bf35ae3bf6d6afa1d7a98a00a68e680d156c6d43740df0",
    ),
    "indexer_KVcompress": (
        "indexer-compressed-kv.f32le.bin",
        [512, 128],
        262_144,
        "3b10d540010d14535b39cf7bd03ad5785d13e03f52b72b8217affe33f4c10cac",
    ),
    "indexer_state_kv": (
        "indexer-state-kv.f32le.bin",
        [8, 256],
        8_192,
        "51bef08ed94d59c7dbe47cd788507806ed69ad8942de4c6ea3bd4ab3a3d83928",
    ),
    "indexer_state_score": (
        "indexer-state-score.i32le.bin",
        [8, 256],
        8_192,
        "101fef43ca2949a9a4645733fb9cc8f50b5495d522c1f585f343df9941b69d33",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "47b6aa6e65c8393b0294b9dde896e440ad1015d143a7c56bfaa0a737c3f03822",
    "hc_attn_post": "1cdeba41649b50ef3dee69af430a77789a0b93f51c5bc66d14ab6abf878b078a",
    "kqv_out": "61d023a92c92d14a99d199a110a0ba37e07c9b3a7861aba562f73e5304276b2d",
    "kqv_back": "dd0f50f9cbdcaf82deb182f8aafd44227772cb79ed5469c5921b8533edad42a6",
    "attn_low": "16d2d7444e51d4c110194ffda9a5b4a25f4ae94c00fbab87bfac8655b5850cd3",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "57abafe9e30f85f462ea1faee338e6622dcb94c3a865a313db2f95e7199f8bd7",
    "ffn_norm": "93452db5b3de691e007491418117b7a84bb5fe58080309f636a9bc3ac18c9ca6",
    "ffn_moe_logits": "4efd25e77fa3befb141d60a668b4669c42b33ebfee1e6a323cd4d6f88b09ec8c",
    "ffn_moe_probs": "00e43d1515a390c212a8e515b1637950837ce6a9f94b9d3af53ad6c23775ded7",
    "ffn_moe_topk": "e5c923ba10f87216990c92b7161f08504c6ed8d56727561fea875bebfe90bfa1",
    "ffn_moe_weights_scaled": "aeb1b4bb1fa33c13075fe251d5adefba6231020e5e5424a337ab99dc8231633f",
    "ffn_moe_weighted_swiglu": "3d8ade6afcb5dae112a2eebc130126028c23b1a4ae139a5b54c8f930cc5e4fdd",
    "ffn_moe_out": "9fccb366044f2beab3004703c09a6d3d213468b80c78f2c2b5d983d3f500b3a9",
    "ffn_shexp": "35e9110c1271998d74f9159ba9a6d1e8cf8e5bf44f7845c2e319eae06eaac0b8",
    "hc_ffn_post": "42035d29d1370cc1fe4eed6be2962e2a997325b67ff721760f3be9fe1ad66a5c",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
