#!/usr/bin/env python3
"""Import repeated full-2K layer-40 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer30_fixtures as importer


importer.LAYER = 40
importer.TEMPLATE_LAYER = 39
importer.CAPTURED_AT_UTC = "2026-08-27T09:11:57Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "be4da5c11958e275cac6c26c781751030b00f17cd8a6863cc5d48c01a3b586c9",
    "attn_norm": "261d6772fda17ffb2fe1fab6788158222d504814002adea707f37da74b80e22d",
    "q_lora": "e9ba9a04329a1162a324cbc892fdb3330166d4b755d1b6e048f59f05e511dae9",
    "q_lora_norm": "f9bfa9110d8faa34609760ce77e23e601b403770099a6012c917e35607930405",
    "KVraw": "c7592a09caa867049f218aa18b3b7059c83be7fb36dc719d893c9b74bb158c38",
    "KVnorm": "ed311df8617731c960e28c760dc13f9eca161d48197acc882ebdb3b1c01ef074",
    "Qraw": "9ce545f1bf1f375f99d4b2c5f3d13ae2ab1ee12e2d1277685b30b723b297d2c5",
    "Qcur": "a368cf1b2ac094fd11601e277c3276265b058c7f093369536c9a55638bdfe1e0",
    "KVrope": "6910fed1a20bd9fb8c4f31c5c4e8ff0dea54ed9c68d1b1fb4b1c1b2e0826a428",
    "KVcur": "9f6effaaeb777361924c15a2ac8c0c7d3e5c6a561968696a0aaa118591190861",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [512, 512],
        1_048_576,
        "a2264abb1579eab2af68b85ed70d335cd2e62c395d3990c8bf1e1f69e405a7d3",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [8, 1024],
        32_768,
        "7eca59437c336f05c89131235cc1ef967872c29eefaf27f54594726c5cd53a71",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [8, 1024],
        32_768,
        "34b1763994b4f1d848e8c2c1a04cb91473345782027140cf78e7f51b78b9fb97",
    ),
    "indexer_KVcompress": (
        "indexer-compressed-kv.f32le.bin",
        [512, 128],
        262_144,
        "c8736b36d0e55b435f6c5007d775b32732cc51445ee71df089688542eb20f04f",
    ),
    "indexer_state_kv": (
        "indexer-state-kv.f32le.bin",
        [8, 256],
        8_192,
        "6a6928741b455ddf6641f6d4c494f8911e45d512fb0459078e277d3a8929f23e",
    ),
    "indexer_state_score": (
        "indexer-state-score.i32le.bin",
        [8, 256],
        8_192,
        "50986e89c923762c3cf531c463f709a5b42eb189850965ebcd5e45839c238882",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "0e8818933e26edf2ecff57536baa04d062224f2aa1a3a60fefe9ce0dd6733329",
    "hc_attn_post": "ee8d9abc0cfec63880be6a6505dd6b403223cc70b53857cd510964e84b8fce05",
    "kqv_out": "6d534cb65fe0800a93e3c4af02e4451ee82cd008fa474b04e477ebc6a999d91c",
    "kqv_back": "c04d112c625fd48117e59ccf5765a7f3da2ffd58fd756d86d6d81405357f60df",
    "attn_low": "630c03544505999413a2d360ec2be91ade964fafc9084310990fb5183a4b3f04",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "84e34413c15f0d658c7b2a884756c4789db6a78ae16dbe2329273002bf64d13f",
    "ffn_norm": "24c09753add73f366c61ab3a579fac52c136051a054736dd0cae55ea3b4c03cf",
    "ffn_moe_logits": "f6fdca1aee52bd7bd064057fb36c72cbccb278cf48e95ed7930161125d1bc69c",
    "ffn_moe_probs": "788b9af7b23c031b24df37632d36200bf4417a8921fb7e2528a404ea4d9adc30",
    "ffn_moe_topk": "b5d75cc45b6014a2d100341c10b62f6ccca44e6c1cc29c0bb0f68057a8b0f79b",
    "ffn_moe_weights_scaled": "e55514f3bd8496df5e80027d2542d63a6d3b8b036cf15673fe834e09efe07b24",
    "ffn_moe_weighted_swiglu": "bca997014272c3cc41ddd7179900edb6baf819c3de211212be574f027d48310a",
    "ffn_moe_out": "00b986d9d49f3ded361243c13f2563b5a58aa8045162bc78c435def7901c8651",
    "ffn_shexp": "aec1a0a18894bfcc35c00456313e620cd4499ea81e33e3a1e2a1d94bef72fb14",
    "hc_ffn_post": "cb43ea75706d35e32eaaa5a9dff0e7977e5a2eb8c1d3ef757eb776df7fc7a379",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
