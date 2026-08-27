#!/usr/bin/env python3
"""Import repeated full-2K layer-38 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer30_fixtures as importer


importer.LAYER = 38
importer.TEMPLATE_LAYER = 37
importer.CAPTURED_AT_UTC = "2026-08-27T07:23:58Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "67a6c6ec36ac6f05c4db4af659f30a9f786798be1b947b5480dc3017f932427b",
    "attn_norm": "65c92a6ff5b1ea525e1423816a90ae9e0eb1f898e8ffe7867c0089c2b6a9093d",
    "q_lora": "9c0c8ae132c4182baeb5062e95b873bfa809c32a5d210e410b70ad399be627f0",
    "q_lora_norm": "b13fb7a881a494a9e2eb1da33e46bda573dab97f6b6749fbdb56e4a318258ab8",
    "KVraw": "7080af0ad91516cd14635ffabb0bd4b939d6f9b863c61c50e54b53276692e1a4",
    "KVnorm": "61e0a7fa905b5a7139644738f9cf340347d5184b48e912d54948fc7be1a70205",
    "Qraw": "917674d74fe03daf009aa12e15314317b0c51b266e9cb2b7017f48ee2697588b",
    "Qcur": "4d35860b366365ff1ce9bfb53ce011a5a24e7353fec4d1c48fdfc04e5560a387",
    "KVrope": "afa0731333fe11b236d6f0c7ed97080c5a9752d607d10e7c9371b04c4f89d535",
    "KVcur": "aee6b52c0cb26980120ef7e30d8e67288ae6d446102a43111e3abd7ab5455f8b",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [512, 512],
        1_048_576,
        "3a8e64d25a6b273d837db1d12edfe8f56db02d268318a82a8994191c2edfefd0",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [8, 1024],
        32_768,
        "71b9e6a68b31e6b736c80cb9434caf7a3df7de4e925d7fe94c7fe7c114534caa",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [8, 1024],
        32_768,
        "650cab65c6c5e74e5b8bac4f589c7987fc172f73886a31d7765888f2ccf2be85",
    ),
    "indexer_KVcompress": (
        "indexer-compressed-kv.f32le.bin",
        [512, 128],
        262_144,
        "309102e5907b093797197164895082810fcc36d1b66a41c46b81a78e37c8683a",
    ),
    "indexer_state_kv": (
        "indexer-state-kv.f32le.bin",
        [8, 256],
        8_192,
        "1f64addbe4fdeaabb8dad58b03aa540bc496f24ad78bf47eaf24b59b4eeb3cf7",
    ),
    "indexer_state_score": (
        "indexer-state-score.i32le.bin",
        [8, 256],
        8_192,
        "0653fa6a3e0410dbaae113129daf17a75da37fd66495b790df8ba73011351013",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "7b5a785bd795f4da556b9c2a605abf044a0bede30d3f28a1e32b471473b964a3",
    "hc_attn_post": "c6db40db77db88f0e615f805cb1140b5c91185334ee6a6b8a3155cab30eff67c",
    "kqv_out": "c98195ddb924eea8a918700e240093c0f95393b65c606fde0d41c1bd08b7008d",
    "kqv_back": "52be2374163d881d6faa56e847c2740939aec8ac327c5075bb37f634bb7c40ae",
    "attn_low": "c01d059c7dcc0c763d0cdb058246823a30459344ed8909464c01b11c35ab5675",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "5cd8a77494867d3c02d7f118483a2f890a1ae81dc42d4f597c68f8b842eacb75",
    "ffn_norm": "d5e7ec98ef83d23f030b5732832d6261e059aec8751b0c32d07ed49b9648b0b5",
    "ffn_moe_logits": "f8fdd00e5fe5755061c2d573f4d52e037f55eada2e7286f5a9bec60bdc99dc79",
    "ffn_moe_probs": "4be2a67f5b0972a7ccd3e93cc9e864cafda5cb877ee03965a3d30598f7a8deeb",
    "ffn_moe_topk": "c1dcec9fd9d19193379cb643910f51a0bd378a28c5706d22c69113375aed4a7b",
    "ffn_moe_weights_scaled": "0989d2e0801cc80338d3234cc62332b9de99f1790ddf4c5a1a129430c1bffcf6",
    "ffn_moe_weighted_swiglu": "bd55a6b994693d7b040085691ef21eab9297b826e7f8846427f553cc1a7134bf",
    "ffn_moe_out": "7675c0026fa5a68568c92fd26a008297f2edda6fdfeaf2d9ebaf957d9435c9f1",
    "ffn_shexp": "b94c82c69a2f71915af9bdf3a1f1d81ccf89237825b8f72b1bf4f39e8749aeab",
    "hc_ffn_post": "e48939d15ee31737a9d8c40676529bf414e3e98cd1eb3eea4eebe20013b6627d",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
