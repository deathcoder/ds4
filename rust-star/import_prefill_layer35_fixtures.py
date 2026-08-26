#!/usr/bin/env python3
"""Import repeated full-2K layer-35 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 35
importer.TEMPLATE_LAYER = 34
importer.CAPTURED_AT_UTC = "2026-08-26T19:08:29Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "a6c6bb789ee094a40c6f33e8b317b492dc58555301f13cf9f40b4a46daafe75a",
    "attn_norm": "19bdb19a78e6946753912a01ca774ef4884471c90efc26d5dc8ff3adf0d2714c",
    "q_lora": "e90fcbfad46fd3d4564d7b51e2ff6dc1f1f9ccd90dfb006807992f74ddbb055f",
    "q_lora_norm": "ac24b547714eb55e031a1e75fc8f41b13d47625fd69e3ca9ac214a774c9b8b1b",
    "KVraw": "588c286a3433e5f155b6c36ed1cf425c1ad94a3db00faaf295cb940d5025138f",
    "KVnorm": "87344f88d34396a6c59ed85c68c0cc5feadb295fc86844dc495e1254047cf171",
    "Qraw": "0b380a1426d003c78185560f8661de221eb36ab9385164aba00f28f325fb6799",
    "Qcur": "591ae4571b85083e5bb79ec1e9b8ba8d70550402770024376a5b9c0f2dc289f2",
    "KVrope": "27e6da8fa923643f0a061c934125ca1c957d01de23a60d295064cc6e5482da5f",
    "KVcur": "df5655d24a78b83764c28daaf31d266a31cb448b3ac0dc2affc42d2cf316b94a",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "9146f04f8a4c355103c8642bbd249cd118b49ddb70c01c9fde83051c17309f25",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [128, 512],
        262_144,
        "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [128, 512],
        262_144,
        "6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "81b45e3fa98b43d43d9ceece105795a9576f15edc4bc97bc120027fd4f1ec111",
    "hc_attn_post": "24c2959728ef972611b699b34808382e2d7fbaac91bbe2ce3c7494b43163bbb0",
    "kqv_out": "7398e46732702246aef42da2fb33316a4c6097e61ca788dfd511ba255caeb47e",
    "kqv_back": "d9229fddcb1afa4ad937e2ae9e35ca20dba70d029cef81e07b0122ea857361e1",
    "attn_low": "f850727446adff36a5840b3b455620f20d132e9f86992d471ae26c8eecd9edf7",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "ffcb410b082fd6d7a3c8809b2d9dac565d65b0b6ad84b30b8eaaee68bdb48ad0",
    "ffn_norm": "8bb64bebd5ffacf0d9fae5a211f190d9c72114730efd6dcf4b697be0e149c17c",
    "ffn_moe_logits": "8f6f88641eb400c74ca9ad696836da68ffcbc8d5e17ce650b12a1a50975d902f",
    "ffn_moe_probs": "798de5d9b07aab7e58f44649c7d8e69185895bd6c298f45d55f63f6d44626061",
    "ffn_moe_topk": "79a140c57d500e5902a7180fbf688819d4075ba53dff6fbe9bf78b4188a3a8c2",
    "ffn_moe_weights_scaled": "69153abb36b2e92f25633aaf677ce282b43ba92ed2e2a24c4c2b17577be17082",
    "ffn_moe_weighted_swiglu": "2fa432bbde50bb2ecedff4394ebbf2ca7e6b70ec86cee6442c142cecd648d396",
    "ffn_moe_out": "d3eb173092f2e13648614e9e3d505ae76152db9e9e3b02da5b2094296af2d50c",
    "ffn_shexp": "7eaa6abd7154fac099ed1e97a95ae3ff350c7df286b06725b5e3bf4918705afc",
    "hc_ffn_post": "985fa31f19063daaa3198b71ebd47b54481c3ff3f62619845d0c5d125b1341a4",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
