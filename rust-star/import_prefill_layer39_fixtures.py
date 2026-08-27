#!/usr/bin/env python3
"""Import repeated full-2K layer-39 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 39
importer.TEMPLATE_LAYER = 38
importer.CAPTURED_AT_UTC = "2026-08-27T08:16:45Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "0ccc4c712e221d622a5adf2f1b512485c696cab7fc3cc39c58f2fba41ab1c8ee",
    "attn_norm": "a274b6c553d978b788eb312b434b8bc7a1787e172fcce96892d63f798b57a308",
    "q_lora": "66feb5c7b8027fe3905f4aedc83d88b159ae0e719788758958a34b1d276cd74f",
    "q_lora_norm": "607997eaad1ed7d96366f15bef3744bab54916c7d6385c5fe4ce37afe49f44b4",
    "KVraw": "9effac21eda00dda0da48f0d12649127544fa019b28ff2ac5491ce7242047f28",
    "KVnorm": "bcb1ca80f183502d3da210dbd893986756fcd7915f38753577bafc3c77b06e9b",
    "Qraw": "ca8ae64e16c3920b48e4957ce2c17b2a5b675dbc0d5b2d5786f1fd3d3329becb",
    "Qcur": "f71d8c8affefaa30e74143b91bb9b1b9abba76c2a1c8955f66676dc4d8f84dc6",
    "KVrope": "877793b4799cb3b9dbad8817a35d864a97773d2c8b09ed1595b1b031c557d604",
    "KVcur": "bd3dbf8770676ef8edbc31804e907a75b6f056fb195319e4675e94773286a9a8",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "8cfbdab5e567e31e1c205175e851e7f164f26baa8166c30ae437edd89fabe6ae",
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
    "attn_out": "3aab12640646a96431b7df4d23012770fb7029626401d63330208b74cb5dbc2b",
    "hc_attn_post": "8fa702cd8e4723cc01915600d2ba1926add7a9f6ce4efaad2d19fa07a6d3cea7",
    "kqv_out": "ba30be04587119cc67faee65a06afa160a3abe4ace8b0f32e814e07066abe1b8",
    "kqv_back": "ea5cfaf777d43d9c8e882cb8167ee2cbc082b5f353a7b06e69e8cf6dd8bca1a2",
    "attn_low": "52d8e2fbaef8e9cda585e4eae5d7963f9eca1003c6f534e9b9af86d75dd2a634",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "69342d0ff43c56e3087a0dbce050e9bb2817a9b02b04db6e605fe4d6a83bffac",
    "ffn_norm": "ff38b8dc11921b6e5c24193220bbb25a3de56c1529b8b80c7cd62b3d1b40ce09",
    "ffn_moe_logits": "0b7f56566c6f20c045c0a840c5ae1a05088682e53cf20e8ee4326df4e8e551d2",
    "ffn_moe_probs": "151a61e9b8bc1f215a02b40ae57f3a8513fce2b651a6ae25823f5a934554152c",
    "ffn_moe_topk": "6b74b28dac30a94d30bc15fe8daa88b4a967a2c69349ecf00e0f1d17d3d443a5",
    "ffn_moe_weights_scaled": "df1ddd001a2267ab690fe46b61e9041c33751c12b946465edbddd88478fcf877",
    "ffn_moe_weighted_swiglu": "c9de7ab84d1b207f83b72492e18525a222df2b6232b33752a2d0a5df603cd4f9",
    "ffn_moe_out": "81246234352fd018e046bfc2081badc3e80b6ca78538aa56f1b374e9f71df49d",
    "ffn_shexp": "7a491239272864be90859650bafdd7cb8560f4f76161d1f5f1a62053d32aeed8",
    "hc_ffn_post": "e07bd53390a908f73666adfbe11dd60a9cb766970bc80f57e8e67f7ef53bc3bb",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
