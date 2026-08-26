#!/usr/bin/env python3
"""Import repeated full-2K layer-31 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 31
importer.TEMPLATE_LAYER = 30
importer.CAPTURED_AT_UTC = "2026-08-26T16:55:06Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "12b2b2b0cfa53554e6c5237cd2f21b5ecb6150e96dd0c9913911ae7261f8d810",
    "attn_norm": "37ff3b2c3f9185400f3122ff765444366ece23a83a3596c281a2229b0944e74b",
    "q_lora": "201d3a4ea3f556c4b0e2f369f9cf5691789e05bb1306094b255dd15a65d7499e",
    "q_lora_norm": "c47c71f4de5dc0263dba99f6d53bae6c0d5b8cb689e3d1532e8c86006953baec",
    "KVraw": "92a8e09d4e230a3feddaf16247c38efab67c453263d234aa952892a6f92b2038",
    "KVnorm": "c74b3c69c6d4c7a8e31635b6ff2a0166cfc094271fff77d3b96ab278f001dbfb",
    "Qraw": "a268151522e562e35c734a3687a1a37d00096810fc5c58706525fcc6e9cda162",
    "Qcur": "e94bf6f34600e40800575c99664eb54c6dd096a9c61beff7cecbeceb54d2bccd",
    "KVrope": "997dedc9dd689f58545e452de32c19e914923ceeb9b272fb253d463268a5067a",
    "KVcur": "078370c32e9d686abc4512acbe5751cc55d70777613c38da5d5422cf61c14e82",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "94517346ca7a05ecf264279d7fd2852fbf4fb44b0078bff7ef7be9f3d8df9e20",
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
    "attn_out": "342c66f81ef8006761954a9892a3a2e00569a6cf7e6e2609e11ef4be91cc92d1",
    "hc_attn_post": "2a8092554a1c34d87d7e0d47f76cce71d005accaff35950cb329de3b2f40e6f0",
    "kqv_out": "626e93eb08893f2667ed06f6d61bacbd5a58479863ee66d9c2ac9ccfb3579e04",
    "kqv_back": "68d65ca58ea3e93192a5844410744a125c73ceae05bdc866014bef41bcd63145",
    "attn_low": "87f9fdac275d969126afdbfc0e1dc71984ae862f77ac3d27e23b6c9f7079c976",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "7cd738454a3b7fbe58af4e15dd1558b28a833e434cb8483ca697df952800a369",
    "ffn_norm": "21305bccdd01de29541b8025b1281dc51e87d8f49b86c22561cf6e31e9cab413",
    "ffn_moe_logits": "aec44ae8da0647843eeaf926945bf089b6534da9f1ff14367412aa6aed1dccc3",
    "ffn_moe_probs": "04e64146ff1e5c42508556c45430d5c5605ae036d87a8984bc15be1423c2f17b",
    "ffn_moe_topk": "57b487a200c5d01ea9a1aad6b8e905aba4da58ec853cf66f559fbfd25e48f06f",
    "ffn_moe_weights_scaled": "10e150e52ba52392b15b22eccc7e50a12b151a8b7d71bf389df3aa2b346f2c31",
    "ffn_moe_weighted_swiglu": "b73617f1d189c8e5437f5008714eb057c1104284a9567e0be83f0c669eafe176",
    "ffn_moe_out": "5866fd01bf3dd3d6dead3e3abe2f38bb5a8881ba0509621bf9609de3f2e63cc0",
    "ffn_shexp": "c257ca075a9ac511355062be95017216da5c7813cc7da6250ee7a97d2ea4c5d5",
    "hc_ffn_post": "a7e51b66d0919245a31803b1bb7b38043e7168d3ed078e759e7c8903d9b70e9d",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
