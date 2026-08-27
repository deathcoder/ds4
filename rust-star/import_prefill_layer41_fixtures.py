#!/usr/bin/env python3
"""Import repeated full-2K layer-41 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 41
importer.TEMPLATE_LAYER = 40
importer.CAPTURED_AT_UTC = "2026-08-27T10:45:26Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "a05637abee3efb22d53f076998b1d252ae42e0c1494142a611a620b9fb1338ce",
    "attn_norm": "e8d86bd6bb4931526394b9254db560ffc8169e97238bc48e30483a1dd7487694",
    "q_lora": "caeb73f2556505842eb6c85bd6ea6089317f8ba4903bd53b3b2b1c6771bab3eb",
    "q_lora_norm": "059f69fc7f9496c5fce729089ea1ec877dbd08957c055f1d664570c2a0f5b73c",
    "KVraw": "79c4094f30a5b55a572348f150080dcda0930958a2561141694ff57a2c74ee3b",
    "KVnorm": "edc63b59677e72ed0e4f3bb0af2ce31b767bb39848f9c5311319397e97ced810",
    "Qraw": "eaacd284e3b0353181e0a0d0306f405040d4109bcdb2f3aae2a4a4cc46b41348",
    "Qcur": "bbda1c43b30ffe9a42cd6bb5b7fc69c070b580ffb0ae56a48eaeea5c8bfbf2f1",
    "KVrope": "41c0e91858e220de891ac557b0ebe5809e8be9b3c8cc49fb1e9008c9ce0c73d5",
    "KVcur": "2aaf8d5f4484cdbc63fd3e4eadd275b924161eec12b2ef96c887d691cbc5fb03",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "11efd3451dad39e69db4910d72fe7f926523f2f439849f222f1a4439da41defd",
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
    "attn_out": "050cf8788cd1df645ab48b1fc3adcf2897bc92f909817d65699ae33996ca8f37",
    "hc_attn_post": "f02626c44874405b05a797f814decb0e499bb03a8790443562dba886d965f280",
    "kqv_out": "9785b8db83e566782b147af2e1824afd12376572be3b6ab2d4b9f30abbf9a6a4",
    "kqv_back": "58e94e37241378cfef7f730fae033c05543873f3153393627f557eac88f35f93",
    "attn_low": "6c96b2d0cf6d7c50831c0df92fbaee41e1d9231b4eef12087eadf0129d1c68df",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "4f6ff570e50dc479a22a78ea4c568de11b023036c0585b27bc451763cf469351",
    "ffn_norm": "0184db6556e8a4ae7647d4b35542ccd9c1651d761e5ecff0523c1fc2972b84f3",
    "ffn_moe_logits": "e2b7871f3edb79b59bc532661d3eb72666a516376605408b8da83cabb97afa17",
    "ffn_moe_probs": "9913a1584e8ecb0e5e5235d0838ac279b7f7828d7df49824bfbd647f33edd1db",
    "ffn_moe_topk": "cf5aa35cac5eba9bf5856490435d941e62bc99b088790d3637bb1a92aade72dc",
    "ffn_moe_weights_scaled": "d626edffb5df54c802d5c46406b15f129c3af5cd820d94ef1250c7ad451e92a8",
    "ffn_moe_weighted_swiglu": "49dacf19579c9fea1171fbec5c42d78ff70e9dc3090fbe2226a4f54dea78ea43",
    "ffn_moe_out": "21a2254c4a084235e37caf80b8b69935edb2cc0517e24c11e20c94f2a7180ac0",
    "ffn_shexp": "e0f41025cc9d4d9ba88b91ebb3b4620f4979850794fc7fa27fe7c038adea6689",
    "hc_ffn_post": "638a8284b59abae7e54e3af9f9cf5ca2bb5e006d021592af89c0801d231a4caf",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
