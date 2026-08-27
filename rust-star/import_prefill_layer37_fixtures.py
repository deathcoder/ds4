#!/usr/bin/env python3
"""Import repeated full-2K layer-37 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 37
importer.TEMPLATE_LAYER = 36
importer.CAPTURED_AT_UTC = "2026-08-27T05:33:33Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "e0c28d9f8cb1c50204b7901a7c9f0122f00f20ec0613ad7a77f4aa91768121a6",
    "attn_norm": "7656a8cefa0f17361d27a900215a261174c40b27db3746adcaecab5174313d49",
    "q_lora": "5882cf8268c0a705056a502ffe150dd3a44ad9416c928e86beb3ac7325fbdb65",
    "q_lora_norm": "f34303b04135d06903f8618de3c3ca482f73cd19b1347fe7ff11822132563686",
    "KVraw": "9dda7f616378a70b40606cfeb0d87a195446c4c87ea5ef238a8f533dd3d322cc",
    "KVnorm": "9e08735d5229cd1a7e51530745339ab17c524326df59e05b133eb682c0175b96",
    "Qraw": "29d94a70cc24917cb40950f60ab295527a809549cb3ae8372ea1438c25206f13",
    "Qcur": "b83488a5a5727f0bff14a7d190393ef9a09d71ca8821fb8a7edfc149317df99f",
    "KVrope": "d9c28c5e9787cf4e6425959ea7ddcfe72425b6aeba6808d8d0821559e7b28de1",
    "KVcur": "1c6a31823c8a9c6bcd6ef78c583f4288af1b9c7cd1445d26d5ffcade067646ab",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "9c7ca6a7763b51ffa8010c7efe90eae800720f1a8aa4dc2968a888733aa73146",
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
    "attn_out": "587cc4d582eb21021c195db444c004682ad3d634ba89cb88d66a3da4e996eb03",
    "hc_attn_post": "fe7e65832bfd87af7af50a0090093f74b98d6491a0a00368792c9ba9645e04cd",
    "kqv_out": "584ef8b436740ce35052091b2c6c5d270aecc61cf3f44543d37192ddd0d3246e",
    "kqv_back": "7cd30b5c480ff0931556034ed70048e39723f943ffcd2a617a2d565caef23dca",
    "attn_low": "cc2810bbbf73defdb8a4449ed55f1f70d5c811757619902fee8ddff36f864f32",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "e6d7a0429e87159a2ef88d4432c060c3188238f97bea4394eb864d6a19481a5e",
    "ffn_norm": "0ee7fe028336f77f12f458eb3535a7cb27771704987e4f070b005d2c89f9a264",
    "ffn_moe_logits": "aee771ced6d13ed206a80bad61a24d7ed65cb859cbf821925764a81a4d0ef0ac",
    "ffn_moe_probs": "2c8e5f87d4d6a655bb8076e0aa300b1412dcddf9b3bf81da1a9ba9d863678052",
    "ffn_moe_topk": "3d6ffa77ba86bef2f50581e5ad912ea3a93a6d21b77716ca7ad0b23757c490df",
    "ffn_moe_weights_scaled": "13379a2ef55e992c099115d21435c031ff4b5ddec33906053cd83ea7ecc44e2f",
    "ffn_moe_weighted_swiglu": "6495bf3fdd8c84f5a6f02f82b75987cfb36eddc90dd8debddb2be61994128b1f",
    "ffn_moe_out": "31ad74acaf3370c4a23e58225dd1e3673969a000effa3c4d19fcaa24d22718c6",
    "ffn_shexp": "b9c8cf37f6ced1824ba1e0d10910ced61aaf922ec2544d784c61c5c02772b0fc",
    "hc_ffn_post": "6c578dfe57825e20d40fa49bfe6d567783ed2d75e7cb1885d8d5d8ce2b72fd74",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
