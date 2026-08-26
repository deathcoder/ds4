#!/usr/bin/env python3
"""Import repeated full-2K layer-33 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 33
importer.TEMPLATE_LAYER = 32
importer.CAPTURED_AT_UTC = "2026-08-26T17:16:42Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "ec3fc8f10bea9102b4ed4c400eff7d1041896533205d62c007a9075e5d7ec96c",
    "attn_norm": "dd21bef2727de7e383085e4786159df6d308eab7a6795a6ced083786f485f07b",
    "q_lora": "a0b4167e2d7a89f55fa1018a098b29f53047314e750b293919abe73869d82e01",
    "q_lora_norm": "adb56b7840b1094311a8e79526cc1365e90ab13dcc319ec061490cb53eb9a1d5",
    "KVraw": "0bba17d08264cf0f4661b4ef1a8e899725d5d4b51c60db8cfb392c2e3e9a041b",
    "KVnorm": "7366ea79ee2e65b6b7f9eaf43f2e7b7a862c86f137f605cea226585877b0105b",
    "Qraw": "a5d43adbb550a75833a7d44819b8a863875b3bb15dd66ad6208af01eecdee262",
    "Qcur": "86c81635529ad337c936d04278789f2051657958e47ea648660726afa05616d8",
    "KVrope": "8d7ebaf4759f5f90d531a22c8c023b689aff917c7cae935c1456b729611c40b2",
    "KVcur": "965d57597b7a0bbd97a66f2904e227002c7afacacda4c970d323daf1f131b848",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "ab2c71a998b8b9aae4596ed58d03912840235e687e6b38985c8d0f6391a54326",
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
    "attn_out": "558e41e477b7117d28cd7847b791cd4ac500544174bcb64efc5475ed90f4086c",
    "hc_attn_post": "9ad78bca17e715ccd847159e47e83f4752731525f3c14186950070e5e279f211",
    "kqv_out": "804955cea51975e7f76cfc1c9ad7fec27da607700f3eb3c70d50d980bae6cf33",
    "kqv_back": "70b3383bf015611da54c4e4191f82cd9f68cc526e04e6d05af21b6f0d61ecf90",
    "attn_low": "092a062dc309db5f46455360efb4b99f0d07630a855c9b9d012f045e9952aea6",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "b2f477e60100b0aa6f3d53bb3cddc6f7d963cdb1ede2a6062130bde61b903a45",
    "ffn_norm": "ba05d7f543c7532a4fb43e3c3b32fbb9715ff78e68a821f2afef541e53ab1882",
    "ffn_moe_logits": "3fcbb0edc8cb68057bf7fa9cd09762b66d701d9a70cc92674bd0df826cfdb342",
    "ffn_moe_probs": "4e39c5ed6f0c6a9c24fdb6b5cdb929887f52f1e0ea3041cd2204fbcd50944fb5",
    "ffn_moe_topk": "49450661ae33d0f2fff28d160bf990e3d2b396fcff5d3ae6d091bbe2c65818b8",
    "ffn_moe_weights_scaled": "7c095be89bedbd2a7946b25628f295c8ca0238f45b0a7dd6547d28ab8ced5038",
    "ffn_moe_weighted_swiglu": "8734609ca964e7ca263be6cf331c7e2e0a931098d793c7ad221b1da1a976b4d6",
    "ffn_moe_out": "6ab4b63f7f4a27dcfb650ba4e379bfaf4da1e28253d3da2f2cc327d47c71e964",
    "ffn_shexp": "9ce220454fbf3ff60e40084fc068ef8476a1bf3730aef59649ef8cbe5e1e368c",
    "hc_ffn_post": "7e925407c7aac9886d5c5e2251f1c10d7e15b2f68c6fa7297a992428daea933b",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
