#!/usr/bin/env python3
"""Import repeated full-2K layer-29 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer27_fixtures as importer


importer.LAYER = 29
importer.TEMPLATE_LAYER = 28
importer.CAPTURED_AT_UTC = "2026-08-26T12:38:25Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "35292fc6ab7200b606258c094ba18f6dd280094273a47f0820f3f33524137868",
    "attn_norm": "652b4384db5658f375519bddf724ac5c2a9b0a480adb24ce344066eaf8e85637",
    "q_lora": "f32955683f079f3380ab7175690d221282f1ba073d86440009893c79f8139a3c",
    "q_lora_norm": "a6d7c819020bcbb378ff6ae54ab70bff4d10a1f6c7794e1d1269433573268930",
    "KVraw": "c622ea967a9935a6db1f1d15cdc73c9632d805f03a18ed28807c68d0844d8749",
    "KVnorm": "66460a4eaa7b5cc9385975269407c052ecb2bd3d456d0635624354aa0f762124",
    "Qraw": "c934ce03e7a13505ddf568ccc078eed04411af1a7a5b6089f1987405ead9a28c",
    "Qcur": "ac9e4fecf31eddf53c8f83a7e9c5cfb327ae3ef53527b65d3adb0750976f1a58",
    "KVrope": "ef1e641a5fae0e0575eb165525fc79971748e598055d1ed0e510e8a3133cd3b5",
    "KVcur": "2e7be15d36e4036b625a132b4161d452567283beed3b22fd567863a5558222ad",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [16, 512],
        32_768,
        "96144d833b62dc63fc9f93cf531eb84ef2c5c549856936ac48dd95ebd720407e",
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
    "attn_out": "fb1629a09c63ce93a7b7425faa5e565fa1c66df288a50bc625cf0f8934072466",
    "hc_attn_post": "13eb2ae539e2b463ba8baa1fc9ae77769de8247f0f12ad5e405d6c352a920598",
    "kqv_out": "b541091c9fb72da257685262077da67a86bc1748a0b638f1582fed1f1b015ae2",
    "kqv_back": "7fb595363cdf148df892130f0ff7f0a2442f152bcbeb2be4d8c48795795a72d9",
    "attn_low": "c17a3c18bd21d0a82ea96c6bd9dc249a7a84e7b247e96972e8aacbb8207195e0",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "520519503d49a76a35d252a2eee1217cfcc8d9fe50008429f083e3b30faed0b7",
    "ffn_norm": "3ce1490f5202c10af319f82bb6227953c131b2b6f211de910b810041dd5ac3c4",
    "ffn_moe_logits": "f2890ecc582a0aa0d36acdba287d5372529720bbc5f7a9a51b1698c2a22a3391",
    "ffn_moe_probs": "14af03c856c7498b72b4bef32c25a5cca8f1a8c9d1c3c40d61fbf152c1f8de9e",
    "ffn_moe_topk": "a9945550e77093fc086416e07ba45726f4b5303cba074c6f5bdfb23228895f68",
    "ffn_moe_weights_scaled": "7e59935155c5fe6728ecfb69e35ea71f1e8eb01bebe811a7fd771bf36e29184f",
    "ffn_moe_weighted_swiglu": "2fc0ad98e8116620354a6397e0d423da721f24e32fe10fbdb4e996425f0c396e",
    "ffn_moe_out": "b043021790668892de7aa282c27cb8c2699d4fdd18f1fb09aa286af3f891c2d4",
    "ffn_shexp": "cdaf9b7097709f4456692080e70f488c237b9f9b6bdf9fbae8deedf7e7b39f48",
    "hc_ffn_post": "7d34181e31267462b0d956e42cc7c4d5f8afda61b7c4534eba837c741783da02",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
