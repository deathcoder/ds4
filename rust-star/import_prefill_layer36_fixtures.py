#!/usr/bin/env python3
"""Import repeated full-2K layer-36 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer30_fixtures as importer


importer.LAYER = 36
importer.TEMPLATE_LAYER = 35
importer.CAPTURED_AT_UTC = "2026-08-26T19:53:24Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "415d629be200b21398616bd859894a654c22a8c937f4e186cc1eaae2596da813",
    "attn_norm": "bdbd5f51dce39e6c8f76bb4f573e2a080e3d66aa58e8d0d30ad421900cb8db14",
    "q_lora": "17c6822a8804d2c3eccaca37eeb504c5fe35e332ae30b13f88bab1e1230e215f",
    "q_lora_norm": "ee1b707de2ad20c8eca39e7d98cbdf43266dbea9c43b25f296e43b4ecf34a3f3",
    "KVraw": "66c42a1067d0afde5077f778fce0393e0eb86dda30be6aa01b92e5002cfef5f0",
    "KVnorm": "07b34decd322afda1fdea5cc5abd8b1044a8bd2133833f49866fd3f0d49d1fb3",
    "Qraw": "d60157780ae6131eecd5f3269ccdaed15d388ef47412d0a1edb5c45c1858adb9",
    "Qcur": "b0c47441cf46b273882bf3b67f88656639a49c71347e5d3cedea29d561fb3869",
    "KVrope": "1660d30abeed92f6fdadbaf682e493f7644a7e5212699138337436dfe59288cc",
    "KVcur": "aecbc6966f46510df109062f8b2162bde82a9ff386df15100b8019c44dfcb52c",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [512, 512],
        1_048_576,
        "249f59f748af6ece1620619a9f38080268ca18a8766858dc7ef5224c7d6df196",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [8, 1024],
        32_768,
        "24d41e389d547e45578bcdd67158592c54e587d1e56c8a266a410610bfedc2a1",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [8, 1024],
        32_768,
        "7b3d372fee155bbf14148f59633df648b5368395f3aec1c949b053435b25c2ad",
    ),
    "indexer_KVcompress": (
        "indexer-compressed-kv.f32le.bin",
        [512, 128],
        262_144,
        "36972a3bd873259edb69c75f99df36caf87978d1cf4215269230553898b2d83c",
    ),
    "indexer_state_kv": (
        "indexer-state-kv.f32le.bin",
        [8, 256],
        8_192,
        "075b65d01eaf033373bab0a1d5d473ae7fed65ef2967e8362f6b52feaaf0b493",
    ),
    "indexer_state_score": (
        "indexer-state-score.i32le.bin",
        [8, 256],
        8_192,
        "814bf95325608e308ef9d1535c9729d82b2ba5e91f12ef8e4da9180513aafff6",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "7d35c7e3de90a053ec227402ca1e2ff869602b5268ec3b4d49891bb56834c228",
    "hc_attn_post": "a1f867455f874131d74fae4b4d980f77122a42dfeb6ec6ba4d925cb0b32c72c7",
    "kqv_out": "16fad344e5cb4fce2fe79f69a110099bcda212b78e4797da0b869f8befc18b0c",
    "kqv_back": "134332176a8eea81372cdc44d83603a67b30ad6a9dd91175658d1c425f898e3d",
    "attn_low": "fc380101c6fb45a3826dde58ea4724bfeea5896bce82e823a8605a979b76a237",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "d01e57c3569001c5dff4d89dbadc84ba69982a2f8dfdb1a9129a8d8a74aa7fc2",
    "ffn_norm": "de87b735e2971f035b2b9cbb07b61253fb51987da66146dcf38785780725bc9b",
    "ffn_moe_logits": "009fdcc627426aebb2b87501e52860ec99c028956c67dc2968f08194d053e8a6",
    "ffn_moe_probs": "82a47febc20c276c96a826c284085cb9d636f6f3bd24c817dcf80399604920c0",
    "ffn_moe_topk": "90597fe9fc0e86d67e9d692ac574d0a3f28737e4af582c7bf072939fe60c6d11",
    "ffn_moe_weights_scaled": "148114c2cf35cf1e16b403b8c3f2ab73b06f243ceda170141a190f2ec1310710",
    "ffn_moe_weighted_swiglu": "94321519c74d20c3134d0bcb427b97a82053870850fab6eb264291f21a377953",
    "ffn_moe_out": "164b36494fa77e8f531d0c30bbb4e4e028ff24b9721c2d8f8903df6795c46ef2",
    "ffn_shexp": "b6d9d5721192395bc08e6a7898d04d68e269568cc605111af795628e4aac8a1d",
    "hc_ffn_post": "eedf2a1085b810052e6bfaeb61583cce28149e2e7ee45aeace2b218d807a6706",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
