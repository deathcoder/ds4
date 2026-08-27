#!/usr/bin/env python3
"""Import repeated full-2K layer-42 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer30_fixtures as importer


importer.LAYER = 42
importer.TEMPLATE_LAYER = 41
importer.CAPTURED_AT_UTC = "2026-08-27T11:55:21Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "00f57f07a901ff14be7dcfd9f3ad89589b49f0a6ae33e5873648a64d2aff3acb",
    "attn_norm": "e06a37523c3b6d0d0fb647c725509c6054a4384cd118c5dc3d3183598f9c87df",
    "q_lora": "821ef87e5384fc1cfee32de428e91d6f26c55b31c9943aa4c13b432f6dd5cd29",
    "q_lora_norm": "033f2844c920bb55a25be9c0e804f2579bb0c19eab55f8599871b52fd46bd3da",
    "KVraw": "f93e5c35797dae61138fdd9216d06a4c690b8030380ab7b748b44da199815d35",
    "KVnorm": "d1698c438e844c76604f621cf1ee780311e7a34747f9eba63e8314e73e555c40",
    "Qraw": "8f53b9d65afc9ecfcca7542b65ff90df121fddc185ee3bb8ccc85b0f500c076e",
    "Qcur": "e37477ddc0caf266049459114f8e94b57a4e0d6e18b8cc07cc9aa0eb21337555",
    "KVrope": "24a2326a285fa0be209e158537f9118bef2b32c073b114e5997f4a7a983f4331",
    "KVcur": "81f0e0e74b4d434af8348d52a1db123eabe635856749b93c43c978125b96abc3",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [512, 512],
        1_048_576,
        "68bcabfac616b7a96a55c52912e576c794220f41f44565231bf7e84581fd9c54",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [8, 1024],
        32_768,
        "7011af0cefd9a2f72a16edcfbf386292846b59bde67986f8d67db0e9b200a8f4",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [8, 1024],
        32_768,
        "b6aeb8320958ceb2072877b0f46ad7c1b347443d371f61fe96294b6bcb03843c",
    ),
    "indexer_KVcompress": (
        "indexer-compressed-kv.f32le.bin",
        [512, 128],
        262_144,
        "61c35b96a2541f4133ac73b979f21bf78b262bb8818aa4702aff3883f7df6529",
    ),
    "indexer_state_kv": (
        "indexer-state-kv.f32le.bin",
        [8, 256],
        8_192,
        "e7bdb13c5fc68e686c5ea659379f8fee557a93a330d01898c43507867b812c27",
    ),
    "indexer_state_score": (
        "indexer-state-score.i32le.bin",
        [8, 256],
        8_192,
        "7d77079fa6a463843fa4606a82a932e488ca16794fb7b7d49defcbb323a200ec",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "c9aa4481bda10de932e381e3d1065ae84ca116d1b3434f9a71a7d670688ef697",
    "hc_attn_post": "332be2de07f72eda3d534cc18ccef8ab46e7c2f3c3b2faa542969448a2c03611",
    "kqv_out": "875dc6aa8e83bf11b932114d29c7d7a10044817c4e78ef91400828e2d418dbc9",
    "kqv_back": "d927152a34c5ddab31d150ce100f7be853f78806fed1ac2fac978a7daa30a9da",
    "attn_low": "996ee8450b517c103bf8af00dfb22fd3c3f916ea587069aae3ebe3eb864d171b",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "3f6572baf197f9e0a5dc1104865b3409e2700e8925acb35d52c4ef8bc26dfccc",
    "ffn_norm": "e69160ed3423af137c1f78ebb40dd3bcafa405cb4733fea3989aaad98558150a",
    "ffn_moe_logits": "d03326ca534813da1a8032bd08dc09e2a6d178a8abfc5ac828964d74cad0b164",
    "ffn_moe_probs": "c44197442a695bd6b9ae7cedaa26d6b3ee423a417bdd35af0ea849b1cf1bd7bb",
    "ffn_moe_topk": "a60245d65466d894a715fe1927741437d107eca338a9859cd78b3fb984dea31a",
    "ffn_moe_weights_scaled": "b59979287a8ec5a92a95965d961fe9cd685dad62bbbfe834a293849f97d4f661",
    "ffn_moe_weighted_swiglu": "a6ee998ff533b497d6b07dfaab421e15ed5bafa76909d27fe65576a3083a9fff",
    "ffn_moe_out": "93018bc58744900dbf62d66c4c28967da9dade22b48631d5555e9359ae417e82",
    "ffn_shexp": "04c7e4e35dc4e3cf7c87153b5eed1ffaa8ae59e5343055250d8a607df05b81b6",
    "hc_ffn_post": "1c4d2b55c3e685e33e62b65eb1501963ef81d073b6f32368c0ed1478c481077d",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
