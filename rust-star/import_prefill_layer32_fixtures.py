#!/usr/bin/env python3
"""Import repeated full-2K layer-32 QKV, compressor, attention, and FFN captures."""

from __future__ import annotations

import import_prefill_layer30_fixtures as importer


importer.LAYER = 32
importer.TEMPLATE_LAYER = 31
importer.CAPTURED_AT_UTC = "2026-08-26T16:28:32Z"
importer.QKV_SHA256 = {
    "hc_attn_pre": "097814b0ebee4a2450d9a52526b4b94b877240ed1f7e50a0046424053b25ec0e",
    "attn_norm": "341fac33f67d40915ecdf3498fb8a1e752aca41c18619a34fd6902b897f3d45d",
    "q_lora": "165ebcd2d0a7674805da4f1cca75998bd0d0a1d854597295f6aaef95304a41fd",
    "q_lora_norm": "2b69d0efe73dbac05068c5795475559be7e07f3d6eeed9032bd04d343c26d8db",
    "KVraw": "77d7bd41bc74e00b7700f0625d1e9c4bdb62051595ef98cb330394f2b07a2b44",
    "KVnorm": "9dc6cadc7af12bd5e23c5dd035cb9d37dd1f9a56d2fd093a73236e974650e627",
    "Qraw": "c8fd5df6517bcf01dfb234e8be9673357165d0151feec065697c5052ca1f1219",
    "Qcur": "dfc0d49029c034dc131b7785d2fe77c677b7e87b4e35e5971a233c1f94f2bb14",
    "KVrope": "8ddb96164aa54404bdfa131c159262632f67f1c6329acdf5df87dbaec326664f",
    "KVcur": "db82c82ef699cdb0b6c41c0b78f7310fcea10f6701c39d613bd8d1c826e6ecd7",
}
importer.COMPRESSOR = {
    "KVcompress": (
        "attention-compressed-kv.f32le.bin",
        [512, 512],
        1_048_576,
        "dda3294aef10b4fc3649ee3e2b2d8728747f85e47d3cd1e7dcb2a971e19f2887",
    ),
    "attn_state_kv": (
        "attention-state-kv.f32le.bin",
        [8, 1024],
        32_768,
        "1ef7f193df9139b649d51128b1010669bca259866add01401ee7938b6a123f8c",
    ),
    "attn_state_score": (
        "attention-state-score.i32le.bin",
        [8, 1024],
        32_768,
        "0177373455a0961de6456de99c3b6c9126123baa239b95b576ec4313c3cbe571",
    ),
    "indexer_KVcompress": (
        "indexer-compressed-kv.f32le.bin",
        [512, 128],
        262_144,
        "2f25bc8075a8c0457f5a4095e3a30afb0228e2f285c38179e7c24d8d8ebeb14d",
    ),
    "indexer_state_kv": (
        "indexer-state-kv.f32le.bin",
        [8, 256],
        8_192,
        "4b67cbd2e64c624b730e1580f17c0fe7884c44292b6dab4e6c74ae606dba4cd3",
    ),
    "indexer_state_score": (
        "indexer-state-score.i32le.bin",
        [8, 256],
        8_192,
        "0f71b7e0995873a73980265263b5235889881dea13f784e786b664271dc2619c",
    ),
}
importer.ATTENTION_SHA256 = {
    "attn_out": "de8804394ba706c2d0846ac0faa3e666a1e57b276b1d27a57e5bb555b1ec61e0",
    "hc_attn_post": "6ff31ecab22d44ab835265a44c13c07fd48d88aab2957577fef0c0f68a079c1a",
    "kqv_out": "62eee0a975c7fe8bfa23ab89e8415f412021bac751f861e7f078ba65cfde2ec3",
    "kqv_back": "4ec1391f50f6f2f0c4e4173abb7f9c008c71791781191d42c7cc45d7dbe9a036",
    "attn_low": "9f8d63d3d0d8f695dc101afd216f78c971cda9b05f533901d27ec28ccc63ee31",
}
importer.FFN_SHA256 = {
    "hc_ffn_pre": "fb6cdedacde21ba261ba9d338f8c34f58923ea534d3b75c05612604905d5d450",
    "ffn_norm": "2261db817804743cb9b42b21680218a0055a0095e3e850d6e29ac48d1b940344",
    "ffn_moe_logits": "1ebe8f4f3ad22f9e1f1b05fe8c1b8effa78439893b0241d74754f81f288243ac",
    "ffn_moe_probs": "28c7165257d9357d95c92090458a077d9aee02dd15c7710a4577b57119fa16d9",
    "ffn_moe_topk": "cb882eb4f66cebd2be99d75a9cd1f8496b5c7d93acf2c53ecaf58c82af82dcb7",
    "ffn_moe_weights_scaled": "ac15ea8226bc532a2ac1cc1f739dea05041689de9f20572897d1b60236721f2f",
    "ffn_moe_weighted_swiglu": "3a2b8138bee5b572fe5c33549af21283f2ad9be697ba12461413af49dee04f2c",
    "ffn_moe_out": "3da55eec134b781e6d838422ed1e908cbf03311fb8bddab523a51790385d7cb4",
    "ffn_shexp": "d124c85d8fbc7bfbf0190775feb5dc51687bc27dcd6850fba2923fb59bd2533f",
    "hc_ffn_post": "441b82e5e6a0d5436bedde9fbe20baabf0abedfa81677c07c10ae3312307aad7",
}


if __name__ == "__main__":
    raise SystemExit(importer.main())
