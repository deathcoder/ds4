//! Minimal Metal ownership and command-dispatch probe.

use crate::gguf::TensorInfo;
use crate::model::MappedModel;
use crate::{Error, Result};
use std::io::Write;

pub const PROBE_SCHEMA: &str = "rust-star-metal-dispatch-probe-v1";
pub const EMBEDDING_PROBE_SCHEMA: &str = "rust-star-f16-embedding-probe-v1";
pub const PROJECTION_PROBE_SCHEMA: &str = "rust-star-q8-0-projection-probe-v1";
pub const INGRESS_PROBE_SCHEMA: &str = "rust-star-layer0-attention-ingress-probe-v1";
pub const ATTENTION_SETUP_PROBE_SCHEMA: &str = "rust-star-layer0-attention-setup-probe-v1";
pub const ROPE_KV_STORE_PROBE_SCHEMA: &str = "rust-star-layer0-rope-kv-store-probe-v1";
pub const ATTENTION_READ_PROBE_SCHEMA: &str = "rust-star-layer0-attention-read-probe-v1";
pub const ATTENTION_OUTPUT_PROBE_SCHEMA: &str = "rust-star-layer0-attention-output-probe-v1";
pub const FFN_ROUTER_PROBE_SCHEMA: &str = "rust-star-layer0-ffn-router-probe-v1";
pub const MOE_OUTPUT_PROBE_SCHEMA: &str = "rust-star-layer0-moe-output-probe-v1";
pub const LAYER0_PROBE_SCHEMA: &str = "rust-star-layer0-complete-probe-v1";
pub const LAYER0_BENCH_SCHEMA: &str = "rust-star-layer0-steady-state-v1";
pub const LAYERS01_PROBE_SCHEMA: &str = "rust-star-layers01-continuous-probe-v1";
pub const LAYERS012_PROBE_SCHEMA: &str = "rust-star-layers012-continuous-probe-v1";
pub const LAYERS012_CHAINED_PROBE_SCHEMA: &str = "rust-star-layers012-chained-probe-v1";
pub const LAYERS0123_PROBE_SCHEMA: &str = "rust-star-layers0123-continuous-probe-v1";
pub const LAYERS0123_CHAINED_PROBE_SCHEMA: &str = "rust-star-layers0123-chained-probe-v1";
pub const LAYERS0123_BENCH_SCHEMA: &str = "rust-star-layers0123-steady-state-v1";
pub const LAYERS0123_DECODE_PROBE_SCHEMA: &str = "rust-star-layers0123-position-advancing-probe-v1";
pub const LAYERS012345_DECODE_PROBE_SCHEMA: &str =
    "rust-star-layers012345-position-advancing-probe-v1";
pub const LAYERS01234567_DECODE_PROBE_SCHEMA: &str =
    "rust-star-layers01234567-position-advancing-probe-v1";
pub const LAYERS0_TO_42_DECODE_PROBE_SCHEMA: &str =
    "rust-star-layers0-42-position-advancing-probe-v1";
pub const DECODER_OUTPUT_PROBE_SCHEMA: &str =
    "rust-star-decoder-output-position-advancing-probe-v1";
pub const RATIO128_COMPRESSOR_REPLAY_PROBE_SCHEMA: &str =
    "rust-star-ratio128-compressor-replay-probe-v1";
pub const PROJECTION_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attn-q-a";
pub const INGRESS_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attention-ingress";
pub const ATTENTION_SETUP_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-qkv-setup";
pub const ROPE_KV_STORE_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-rope-kv-store";
pub const ATTENTION_READ_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attention-read";
pub const ATTENTION_OUTPUT_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attention-output";
pub const FFN_ROUTER_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-ffn-router";
pub const MOE_OUTPUT_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-moe-output";
pub const LAYER0_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-complete";
pub const LAYER1_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer1-pos1-complete";
pub const LAYER2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer2-pos1-complete";
pub const LAYER3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer3-pos1-complete";
pub const LAYER4_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer4-pos1-complete";
pub const LAYER5_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer5-pos1-complete";
pub const LAYER6_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer6-pos1-complete";
pub const LAYER7_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer7-pos1-complete";
pub const LAYER0_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos2-complete";
pub const LAYER1_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer1-pos2-complete";
pub const LAYER2_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer2-pos2-complete";
pub const LAYER3_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer3-pos2-complete";
pub const LAYER4_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer4-pos2-complete";
pub const LAYER5_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer5-pos2-complete";
pub const LAYER6_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer6-pos2-complete";
pub const LAYER7_POS2_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer7-pos2-complete";
pub const LAYER0_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos3-complete";
pub const LAYER1_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer1-pos3-complete";
pub const LAYER2_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer2-pos3-complete";
pub const LAYER3_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer3-pos3-complete";
pub const LAYER4_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer4-pos3-complete";
pub const LAYER5_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer5-pos3-complete";
pub const LAYER6_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer6-pos3-complete";
pub const LAYER7_POS3_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer7-pos3-complete";
pub const LAYER3_POS127_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-layer3-pos127-compressor-replay";
pub const LAYER5_POS127_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-layer5-pos127-compressor-replay";
pub const DEFAULT_ELEMENTS: u64 = 4096;
pub const DEFAULT_ITERATIONS: u64 = 100;
const MAX_ELEMENTS: u64 = 16 * 1024 * 1024;
const MAX_ITERATIONS: u64 = 100_000;
const MAX_THREAD_INVOCATIONS: u64 = 1_000_000_000;
const MAX_LAYER0_EXECUTIONS: u32 = 1000;
const MAX_EMBEDDING_TOKENS: usize = 64;
const OUTPUT_HEAD_POS1_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos1-v1/output-hc-pre.f32le.bin");
const OUTPUT_HEAD_POS1_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos1-v1/output-hc-weights.f32le.bin");
const OUTPUT_HEAD_POS1_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos1-v1/output-hc.f32le.bin");
const OUTPUT_HEAD_POS1_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos1-v1/output-norm.f32le.bin");
const OUTPUT_HEAD_POS1_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos1-v1/logits.f32le.bin");
const OUTPUT_HEAD_POS2_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos2-v1/output-hc-pre.f32le.bin");
const OUTPUT_HEAD_POS2_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos2-v1/output-hc-weights.f32le.bin");
const OUTPUT_HEAD_POS2_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos2-v1/output-hc.f32le.bin");
const OUTPUT_HEAD_POS2_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos2-v1/output-norm.f32le.bin");
const OUTPUT_HEAD_POS2_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos2-v1/logits.f32le.bin");
const OUTPUT_HEAD_POS3_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos3-v1/output-hc-pre.f32le.bin");
const OUTPUT_HEAD_POS3_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos3-v1/output-hc-weights.f32le.bin");
const OUTPUT_HEAD_POS3_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos3-v1/output-hc.f32le.bin");
const OUTPUT_HEAD_POS3_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos3-v1/output-norm.f32le.bin");
const OUTPUT_HEAD_POS3_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos3-v1/logits.f32le.bin");
const PROJECTION_TENSOR: &str = "blk.0.attn_q_a.weight";
const PROJECTION_INPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/q8-attn-q-a-v1/activation.f32le.bin");
const PROJECTION_OUTPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/q8-attn-q-a-v1/output.f32le.bin");
const INGRESS_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/hc-mixes.f32le.bin");
const INGRESS_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/hc-pre.f32le.bin");
const INGRESS_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/hc-post.f32le.bin");
const INGRESS_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/hc-combination.f32le.bin");
const INGRESS_COLLAPSED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/hc-collapsed.f32le.bin");
const INGRESS_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/attn-norm.f32le.bin");
const INGRESS_Q_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-ingress-v1/q-lora.f32le.bin");
const SETUP_Q_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-qkv-setup-v1/q-lora-norm.f32le.bin");
const SETUP_KV_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-qkv-setup-v1/kv-raw.f32le.bin");
const SETUP_KV_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-qkv-setup-v1/kv-norm.f32le.bin");
const SETUP_Q_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-qkv-setup-v1/q-raw.f32le.bin");
const ROPE_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-rope-kv-store-v1/q-cur.f32le.bin");
const ROPE_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-rope-kv-store-v1/kv-rope.f32le.bin");
const ROPE_KV_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-rope-kv-store-v1/kv-cur.f32le.bin");
const ROPE_CACHE_ROW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-rope-kv-store-v1/cache-row.f32le.bin");
const ATTENTION_CACHE_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-read-v1/cache-row0.f32le.bin");
const ATTENTION_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-read-v1/q-cur.f32le.bin");
const ATTENTION_CACHE_ROW1_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-read-v1/cache-row1.f32le.bin");
const ATTENTION_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-read-v1/kqv-back.f32le.bin");
const OUTPUT_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-output-v1/kqv-back.f32le.bin");
const OUTPUT_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-output-v1/attn-low.f32le.bin");
const OUTPUT_ATTN_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-output-v1/attn-out.f32le.bin");
const OUTPUT_HC_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-attention-output-v1/hc-attn-post.f32le.bin");
const FFN_INPUT_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/hc-attn-post.f32le.bin");
const FFN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/hc-mixes.f32le.bin");
const FFN_PRE_BYTES: &[u8] = include_bytes!("../../fixtures/layer0-ffn-router-v1/hc-pre.f32le.bin");
const FFN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/hc-post.f32le.bin");
const FFN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/hc-combination.f32le.bin");
const FFN_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/ffn-cur.f32le.bin");
const FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/ffn-norm.f32le.bin");
const ROUTER_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/router-logits.f32le.bin");
const ROUTER_PROBS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/router-probs.f32le.bin");
const ROUTER_SELECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/router-selected.i32le.bin");
const ROUTER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-ffn-router-v1/router-weights.f32le.bin");
const MOE_FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/ffn-norm.f32le.bin");
const MOE_SELECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/router-selected.i32le.bin");
const MOE_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/router-weights.f32le.bin");
const MOE_INPUT_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/hc-attn-post.f32le.bin");
const MOE_PRE_BYTES: &[u8] = include_bytes!("../../fixtures/layer0-moe-output-v1/hc-pre.f32le.bin");
const MOE_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/hc-post.f32le.bin");
const MOE_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/hc-combination.f32le.bin");
const MOE_ROUTED_MID_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/routed-mid.f32le.bin");
const MOE_ROUTED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/routed-out.f32le.bin");
const MOE_SHARED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/shared-out.f32le.bin");
const MOE_HC_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer0-moe-output-v1/hc-ffn-post.f32le.bin");
const LAYER1_CACHE_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/cache-row0.f32le.bin");
const LAYER1_ATTN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-mixes.f32le.bin");
const LAYER1_ATTN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-pre.f32le.bin");
const LAYER1_ATTN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-post.f32le.bin");
const LAYER1_ATTN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-combination.f32le.bin");
const LAYER1_ATTN_COLLAPSED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-collapsed.f32le.bin");
const LAYER1_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-norm.f32le.bin");
const LAYER1_Q_LORA_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/q-lora.f32le.bin");
const LAYER1_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/q-lora-norm.f32le.bin");
const LAYER1_KV_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/kv-raw.f32le.bin");
const LAYER1_Q_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/q-raw.f32le.bin");
const LAYER1_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/q-cur.f32le.bin");
const LAYER1_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/kv-rope.f32le.bin");
const LAYER1_KV_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/kv-cur.f32le.bin");
const LAYER1_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/kqv-back.f32le.bin");
const LAYER1_ATTN_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-low.f32le.bin");
const LAYER1_ATTN_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/attn-out.f32le.bin");
const LAYER1_ATTN_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/hc-attn-post.f32le.bin");
const LAYER1_FFN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/ffn-mixes.f32le.bin");
const LAYER1_FFN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/ffn-pre.f32le.bin");
const LAYER1_FFN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/ffn-post.f32le.bin");
const LAYER1_FFN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/ffn-combination.f32le.bin");
const LAYER1_FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/ffn-norm.f32le.bin");
const LAYER1_ROUTER_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/router-logits.f32le.bin");
const LAYER1_ROUTER_PROBS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/router-probs.f32le.bin");
const LAYER1_SELECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/router-selected.i32le.bin");
const LAYER1_ROUTER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/router-weights.f32le.bin");
const LAYER1_ROUTED_MID_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/routed-mid.f32le.bin");
const LAYER1_ROUTED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/routed-out.f32le.bin");
const LAYER1_SHARED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/shared-out.f32le.bin");
const LAYER1_FINAL_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer1-complete-v1/hc-ffn-post.f32le.bin");
const LAYER2_CACHE_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/cache-row0.f32le.bin");
const LAYER2_ATTN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-mixes.f32le.bin");
const LAYER2_ATTN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-pre.f32le.bin");
const LAYER2_ATTN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-post.f32le.bin");
const LAYER2_ATTN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-combination.f32le.bin");
const LAYER2_ATTN_COLLAPSED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-collapsed.f32le.bin");
const LAYER2_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-norm.f32le.bin");
const LAYER2_Q_LORA_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/q-lora.f32le.bin");
const LAYER2_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/q-lora-norm.f32le.bin");
const LAYER2_KV_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/kv-raw.f32le.bin");
const LAYER2_Q_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/q-raw.f32le.bin");
const LAYER2_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/q-cur.f32le.bin");
const LAYER2_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/kv-rope.f32le.bin");
const LAYER2_KV_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/kv-cur.f32le.bin");
const LAYER2_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/kqv-back.f32le.bin");
const LAYER2_ATTN_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-low.f32le.bin");
const LAYER2_ATTN_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/attn-out.f32le.bin");
const LAYER2_ATTN_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/hc-attn-post.f32le.bin");
const LAYER2_FFN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/ffn-mixes.f32le.bin");
const LAYER2_FFN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/ffn-pre.f32le.bin");
const LAYER2_FFN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/ffn-post.f32le.bin");
const LAYER2_FFN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/ffn-combination.f32le.bin");
const LAYER2_FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/ffn-norm.f32le.bin");
const LAYER2_ROUTER_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/router-logits.f32le.bin");
const LAYER2_ROUTER_PROBS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/router-probs.f32le.bin");
const LAYER2_SELECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/router-selected.i32le.bin");
const LAYER2_ROUTER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/router-weights.f32le.bin");
const LAYER2_ROUTED_MID_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/routed-mid.f32le.bin");
const LAYER2_ROUTED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/routed-out.f32le.bin");
const LAYER2_SHARED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/shared-out.f32le.bin");
const LAYER2_FINAL_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-complete-v1/hc-ffn-post.f32le.bin");
const LAYER3_CACHE_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/cache-row0.f32le.bin");
const LAYER3_ATTN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-mixes.f32le.bin");
const LAYER3_ATTN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-pre.f32le.bin");
const LAYER3_ATTN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-post.f32le.bin");
const LAYER3_ATTN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-combination.f32le.bin");
const LAYER3_ATTN_COLLAPSED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-collapsed.f32le.bin");
const LAYER3_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-norm.f32le.bin");
const LAYER3_Q_LORA_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/q-lora.f32le.bin");
const LAYER3_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/q-lora-norm.f32le.bin");
const LAYER3_KV_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/kv-raw.f32le.bin");
const LAYER3_Q_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/q-raw.f32le.bin");
const LAYER3_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/q-cur.f32le.bin");
const LAYER3_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/kv-rope.f32le.bin");
const LAYER3_KV_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/kv-cur.f32le.bin");
const LAYER3_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/kqv-back.f32le.bin");
const LAYER3_ATTN_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-low.f32le.bin");
const LAYER3_ATTN_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/attn-out.f32le.bin");
const LAYER3_ATTN_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/hc-attn-post.f32le.bin");
const LAYER3_FFN_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/ffn-mixes.f32le.bin");
const LAYER3_FFN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/ffn-pre.f32le.bin");
const LAYER3_FFN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/ffn-post.f32le.bin");
const LAYER3_FFN_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/ffn-combination.f32le.bin");
const LAYER3_FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/ffn-norm.f32le.bin");
const LAYER3_ROUTER_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/router-logits.f32le.bin");
const LAYER3_ROUTER_PROBS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/router-probs.f32le.bin");
const LAYER3_SELECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/router-selected.i32le.bin");
const LAYER3_ROUTER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/router-weights.f32le.bin");
const LAYER3_ROUTED_MID_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/routed-mid.f32le.bin");
const LAYER3_ROUTED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/routed-out.f32le.bin");
const LAYER3_SHARED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/shared-out.f32le.bin");
const LAYER3_FINAL_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-complete-v1/hc-ffn-post.f32le.bin");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeConfig {
    pub elements: u64,
    pub iterations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layer0BenchConfig {
    pub warmup_iterations: u32,
    pub iterations: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layers0123BenchConfig {
    pub warmup_iterations: u32,
    pub iterations: u32,
}

impl Default for Layers0123BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            iterations: 20,
        }
    }
}

impl Layers0123BenchConfig {
    pub fn validate(self) -> Result<Self> {
        if self.iterations == 0 {
            return Err(Error::invalid(
                "layers-0/1/2/3 benchmark iterations must be positive",
            ));
        }
        let total = self
            .warmup_iterations
            .checked_add(self.iterations)
            .ok_or_else(|| Error::invalid("four-layer benchmark iteration count overflows"))?;
        if total > MAX_LAYER0_EXECUTIONS {
            return Err(Error::invalid(format!(
                "four-layer benchmark warmup+iterations must not exceed {MAX_LAYER0_EXECUTIONS}"
            )));
        }
        Ok(self)
    }
}

impl Default for Layer0BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            iterations: 30,
        }
    }
}

impl Layer0BenchConfig {
    pub fn validate(self) -> Result<Self> {
        if self.iterations == 0 {
            return Err(Error::invalid(
                "layer-0 benchmark iterations must be positive",
            ));
        }
        let total = self
            .warmup_iterations
            .checked_add(self.iterations)
            .ok_or_else(|| Error::invalid("layer-0 benchmark iteration count overflows"))?;
        if total > MAX_LAYER0_EXECUTIONS {
            return Err(Error::invalid(format!(
                "layer-0 benchmark warmup+iterations must not exceed {MAX_LAYER0_EXECUTIONS}"
            )));
        }
        Ok(self)
    }
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            elements: DEFAULT_ELEMENTS,
            iterations: DEFAULT_ITERATIONS,
        }
    }
}

impl ProbeConfig {
    pub fn validate(self) -> Result<Self> {
        if self.elements == 0 || self.elements > MAX_ELEMENTS {
            return Err(Error::invalid(format!(
                "Metal probe elements must be in 1..={MAX_ELEMENTS}"
            )));
        }
        if self.iterations == 0 || self.iterations > MAX_ITERATIONS {
            return Err(Error::invalid(format!(
                "Metal probe iterations must be in 1..={MAX_ITERATIONS}"
            )));
        }
        let work = self
            .elements
            .checked_mul(self.iterations)
            .ok_or_else(|| Error::invalid("Metal probe work size overflows"))?;
        if work > MAX_THREAD_INVOCATIONS {
            return Err(Error::invalid(format!(
                "Metal probe elements*iterations must not exceed {MAX_THREAD_INVOCATIONS}"
            )));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub device_name: String,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_bytes: u64,
    pub max_total_threads_per_threadgroup: u64,
    pub elements: u64,
    pub iterations: u64,
    pub buffer_bytes: u64,
    pub checksum: u64,
    pub setup_ms: f64,
    pub compile_ms: f64,
    pub warmup_wall_ms: f64,
    pub warmup_gpu_ms: f64,
    pub roundtrip_wall_ms: f64,
    pub roundtrip_gpu_ms: f64,
    pub batched_wall_ms: f64,
    pub batched_gpu_ms: f64,
}

#[derive(Clone, Debug)]
pub struct EmbeddingProbeReport {
    pub tensor_name: String,
    pub tokens: Vec<u32>,
    pub embedding_elements: u64,
    pub output_elements: u64,
    pub model_bytes: u64,
    pub tensor_offset: u64,
    pub tensor_bytes: u64,
    pub page_offset: u64,
    pub buffer_bytes: u64,
    pub inner_offset: u64,
    pub max_buffer_length: u64,
    pub no_copy_pointer_match: bool,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub checksum: u64,
}

#[derive(Clone, Debug)]
pub struct ProjectionProbeReport {
    pub fixture_id: &'static str,
    pub tensor_name: String,
    pub input_elements: u64,
    pub output_elements: u64,
    pub model_bytes: u64,
    pub tensor_offset: u64,
    pub tensor_bytes: u64,
    pub page_offset: u64,
    pub buffer_bytes: u64,
    pub inner_offset: u64,
    pub max_buffer_length: u64,
    pub no_copy_pointer_match: bool,
    pub simdgroups: u32,
    pub rows_per_threadgroup: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub input_checksum: u64,
    pub output_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct IngressProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub mixes_checksum: u64,
    pub split_checksum: u64,
    pub collapsed_checksum: u64,
    pub attn_norm_checksum: u64,
    pub q_lora_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct AttentionSetupProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub q_lora_norm_checksum: u64,
    pub kv_raw_checksum: u64,
    pub kv_norm_checksum: u64,
    pub q_raw_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct RopeKvStoreProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub cache_capacity_rows: u32,
    pub cache_target_row: u32,
    pub cache_guard_rows_intact: bool,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub q_cur_checksum: u64,
    pub kv_rope_checksum: u64,
    pub kv_cur_checksum: u64,
    pub cache_row_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct AttentionReadProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub cache_capacity_rows: u32,
    pub cache_rows_read: u32,
    pub cache_row0_preserved: bool,
    pub cache_guard_row_intact: bool,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub attention_raw_checksum: u64,
    pub attention_back_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct AttentionOutputProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub output_groups: u32,
    pub output_rank: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub attention_low_checksum: u64,
    pub attention_out_checksum: u64,
    pub hc_post_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct FfnRouterProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub expert_count: u32,
    pub selected_experts: Vec<i32>,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub hc_mixes_checksum: u64,
    pub hc_split_checksum: u64,
    pub ffn_cur_checksum: u64,
    pub ffn_norm_checksum: u64,
    pub router_logits_checksum: u64,
    pub router_probs_checksum: u64,
    pub router_weights_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct MoeOutputProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub selected_experts: Vec<i32>,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub routed_mid_checksum: u64,
    pub routed_out_checksum: u64,
    pub shared_out_checksum: u64,
    pub hc_post_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct Layer0ProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub dispatches: u32,
    pub command_buffers: u32,
    pub selected_experts: Vec<i32>,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub attention_hc_checksum: u64,
    pub ffn_norm_checksum: u64,
    pub router_weights_checksum: u64,
    pub routed_mid_checksum: u64,
    pub routed_out_checksum: u64,
    pub shared_out_checksum: u64,
    pub final_hc_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct LayerSequenceProbeReport {
    pub layers: Vec<Layer0ProbeReport>,
    pub command_buffers: u32,
    pub retained_hc_handoff: bool,
    pub kv_cache_layers: u32,
}

pub type Layers01ProbeReport = LayerSequenceProbeReport;
pub type Layers012ProbeReport = LayerSequenceProbeReport;
pub type Layers0123ProbeReport = LayerSequenceProbeReport;

#[derive(Clone, Debug)]
pub struct Layers012ChainedProbeReport {
    pub layers: Vec<Layer0ProbeReport>,
    pub command_buffers: u32,
    pub host_waits: u32,
    pub retained_hc_handoff: bool,
    pub kv_cache_layers: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
}

pub type Layers0123ChainedProbeReport = Layers012ChainedProbeReport;

#[derive(Clone, Copy, Debug)]
pub struct TimingSummary {
    pub median_ms: f64,
    pub mad_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

#[derive(Clone, Debug)]
pub struct Layer0BenchReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub warmup_iterations: u32,
    pub iterations: u32,
    pub dispatches_per_iteration: u32,
    pub command_buffers_per_iteration: u32,
    pub selected_experts: Vec<i32>,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms_samples: Vec<f64>,
    pub gpu_ms_samples: Vec<f64>,
    pub wall: TimingSummary,
    pub gpu: TimingSummary,
    pub repeat_bitwise_match: bool,
    pub final_hc_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct Layers0123BenchReport {
    pub warmup_iterations: u32,
    pub iterations: u32,
    pub command_buffers_per_iteration: u32,
    pub host_waits_per_iteration: u32,
    pub retained_hc_handoff: bool,
    pub kv_cache_layers: u32,
    pub wall_ms_samples: Vec<f64>,
    pub gpu_ms_samples: Vec<f64>,
    pub wall: TimingSummary,
    pub gpu: TimingSummary,
    pub final_layers: Vec<Layer0ProbeReport>,
}

#[derive(Clone, Debug)]
pub struct Layers0123DecodeStepReport {
    pub position: u32,
    pub token: u32,
    pub cache_rows: u32,
    pub layers: Vec<Layer0ProbeReport>,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub output_hc_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct Layers0123DecodeProbeReport {
    pub steps: Vec<Layers0123DecodeStepReport>,
    pub command_buffers_per_step: u32,
    pub host_waits_per_step: u32,
    pub kv_cache_layers: u32,
    pub cache_capacity_rows: u32,
    pub output_hc_elements: u32,
}

pub type Layers012345DecodeStepReport = Layers0123DecodeStepReport;
pub type Layers012345DecodeProbeReport = Layers0123DecodeProbeReport;
pub type Layers01234567DecodeStepReport = Layers0123DecodeStepReport;
pub type Layers01234567DecodeProbeReport = Layers0123DecodeProbeReport;
pub type Layers0To42DecodeStepReport = Layers0123DecodeStepReport;
pub type Layers0To42DecodeProbeReport = Layers0123DecodeProbeReport;

#[derive(Clone, Debug)]
pub struct OutputHeadProbeReport {
    pub fixture_id: &'static str,
    pub dispatches: u32,
    pub command_buffers: u32,
    pub host_waits: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub hc_pre_checksum: u64,
    pub hc_weights_checksum: u64,
    pub hc_checksum: u64,
    pub norm_checksum: u64,
    pub logits_checksum: u64,
    pub selected_token: u32,
}

#[derive(Clone, Debug)]
pub struct DecoderOutputStepReport {
    pub position: u32,
    pub input_token: u32,
    pub cache_rows: u32,
    pub layers: Vec<Layer0ProbeReport>,
    pub transformer_wall_ms: f64,
    pub transformer_gpu_ms: f64,
    pub output_head: OutputHeadProbeReport,
}

#[derive(Clone, Debug)]
pub struct DecoderOutputProbeReport {
    pub steps: Vec<DecoderOutputStepReport>,
    pub command_buffers_per_step: u32,
    pub host_waits_per_step: u32,
    pub kv_cache_layers: u32,
    pub cache_capacity_rows: u32,
    pub logits_elements: u32,
}

#[derive(Clone, Debug)]
pub struct Ratio128CompressorLayerReport {
    pub layer: u32,
    pub fixture_id: &'static str,
    pub activation_rows: u32,
    pub state_rows: u32,
    pub dispatches: u32,
    pub command_buffers: u32,
    pub host_waits: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub input_checksum: u64,
    pub output_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct Ratio128CompressorReplayProbeReport {
    pub layers: Vec<Ratio128CompressorLayerReport>,
    pub final_position: u32,
    pub externally_supplied_activation_rows: u32,
}

pub fn write_ingress_probe_json<W: Write>(
    output: &mut W,
    report: &IngressProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{INGRESS_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"operations\": [\"kernel_get_rows_f16\", \"kernel_repeat_f32\", \"kernel_rms_norm_f32_4\", \"kernel_mul_mv_f16_f32_4\", \"kernel_dsv4_hc_split_weighted_sum_norm4\", \"kernel_mul_mv_q8_0_f32\"],\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"hc_mixes\": {},\n    \"hc_split\": {},\n    \"hc_collapsed\": {},\n    \"attn_norm\": {},\n    \"q_lora\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.mixes_checksum,
        report.split_checksum,
        report.collapsed_checksum,
        report.attn_norm_checksum,
        report.q_lora_checksum,
    )?;
    Ok(())
}

pub fn write_attention_setup_probe_json<W: Write>(
    output: &mut W,
    report: &AttentionSetupProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{ATTENTION_SETUP_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"q_lora_norm\": {},\n    \"kv_raw\": {},\n    \"kv_norm\": {},\n    \"q_raw\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.q_lora_norm_checksum,
        report.kv_raw_checksum,
        report.kv_norm_checksum,
        report.q_raw_checksum,
    )?;
    Ok(())
}

pub fn write_rope_kv_store_probe_json<W: Write>(
    output: &mut W,
    report: &RopeKvStoreProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{ROPE_KV_STORE_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"cache\": {{\n    \"capacity_rows\": {},\n    \"target_row\": {},\n    \"guard_rows_intact\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"q_cur\": {},\n    \"kv_rope\": {},\n    \"kv_cur\": {},\n    \"cache_row\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.cache_capacity_rows,
        report.cache_target_row,
        report.cache_guard_rows_intact,
        report.wall_ms,
        report.gpu_ms,
        report.q_cur_checksum,
        report.kv_rope_checksum,
        report.kv_cur_checksum,
        report.cache_row_checksum,
    )?;
    Ok(())
}

pub fn write_attention_read_probe_json<W: Write>(
    output: &mut W,
    report: &AttentionReadProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{ATTENTION_READ_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"cache\": {{\n    \"capacity_rows\": {},\n    \"rows_read\": {},\n    \"row0_preserved\": {},\n    \"guard_row_intact\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"attention_raw\": {},\n    \"kqv_back\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.cache_capacity_rows,
        report.cache_rows_read,
        report.cache_row0_preserved,
        report.cache_guard_row_intact,
        report.wall_ms,
        report.gpu_ms,
        report.attention_raw_checksum,
        report.attention_back_checksum,
    )?;
    Ok(())
}

pub fn write_attention_output_probe_json<W: Write>(
    output: &mut W,
    report: &AttentionOutputProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{ATTENTION_OUTPUT_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"projection\": {{\n    \"groups\": {},\n    \"rank_per_group\": {}\n  }},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"attn_low\": {},\n    \"attn_out\": {},\n    \"hc_attn_post\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.output_groups,
        report.output_rank,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.attention_low_checksum,
        report.attention_out_checksum,
        report.hc_post_checksum,
    )?;
    Ok(())
}

pub fn write_ffn_router_probe_json<W: Write>(
    output: &mut W,
    report: &FfnRouterProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{FFN_ROUTER_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"router\": {{\n    \"expert_count\": {},\n    \"selected_experts\": {:?}\n  }},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"hc_mixes\": {},\n    \"hc_split\": {},\n    \"ffn_cur\": {},\n    \"ffn_norm\": {},\n    \"router_logits\": {},\n    \"router_probs\": {},\n    \"router_weights\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.expert_count,
        report.selected_experts,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.hc_mixes_checksum,
        report.hc_split_checksum,
        report.ffn_cur_checksum,
        report.ffn_norm_checksum,
        report.router_logits_checksum,
        report.router_probs_checksum,
        report.router_weights_checksum,
    )?;
    Ok(())
}

pub fn write_moe_output_probe_json<W: Write>(
    output: &mut W,
    report: &MoeOutputProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{MOE_OUTPUT_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"selected_experts\": {:?},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"routed_mid\": {},\n    \"routed_out\": {},\n    \"shared_out\": {},\n    \"hc_ffn_post\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.selected_experts,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.routed_mid_checksum,
        report.routed_out_checksum,
        report.shared_out_checksum,
        report.hc_post_checksum,
    )?;
    Ok(())
}

pub fn write_layer0_probe_json<W: Write>(output: &mut W, report: &Layer0ProbeReport) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{LAYER0_PROBE_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"dispatches\": {},\n  \"command_buffers\": {},\n  \"selected_experts\": {:?},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"hc_attn_post\": {},\n    \"ffn_norm\": {},\n    \"router_weights\": {},\n    \"routed_mid\": {},\n    \"routed_out\": {},\n    \"shared_out\": {},\n    \"hc_ffn_post\": {}\n  }},\n  \"c0_bitwise_match\": true\n}}\n",
        report.fixture_id,
        report.token,
        report.dispatches,
        report.command_buffers,
        report.selected_experts,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.attention_hc_checksum,
        report.ffn_norm_checksum,
        report.router_weights_checksum,
        report.routed_mid_checksum,
        report.routed_out_checksum,
        report.shared_out_checksum,
        report.final_hc_checksum,
    )?;
    Ok(())
}

pub fn write_layers01_probe_json<W: Write>(
    output: &mut W,
    report: &Layers01ProbeReport,
) -> Result<()> {
    write_layer_sequence_probe_json(output, report, LAYERS01_PROBE_SCHEMA, 2)
}

pub fn write_layers012_probe_json<W: Write>(
    output: &mut W,
    report: &Layers012ProbeReport,
) -> Result<()> {
    write_layer_sequence_probe_json(output, report, LAYERS012_PROBE_SCHEMA, 3)
}

pub fn write_layers0123_probe_json<W: Write>(
    output: &mut W,
    report: &Layers0123ProbeReport,
) -> Result<()> {
    write_layer_sequence_probe_json(output, report, LAYERS0123_PROBE_SCHEMA, 4)
}

pub fn write_layers012_chained_probe_json<W: Write>(
    output: &mut W,
    report: &Layers012ChainedProbeReport,
) -> Result<()> {
    write_chained_layer_sequence_probe_json(output, report, LAYERS012_CHAINED_PROBE_SCHEMA, 3)
}

pub fn write_layers0123_chained_probe_json<W: Write>(
    output: &mut W,
    report: &Layers0123ChainedProbeReport,
) -> Result<()> {
    write_chained_layer_sequence_probe_json(output, report, LAYERS0123_CHAINED_PROBE_SCHEMA, 4)
}

fn write_chained_layer_sequence_probe_json<W: Write>(
    output: &mut W,
    report: &Layers012ChainedProbeReport,
    schema: &str,
    expected_layers: usize,
) -> Result<()> {
    if report.layers.len() != expected_layers
        || report.command_buffers != expected_layers as u32
        || report.host_waits != 1
        || report.kv_cache_layers != expected_layers as u32
        || !report.wall_ms.is_finite()
        || report.wall_ms <= 0.0
        || !report.gpu_ms.is_finite()
        || report.gpu_ms < 0.0
    {
        return Err(Error::invalid(
            "chained layer-sequence report has inconsistent scheduling metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{}\",\n  \"command_buffers\": {},\n  \"host_waits\": {},\n  \"retained_hc_handoff\": {},\n  \"kv_cache_layers\": {},\n  \"timing\": {{\n    \"chain_wall_ms\": {:.6},\n    \"summed_command_gpu_ms\": {:.6}\n  }},\n  \"layers\": [",
        schema,
        report.command_buffers,
        report.host_waits,
        report.retained_hc_handoff,
        report.kv_cache_layers,
        report.wall_ms,
        report.gpu_ms,
    )?;
    for (index, layer) in report.layers.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(
            output,
            "\n    {{\"layer\": {}, \"fixture\": \"{}\", \"dispatches\": {}, \"selected_experts\": {:?}, \"wrapped_model_ranges\": {}, \"pointer_matches\": {}, \"gpu_ms\": {:.6}, \"final_hc_checksum\": {}, \"c0_bitwise_match\": true}}",
            index,
            layer.fixture_id,
            layer.dispatches,
            layer.selected_experts,
            layer.wrapped_model_ranges,
            layer.pointer_matches,
            layer.gpu_ms,
            layer.final_hc_checksum,
        )?;
    }
    write!(output, "\n  ],\n  \"c0_bitwise_match\": true\n}}\n")?;
    Ok(())
}

fn write_layer_sequence_probe_json<W: Write>(
    output: &mut W,
    report: &LayerSequenceProbeReport,
    schema: &str,
    expected_layers: usize,
) -> Result<()> {
    if report.layers.len() != expected_layers
        || report.command_buffers != expected_layers as u32
        || report.kv_cache_layers != expected_layers as u32
    {
        return Err(Error::invalid(
            "layer-sequence report has inconsistent layer ownership metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{}\",\n  \"command_buffers\": {},\n  \"retained_hc_handoff\": {},\n  \"kv_cache_layers\": {},\n  \"layers\": [",
        schema,
        report.command_buffers,
        report.retained_hc_handoff,
        report.kv_cache_layers,
    )?;
    for (index, layer) in report.layers.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(
            output,
            "\n    {{\"layer\": {}, \"fixture\": \"{}\", \"dispatches\": {}, \"selected_experts\": {:?}, \"wrapped_model_ranges\": {}, \"pointer_matches\": {}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"final_hc_checksum\": {}, \"c0_bitwise_match\": true}}",
            index,
            layer.fixture_id,
            layer.dispatches,
            layer.selected_experts,
            layer.wrapped_model_ranges,
            layer.pointer_matches,
            layer.wall_ms,
            layer.gpu_ms,
            layer.final_hc_checksum,
        )?;
    }
    write!(output, "\n  ],\n  \"c0_bitwise_match\": true\n}}\n")?;
    Ok(())
}

pub fn write_layer0_bench_json<W: Write>(output: &mut W, report: &Layer0BenchReport) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{LAYER0_BENCH_SCHEMA}\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"warmup_iterations\": {},\n  \"measured_iterations\": {},\n  \"dispatches_per_iteration\": {},\n  \"command_buffers_per_iteration\": {},\n  \"selected_experts\": {:?},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"wall_ms\": {{\n    \"samples\": [",
        report.fixture_id,
        report.token,
        report.warmup_iterations,
        report.iterations,
        report.dispatches_per_iteration,
        report.command_buffers_per_iteration,
        report.selected_experts,
        report.wrapped_model_ranges,
        report.pointer_matches,
    )?;
    write_timing_samples(output, &report.wall_ms_samples)?;
    write!(
        output,
        "],\n    \"median\": {:.6},\n    \"mad\": {:.6},\n    \"min\": {:.6},\n    \"max\": {:.6}\n  }},\n  \"gpu_ms\": {{\n    \"samples\": [",
        report.wall.median_ms, report.wall.mad_ms, report.wall.min_ms, report.wall.max_ms,
    )?;
    write_timing_samples(output, &report.gpu_ms_samples)?;
    write!(
        output,
        "],\n    \"median\": {:.6},\n    \"mad\": {:.6},\n    \"min\": {:.6},\n    \"max\": {:.6}\n  }},\n  \"checksums\": {{\n    \"hc_ffn_post\": {}\n  }},\n  \"repeat_bitwise_match\": {},\n  \"c0_bitwise_match\": true\n}}\n",
        report.gpu.median_ms,
        report.gpu.mad_ms,
        report.gpu.min_ms,
        report.gpu.max_ms,
        report.final_hc_checksum,
        report.repeat_bitwise_match,
    )?;
    Ok(())
}

pub fn write_layers0123_bench_json<W: Write>(
    output: &mut W,
    report: &Layers0123BenchReport,
) -> Result<()> {
    if report.iterations == 0
        || report.wall_ms_samples.len() != report.iterations as usize
        || report.gpu_ms_samples.len() != report.iterations as usize
        || report.command_buffers_per_iteration != 4
        || report.host_waits_per_iteration != 1
        || report.kv_cache_layers != 4
        || report.final_layers.len() != 4
    {
        return Err(Error::invalid(
            "four-layer benchmark report has inconsistent scheduling metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{LAYERS0123_BENCH_SCHEMA}\",\n  \"warmup_iterations\": {},\n  \"measured_iterations\": {},\n  \"command_buffers_per_iteration\": {},\n  \"host_waits_per_iteration\": {},\n  \"retained_hc_handoff\": {},\n  \"kv_cache_layers\": {},\n  \"correctness_readback\": \"after-final-measured-iteration\",\n  \"wall_ms\": {{\n    \"samples\": [",
        report.warmup_iterations,
        report.iterations,
        report.command_buffers_per_iteration,
        report.host_waits_per_iteration,
        report.retained_hc_handoff,
        report.kv_cache_layers,
    )?;
    write_timing_samples(output, &report.wall_ms_samples)?;
    write!(
        output,
        "],\n    \"median\": {:.6},\n    \"mad\": {:.6},\n    \"min\": {:.6},\n    \"max\": {:.6}\n  }},\n  \"gpu_ms\": {{\n    \"samples\": [",
        report.wall.median_ms, report.wall.mad_ms, report.wall.min_ms, report.wall.max_ms,
    )?;
    write_timing_samples(output, &report.gpu_ms_samples)?;
    write!(
        output,
        "],\n    \"median\": {:.6},\n    \"mad\": {:.6},\n    \"min\": {:.6},\n    \"max\": {:.6}\n  }},\n  \"final_layers\": [",
        report.gpu.median_ms, report.gpu.mad_ms, report.gpu.min_ms, report.gpu.max_ms,
    )?;
    for (index, layer) in report.final_layers.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(
            output,
            "\n    {{\"layer\": {index}, \"fixture\": \"{}\", \"selected_experts\": {:?}, \"final_hc_checksum\": {}, \"c0_bitwise_match\": true}}",
            layer.fixture_id, layer.selected_experts, layer.final_hc_checksum,
        )?;
    }
    write!(output, "\n  ],\n  \"final_c0_bitwise_match\": true\n}}\n")?;
    Ok(())
}

pub fn write_layers0123_decode_probe_json<W: Write>(
    output: &mut W,
    report: &Layers0123DecodeProbeReport,
) -> Result<()> {
    write_position_advancing_probe_json(output, report, LAYERS0123_DECODE_PROBE_SCHEMA, 4)
}

pub fn write_layers012345_decode_probe_json<W: Write>(
    output: &mut W,
    report: &Layers012345DecodeProbeReport,
) -> Result<()> {
    write_position_advancing_probe_json(output, report, LAYERS012345_DECODE_PROBE_SCHEMA, 6)
}

pub fn write_layers01234567_decode_probe_json<W: Write>(
    output: &mut W,
    report: &Layers01234567DecodeProbeReport,
) -> Result<()> {
    write_position_advancing_probe_json(output, report, LAYERS01234567_DECODE_PROBE_SCHEMA, 8)
}

pub fn write_layers0_to_42_decode_probe_json<W: Write>(
    output: &mut W,
    report: &Layers0To42DecodeProbeReport,
) -> Result<()> {
    write_position_advancing_probe_json(output, report, LAYERS0_TO_42_DECODE_PROBE_SCHEMA, 43)
}

pub fn write_decoder_output_probe_json<W: Write>(
    output: &mut W,
    report: &DecoderOutputProbeReport,
) -> Result<()> {
    if report.steps.len() != 3
        || report.command_buffers_per_step != 44
        || report.host_waits_per_step != 2
        || report.kv_cache_layers != 43
        || report.cache_capacity_rows != 4
        || report.logits_elements != 129280
    {
        return Err(Error::invalid(
            "decoder-output report has inconsistent boundary metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{DECODER_OUTPUT_PROBE_SCHEMA}\",\n  \"selection\": \"lowest-token-id-argmax\",\n  \"input_tokens\": [201, 361, 1915],\n  \"command_buffers_per_step\": {},\n  \"host_waits_per_step\": {},\n  \"kv_cache_layers\": {},\n  \"cache_capacity_rows\": {},\n  \"logits_elements\": {},\n  \"steps\": [",
        report.command_buffers_per_step,
        report.host_waits_per_step,
        report.kv_cache_layers,
        report.cache_capacity_rows,
        report.logits_elements,
    )?;
    for (index, step) in report.steps.iter().enumerate() {
        let expected_position = index as u32 + 1;
        let expected_input = [201_u32, 361, 1915][index];
        let expected_selected = [361_u32, 1915, 262][index];
        if step.position != expected_position
            || step.input_token != expected_input
            || step.cache_rows != expected_position + 1
            || step.layers.len() != 43
            || step.output_head.dispatches != 5
            || step.output_head.command_buffers != 1
            || step.output_head.host_waits != 1
            || step.output_head.wrapped_model_ranges != 5
            || step.output_head.pointer_matches != 5
            || step.output_head.selected_token != expected_selected
            || !step.transformer_wall_ms.is_finite()
            || step.transformer_wall_ms <= 0.0
            || !step.transformer_gpu_ms.is_finite()
            || step.transformer_gpu_ms < 0.0
            || !step.output_head.wall_ms.is_finite()
            || step.output_head.wall_ms <= 0.0
            || !step.output_head.gpu_ms.is_finite()
            || step.output_head.gpu_ms < 0.0
        {
            return Err(Error::invalid(
                "decoder-output step has inconsistent scheduling metadata",
            ));
        }
        if index != 0 {
            write!(output, ",")?;
        }
        write!(
            output,
            "\n    {{\n      \"position\": {},\n      \"input_token\": {},\n      \"cache_rows\": {},\n      \"transformer\": {{\"layers\": 43, \"wall_ms\": {:.6}, \"summed_gpu_ms\": {:.6}, \"c0_bitwise_match\": true}},\n      \"output_head\": {{\n        \"fixture\": \"{}\",\n        \"dispatches\": {},\n        \"command_buffers\": {},\n        \"host_waits\": {},\n        \"mapping\": {{\"wrapped_model_ranges\": {}, \"pointer_matches\": {}}},\n        \"timing\": {{\"wall_ms\": {:.6}, \"gpu_ms\": {:.6}}},\n        \"checksums\": {{\"hc_pre\": {}, \"hc_weights\": {}, \"hc\": {}, \"norm\": {}, \"logits\": {}}},\n        \"selected_token\": {},\n        \"full_logits_c0_bitwise_match\": true\n      }}\n    }}",
            step.position,
            step.input_token,
            step.cache_rows,
            step.transformer_wall_ms,
            step.transformer_gpu_ms,
            step.output_head.fixture_id,
            step.output_head.dispatches,
            step.output_head.command_buffers,
            step.output_head.host_waits,
            step.output_head.wrapped_model_ranges,
            step.output_head.pointer_matches,
            step.output_head.wall_ms,
            step.output_head.gpu_ms,
            step.output_head.hc_pre_checksum,
            step.output_head.hc_weights_checksum,
            step.output_head.hc_checksum,
            step.output_head.norm_checksum,
            step.output_head.logits_checksum,
            step.output_head.selected_token,
        )?;
    }
    write!(
        output,
        "\n  ],\n  \"closed_loop_sampling\": false,\n  \"externally_supplied_decode_inputs\": true,\n  \"full_logits_c0_bitwise_match\": true,\n  \"c0_bitwise_match\": true\n}}\n"
    )?;
    Ok(())
}

pub fn write_ratio128_compressor_replay_probe_json<W: Write>(
    output: &mut W,
    report: &Ratio128CompressorReplayProbeReport,
) -> Result<()> {
    if report.layers.len() != 2
        || report.final_position != 127
        || report.externally_supplied_activation_rows != 128
        || report
            .layers
            .iter()
            .map(|layer| layer.layer)
            .ne([3_u32, 5_u32])
    {
        return Err(Error::invalid(
            "ratio-128 compressor replay report has inconsistent boundary metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{RATIO128_COMPRESSOR_REPLAY_PROBE_SCHEMA}\",\n  \"boundary\": \"oracle-attn-norm-activation-replay\",\n  \"final_position\": {},\n  \"externally_supplied_activation_rows\": {},\n  \"layers\": [",
        report.final_position, report.externally_supplied_activation_rows,
    )?;
    for (index, layer) in report.layers.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(
            output,
            "\n    {{\n      \"layer\": {},\n      \"fixture\": \"{}\",\n      \"activation_rows\": {},\n      \"state_rows\": {},\n      \"dispatches\": {},\n      \"command_buffers\": {},\n      \"host_waits\": {},\n      \"mapping\": {{\"wrapped_model_ranges\": {}, \"pointer_matches\": {}}},\n      \"timing\": {{\"wall_ms\": {:.6}, \"gpu_ms\": {:.6}}},\n      \"checksums\": {{\"attn_norm_sequence\": {}, \"compressed_kv_row\": {}}},\n      \"c0_bitwise_match\": true\n    }}",
            layer.layer,
            layer.fixture_id,
            layer.activation_rows,
            layer.state_rows,
            layer.dispatches,
            layer.command_buffers,
            layer.host_waits,
            layer.wrapped_model_ranges,
            layer.pointer_matches,
            layer.wall_ms,
            layer.gpu_ms,
            layer.input_checksum,
            layer.output_checksum,
        )?;
    }
    write!(
        output,
        "\n  ],\n  \"sampling_performed\": false,\n  \"full_decoder_claim\": false,\n  \"c0_bitwise_match\": true\n}}\n"
    )?;
    Ok(())
}

fn write_position_advancing_probe_json<W: Write>(
    output: &mut W,
    report: &Layers0123DecodeProbeReport,
    schema: &str,
    expected_layers: u32,
) -> Result<()> {
    if report.steps.len() != 3
        || report.command_buffers_per_step != expected_layers
        || report.host_waits_per_step != 1
        || report.kv_cache_layers != expected_layers
        || report.cache_capacity_rows != 4
        || report.output_hc_elements != 16384
    {
        return Err(Error::invalid(
            "position-advancing report has inconsistent ownership metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{}\",\n  \"command_buffers_per_step\": {},\n  \"host_waits_per_step\": {},\n  \"kv_cache_layers\": {},\n  \"cache_capacity_rows\": {},\n  \"output_hc_elements\": {},\n  \"steps\": [",
        schema,
        report.command_buffers_per_step,
        report.host_waits_per_step,
        report.kv_cache_layers,
        report.cache_capacity_rows,
        report.output_hc_elements,
    )?;
    for (step_index, step) in report.steps.iter().enumerate() {
        if step_index != 0 {
            write!(output, ",")?;
        }
        if step.layers.len() != expected_layers as usize || step.cache_rows != step.position + 1 {
            return Err(Error::invalid(
                "position-advancing step has inconsistent layer/cache metadata",
            ));
        }
        write!(
            output,
            "\n    {{\n      \"position\": {},\n      \"token\": {},\n      \"cache_rows\": {},\n      \"wall_ms\": {:.6},\n      \"summed_gpu_ms\": {:.6},\n      \"output_hc_checksum\": {},\n      \"layers\": [",
            step.position,
            step.token,
            step.cache_rows,
            step.wall_ms,
            step.gpu_ms,
            step.output_hc_checksum,
        )?;
        for (layer_index, layer) in step.layers.iter().enumerate() {
            if layer_index != 0 {
                write!(output, ",")?;
            }
            write!(
                output,
                "\n        {{\"layer\": {layer_index}, \"fixture\": \"{}\", \"selected_experts\": {:?}, \"final_hc_checksum\": {}, \"c0_bitwise_match\": true}}",
                layer.fixture_id, layer.selected_experts, layer.final_hc_checksum,
            )?;
        }
        write!(
            output,
            "\n      ],\n      \"c0_bitwise_match\": true\n    }}"
        )?;
    }
    write!(
        output,
        "\n  ],\n  \"cache_growth_exact\": true,\n  \"output_handoff_exact\": true,\n  \"c0_bitwise_match\": true\n}}\n"
    )?;
    Ok(())
}

fn write_timing_samples<W: Write>(output: &mut W, samples: &[f64]) -> Result<()> {
    for (index, sample) in samples.iter().enumerate() {
        if index != 0 {
            write!(output, ", ")?;
        }
        write!(output, "{sample:.6}")?;
    }
    Ok(())
}

fn summarize_timing(samples: &[f64]) -> Result<TimingSummary> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|sample| !sample.is_finite() || *sample < 0.0)
    {
        return Err(Error::invalid("benchmark returned invalid timing samples"));
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let median_ms = median_sorted(&ordered);
    let mut deviations: Vec<f64> = ordered
        .iter()
        .map(|sample| (sample - median_ms).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    Ok(TimingSummary {
        median_ms,
        mad_ms: median_sorted(&deviations),
        min_ms: ordered[0],
        max_ms: ordered[ordered.len() - 1],
    })
}

fn median_sorted(ordered: &[f64]) -> f64 {
    let middle = ordered.len() / 2;
    if ordered.len() % 2 == 0 {
        (ordered[middle - 1] + ordered[middle]) / 2.0
    } else {
        ordered[middle]
    }
}

impl ProbeReport {
    pub fn roundtrip_dispatches_per_second(&self) -> f64 {
        rate(self.iterations, self.roundtrip_wall_ms)
    }

    pub fn batched_dispatches_per_second(&self) -> f64 {
        rate(self.iterations, self.batched_wall_ms)
    }

    pub fn roundtrip_thread_invocations_per_second(&self) -> f64 {
        work_rate(self.elements, self.iterations, self.roundtrip_wall_ms)
    }

    pub fn batched_thread_invocations_per_second(&self) -> f64 {
        work_rate(self.elements, self.iterations, self.batched_wall_ms)
    }
}

fn rate(iterations: u64, milliseconds: f64) -> f64 {
    if milliseconds > 0.0 {
        iterations as f64 * 1000.0 / milliseconds
    } else {
        0.0
    }
}

fn work_rate(elements: u64, iterations: u64, milliseconds: f64) -> f64 {
    if milliseconds > 0.0 {
        elements as f64 * iterations as f64 * 1000.0 / milliseconds
    } else {
        0.0
    }
}

pub fn write_probe_json<W: Write>(output: &mut W, report: &ProbeReport) -> Result<()> {
    output.write_all(b"{\n  \"schema\": \"")?;
    output.write_all(PROBE_SCHEMA.as_bytes())?;
    output.write_all(b"\",\n  \"device\": {\n    \"name\": ")?;
    crate::artifact::write_json_string(output, &report.device_name)?;
    write!(
        output,
        ",\n    \"has_unified_memory\": {},\n    \"recommended_max_working_set_bytes\": {},\n    \"max_total_threads_per_threadgroup\": {}\n  }},\n  \"configuration\": {{\n    \"elements\": {},\n    \"iterations\": {},\n    \"buffer_bytes\": {}\n  }},\n  \"setup_ms\": {:.6},\n  \"compile_ms\": {:.6},\n  \"warmup\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"roundtrip\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"dispatches_per_second\": {:.6},\n    \"thread_invocations_per_second\": {:.6}\n  }},\n  \"batched\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"dispatches_per_second\": {:.6},\n    \"thread_invocations_per_second\": {:.6}\n  }},\n  \"checksum\": {}\n}}\n",
        report.has_unified_memory,
        report.recommended_max_working_set_bytes,
        report.max_total_threads_per_threadgroup,
        report.elements,
        report.iterations,
        report.buffer_bytes,
        report.setup_ms,
        report.compile_ms,
        report.warmup_wall_ms,
        report.warmup_gpu_ms,
        report.roundtrip_wall_ms,
        report.roundtrip_gpu_ms,
        report.roundtrip_dispatches_per_second(),
        report.roundtrip_thread_invocations_per_second(),
        report.batched_wall_ms,
        report.batched_gpu_ms,
        report.batched_dispatches_per_second(),
        report.batched_thread_invocations_per_second(),
        report.checksum,
    )?;
    Ok(())
}

pub fn write_embedding_probe_json<W: Write>(
    output: &mut W,
    report: &EmbeddingProbeReport,
) -> Result<()> {
    output.write_all(b"{\n  \"schema\": \"")?;
    output.write_all(EMBEDDING_PROBE_SCHEMA.as_bytes())?;
    output.write_all(b"\",\n  \"kernel\": \"kernel_get_rows_f16\",\n  \"tensor\": ")?;
    crate::artifact::write_json_string(output, &report.tensor_name)?;
    output.write_all(b",\n  \"tokens\": [")?;
    for (index, token) in report.tokens.iter().enumerate() {
        if index != 0 {
            output.write_all(b", ")?;
        }
        write!(output, "{token}")?;
    }
    write!(
        output,
        "],\n  \"embedding_elements\": {},\n  \"output_elements\": {},\n  \"mapping\": {{\n    \"model_bytes\": {},\n    \"tensor_offset\": {},\n    \"tensor_bytes\": {},\n    \"page_offset\": {},\n    \"buffer_bytes\": {},\n    \"inner_offset\": {},\n    \"max_buffer_length\": {},\n    \"no_copy_pointer_match\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksum\": {},\n  \"c0_bitwise_match\": true\n}}\n",
        report.embedding_elements,
        report.output_elements,
        report.model_bytes,
        report.tensor_offset,
        report.tensor_bytes,
        report.page_offset,
        report.buffer_bytes,
        report.inner_offset,
        report.max_buffer_length,
        report.no_copy_pointer_match,
        report.wall_ms,
        report.gpu_ms,
        report.checksum,
    )?;
    Ok(())
}

pub fn write_projection_probe_json<W: Write>(
    output: &mut W,
    report: &ProjectionProbeReport,
) -> Result<()> {
    output.write_all(b"{\n  \"schema\": \"")?;
    output.write_all(PROJECTION_PROBE_SCHEMA.as_bytes())?;
    output.write_all(b"\",\n  \"fixture\": ")?;
    crate::artifact::write_json_string(output, report.fixture_id)?;
    output.write_all(b",\n  \"kernel\": \"kernel_mul_mv_q8_0_f32\",\n  \"tensor\": ")?;
    crate::artifact::write_json_string(output, &report.tensor_name)?;
    write!(
        output,
        ",\n  \"input_elements\": {},\n  \"output_elements\": {},\n  \"dispatch\": {{\n    \"simdgroups\": {},\n    \"rows_per_threadgroup\": {}\n  }},\n  \"mapping\": {{\n    \"model_bytes\": {},\n    \"tensor_offset\": {},\n    \"tensor_bytes\": {},\n    \"page_offset\": {},\n    \"buffer_bytes\": {},\n    \"inner_offset\": {},\n    \"max_buffer_length\": {},\n    \"no_copy_pointer_match\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"input_checksum\": {},\n  \"output_checksum\": {},\n  \"c0_bitwise_match\": true\n}}\n",
        report.input_elements,
        report.output_elements,
        report.simdgroups,
        report.rows_per_threadgroup,
        report.model_bytes,
        report.tensor_offset,
        report.tensor_bytes,
        report.page_offset,
        report.buffer_bytes,
        report.inner_offset,
        report.max_buffer_length,
        report.no_copy_pointer_match,
        report.wall_ms,
        report.gpu_ms,
        report.input_checksum,
        report.output_checksum,
    )?;
    Ok(())
}

fn validate_embedding_inputs(
    model: &MappedModel,
    tensor: &TensorInfo,
    tokens: &[u32],
) -> Result<(u32, u32, usize)> {
    if tensor.tensor_type.id != 1 {
        return Err(Error::invalid(format!(
            "embedding tensor must be F16, found {}",
            tensor.tensor_type.name
        )));
    }
    if tensor.dimensions.len() != 2 {
        return Err(Error::invalid("embedding tensor must have rank 2"));
    }
    let n_embd = u32::try_from(tensor.dimensions[0])
        .map_err(|_| Error::invalid("embedding width exceeds u32"))?;
    let n_vocab = u32::try_from(tensor.dimensions[1])
        .map_err(|_| Error::invalid("embedding vocabulary exceeds u32"))?;
    if n_embd == 0 || n_vocab == 0 {
        return Err(Error::invalid("embedding dimensions must be nonzero"));
    }
    if tokens.is_empty() || tokens.len() > MAX_EMBEDDING_TOKENS {
        return Err(Error::invalid(format!(
            "embedding probe requires 1..={MAX_EMBEDDING_TOKENS} tokens"
        )));
    }
    for token in tokens {
        if *token >= n_vocab {
            return Err(Error::invalid(format!(
                "embedding token {token} is outside vocabulary {n_vocab}"
            )));
        }
    }
    let expected_bytes = u64::from(n_embd)
        .checked_mul(u64::from(n_vocab))
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Error::invalid("embedding tensor size overflows"))?;
    if tensor.bytes != expected_bytes {
        return Err(Error::invalid(format!(
            "embedding tensor has {} bytes, expected {expected_bytes}",
            tensor.bytes
        )));
    }
    let output_elements = usize::try_from(n_embd)
        .ok()
        .and_then(|width| width.checked_mul(tokens.len()))
        .ok_or_else(|| Error::invalid("embedding probe output size overflows"))?;
    model.tensor_bytes(tensor)?;
    Ok((n_embd, n_vocab, output_elements))
}

fn expected_f16_rows(
    model: &MappedModel,
    tensor: &TensorInfo,
    tokens: &[u32],
    n_embd: u32,
) -> Result<Vec<f32>> {
    let source = model.tensor_bytes(tensor)?;
    let width = n_embd as usize;
    let row_bytes = width
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("embedding row size overflows"))?;
    let mut output = Vec::with_capacity(
        width
            .checked_mul(tokens.len())
            .ok_or_else(|| Error::invalid("embedding reference size overflows"))?,
    );
    for token in tokens {
        let start = (*token as usize)
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::invalid("embedding row offset overflows"))?;
        let row = source
            .get(start..start + row_bytes)
            .ok_or_else(|| Error::invalid("embedding row is outside the tensor"))?;
        for bytes in row.chunks_exact(2) {
            let value = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            if !value.is_finite() {
                return Err(Error::invalid(
                    "embedding tensor contains a non-finite F16 value",
                ));
            }
            output.push(value);
        }
    }
    Ok(output)
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = u32::from(value & 0x03ff);
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << shift) & 0x03ff;
            let f32_exponent = 113_u32 - shift;
            sign | (f32_exponent << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(bits)
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | u16::from(mantissa != 0);
    }
    let mut half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = significand >> shift;
        let remainder = significand & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mantissa & 1) != 0) {
            half_mantissa += 1;
        }
        return sign | half_mantissa as u16;
    }
    let mut half_mantissa = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1) != 0) {
        half_mantissa += 1;
        if half_mantissa == 0x400 {
            half_mantissa = 0;
            half_exponent += 1;
            if half_exponent >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((half_exponent as u16) << 10) | half_mantissa as u16
}

fn f16_round_f32(value: f32) -> f32 {
    f16_to_f32(f32_to_f16(value))
}

fn checksum_f32(values: &[f32]) -> u64 {
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        checksum ^= u64::from(value.to_bits());
        checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
    }
    checksum
}

fn decode_f32_fixture(bytes: &[u8], label: &str) -> Result<Vec<f32>> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(Error::invalid(format!(
            "{label} fixture must contain nonempty little-endian FP32 data"
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        if !value.is_finite() {
            return Err(Error::invalid(format!(
                "{label} fixture contains a non-finite FP32 value"
            )));
        }
        values.push(value);
    }
    Ok(values)
}

struct OutputHeadExpected {
    fixture_id: &'static str,
    hc_pre: Vec<f32>,
    hc_weights: Vec<f32>,
    hc: Vec<f32>,
    norm: Vec<f32>,
    logits: Vec<f32>,
    selected_token: u32,
}

fn output_head_expected(position: u32) -> Result<OutputHeadExpected> {
    let (fixture_id, pre, weights, hc, norm, logits, selected_token) = match position {
        1 => (
            "dwarfstar-oracle-v1-output-head-pos1",
            OUTPUT_HEAD_POS1_PRE_BYTES,
            OUTPUT_HEAD_POS1_WEIGHTS_BYTES,
            OUTPUT_HEAD_POS1_HC_BYTES,
            OUTPUT_HEAD_POS1_NORM_BYTES,
            OUTPUT_HEAD_POS1_LOGITS_BYTES,
            361,
        ),
        2 => (
            "dwarfstar-oracle-v1-output-head-pos2",
            OUTPUT_HEAD_POS2_PRE_BYTES,
            OUTPUT_HEAD_POS2_WEIGHTS_BYTES,
            OUTPUT_HEAD_POS2_HC_BYTES,
            OUTPUT_HEAD_POS2_NORM_BYTES,
            OUTPUT_HEAD_POS2_LOGITS_BYTES,
            1915,
        ),
        3 => (
            "dwarfstar-oracle-v1-output-head-pos3",
            OUTPUT_HEAD_POS3_PRE_BYTES,
            OUTPUT_HEAD_POS3_WEIGHTS_BYTES,
            OUTPUT_HEAD_POS3_HC_BYTES,
            OUTPUT_HEAD_POS3_NORM_BYTES,
            OUTPUT_HEAD_POS3_LOGITS_BYTES,
            262,
        ),
        _ => {
            return Err(Error::invalid(
                "output-head fixtures cover positions 1 through 3",
            ))
        }
    };
    let expected = OutputHeadExpected {
        fixture_id,
        hc_pre: decode_f32_fixture(pre, "output HC pre")?,
        hc_weights: decode_f32_fixture(weights, "output HC weights")?,
        hc: decode_f32_fixture(hc, "output HC")?,
        norm: decode_f32_fixture(norm, "output norm")?,
        logits: decode_f32_fixture(logits, "output logits")?,
        selected_token,
    };
    if expected.hc_pre.len() != 4
        || expected.hc_weights.len() != 4
        || expected.hc.len() != 4096
        || expected.norm.len() != 4096
        || expected.logits.len() != 129280
        || lowest_id_argmax(&expected.logits)? != selected_token
    {
        return Err(Error::invalid(
            "output-head fixture shape or selection is invalid",
        ));
    }
    Ok(expected)
}

fn lowest_id_argmax(values: &[f32]) -> Result<u32> {
    let first = *values
        .first()
        .ok_or_else(|| Error::invalid("cannot select from empty logits"))?;
    if !first.is_finite() {
        return Err(Error::invalid("logits contain a non-finite value"));
    }
    let mut best_id = 0_u32;
    let mut best = first;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(Error::invalid("logits contain a non-finite value"));
        }
        if value > best {
            best = value;
            best_id = index as u32;
        }
    }
    Ok(best_id)
}

fn ratio128_compressor_fixture(layer: u32) -> Result<(&'static str, Vec<f32>, Vec<f32>)> {
    let (fixture_id, inputs, output) = match layer {
        3 => (
            LAYER3_POS127_COMPRESSOR_FIXTURE_ID,
            LAYER3_POS127_COMPRESSOR_INPUT_BYTES,
            LAYER3_POS127_COMPRESSED_KV_BYTES,
        ),
        5 => (
            LAYER5_POS127_COMPRESSOR_FIXTURE_ID,
            LAYER5_POS127_COMPRESSOR_INPUT_BYTES,
            LAYER5_POS127_COMPRESSED_KV_BYTES,
        ),
        _ => {
            return Err(Error::invalid(format!(
                "layer-{layer} ratio-128 compressor replay fixture is not captured"
            )))
        }
    };
    let inputs = decode_f32_fixture(inputs, &format!("layer-{layer} compressor inputs"))?;
    let output = decode_f32_fixture(output, &format!("layer-{layer} compressed KV row"))?;
    if inputs.len() != 128 * 4096 || output.len() != 512 {
        return Err(Error::invalid(format!(
            "layer-{layer} ratio-128 compressor fixture dimensions are invalid"
        )));
    }
    Ok((fixture_id, inputs, output))
}

fn projection_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        decode_f32_fixture(PROJECTION_INPUT_BYTES, "projection input")?,
        decode_f32_fixture(PROJECTION_OUTPUT_BYTES, "projection output")?,
    ))
}

fn ingress_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mixes = decode_f32_fixture(INGRESS_MIXES_BYTES, "ingress HC mixes")?;
    let mut split = decode_f32_fixture(INGRESS_PRE_BYTES, "ingress HC pre weights")?;
    split.extend(decode_f32_fixture(
        INGRESS_POST_BYTES,
        "ingress HC post weights",
    )?);
    split.extend(decode_f32_fixture(
        INGRESS_COMB_BYTES,
        "ingress HC combination",
    )?);
    Ok((
        mixes,
        split,
        decode_f32_fixture(INGRESS_COLLAPSED_BYTES, "ingress HC collapsed")?,
        decode_f32_fixture(INGRESS_NORM_BYTES, "ingress attention norm")?,
        decode_f32_fixture(INGRESS_Q_BYTES, "ingress Q lora")?,
    ))
}

fn attention_setup_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    Ok((
        decode_f32_fixture(SETUP_Q_NORM_BYTES, "setup Q-Lora norm")?,
        decode_f32_fixture(SETUP_KV_RAW_BYTES, "setup KV raw")?,
        decode_f32_fixture(SETUP_KV_NORM_BYTES, "setup KV norm")?,
        decode_f32_fixture(SETUP_Q_RAW_BYTES, "setup Q raw")?,
    ))
}

fn rope_kv_store_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    Ok((
        decode_f32_fixture(ROPE_Q_CUR_BYTES, "RoPE Q current")?,
        decode_f32_fixture(ROPE_KV_ROPE_BYTES, "RoPE KV pre-store")?,
        decode_f32_fixture(ROPE_KV_CUR_BYTES, "RoPE KV post-FP8")?,
        decode_f32_fixture(ROPE_CACHE_ROW_BYTES, "RoPE KV cache row")?,
    ))
}

fn attention_read_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    Ok((
        decode_f32_fixture(ATTENTION_CACHE_ROW0_BYTES, "attention cache row 0")?,
        decode_f32_fixture(ATTENTION_CACHE_ROW1_BYTES, "attention cache row 1")?,
        decode_f32_fixture(ATTENTION_BACK_BYTES, "attention inverse-RoPE output")?,
    ))
}

fn attention_output_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    Ok((
        decode_f32_fixture(OUTPUT_KQV_BACK_BYTES, "output KQv back")?,
        decode_f32_fixture(OUTPUT_LOW_BYTES, "attention output low")?,
        decode_f32_fixture(OUTPUT_ATTN_BYTES, "attention output projection")?,
        decode_f32_fixture(OUTPUT_HC_POST_BYTES, "attention HC post-state")?,
    ))
}

type FfnRouterFixture = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<i32>,
    Vec<f32>,
);

fn ffn_router_fixture() -> Result<FfnRouterFixture> {
    let mut split = decode_f32_fixture(FFN_PRE_BYTES, "FFN HC pre weights")?;
    split.extend(decode_f32_fixture(FFN_POST_BYTES, "FFN HC post weights")?);
    split.extend(decode_f32_fixture(FFN_COMB_BYTES, "FFN HC combination")?);
    Ok((
        decode_f32_fixture(FFN_INPUT_HC_BYTES, "FFN input HC state")?,
        decode_f32_fixture(FFN_MIXES_BYTES, "FFN HC mixes")?,
        split,
        decode_f32_fixture(FFN_CUR_BYTES, "FFN collapsed state")?,
        decode_f32_fixture(FFN_NORM_BYTES, "FFN normalized state")?,
        decode_f32_fixture(ROUTER_LOGITS_BYTES, "router logits")?,
        decode_f32_fixture(ROUTER_PROBS_BYTES, "router probabilities")?,
        decode_i32_fixture(ROUTER_SELECTED_BYTES, "router selected experts")?,
        decode_f32_fixture(ROUTER_WEIGHTS_BYTES, "router weights")?,
    ))
}

type MoeOutputFixture = (
    Vec<f32>,
    Vec<i32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
);

fn moe_output_fixture() -> Result<MoeOutputFixture> {
    let mut split = decode_f32_fixture(MOE_PRE_BYTES, "MoE HC pre weights")?;
    split.extend(decode_f32_fixture(MOE_POST_BYTES, "MoE HC post weights")?);
    split.extend(decode_f32_fixture(MOE_COMB_BYTES, "MoE HC combination")?);
    Ok((
        decode_f32_fixture(MOE_FFN_NORM_BYTES, "MoE normalized input")?,
        decode_i32_fixture(MOE_SELECTED_BYTES, "MoE selected experts")?,
        decode_f32_fixture(MOE_WEIGHTS_BYTES, "MoE router weights")?,
        decode_f32_fixture(MOE_INPUT_HC_BYTES, "MoE residual HC state")?,
        split,
        decode_f32_fixture(MOE_ROUTED_MID_BYTES, "MoE routed activation")?,
        decode_f32_fixture(MOE_ROUTED_OUT_BYTES, "MoE routed output")?,
        decode_f32_fixture(MOE_SHARED_OUT_BYTES, "MoE shared output")?,
        decode_f32_fixture(MOE_HC_POST_BYTES, "MoE HC post-state")?,
    ))
}

struct LayerExpected {
    fixture_id: &'static str,
    cache_row0: Vec<f32>,
    attention_mixes: Vec<f32>,
    attention_split: Vec<f32>,
    attention_collapsed: Vec<f32>,
    attention_norm: Vec<f32>,
    q_lora: Vec<f32>,
    q_lora_norm: Vec<f32>,
    kv_raw: Vec<f32>,
    q_raw: Vec<f32>,
    q_cur: Vec<f32>,
    kv_rope: Vec<f32>,
    kv_cur: Vec<f32>,
    attention_back: Vec<f32>,
    attention_low: Vec<f32>,
    attention_out: Vec<f32>,
    attention_hc: Vec<f32>,
    ffn_mixes: Vec<f32>,
    ffn_split: Vec<f32>,
    ffn_norm: Vec<f32>,
    router_logits: Vec<f32>,
    router_probs: Vec<f32>,
    selected: Vec<i32>,
    router_weights: Vec<f32>,
    routed_mid: Vec<f32>,
    routed_out: Vec<f32>,
    shared_out: Vec<f32>,
    final_hc: Vec<f32>,
    compressed_kv: Vec<f32>,
}

struct LayerExpectedBytes {
    fixture_id: &'static str,
    attention_mixes: &'static [u8],
    attention_pre: &'static [u8],
    attention_post: &'static [u8],
    attention_comb: &'static [u8],
    attention_collapsed: &'static [u8],
    attention_norm: &'static [u8],
    q_lora: &'static [u8],
    q_lora_norm: &'static [u8],
    kv_raw: &'static [u8],
    q_raw: &'static [u8],
    q_cur: &'static [u8],
    kv_rope: &'static [u8],
    kv_cur: &'static [u8],
    attention_back: &'static [u8],
    attention_low: &'static [u8],
    attention_out: &'static [u8],
    attention_hc: &'static [u8],
    ffn_mixes: &'static [u8],
    ffn_pre: &'static [u8],
    ffn_post: &'static [u8],
    ffn_comb: &'static [u8],
    ffn_norm: &'static [u8],
    router_logits: &'static [u8],
    router_probs: &'static [u8],
    selected: &'static [u8],
    router_weights: &'static [u8],
    routed_mid: &'static [u8],
    routed_out: &'static [u8],
    shared_out: &'static [u8],
    final_hc: &'static [u8],
}

macro_rules! complete_decode_fixture {
    ($name:ident, $layer:literal, $position:literal, $fixture_id:expr) => {
        const $name: LayerExpectedBytes =
    { complete_decode_fixture!(@value $layer, $position, $fixture_id) };
    };
    (@value $layer:literal, $position:literal, $fixture_id:expr) => {
        LayerExpectedBytes {
            fixture_id: $fixture_id,
            attention_mixes: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-mixes.f32le.bin"
            )),
            attention_pre: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-pre.f32le.bin"
            )),
            attention_post: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-post.f32le.bin"
            )),
            attention_comb: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-combination.f32le.bin"
            )),
            attention_collapsed: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-collapsed.f32le.bin"
            )),
            attention_norm: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-norm.f32le.bin"
            )),
            q_lora: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/q-lora.f32le.bin"
            )),
            q_lora_norm: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/q-lora-norm.f32le.bin"
            )),
            kv_raw: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/kv-raw.f32le.bin"
            )),
            q_raw: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/q-raw.f32le.bin"
            )),
            q_cur: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/q-cur.f32le.bin"
            )),
            kv_rope: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/kv-rope.f32le.bin"
            )),
            kv_cur: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/kv-cur.f32le.bin"
            )),
            attention_back: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/kqv-back.f32le.bin"
            )),
            attention_low: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-low.f32le.bin"
            )),
            attention_out: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/attn-out.f32le.bin"
            )),
            attention_hc: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/hc-attn-post.f32le.bin"
            )),
            ffn_mixes: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/ffn-mixes.f32le.bin"
            )),
            ffn_pre: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/ffn-pre.f32le.bin"
            )),
            ffn_post: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/ffn-post.f32le.bin"
            )),
            ffn_comb: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/ffn-combination.f32le.bin"
            )),
            ffn_norm: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/ffn-norm.f32le.bin"
            )),
            router_logits: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/router-logits.f32le.bin"
            )),
            router_probs: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/router-probs.f32le.bin"
            )),
            selected: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/router-selected.i32le.bin"
            )),
            router_weights: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/router-weights.f32le.bin"
            )),
            routed_mid: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/routed-mid.f32le.bin"
            )),
            routed_out: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/routed-out.f32le.bin"
            )),
            shared_out: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/shared-out.f32le.bin"
            )),
            final_hc: include_bytes!(concat!(
                "../../fixtures/layer",
                stringify!($layer),
                "-pos",
                stringify!($position),
                "-complete-v1/hc-ffn-post.f32le.bin"
            )),
        }
    };
}

complete_decode_fixture!(LAYER0_POS2_BYTES, 0, 2, LAYER0_POS2_FIXTURE_ID);
complete_decode_fixture!(LAYER1_POS2_BYTES, 1, 2, LAYER1_POS2_FIXTURE_ID);
complete_decode_fixture!(LAYER2_POS2_BYTES, 2, 2, LAYER2_POS2_FIXTURE_ID);
complete_decode_fixture!(LAYER3_POS2_BYTES, 3, 2, LAYER3_POS2_FIXTURE_ID);
complete_decode_fixture!(LAYER0_POS3_BYTES, 0, 3, LAYER0_POS3_FIXTURE_ID);
complete_decode_fixture!(LAYER1_POS3_BYTES, 1, 3, LAYER1_POS3_FIXTURE_ID);
complete_decode_fixture!(LAYER2_POS3_BYTES, 2, 3, LAYER2_POS3_FIXTURE_ID);
complete_decode_fixture!(LAYER3_POS3_BYTES, 3, 3, LAYER3_POS3_FIXTURE_ID);

macro_rules! complete_decode_fixture_registry {
    ($($layer:literal),+ $(,)?) => {
        const LATER_POS1_BYTES: &[LayerExpectedBytes] = &[
            $(complete_decode_fixture!(@value $layer, 1,
                concat!("dwarfstar-oracle-v1-layer", stringify!($layer), "-pos1-complete"))),+
        ];
        const LATER_POS2_BYTES: &[LayerExpectedBytes] = &[
            $(complete_decode_fixture!(@value $layer, 2,
                concat!("dwarfstar-oracle-v1-layer", stringify!($layer), "-pos2-complete"))),+
        ];
        const LATER_POS3_BYTES: &[LayerExpectedBytes] = &[
            $(complete_decode_fixture!(@value $layer, 3,
                concat!("dwarfstar-oracle-v1-layer", stringify!($layer), "-pos3-complete"))),+
        ];
        const LATER_POS0_COMPRESSOR_PRIME_BYTES: &[&[u8]] = &[
            $(include_bytes!(concat!(
                "../../fixtures/layer", stringify!($layer),
                "-pos0-compressor-prime-v1/attn-norm.f32le.bin"
            ))),+
        ];
        const LATER_CACHE_ROW0_BYTES: &[&[u8]] = &[
            $(include_bytes!(concat!(
                "../../fixtures/layer", stringify!($layer),
                "-pos1-complete-v1/cache-row0.f32le.bin"
            ))),+
        ];
    };
}

complete_decode_fixture_registry!(
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
);

macro_rules! compressed_kv_fixture_registry {
    ($($layer:literal),+ $(,)?) => {
        const LATER_POS3_COMPRESSED_KV_BYTES: &[&[u8]] = &[
            $(include_bytes!(concat!(
                "../../fixtures/layer", stringify!($layer),
                "-pos3-complete-v1/compressed-kv-row0.f32le.bin"
            ))),+
        ];
    };
}

compressed_kv_fixture_registry!(
    4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42
);

const LAYER2_POS3_COMPRESSED_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-pos3-complete-v1/compressed-kv-row0.f32le.bin");
const LAYER2_POS0_COMPRESSOR_PRIME_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer2-pos0-compressor-prime-v1/attn-norm.f32le.bin");
const LAYER3_POS0_COMPRESSOR_PRIME_BYTES: &[u8] =
    include_bytes!("../../fixtures/layer3-pos0-compressor-prime-v1/attn-norm.f32le.bin");
const LAYER3_POS127_COMPRESSOR_INPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/layer3-pos127-compressor-replay-v1/attn-norm-sequence.f32le.bin"
);
const LAYER3_POS127_COMPRESSED_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/layer3-pos127-compressor-replay-v1/compressed-kv-row0.f32le.bin"
);
const LAYER5_POS127_COMPRESSOR_INPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/layer5-pos127-compressor-replay-v1/attn-norm-sequence.f32le.bin"
);
const LAYER5_POS127_COMPRESSED_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/layer5-pos127-compressor-replay-v1/compressed-kv-row0.f32le.bin"
);

fn later_fixture_index(layer_index: u32) -> Option<usize> {
    if (4..=42).contains(&layer_index) {
        Some((layer_index - 4) as usize)
    } else {
        None
    }
}

fn compressor_prime_bytes(layer_index: u32) -> Option<&'static [u8]> {
    match layer_index {
        2 => Some(LAYER2_POS0_COMPRESSOR_PRIME_BYTES),
        3 => Some(LAYER3_POS0_COMPRESSOR_PRIME_BYTES),
        _ => later_fixture_index(layer_index).map(|index| LATER_POS0_COMPRESSOR_PRIME_BYTES[index]),
    }
}

fn compressed_kv_bytes(layer_index: u32) -> Option<&'static [u8]> {
    if (4..=42).contains(&layer_index) && layer_index % 2 == 0 {
        Some(LATER_POS3_COMPRESSED_KV_BYTES[((layer_index - 4) / 2) as usize])
    } else {
        None
    }
}

fn apply_complete_decode_fixture(
    layer_index: u32,
    position: u32,
    mut expected: LayerExpected,
) -> Result<LayerExpected> {
    let bytes = if let Some(index) = later_fixture_index(layer_index) {
        match position {
            1 => &LATER_POS1_BYTES[index],
            2 => &LATER_POS2_BYTES[index],
            3 => &LATER_POS3_BYTES[index],
            _ => {
                return Err(Error::invalid(
                    "complete decode fixture requires position one through three",
                ))
            }
        }
    } else {
        match (position, layer_index) {
            (2, 0) => &LAYER0_POS2_BYTES,
            (2, 1) => &LAYER1_POS2_BYTES,
            (2, 2) => &LAYER2_POS2_BYTES,
            (2, 3) => &LAYER3_POS2_BYTES,
            (3, 0) => &LAYER0_POS3_BYTES,
            (3, 1) => &LAYER1_POS3_BYTES,
            (3, 2) => &LAYER2_POS3_BYTES,
            (3, 3) => &LAYER3_POS3_BYTES,
            _ => {
                return Err(Error::invalid(
                    "complete decode fixture is outside the retained layer/position frontier",
                ))
            }
        }
    };
    let label = format!("layer-{layer_index} position-{position}");
    let decode = |data, name: &str| decode_f32_fixture(data, &format!("{label} {name}"));
    expected.fixture_id = bytes.fixture_id;
    expected.attention_mixes = decode(bytes.attention_mixes, "attention HC mixes")?;
    expected.attention_split = decode(bytes.attention_pre, "attention HC pre weights")?;
    expected
        .attention_split
        .extend(decode(bytes.attention_post, "attention HC post weights")?);
    expected
        .attention_split
        .extend(decode(bytes.attention_comb, "attention HC combination")?);
    expected.attention_collapsed = decode(bytes.attention_collapsed, "attention collapsed")?;
    expected.attention_norm = decode(bytes.attention_norm, "attention norm")?;
    expected.q_lora = decode(bytes.q_lora, "Q-Lora")?;
    expected.q_lora_norm = decode(bytes.q_lora_norm, "normalized Q-Lora")?;
    expected.kv_raw = decode(bytes.kv_raw, "raw KV")?;
    expected.q_raw = decode(bytes.q_raw, "raw Q")?;
    expected.q_cur = decode(bytes.q_cur, "Q current")?;
    expected.kv_rope = decode(bytes.kv_rope, "KV before FP8 store")?;
    expected.kv_cur = decode(bytes.kv_cur, "KV current")?;
    expected.attention_back = decode(bytes.attention_back, "inverse-RoPE attention")?;
    expected.attention_low = decode(bytes.attention_low, "low-rank attention output")?;
    expected.attention_out = decode(bytes.attention_out, "attention output")?;
    expected.attention_hc = decode(bytes.attention_hc, "attention HC")?;
    expected.ffn_mixes = decode(bytes.ffn_mixes, "FFN HC mixes")?;
    expected.ffn_split = decode(bytes.ffn_pre, "FFN HC pre weights")?;
    expected
        .ffn_split
        .extend(decode(bytes.ffn_post, "FFN HC post weights")?);
    expected
        .ffn_split
        .extend(decode(bytes.ffn_comb, "FFN HC combination")?);
    expected.ffn_norm = decode(bytes.ffn_norm, "FFN norm")?;
    expected.router_logits = decode(bytes.router_logits, "router logits")?;
    expected.router_probs = decode(bytes.router_probs, "router probabilities")?;
    expected.selected = decode_i32_fixture(bytes.selected, &format!("{label} selected experts"))?;
    expected.router_weights = decode(bytes.router_weights, "router weights")?;
    expected.routed_mid = decode(bytes.routed_mid, "routed activation")?;
    expected.routed_out = decode(bytes.routed_out, "routed output")?;
    expected.shared_out = decode(bytes.shared_out, "shared output")?;
    expected.final_hc = decode(bytes.final_hc, "final HC")?;
    expected.compressed_kv = if position == 3 && layer_index == 2 {
        decode_f32_fixture(
            LAYER2_POS3_COMPRESSED_KV_BYTES,
            "layer-2 position-3 compressed KV row",
        )?
    } else if position == 3 {
        match compressed_kv_bytes(layer_index) {
            Some(bytes) => decode_f32_fixture(
                bytes,
                &format!("layer-{layer_index} position-3 compressed KV row"),
            )?,
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };
    Ok(expected)
}

fn layer_expected(layer_index: u32, position: u32) -> Result<LayerExpected> {
    if position == 2 || position == 3 {
        return apply_complete_decode_fixture(
            layer_index,
            position,
            layer_expected(layer_index, 1)?,
        );
    }
    if position == 1 && later_fixture_index(layer_index).is_some() {
        let mut expected =
            apply_complete_decode_fixture(layer_index, position, layer_expected(3, position)?)?;
        let cache_row = LATER_CACHE_ROW0_BYTES[later_fixture_index(layer_index).unwrap()];
        expected.cache_row0 =
            decode_f32_fixture(cache_row, &format!("layer-{layer_index} cache row 0"))?;
        return Ok(expected);
    }
    if position != 1 {
        return Err(Error::invalid(
            "complete layer fixtures currently cover decode position 1",
        ));
    }
    if layer_index == 0 {
        let (attention_mixes, attention_split, attention_collapsed, attention_norm, q_lora) =
            ingress_fixture()?;
        let (q_lora_norm, kv_raw, _, q_raw) = attention_setup_fixture()?;
        let (cache_row0, _, attention_back) = attention_read_fixture()?;
        let (q_cur, kv_rope, kv_cur, _) = rope_kv_store_fixture()?;
        let (_, attention_low, attention_out, attention_hc) = attention_output_fixture()?;
        let (
            _,
            ffn_mixes,
            ffn_split,
            _,
            ffn_norm,
            router_logits,
            router_probs,
            selected,
            router_weights,
        ) = ffn_router_fixture()?;
        let (_, _, _, _, _, routed_mid, routed_out, shared_out, final_hc) = moe_output_fixture()?;
        return Ok(LayerExpected {
            fixture_id: LAYER0_FIXTURE_ID,
            cache_row0,
            attention_mixes,
            attention_split,
            attention_collapsed,
            attention_norm,
            q_lora,
            q_lora_norm,
            kv_raw,
            q_raw,
            q_cur,
            kv_rope,
            kv_cur,
            attention_back,
            attention_low,
            attention_out,
            attention_hc,
            ffn_mixes,
            ffn_split,
            ffn_norm,
            router_logits,
            router_probs,
            selected,
            router_weights,
            routed_mid,
            routed_out,
            shared_out,
            final_hc,
            compressed_kv: Vec::new(),
        });
    }
    let (
        fixture_id,
        label,
        cache_row0,
        attention_mixes,
        attention_pre,
        attention_post,
        attention_comb,
        attention_collapsed,
        attention_norm,
        q_lora,
        q_lora_norm,
        kv_raw,
        q_raw,
        q_cur,
        kv_rope,
        kv_cur,
        attention_back,
        attention_low,
        attention_out,
        attention_hc,
        ffn_mixes,
        ffn_pre,
        ffn_post,
        ffn_comb,
        ffn_norm,
        router_logits,
        router_probs,
        selected,
        router_weights,
        routed_mid,
        routed_out,
        shared_out,
        final_hc,
    ) = match layer_index {
        1 => (
            LAYER1_FIXTURE_ID,
            "layer-1",
            LAYER1_CACHE_ROW0_BYTES,
            LAYER1_ATTN_MIXES_BYTES,
            LAYER1_ATTN_PRE_BYTES,
            LAYER1_ATTN_POST_BYTES,
            LAYER1_ATTN_COMB_BYTES,
            LAYER1_ATTN_COLLAPSED_BYTES,
            LAYER1_ATTN_NORM_BYTES,
            LAYER1_Q_LORA_BYTES,
            LAYER1_Q_LORA_NORM_BYTES,
            LAYER1_KV_RAW_BYTES,
            LAYER1_Q_RAW_BYTES,
            LAYER1_Q_CUR_BYTES,
            LAYER1_KV_ROPE_BYTES,
            LAYER1_KV_CUR_BYTES,
            LAYER1_KQV_BACK_BYTES,
            LAYER1_ATTN_LOW_BYTES,
            LAYER1_ATTN_OUT_BYTES,
            LAYER1_ATTN_HC_BYTES,
            LAYER1_FFN_MIXES_BYTES,
            LAYER1_FFN_PRE_BYTES,
            LAYER1_FFN_POST_BYTES,
            LAYER1_FFN_COMB_BYTES,
            LAYER1_FFN_NORM_BYTES,
            LAYER1_ROUTER_LOGITS_BYTES,
            LAYER1_ROUTER_PROBS_BYTES,
            LAYER1_SELECTED_BYTES,
            LAYER1_ROUTER_WEIGHTS_BYTES,
            LAYER1_ROUTED_MID_BYTES,
            LAYER1_ROUTED_OUT_BYTES,
            LAYER1_SHARED_OUT_BYTES,
            LAYER1_FINAL_HC_BYTES,
        ),
        2 => (
            LAYER2_FIXTURE_ID,
            "layer-2",
            LAYER2_CACHE_ROW0_BYTES,
            LAYER2_ATTN_MIXES_BYTES,
            LAYER2_ATTN_PRE_BYTES,
            LAYER2_ATTN_POST_BYTES,
            LAYER2_ATTN_COMB_BYTES,
            LAYER2_ATTN_COLLAPSED_BYTES,
            LAYER2_ATTN_NORM_BYTES,
            LAYER2_Q_LORA_BYTES,
            LAYER2_Q_LORA_NORM_BYTES,
            LAYER2_KV_RAW_BYTES,
            LAYER2_Q_RAW_BYTES,
            LAYER2_Q_CUR_BYTES,
            LAYER2_KV_ROPE_BYTES,
            LAYER2_KV_CUR_BYTES,
            LAYER2_KQV_BACK_BYTES,
            LAYER2_ATTN_LOW_BYTES,
            LAYER2_ATTN_OUT_BYTES,
            LAYER2_ATTN_HC_BYTES,
            LAYER2_FFN_MIXES_BYTES,
            LAYER2_FFN_PRE_BYTES,
            LAYER2_FFN_POST_BYTES,
            LAYER2_FFN_COMB_BYTES,
            LAYER2_FFN_NORM_BYTES,
            LAYER2_ROUTER_LOGITS_BYTES,
            LAYER2_ROUTER_PROBS_BYTES,
            LAYER2_SELECTED_BYTES,
            LAYER2_ROUTER_WEIGHTS_BYTES,
            LAYER2_ROUTED_MID_BYTES,
            LAYER2_ROUTED_OUT_BYTES,
            LAYER2_SHARED_OUT_BYTES,
            LAYER2_FINAL_HC_BYTES,
        ),
        3 => (
            LAYER3_FIXTURE_ID,
            "layer-3",
            LAYER3_CACHE_ROW0_BYTES,
            LAYER3_ATTN_MIXES_BYTES,
            LAYER3_ATTN_PRE_BYTES,
            LAYER3_ATTN_POST_BYTES,
            LAYER3_ATTN_COMB_BYTES,
            LAYER3_ATTN_COLLAPSED_BYTES,
            LAYER3_ATTN_NORM_BYTES,
            LAYER3_Q_LORA_BYTES,
            LAYER3_Q_LORA_NORM_BYTES,
            LAYER3_KV_RAW_BYTES,
            LAYER3_Q_RAW_BYTES,
            LAYER3_Q_CUR_BYTES,
            LAYER3_KV_ROPE_BYTES,
            LAYER3_KV_CUR_BYTES,
            LAYER3_KQV_BACK_BYTES,
            LAYER3_ATTN_LOW_BYTES,
            LAYER3_ATTN_OUT_BYTES,
            LAYER3_ATTN_HC_BYTES,
            LAYER3_FFN_MIXES_BYTES,
            LAYER3_FFN_PRE_BYTES,
            LAYER3_FFN_POST_BYTES,
            LAYER3_FFN_COMB_BYTES,
            LAYER3_FFN_NORM_BYTES,
            LAYER3_ROUTER_LOGITS_BYTES,
            LAYER3_ROUTER_PROBS_BYTES,
            LAYER3_SELECTED_BYTES,
            LAYER3_ROUTER_WEIGHTS_BYTES,
            LAYER3_ROUTED_MID_BYTES,
            LAYER3_ROUTED_OUT_BYTES,
            LAYER3_SHARED_OUT_BYTES,
            LAYER3_FINAL_HC_BYTES,
        ),

        _ => {
            return Err(Error::invalid(
                "the persistent layer executor currently supports layers 0 through 3",
            ))
        }
    };
    let mut attention_split =
        decode_f32_fixture(attention_pre, &format!("{label} attention HC pre weights"))?;
    attention_split.extend(decode_f32_fixture(
        attention_post,
        &format!("{label} attention HC post weights"),
    )?);
    attention_split.extend(decode_f32_fixture(
        attention_comb,
        &format!("{label} attention HC combination"),
    )?);
    let mut ffn_split = decode_f32_fixture(ffn_pre, &format!("{label} FFN HC pre weights"))?;
    ffn_split.extend(decode_f32_fixture(
        ffn_post,
        &format!("{label} FFN HC post weights"),
    )?);
    ffn_split.extend(decode_f32_fixture(
        ffn_comb,
        &format!("{label} FFN HC combination"),
    )?);
    Ok(LayerExpected {
        fixture_id,
        cache_row0: decode_f32_fixture(cache_row0, &format!("{label} cache row 0"))?,
        attention_mixes: decode_f32_fixture(
            attention_mixes,
            &format!("{label} attention HC mixes"),
        )?,
        attention_split,
        attention_collapsed: decode_f32_fixture(
            attention_collapsed,
            &format!("{label} attention collapsed state"),
        )?,
        attention_norm: decode_f32_fixture(attention_norm, &format!("{label} attention norm"))?,
        q_lora: decode_f32_fixture(q_lora, &format!("{label} Q-Lora"))?,
        q_lora_norm: decode_f32_fixture(q_lora_norm, &format!("{label} normalized Q-Lora"))?,
        kv_raw: decode_f32_fixture(kv_raw, &format!("{label} raw KV"))?,
        q_raw: decode_f32_fixture(q_raw, &format!("{label} raw Q"))?,
        q_cur: decode_f32_fixture(q_cur, &format!("{label} Q current"))?,
        kv_rope: decode_f32_fixture(kv_rope, &format!("{label} KV before FP8 store"))?,
        kv_cur: decode_f32_fixture(kv_cur, &format!("{label} KV current"))?,
        attention_back: decode_f32_fixture(
            attention_back,
            &format!("{label} inverse-RoPE attention output"),
        )?,
        attention_low: decode_f32_fixture(
            attention_low,
            &format!("{label} low-rank attention output"),
        )?,
        attention_out: decode_f32_fixture(attention_out, &format!("{label} attention output"))?,
        attention_hc: decode_f32_fixture(attention_hc, &format!("{label} attention HC"))?,
        ffn_mixes: decode_f32_fixture(ffn_mixes, &format!("{label} FFN HC mixes"))?,
        ffn_split,
        ffn_norm: decode_f32_fixture(ffn_norm, &format!("{label} FFN norm"))?,
        router_logits: decode_f32_fixture(router_logits, &format!("{label} router logits"))?,
        router_probs: decode_f32_fixture(router_probs, &format!("{label} router probabilities"))?,
        selected: decode_i32_fixture(selected, &format!("{label} selected experts"))?,
        router_weights: decode_f32_fixture(router_weights, &format!("{label} router weights"))?,
        routed_mid: decode_f32_fixture(routed_mid, &format!("{label} routed activation"))?,
        routed_out: decode_f32_fixture(routed_out, &format!("{label} routed output"))?,
        shared_out: decode_f32_fixture(shared_out, &format!("{label} shared output"))?,
        final_hc: decode_f32_fixture(final_hc, &format!("{label} final HC"))?,
        compressed_kv: Vec::new(),
    })
}

fn decode_i32_fixture(bytes: &[u8], label: &str) -> Result<Vec<i32>> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(Error::invalid(format!(
            "{label} fixture must contain nonempty little-endian I32 data"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn exact_tensor<'a>(
    model: &'a MappedModel,
    name: &str,
    kind: u32,
    dimensions: &[u64],
) -> Result<&'a TensorInfo> {
    let tensor = model.tensor(name)?;
    if tensor.tensor_type.id != kind || tensor.dimensions != dimensions {
        return Err(Error::invalid(format!(
            "attention ingress tensor {name} has unexpected type or dimensions"
        )));
    }
    model.tensor_bytes(tensor)?;
    Ok(tensor)
}

fn validate_projection_inputs(
    model: &MappedModel,
    tensor: &TensorInfo,
    input: &[f32],
    expected: &[f32],
) -> Result<(u32, u32)> {
    if tensor.name != PROJECTION_TENSOR {
        return Err(Error::invalid(format!(
            "projection tensor must be {PROJECTION_TENSOR}"
        )));
    }
    if tensor.tensor_type.id != 8 {
        return Err(Error::invalid(format!(
            "projection tensor must be Q8_0, found {}",
            tensor.tensor_type.name
        )));
    }
    if tensor.dimensions.len() != 2 {
        return Err(Error::invalid("projection tensor must have rank 2"));
    }
    let input_elements = u32::try_from(tensor.dimensions[0])
        .map_err(|_| Error::invalid("projection input width exceeds u32"))?;
    let output_elements = u32::try_from(tensor.dimensions[1])
        .map_err(|_| Error::invalid("projection output width exceeds u32"))?;
    if input_elements == 0 || input_elements % 32 != 0 || output_elements == 0 {
        return Err(Error::invalid("projection tensor dimensions are invalid"));
    }
    if input.len() != input_elements as usize || expected.len() != output_elements as usize {
        return Err(Error::invalid(format!(
            "projection fixture dimensions differ from tensor {}x{}",
            input_elements, output_elements
        )));
    }
    let row_bytes = u64::from(input_elements / 32)
        .checked_mul(34)
        .ok_or_else(|| Error::invalid("projection row size overflows"))?;
    let expected_bytes = row_bytes
        .checked_mul(u64::from(output_elements))
        .ok_or_else(|| Error::invalid("projection tensor size overflows"))?;
    if tensor.bytes != expected_bytes {
        return Err(Error::invalid(format!(
            "projection tensor has {} bytes, expected {expected_bytes}",
            tensor.bytes
        )));
    }
    model.tensor_bytes(tensor)?;
    Ok((input_elements, output_elements))
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr;

    const ERROR_BYTES: usize = 1024;
    const COMMAND_SYNCHRONIZED: u32 = 0;
    const COMMAND_CHAINED_ENQUEUE: u32 = 1;
    const COMMAND_CHAINED_FINAL: u32 = 2;
    const COMMAND_CHAINED_COLLECT: u32 = 3;
    const COMMAND_CHAINED_TIMING: u32 = 4;

    #[repr(C)]
    struct RawProbeResult {
        elements: u64,
        iterations: u64,
        recommended_max_working_set_bytes: u64,
        buffer_bytes: u64,
        max_total_threads_per_threadgroup: u64,
        checksum: u64,
        has_unified_memory: u32,
        reserved: u32,
        setup_ms: f64,
        compile_ms: f64,
        warmup_wall_ms: f64,
        warmup_gpu_ms: f64,
        roundtrip_wall_ms: f64,
        roundtrip_gpu_ms: f64,
        batched_wall_ms: f64,
        batched_gpu_ms: f64,
        device_name: [c_char; 256],
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawEmbeddingProbeResult {
        model_bytes: u64,
        tensor_offset: u64,
        tensor_bytes: u64,
        page_offset: u64,
        buffer_bytes: u64,
        inner_offset: u64,
        output_elements: u64,
        max_buffer_length: u64,
        no_copy_pointer_match: u32,
        reserved: u32,
        wall_ms: f64,
        gpu_ms: f64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawProjectionProbeResult {
        model_bytes: u64,
        tensor_offset: u64,
        tensor_bytes: u64,
        page_offset: u64,
        buffer_bytes: u64,
        inner_offset: u64,
        input_elements: u64,
        output_elements: u64,
        max_buffer_length: u64,
        no_copy_pointer_match: u32,
        simdgroups: u32,
        rows_per_threadgroup: u32,
        reserved: u32,
        wall_ms: f64,
        gpu_ms: f64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawIngressProbeResult {
        model_bytes: u64,
        max_buffer_length: u64,
        wrapped_model_ranges: u32,
        pointer_matches: u32,
        wall_ms: f64,
        gpu_ms: f64,
    }

    #[repr(C)]
    struct RawLayer0Extension {
        hc_ffn_fn_offset: u64,
        hc_ffn_fn_bytes: u64,
        hc_ffn_scale_offset: u64,
        hc_ffn_scale_bytes: u64,
        hc_ffn_base_offset: u64,
        hc_ffn_base_bytes: u64,
        ffn_norm_offset: u64,
        ffn_norm_bytes: u64,
        router_gate_offset: u64,
        router_gate_bytes: u64,
        router_aux_offset: u64,
        router_aux_bytes: u64,
        routed_gate_offset: u64,
        routed_gate_bytes: u64,
        routed_up_offset: u64,
        routed_up_bytes: u64,
        routed_down_offset: u64,
        routed_down_bytes: u64,
        shared_gate_offset: u64,
        shared_gate_bytes: u64,
        shared_up_offset: u64,
        shared_up_bytes: u64,
        shared_down_offset: u64,
        shared_down_bytes: u64,
        attn_compressor_ape_offset: u64,
        attn_compressor_ape_bytes: u64,
        attn_compressor_kv_offset: u64,
        attn_compressor_kv_bytes: u64,
        attn_compressor_gate_offset: u64,
        attn_compressor_gate_bytes: u64,
        attn_compressor_norm_offset: u64,
        attn_compressor_norm_bytes: u64,
        indexer_compressor_ape_offset: u64,
        indexer_compressor_ape_bytes: u64,
        indexer_compressor_kv_offset: u64,
        indexer_compressor_kv_bytes: u64,
        indexer_compressor_gate_offset: u64,
        indexer_compressor_gate_bytes: u64,
        indexer_compressor_norm_offset: u64,
        indexer_compressor_norm_bytes: u64,
        compressor_prime_attn_norm: *const f32,
        compressed_kv_row: *mut f32,
        compressed_indexer_row: *mut f32,
        ffn_mixes: *mut f32,
        ffn_split: *mut f32,
        ffn_norm: *mut f32,
        router_logits: *mut f32,
        router_probs: *mut f32,
        selected: *mut i32,
        router_weights: *mut f32,
        routed_mid: *mut f32,
        routed_out: *mut f32,
        shared_out: *mut f32,
        after_ffn_hc: *mut f32,
        warmup_iterations: u32,
        measured_iterations: u32,
        wall_ms_samples: *mut f64,
        gpu_ms_samples: *mut f64,
        repeat_bitwise_matches: *mut u32,
        layer_index: u32,
        reuse_previous_hc: u32,
        command_mode: u32,
        chain_final_layer: u32,
        position: u32,
    }

    impl Default for RawProbeResult {
        fn default() -> Self {
            Self {
                elements: 0,
                iterations: 0,
                recommended_max_working_set_bytes: 0,
                buffer_bytes: 0,
                max_total_threads_per_threadgroup: 0,
                checksum: 0,
                has_unified_memory: 0,
                reserved: 0,
                setup_ms: 0.0,
                compile_ms: 0.0,
                warmup_wall_ms: 0.0,
                warmup_gpu_ms: 0.0,
                roundtrip_wall_ms: 0.0,
                roundtrip_gpu_ms: 0.0,
                batched_wall_ms: 0.0,
                batched_gpu_ms: 0.0,
                device_name: [0; 256],
            }
        }
    }

    extern "C" {
        fn rust_star_metal_create(
            context_out: *mut *mut c_void,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_probe(
            context: *mut c_void,
            elements: u64,
            iterations: u64,
            result: *mut RawProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_f16_get_rows(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            tensor_offset: u64,
            tensor_bytes: u64,
            n_vocab: u32,
            n_embd: u32,
            tokens: *const u32,
            token_count: u32,
            output: *mut f32,
            output_elements: u64,
            result: *mut RawEmbeddingProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_q8_0_projection(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            tensor_offset: u64,
            tensor_bytes: u64,
            input_elements: u32,
            output_elements: u32,
            input: *const f32,
            output: *mut f32,
            result: *mut RawProjectionProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_ratio128_compressor_replay(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            layer_index: u32,
            ape_offset: u64,
            ape_bytes: u64,
            kv_offset: u64,
            kv_bytes: u64,
            gate_offset: u64,
            gate_bytes: u64,
            norm_offset: u64,
            norm_bytes: u64,
            activation_sequence: *const f32,
            activation_elements: u64,
            output: *mut f32,
            output_elements: u64,
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_attention_ingress(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            token: u32,
            n_vocab: u32,
            embedding_offset: u64,
            embedding_bytes: u64,
            hc_fn_offset: u64,
            hc_fn_bytes: u64,
            hc_scale_offset: u64,
            hc_scale_bytes: u64,
            hc_base_offset: u64,
            hc_base_bytes: u64,
            attn_norm_offset: u64,
            attn_norm_bytes: u64,
            q_a_offset: u64,
            q_a_bytes: u64,
            q_a_norm_offset: u64,
            q_a_norm_bytes: u64,
            kv_offset: u64,
            kv_bytes: u64,
            kv_norm_offset: u64,
            kv_norm_bytes: u64,
            q_b_offset: u64,
            q_b_bytes: u64,
            attn_sinks_offset: u64,
            attn_sinks_bytes: u64,
            attn_output_a_offset: u64,
            attn_output_a_bytes: u64,
            attn_output_b_offset: u64,
            attn_output_b_bytes: u64,
            mixes: *mut f32,
            split: *mut f32,
            collapsed: *mut f32,
            attn_norm: *mut f32,
            q_lora: *mut f32,
            q_lora_norm: *mut f32,
            kv_raw: *mut f32,
            kv_norm: *mut f32,
            q_raw: *mut f32,
            q_cur: *mut f32,
            kv_rope: *mut f32,
            kv_cur: *mut f32,
            cache_rows: *mut f32,
            cache_row0: *const f32,
            attention_raw: *mut f32,
            attention_back: *mut f32,
            attention_low: *mut f32,
            attention_out: *mut f32,
            after_attention_hc: *mut f32,
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
            layer0: *const RawLayer0Extension,
        ) -> i32;
        fn rust_star_metal_run_output_head(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            hc_fn_offset: u64,
            hc_fn_bytes: u64,
            hc_scale_offset: u64,
            hc_scale_bytes: u64,
            hc_base_offset: u64,
            hc_base_bytes: u64,
            output_norm_offset: u64,
            output_norm_bytes: u64,
            output_offset: u64,
            output_bytes: u64,
            hc_pre: *mut f32,
            hc_weights: *mut f32,
            hc: *mut f32,
            norm: *mut f32,
            logits: *mut f32,
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_ffn_router(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            hc_fn_offset: u64,
            hc_fn_bytes: u64,
            hc_scale_offset: u64,
            hc_scale_bytes: u64,
            hc_base_offset: u64,
            hc_base_bytes: u64,
            ffn_norm_offset: u64,
            ffn_norm_bytes: u64,
            gate_offset: u64,
            gate_bytes: u64,
            bias_offset: u64,
            bias_bytes: u64,
            hash_offset: u64,
            hash_bytes: u64,
            after_attention_hc: *const f32,
            mixes: *mut f32,
            split: *mut f32,
            ffn_cur: *mut f32,
            ffn_norm: *mut f32,
            logits: *mut f32,
            probs: *mut f32,
            selected: *mut i32,
            weights: *mut f32,
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_moe_output(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            routed_gate_offset: u64,
            routed_gate_bytes: u64,
            routed_up_offset: u64,
            routed_up_bytes: u64,
            routed_down_offset: u64,
            routed_down_bytes: u64,
            shared_gate_offset: u64,
            shared_gate_bytes: u64,
            shared_up_offset: u64,
            shared_up_bytes: u64,
            shared_down_offset: u64,
            shared_down_bytes: u64,
            ffn_norm: *const f32,
            selected: *const i32,
            weights: *const f32,
            after_attention_hc: *const f32,
            split: *const f32,
            routed_mid: *mut f32,
            routed_out: *mut f32,
            shared_out: *mut f32,
            after_ffn_hc: *mut f32,
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_destroy(context: *mut c_void);
    }

    struct Context(*mut c_void);

    pub struct LayerExecutor<'a> {
        model: &'a MappedModel,
        context: Context,
        next_layer: u32,
        poisoned: bool,
    }

    struct Layer0Execution {
        report: Layer0ProbeReport,
        wall_ms_samples: Vec<f64>,
        gpu_ms_samples: Vec<f64>,
        repeat_bitwise_match: bool,
    }

    #[derive(Clone, Copy)]
    struct ModelSpan {
        absolute_offset: u64,
        bytes: u64,
    }

    impl From<&TensorInfo> for ModelSpan {
        fn from(tensor: &TensorInfo) -> Self {
            Self {
                absolute_offset: tensor.absolute_offset,
                bytes: tensor.bytes,
            }
        }
    }

    struct CompressorSpans {
        ape: ModelSpan,
        kv: ModelSpan,
        gate: ModelSpan,
        norm: ModelSpan,
    }

    struct PreparedLayerExecution {
        layer_index: u32,
        embedding: ModelSpan,
        hc_fn: ModelSpan,
        hc_scale: ModelSpan,
        hc_base: ModelSpan,
        norm_weight: ModelSpan,
        q_a: ModelSpan,
        q_a_norm: ModelSpan,
        kv: ModelSpan,
        kv_norm_weight: ModelSpan,
        q_b: ModelSpan,
        sinks: ModelSpan,
        output_a: ModelSpan,
        output_b: ModelSpan,
        ffn_hc_fn: ModelSpan,
        ffn_hc_scale: ModelSpan,
        ffn_hc_base: ModelSpan,
        ffn_norm_weight: ModelSpan,
        router_gate: ModelSpan,
        router_aux: ModelSpan,
        routed_gate: ModelSpan,
        routed_up: ModelSpan,
        routed_down: ModelSpan,
        shared_gate: ModelSpan,
        shared_up: ModelSpan,
        shared_down: ModelSpan,
        attention_compressor: Option<CompressorSpans>,
        indexer_compressor: Option<CompressorSpans>,
        compressor_prime: Vec<f32>,
        compressed_kv: Vec<f32>,
        compressed_indexer: Vec<f32>,
        expected: LayerExpected,
        expected_cache_rows: Vec<f32>,
        mixes: Vec<f32>,
        split: Vec<f32>,
        collapsed: Vec<f32>,
        norm: Vec<f32>,
        q_lora: Vec<f32>,
        q_lora_norm: Vec<f32>,
        kv_raw: Vec<f32>,
        kv_after_store: Vec<f32>,
        q_raw: Vec<f32>,
        q_cur: Vec<f32>,
        kv_rope: Vec<f32>,
        kv_cur: Vec<f32>,
        cache_rows: Vec<f32>,
        attention_raw: Vec<f32>,
        attention_back: Vec<f32>,
        attention_low: Vec<f32>,
        attention_out: Vec<f32>,
        after_attention_hc: Vec<f32>,
        ffn_mixes: Vec<f32>,
        ffn_split: Vec<f32>,
        ffn_norm: Vec<f32>,
        router_logits: Vec<f32>,
        router_probs: Vec<f32>,
        selected: Vec<i32>,
        router_weights: Vec<f32>,
        routed_mid: Vec<f32>,
        routed_out: Vec<f32>,
        shared_out: Vec<f32>,
        after_ffn_hc: Vec<f32>,
        wall_ms_samples: Vec<f64>,
        gpu_ms_samples: Vec<f64>,
        repeat_bitwise_matches: u32,
    }

    impl PreparedLayerExecution {
        fn new(
            model: &MappedModel,
            layer_index: u32,
            position: u32,
            measured_iterations: u32,
        ) -> Result<Self> {
            let tensor_name = |suffix: &str| format!("blk.{layer_index}.{suffix}");
            let span = |name: &str, kind: u32, dimensions: &[u64]| -> Result<ModelSpan> {
                Ok(exact_tensor(model, name, kind, dimensions)?.into())
            };
            let router_aux = if layer_index < 3 {
                span(&tensor_name("ffn_gate_tid2eid.weight"), 26, &[6, 129280])?
            } else {
                span(&tensor_name("exp_probs_b.bias"), 0, &[256])?
            };
            let expected = layer_expected(layer_index, position)?;
            let expected_cache_rows = expected.cache_row0.clone();
            let (attention_compressor, compressor_prime) = if layer_index >= 2 {
                let ratio = if layer_index % 2 == 0 { 4 } else { 128 };
                let width = if ratio == 4 { 1024 } else { 512 };
                let prime_bytes = compressor_prime_bytes(layer_index).ok_or_else(|| {
                    Error::invalid(format!(
                        "layer-{layer_index} compressor prime is not captured"
                    ))
                })?;
                (
                    Some(CompressorSpans {
                        ape: span(
                            &tensor_name("attn_compressor_ape.weight"),
                            1,
                            &[width, ratio],
                        )?,
                        kv: span(&tensor_name("attn_compressor_kv.weight"), 1, &[4096, width])?,
                        gate: span(
                            &tensor_name("attn_compressor_gate.weight"),
                            1,
                            &[4096, width],
                        )?,
                        norm: span(&tensor_name("attn_compressor_norm.weight"), 0, &[512])?,
                    }),
                    decode_f32_fixture(
                        prime_bytes,
                        &format!("layer-{layer_index} compressor prime"),
                    )?,
                )
            } else {
                (None, Vec::new())
            };
            let indexer_compressor = if layer_index >= 2 && layer_index % 2 == 0 {
                Some(CompressorSpans {
                    ape: span(&tensor_name("indexer_compressor_ape.weight"), 1, &[256, 4])?,
                    kv: span(
                        &tensor_name("indexer_compressor_kv.weight"),
                        1,
                        &[4096, 256],
                    )?,
                    gate: span(
                        &tensor_name("indexer_compressor_gate.weight"),
                        1,
                        &[4096, 256],
                    )?,
                    norm: span(&tensor_name("indexer_compressor_norm.weight"), 0, &[128])?,
                })
            } else {
                None
            };
            Ok(Self {
                layer_index,
                embedding: span("token_embd.weight", 1, &[4096, 129280])?,
                hc_fn: span(&tensor_name("hc_attn_fn.weight"), 1, &[16384, 24])?,
                hc_scale: span(&tensor_name("hc_attn_scale.weight"), 0, &[3])?,
                hc_base: span(&tensor_name("hc_attn_base.weight"), 0, &[24])?,
                norm_weight: span(&tensor_name("attn_norm.weight"), 0, &[4096])?,
                q_a: span(&tensor_name("attn_q_a.weight"), 8, &[4096, 1024])?,
                q_a_norm: span(&tensor_name("attn_q_a_norm.weight"), 0, &[1024])?,
                kv: span(&tensor_name("attn_kv.weight"), 8, &[4096, 512])?,
                kv_norm_weight: span(&tensor_name("attn_kv_a_norm.weight"), 0, &[512])?,
                q_b: span(&tensor_name("attn_q_b.weight"), 8, &[1024, 32768])?,
                sinks: span(&tensor_name("attn_sinks.weight"), 0, &[64])?,
                output_a: span(&tensor_name("attn_output_a.weight"), 8, &[4096, 8192])?,
                output_b: span(&tensor_name("attn_output_b.weight"), 8, &[8192, 4096])?,
                ffn_hc_fn: span(&tensor_name("hc_ffn_fn.weight"), 1, &[16384, 24])?,
                ffn_hc_scale: span(&tensor_name("hc_ffn_scale.weight"), 0, &[3])?,
                ffn_hc_base: span(&tensor_name("hc_ffn_base.weight"), 0, &[24])?,
                ffn_norm_weight: span(&tensor_name("ffn_norm.weight"), 0, &[4096])?,
                router_gate: span(&tensor_name("ffn_gate_inp.weight"), 1, &[4096, 256])?,
                router_aux,
                routed_gate: span(&tensor_name("ffn_gate_exps.weight"), 16, &[4096, 2048, 256])?,
                routed_up: span(&tensor_name("ffn_up_exps.weight"), 16, &[4096, 2048, 256])?,
                routed_down: span(&tensor_name("ffn_down_exps.weight"), 10, &[2048, 4096, 256])?,
                shared_gate: span(&tensor_name("ffn_gate_shexp.weight"), 8, &[4096, 2048])?,
                shared_up: span(&tensor_name("ffn_up_shexp.weight"), 8, &[4096, 2048])?,
                shared_down: span(&tensor_name("ffn_down_shexp.weight"), 8, &[2048, 4096])?,
                attention_compressor,
                indexer_compressor,
                compressor_prime,
                compressed_kv: vec![0.0; 512],
                compressed_indexer: vec![0.0; 128],
                expected,
                expected_cache_rows,
                mixes: vec![0.0; 24],
                split: vec![0.0; 24],
                collapsed: vec![0.0; 4096],
                norm: vec![0.0; 4096],
                q_lora: vec![0.0; 1024],
                q_lora_norm: vec![0.0; 1024],
                kv_raw: vec![0.0; 512],
                kv_after_store: vec![0.0; 512],
                q_raw: vec![0.0; 32768],
                q_cur: vec![0.0; 32768],
                kv_rope: vec![0.0; 512],
                kv_cur: vec![0.0; 512],
                cache_rows: vec![0.0; 4 * 512],
                attention_raw: vec![0.0; 32768],
                attention_back: vec![0.0; 32768],
                attention_low: vec![0.0; 8192],
                attention_out: vec![0.0; 4096],
                after_attention_hc: vec![0.0; 4 * 4096],
                ffn_mixes: vec![0.0; 24],
                ffn_split: vec![0.0; 24],
                ffn_norm: vec![0.0; 4096],
                router_logits: vec![0.0; 256],
                router_probs: vec![0.0; 256],
                selected: vec![0; 6],
                router_weights: vec![0.0; 6],
                routed_mid: vec![0.0; 6 * 2048],
                routed_out: vec![0.0; 4096],
                shared_out: vec![0.0; 4096],
                after_ffn_hc: vec![0.0; 4 * 4096],
                wall_ms_samples: vec![0.0; measured_iterations as usize],
                gpu_ms_samples: vec![0.0; measured_iterations as usize],
                repeat_bitwise_matches: 1,
            })
        }
    }

    impl Drop for Context {
        fn drop(&mut self) {
            unsafe { rust_star_metal_destroy(self.0) };
        }
    }

    impl Context {
        fn new() -> Result<Self> {
            let mut error = [0 as c_char; ERROR_BYTES];
            let mut pointer = ptr::null_mut();
            let created =
                unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
            if created == 0 || pointer.is_null() {
                return Err(Error::invalid(format!(
                    "Metal initialization failed: {}",
                    error_text(&error)
                )));
            }
            Ok(Self(pointer))
        }
    }

    impl<'a> LayerExecutor<'a> {
        pub fn new(model: &'a MappedModel) -> Result<Self> {
            Ok(Self {
                model,
                context: Context::new()?,
                next_layer: 0,
                poisoned: false,
            })
        }

        pub fn execute_layer(&mut self, layer_index: u32) -> Result<Layer0ProbeReport> {
            if self.poisoned {
                return Err(Error::invalid(
                    "persistent layer executor cannot continue after a failed execution",
                ));
            }
            if layer_index != self.next_layer {
                return Err(Error::invalid(format!(
                    "persistent layer executor expected layer {}, received layer {layer_index}",
                    self.next_layer
                )));
            }
            self.poisoned = true;
            let execution = run_layer_iterations(
                self.model,
                &self.context,
                layer_index,
                0,
                1,
                COMMAND_SYNCHRONIZED,
                0,
            )?;
            self.next_layer += 1;
            self.poisoned = false;
            Ok(execution.report)
        }
    }

    fn error_text(buffer: &[c_char; ERROR_BYTES]) -> String {
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn validate_times(raw: &RawProbeResult) -> Result<()> {
        for (name, value) in [
            ("setup_ms", raw.setup_ms),
            ("compile_ms", raw.compile_ms),
            ("warmup_wall_ms", raw.warmup_wall_ms),
            ("warmup_gpu_ms", raw.warmup_gpu_ms),
            ("roundtrip_wall_ms", raw.roundtrip_wall_ms),
            ("roundtrip_gpu_ms", raw.roundtrip_gpu_ms),
            ("batched_wall_ms", raw.batched_wall_ms),
            ("batched_gpu_ms", raw.batched_gpu_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal probe returned invalid {name}"
                )));
            }
        }
        if raw.roundtrip_wall_ms == 0.0 || raw.batched_wall_ms == 0.0 {
            return Err(Error::invalid("Metal probe returned a zero timed interval"));
        }
        Ok(())
    }

    pub fn run_probe(config: ProbeConfig) -> Result<ProbeReport> {
        let config = config.validate()?;
        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_probe(
                context.0,
                config.elements,
                config.iterations,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal dispatch probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.elements != config.elements || raw.iterations != config.iterations {
            return Err(Error::invalid("Metal probe returned different dimensions"));
        }
        validate_times(&raw)?;
        let device_name = unsafe { CStr::from_ptr(raw.device_name.as_ptr()) }
            .to_str()
            .map_err(|_| Error::invalid("Metal device name is not UTF-8"))?
            .to_owned();
        if device_name.is_empty() {
            return Err(Error::invalid("Metal device name is empty"));
        }
        Ok(ProbeReport {
            device_name,
            has_unified_memory: raw.has_unified_memory != 0,
            recommended_max_working_set_bytes: raw.recommended_max_working_set_bytes,
            max_total_threads_per_threadgroup: raw.max_total_threads_per_threadgroup,
            elements: raw.elements,
            iterations: raw.iterations,
            buffer_bytes: raw.buffer_bytes,
            checksum: raw.checksum,
            setup_ms: raw.setup_ms,
            compile_ms: raw.compile_ms,
            warmup_wall_ms: raw.warmup_wall_ms,
            warmup_gpu_ms: raw.warmup_gpu_ms,
            roundtrip_wall_ms: raw.roundtrip_wall_ms,
            roundtrip_gpu_ms: raw.roundtrip_gpu_ms,
            batched_wall_ms: raw.batched_wall_ms,
            batched_gpu_ms: raw.batched_gpu_ms,
        })
    }

    pub fn run_f16_embedding_probe(
        model: &MappedModel,
        tensor: &TensorInfo,
        tokens: &[u32],
    ) -> Result<EmbeddingProbeReport> {
        let (n_embd, n_vocab, output_elements) = validate_embedding_inputs(model, tensor, tokens)?;
        let expected = expected_f16_rows(model, tensor, tokens, n_embd)?;
        let mut actual = vec![0.0_f32; output_elements];
        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawEmbeddingProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_f16_get_rows(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                tensor.absolute_offset,
                tensor.bytes,
                n_vocab,
                n_embd,
                tokens.as_ptr(),
                tokens.len() as u32,
                actual.as_mut_ptr(),
                actual.len() as u64,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal F16 embedding probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.tensor_offset != tensor.absolute_offset
            || raw.tensor_bytes != tensor.bytes
            || raw.output_elements != actual.len() as u64
        {
            return Err(Error::invalid(
                "Metal F16 embedding probe returned different dimensions",
            ));
        }
        if raw.no_copy_pointer_match == 0 {
            return Err(Error::invalid(
                "Metal bytes-no-copy buffer did not retain the mmap pointer",
            ));
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "kernel_get_rows_f16 C0 mismatch at output {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal F16 embedding probe returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal F16 embedding probe returned a zero wall interval",
            ));
        }
        Ok(EmbeddingProbeReport {
            tensor_name: tensor.name.clone(),
            tokens: tokens.to_vec(),
            embedding_elements: u64::from(n_embd),
            output_elements: raw.output_elements,
            model_bytes: raw.model_bytes,
            tensor_offset: raw.tensor_offset,
            tensor_bytes: raw.tensor_bytes,
            page_offset: raw.page_offset,
            buffer_bytes: raw.buffer_bytes,
            inner_offset: raw.inner_offset,
            max_buffer_length: raw.max_buffer_length,
            no_copy_pointer_match: true,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            checksum: checksum_f32(&actual),
        })
    }

    pub fn run_q8_projection_probe(
        model: &MappedModel,
        tensor: &TensorInfo,
    ) -> Result<ProjectionProbeReport> {
        let (input, expected) = projection_fixture()?;
        let (input_elements, output_elements) =
            validate_projection_inputs(model, tensor, &input, &expected)?;
        let mut actual = vec![0.0_f32; output_elements as usize];
        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawProjectionProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_q8_0_projection(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                tensor.absolute_offset,
                tensor.bytes,
                input_elements,
                output_elements,
                input.as_ptr(),
                actual.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal Q8_0 projection probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.tensor_offset != tensor.absolute_offset
            || raw.tensor_bytes != tensor.bytes
            || raw.input_elements != u64::from(input_elements)
            || raw.output_elements != u64::from(output_elements)
        {
            return Err(Error::invalid(
                "Metal Q8_0 projection probe returned different dimensions",
            ));
        }
        if raw.no_copy_pointer_match == 0 {
            return Err(Error::invalid(
                "Metal Q8_0 bytes-no-copy buffer did not retain the mmap pointer",
            ));
        }
        if raw.simdgroups != 4 || raw.rows_per_threadgroup != 2 {
            return Err(Error::invalid(
                "Metal Q8_0 projection used unexpected dispatch geometry",
            ));
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "kernel_mul_mv_q8_0_f32 C0 mismatch at output {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal Q8_0 projection probe returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal Q8_0 projection probe returned a zero wall interval",
            ));
        }
        Ok(ProjectionProbeReport {
            fixture_id: PROJECTION_FIXTURE_ID,
            tensor_name: tensor.name.clone(),
            input_elements: raw.input_elements,
            output_elements: raw.output_elements,
            model_bytes: raw.model_bytes,
            tensor_offset: raw.tensor_offset,
            tensor_bytes: raw.tensor_bytes,
            page_offset: raw.page_offset,
            buffer_bytes: raw.buffer_bytes,
            inner_offset: raw.inner_offset,
            max_buffer_length: raw.max_buffer_length,
            no_copy_pointer_match: true,
            simdgroups: raw.simdgroups,
            rows_per_threadgroup: raw.rows_per_threadgroup,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            input_checksum: checksum_f32(&input),
            output_checksum: checksum_f32(&actual),
        })
    }

    pub fn run_attention_ingress_probe(model: &MappedModel) -> Result<IngressProbeReport> {
        const TOKEN: u32 = 201;
        let embedding = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let hc_fn = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_attn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_attn_base.weight", 0, &[24])?;
        let norm_weight = exact_tensor(model, "blk.0.attn_norm.weight", 0, &[4096])?;
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let (expected_mixes, expected_split, expected_collapsed, expected_norm, expected_q) =
            ingress_fixture()?;
        let mut mixes = vec![0.0_f32; 24];
        let mut split = vec![0.0_f32; 24];
        let mut collapsed = vec![0.0_f32; 4096];
        let mut norm = vec![0.0_f32; 4096];
        let mut q = vec![0.0_f32; 1024];

        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_attention_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                TOKEN,
                129280,
                embedding.absolute_offset,
                embedding.bytes,
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                q_a.absolute_offset,
                q_a.bytes,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                collapsed.as_mut_ptr(),
                norm.as_mut_ptr(),
                q.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
                ptr::null(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 attention ingress probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 6
            || raw.pointer_matches != 6
        {
            return Err(Error::invalid(
                "Metal attention ingress did not preserve all six mmap-backed model ranges",
            ));
        }
        for (label, actual, expected) in [
            (
                "hc_attn_pre_mixes",
                mixes.as_slice(),
                expected_mixes.as_slice(),
            ),
            ("hc_split", split.as_slice(), expected_split.as_slice()),
            (
                "hc_attn_pre",
                collapsed.as_slice(),
                expected_collapsed.as_slice(),
            ),
            ("attn_norm", norm.as_slice(), expected_norm.as_slice()),
            ("q_lora", q.as_slice(), expected_q.as_slice()),
        ] {
            if actual.len() != expected.len() {
                return Err(Error::invalid(format!("{label} fixture length mismatch")));
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "attention ingress C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal attention ingress returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal attention ingress returned a zero wall interval",
            ));
        }
        Ok(IngressProbeReport {
            fixture_id: INGRESS_FIXTURE_ID,
            token: TOKEN,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            mixes_checksum: checksum_f32(&mixes),
            split_checksum: checksum_f32(&split),
            collapsed_checksum: checksum_f32(&collapsed),
            attn_norm_checksum: checksum_f32(&norm),
            q_lora_checksum: checksum_f32(&q),
        })
    }

    pub fn run_attention_setup_probe(model: &MappedModel) -> Result<AttentionSetupProbeReport> {
        const TOKEN: u32 = 201;
        let embedding = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let hc_fn = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_attn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_attn_base.weight", 0, &[24])?;
        let norm_weight = exact_tensor(model, "blk.0.attn_norm.weight", 0, &[4096])?;
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = exact_tensor(model, "blk.0.attn_q_a_norm.weight", 0, &[1024])?;
        let kv = exact_tensor(model, "blk.0.attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm_weight = exact_tensor(model, "blk.0.attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        let (expected_q_norm, expected_kv_raw, expected_kv_norm, expected_q_raw) =
            attention_setup_fixture()?;

        let mut mixes = vec![0.0_f32; 24];
        let mut split = vec![0.0_f32; 24];
        let mut collapsed = vec![0.0_f32; 4096];
        let mut norm = vec![0.0_f32; 4096];
        let mut q_lora = vec![0.0_f32; 1024];
        let mut q_lora_norm = vec![0.0_f32; 1024];
        let mut kv_raw = vec![0.0_f32; 512];
        let mut kv_norm = vec![0.0_f32; 512];
        let mut q_raw = vec![0.0_f32; 32768];

        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_attention_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                TOKEN,
                129280,
                embedding.absolute_offset,
                embedding.bytes,
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                q_a.absolute_offset,
                q_a.bytes,
                q_a_norm.absolute_offset,
                q_a_norm.bytes,
                kv.absolute_offset,
                kv.bytes,
                kv_norm_weight.absolute_offset,
                kv_norm_weight.bytes,
                q_b.absolute_offset,
                q_b.bytes,
                0,
                0,
                0,
                0,
                0,
                0,
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                collapsed.as_mut_ptr(),
                norm.as_mut_ptr(),
                q_lora.as_mut_ptr(),
                q_lora_norm.as_mut_ptr(),
                kv_raw.as_mut_ptr(),
                kv_norm.as_mut_ptr(),
                q_raw.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
                ptr::null(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 attention setup probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 10
            || raw.pointer_matches != 10
        {
            return Err(Error::invalid(
                "Metal attention setup did not preserve all ten mmap-backed model ranges",
            ));
        }
        for (label, actual, expected) in [
            (
                "q_lora_norm",
                q_lora_norm.as_slice(),
                expected_q_norm.as_slice(),
            ),
            ("KVraw", kv_raw.as_slice(), expected_kv_raw.as_slice()),
            ("KVnorm", kv_norm.as_slice(), expected_kv_norm.as_slice()),
            ("Qraw", q_raw.as_slice(), expected_q_raw.as_slice()),
        ] {
            if actual.len() != expected.len() {
                return Err(Error::invalid(format!("{label} fixture length mismatch")));
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "attention setup C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal attention setup returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal attention setup returned a zero wall interval",
            ));
        }
        Ok(AttentionSetupProbeReport {
            fixture_id: ATTENTION_SETUP_FIXTURE_ID,
            token: TOKEN,
            dispatches: 9,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            q_lora_norm_checksum: checksum_f32(&q_lora_norm),
            kv_raw_checksum: checksum_f32(&kv_raw),
            kv_norm_checksum: checksum_f32(&kv_norm),
            q_raw_checksum: checksum_f32(&q_raw),
        })
    }

    pub fn run_rope_kv_store_probe(model: &MappedModel) -> Result<RopeKvStoreProbeReport> {
        const TOKEN: u32 = 201;
        const CACHE_ROWS: usize = 3;
        const CACHE_ROW: usize = 1;
        const CACHE_GUARD: f32 = -12345.5;
        let embedding = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let hc_fn = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_attn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_attn_base.weight", 0, &[24])?;
        let norm_weight = exact_tensor(model, "blk.0.attn_norm.weight", 0, &[4096])?;
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = exact_tensor(model, "blk.0.attn_q_a_norm.weight", 0, &[1024])?;
        let kv = exact_tensor(model, "blk.0.attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm_weight = exact_tensor(model, "blk.0.attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        let (expected_q_norm, expected_kv_raw, _, expected_q_raw) = attention_setup_fixture()?;
        let (expected_q_cur, expected_kv_rope, expected_kv_cur, expected_cache_row) =
            rope_kv_store_fixture()?;

        let mut mixes = vec![0.0_f32; 24];
        let mut split = vec![0.0_f32; 24];
        let mut collapsed = vec![0.0_f32; 4096];
        let mut norm = vec![0.0_f32; 4096];
        let mut q_lora = vec![0.0_f32; 1024];
        let mut q_lora_norm = vec![0.0_f32; 1024];
        let mut kv_raw = vec![0.0_f32; 512];
        let mut kv_after_store = vec![0.0_f32; 512];
        let mut q_raw = vec![0.0_f32; 32768];
        let mut q_cur = vec![0.0_f32; 32768];
        let mut kv_rope = vec![0.0_f32; 512];
        let mut kv_cur = vec![0.0_f32; 512];
        let mut cache_rows = vec![0.0_f32; CACHE_ROWS * 512];

        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_attention_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                TOKEN,
                129280,
                embedding.absolute_offset,
                embedding.bytes,
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                q_a.absolute_offset,
                q_a.bytes,
                q_a_norm.absolute_offset,
                q_a_norm.bytes,
                kv.absolute_offset,
                kv.bytes,
                kv_norm_weight.absolute_offset,
                kv_norm_weight.bytes,
                q_b.absolute_offset,
                q_b.bytes,
                0,
                0,
                0,
                0,
                0,
                0,
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                collapsed.as_mut_ptr(),
                norm.as_mut_ptr(),
                q_lora.as_mut_ptr(),
                q_lora_norm.as_mut_ptr(),
                kv_raw.as_mut_ptr(),
                kv_after_store.as_mut_ptr(),
                q_raw.as_mut_ptr(),
                q_cur.as_mut_ptr(),
                kv_rope.as_mut_ptr(),
                kv_cur.as_mut_ptr(),
                cache_rows.as_mut_ptr(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
                ptr::null(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 RoPE/KV-store probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 10
            || raw.pointer_matches != 10
        {
            return Err(Error::invalid(
                "Metal RoPE/KV-store path did not preserve all ten mmap-backed model ranges",
            ));
        }
        for (label, actual, expected) in [
            (
                "q_lora_norm",
                q_lora_norm.as_slice(),
                expected_q_norm.as_slice(),
            ),
            ("KVraw", kv_raw.as_slice(), expected_kv_raw.as_slice()),
            ("Qraw", q_raw.as_slice(), expected_q_raw.as_slice()),
            ("Qcur", q_cur.as_slice(), expected_q_cur.as_slice()),
            ("KVrope", kv_rope.as_slice(), expected_kv_rope.as_slice()),
            ("KVcur", kv_cur.as_slice(), expected_kv_cur.as_slice()),
            (
                "cache_row",
                &cache_rows[CACHE_ROW * 512..(CACHE_ROW + 1) * 512],
                expected_cache_row.as_slice(),
            ),
        ] {
            if actual.len() != expected.len() {
                return Err(Error::invalid(format!("{label} fixture length mismatch")));
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "RoPE/KV-store C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        if kv_after_store
            .iter()
            .zip(&kv_cur)
            .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            return Err(Error::invalid(
                "KV post-store output aliases do not match by bit pattern",
            ));
        }
        let guard_bits = CACHE_GUARD.to_bits();
        let guards_intact = cache_rows[..512]
            .iter()
            .chain(&cache_rows[2 * 512..])
            .all(|value| value.to_bits() == guard_bits);
        if !guards_intact {
            return Err(Error::invalid(
                "KV cache store modified a neighboring guard row",
            ));
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal RoPE/KV-store returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal RoPE/KV-store returned a zero wall interval",
            ));
        }
        Ok(RopeKvStoreProbeReport {
            fixture_id: ROPE_KV_STORE_FIXTURE_ID,
            token: TOKEN,
            dispatches: 12,
            cache_capacity_rows: CACHE_ROWS as u32,
            cache_target_row: CACHE_ROW as u32,
            cache_guard_rows_intact: guards_intact,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            q_cur_checksum: checksum_f32(&q_cur),
            kv_rope_checksum: checksum_f32(&kv_rope),
            kv_cur_checksum: checksum_f32(&kv_cur),
            cache_row_checksum: checksum_f32(&cache_rows[CACHE_ROW * 512..(CACHE_ROW + 1) * 512]),
        })
    }

    pub fn run_attention_read_probe(model: &MappedModel) -> Result<AttentionReadProbeReport> {
        const TOKEN: u32 = 201;
        const CACHE_ROWS: usize = 3;
        const CACHE_GUARD: f32 = -12345.5;
        let embedding = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let hc_fn = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_attn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_attn_base.weight", 0, &[24])?;
        let norm_weight = exact_tensor(model, "blk.0.attn_norm.weight", 0, &[4096])?;
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = exact_tensor(model, "blk.0.attn_q_a_norm.weight", 0, &[1024])?;
        let kv = exact_tensor(model, "blk.0.attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm_weight = exact_tensor(model, "blk.0.attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        let sinks = exact_tensor(model, "blk.0.attn_sinks.weight", 0, &[64])?;
        let (expected_cache_row0, expected_cache_row1, expected_attention_back) =
            attention_read_fixture()?;
        let expected_q_cur = decode_f32_fixture(ATTENTION_Q_CUR_BYTES, "attention Q current")?;

        let mut mixes = vec![0.0_f32; 24];
        let mut split = vec![0.0_f32; 24];
        let mut collapsed = vec![0.0_f32; 4096];
        let mut norm = vec![0.0_f32; 4096];
        let mut q_lora = vec![0.0_f32; 1024];
        let mut q_lora_norm = vec![0.0_f32; 1024];
        let mut kv_raw = vec![0.0_f32; 512];
        let mut kv_after_store = vec![0.0_f32; 512];
        let mut q_raw = vec![0.0_f32; 32768];
        let mut q_cur = vec![0.0_f32; 32768];
        let mut kv_rope = vec![0.0_f32; 512];
        let mut kv_cur = vec![0.0_f32; 512];
        let mut cache_rows = vec![0.0_f32; CACHE_ROWS * 512];
        let mut attention_raw = vec![0.0_f32; 32768];
        let mut attention_back = vec![0.0_f32; 32768];

        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_attention_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                TOKEN,
                129280,
                embedding.absolute_offset,
                embedding.bytes,
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                q_a.absolute_offset,
                q_a.bytes,
                q_a_norm.absolute_offset,
                q_a_norm.bytes,
                kv.absolute_offset,
                kv.bytes,
                kv_norm_weight.absolute_offset,
                kv_norm_weight.bytes,
                q_b.absolute_offset,
                q_b.bytes,
                sinks.absolute_offset,
                sinks.bytes,
                0,
                0,
                0,
                0,
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                collapsed.as_mut_ptr(),
                norm.as_mut_ptr(),
                q_lora.as_mut_ptr(),
                q_lora_norm.as_mut_ptr(),
                kv_raw.as_mut_ptr(),
                kv_after_store.as_mut_ptr(),
                q_raw.as_mut_ptr(),
                q_cur.as_mut_ptr(),
                kv_rope.as_mut_ptr(),
                kv_cur.as_mut_ptr(),
                cache_rows.as_mut_ptr(),
                expected_cache_row0.as_ptr(),
                attention_raw.as_mut_ptr(),
                attention_back.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
                ptr::null(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 attention-read probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 11
            || raw.pointer_matches != 11
        {
            return Err(Error::invalid(
                "Metal attention-read path did not preserve all eleven mmap-backed model ranges",
            ));
        }
        for (label, actual, expected) in [
            ("Qcur", q_cur.as_slice(), expected_q_cur.as_slice()),
            (
                "cache_row0",
                &cache_rows[..512],
                expected_cache_row0.as_slice(),
            ),
            (
                "cache_row1",
                &cache_rows[512..1024],
                expected_cache_row1.as_slice(),
            ),
            (
                "kqv_back",
                attention_back.as_slice(),
                expected_attention_back.as_slice(),
            ),
        ] {
            if actual.len() != expected.len() {
                return Err(Error::invalid(format!("{label} fixture length mismatch")));
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "attention-read C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        let guard_bits = CACHE_GUARD.to_bits();
        let guard_intact = cache_rows[1024..]
            .iter()
            .all(|value| value.to_bits() == guard_bits);
        if !guard_intact {
            return Err(Error::invalid(
                "attention read modified the neighboring cache guard row",
            ));
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal attention-read returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal attention-read returned a zero wall interval",
            ));
        }
        Ok(AttentionReadProbeReport {
            fixture_id: ATTENTION_READ_FIXTURE_ID,
            token: TOKEN,
            dispatches: 17,
            cache_capacity_rows: CACHE_ROWS as u32,
            cache_rows_read: 2,
            cache_row0_preserved: true,
            cache_guard_row_intact: guard_intact,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            attention_raw_checksum: checksum_f32(&attention_raw),
            attention_back_checksum: checksum_f32(&attention_back),
        })
    }

    pub fn run_attention_output_probe(model: &MappedModel) -> Result<AttentionOutputProbeReport> {
        const TOKEN: u32 = 201;
        let embedding = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let hc_fn = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_attn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_attn_base.weight", 0, &[24])?;
        let norm_weight = exact_tensor(model, "blk.0.attn_norm.weight", 0, &[4096])?;
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = exact_tensor(model, "blk.0.attn_q_a_norm.weight", 0, &[1024])?;
        let kv = exact_tensor(model, "blk.0.attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm_weight = exact_tensor(model, "blk.0.attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        let sinks = exact_tensor(model, "blk.0.attn_sinks.weight", 0, &[64])?;
        let output_a = exact_tensor(model, "blk.0.attn_output_a.weight", 8, &[4096, 8192])?;
        let output_b = exact_tensor(model, "blk.0.attn_output_b.weight", 8, &[8192, 4096])?;
        let (expected_cache_row0, _, _) = attention_read_fixture()?;
        let (expected_back, expected_low, expected_out, expected_hc_post) =
            attention_output_fixture()?;

        let mut mixes = vec![0.0_f32; 24];
        let mut split = vec![0.0_f32; 24];
        let mut collapsed = vec![0.0_f32; 4096];
        let mut norm = vec![0.0_f32; 4096];
        let mut q_lora = vec![0.0_f32; 1024];
        let mut q_lora_norm = vec![0.0_f32; 1024];
        let mut kv_raw = vec![0.0_f32; 512];
        let mut kv_after_store = vec![0.0_f32; 512];
        let mut q_raw = vec![0.0_f32; 32768];
        let mut q_cur = vec![0.0_f32; 32768];
        let mut kv_rope = vec![0.0_f32; 512];
        let mut kv_cur = vec![0.0_f32; 512];
        let mut cache_rows = vec![0.0_f32; 3 * 512];
        let mut attention_raw = vec![0.0_f32; 32768];
        let mut attention_back = vec![0.0_f32; 32768];
        let mut attention_low = vec![0.0_f32; 8192];
        let mut attention_out = vec![0.0_f32; 4096];
        let mut after_attention_hc = vec![0.0_f32; 4 * 4096];

        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_attention_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                TOKEN,
                129280,
                embedding.absolute_offset,
                embedding.bytes,
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                q_a.absolute_offset,
                q_a.bytes,
                q_a_norm.absolute_offset,
                q_a_norm.bytes,
                kv.absolute_offset,
                kv.bytes,
                kv_norm_weight.absolute_offset,
                kv_norm_weight.bytes,
                q_b.absolute_offset,
                q_b.bytes,
                sinks.absolute_offset,
                sinks.bytes,
                output_a.absolute_offset,
                output_a.bytes,
                output_b.absolute_offset,
                output_b.bytes,
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                collapsed.as_mut_ptr(),
                norm.as_mut_ptr(),
                q_lora.as_mut_ptr(),
                q_lora_norm.as_mut_ptr(),
                kv_raw.as_mut_ptr(),
                kv_after_store.as_mut_ptr(),
                q_raw.as_mut_ptr(),
                q_cur.as_mut_ptr(),
                kv_rope.as_mut_ptr(),
                kv_cur.as_mut_ptr(),
                cache_rows.as_mut_ptr(),
                expected_cache_row0.as_ptr(),
                attention_raw.as_mut_ptr(),
                attention_back.as_mut_ptr(),
                attention_low.as_mut_ptr(),
                attention_out.as_mut_ptr(),
                after_attention_hc.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
                ptr::null(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 attention-output probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 13
            || raw.pointer_matches != 13
        {
            return Err(Error::invalid(
                "Metal attention-output path did not preserve all thirteen mmap-backed model ranges",
            ));
        }
        for (label, actual, expected) in [
            (
                "kqv_back",
                attention_back.as_slice(),
                expected_back.as_slice(),
            ),
            (
                "attn_low",
                attention_low.as_slice(),
                expected_low.as_slice(),
            ),
            (
                "attn_out",
                attention_out.as_slice(),
                expected_out.as_slice(),
            ),
            (
                "hc_attn_post",
                after_attention_hc.as_slice(),
                expected_hc_post.as_slice(),
            ),
        ] {
            if actual.len() != expected.len() {
                return Err(Error::invalid(format!("{label} fixture length mismatch")));
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "attention-output C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal attention-output returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal attention-output returned a zero wall interval",
            ));
        }
        Ok(AttentionOutputProbeReport {
            fixture_id: ATTENTION_OUTPUT_FIXTURE_ID,
            token: TOKEN,
            dispatches: 19,
            output_groups: 8,
            output_rank: 1024,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            attention_low_checksum: checksum_f32(&attention_low),
            attention_out_checksum: checksum_f32(&attention_out),
            hc_post_checksum: checksum_f32(&after_attention_hc),
        })
    }

    pub fn run_layer0_probe(model: &MappedModel) -> Result<Layer0ProbeReport> {
        LayerExecutor::new(model)?.execute_layer(0)
    }

    pub fn run_layers01_probe(model: &MappedModel) -> Result<Layers01ProbeReport> {
        let mut executor = LayerExecutor::new(model)?;
        let layer0 = executor.execute_layer(0)?;
        let layer1 = executor.execute_layer(1)?;
        Ok(LayerSequenceProbeReport {
            layers: vec![layer0, layer1],
            command_buffers: 2,
            retained_hc_handoff: true,
            kv_cache_layers: 2,
        })
    }

    pub fn run_layers012_probe(model: &MappedModel) -> Result<Layers012ProbeReport> {
        let mut executor = LayerExecutor::new(model)?;
        let layer0 = executor.execute_layer(0)?;
        let layer1 = executor.execute_layer(1)?;
        let layer2 = executor.execute_layer(2)?;
        Ok(LayerSequenceProbeReport {
            layers: vec![layer0, layer1, layer2],
            command_buffers: 3,
            retained_hc_handoff: true,
            kv_cache_layers: 3,
        })
    }

    pub fn run_layers0123_probe(model: &MappedModel) -> Result<Layers0123ProbeReport> {
        let mut executor = LayerExecutor::new(model)?;
        let mut layers = Vec::with_capacity(4);
        for layer_index in 0..=3 {
            layers.push(executor.execute_layer(layer_index)?);
        }
        Ok(LayerSequenceProbeReport {
            layers,
            command_buffers: 4,
            retained_hc_handoff: true,
            kv_cache_layers: 4,
        })
    }

    pub fn run_layers012_chained_probe(model: &MappedModel) -> Result<Layers012ChainedProbeReport> {
        let context = Context::new()?;
        run_layer_iterations(model, &context, 0, 0, 1, COMMAND_CHAINED_ENQUEUE, 2)?;
        run_layer_iterations(model, &context, 1, 0, 1, COMMAND_CHAINED_ENQUEUE, 2)?;
        run_layer_iterations(model, &context, 2, 0, 1, COMMAND_CHAINED_FINAL, 2)?;

        let layer0 =
            run_layer_iterations(model, &context, 0, 0, 1, COMMAND_CHAINED_COLLECT, 2)?.report;
        let layer1 =
            run_layer_iterations(model, &context, 1, 0, 1, COMMAND_CHAINED_COLLECT, 2)?.report;
        let layer2 =
            run_layer_iterations(model, &context, 2, 0, 1, COMMAND_CHAINED_COLLECT, 2)?.report;
        let wall_ms = layer0.wall_ms;
        let gpu_ms = layer0.gpu_ms + layer1.gpu_ms + layer2.gpu_ms;
        Ok(Layers012ChainedProbeReport {
            layers: vec![layer0, layer1, layer2],
            command_buffers: 3,
            host_waits: 1,
            retained_hc_handoff: true,
            kv_cache_layers: 3,
            wall_ms,
            gpu_ms,
        })
    }

    pub fn run_layers0123_chained_probe(
        model: &MappedModel,
    ) -> Result<Layers0123ChainedProbeReport> {
        let context = Context::new()?;
        for layer_index in 0..3 {
            run_layer_iterations(
                model,
                &context,
                layer_index,
                0,
                1,
                COMMAND_CHAINED_ENQUEUE,
                3,
            )?;
        }
        run_layer_iterations(model, &context, 3, 0, 1, COMMAND_CHAINED_FINAL, 3)?;

        let mut layers = Vec::with_capacity(4);
        for layer_index in 0..=3 {
            layers.push(
                run_layer_iterations(
                    model,
                    &context,
                    layer_index,
                    0,
                    1,
                    COMMAND_CHAINED_COLLECT,
                    3,
                )?
                .report,
            );
        }
        let wall_ms = layers[0].wall_ms;
        let gpu_ms = layers.iter().map(|layer| layer.gpu_ms).sum();
        Ok(Layers012ChainedProbeReport {
            layers,
            command_buffers: 4,
            host_waits: 1,
            retained_hc_handoff: true,
            kv_cache_layers: 4,
            wall_ms,
            gpu_ms,
        })
    }

    fn submit_prepared_layers(
        model: &MappedModel,
        context: &Context,
        layers: &mut [PreparedLayerExecution],
        token: u32,
        position: u32,
    ) -> Result<()> {
        if !(3..=43).contains(&layers.len()) {
            return Err(Error::invalid(
                "prepared submission requires three through forty-three contiguous layers",
            ));
        }
        let final_layer = layers.len() as u32 - 1;
        for (layer_index, layer) in layers.iter_mut().enumerate() {
            if layer.layer_index != layer_index as u32 {
                return Err(Error::invalid(
                    "prepared submission layers must be contiguous and zero-based",
                ));
            }
            let command_mode = if layer_index as u32 == final_layer {
                COMMAND_CHAINED_FINAL
            } else {
                COMMAND_CHAINED_ENQUEUE
            };
            run_prepared_layer_iterations(
                model,
                context,
                layer,
                token,
                position,
                0,
                1,
                command_mode,
                final_layer,
            )?;
        }
        Ok(())
    }

    pub fn run_layers0123_bench(
        model: &MappedModel,
        config: Layers0123BenchConfig,
    ) -> Result<Layers0123BenchReport> {
        let config = config.validate()?;
        let context = Context::new()?;
        let mut layers = (0..=3)
            .map(|layer_index| PreparedLayerExecution::new(model, layer_index, 1, 1))
            .collect::<Result<Vec<_>>>()?;

        for _ in 0..config.warmup_iterations {
            submit_prepared_layers(model, &context, &mut layers, 201, 1)?;
        }

        let mut wall_ms_samples = Vec::with_capacity(config.iterations as usize);
        let mut gpu_ms_samples = Vec::with_capacity(config.iterations as usize);
        for _ in 0..config.iterations {
            submit_prepared_layers(model, &context, &mut layers, 201, 1)?;
            let timing = run_prepared_layer_iterations(
                model,
                &context,
                &mut layers[0],
                201,
                1,
                0,
                1,
                COMMAND_CHAINED_TIMING,
                3,
            )?;
            wall_ms_samples.push(timing.report.wall_ms);
            gpu_ms_samples.push(timing.report.gpu_ms);
        }

        let mut final_layers = Vec::with_capacity(4);
        for layer in &mut layers {
            final_layers.push(
                run_prepared_layer_iterations(
                    model,
                    &context,
                    layer,
                    201,
                    1,
                    0,
                    1,
                    COMMAND_CHAINED_COLLECT,
                    3,
                )?
                .report,
            );
        }

        Ok(Layers0123BenchReport {
            warmup_iterations: config.warmup_iterations,
            iterations: config.iterations,
            command_buffers_per_iteration: 4,
            host_waits_per_iteration: 1,
            retained_hc_handoff: true,
            kv_cache_layers: 4,
            wall: summarize_timing(&wall_ms_samples)?,
            gpu: summarize_timing(&gpu_ms_samples)?,
            wall_ms_samples,
            gpu_ms_samples,
            final_layers,
        })
    }

    fn run_retained_output_head(
        model: &MappedModel,
        context: &Context,
        position: u32,
    ) -> Result<OutputHeadProbeReport> {
        let expected = output_head_expected(position)?;
        let hc_fn = exact_tensor(model, "output_hc_fn.weight", 1, &[16384, 4])?;
        let hc_scale = exact_tensor(model, "output_hc_scale.weight", 0, &[1])?;
        let hc_base = exact_tensor(model, "output_hc_base.weight", 0, &[4])?;
        let output_norm = exact_tensor(model, "output_norm.weight", 0, &[4096])?;
        let output = exact_tensor(model, "output.weight", 8, &[4096, 129280])?;
        let mut hc_pre = vec![0.0_f32; 4];
        let mut hc_weights = vec![0.0_f32; 4];
        let mut hc = vec![0.0_f32; 4096];
        let mut norm = vec![0.0_f32; 4096];
        let mut logits = vec![0.0_f32; 129280];
        let mut raw = RawIngressProbeResult::default();
        let mut error = [0 as c_char; ERROR_BYTES];
        let succeeded = unsafe {
            rust_star_metal_run_output_head(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                output_norm.absolute_offset,
                output_norm.bytes,
                output.absolute_offset,
                output.bytes,
                hc_pre.as_mut_ptr(),
                hc_weights.as_mut_ptr(),
                hc.as_mut_ptr(),
                norm.as_mut_ptr(),
                logits.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal output-head probe failed at position {position}: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 5
            || raw.pointer_matches != 5
        {
            return Err(Error::invalid(
                "Metal output head did not preserve all five mmap-backed model ranges",
            ));
        }
        for (label, actual, reference) in [
            (
                "output_hc_pre",
                hc_pre.as_slice(),
                expected.hc_pre.as_slice(),
            ),
            (
                "output_hc_weights",
                hc_weights.as_slice(),
                expected.hc_weights.as_slice(),
            ),
            ("output_hc", hc.as_slice(), expected.hc.as_slice()),
            ("output_norm", norm.as_slice(), expected.norm.as_slice()),
            ("logits", logits.as_slice(), expected.logits.as_slice()),
        ] {
            for (index, (actual, reference)) in actual.iter().zip(reference).enumerate() {
                if actual.to_bits() != reference.to_bits() {
                    return Err(Error::invalid(format!(
                        "output-head C0 mismatch in {label}[{index}] at position {position}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        reference.to_bits()
                    )));
                }
            }
        }
        let selected_token = lowest_id_argmax(&logits)?;
        if selected_token != expected.selected_token {
            return Err(Error::invalid(format!(
                "output-head selection mismatch at position {position}: actual={selected_token} expected={}",
                expected.selected_token
            )));
        }
        if !raw.wall_ms.is_finite()
            || raw.wall_ms <= 0.0
            || !raw.gpu_ms.is_finite()
            || raw.gpu_ms < 0.0
        {
            return Err(Error::invalid("Metal output head returned invalid timing"));
        }
        Ok(OutputHeadProbeReport {
            fixture_id: expected.fixture_id,
            dispatches: 5,
            command_buffers: 1,
            host_waits: 1,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            hc_pre_checksum: checksum_f32(&hc_pre),
            hc_weights_checksum: checksum_f32(&hc_weights),
            hc_checksum: checksum_f32(&hc),
            norm_checksum: checksum_f32(&norm),
            logits_checksum: checksum_f32(&logits),
            selected_token,
        })
    }

    fn run_position_advancing_probe(
        model: &MappedModel,
        layer_count: u32,
    ) -> Result<Layers0123DecodeProbeReport> {
        if layer_count != 4 && layer_count != 6 && layer_count != 8 && layer_count != 43 {
            return Err(Error::invalid(
                "position-advancing probe requires four, six, eight, or forty-three layers",
            ));
        }
        let context = Context::new()?;
        let final_layer = layer_count - 1;
        let mut layers = (0..layer_count)
            .map(|layer_index| PreparedLayerExecution::new(model, layer_index, 1, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut steps = Vec::with_capacity(3);

        for (position, token) in [(1_u32, 201_u32), (2_u32, 361_u32), (3_u32, 1915_u32)] {
            if position > 1 {
                for (layer_index, layer) in layers.iter_mut().enumerate() {
                    layer.expected = layer_expected(layer_index as u32, position)?;
                }
            }
            submit_prepared_layers(model, &context, &mut layers, token, position)?;
            let mut reports = Vec::with_capacity(layer_count as usize);
            for layer in &mut layers {
                reports.push(
                    run_prepared_layer_iterations(
                        model,
                        &context,
                        layer,
                        token,
                        position,
                        0,
                        1,
                        COMMAND_CHAINED_COLLECT,
                        final_layer,
                    )?
                    .report,
                );
            }
            let wall_ms = reports[0].wall_ms;
            let gpu_ms = reports.iter().map(|layer| layer.gpu_ms).sum();
            let output_hc_checksum = reports[final_layer as usize].final_hc_checksum;
            steps.push(Layers0123DecodeStepReport {
                position,
                token,
                cache_rows: position + 1,
                layers: reports,
                wall_ms,
                gpu_ms,
                output_hc_checksum,
            });
        }

        Ok(Layers0123DecodeProbeReport {
            steps,
            command_buffers_per_step: layer_count,
            host_waits_per_step: 1,
            kv_cache_layers: layer_count,
            cache_capacity_rows: 4,
            output_hc_elements: 4 * 4096,
        })
    }

    pub fn run_layers0123_decode_probe(model: &MappedModel) -> Result<Layers0123DecodeProbeReport> {
        run_position_advancing_probe(model, 4)
    }

    pub fn run_layers012345_decode_probe(
        model: &MappedModel,
    ) -> Result<Layers012345DecodeProbeReport> {
        run_position_advancing_probe(model, 6)
    }

    pub fn run_layers01234567_decode_probe(
        model: &MappedModel,
    ) -> Result<Layers01234567DecodeProbeReport> {
        run_position_advancing_probe(model, 8)
    }

    pub fn run_layers0_to_42_decode_probe(
        model: &MappedModel,
    ) -> Result<Layers0To42DecodeProbeReport> {
        run_position_advancing_probe(model, 43)
    }

    pub fn run_decoder_output_probe(model: &MappedModel) -> Result<DecoderOutputProbeReport> {
        let context = Context::new()?;
        let mut layers = (0..43)
            .map(|layer_index| PreparedLayerExecution::new(model, layer_index, 1, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut steps = Vec::with_capacity(3);

        for (position, input_token) in [(1_u32, 201_u32), (2_u32, 361_u32), (3_u32, 1915_u32)] {
            if position > 1 {
                for (layer_index, layer) in layers.iter_mut().enumerate() {
                    layer.expected = layer_expected(layer_index as u32, position)?;
                }
            }
            submit_prepared_layers(model, &context, &mut layers, input_token, position)?;
            let output_head = run_retained_output_head(model, &context, position)?;
            let mut reports = Vec::with_capacity(43);
            for layer in &mut layers {
                reports.push(
                    run_prepared_layer_iterations(
                        model,
                        &context,
                        layer,
                        input_token,
                        position,
                        0,
                        1,
                        COMMAND_CHAINED_COLLECT,
                        42,
                    )?
                    .report,
                );
            }
            steps.push(DecoderOutputStepReport {
                position,
                input_token,
                cache_rows: position + 1,
                transformer_wall_ms: reports[0].wall_ms,
                transformer_gpu_ms: reports.iter().map(|layer| layer.gpu_ms).sum(),
                layers: reports,
                output_head,
            });
        }

        Ok(DecoderOutputProbeReport {
            steps,
            command_buffers_per_step: 44,
            host_waits_per_step: 2,
            kv_cache_layers: 43,
            cache_capacity_rows: 4,
            logits_elements: 129280,
        })
    }

    pub fn run_ratio128_compressor_replay_probe(
        model: &MappedModel,
    ) -> Result<Ratio128CompressorReplayProbeReport> {
        let context = Context::new()?;
        let mut layers = Vec::with_capacity(2);
        for layer in [3_u32, 5_u32] {
            let tensor_name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let ape = exact_tensor(
                model,
                &tensor_name("attn_compressor_ape.weight"),
                1,
                &[512, 128],
            )?;
            let kv = exact_tensor(
                model,
                &tensor_name("attn_compressor_kv.weight"),
                1,
                &[4096, 512],
            )?;
            let gate = exact_tensor(
                model,
                &tensor_name("attn_compressor_gate.weight"),
                1,
                &[4096, 512],
            )?;
            let norm = exact_tensor(
                model,
                &tensor_name("attn_compressor_norm.weight"),
                0,
                &[512],
            )?;
            let (fixture_id, inputs, expected) = ratio128_compressor_fixture(layer)?;
            let mut output = vec![0.0_f32; 512];
            let mut raw = RawIngressProbeResult::default();
            let mut error = [0 as c_char; ERROR_BYTES];
            let succeeded = unsafe {
                rust_star_metal_run_ratio128_compressor_replay(
                    context.0,
                    model.mapping_pointer(),
                    model.bytes(),
                    layer,
                    ape.absolute_offset,
                    ape.bytes,
                    kv.absolute_offset,
                    kv.bytes,
                    gate.absolute_offset,
                    gate.bytes,
                    norm.absolute_offset,
                    norm.bytes,
                    inputs.as_ptr(),
                    inputs.len() as u64,
                    output.as_mut_ptr(),
                    output.len() as u64,
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if succeeded == 0 {
                return Err(Error::invalid(format!(
                    "Metal layer-{layer} ratio-128 compressor replay failed: {}",
                    error_text(&error)
                )));
            }
            if raw.model_bytes != model.bytes()
                || raw.wrapped_model_ranges != 4
                || raw.pointer_matches != 4
            {
                return Err(Error::invalid(format!(
                    "layer-{layer} ratio-128 replay did not preserve all four mmap-backed model ranges"
                )));
            }
            for (index, (actual, expected)) in output.iter().zip(&expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "layer-{layer} ratio-128 compressed KV C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits()
                    )));
                }
            }
            if !raw.wall_ms.is_finite()
                || raw.wall_ms <= 0.0
                || !raw.gpu_ms.is_finite()
                || raw.gpu_ms < 0.0
            {
                return Err(Error::invalid(format!(
                    "layer-{layer} ratio-128 replay returned invalid timing"
                )));
            }
            layers.push(Ratio128CompressorLayerReport {
                layer,
                fixture_id,
                activation_rows: 128,
                state_rows: 128,
                dispatches: 263,
                command_buffers: 1,
                host_waits: 1,
                wrapped_model_ranges: raw.wrapped_model_ranges,
                pointer_matches: raw.pointer_matches,
                wall_ms: raw.wall_ms,
                gpu_ms: raw.gpu_ms,
                input_checksum: checksum_f32(&inputs),
                output_checksum: checksum_f32(&output),
            });
        }
        Ok(Ratio128CompressorReplayProbeReport {
            layers,
            final_position: 127,
            externally_supplied_activation_rows: 128,
        })
    }

    pub fn run_layer0_bench(
        model: &MappedModel,
        config: Layer0BenchConfig,
    ) -> Result<Layer0BenchReport> {
        let config = config.validate()?;
        let context = Context::new()?;
        let execution = run_layer_iterations(
            model,
            &context,
            COMMAND_SYNCHRONIZED,
            config.warmup_iterations,
            config.iterations,
            0,
            0,
        )?;
        let wall = summarize_timing(&execution.wall_ms_samples)?;
        let gpu = summarize_timing(&execution.gpu_ms_samples)?;
        Ok(Layer0BenchReport {
            fixture_id: execution.report.fixture_id,
            token: execution.report.token,
            warmup_iterations: config.warmup_iterations,
            iterations: config.iterations,
            dispatches_per_iteration: execution.report.dispatches,
            command_buffers_per_iteration: execution.report.command_buffers,
            selected_experts: execution.report.selected_experts,
            wrapped_model_ranges: execution.report.wrapped_model_ranges,
            pointer_matches: execution.report.pointer_matches,
            wall_ms_samples: execution.wall_ms_samples,
            gpu_ms_samples: execution.gpu_ms_samples,
            wall,
            gpu,
            repeat_bitwise_match: execution.repeat_bitwise_match,
            final_hc_checksum: execution.report.final_hc_checksum,
        })
    }

    fn run_layer_iterations(
        model: &MappedModel,
        context: &Context,
        layer_index: u32,
        warmup_iterations: u32,
        measured_iterations: u32,
        command_mode: u32,
        chain_final_layer: u32,
    ) -> Result<Layer0Execution> {
        let mut prepared = PreparedLayerExecution::new(model, layer_index, 1, measured_iterations)?;
        run_prepared_layer_iterations(
            model,
            context,
            &mut prepared,
            201,
            1,
            warmup_iterations,
            measured_iterations,
            command_mode,
            chain_final_layer,
        )
    }

    fn run_prepared_layer_iterations(
        model: &MappedModel,
        context: &Context,
        prepared: &mut PreparedLayerExecution,
        token: u32,
        position: u32,
        warmup_iterations: u32,
        measured_iterations: u32,
        command_mode: u32,
        chain_final_layer: u32,
    ) -> Result<Layer0Execution> {
        let layer_index = prepared.layer_index;
        let PreparedLayerExecution {
            layer_index: _,
            embedding,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            q_a,
            q_a_norm,
            kv,
            kv_norm_weight,
            q_b,
            sinks,
            output_a,
            output_b,
            ffn_hc_fn,
            ffn_hc_scale,
            ffn_hc_base,
            ffn_norm_weight,
            router_gate,
            router_aux,
            routed_gate,
            routed_up,
            routed_down,
            shared_gate,
            shared_up,
            shared_down,
            attention_compressor,
            indexer_compressor,
            compressor_prime,
            compressed_kv,
            compressed_indexer,
            expected,
            expected_cache_rows,
            mixes,
            split,
            collapsed,
            norm,
            q_lora,
            q_lora_norm,
            kv_raw,
            kv_after_store,
            q_raw,
            q_cur,
            kv_rope,
            kv_cur,
            cache_rows,
            attention_raw,
            attention_back,
            attention_low,
            attention_out,
            after_attention_hc,
            ffn_mixes,
            ffn_split,
            ffn_norm,
            router_logits,
            router_probs,
            selected,
            router_weights,
            routed_mid,
            routed_out,
            shared_out,
            after_ffn_hc,
            wall_ms_samples,
            gpu_ms_samples,
            repeat_bitwise_matches,
        } = prepared;

        let compressor_fields = |spans: &Option<CompressorSpans>| {
            spans.as_ref().map_or([0_u64; 8], |spans| {
                [
                    spans.ape.absolute_offset,
                    spans.ape.bytes,
                    spans.kv.absolute_offset,
                    spans.kv.bytes,
                    spans.gate.absolute_offset,
                    spans.gate.bytes,
                    spans.norm.absolute_offset,
                    spans.norm.bytes,
                ]
            })
        };
        let attention_compressor_fields = compressor_fields(attention_compressor);
        let indexer_compressor_fields = compressor_fields(indexer_compressor);

        let layer0 = RawLayer0Extension {
            hc_ffn_fn_offset: ffn_hc_fn.absolute_offset,
            hc_ffn_fn_bytes: ffn_hc_fn.bytes,
            hc_ffn_scale_offset: ffn_hc_scale.absolute_offset,
            hc_ffn_scale_bytes: ffn_hc_scale.bytes,
            hc_ffn_base_offset: ffn_hc_base.absolute_offset,
            hc_ffn_base_bytes: ffn_hc_base.bytes,
            ffn_norm_offset: ffn_norm_weight.absolute_offset,
            ffn_norm_bytes: ffn_norm_weight.bytes,
            router_gate_offset: router_gate.absolute_offset,
            router_gate_bytes: router_gate.bytes,
            router_aux_offset: router_aux.absolute_offset,
            router_aux_bytes: router_aux.bytes,
            routed_gate_offset: routed_gate.absolute_offset,
            routed_gate_bytes: routed_gate.bytes,
            routed_up_offset: routed_up.absolute_offset,
            routed_up_bytes: routed_up.bytes,
            routed_down_offset: routed_down.absolute_offset,
            routed_down_bytes: routed_down.bytes,
            shared_gate_offset: shared_gate.absolute_offset,
            shared_gate_bytes: shared_gate.bytes,
            shared_up_offset: shared_up.absolute_offset,
            shared_up_bytes: shared_up.bytes,
            shared_down_offset: shared_down.absolute_offset,
            shared_down_bytes: shared_down.bytes,
            attn_compressor_ape_offset: attention_compressor_fields[0],
            attn_compressor_ape_bytes: attention_compressor_fields[1],
            attn_compressor_kv_offset: attention_compressor_fields[2],
            attn_compressor_kv_bytes: attention_compressor_fields[3],
            attn_compressor_gate_offset: attention_compressor_fields[4],
            attn_compressor_gate_bytes: attention_compressor_fields[5],
            attn_compressor_norm_offset: attention_compressor_fields[6],
            attn_compressor_norm_bytes: attention_compressor_fields[7],
            indexer_compressor_ape_offset: indexer_compressor_fields[0],
            indexer_compressor_ape_bytes: indexer_compressor_fields[1],
            indexer_compressor_kv_offset: indexer_compressor_fields[2],
            indexer_compressor_kv_bytes: indexer_compressor_fields[3],
            indexer_compressor_gate_offset: indexer_compressor_fields[4],
            indexer_compressor_gate_bytes: indexer_compressor_fields[5],
            indexer_compressor_norm_offset: indexer_compressor_fields[6],
            indexer_compressor_norm_bytes: indexer_compressor_fields[7],
            compressor_prime_attn_norm: if compressor_prime.is_empty() {
                ptr::null()
            } else {
                compressor_prime.as_ptr()
            },
            compressed_kv_row: compressed_kv.as_mut_ptr(),
            compressed_indexer_row: compressed_indexer.as_mut_ptr(),
            ffn_mixes: ffn_mixes.as_mut_ptr(),
            ffn_split: ffn_split.as_mut_ptr(),
            ffn_norm: ffn_norm.as_mut_ptr(),
            router_logits: router_logits.as_mut_ptr(),
            router_probs: router_probs.as_mut_ptr(),
            selected: selected.as_mut_ptr(),
            router_weights: router_weights.as_mut_ptr(),
            routed_mid: routed_mid.as_mut_ptr(),
            routed_out: routed_out.as_mut_ptr(),
            shared_out: shared_out.as_mut_ptr(),
            after_ffn_hc: after_ffn_hc.as_mut_ptr(),
            warmup_iterations,
            measured_iterations,
            wall_ms_samples: wall_ms_samples.as_mut_ptr(),
            gpu_ms_samples: gpu_ms_samples.as_mut_ptr(),
            repeat_bitwise_matches,
            layer_index,
            reuse_previous_hc: u32::from(layer_index != 0),
            command_mode,
            chain_final_layer,
            position,
        };

        let mut error = [0 as c_char; ERROR_BYTES];
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_attention_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                token,
                129280,
                embedding.absolute_offset,
                embedding.bytes,
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                q_a.absolute_offset,
                q_a.bytes,
                q_a_norm.absolute_offset,
                q_a_norm.bytes,
                kv.absolute_offset,
                kv.bytes,
                kv_norm_weight.absolute_offset,
                kv_norm_weight.bytes,
                q_b.absolute_offset,
                q_b.bytes,
                sinks.absolute_offset,
                sinks.bytes,
                output_a.absolute_offset,
                output_a.bytes,
                output_b.absolute_offset,
                output_b.bytes,
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                collapsed.as_mut_ptr(),
                norm.as_mut_ptr(),
                q_lora.as_mut_ptr(),
                q_lora_norm.as_mut_ptr(),
                kv_raw.as_mut_ptr(),
                kv_after_store.as_mut_ptr(),
                q_raw.as_mut_ptr(),
                q_cur.as_mut_ptr(),
                kv_rope.as_mut_ptr(),
                kv_cur.as_mut_ptr(),
                cache_rows.as_mut_ptr(),
                expected.cache_row0.as_ptr(),
                attention_raw.as_mut_ptr(),
                attention_back.as_mut_ptr(),
                attention_low.as_mut_ptr(),
                attention_out.as_mut_ptr(),
                after_attention_hc.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
                &layer0,
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal complete layer-{layer_index} probe failed: {}",
                error_text(&error)
            )));
        }
        let expected_wrapped_ranges = if layer_index >= 2 {
            if layer_index % 2 == 0 {
                33
            } else {
                29
            }
        } else {
            25
        };
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != expected_wrapped_ranges
            || raw.pointer_matches != expected_wrapped_ranges
        {
            return Err(Error::invalid(
                "Metal complete layer path did not preserve every required mmap-backed model range",
            ));
        }
        let dispatches = match (layer_index, layer_index % 2, position) {
            (0, _, _) => 30,
            (layer, 0, 1) if layer >= 2 => 36,
            (layer, 0, 3) if layer >= 2 => 48,
            (layer, 0, _) if layer >= 2 => 32,
            (layer, 1, 1) if layer >= 3 => 32,
            (layer, 1, _) if layer >= 3 => 30,
            _ => 28,
        };
        if matches!(
            command_mode,
            COMMAND_CHAINED_ENQUEUE | COMMAND_CHAINED_FINAL
        ) {
            return Ok(Layer0Execution {
                report: Layer0ProbeReport {
                    fixture_id: expected.fixture_id,
                    token,
                    dispatches,
                    command_buffers: 1,
                    selected_experts: expected.selected.clone(),
                    wrapped_model_ranges: raw.wrapped_model_ranges,
                    pointer_matches: raw.pointer_matches,
                    wall_ms: 0.0,
                    gpu_ms: 0.0,
                    attention_hc_checksum: 0,
                    ffn_norm_checksum: 0,
                    router_weights_checksum: 0,
                    routed_mid_checksum: 0,
                    routed_out_checksum: 0,
                    shared_out_checksum: 0,
                    final_hc_checksum: 0,
                },
                wall_ms_samples: wall_ms_samples.clone(),
                gpu_ms_samples: gpu_ms_samples.clone(),
                repeat_bitwise_match: true,
            });
        }
        if command_mode == COMMAND_CHAINED_TIMING {
            if !raw.wall_ms.is_finite()
                || raw.wall_ms <= 0.0
                || !raw.gpu_ms.is_finite()
                || raw.gpu_ms < 0.0
            {
                return Err(Error::invalid(
                    "Metal four-layer chain returned invalid timing",
                ));
            }
            return Ok(Layer0Execution {
                report: Layer0ProbeReport {
                    fixture_id: expected.fixture_id,
                    token,
                    dispatches,
                    command_buffers: 1,
                    selected_experts: expected.selected.clone(),
                    wrapped_model_ranges: raw.wrapped_model_ranges,
                    pointer_matches: raw.pointer_matches,
                    wall_ms: raw.wall_ms,
                    gpu_ms: raw.gpu_ms,
                    attention_hc_checksum: 0,
                    ffn_norm_checksum: 0,
                    router_weights_checksum: 0,
                    routed_mid_checksum: 0,
                    routed_out_checksum: 0,
                    shared_out_checksum: 0,
                    final_hc_checksum: 0,
                },
                wall_ms_samples: wall_ms_samples.clone(),
                gpu_ms_samples: gpu_ms_samples.clone(),
                repeat_bitwise_match: true,
            });
        }
        if *selected != expected.selected {
            return Err(Error::invalid(format!(
                "complete layer-{layer_index} selected experts differ: actual={selected:?} expected={:?}",
                expected.selected
            )));
        }
        if *repeat_bitwise_matches != 1 {
            return Err(Error::invalid(
                "complete layer-0 outputs changed across measured iterations",
            ));
        }
        let expected_prior_cache_elements = position as usize * 512;
        if expected_cache_rows.len() != expected_prior_cache_elements {
            return Err(Error::invalid(format!(
                "complete layer-{layer_index} cache history has {} rows before position {position}",
                expected_cache_rows.len() / 512
            )));
        }
        for (index, (actual, expected_value)) in cache_rows
            .iter()
            .take(expected_prior_cache_elements)
            .zip(expected_cache_rows.iter())
            .enumerate()
        {
            if actual.to_bits() != expected_value.to_bits() {
                return Err(Error::invalid(format!(
                    "complete layer-{layer_index} C0 mismatch in retained cache[{index}]: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected_value.to_bits()
                )));
            }
        }
        let target_start = expected_prior_cache_elements;
        let stored_cache_row = expected
            .kv_cur
            .iter()
            .copied()
            .map(f16_round_f32)
            .collect::<Vec<_>>();
        for (index, (actual, expected_value)) in cache_rows[target_start..target_start + 512]
            .iter()
            .zip(stored_cache_row.iter())
            .enumerate()
        {
            if actual.to_bits() != expected_value.to_bits() {
                return Err(Error::invalid(format!(
                    "complete layer-{layer_index} C0 mismatch in cache row {position}[{index}]: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected_value.to_bits()
                )));
            }
        }
        expected_cache_rows.extend_from_slice(&stored_cache_row);
        if !expected.compressed_kv.is_empty() {
            for (index, (actual, expected)) in compressed_kv
                .iter()
                .zip(expected.compressed_kv.iter())
                .enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "complete layer-{layer_index} C0 mismatch in compressed KV row[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "hc_attn_pre_mixes",
                mixes.as_slice(),
                expected.attention_mixes.as_slice(),
            ),
            (
                "hc_attn_pre_split",
                split.as_slice(),
                expected.attention_split.as_slice(),
            ),
            (
                "hc_attn_pre",
                collapsed.as_slice(),
                expected.attention_collapsed.as_slice(),
            ),
            (
                "attn_norm",
                norm.as_slice(),
                expected.attention_norm.as_slice(),
            ),
            ("q_lora", q_lora.as_slice(), expected.q_lora.as_slice()),
            (
                "q_lora_norm",
                q_lora_norm.as_slice(),
                expected.q_lora_norm.as_slice(),
            ),
            ("kv_raw", kv_raw.as_slice(), expected.kv_raw.as_slice()),
            ("q_raw", q_raw.as_slice(), expected.q_raw.as_slice()),
            ("q_cur", q_cur.as_slice(), expected.q_cur.as_slice()),
            ("kv_rope", kv_rope.as_slice(), expected.kv_rope.as_slice()),
            ("kv_cur", kv_cur.as_slice(), expected.kv_cur.as_slice()),
            (
                "kqv_back",
                attention_back.as_slice(),
                expected.attention_back.as_slice(),
            ),
            (
                "attn_low",
                attention_low.as_slice(),
                expected.attention_low.as_slice(),
            ),
            (
                "attn_out",
                attention_out.as_slice(),
                expected.attention_out.as_slice(),
            ),
            (
                "hc_attn_post",
                after_attention_hc.as_slice(),
                expected.attention_hc.as_slice(),
            ),
            (
                "hc_ffn_pre_mixes",
                ffn_mixes.as_slice(),
                expected.ffn_mixes.as_slice(),
            ),
            (
                "hc_ffn_pre_split",
                ffn_split.as_slice(),
                expected.ffn_split.as_slice(),
            ),
            (
                "ffn_norm",
                ffn_norm.as_slice(),
                expected.ffn_norm.as_slice(),
            ),
            (
                "router_logits",
                router_logits.as_slice(),
                expected.router_logits.as_slice(),
            ),
            (
                "router_probs",
                router_probs.as_slice(),
                expected.router_probs.as_slice(),
            ),
            (
                "router_weights",
                router_weights.as_slice(),
                expected.router_weights.as_slice(),
            ),
            (
                "routed_mid",
                routed_mid.as_slice(),
                expected.routed_mid.as_slice(),
            ),
            (
                "routed_out",
                routed_out.as_slice(),
                expected.routed_out.as_slice(),
            ),
            (
                "shared_out",
                shared_out.as_slice(),
                expected.shared_out.as_slice(),
            ),
            (
                "hc_ffn_post",
                after_ffn_hc.as_slice(),
                expected.final_hc.as_slice(),
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "complete layer-{layer_index} C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        if !raw.wall_ms.is_finite()
            || raw.wall_ms <= 0.0
            || !raw.gpu_ms.is_finite()
            || raw.gpu_ms < 0.0
        {
            return Err(Error::invalid(
                "Metal complete layer returned invalid timing",
            ));
        }
        Ok(Layer0Execution {
            report: Layer0ProbeReport {
                fixture_id: expected.fixture_id,
                token,
                dispatches,
                command_buffers: 1,
                selected_experts: selected.clone(),
                wrapped_model_ranges: raw.wrapped_model_ranges,
                pointer_matches: raw.pointer_matches,
                wall_ms: raw.wall_ms,
                gpu_ms: raw.gpu_ms,
                attention_hc_checksum: checksum_f32(&after_attention_hc),
                ffn_norm_checksum: checksum_f32(&ffn_norm),
                router_weights_checksum: checksum_f32(&router_weights),
                routed_mid_checksum: checksum_f32(&routed_mid),
                routed_out_checksum: checksum_f32(&routed_out),
                shared_out_checksum: checksum_f32(&shared_out),
                final_hc_checksum: checksum_f32(&after_ffn_hc),
            },
            wall_ms_samples: wall_ms_samples.clone(),
            gpu_ms_samples: gpu_ms_samples.clone(),
            repeat_bitwise_match: *repeat_bitwise_matches == 1,
        })
    }

    pub fn run_ffn_router_probe(model: &MappedModel) -> Result<FfnRouterProbeReport> {
        const TOKEN: u32 = 201;
        let hc_fn = exact_tensor(model, "blk.0.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_ffn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_ffn_base.weight", 0, &[24])?;
        let norm_weight = exact_tensor(model, "blk.0.ffn_norm.weight", 0, &[4096])?;
        let gate = exact_tensor(model, "blk.0.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let hash = exact_tensor(model, "blk.0.ffn_gate_tid2eid.weight", 26, &[6, 129280])?;
        let (
            input_hc,
            expected_mixes,
            expected_split,
            expected_cur,
            expected_norm,
            expected_logits,
            expected_probs,
            expected_selected,
            expected_weights,
        ) = ffn_router_fixture()?;

        let mut mixes = vec![0.0_f32; 24];
        let mut split = vec![0.0_f32; 24];
        let mut ffn_cur = vec![0.0_f32; 4096];
        let mut ffn_norm = vec![0.0_f32; 4096];
        let mut logits = vec![0.0_f32; 256];
        let mut probs = vec![0.0_f32; 256];
        let mut selected = vec![0_i32; 6];
        let mut weights = vec![0.0_f32; 6];
        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_ffn_router(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                hc_fn.absolute_offset,
                hc_fn.bytes,
                hc_scale.absolute_offset,
                hc_scale.bytes,
                hc_base.absolute_offset,
                hc_base.bytes,
                norm_weight.absolute_offset,
                norm_weight.bytes,
                gate.absolute_offset,
                gate.bytes,
                0,
                0,
                hash.absolute_offset,
                hash.bytes,
                input_hc.as_ptr(),
                mixes.as_mut_ptr(),
                split.as_mut_ptr(),
                ffn_cur.as_mut_ptr(),
                ffn_norm.as_mut_ptr(),
                logits.as_mut_ptr(),
                probs.as_mut_ptr(),
                selected.as_mut_ptr(),
                weights.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 FFN router probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 6
            || raw.pointer_matches != 6
        {
            return Err(Error::invalid(
                "Metal FFN router path did not preserve all six mmap-backed model ranges",
            ));
        }
        if selected != expected_selected {
            return Err(Error::invalid(format!(
                "FFN router C0 mismatch in selected experts: actual={selected:?} expected={expected_selected:?}"
            )));
        }
        for (label, actual, expected) in [
            (
                "hc_ffn_pre_mixes",
                mixes.as_slice(),
                expected_mixes.as_slice(),
            ),
            (
                "hc_ffn_pre_split",
                split.as_slice(),
                expected_split.as_slice(),
            ),
            ("hc_ffn_pre", ffn_cur.as_slice(), expected_cur.as_slice()),
            ("ffn_norm", ffn_norm.as_slice(), expected_norm.as_slice()),
            (
                "ffn_moe_logits",
                logits.as_slice(),
                expected_logits.as_slice(),
            ),
            ("ffn_moe_probs", probs.as_slice(), expected_probs.as_slice()),
            (
                "ffn_moe_weights_scaled",
                weights.as_slice(),
                expected_weights.as_slice(),
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "FFN router C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal FFN router returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal FFN router returned a zero wall interval",
            ));
        }
        Ok(FfnRouterProbeReport {
            fixture_id: FFN_ROUTER_FIXTURE_ID,
            token: TOKEN,
            dispatches: 7,
            expert_count: 256,
            selected_experts: selected,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            hc_mixes_checksum: checksum_f32(&mixes),
            hc_split_checksum: checksum_f32(&split),
            ffn_cur_checksum: checksum_f32(&ffn_cur),
            ffn_norm_checksum: checksum_f32(&ffn_norm),
            router_logits_checksum: checksum_f32(&logits),
            router_probs_checksum: checksum_f32(&probs),
            router_weights_checksum: checksum_f32(&weights),
        })
    }

    pub fn run_moe_output_probe(model: &MappedModel) -> Result<MoeOutputProbeReport> {
        const TOKEN: u32 = 201;
        let routed_gate =
            exact_tensor(model, "blk.0.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_up = exact_tensor(model, "blk.0.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_down =
            exact_tensor(model, "blk.0.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let shared_gate = exact_tensor(model, "blk.0.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let shared_up = exact_tensor(model, "blk.0.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let shared_down = exact_tensor(model, "blk.0.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let (
            ffn_norm,
            selected,
            weights,
            input_hc,
            split,
            expected_routed_mid,
            expected_routed_out,
            expected_shared_out,
            expected_hc_post,
        ) = moe_output_fixture()?;
        let mut routed_mid = vec![0.0_f32; 6 * 2048];
        let mut routed_out = vec![0.0_f32; 4096];
        let mut shared_out = vec![0.0_f32; 4096];
        let mut hc_post = vec![0.0_f32; 4 * 4096];
        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawIngressProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_moe_output(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                routed_gate.absolute_offset,
                routed_gate.bytes,
                routed_up.absolute_offset,
                routed_up.bytes,
                routed_down.absolute_offset,
                routed_down.bytes,
                shared_gate.absolute_offset,
                shared_gate.bytes,
                shared_up.absolute_offset,
                shared_up.bytes,
                shared_down.absolute_offset,
                shared_down.bytes,
                ffn_norm.as_ptr(),
                selected.as_ptr(),
                weights.as_ptr(),
                input_hc.as_ptr(),
                split.as_ptr(),
                routed_mid.as_mut_ptr(),
                routed_out.as_mut_ptr(),
                shared_out.as_mut_ptr(),
                hc_post.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal layer-0 MoE output probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 6
            || raw.pointer_matches != 6
        {
            return Err(Error::invalid(
                "Metal MoE output path did not preserve all six mmap-backed model ranges",
            ));
        }
        for (label, actual, expected) in [
            (
                "ffn_moe_weighted_swiglu",
                routed_mid.as_slice(),
                expected_routed_mid.as_slice(),
            ),
            (
                "ffn_moe_out",
                routed_out.as_slice(),
                expected_routed_out.as_slice(),
            ),
            (
                "ffn_shexp",
                shared_out.as_slice(),
                expected_shared_out.as_slice(),
            ),
            (
                "hc_ffn_post",
                hc_post.as_slice(),
                expected_hc_post.as_slice(),
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "MoE output C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits()
                    )));
                }
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal MoE output returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal MoE output returned a zero wall interval",
            ));
        }
        Ok(MoeOutputProbeReport {
            fixture_id: MOE_OUTPUT_FIXTURE_ID,
            token: TOKEN,
            dispatches: 4,
            selected_experts: selected,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            routed_mid_checksum: checksum_f32(&routed_mid),
            routed_out_checksum: checksum_f32(&routed_out),
            shared_out_checksum: checksum_f32(&shared_out),
            hc_post_checksum: checksum_f32(&hc_post),
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    pub struct LayerExecutor<'a> {
        model: &'a MappedModel,
        next_layer: u32,
    }

    impl<'a> LayerExecutor<'a> {
        pub fn new(model: &'a MappedModel) -> Result<Self> {
            Ok(Self {
                model,
                next_layer: 0,
            })
        }

        pub fn execute_layer(&mut self, layer_index: u32) -> Result<Layer0ProbeReport> {
            if layer_index != self.next_layer {
                return Err(Error::invalid(format!(
                    "persistent layer executor expected layer {}, received layer {layer_index}",
                    self.next_layer
                )));
            }
            let _ = layer_expected(layer_index, 1)?;
            let _ = exact_tensor(
                self.model,
                &format!("blk.{layer_index}.ffn_down_shexp.weight"),
                8,
                &[2048, 4096],
            )?;
            Err(Error::invalid(
                "the persistent Metal layer executor is available only on macOS",
            ))
        }
    }

    pub fn run_probe(config: ProbeConfig) -> Result<ProbeReport> {
        config.validate()?;
        Err(Error::invalid(
            "the Metal dispatch probe is available only on macOS",
        ))
    }

    pub fn run_layers0123_bench(
        model: &MappedModel,
        config: Layers0123BenchConfig,
    ) -> Result<Layers0123BenchReport> {
        let _ = config.validate()?;
        for layer_index in 0..=3 {
            let _ = layer_expected(layer_index, 1)?;
        }
        let _ = exact_tensor(model, "blk.3.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the Metal layers-0/1/2/3 benchmark is available only on macOS",
        ))
    }

    pub fn run_layers0123_decode_probe(model: &MappedModel) -> Result<Layers0123DecodeProbeReport> {
        for layer_index in 0..=3 {
            let _ = layer_expected(layer_index, 1)?;
            let _ = layer_expected(layer_index, 2)?;
            let _ = layer_expected(layer_index, 3)?;
        }
        let _ = exact_tensor(model, "blk.3.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the Metal position-advancing four-layer probe is available only on macOS",
        ))
    }

    pub fn run_layers012345_decode_probe(
        model: &MappedModel,
    ) -> Result<Layers012345DecodeProbeReport> {
        for layer_index in 0..=5 {
            let _ = layer_expected(layer_index, 1)?;
            let _ = layer_expected(layer_index, 2)?;
            let _ = layer_expected(layer_index, 3)?;
        }
        let _ = exact_tensor(model, "blk.5.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the Metal position-advancing six-layer probe is available only on macOS",
        ))
    }

    pub fn run_layers01234567_decode_probe(
        model: &MappedModel,
    ) -> Result<Layers01234567DecodeProbeReport> {
        for layer_index in 0..=7 {
            let _ = layer_expected(layer_index, 1)?;
            let _ = layer_expected(layer_index, 2)?;
            let _ = layer_expected(layer_index, 3)?;
        }
        let _ = exact_tensor(model, "blk.7.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the Metal position-advancing eight-layer probe is available only on macOS",
        ))
    }

    pub fn run_layers0_to_42_decode_probe(
        model: &MappedModel,
    ) -> Result<Layers0To42DecodeProbeReport> {
        for layer_index in 0..=42 {
            let _ = layer_expected(layer_index, 1)?;
            let _ = layer_expected(layer_index, 2)?;
            let _ = layer_expected(layer_index, 3)?;
        }
        let _ = exact_tensor(model, "blk.42.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the Metal position-advancing forty-three-layer probe is available only on macOS",
        ))
    }

    pub fn run_decoder_output_probe(model: &MappedModel) -> Result<DecoderOutputProbeReport> {
        for position in 1..=3 {
            let _ = output_head_expected(position)?;
        }
        let _ = exact_tensor(model, "output_hc_fn.weight", 1, &[16384, 4])?;
        let _ = exact_tensor(model, "output_hc_scale.weight", 0, &[1])?;
        let _ = exact_tensor(model, "output_hc_base.weight", 0, &[4])?;
        let _ = exact_tensor(model, "output_norm.weight", 0, &[4096])?;
        let _ = exact_tensor(model, "output.weight", 8, &[4096, 129280])?;
        Err(Error::invalid(
            "the Metal decoder-output probe is available only on macOS",
        ))
    }

    pub fn run_ratio128_compressor_replay_probe(
        model: &MappedModel,
    ) -> Result<Ratio128CompressorReplayProbeReport> {
        for layer in [3_u32, 5_u32] {
            let _ = ratio128_compressor_fixture(layer)?;
            let _ = exact_tensor(
                model,
                &format!("blk.{layer}.attn_compressor_kv.weight"),
                1,
                &[4096, 512],
            )?;
        }
        Err(Error::invalid(
            "the Metal ratio-128 compressor replay probe is available only on macOS",
        ))
    }

    pub fn run_layers012_chained_probe(model: &MappedModel) -> Result<Layers012ChainedProbeReport> {
        let _ = layer_expected(0, 1)?;
        let _ = layer_expected(1, 1)?;
        let _ = layer_expected(2, 1)?;
        let _ = exact_tensor(model, "blk.2.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the chained Metal layers-0/1/2 probe is available only on macOS",
        ))
    }

    pub fn run_layers0123_chained_probe(
        model: &MappedModel,
    ) -> Result<Layers0123ChainedProbeReport> {
        for layer_index in 0..=3 {
            let _ = layer_expected(layer_index, 1)?;
        }
        let _ = exact_tensor(model, "blk.3.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the chained Metal layers-0/1/2/3 probe is available only on macOS",
        ))
    }

    pub fn run_f16_embedding_probe(
        model: &MappedModel,
        tensor: &TensorInfo,
        tokens: &[u32],
    ) -> Result<EmbeddingProbeReport> {
        validate_embedding_inputs(model, tensor, tokens)?;
        Err(Error::invalid(
            "the Metal F16 embedding probe is available only on macOS",
        ))
    }

    pub fn run_q8_projection_probe(
        model: &MappedModel,
        tensor: &TensorInfo,
    ) -> Result<ProjectionProbeReport> {
        let (input, expected) = projection_fixture()?;
        validate_projection_inputs(model, tensor, &input, &expected)?;
        Err(Error::invalid(
            "the Metal Q8_0 projection probe is available only on macOS",
        ))
    }

    pub fn run_attention_ingress_probe(model: &MappedModel) -> Result<IngressProbeReport> {
        let _ = ingress_fixture()?;
        let _ = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        Err(Error::invalid(
            "the Metal layer-0 attention ingress probe is available only on macOS",
        ))
    }

    pub fn run_attention_setup_probe(model: &MappedModel) -> Result<AttentionSetupProbeReport> {
        let _ = attention_setup_fixture()?;
        let _ = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        Err(Error::invalid(
            "the Metal layer-0 attention setup probe is available only on macOS",
        ))
    }

    pub fn run_rope_kv_store_probe(model: &MappedModel) -> Result<RopeKvStoreProbeReport> {
        let _ = attention_setup_fixture()?;
        let _ = rope_kv_store_fixture()?;
        let _ = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        Err(Error::invalid(
            "the Metal layer-0 RoPE/KV-store probe is available only on macOS",
        ))
    }

    pub fn run_attention_read_probe(model: &MappedModel) -> Result<AttentionReadProbeReport> {
        let _ = attention_read_fixture()?;
        let _ = exact_tensor(model, "blk.0.attn_sinks.weight", 0, &[64])?;
        Err(Error::invalid(
            "the Metal layer-0 attention-read probe is available only on macOS",
        ))
    }

    pub fn run_attention_output_probe(model: &MappedModel) -> Result<AttentionOutputProbeReport> {
        let _ = attention_output_fixture()?;
        let _ = exact_tensor(model, "blk.0.attn_output_a.weight", 8, &[4096, 8192])?;
        let _ = exact_tensor(model, "blk.0.attn_output_b.weight", 8, &[8192, 4096])?;
        Err(Error::invalid(
            "the Metal layer-0 attention-output probe is available only on macOS",
        ))
    }

    pub fn run_ffn_router_probe(model: &MappedModel) -> Result<FfnRouterProbeReport> {
        let _ = ffn_router_fixture()?;
        let _ = exact_tensor(model, "blk.0.ffn_gate_inp.weight", 1, &[4096, 256])?;
        Err(Error::invalid(
            "the Metal layer-0 FFN router probe is available only on macOS",
        ))
    }

    pub fn run_moe_output_probe(model: &MappedModel) -> Result<MoeOutputProbeReport> {
        let _ = moe_output_fixture()?;
        let _ = exact_tensor(model, "blk.0.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        Err(Error::invalid(
            "the Metal layer-0 MoE output probe is available only on macOS",
        ))
    }

    pub fn run_layer0_probe(model: &MappedModel) -> Result<Layer0ProbeReport> {
        let _ = attention_output_fixture()?;
        let _ = ffn_router_fixture()?;
        let _ = moe_output_fixture()?;
        let _ = exact_tensor(model, "blk.0.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the complete Metal layer-0 probe is available only on macOS",
        ))
    }

    pub fn run_layers01_probe(model: &MappedModel) -> Result<Layers01ProbeReport> {
        let _ = layer_expected(0, 1)?;
        let _ = layer_expected(1, 1)?;
        let _ = exact_tensor(model, "blk.1.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the continuous Metal layers-0/1 probe is available only on macOS",
        ))
    }

    pub fn run_layers012_probe(model: &MappedModel) -> Result<Layers012ProbeReport> {
        let _ = layer_expected(0, 1)?;
        let _ = layer_expected(1, 1)?;
        let _ = layer_expected(2, 1)?;
        let _ = exact_tensor(model, "blk.2.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the continuous Metal layers-0/1/2 probe is available only on macOS",
        ))
    }

    pub fn run_layers0123_probe(model: &MappedModel) -> Result<Layers0123ProbeReport> {
        for layer_index in 0..=3 {
            let _ = layer_expected(layer_index, 1)?;
        }
        let _ = exact_tensor(model, "blk.3.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the continuous Metal layers-0/1/2/3 probe is available only on macOS",
        ))
    }

    pub fn run_layer0_bench(
        model: &MappedModel,
        config: Layer0BenchConfig,
    ) -> Result<Layer0BenchReport> {
        let _ = config.validate()?;
        let _ = attention_output_fixture()?;
        let _ = ffn_router_fixture()?;
        let _ = moe_output_fixture()?;
        let _ = exact_tensor(model, "blk.0.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the Metal layer-0 steady-state benchmark is available only on macOS",
        ))
    }
}

pub use imp::{
    run_attention_ingress_probe, run_attention_output_probe, run_attention_read_probe,
    run_attention_setup_probe, run_decoder_output_probe, run_f16_embedding_probe,
    run_ffn_router_probe, run_layer0_bench, run_layer0_probe, run_layers01234567_decode_probe,
    run_layers012345_decode_probe, run_layers0123_bench, run_layers0123_chained_probe,
    run_layers0123_decode_probe, run_layers0123_probe, run_layers012_chained_probe,
    run_layers012_probe, run_layers01_probe, run_layers0_to_42_decode_probe, run_moe_output_probe,
    run_probe, run_q8_projection_probe, run_ratio128_compressor_replay_probe,
    run_rope_kv_store_probe, LayerExecutor,
};

#[cfg(test)]
mod tests {
    use super::*;

    const POSITION2_FIXTURE_IDS: [&str; 8] = [
        LAYER0_POS2_FIXTURE_ID,
        LAYER1_POS2_FIXTURE_ID,
        LAYER2_POS2_FIXTURE_ID,
        LAYER3_POS2_FIXTURE_ID,
        LAYER4_POS2_FIXTURE_ID,
        LAYER5_POS2_FIXTURE_ID,
        LAYER6_POS2_FIXTURE_ID,
        LAYER7_POS2_FIXTURE_ID,
    ];
    const POSITION3_FIXTURE_IDS: [&str; 8] = [
        LAYER0_POS3_FIXTURE_ID,
        LAYER1_POS3_FIXTURE_ID,
        LAYER2_POS3_FIXTURE_ID,
        LAYER3_POS3_FIXTURE_ID,
        LAYER4_POS3_FIXTURE_ID,
        LAYER5_POS3_FIXTURE_ID,
        LAYER6_POS3_FIXTURE_ID,
        LAYER7_POS3_FIXTURE_ID,
    ];

    fn report() -> ProbeReport {
        ProbeReport {
            device_name: "Apple GPU \"fixture\"".to_owned(),
            has_unified_memory: true,
            recommended_max_working_set_bytes: 100,
            max_total_threads_per_threadgroup: 1024,
            elements: 4,
            iterations: 2,
            buffer_bytes: 16,
            checksum: 42,
            setup_ms: 1.0,
            compile_ms: 2.0,
            warmup_wall_ms: 3.0,
            warmup_gpu_ms: 1.0,
            roundtrip_wall_ms: 4.0,
            roundtrip_gpu_ms: 2.0,
            batched_wall_ms: 1.0,
            batched_gpu_ms: 0.5,
        }
    }

    fn embedding_report() -> EmbeddingProbeReport {
        EmbeddingProbeReport {
            tensor_name: "token_embd.weight".to_owned(),
            tokens: vec![0, 7],
            embedding_elements: 4,
            output_elements: 8,
            model_bytes: 1000,
            tensor_offset: 33,
            tensor_bytes: 64,
            page_offset: 0,
            buffer_bytes: 4096,
            inner_offset: 33,
            max_buffer_length: 1 << 30,
            no_copy_pointer_match: true,
            wall_ms: 1.5,
            gpu_ms: 0.5,
            checksum: 99,
        }
    }

    fn projection_report() -> ProjectionProbeReport {
        ProjectionProbeReport {
            fixture_id: PROJECTION_FIXTURE_ID,
            tensor_name: PROJECTION_TENSOR.to_owned(),
            input_elements: 4096,
            output_elements: 1024,
            model_bytes: 100_000,
            tensor_offset: 4097,
            tensor_bytes: 4_456_448,
            page_offset: 4096,
            buffer_bytes: 4_460_544,
            inner_offset: 1,
            max_buffer_length: 1 << 30,
            no_copy_pointer_match: true,
            simdgroups: 4,
            rows_per_threadgroup: 2,
            wall_ms: 1.0,
            gpu_ms: 0.5,
            input_checksum: 11,
            output_checksum: 12,
        }
    }

    fn ratio128_compressor_report() -> Ratio128CompressorReplayProbeReport {
        Ratio128CompressorReplayProbeReport {
            layers: [
                (3, LAYER3_POS127_COMPRESSOR_FIXTURE_ID, 11, 12),
                (5, LAYER5_POS127_COMPRESSOR_FIXTURE_ID, 13, 14),
            ]
            .into_iter()
            .map(|(layer, fixture_id, input_checksum, output_checksum)| {
                Ratio128CompressorLayerReport {
                    layer,
                    fixture_id,
                    activation_rows: 128,
                    state_rows: 128,
                    dispatches: 263,
                    command_buffers: 1,
                    host_waits: 1,
                    wrapped_model_ranges: 4,
                    pointer_matches: 4,
                    wall_ms: 2.0,
                    gpu_ms: 1.0,
                    input_checksum,
                    output_checksum,
                }
            })
            .collect(),
            final_position: 127,
            externally_supplied_activation_rows: 128,
        }
    }

    fn ingress_report() -> IngressProbeReport {
        IngressProbeReport {
            fixture_id: INGRESS_FIXTURE_ID,
            token: 201,
            wrapped_model_ranges: 6,
            pointer_matches: 6,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            mixes_checksum: 1,
            split_checksum: 2,
            collapsed_checksum: 3,
            attn_norm_checksum: 4,
            q_lora_checksum: 5,
        }
    }

    fn attention_setup_report() -> AttentionSetupProbeReport {
        AttentionSetupProbeReport {
            fixture_id: ATTENTION_SETUP_FIXTURE_ID,
            token: 201,
            dispatches: 9,
            wrapped_model_ranges: 10,
            pointer_matches: 10,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            q_lora_norm_checksum: 1,
            kv_raw_checksum: 2,
            kv_norm_checksum: 3,
            q_raw_checksum: 4,
        }
    }

    fn rope_kv_store_report() -> RopeKvStoreProbeReport {
        RopeKvStoreProbeReport {
            fixture_id: ROPE_KV_STORE_FIXTURE_ID,
            token: 201,
            dispatches: 12,
            cache_capacity_rows: 3,
            cache_target_row: 1,
            cache_guard_rows_intact: true,
            wrapped_model_ranges: 10,
            pointer_matches: 10,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            q_cur_checksum: 1,
            kv_rope_checksum: 2,
            kv_cur_checksum: 3,
            cache_row_checksum: 4,
        }
    }

    fn attention_read_report() -> AttentionReadProbeReport {
        AttentionReadProbeReport {
            fixture_id: ATTENTION_READ_FIXTURE_ID,
            token: 201,
            dispatches: 17,
            cache_capacity_rows: 3,
            cache_rows_read: 2,
            cache_row0_preserved: true,
            cache_guard_row_intact: true,
            wrapped_model_ranges: 11,
            pointer_matches: 11,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            attention_raw_checksum: 5,
            attention_back_checksum: 6,
        }
    }

    fn attention_output_report() -> AttentionOutputProbeReport {
        AttentionOutputProbeReport {
            fixture_id: ATTENTION_OUTPUT_FIXTURE_ID,
            token: 201,
            dispatches: 19,
            output_groups: 8,
            output_rank: 1024,
            wrapped_model_ranges: 13,
            pointer_matches: 13,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            attention_low_checksum: 7,
            attention_out_checksum: 8,
            hc_post_checksum: 9,
        }
    }

    fn ffn_router_report() -> FfnRouterProbeReport {
        FfnRouterProbeReport {
            fixture_id: FFN_ROUTER_FIXTURE_ID,
            token: 201,
            dispatches: 7,
            expert_count: 256,
            selected_experts: vec![17, 42, 63, 99, 101, 201],
            wrapped_model_ranges: 6,
            pointer_matches: 6,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            hc_mixes_checksum: 1,
            hc_split_checksum: 2,
            ffn_cur_checksum: 3,
            ffn_norm_checksum: 4,
            router_logits_checksum: 5,
            router_probs_checksum: 6,
            router_weights_checksum: 7,
        }
    }

    fn moe_output_report() -> MoeOutputProbeReport {
        MoeOutputProbeReport {
            fixture_id: MOE_OUTPUT_FIXTURE_ID,
            token: 201,
            dispatches: 4,
            selected_experts: vec![25, 174, 215, 58, 48, 60],
            wrapped_model_ranges: 6,
            pointer_matches: 6,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            routed_mid_checksum: 1,
            routed_out_checksum: 2,
            shared_out_checksum: 3,
            hc_post_checksum: 4,
        }
    }

    fn layer0_report() -> Layer0ProbeReport {
        Layer0ProbeReport {
            fixture_id: LAYER0_FIXTURE_ID,
            token: 201,
            dispatches: 30,
            command_buffers: 1,
            selected_experts: vec![25, 174, 215, 58, 48, 60],
            wrapped_model_ranges: 25,
            pointer_matches: 25,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            attention_hc_checksum: 1,
            ffn_norm_checksum: 2,
            router_weights_checksum: 3,
            routed_mid_checksum: 4,
            routed_out_checksum: 5,
            shared_out_checksum: 6,
            final_hc_checksum: 7,
        }
    }

    fn layer0_bench_report() -> Layer0BenchReport {
        Layer0BenchReport {
            fixture_id: LAYER0_FIXTURE_ID,
            token: 201,
            warmup_iterations: 2,
            iterations: 3,
            dispatches_per_iteration: 30,
            command_buffers_per_iteration: 1,
            selected_experts: vec![25, 174, 215, 58, 48, 60],
            wrapped_model_ranges: 25,
            pointer_matches: 25,
            wall_ms_samples: vec![1.0, 2.0, 4.0],
            gpu_ms_samples: vec![0.5, 0.75, 1.0],
            wall: TimingSummary {
                median_ms: 2.0,
                mad_ms: 1.0,
                min_ms: 1.0,
                max_ms: 4.0,
            },
            gpu: TimingSummary {
                median_ms: 0.75,
                mad_ms: 0.25,
                min_ms: 0.5,
                max_ms: 1.0,
            },
            repeat_bitwise_match: true,
            final_hc_checksum: 7,
        }
    }

    fn layers01_report() -> Layers01ProbeReport {
        let mut layer1 = layer0_report();
        layer1.fixture_id = LAYER1_FIXTURE_ID;
        layer1.dispatches = 28;
        layer1.selected_experts = vec![228, 208, 35, 27, 113, 12];
        layer1.final_hc_checksum = 8;
        LayerSequenceProbeReport {
            layers: vec![layer0_report(), layer1],
            command_buffers: 2,
            retained_hc_handoff: true,
            kv_cache_layers: 2,
        }
    }

    fn layers012_report() -> Layers012ProbeReport {
        let mut report = layers01_report();
        let mut layer2 = layer0_report();
        layer2.fixture_id = LAYER2_FIXTURE_ID;
        layer2.dispatches = 28;
        layer2.selected_experts = vec![8, 188, 195, 75, 96, 176];
        layer2.final_hc_checksum = 9;
        report.layers.push(layer2);
        report.command_buffers = 3;
        report.kv_cache_layers = 3;
        report
    }

    fn layers012_chained_report() -> Layers012ChainedProbeReport {
        Layers012ChainedProbeReport {
            layers: layers012_report().layers,
            command_buffers: 3,
            host_waits: 1,
            retained_hc_handoff: true,
            kv_cache_layers: 3,
            wall_ms: 4.0,
            gpu_ms: 3.0,
        }
    }

    fn layers0123_report() -> Layers0123ProbeReport {
        let mut report = layers012_report();
        let mut layer3 = layer0_report();
        layer3.fixture_id = LAYER3_FIXTURE_ID;
        layer3.dispatches = 28;
        layer3.selected_experts = vec![1, 58, 68, 240, 20, 24];
        layer3.final_hc_checksum = 10;
        report.layers.push(layer3);
        report.command_buffers = 4;
        report.kv_cache_layers = 4;
        report
    }

    fn layers0123_chained_report() -> Layers0123ChainedProbeReport {
        Layers012ChainedProbeReport {
            layers: layers0123_report().layers,
            command_buffers: 4,
            host_waits: 1,
            retained_hc_handoff: true,
            kv_cache_layers: 4,
            wall_ms: 5.0,
            gpu_ms: 4.0,
        }
    }

    fn layers0123_bench_report() -> Layers0123BenchReport {
        Layers0123BenchReport {
            warmup_iterations: 2,
            iterations: 3,
            command_buffers_per_iteration: 4,
            host_waits_per_iteration: 1,
            retained_hc_handoff: true,
            kv_cache_layers: 4,
            wall_ms_samples: vec![4.0, 3.0, 5.0],
            gpu_ms_samples: vec![2.0, 1.0, 3.0],
            wall: TimingSummary {
                median_ms: 4.0,
                mad_ms: 1.0,
                min_ms: 3.0,
                max_ms: 5.0,
            },
            gpu: TimingSummary {
                median_ms: 2.0,
                mad_ms: 1.0,
                min_ms: 1.0,
                max_ms: 3.0,
            },
            final_layers: layers0123_report().layers,
        }
    }

    fn layers0123_decode_report() -> Layers0123DecodeProbeReport {
        let position1_layers = layers0123_report().layers;
        let mut position2_layers = position1_layers.clone();
        for (layer_index, layer) in position2_layers.iter_mut().enumerate() {
            layer.fixture_id = POSITION2_FIXTURE_IDS[layer_index];
            layer.token = 361;
        }
        let mut position3_layers = position2_layers.clone();
        for (layer_index, layer) in position3_layers.iter_mut().enumerate() {
            layer.fixture_id = POSITION3_FIXTURE_IDS[layer_index];
            layer.token = 1915;
        }
        Layers0123DecodeProbeReport {
            steps: vec![
                Layers0123DecodeStepReport {
                    position: 1,
                    token: 201,
                    cache_rows: 2,
                    layers: position1_layers,
                    wall_ms: 5.0,
                    gpu_ms: 4.0,
                    output_hc_checksum: 10,
                },
                Layers0123DecodeStepReport {
                    position: 2,
                    token: 361,
                    cache_rows: 3,
                    layers: position2_layers,
                    wall_ms: 6.0,
                    gpu_ms: 5.0,
                    output_hc_checksum: 11,
                },
                Layers0123DecodeStepReport {
                    position: 3,
                    token: 1915,
                    cache_rows: 4,
                    layers: position3_layers,
                    wall_ms: 7.0,
                    gpu_ms: 6.0,
                    output_hc_checksum: 12,
                },
            ],
            command_buffers_per_step: 4,
            host_waits_per_step: 1,
            kv_cache_layers: 4,
            cache_capacity_rows: 4,
            output_hc_elements: 4 * 4096,
        }
    }

    fn layers012345_decode_report() -> Layers012345DecodeProbeReport {
        let mut report = layers0123_decode_report();
        for step in &mut report.steps {
            let mut layer4 = layer0_report();
            layer4.fixture_id = match step.position {
                1 => LAYER4_FIXTURE_ID,
                2 => LAYER4_POS2_FIXTURE_ID,
                _ => LAYER4_POS3_FIXTURE_ID,
            };
            layer4.token = step.token;
            layer4.dispatches = if step.position == 1 {
                36
            } else if step.position == 3 {
                48
            } else {
                32
            };
            layer4.selected_experts = match step.position {
                1 => vec![170, 98, 161, 61, 63, 187],
                2 => vec![170, 28, 97, 3, 98, 12],
                _ => vec![242, 170, 184, 182, 97, 86],
            };
            let mut layer5 = layer0_report();
            layer5.fixture_id = match step.position {
                1 => LAYER5_FIXTURE_ID,
                2 => LAYER5_POS2_FIXTURE_ID,
                _ => LAYER5_POS3_FIXTURE_ID,
            };
            layer5.token = step.token;
            layer5.dispatches = if step.position == 1 { 32 } else { 30 };
            layer5.selected_experts = match step.position {
                1 => vec![70, 210, 35, 48, 118, 72],
                2 => vec![210, 70, 225, 255, 206, 10],
                _ => vec![70, 240, 210, 110, 194, 77],
            };
            layer5.final_hc_checksum = 20 + u64::from(step.position);
            step.output_hc_checksum = layer5.final_hc_checksum;
            step.layers.extend([layer4, layer5]);
        }
        report.command_buffers_per_step = 6;
        report.kv_cache_layers = 6;
        report
    }

    fn layers01234567_decode_report() -> Layers01234567DecodeProbeReport {
        let mut report = layers012345_decode_report();
        for step in &mut report.steps {
            let mut layer6 = layer0_report();
            layer6.fixture_id = match step.position {
                1 => LAYER6_FIXTURE_ID,
                2 => LAYER6_POS2_FIXTURE_ID,
                _ => LAYER6_POS3_FIXTURE_ID,
            };
            layer6.token = step.token;
            layer6.dispatches = if step.position == 1 {
                36
            } else if step.position == 3 {
                48
            } else {
                32
            };
            layer6.selected_experts = match step.position {
                1 => vec![82, 171, 1, 57, 16, 108],
                2 => vec![82, 202, 38, 74, 12, 182],
                _ => vec![74, 182, 108, 83, 146, 166],
            };
            let mut layer7 = layer0_report();
            layer7.fixture_id = match step.position {
                1 => LAYER7_FIXTURE_ID,
                2 => LAYER7_POS2_FIXTURE_ID,
                _ => LAYER7_POS3_FIXTURE_ID,
            };
            layer7.token = step.token;
            layer7.dispatches = if step.position == 1 { 32 } else { 30 };
            layer7.selected_experts = match step.position {
                1 => vec![29, 122, 5, 241, 162, 62],
                2 => vec![122, 29, 66, 120, 213, 223],
                _ => vec![5, 122, 71, 162, 93, 241],
            };
            layer7.final_hc_checksum = 30 + u64::from(step.position);
            step.output_hc_checksum = layer7.final_hc_checksum;
            step.layers.extend([layer6, layer7]);
        }
        report.command_buffers_per_step = 8;
        report.kv_cache_layers = 8;
        report
    }

    fn layers0_to_42_decode_report() -> Layers0To42DecodeProbeReport {
        let mut report = layers01234567_decode_report();
        for step in &mut report.steps {
            for layer_index in 8..=42 {
                let registry_index = later_fixture_index(layer_index).unwrap();
                let mut layer = layer0_report();
                layer.fixture_id = match step.position {
                    1 => LATER_POS1_BYTES[registry_index].fixture_id,
                    2 => LATER_POS2_BYTES[registry_index].fixture_id,
                    _ => LATER_POS3_BYTES[registry_index].fixture_id,
                };
                layer.token = step.token;
                layer.final_hc_checksum = 100 + u64::from(layer_index * 3 + step.position);
                step.output_hc_checksum = layer.final_hc_checksum;
                step.layers.push(layer);
            }
        }
        report.command_buffers_per_step = 43;
        report.kv_cache_layers = 43;
        report
    }

    fn decoder_output_report() -> DecoderOutputProbeReport {
        let transformer = layers0_to_42_decode_report();
        let selected = [361_u32, 1915, 262];
        let fixtures = [
            "dwarfstar-oracle-v1-output-head-pos1",
            "dwarfstar-oracle-v1-output-head-pos2",
            "dwarfstar-oracle-v1-output-head-pos3",
        ];
        DecoderOutputProbeReport {
            steps: transformer
                .steps
                .into_iter()
                .enumerate()
                .map(|(index, step)| DecoderOutputStepReport {
                    position: step.position,
                    input_token: step.token,
                    cache_rows: step.cache_rows,
                    layers: step.layers,
                    transformer_wall_ms: step.wall_ms,
                    transformer_gpu_ms: step.gpu_ms,
                    output_head: OutputHeadProbeReport {
                        fixture_id: fixtures[index],
                        dispatches: 5,
                        command_buffers: 1,
                        host_waits: 1,
                        wrapped_model_ranges: 5,
                        pointer_matches: 5,
                        wall_ms: 2.0,
                        gpu_ms: 1.0,
                        hc_pre_checksum: 1,
                        hc_weights_checksum: 2,
                        hc_checksum: 3,
                        norm_checksum: 4,
                        logits_checksum: 5,
                        selected_token: selected[index],
                    },
                })
                .collect(),
            command_buffers_per_step: 44,
            host_waits_per_step: 2,
            kv_cache_layers: 43,
            cache_capacity_rows: 4,
            logits_elements: 129280,
        }
    }

    #[test]
    fn validates_probe_work_bounds() {
        assert!(ProbeConfig::default().validate().is_ok());
        assert!(ProbeConfig {
            elements: 0,
            iterations: 1,
        }
        .validate()
        .is_err());
        assert!(ProbeConfig {
            elements: MAX_ELEMENTS,
            iterations: MAX_ITERATIONS,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn validates_layer0_bench_bounds() {
        assert!(Layer0BenchConfig::default().validate().is_ok());
        assert!(Layer0BenchConfig {
            warmup_iterations: 0,
            iterations: 0,
        }
        .validate()
        .is_err());
        assert!(Layer0BenchConfig {
            warmup_iterations: 1,
            iterations: MAX_LAYER0_EXECUTIONS,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn validates_layers0123_bench_bounds() {
        assert!(Layers0123BenchConfig::default().validate().is_ok());
        assert!(Layers0123BenchConfig {
            warmup_iterations: 0,
            iterations: 0,
        }
        .validate()
        .is_err());
        assert!(Layers0123BenchConfig {
            warmup_iterations: 1,
            iterations: MAX_LAYER0_EXECUTIONS,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn writes_stable_probe_json() {
        let mut output = Vec::new();
        write_probe_json(&mut output, &report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{PROBE_SCHEMA}\"")));
        assert!(text.contains("Apple GPU \\\"fixture\\\""));
        assert!(text.contains("\"dispatches_per_second\": 2000"));
        assert!(text.contains("\"checksum\": 42"));
    }

    #[test]
    fn writes_stable_layers01_probe_json() {
        let mut output = Vec::new();
        write_layers01_probe_json(&mut output, &layers01_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYERS01_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"retained_hc_handoff\": true"));
        assert!(text.contains("\"kv_cache_layers\": 2"));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER1_FIXTURE_ID}\"")));
        assert!(text.contains("\"dispatches\": 28"));
    }

    #[test]
    fn writes_stable_layers012_probe_json() {
        let mut output = Vec::new();
        write_layers012_probe_json(&mut output, &layers012_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYERS012_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"kv_cache_layers\": 3"));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER2_FIXTURE_ID}\"")));
        assert!(text.contains("\"selected_experts\": [8, 188, 195, 75, 96, 176]"));
    }

    #[test]
    fn writes_stable_layers012_chained_probe_json() {
        let mut output = Vec::new();
        write_layers012_chained_probe_json(&mut output, &layers012_chained_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYERS012_CHAINED_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"host_waits\": 1"));
        assert!(text.contains("\"chain_wall_ms\": 4.000000"));
        assert!(text.contains("\"summed_command_gpu_ms\": 3.000000"));
    }

    #[test]
    fn writes_stable_layers0123_probe_json() {
        let mut output = Vec::new();
        write_layers0123_probe_json(&mut output, &layers0123_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYERS0123_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"kv_cache_layers\": 4"));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER3_FIXTURE_ID}\"")));
        assert!(text.contains("\"selected_experts\": [1, 58, 68, 240, 20, 24]"));
    }

    #[test]
    fn writes_stable_layers0123_chained_probe_json() {
        let mut output = Vec::new();
        write_layers0123_chained_probe_json(&mut output, &layers0123_chained_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{LAYERS0123_CHAINED_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"command_buffers\": 4"));
        assert!(text.contains("\"chain_wall_ms\": 5.000000"));
        assert!(text.contains("\"summed_command_gpu_ms\": 4.000000"));
    }

    #[test]
    fn writes_stable_layers0123_bench_json() {
        let mut output = Vec::new();
        write_layers0123_bench_json(&mut output, &layers0123_bench_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYERS0123_BENCH_SCHEMA}\"")));
        assert!(text.contains("\"correctness_readback\": \"after-final-measured-iteration\""));
        assert!(text.contains("\"samples\": [4.000000, 3.000000, 5.000000]"));
        assert!(text.contains("\"final_c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_layers0123_decode_probe_json() {
        let mut output = Vec::new();
        write_layers0123_decode_probe_json(&mut output, &layers0123_decode_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYERS0123_DECODE_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"position\": 2"));
        assert!(text.contains("\"token\": 361"));
        assert!(text.contains("\"cache_rows\": 3"));
        assert!(text.contains("\"position\": 3"));
        assert!(text.contains("\"token\": 1915"));
        assert!(text.contains("\"cache_rows\": 4"));
        assert!(text.contains(&format!("\"fixture\": \"{}\"", POSITION3_FIXTURE_IDS[2])));
        assert!(text.contains(&format!("\"fixture\": \"{}\"", POSITION2_FIXTURE_IDS[3])));
        assert!(text.contains("\"cache_growth_exact\": true"));
        assert!(text.contains("\"output_handoff_exact\": true"));
    }

    #[test]
    fn writes_stable_layers012345_decode_probe_json() {
        let mut output = Vec::new();
        write_layers012345_decode_probe_json(&mut output, &layers012345_decode_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{LAYERS012345_DECODE_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"command_buffers_per_step\": 6"));
        assert!(text.contains("\"kv_cache_layers\": 6"));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER4_POS3_FIXTURE_ID}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER5_POS3_FIXTURE_ID}\"")));
    }

    #[test]
    fn writes_stable_layers01234567_decode_probe_json() {
        let mut output = Vec::new();
        write_layers01234567_decode_probe_json(&mut output, &layers01234567_decode_report())
            .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{LAYERS01234567_DECODE_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"command_buffers_per_step\": 8"));
        assert!(text.contains("\"kv_cache_layers\": 8"));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER6_POS3_FIXTURE_ID}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{LAYER7_POS3_FIXTURE_ID}\"")));
    }

    #[test]
    fn writes_stable_layers0_to_42_decode_probe_json() {
        let mut output = Vec::new();
        write_layers0_to_42_decode_probe_json(&mut output, &layers0_to_42_decode_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{LAYERS0_TO_42_DECODE_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"command_buffers_per_step\": 43"));
        assert!(text.contains("\"kv_cache_layers\": 43"));
        assert!(text.contains("\"layer\": 42"));
        assert!(text.contains("\"fixture\": \"dwarfstar-oracle-v1-layer42-pos3-complete\""));
    }

    #[test]
    fn output_head_fixtures_have_target_shapes_and_selection() {
        for (position, selected) in [(1, 361), (2, 1915), (3, 262)] {
            let fixture = output_head_expected(position).unwrap();
            assert_eq!(fixture.hc_pre.len(), 4);
            assert_eq!(fixture.hc_weights.len(), 4);
            assert_eq!(fixture.hc.len(), 4096);
            assert_eq!(fixture.norm.len(), 4096);
            assert_eq!(fixture.logits.len(), 129280);
            assert_eq!(fixture.selected_token, selected);
            assert_eq!(lowest_id_argmax(&fixture.logits).unwrap(), selected);
        }
    }

    #[test]
    fn writes_stable_decoder_output_probe_json() {
        let mut output = Vec::new();
        write_decoder_output_probe_json(&mut output, &decoder_output_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{DECODER_OUTPUT_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"command_buffers_per_step\": 44"));
        assert!(text.contains("\"host_waits_per_step\": 2"));
        assert!(text.contains("\"selected_token\": 262"));
        assert!(text.contains("\"closed_loop_sampling\": false"));
        assert!(text.contains("\"full_logits_c0_bitwise_match\": true"));
    }

    #[test]
    fn complete_later_layer_registry_has_target_shapes() {
        assert_eq!(LATER_POS1_BYTES.len(), 39);
        assert_eq!(LATER_POS2_BYTES.len(), 39);
        assert_eq!(LATER_POS3_BYTES.len(), 39);
        assert_eq!(LATER_POS0_COMPRESSOR_PRIME_BYTES.len(), 39);
        assert_eq!(LATER_CACHE_ROW0_BYTES.len(), 39);
        assert_eq!(LATER_POS3_COMPRESSED_KV_BYTES.len(), 20);
        for layer_index in 4..=42 {
            assert_eq!(compressor_prime_bytes(layer_index).unwrap().len(), 4096 * 4);
            for position in 1..=3 {
                let fixture = layer_expected(layer_index, position).unwrap();
                assert_eq!(
                    fixture.fixture_id,
                    format!("dwarfstar-oracle-v1-layer{layer_index}-pos{position}-complete")
                );
                assert_eq!(fixture.cache_row0.len(), 512);
                assert_eq!(fixture.kv_cur.len(), 512);
                assert_eq!(fixture.attention_hc.len(), 4 * 4096);
                assert_eq!(fixture.selected.len(), 6);
                assert_eq!(fixture.final_hc.len(), 4 * 4096);
                assert_eq!(
                    fixture.compressed_kv.len(),
                    if position == 3 && layer_index % 2 == 0 {
                        512
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn position2_complete_fixtures_have_target_shapes() {
        let selected = [
            vec![191, 152, 163, 99, 109, 156],
            vec![46, 252, 230, 133, 42, 183],
            vec![29, 164, 162, 130, 7, 179],
            vec![211, 122, 64, 68, 1, 24],
        ];
        for layer_index in 0..=3 {
            let fixture = layer_expected(layer_index, 2).unwrap();
            assert_eq!(
                fixture.fixture_id,
                POSITION2_FIXTURE_IDS[layer_index as usize]
            );
            assert_eq!(fixture.cache_row0.len(), 512);
            assert_eq!(fixture.kv_cur.len(), 512);
            assert_eq!(fixture.attention_hc.len(), 4 * 4096);
            assert_eq!(fixture.selected, selected[layer_index as usize]);
            assert_eq!(fixture.final_hc.len(), 4 * 4096);
        }
    }

    #[test]
    fn position3_complete_fixtures_have_target_shapes() {
        let selected = [
            vec![133, 217, 222, 94, 234, 246],
            vec![107, 58, 141, 226, 233, 88],
            vec![90, 98, 196, 23, 62, 19],
            vec![64, 87, 198, 214, 128, 1],
        ];
        for layer_index in 0..=3 {
            let fixture = layer_expected(layer_index, 3).unwrap();
            assert_eq!(
                fixture.fixture_id,
                POSITION3_FIXTURE_IDS[layer_index as usize]
            );
            assert_eq!(fixture.kv_cur.len(), 512);
            assert_eq!(fixture.selected, selected[layer_index as usize]);
            assert_eq!(fixture.final_hc.len(), 4 * 4096);
            assert_eq!(
                fixture.compressed_kv.len(),
                if layer_index == 2 { 512 } else { 0 }
            );
        }
    }

    #[test]
    fn layer45_position_advancing_fixtures_have_target_shapes() {
        let selected = [
            [
                vec![170, 98, 161, 61, 63, 187],
                vec![170, 28, 97, 3, 98, 12],
                vec![242, 170, 184, 182, 97, 86],
            ],
            [
                vec![70, 210, 35, 48, 118, 72],
                vec![210, 70, 225, 255, 206, 10],
                vec![70, 240, 210, 110, 194, 77],
            ],
        ];
        for layer_index in 4..=5 {
            for position in 1..=3 {
                let fixture = layer_expected(layer_index, position).unwrap();
                assert_eq!(fixture.cache_row0.len(), 512);
                assert_eq!(fixture.kv_cur.len(), 512);
                assert_eq!(fixture.attention_hc.len(), 4 * 4096);
                assert_eq!(
                    fixture.selected,
                    selected[(layer_index - 4) as usize][(position - 1) as usize]
                );
                assert_eq!(fixture.final_hc.len(), 4 * 4096);
                assert_eq!(
                    fixture.compressed_kv.len(),
                    if layer_index == 4 && position == 3 {
                        512
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn layer67_position_advancing_fixtures_have_target_shapes() {
        let selected = [
            [
                vec![82, 171, 1, 57, 16, 108],
                vec![82, 202, 38, 74, 12, 182],
                vec![74, 182, 108, 83, 146, 166],
            ],
            [
                vec![29, 122, 5, 241, 162, 62],
                vec![122, 29, 66, 120, 213, 223],
                vec![5, 122, 71, 162, 93, 241],
            ],
        ];
        for layer_index in 6..=7 {
            for position in 1..=3 {
                let fixture = layer_expected(layer_index, position).unwrap();
                assert_eq!(fixture.cache_row0.len(), 512);
                assert_eq!(fixture.kv_cur.len(), 512);
                assert_eq!(fixture.attention_hc.len(), 4 * 4096);
                assert_eq!(
                    fixture.selected,
                    selected[(layer_index - 6) as usize][(position - 1) as usize]
                );
                assert_eq!(fixture.final_hc.len(), 4 * 4096);
                assert_eq!(
                    fixture.compressed_kv.len(),
                    if layer_index == 6 && position == 3 {
                        512
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn layer1_complete_fixture_has_target_shapes() {
        let fixture = layer_expected(1, 1).unwrap();
        assert_eq!(fixture.cache_row0.len(), 512);
        assert_eq!(fixture.attention_hc.len(), 4 * 4096);
        assert_eq!(fixture.ffn_mixes.len(), 24);
        assert_eq!(fixture.ffn_split.len(), 24);
        assert_eq!(fixture.router_logits.len(), 256);
        assert_eq!(fixture.selected, vec![228, 208, 35, 27, 113, 12]);
        assert_eq!(fixture.routed_mid.len(), 6 * 2048);
        assert_eq!(fixture.final_hc.len(), 4 * 4096);
    }

    #[test]
    fn layer2_complete_fixture_has_target_shapes() {
        let fixture = layer_expected(2, 1).unwrap();
        assert_eq!(fixture.cache_row0.len(), 512);
        assert_eq!(fixture.attention_hc.len(), 4 * 4096);
        assert_eq!(fixture.ffn_mixes.len(), 24);
        assert_eq!(fixture.ffn_split.len(), 24);
        assert_eq!(fixture.router_logits.len(), 256);
        assert_eq!(fixture.selected, vec![8, 188, 195, 75, 96, 176]);
        assert_eq!(fixture.routed_mid.len(), 6 * 2048);
        assert_eq!(fixture.final_hc.len(), 4 * 4096);
    }

    #[test]
    fn layer3_complete_fixture_has_target_shapes() {
        let fixture = layer_expected(3, 1).unwrap();
        assert_eq!(fixture.cache_row0.len(), 512);
        assert_eq!(fixture.attention_hc.len(), 4 * 4096);
        assert_eq!(fixture.ffn_mixes.len(), 24);
        assert_eq!(fixture.ffn_split.len(), 24);
        assert_eq!(fixture.router_logits.len(), 256);
        assert_eq!(fixture.selected, vec![1, 58, 68, 240, 20, 24]);
        assert_eq!(fixture.routed_mid.len(), 6 * 2048);
        assert_eq!(fixture.final_hc.len(), 4 * 4096);
    }

    #[test]
    fn writes_stable_embedding_probe_json() {
        let mut output = Vec::new();
        write_embedding_probe_json(&mut output, &embedding_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{EMBEDDING_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"kernel\": \"kernel_get_rows_f16\""));
        assert!(text.contains("\"tokens\": [0, 7]"));
        assert!(text.contains("\"no_copy_pointer_match\": true"));
        assert!(text.contains("\"c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_projection_probe_json() {
        let mut output = Vec::new();
        write_projection_probe_json(&mut output, &projection_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{PROJECTION_PROBE_SCHEMA}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{PROJECTION_FIXTURE_ID}\"")));
        assert!(text.contains("\"kernel\": \"kernel_mul_mv_q8_0_f32\""));
        assert!(text.contains("\"simdgroups\": 4"));
        assert!(text.contains("\"c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_ratio128_compressor_replay_json() {
        let mut output = Vec::new();
        write_ratio128_compressor_replay_probe_json(&mut output, &ratio128_compressor_report())
            .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{RATIO128_COMPRESSOR_REPLAY_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"final_position\": 127"));
        assert!(text.contains("\"layer\": 3"));
        assert!(text.contains("\"layer\": 5"));
        assert!(text.contains("\"sampling_performed\": false"));
        assert!(text.contains("\"full_decoder_claim\": false"));
    }

    #[test]
    fn writes_stable_ingress_probe_json() {
        let mut output = Vec::new();
        write_ingress_probe_json(&mut output, &ingress_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{INGRESS_PROBE_SCHEMA}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{INGRESS_FIXTURE_ID}\"")));
        assert!(text.contains("\"pointer_matches\": 6"));
        assert!(text.contains("\"c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_attention_setup_probe_json() {
        let mut output = Vec::new();
        write_attention_setup_probe_json(&mut output, &attention_setup_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{ATTENTION_SETUP_PROBE_SCHEMA}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{ATTENTION_SETUP_FIXTURE_ID}\"")));
        assert!(text.contains("\"dispatches\": 9"));
        assert!(text.contains("\"pointer_matches\": 10"));
        assert!(text.contains("\"c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_rope_kv_store_probe_json() {
        let mut output = Vec::new();
        write_rope_kv_store_probe_json(&mut output, &rope_kv_store_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{ROPE_KV_STORE_PROBE_SCHEMA}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{ROPE_KV_STORE_FIXTURE_ID}\"")));
        assert!(text.contains("\"dispatches\": 12"));
        assert!(text.contains("\"guard_rows_intact\": true"));
        assert!(text.contains("\"c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_attention_read_probe_json() {
        let mut output = Vec::new();
        write_attention_read_probe_json(&mut output, &attention_read_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{ATTENTION_READ_PROBE_SCHEMA}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{ATTENTION_READ_FIXTURE_ID}\"")));
        assert!(text.contains("\"rows_read\": 2"));
        assert!(text.contains("\"kqv_back\": 6"));
    }

    #[test]
    fn writes_stable_attention_output_probe_json() {
        let mut output = Vec::new();
        write_attention_output_probe_json(&mut output, &attention_output_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{ATTENTION_OUTPUT_PROBE_SCHEMA}\"")));
        assert!(text.contains(&format!("\"fixture\": \"{ATTENTION_OUTPUT_FIXTURE_ID}\"")));
        assert!(text.contains("\"groups\": 8"));
        assert!(text.contains("\"hc_attn_post\": 9"));
    }

    #[test]
    fn writes_stable_ffn_router_probe_json() {
        let mut output = Vec::new();
        write_ffn_router_probe_json(&mut output, &ffn_router_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{FFN_ROUTER_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"dispatches\": 7"));
        assert!(text.contains("\"selected_experts\": [17, 42, 63, 99, 101, 201]"));
        assert!(text.contains("\"c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_moe_output_probe_json() {
        let mut output = Vec::new();
        write_moe_output_probe_json(&mut output, &moe_output_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{MOE_OUTPUT_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"dispatches\": 4"));
        assert!(text.contains("\"selected_experts\": [25, 174, 215, 58, 48, 60]"));
        assert!(text.contains("\"hc_ffn_post\": 4"));
    }

    #[test]
    fn writes_stable_layer0_probe_json() {
        let mut output = Vec::new();
        write_layer0_probe_json(&mut output, &layer0_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYER0_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"dispatches\": 30"));
        assert!(text.contains("\"command_buffers\": 1"));
        assert!(text.contains("\"pointer_matches\": 25"));
        assert!(text.contains("\"hc_ffn_post\": 7"));
    }

    #[test]
    fn writes_stable_layer0_bench_json() {
        let mut output = Vec::new();
        write_layer0_bench_json(&mut output, &layer0_bench_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{LAYER0_BENCH_SCHEMA}\"")));
        assert!(text.contains("\"warmup_iterations\": 2"));
        assert!(text.contains("\"samples\": [1.000000, 2.000000, 4.000000]"));
        assert!(text.contains("\"median\": 0.750000"));
        assert!(text.contains("\"repeat_bitwise_match\": true"));
    }

    #[test]
    fn rope_kv_store_fixture_has_target_shapes() {
        let (q_cur, kv_rope, kv_cur, cache_row) = rope_kv_store_fixture().unwrap();
        assert_eq!(q_cur.len(), 64 * 512);
        assert_eq!(kv_rope.len(), 512);
        assert_eq!(kv_cur.len(), 512);
        assert_eq!(cache_row.len(), 512);
        assert!(q_cur
            .iter()
            .chain(&kv_rope)
            .chain(&kv_cur)
            .chain(&cache_row)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn attention_read_fixture_has_target_shapes() {
        let (cache_row0, cache_row1, attention_back) = attention_read_fixture().unwrap();
        assert_eq!(cache_row0.len(), 512);
        assert_eq!(cache_row1.len(), 512);
        assert_eq!(attention_back.len(), 64 * 512);
    }

    #[test]
    fn attention_output_fixture_has_target_shapes() {
        let (back, low, out, hc_post) = attention_output_fixture().unwrap();
        assert_eq!(back.len(), 64 * 512);
        assert_eq!(low.len(), 8 * 1024);
        assert_eq!(out.len(), 4096);
        assert_eq!(hc_post.len(), 4 * 4096);
    }

    #[test]
    fn ffn_router_fixture_has_target_shapes() {
        let (input, mixes, split, cur, norm, logits, probs, selected, weights) =
            ffn_router_fixture().unwrap();
        assert_eq!(input.len(), 4 * 4096);
        assert_eq!(mixes.len(), 24);
        assert_eq!(split.len(), 24);
        assert_eq!(cur.len(), 4096);
        assert_eq!(norm.len(), 4096);
        assert_eq!(logits.len(), 256);
        assert_eq!(probs.len(), 256);
        assert_eq!(selected.len(), 6);
        assert_eq!(weights.len(), 6);
    }

    #[test]
    fn moe_output_fixture_has_target_shapes() {
        let (norm, selected, weights, input_hc, split, routed_mid, routed_out, shared_out, hc_post) =
            moe_output_fixture().unwrap();
        assert_eq!(norm.len(), 4096);
        assert_eq!(selected, [25, 174, 215, 58, 48, 60]);
        assert_eq!(weights.len(), 6);
        assert_eq!(input_hc.len(), 4 * 4096);
        assert_eq!(split.len(), 24);
        assert_eq!(routed_mid.len(), 6 * 2048);
        assert_eq!(routed_out.len(), 4096);
        assert_eq!(shared_out.len(), 4096);
        assert_eq!(hc_post.len(), 4 * 4096);
    }

    #[test]
    fn projection_fixture_is_finite_and_has_target_shape() {
        let (input, output) = projection_fixture().unwrap();
        assert_eq!(input.len(), 4096);
        assert_eq!(output.len(), 1024);
        assert!(input.iter().all(|value| value.is_finite()));
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(checksum_f32(&input), 6_001_855_774_483_604_828);
        assert_eq!(checksum_f32(&output), 13_770_952_831_385_691_371);
    }

    #[test]
    fn ratio128_compressor_fixtures_have_target_shapes() {
        for layer in [3_u32, 5_u32] {
            let (fixture_id, inputs, output) = ratio128_compressor_fixture(layer).unwrap();
            assert_eq!(
                fixture_id,
                if layer == 3 {
                    LAYER3_POS127_COMPRESSOR_FIXTURE_ID
                } else {
                    LAYER5_POS127_COMPRESSOR_FIXTURE_ID
                }
            );
            assert_eq!(inputs.len(), 128 * 4096);
            assert_eq!(output.len(), 512);
            assert!(inputs.iter().chain(&output).all(|value| value.is_finite()));
        }
    }

    #[test]
    fn converts_f16_reference_values_exactly() {
        for (half, single) in [
            (0x0000, 0x0000_0000),
            (0x8000, 0x8000_0000),
            (0x3c00, 0x3f80_0000),
            (0xc000, 0xc000_0000),
            (0x7bff, 0x477f_e000),
            (0x0400, 0x3880_0000),
            (0x0001, 0x3380_0000),
        ] {
            assert_eq!(f16_to_f32(half).to_bits(), single, "half={half:#06x}");
        }
    }

    #[test]
    fn rounds_raw_cache_values_through_f16_exactly() {
        for (single, half, rounded) in [
            (0x0000_0000, 0x0000, 0x0000_0000),
            (0x8000_0000, 0x8000, 0x8000_0000),
            (0x3f80_0000, 0x3c00, 0x3f80_0000),
            (0xbe0f_ffff, 0xb080, 0xbe10_0000),
            (0x3380_0000, 0x0001, 0x3380_0000),
        ] {
            let value = f32::from_bits(single);
            assert_eq!(f32_to_f16(value), half, "single={single:#010x}");
            assert_eq!(f16_round_f32(value).to_bits(), rounded);
        }
    }

    #[test]
    fn embedding_checksum_preserves_float_bits() {
        assert_ne!(checksum_f32(&[0.0]), checksum_f32(&[-0.0]));
        assert_eq!(checksum_f32(&[1.0, 2.0]), checksum_f32(&[1.0, 2.0]));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_probe_is_explicitly_unsupported() {
        let error = run_probe(ProbeConfig::default()).unwrap_err().to_string();
        assert!(error.contains("only on macOS"), "{error}");
    }
}
