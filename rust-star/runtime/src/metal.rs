//! Minimal Metal ownership and command-dispatch probe.

use crate::gguf::TensorInfo;
use crate::model::MappedModel;
use crate::{Error, Result};
use std::io::Write;

pub const PROBE_SCHEMA: &str = "rust-star-metal-dispatch-probe-v1";
pub const EMBEDDING_PROBE_SCHEMA: &str = "rust-star-f16-embedding-probe-v1";
pub const PROJECTION_PROBE_SCHEMA: &str = "rust-star-q8-0-projection-probe-v1";
pub const PREFILL_Q8_BOUNDARY_PROBE_SCHEMA: &str = "rust-star-prefill-q8-boundary-probe-v1";
pub const PREFILL_QKV_BOUNDARY_PROBE_SCHEMA: &str = "rust-star-prefill-qkv-boundary-probe-v1";
pub const PREFILL_LAYER0_BOUNDARY_PROBE_SCHEMA: &str = "rust-star-prefill-layer0-boundary-probe-v5";
pub const PREFILL_LAYERS01_BOUNDARY_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers01-boundary-probe-v1";
pub const PREFILL_LAYERS01_COMPLETE_BOUNDARY_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers01-complete-boundary-probe-v1";
pub const PREFILL_LAYERS01_ROW_COVERAGE_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers01-row-coverage-probe-v1";
pub const PREFILL_LAYERS01_LIVE_KV_CHAIN_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers01-live-kv-chain-probe-v1";
pub const PREFILL_LAYERS01_LIVE_KV_LOOP_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers01-live-kv-loop-probe-v1";
pub const PREFILL_LAYERS012_KVNORM_LOOP_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers012-kvnorm-loop-probe-v1";
pub const PREFILL_LAYERS012_KV_STATE_LOOP_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers012-kv-state-loop-probe-v1";
pub const PREFILL_LAYERS012_COMPRESSOR_LOOP_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers012-compressor-loop-probe-v1";
pub const PREFILL_LAYERS012_ATTENTION_LOOP_PROBE_SCHEMA: &str =
    "rust-star-prefill-layers012-attention-loop-probe-v1";
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
    "rust-star-decoder-output-position-advancing-probe-v2";
pub const CLOSED_LOOP_DECODER_PROBE_SCHEMA: &str = "rust-star-closed-loop-decoder-diagnostic-v2";
pub const POSITION127_DECODER_PROBE_SCHEMA: &str =
    "rust-star-position127-decoder-frontier-diagnostic-v1";
pub const COLD_PREFILL_DECODER_PROBE_SCHEMA: &str = "rust-star-cold-prefill-decoder-diagnostic-v1";
pub const PREFILL_FRONTIER_PROBE_SCHEMA: &str = "rust-star-prefill-frontier-diagnostic-v1";
pub const RATIO128_COMPRESSOR_REPLAY_PROBE_SCHEMA: &str =
    "rust-star-ratio128-compressor-replay-probe-v1";
pub const SPARSE_INDEXED_ATTENTION_PROBE_SCHEMA: &str =
    "rust-star-sparse-indexed-attention-boundary-v2";
pub const RETAINED_SPARSE_BOUNDARY_PROBE_SCHEMA: &str = "rust-star-retained-sparse-boundary-v1";
pub const RETAINED_SPARSE_MULTIMERGE_PROBE_SCHEMA: &str = "rust-star-retained-sparse-multimerge-v1";
pub const SPARSE_INDEXED_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-layer2-pos2051-sparse-indexed-attention";
pub const SPARSE_INDEXED_ATTENTION_DEFAULT_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-layer2-pos4099-sparse-indexed-attention";
pub const RETAINED_SPARSE_BOUNDARY_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-retained-layer2-pos4099-sparse";
pub const RETAINED_SPARSE_MULTIMERGE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-retained-layer2-pos8195-sparse-multimerge";
pub const PROJECTION_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attn-q-a";
pub const PREFILL_Q8_BOUNDARY_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-q8-boundary-2048";
pub const PREFILL_QKV_BOUNDARY_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-qkv-boundary-2048";
pub const PREFILL_KV_STATE_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-kv-state-2048";
pub const PREFILL_ATTENTION_READ_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-attention-read-2048";
pub const PREFILL_ATTENTION_OUTPUT_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-attention-output-2048";
pub const PREFILL_FFN_OUTPUT_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-ffn-output-2048";
pub const PREFILL_LAYER1_INGRESS_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer1-ingress-2048";
pub const PREFILL_LAYER1_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer1-complete-2048";
pub const PREFILL_LAYER2_KVNORM_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-layer2-kvnorm-2048";
pub const PREFILL_LAYER2_KV_STATE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer2-kv-state-2048";
pub const PREFILL_LAYER2_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer2-compressors-2048";
pub const PREFILL_LAYER2_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer2-attention-2048";
pub const PREFILL_LAYER2_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer2-complete-2048";
pub const PREFILL_LAYER3_INGRESS_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer3-ingress-2048";
pub const PREFILL_LAYER3_KV_STATE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer3-kv-state-2048";
pub const PREFILL_LAYER3_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer3-compressor-2048";
pub const PREFILL_LAYER3_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer3-attention-2048";
pub const PREFILL_LAYER3_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer3-complete-2048";
pub const PREFILL_LAYER4_QKV_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-layer4-qkv-2048";
pub const PREFILL_LAYER4_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer4-compressors-2048";
pub const PREFILL_LAYER4_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer4-attention-2048";
pub const PREFILL_LAYER4_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer4-complete-2048";
pub const PREFILL_LAYER5_QKV_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-layer5-qkv-2048";
pub const PREFILL_LAYER5_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer5-compressor-2048";
pub const PREFILL_LAYER5_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer5-attention-2048";
pub const PREFILL_LAYER5_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer5-complete-2048";
pub const PREFILL_LAYER6_QKV_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-layer6-qkv-2048";
pub const PREFILL_LAYER6_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer6-compressors-2048";
pub const PREFILL_LAYER6_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer6-attention-2048";
pub const PREFILL_LAYER6_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer6-complete-2048";
pub const PREFILL_LAYER7_QKV_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-layer7-qkv-2048";
pub const PREFILL_LAYER7_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer7-compressor-2048";
pub const PREFILL_LAYER7_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer7-attention-2048";
pub const PREFILL_LAYER7_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer7-complete-2048";
pub const PREFILL_LAYER8_QKV_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-layer8-qkv-2048";
pub const PREFILL_LAYER8_COMPRESSOR_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer8-compressor-2048";
pub const PREFILL_LAYER8_ATTENTION_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer8-attention-2048";
pub const PREFILL_LAYER8_COMPLETE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layer8-complete-2048";
pub const PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE_ID: &str =
    "dwarfstar-oracle-v1-prefill-layers01-previous-tile-2048";
pub const PREFILL_HC_INGRESS_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-hc-ingress-2048";
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
pub const POSITION127_DECODER_FIXTURE_ID: &str = "dwarfstar-oracle-v1-decoder-frontier-pos127";
pub const COLD_PREFILL_FIXTURE_ID: &str = "dwarfstar-oracle-v1-cold-prefill-pos0";
pub const PREFILL_FRONTIER_2048_FIXTURE_ID: &str = "dwarfstar-oracle-v1-prefill-frontier-2048";
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
const OUTPUT_HEAD_POS4_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos4-v1/output-hc-pre.f32le.bin");
const OUTPUT_HEAD_POS4_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos4-v1/output-hc-weights.f32le.bin");
const OUTPUT_HEAD_POS4_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos4-v1/output-hc.f32le.bin");
const OUTPUT_HEAD_POS4_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos4-v1/output-norm.f32le.bin");
const OUTPUT_HEAD_POS4_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/output-head-pos4-v1/logits.f32le.bin");
const POSITION127_TOKEN_IDS_BYTES: &[u8] =
    include_bytes!("../../fixtures/decoder-frontier-pos127-v1/token-ids.u32le.bin");
const POSITION127_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/decoder-frontier-pos127-v1/logits.f32le.bin");
const COLD_PREFILL_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/cold-prefill-pos0-v1/logits.f32le.bin");
const PREFILL_FRONTIER_2048_TOKEN_IDS_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-frontier-2048-v1/token-ids.u32le.bin");
const PREFILL_FRONTIER_2048_BATCH_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-frontier-2048-v1/batch-prefill-logits.f32le.bin");
const PREFILL_FRONTIER_2048_DECODE_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-frontier-2048-v1/decode-replay-logits.f32le.bin");
const PROJECTION_TENSOR: &str = "blk.0.attn_q_a.weight";
const PROJECTION_INPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/q8-attn-q-a-v1/activation.f32le.bin");
const PROJECTION_OUTPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/q8-attn-q-a-v1/output.f32le.bin");
const PREFILL_Q8_INPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-q8-boundary-2048-v1/attn-norm-final-tile.f32le.bin");
const SPARSE_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/q-lora-norm.f32le.bin");
const SPARSE_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/attn-norm.f32le.bin");
const SPARSE_Q_CURRENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/q-current.f32le.bin");
const SPARSE_RAW_CACHE_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/raw-cache.f32le.bin");
const SPARSE_ATTN_COMP_CACHE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/sparse-indexed-attention-pos2051-v1/attention-comp-cache.f32le.bin"
);
const SPARSE_INDEX_COMP_CACHE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/sparse-indexed-attention-pos2051-v1/indexer-comp-cache.f32le.bin"
);
const SPARSE_INDEXER_Q_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/indexer-q.f32le.bin");
const SPARSE_INDEXER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/indexer-weights.f32le.bin");
const SPARSE_INDEXER_SCORES_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/indexer-scores.f32le.bin");
const SPARSE_INDEXER_TOPK_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/indexer-topk.i32le.bin");
const SPARSE_KQV_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/kqv-out.f32le.bin");
const SPARSE_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos2051-v1/kqv-back.f32le.bin");
const SPARSE_DEFAULT_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/q-lora-norm.f32le.bin");
const SPARSE_DEFAULT_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/attn-norm.f32le.bin");
const SPARSE_DEFAULT_Q_CURRENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/q-current.f32le.bin");
const SPARSE_DEFAULT_RAW_CACHE_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/raw-cache.f32le.bin");
const SPARSE_DEFAULT_ATTN_COMP_CACHE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/sparse-indexed-attention-pos4099-v1/attention-comp-cache.f32le.bin"
);
const SPARSE_DEFAULT_INDEX_COMP_CACHE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/sparse-indexed-attention-pos4099-v1/indexer-comp-cache.f32le.bin"
);
const SPARSE_DEFAULT_INDEXER_Q_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/indexer-q.f32le.bin");
const SPARSE_DEFAULT_INDEXER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/indexer-weights.f32le.bin");
const SPARSE_DEFAULT_INDEXER_SCORES_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/indexer-scores.f32le.bin");
const SPARSE_DEFAULT_INDEXER_TOPK_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/indexer-topk.i32le.bin");
const SPARSE_DEFAULT_KQV_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/kqv-out.f32le.bin");
const SPARSE_DEFAULT_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/sparse-indexed-attention-pos4099-v1/kqv-back.f32le.bin");
const RETAINED_SPARSE_INPUT_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/retained-input-hc.f32le.bin");
const RETAINED_SPARSE_RAW_PRIOR_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/raw-cache-prior.f32le.bin");
const RETAINED_SPARSE_ATTN_COMP_PRIOR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/attention-compressed-prior.f32le.bin"
);
const RETAINED_SPARSE_INDEX_COMP_PRIOR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-compressed-prior.f32le.bin"
);
const RETAINED_SPARSE_ATTN_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/attention-state-kv-pre.f32le.bin"
);
const RETAINED_SPARSE_ATTN_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/attention-state-score-pre-bits.i32le.bin"
);
const RETAINED_SPARSE_INDEX_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-state-kv-pre.f32le.bin"
);
const RETAINED_SPARSE_INDEX_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-state-score-pre-bits.i32le.bin"
);
const RETAINED_SPARSE_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/q-lora-norm.f32le.bin");
const RETAINED_SPARSE_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/attn-norm.f32le.bin");
const RETAINED_SPARSE_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/q-cur.f32le.bin");
const RETAINED_SPARSE_KV_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/kv-cur.f32le.bin");
const RETAINED_SPARSE_COMPRESSED_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/compressed-kv-row0.f32le.bin");
const RETAINED_SPARSE_COMPRESSED_INDEXER_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos4099-v1/compressed-indexer-row1024.f32le.bin"
);
const RETAINED_SPARSE_INDEXER_Q_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-q.f32le.bin");
const RETAINED_SPARSE_INDEXER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-weights.f32le.bin");
const RETAINED_SPARSE_INDEXER_SCORES_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-scores.f32le.bin");
const RETAINED_SPARSE_INDEXER_TOPK_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/indexer-topk.i32le.bin");
const RETAINED_SPARSE_KQV_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/kqv-out.f32le.bin");
const RETAINED_SPARSE_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/kqv-back.f32le.bin");
const RETAINED_SPARSE_ATTN_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/attn-low.f32le.bin");
const RETAINED_SPARSE_ATTN_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/attn-out.f32le.bin");
const RETAINED_SPARSE_HC_ATTN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos4099-v1/hc-attn-post.f32le.bin");
const RETAINED_MULTIMERGE_INPUT_HC_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/retained-input-hc.f32le.bin");
const RETAINED_MULTIMERGE_RAW_PRIOR_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/raw-cache-prior.f32le.bin");
const RETAINED_MULTIMERGE_ATTN_COMP_PRIOR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/attention-compressed-prior.f32le.bin"
);
const RETAINED_MULTIMERGE_INDEX_COMP_PRIOR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-compressed-prior.f32le.bin"
);
const RETAINED_MULTIMERGE_ATTN_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/attention-state-kv-pre.f32le.bin"
);
const RETAINED_MULTIMERGE_ATTN_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/attention-state-score-pre-bits.i32le.bin"
);
const RETAINED_MULTIMERGE_INDEX_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-state-kv-pre.f32le.bin"
);
const RETAINED_MULTIMERGE_INDEX_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-state-score-pre-bits.i32le.bin"
);
const RETAINED_MULTIMERGE_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/q-lora-norm.f32le.bin");
const RETAINED_MULTIMERGE_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/attn-norm.f32le.bin");
const RETAINED_MULTIMERGE_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/q-cur.f32le.bin");
const RETAINED_MULTIMERGE_KV_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/kv-cur.f32le.bin");
const RETAINED_MULTIMERGE_COMPRESSED_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/compressed-kv-row0.f32le.bin");
const RETAINED_MULTIMERGE_COMPRESSED_INDEXER_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/compressed-indexer-row2048.f32le.bin"
);
const RETAINED_MULTIMERGE_INDEXER_Q_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-q.f32le.bin");
const RETAINED_MULTIMERGE_INDEXER_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-weights.f32le.bin");
const RETAINED_MULTIMERGE_INDEXER_SCORES_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-scores.f32le.bin");
const RETAINED_MULTIMERGE_INDEXER_TOPK_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/indexer-topk.i32le.bin");
const RETAINED_MULTIMERGE_KQV_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/kqv-out.f32le.bin");
const RETAINED_MULTIMERGE_KQV_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/kqv-back.f32le.bin");
const RETAINED_MULTIMERGE_ATTN_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/attn-low.f32le.bin");
const RETAINED_MULTIMERGE_ATTN_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/attn-out.f32le.bin");
const RETAINED_MULTIMERGE_HC_ATTN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-attn-post.f32le.bin");
const RETAINED_MULTIMERGE_HC_ATTN_PRE_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-attn-pre-mixes.f32le.bin");
const RETAINED_MULTIMERGE_HC_ATTN_PRE_WEIGHTS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/hc-attn-pre-weights.f32le.bin"
);
const RETAINED_MULTIMERGE_HC_ATTN_PRE_POST_WEIGHTS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/hc-attn-pre-post-weights.f32le.bin"
);
const RETAINED_MULTIMERGE_HC_ATTN_PRE_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-attn-pre-comb.f32le.bin");
const RETAINED_MULTIMERGE_HC_ATTN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-attn-pre.f32le.bin");
const RETAINED_MULTIMERGE_Q_LORA_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/q-lora.f32le.bin");
const RETAINED_MULTIMERGE_KV_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/kv-raw.f32le.bin");
const RETAINED_MULTIMERGE_KV_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/kv-norm.f32le.bin");
const RETAINED_MULTIMERGE_Q_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/q-raw.f32le.bin");
const RETAINED_MULTIMERGE_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/kv-rope.f32le.bin");
const RETAINED_MULTIMERGE_HC_FFN_PRE_MIXES_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-ffn-pre-mixes.f32le.bin");
const RETAINED_MULTIMERGE_HC_FFN_PRE_WEIGHTS_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-ffn-pre-weights.f32le.bin");
const RETAINED_MULTIMERGE_HC_FFN_PRE_POST_WEIGHTS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/hc-ffn-pre-post-weights.f32le.bin"
);
const RETAINED_MULTIMERGE_HC_FFN_PRE_COMB_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-ffn-pre-comb.f32le.bin");
const RETAINED_MULTIMERGE_HC_FFN_PRE_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-ffn-pre.f32le.bin");
const RETAINED_MULTIMERGE_FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-norm.f32le.bin");
const RETAINED_MULTIMERGE_FFN_MOE_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-moe-logits.f32le.bin");
const RETAINED_MULTIMERGE_FFN_MOE_PROBS_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-moe-probs.f32le.bin");
const RETAINED_MULTIMERGE_FFN_MOE_TOPK_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-moe-topk.i32le.bin");
const RETAINED_MULTIMERGE_FFN_MOE_WEIGHTS_SCALED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-moe-weights-scaled.f32le.bin"
);
const RETAINED_MULTIMERGE_FFN_MOE_WEIGHTED_SWIGLU_BYTES: &[u8] = include_bytes!(
    "../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-moe-weighted-swiglu.f32le.bin"
);
const RETAINED_MULTIMERGE_FFN_MOE_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-moe-out.f32le.bin");
const RETAINED_MULTIMERGE_FFN_SHEXP_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/ffn-shexp.f32le.bin");
const RETAINED_MULTIMERGE_HC_FFN_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/retained-sparse-layer2-pos8195-v1/hc-ffn-post.f32le.bin");
const PREFILL_Q8_BATCH_OUTPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-q8-boundary-2048-v1/q-lora-batch-final-tile.f32le.bin");
const PREFILL_Q8_DECODE_OUTPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-q8-boundary-2048-v1/q-lora-decode-final-row.f32le.bin");
const PREFILL_QKV_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/attn-norm-final-tile.f32le.bin");
const PREFILL_QKV_Q_LORA_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/q-lora-final-tile.f32le.bin");
const PREFILL_QKV_Q_LORA_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/q-lora-norm-final-tile.f32le.bin");
const PREFILL_QKV_KV_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/kv-raw-final-tile.f32le.bin");
const PREFILL_QKV_KV_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/kv-norm-final-tile.f32le.bin");
const PREFILL_QKV_Q_RAW_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/q-raw-final-tile.f32le.bin");
const PREFILL_QKV_Q_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-qkv-boundary-2048-v1/q-current-final-tile.f32le.bin");
const PREFILL_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-kv-state-2048-v1/kv-rope-final-tile.f32le.bin");
const PREFILL_KV_CURRENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-kv-state-2048-v1/kv-current-final-tile.f32le.bin");
const PREFILL_RAW_CACHE_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-kv-state-2048-v1/raw-cache-final-tile.f32le.bin");
const PREFILL_ATTENTION_KV_PREFIX_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-attention-read-2048-v1/kv-current-prefix.f32le.bin");
const PREFILL_ATTENTION_OUTPUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-attention-read-2048-v1/kqv-output-final-tile.f32le.bin");
const PREFILL_ATTENTION_BACK_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-attention-read-2048-v1/kqv-back-final-tile.f32le.bin");
const PREFILL_ATTENTION_LOW_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-attention-output-2048-v1/attn-low-final-tile.f32le.bin");
const PREFILL_ATTENTION_PROJECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-attention-output-2048-v1/attn-out-final-tile.f32le.bin");
const PREFILL_ATTENTION_HC_POST_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-attention-output-2048-v1/hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_FFN_CUR_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/hc-ffn-pre-final-tile.f32le.bin");
const PREFILL_FFN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/ffn-norm-final-tile.f32le.bin");
const PREFILL_FFN_ROUTER_LOGITS_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/ffn-moe-logits-final-tile.f32le.bin");
const PREFILL_FFN_ROUTER_PROBS_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/ffn-moe-probs-final-tile.f32le.bin");
const PREFILL_FFN_SELECTED_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/ffn-moe-topk-final-tile.i32le.bin");
const PREFILL_FFN_ROUTER_WEIGHTS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-ffn-output-2048-v1/ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_FFN_ROUTED_MID_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-ffn-output-2048-v1/ffn-moe-weighted-swiglu-final-tile.f32le.bin"
);
const PREFILL_FFN_ROUTED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/ffn-moe-out-final-tile.f32le.bin");
const PREFILL_FFN_SHARED_OUT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/ffn-shexp-final-tile.f32le.bin");
const PREFILL_FFN_HC_POST_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-ffn-output-2048-v1/hc-ffn-post-final-tile.f32le.bin");
const PREFILL_LAYER1_HC_PRE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-ingress-2048-v1/layer1-hc-attn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER1_ATTN_NORM_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-ingress-2048-v1/layer1-attn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER1_Q_LORA_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-ingress-2048-v1/layer1-q-lora-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_Q_NORM_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-q-lora-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_KV_NORM_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-kvnorm-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_Q_CUR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-qcur-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_KV_ROPE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-kvrope-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_KV_CUR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-kvcur-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_KV_PREFIX_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-kv-current-prefix.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_KQV_OUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-kqv-out-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_KQV_BACK_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-kqv-back-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ATTN_LOW_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-attn-low-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ATTN_OUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-attn-out-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_HC_ATTN_POST_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_HC_FFN_PRE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_FFN_NORM_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ROUTER_LOGITS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-moe-logits-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ROUTER_PROBS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-moe-probs-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_SELECTED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ROUTER_WEIGHTS_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ROUTED_MID_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-moe-weighted-swiglu-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_ROUTED_OUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_SHARED_OUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER1_COMPLETE_HC_FFN_POST_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer1-complete-2048-v1/layer1-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_PREVIOUS_LAYER0_KV_CUR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layers01-previous-tile-2048-v1/layer0-kv-current.f32le.bin"
);
const PREFILL_PREVIOUS_LAYER0_HC_POST_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layers01-previous-tile-2048-v1/layer0-hc-ffn-post.f32le.bin"
);
const PREFILL_PREVIOUS_LAYER0_SELECTED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layers01-previous-tile-2048-v1/layer0-selected.i32le.bin"
);
const PREFILL_PREVIOUS_LAYER1_KV_CUR_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layers01-previous-tile-2048-v1/layer1-kv-current.f32le.bin"
);
const PREFILL_PREVIOUS_LAYER1_HC_POST_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layers01-previous-tile-2048-v1/layer1-hc-ffn-post.f32le.bin"
);
const PREFILL_PREVIOUS_LAYER1_SELECTED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layers01-previous-tile-2048-v1/layer1-selected.i32le.bin"
);
const PREFILL_LAYER2_KV_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer2-kvnorm-2048-v1/layer2-kv-norm.f32le.bin");
const PREFILL_LAYER2_KV_ROPE_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer2-kv-state-2048-v1/layer2-kv-rope.f32le.bin");
const PREFILL_LAYER2_KV_CURRENT_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer2-kv-state-2048-v1/layer2-kv-current.f32le.bin");
const PREFILL_LAYER2_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-compressors-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER2_ATTN_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-compressors-2048-v1/attention-state-kv.f32le.bin"
);
const PREFILL_LAYER2_ATTN_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-compressors-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER2_INDEXER_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-compressors-2048-v1/indexer-compressed-kv.f32le.bin"
);
const PREFILL_LAYER2_INDEXER_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer2-compressors-2048-v1/indexer-state-kv.f32le.bin");
const PREFILL_LAYER2_INDEXER_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-compressors-2048-v1/indexer-state-score.i32le.bin"
);
const PREFILL_LAYER4_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-compressors-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER4_ATTN_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-compressors-2048-v1/attention-state-kv.f32le.bin"
);
const PREFILL_LAYER4_ATTN_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-compressors-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER4_INDEXER_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-compressors-2048-v1/indexer-compressed-kv.f32le.bin"
);
const PREFILL_LAYER4_INDEXER_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer4-compressors-2048-v1/indexer-state-kv.f32le.bin");
const PREFILL_LAYER4_INDEXER_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-compressors-2048-v1/indexer-state-score.i32le.bin"
);
const PREFILL_LAYER6_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-compressors-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER6_ATTN_STATE_KV_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-compressors-2048-v1/attention-state-kv.f32le.bin"
);
const PREFILL_LAYER6_ATTN_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-compressors-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER6_INDEXER_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-compressors-2048-v1/indexer-compressed-kv.f32le.bin"
);
const PREFILL_LAYER6_INDEXER_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer6-compressors-2048-v1/indexer-state-kv.f32le.bin");
const PREFILL_LAYER6_INDEXER_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-compressors-2048-v1/indexer-state-score.i32le.bin"
);
const PREFILL_LAYER6_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-attention-2048-v1/layer6-attention-output.f32le.bin"
);
const PREFILL_LAYER6_ATTENTION_OUTPUT_CHECKSUM: u64 = 0xfc80_59f6_f92b_0be9;
const PREFILL_LAYER6_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-attention-2048-v1/layer6-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER6_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0x2c16_8c26_633d_52a3;
const PREFILL_LAYER6_KQV_OUT_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer6-attention-2048-v1/layer6-kqv-out-row0.f32le.bin");
const PREFILL_LAYER6_KQV_BACK_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-attention-2048-v1/layer6-kqv-back-row0.f32le.bin"
);
const PREFILL_LAYER6_ATTN_LOW_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-attention-2048-v1/layer6-attn-low-row0.f32le.bin"
);
const PREFILL_LAYER7_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-compressor-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER7_ATTN_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer7-compressor-2048-v1/attention-state-kv.f32le.bin");
const PREFILL_LAYER7_ATTN_STATE_SCORE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-compressor-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER7_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-attention-2048-v1/layer7-attention-output.f32le.bin"
);
const PREFILL_LAYER7_ATTENTION_OUTPUT_CHECKSUM: u64 = 0x06b5_6be5_5f70_8ac7;
const PREFILL_LAYER7_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-attention-2048-v1/layer7-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER7_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0x411a_0c36_d954_e04c;
const PREFILL_LAYER7_KQV_OUT_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer7-attention-2048-v1/layer7-kqv-out-row0.f32le.bin");
const PREFILL_LAYER7_KQV_BACK_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-attention-2048-v1/layer7-kqv-back-row0.f32le.bin"
);
const PREFILL_LAYER7_ATTN_LOW_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-attention-2048-v1/layer7-attn-low-row0.f32le.bin"
);
const PREFILL_LAYER8_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-compressor-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER8_ATTN_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer8-compressor-2048-v1/attention-state-kv.f32le.bin");
const PREFILL_LAYER8_ATTN_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-compressor-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER8_INDEXER_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-compressor-2048-v1/indexer-compressed-kv.f32le.bin"
);
const PREFILL_LAYER8_INDEXER_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer8-compressor-2048-v1/indexer-state-kv.f32le.bin");
const PREFILL_LAYER8_INDEXER_STATE_SCORE_BITS: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-compressor-2048-v1/indexer-state-score.i32le.bin"
);
const PREFILL_LAYER8_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-attention-2048-v1/layer8-attention-output.f32le.bin"
);
const PREFILL_LAYER8_ATTENTION_OUTPUT_CHECKSUM: u64 = 0xc733_5752_e51e_1255;
const PREFILL_LAYER8_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-attention-2048-v1/layer8-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER8_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0xb7c5_c48b_7801_39fc;
const PREFILL_LAYER8_KQV_OUT_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer8-attention-2048-v1/layer8-kqv-out-row0.f32le.bin");
const PREFILL_LAYER8_KQV_BACK_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-attention-2048-v1/layer8-kqv-back-row0.f32le.bin"
);
const PREFILL_LAYER8_ATTN_LOW_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-attention-2048-v1/layer8-attn-low-row0.f32le.bin"
);
const PREFILL_LAYER4_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-attention-2048-v1/layer4-attention-output.f32le.bin"
);
const PREFILL_LAYER4_ATTENTION_OUTPUT_CHECKSUM: u64 = 0x0b96_254b_94dd_8c53;
const PREFILL_LAYER4_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-attention-2048-v1/layer4-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER4_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0x679b_5a42_06be_3400;
const PREFILL_LAYER4_KQV_OUT_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer4-attention-2048-v1/layer4-kqv-out-row0.f32le.bin");
const PREFILL_LAYER4_KQV_BACK_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-attention-2048-v1/layer4-kqv-back-row0.f32le.bin"
);
const PREFILL_LAYER4_ATTN_LOW_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-attention-2048-v1/layer4-attn-low-row0.f32le.bin"
);
const PREFILL_LAYER4_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER4_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER4_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER4_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER4_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER4_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER4_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer4-complete-2048-v1/layer4-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER4_HC_FFN_POST_FULL_CHECKSUM: u64 = 0x7d02_e36d_de13_d258;
const PREFILL_LAYER5_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER5_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER5_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER5_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER5_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER5_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER5_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-complete-2048-v1/layer5-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER5_HC_FFN_POST_FULL_CHECKSUM: u64 = 0x7759_cb71_bf9c_817b;
const PREFILL_LAYER6_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER6_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER6_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER6_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER6_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER6_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER6_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer6-complete-2048-v1/layer6-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER6_HC_FFN_POST_FULL_CHECKSUM: u64 = 0x0ea8_e5a3_2059_5193;
const PREFILL_LAYER7_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER7_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER7_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER7_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER7_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER7_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER7_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer7-complete-2048-v1/layer7-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER7_HC_FFN_POST_FULL_CHECKSUM: u64 = 0xf0fc_a3b4_33f3_4a72;
const PREFILL_LAYER8_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER8_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER8_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER8_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER8_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER8_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER8_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer8-complete-2048-v1/layer8-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER8_HC_FFN_POST_FULL_CHECKSUM: u64 = 0xd145_dd96_16ce_0402;
const PREFILL_LAYER2_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-attention-2048-v1/layer2-attention-output.f32le.bin"
);
const PREFILL_LAYER2_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER2_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0x82a4_9f8f_e742_1325;
const PREFILL_LAYER2_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER2_HC_FFN_POST_FULL_CHECKSUM: u64 = 0xbdb1_d4ab_45ce_d224;
const PREFILL_LAYER2_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER2_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER2_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER2_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER2_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER2_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer2-complete-2048-v1/layer2-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER3_HC_ATTN_PRE_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-ingress-2048-v1/layer3-hc-attn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER3_HC_ATTN_PRE_FULL_CHECKSUM: u64 = 0xcb93_e0a2_51fd_7280;
const PREFILL_LAYER3_ATTN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-ingress-2048-v1/layer3-attn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER3_ATTN_NORM_FULL_CHECKSUM: u64 = 0xe64f_ba2f_04bf_cb54;
const PREFILL_LAYER3_Q_LORA_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-ingress-2048-v1/layer3-q-lora-final-tile.f32le.bin"
);
const PREFILL_LAYER3_Q_LORA_FULL_CHECKSUM: u64 = 0xd018_7575_8ab4_b722;
const PREFILL_LAYER3_Q_LORA_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-q-lora-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER3_Q_LORA_NORM_FULL_CHECKSUM: u64 = 0xe2f3_b81e_d020_c697;
const PREFILL_LAYER3_KV_RAW_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-kv-raw-final-tile.f32le.bin"
);
const PREFILL_LAYER3_KV_RAW_FULL_CHECKSUM: u64 = 0x3ad7_b41f_46f0_c66b;
const PREFILL_LAYER3_KV_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-kv-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER3_KV_NORM_FULL_CHECKSUM: u64 = 0x1c87_3bfc_b23b_5749;
const PREFILL_LAYER3_Q_RAW_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-q-raw-final-tile.f32le.bin"
);
const PREFILL_LAYER3_Q_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-q-current-final-tile.f32le.bin"
);
const PREFILL_LAYER3_KV_ROPE_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-kv-rope-final-tile.f32le.bin"
);
const PREFILL_LAYER3_KV_ROPE_FULL_CHECKSUM: u64 = 0x4955_1ba5_ae96_2613;
const PREFILL_LAYER3_KV_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-kv-state-2048-v1/layer3-kv-current-final-tile.f32le.bin"
);
const PREFILL_LAYER3_KV_CUR_FULL_CHECKSUM: u64 = 0x7454_7b0c_9158_4ad8;
const PREFILL_LAYER3_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-compressor-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER3_ATTN_COMPRESSED_CHECKSUM: u64 = 0x8c62_768a_54d0_d439;
const PREFILL_LAYER3_ATTN_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer3-compressor-2048-v1/attention-state-kv.f32le.bin");
const PREFILL_LAYER3_ATTN_STATE_KV_CHECKSUM: u64 = 0xeb05_052e_a5b6_2325;
const PREFILL_LAYER3_ATTN_STATE_SCORE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-compressor-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER3_ATTN_STATE_SCORE_CHECKSUM: u64 = 0x5512_b2ce_2eb6_2325;
const PREFILL_LAYER5_ATTN_COMPRESSED_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-compressor-2048-v1/attention-compressed-kv.f32le.bin"
);
const PREFILL_LAYER5_ATTN_STATE_KV_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer5-compressor-2048-v1/attention-state-kv.f32le.bin");
const PREFILL_LAYER5_ATTN_STATE_SCORE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-compressor-2048-v1/attention-state-score.i32le.bin"
);
const PREFILL_LAYER5_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-attention-2048-v1/layer5-attention-output.f32le.bin"
);
const PREFILL_LAYER5_ATTENTION_OUTPUT_CHECKSUM: u64 = 0xe460_45ae_4f35_1604;
const PREFILL_LAYER5_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-attention-2048-v1/layer5-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER5_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0x02a4_5603_b5ea_5daa;
const PREFILL_LAYER5_KQV_OUT_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer5-attention-2048-v1/layer5-kqv-out-row0.f32le.bin");
const PREFILL_LAYER5_KQV_BACK_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-attention-2048-v1/layer5-kqv-back-row0.f32le.bin"
);
const PREFILL_LAYER5_ATTN_LOW_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer5-attention-2048-v1/layer5-attn-low-row0.f32le.bin"
);
const PREFILL_LAYER3_ATTENTION_OUTPUT_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-attention-2048-v1/layer3-attention-output.f32le.bin"
);
const PREFILL_LAYER3_ATTENTION_OUTPUT_CHECKSUM: u64 = 0x66cd_da01_32c4_d73c;
const PREFILL_LAYER3_HC_ATTN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-attention-2048-v1/layer3-hc-attn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER3_HC_ATTN_POST_FULL_CHECKSUM: u64 = 0x00ab_ade4_3231_7930;
const PREFILL_LAYER3_FFN_CUR_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-hc-ffn-pre-final-tile.f32le.bin"
);
const PREFILL_LAYER3_FFN_NORM_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-ffn-norm-final-tile.f32le.bin"
);
const PREFILL_LAYER3_ROUTER_SELECTED_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-ffn-moe-topk-final-tile.i32le.bin"
);
const PREFILL_LAYER3_ROUTER_WEIGHTS_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-ffn-moe-weights-scaled-final-tile.f32le.bin"
);
const PREFILL_LAYER3_ROUTED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-ffn-moe-out-final-tile.f32le.bin"
);
const PREFILL_LAYER3_SHARED_OUT_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-ffn-shexp-final-tile.f32le.bin"
);
const PREFILL_LAYER3_HC_FFN_POST_FINAL_TILE_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-complete-2048-v1/layer3-hc-ffn-post-final-tile.f32le.bin"
);
const PREFILL_LAYER3_HC_FFN_POST_FULL_CHECKSUM: u64 = 0x02e3_0d5f_e341_ae76;
const PREFILL_LAYER4_QKV_FINAL_TILE_BYTES: [&[u8]; 10] = [
    include_bytes!(
        "../../fixtures/prefill-layer4-qkv-2048-v1/layer4-hc-attn-pre-final-tile.f32le.bin"
    ),
    include_bytes!(
        "../../fixtures/prefill-layer4-qkv-2048-v1/layer4-attn-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer4-qkv-2048-v1/layer4-q-lora-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer4-qkv-2048-v1/layer4-q-lora-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer4-qkv-2048-v1/layer4-kv-raw-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer4-qkv-2048-v1/layer4-kv-norm-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer4-qkv-2048-v1/layer4-q-raw-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer4-qkv-2048-v1/layer4-q-current-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer4-qkv-2048-v1/layer4-kv-rope-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer4-qkv-2048-v1/layer4-kv-current-final-tile.f32le.bin"
    ),
];
const PREFILL_LAYER5_QKV_FINAL_TILE_BYTES: [&[u8]; 10] = [
    include_bytes!(
        "../../fixtures/prefill-layer5-qkv-2048-v1/layer5-hc-attn-pre-final-tile.f32le.bin"
    ),
    include_bytes!(
        "../../fixtures/prefill-layer5-qkv-2048-v1/layer5-attn-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer5-qkv-2048-v1/layer5-q-lora-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer5-qkv-2048-v1/layer5-q-lora-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer5-qkv-2048-v1/layer5-kv-raw-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer5-qkv-2048-v1/layer5-kv-norm-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer5-qkv-2048-v1/layer5-q-raw-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer5-qkv-2048-v1/layer5-q-current-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer5-qkv-2048-v1/layer5-kv-rope-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer5-qkv-2048-v1/layer5-kv-current-final-tile.f32le.bin"
    ),
];
const PREFILL_LAYER6_QKV_FINAL_TILE_BYTES: [&[u8]; 10] = [
    include_bytes!(
        "../../fixtures/prefill-layer6-qkv-2048-v1/layer6-hc-attn-pre-final-tile.f32le.bin"
    ),
    include_bytes!(
        "../../fixtures/prefill-layer6-qkv-2048-v1/layer6-attn-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer6-qkv-2048-v1/layer6-q-lora-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer6-qkv-2048-v1/layer6-q-lora-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer6-qkv-2048-v1/layer6-kv-raw-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer6-qkv-2048-v1/layer6-kv-norm-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer6-qkv-2048-v1/layer6-q-raw-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer6-qkv-2048-v1/layer6-q-current-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer6-qkv-2048-v1/layer6-kv-rope-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer6-qkv-2048-v1/layer6-kv-current-final-tile.f32le.bin"
    ),
];
const PREFILL_LAYER7_QKV_FINAL_TILE_BYTES: [&[u8]; 10] = [
    include_bytes!(
        "../../fixtures/prefill-layer7-qkv-2048-v1/layer7-hc-attn-pre-final-tile.f32le.bin"
    ),
    include_bytes!(
        "../../fixtures/prefill-layer7-qkv-2048-v1/layer7-attn-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer7-qkv-2048-v1/layer7-q-lora-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer7-qkv-2048-v1/layer7-q-lora-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer7-qkv-2048-v1/layer7-kv-raw-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer7-qkv-2048-v1/layer7-kv-norm-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer7-qkv-2048-v1/layer7-q-raw-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer7-qkv-2048-v1/layer7-q-current-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer7-qkv-2048-v1/layer7-kv-rope-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer7-qkv-2048-v1/layer7-kv-current-final-tile.f32le.bin"
    ),
];
const PREFILL_LAYER8_QKV_FINAL_TILE_BYTES: [&[u8]; 10] = [
    include_bytes!(
        "../../fixtures/prefill-layer8-qkv-2048-v1/layer8-hc-attn-pre-final-tile.f32le.bin"
    ),
    include_bytes!(
        "../../fixtures/prefill-layer8-qkv-2048-v1/layer8-attn-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer8-qkv-2048-v1/layer8-q-lora-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer8-qkv-2048-v1/layer8-q-lora-norm-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer8-qkv-2048-v1/layer8-kv-raw-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer8-qkv-2048-v1/layer8-kv-norm-final-tile.f32le.bin"),
    include_bytes!("../../fixtures/prefill-layer8-qkv-2048-v1/layer8-q-raw-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer8-qkv-2048-v1/layer8-q-current-final-tile.f32le.bin"
    ),
    include_bytes!("../../fixtures/prefill-layer8-qkv-2048-v1/layer8-kv-rope-final-tile.f32le.bin"),
    include_bytes!(
        "../../fixtures/prefill-layer8-qkv-2048-v1/layer8-kv-current-final-tile.f32le.bin"
    ),
];
const PREFILL_LAYER3_KQV_OUT_ROW0_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-layer3-attention-2048-v1/layer3-kqv-out-row0.f32le.bin");
const PREFILL_LAYER3_KQV_BACK_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-attention-2048-v1/layer3-kqv-back-row0.f32le.bin"
);
const PREFILL_LAYER3_ATTN_LOW_ROW0_BYTES: &[u8] = include_bytes!(
    "../../fixtures/prefill-layer3-attention-2048-v1/layer3-attn-low-row0.f32le.bin"
);
const PREFILL_HC_TOKEN_IDS_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-hc-ingress-2048-v1/token-ids-final-tile.i32le.bin");
const PREFILL_HC_COLLAPSED_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-hc-ingress-2048-v1/hc-attn-pre-final-tile.f32le.bin");
const PREFILL_HC_ATTN_NORM_BYTES: &[u8] =
    include_bytes!("../../fixtures/prefill-hc-ingress-2048-v1/attn-norm-final-tile.f32le.bin");
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
pub struct PrefillQ8BoundaryProbeReport {
    pub fixture_id: &'static str,
    pub tensor_name: String,
    pub rows: u64,
    pub input_elements_per_row: u64,
    pub output_elements_per_row: u64,
    pub no_copy_pointer_match: bool,
    pub batch_threads_per_threadgroup: u32,
    pub batch_threadgroups_x: u32,
    pub batch_threadgroups_y: u32,
    pub batch_wall_ms: f64,
    pub batch_gpu_ms: f64,
    pub decode_wall_ms: f64,
    pub decode_gpu_ms: f64,
    pub input_checksum: u64,
    pub batch_output_checksum: u64,
    pub decode_output_checksum: u64,
    pub final_row_mismatches: u64,
    pub final_row_max_abs_error: f32,
}

#[derive(Clone, Debug)]
pub struct PrefillQkvBoundaryProbeReport {
    pub fixture_id: &'static str,
    pub rows: u64,
    pub position_start: u32,
    pub dispatches: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub checksums: [u64; 7],
}

#[derive(Clone, Debug)]
pub struct PrefillLayer0BoundaryProbeReport {
    pub ingress_fixture_id: &'static str,
    pub qkv_fixture_id: &'static str,
    pub kv_state_fixture_id: &'static str,
    pub attention_fixture_id: &'static str,
    pub attention_output_fixture_id: &'static str,
    pub ffn_output_fixture_id: &'static str,
    pub rows: u64,
    pub position_start: u32,
    pub dispatches: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub raw_cache_rows: u32,
    pub raw_cache_target_row: u32,
    pub raw_cache_guard_rows: u32,
    pub attention_kv_rows: u32,
    pub attention_kv_prefix_rows: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub checksums: [u64; 28],
}

#[derive(Clone, Debug)]
pub struct PrefillLayers01BoundaryProbeReport {
    pub layer0: PrefillLayer0BoundaryProbeReport,
    pub layer1_fixture_id: &'static str,
    pub checksums: [u64; 3],
}

#[derive(Clone, Debug)]
pub struct PrefillLayers01CompleteBoundaryProbeReport {
    pub layers01: PrefillLayers01BoundaryProbeReport,
    pub complete_fixture_id: &'static str,
    pub checksums: [u64; 20],
}

#[derive(Clone, Debug)]
pub struct PrefillLayers01RowCoverageProbeReport {
    pub previous_fixture_id: &'static str,
    pub previous_position_start: u32,
    pub previous_position_end: u32,
    pub previous_dispatches: u32,
    pub previous_wrapped_model_ranges: u32,
    pub previous_pointer_matches: u32,
    pub previous_raw_cache_target_row: u32,
    pub previous_wall_ms: f64,
    pub previous_gpu_ms: f64,
    pub previous_checksums: [u64; 6],
    pub final_tile: PrefillLayers01CompleteBoundaryProbeReport,
}

#[derive(Clone, Debug)]
pub struct PrefillLayers01LiveKvChainProbeReport {
    pub tiles: PrefillLayers01RowCoverageProbeReport,
    pub retained_kv_rows_after_first_tile: u32,
    pub retained_kv_rows_after_final_tile: u32,
}

#[derive(Clone, Debug)]
pub struct PrefillLayers01LiveKvLoopProbeReport {
    pub tiles: Vec<PrefillLayer0BoundaryProbeReport>,
    pub final_tile: PrefillLayers01CompleteBoundaryProbeReport,
}

#[derive(Clone, Debug)]
pub struct PrefillLayers012KvnormLoopProbeReport {
    pub tiles: Vec<PrefillLayer0BoundaryProbeReport>,
    pub layer2_fixture_id: &'static str,
    pub layer2_kvnorm_checksums: Vec<u64>,
    pub final_tile: PrefillLayers01CompleteBoundaryProbeReport,
}

#[derive(Clone, Debug)]
pub struct PrefillLayers012KvStateLoopProbeReport {
    pub tiles: Vec<PrefillLayer0BoundaryProbeReport>,
    pub layer2_kvnorm_fixture_id: &'static str,
    pub layer2_kv_state_fixture_id: &'static str,
    pub layer2_checksums: Vec<[u64; 3]>,
    pub final_tile: PrefillLayers01CompleteBoundaryProbeReport,
}

#[derive(Clone, Debug)]
pub struct PrefillLayers012CompressorLoopProbeReport {
    pub tiles: Vec<PrefillLayer0BoundaryProbeReport>,
    pub layer2_kvnorm_fixture_id: &'static str,
    pub layer2_kv_state_fixture_id: &'static str,
    pub layer2_compressor_fixture_id: &'static str,
    pub layer2_checksums: Vec<[u64; 3]>,
    pub layer2_compressor_checksums: Vec<[u64; 6]>,
    pub final_tile: PrefillLayers01CompleteBoundaryProbeReport,
}

#[derive(Clone, Debug)]
pub struct PrefillLayers012AttentionLoopProbeReport {
    pub compressor: PrefillLayers012CompressorLoopProbeReport,
    pub attention_fixture_id: &'static str,
    pub attention_hc_fixture_id: &'static str,
    pub layer3_ingress_fixture_id: &'static str,
    pub layer3_kv_state_fixture_id: &'static str,
    pub layer3_compressor_fixture_id: &'static str,
    pub layer3_attention_fixture_id: &'static str,
    pub layer3_complete_fixture_id: &'static str,
    pub layer4_qkv_fixture_id: &'static str,
    pub layer4_compressor_fixture_id: &'static str,
    pub layer4_attention_fixture_id: &'static str,
    pub layer4_complete_fixture_id: &'static str,
    pub layer5_qkv_fixture_id: &'static str,
    pub layer5_compressor_fixture_id: &'static str,
    pub layer5_attention_fixture_id: &'static str,
    pub layer5_complete_fixture_id: &'static str,
    pub layer6_qkv_fixture_id: &'static str,
    pub layer6_compressor_fixture_id: &'static str,
    pub layer6_attention_fixture_id: &'static str,
    pub layer6_complete_fixture_id: &'static str,
    pub layer7_qkv_fixture_id: &'static str,
    pub layer7_compressor_fixture_id: &'static str,
    pub layer7_attention_fixture_id: &'static str,
    pub layer7_complete_fixture_id: &'static str,
    pub layer8_qkv_fixture_id: &'static str,
    pub layer8_compressor_fixture_id: &'static str,
    pub layer8_attention_fixture_id: &'static str,
    pub layer8_complete_fixture_id: &'static str,
    pub rows: u32,
    pub raw_kv_rows: u32,
    pub compressed_kv_rows: u32,
    pub layer3_compressed_kv_rows: u32,
    pub dispatches: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub output_checksum: u64,
    pub after_attention_hc_checksum: u64,
    pub after_ffn_hc_checksum: u64,
    pub layer3_hc_attn_pre_checksum: u64,
    pub layer3_attn_norm_checksum: u64,
    pub layer3_q_lora_checksum: u64,
    pub layer3_q_lora_norm_checksum: u64,
    pub layer3_kv_raw_checksum: u64,
    pub layer3_kv_norm_checksum: u64,
    pub layer3_q_raw_final_tile_checksum: u64,
    pub layer3_q_cur_final_tile_checksum: u64,
    pub layer3_kv_rope_checksum: u64,
    pub layer3_kv_cur_checksum: u64,
    pub layer3_attn_compressed_checksum: u64,
    pub layer3_attn_state_kv_checksum: u64,
    pub layer3_attn_state_score_checksum: u64,
    pub layer3_attention_output_checksum: u64,
    pub layer3_after_attention_hc_checksum: u64,
    pub layer3_after_ffn_hc_checksum: u64,
    pub layer4_qkv_checksums: [u64; 10],
    pub layer4_compressor_checksums: [u64; 6],
    pub layer4_attention_output_checksum: u64,
    pub layer4_after_attention_hc_checksum: u64,
    pub layer4_after_ffn_hc_checksum: u64,
    pub layer5_qkv_checksums: [u64; 10],
    pub layer5_compressor_checksums: [u64; 3],
    pub layer5_attention_output_checksum: u64,
    pub layer5_after_attention_hc_checksum: u64,
    pub layer5_after_ffn_hc_checksum: u64,
    pub layer6_qkv_checksums: [u64; 10],
    pub layer6_compressor_checksums: [u64; 6],
    pub layer6_attention_output_checksum: u64,
    pub layer6_after_attention_hc_checksum: u64,
    pub layer6_after_ffn_hc_checksum: u64,
    pub layer7_qkv_checksums: [u64; 10],
    pub layer7_compressor_checksums: [u64; 3],
    pub layer7_attention_output_checksum: u64,
    pub layer7_after_attention_hc_checksum: u64,
    pub layer7_after_ffn_hc_checksum: u64,
    pub layer8_qkv_checksums: [u64; 10],
    pub layer8_compressor_checksums: [u64; 6],
    pub layer8_attention_output_checksum: u64,
    pub layer8_after_attention_hc_checksum: u64,
    pub layer8_after_ffn_hc_checksum: u64,
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
    pub closed_loop_sampling: bool,
    pub externally_supplied_decode_inputs: bool,
}

#[derive(Clone, Debug)]
pub struct TimedDecoderStepReport {
    pub position: u32,
    pub input_token: u32,
    pub selected_token: u32,
    pub wall_ms: f64,
    pub output_head_gpu_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ClosedLoopDecoderProbeReport {
    pub correctness: DecoderOutputProbeReport,
    pub timed_steps: Vec<TimedDecoderStepReport>,
    pub pipeline_prepare_ms: f64,
    pub generation_wall_ms: f64,
    pub generation_tps: f64,
    pub first_token_ms: f64,
    pub steady_wall_ms: f64,
    pub steady_tps: f64,
}

#[derive(Clone, Debug)]
pub struct Position127DecoderProbeReport {
    pub fixture_id: &'static str,
    pub committed_tokens: Vec<u32>,
    pub evaluated_positions: u32,
    pub final_position: u32,
    pub cache_capacity_rows: u32,
    pub compressed_cache_capacity_rows: u32,
    pub command_buffers_per_position: u32,
    pub host_waits_per_position: u32,
    pub wall_ms: f64,
    pub eval_tps: f64,
    pub final_logits_checksum: u64,
    pub ratio128_layer3_checksum: u64,
    pub ratio128_layer5_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct ColdPrefillDecoderProbeReport {
    pub fixture_id: &'static str,
    pub prompt_token: u32,
    pub committed_tokens: Vec<u32>,
    pub prefill_wall_ms: f64,
    pub prefill_logits_checksum: u64,
    pub decode_wall_ms: f64,
    pub decode_tps: f64,
    pub final_logits_checksum: u64,
    pub ratio128_layer3_checksum: u64,
    pub ratio128_layer5_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct PrefillFrontierProbeReport {
    pub fixture_id: &'static str,
    pub context_capacity: u32,
    pub prompt_tokens: u32,
    pub final_position: u32,
    pub raw_cache_capacity_rows: u32,
    pub ratio4_compressed_capacity_rows: u32,
    pub ratio128_compressed_capacity_rows: u32,
    pub selected_token: u32,
    pub wall_ms: f64,
    pub prefill_tps: f64,
    pub decode_logits_checksum: u64,
    pub batch_logits_mismatch_count: u32,
    pub batch_logits_max_abs_error: f32,
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

#[derive(Clone, Debug)]
pub struct SparseIndexedAttentionProbeReport {
    pub fixture_id: &'static str,
    pub position: u32,
    pub compressed_rows: u32,
    pub raw_rows: u32,
    pub top_k: u32,
    pub diagnostic_threshold_override: u32,
    pub pinned_default_threshold: u32,
    pub first_default_sparse_rows: u32,
    pub dispatches: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub split_count: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub indexer_q_checksum: u64,
    pub indexer_weights_checksum: u64,
    pub indexer_scores_checksum: u64,
    pub indexer_topk_checksum: u64,
    pub kqv_out_checksum: u64,
    pub kqv_back_checksum: u64,
}

#[derive(Clone, Debug)]
pub struct RetainedSparseBoundaryProbeReport {
    pub fixture_id: &'static str,
    pub token: u32,
    pub layer: u32,
    pub position: u32,
    pub raw_rows: u32,
    pub seeded_raw_rows: u32,
    pub compressed_rows: u32,
    pub seeded_compressed_rows: u32,
    pub top_k: u32,
    pub sort_blocks: u32,
    pub merge_passes: u32,
    pub topk_work_width: u32,
    pub dispatches: u32,
    pub wrapped_model_ranges: u32,
    pub pointer_matches: u32,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub exact_tensor_checks: u32,
    pub q_current_checksum: u64,
    pub compressed_kv_checksum: u64,
    pub compressed_indexer_checksum: u64,
    pub indexer_scores_checksum: u64,
    pub indexer_topk_checksum: u64,
    pub kqv_out_checksum: u64,
    pub kqv_back_checksum: u64,
    pub attention_hc_checksum: u64,
    pub selected_experts_checksum: u64,
    pub final_hc_checksum: u64,
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
    if report.steps.len() != 4
        || report.command_buffers_per_step != 44
        || report.host_waits_per_step != 2
        || report.kv_cache_layers != 43
        || report.cache_capacity_rows != 128
        || report.logits_elements != 129280
        || report.closed_loop_sampling == report.externally_supplied_decode_inputs
    {
        return Err(Error::invalid(
            "decoder-output report has inconsistent boundary metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{DECODER_OUTPUT_PROBE_SCHEMA}\",\n  \"selection\": \"lowest-token-id-argmax\",\n  \"input_tokens\": [201, 361, 1915, 262],\n  \"command_buffers_per_step\": {},\n  \"host_waits_per_step\": {},\n  \"kv_cache_layers\": {},\n  \"cache_capacity_rows\": {},\n  \"logits_elements\": {},\n  \"steps\": [",
        report.command_buffers_per_step,
        report.host_waits_per_step,
        report.kv_cache_layers,
        report.cache_capacity_rows,
        report.logits_elements,
    )?;
    for (index, step) in report.steps.iter().enumerate() {
        let expected_position = index as u32 + 1;
        let expected_input = [201_u32, 361, 1915, 262][index];
        let expected_selected = [361_u32, 1915, 262, 1554][index];
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
        "\n  ],\n  \"closed_loop_sampling\": {},\n  \"externally_supplied_decode_inputs\": {},\n  \"full_logits_c0_bitwise_match\": true,\n  \"c0_bitwise_match\": true\n}}\n",
        report.closed_loop_sampling,
        report.externally_supplied_decode_inputs,
    )?;
    Ok(())
}

pub fn write_closed_loop_decoder_probe_json<W: Write>(
    output: &mut W,
    report: &ClosedLoopDecoderProbeReport,
) -> Result<()> {
    let expected_inputs = [201_u32, 361, 1915, 262];
    let expected_outputs = [361_u32, 1915, 262, 1554];
    if report.correctness.steps.len() != 4
        || report.correctness.command_buffers_per_step != 44
        || report.correctness.host_waits_per_step != 2
        || report.correctness.kv_cache_layers != 43
        || report.correctness.logits_elements != 129280
        || !report.correctness.closed_loop_sampling
        || report.correctness.externally_supplied_decode_inputs
        || report.timed_steps.len() != 4
        || !report.pipeline_prepare_ms.is_finite()
        || report.pipeline_prepare_ms <= 0.0
        || !report.generation_wall_ms.is_finite()
        || report.generation_wall_ms <= 0.0
        || !report.generation_tps.is_finite()
        || report.generation_tps <= 0.0
        || !report.first_token_ms.is_finite()
        || report.first_token_ms <= 0.0
        || !report.steady_wall_ms.is_finite()
        || report.steady_wall_ms <= 0.0
        || !report.steady_tps.is_finite()
        || report.steady_tps <= 0.0
    {
        return Err(Error::invalid(
            "closed-loop decoder report has inconsistent aggregate metadata",
        ));
    }
    for (index, (correctness, timed)) in report
        .correctness
        .steps
        .iter()
        .zip(&report.timed_steps)
        .enumerate()
    {
        if correctness.position != index as u32 + 1
            || correctness.input_token != expected_inputs[index]
            || correctness.output_head.selected_token != expected_outputs[index]
            || timed.position != correctness.position
            || timed.input_token != correctness.input_token
            || timed.selected_token != correctness.output_head.selected_token
            || !timed.wall_ms.is_finite()
            || timed.wall_ms <= 0.0
            || !timed.output_head_gpu_ms.is_finite()
            || timed.output_head_gpu_ms < 0.0
        {
            return Err(Error::invalid(
                "closed-loop decoder report has inconsistent step metadata",
            ));
        }
    }
    let generation_wall_ms = report
        .timed_steps
        .iter()
        .map(|step| step.wall_ms)
        .sum::<f64>();
    let steady_wall_ms = report.timed_steps[1..]
        .iter()
        .map(|step| step.wall_ms)
        .sum::<f64>();
    if report.generation_wall_ms.to_bits() != generation_wall_ms.to_bits()
        || report.first_token_ms.to_bits() != report.timed_steps[0].wall_ms.to_bits()
        || report.steady_wall_ms.to_bits() != steady_wall_ms.to_bits()
        || report.generation_tps.to_bits() != (4000.0 / generation_wall_ms).to_bits()
        || report.steady_tps.to_bits() != (3000.0 / steady_wall_ms).to_bits()
    {
        return Err(Error::invalid(
            "closed-loop decoder report has inconsistent timing aggregates",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{CLOSED_LOOP_DECODER_PROBE_SCHEMA}\",\n  \"classification\": \"diagnostic\",\n  \"selection\": \"lowest-token-id-argmax\",\n  \"bootstrap_input_token\": 201,\n  \"generated_tokens\": [361, 1915, 262, 1554],\n  \"correctness\": {{\n    \"positions\": 4,\n    \"transformer_layers\": 43,\n    \"logits_per_position\": 129280,\n    \"closed_loop_sampling\": true,\n    \"externally_supplied_decode_inputs\": false,\n    \"full_logits_c0_bitwise_match\": true,\n    \"c0_bitwise_match\": true\n  }},\n  \"timed_path\": {{\n    \"pipeline_prepare_ms\": {:.6},\n    \"pipeline_preparation_in_interval\": false,\n    \"command_buffers_per_step\": 44,\n    \"host_waits_per_step\": 2,\n    \"correctness_readback_in_interval\": false,\n    \"sampling_logits_readback_in_interval\": true,\n    \"argmax_in_interval\": true,\n    \"steps\": [",
        report.pipeline_prepare_ms,
    )?;
    for (index, step) in report.timed_steps.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(
            output,
            "\n      {{\"position\": {}, \"input_token\": {}, \"selected_token\": {}, \"wall_ms\": {:.6}, \"output_head_gpu_ms\": {:.6}}}",
            step.position,
            step.input_token,
            step.selected_token,
            step.wall_ms,
            step.output_head_gpu_ms,
        )?;
    }
    write!(
        output,
        "\n    ],\n    \"metrics\": {{\"gen_tokens\": 4, \"gen_steady_tokens\": 3, \"gen_ms\": {:.6}, \"gen_tps\": {:.6}, \"gen_first_ms\": {:.6}, \"gen_steady_ms\": {:.6}, \"gen_steady_tps\": {:.6}}}\n  }},\n  \"paired_protocol_eligible\": false,\n  \"paired_protocol_blocker\": \"captured initial state and only four committed positions\"\n}}\n",
        report.generation_wall_ms,
        report.generation_tps,
        report.first_token_ms,
        report.steady_wall_ms,
        report.steady_tps,
    )?;
    Ok(())
}

pub fn write_position127_decoder_probe_json<W: Write>(
    output: &mut W,
    report: &Position127DecoderProbeReport,
) -> Result<()> {
    let (expected_tokens, expected_logits) = position127_decoder_fixture()?;
    let expected_layer3 = decode_f32_fixture(
        LAYER3_POS127_COMPRESSED_KV_BYTES,
        "layer-3 position-127 compressed KV row",
    )?;
    let expected_layer5 = decode_f32_fixture(
        LAYER5_POS127_COMPRESSED_KV_BYTES,
        "layer-5 position-127 compressed KV row",
    )?;
    if report.fixture_id != POSITION127_DECODER_FIXTURE_ID
        || report.committed_tokens != expected_tokens
        || report.evaluated_positions != 127
        || report.final_position != 127
        || report.cache_capacity_rows != 128
        || report.compressed_cache_capacity_rows != 32
        || report.command_buffers_per_position != 44
        || report.host_waits_per_position != 2
        || !report.wall_ms.is_finite()
        || report.wall_ms <= 0.0
        || !report.eval_tps.is_finite()
        || report.eval_tps <= 0.0
        || report.eval_tps.to_bits() != (127000.0 / report.wall_ms).to_bits()
        || report.final_logits_checksum != checksum_f32(&expected_logits)
        || report.ratio128_layer3_checksum != checksum_f32(&expected_layer3)
        || report.ratio128_layer5_checksum != checksum_f32(&expected_layer5)
    {
        return Err(Error::invalid(
            "position-127 decoder report has inconsistent frontier metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{POSITION127_DECODER_PROBE_SCHEMA}\",\n  \"classification\": \"diagnostic\",\n  \"fixture\": \"{}\",\n  \"bootstrap_prompt_token\": 36662,\n  \"committed_tokens\": [",
        report.fixture_id,
    )?;
    for (index, token) in report.committed_tokens.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(output, "{token}")?;
    }
    write!(
        output,
        "],\n  \"frontier\": {{\"evaluated_positions\": {}, \"final_position\": {}, \"raw_cache_capacity_rows\": {}, \"compressed_cache_capacity_rows\": {}}},\n  \"schedule\": {{\"command_buffers_per_position\": {}, \"host_waits_per_position\": {}}},\n  \"timing\": {{\"wall_ms\": {:.6}, \"evaluated_positions_per_second\": {:.6}, \"correctness_readback_in_interval\": false}},\n  \"checksums\": {{\"final_logits\": {}, \"layer3_ratio128_row0\": {}, \"layer5_ratio128_row0\": {}}},\n  \"transcript_token_match\": true,\n  \"final_logits_c0_bitwise_match\": true,\n  \"integrated_ratio128_rows_c0_bitwise_match\": true,\n  \"paired_protocol_eligible\": false,\n  \"paired_protocol_blocker\": \"captured initial state; cold prefill is covered by a separate control\"\n}}\n",
        report.evaluated_positions,
        report.final_position,
        report.cache_capacity_rows,
        report.compressed_cache_capacity_rows,
        report.command_buffers_per_position,
        report.host_waits_per_position,
        report.wall_ms,
        report.eval_tps,
        report.final_logits_checksum,
        report.ratio128_layer3_checksum,
        report.ratio128_layer5_checksum,
    )?;
    Ok(())
}

pub fn write_cold_prefill_decoder_probe_json<W: Write>(
    output: &mut W,
    report: &ColdPrefillDecoderProbeReport,
) -> Result<()> {
    let (expected_tokens, expected_logits) = position127_decoder_fixture()?;
    let expected_prefill_logits = cold_prefill_fixture()?;
    let expected_layer3 = decode_f32_fixture(
        LAYER3_POS127_COMPRESSED_KV_BYTES,
        "layer-3 position-127 compressed KV row",
    )?;
    let expected_layer5 = decode_f32_fixture(
        LAYER5_POS127_COMPRESSED_KV_BYTES,
        "layer-5 position-127 compressed KV row",
    )?;
    if report.fixture_id != COLD_PREFILL_FIXTURE_ID
        || report.prompt_token != 36662
        || report.committed_tokens != expected_tokens
        || !report.prefill_wall_ms.is_finite()
        || report.prefill_wall_ms <= 0.0
        || report.prefill_logits_checksum != checksum_f32(&expected_prefill_logits)
        || !report.decode_wall_ms.is_finite()
        || report.decode_wall_ms <= 0.0
        || !report.decode_tps.is_finite()
        || report.decode_tps <= 0.0
        || report.decode_tps.to_bits() != (127000.0 / report.decode_wall_ms).to_bits()
        || report.final_logits_checksum != checksum_f32(&expected_logits)
        || report.ratio128_layer3_checksum != checksum_f32(&expected_layer3)
        || report.ratio128_layer5_checksum != checksum_f32(&expected_layer5)
        || lowest_id_argmax(&expected_prefill_logits)? != expected_tokens[0]
    {
        return Err(Error::invalid(
            "cold-prefill decoder report has inconsistent frontier metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{COLD_PREFILL_DECODER_PROBE_SCHEMA}\",\n  \"classification\": \"diagnostic\",\n  \"fixture\": \"{}\",\n  \"prompt_token\": {},\n  \"committed_tokens\": [",
        report.fixture_id,
        report.prompt_token,
    )?;
    for (index, token) in report.committed_tokens.iter().enumerate() {
        if index != 0 {
            write!(output, ",")?;
        }
        write!(output, "{token}")?;
    }
    write!(
        output,
        "],\n  \"state_initialization\": \"cold-empty-kv-and-compressor-state\",\n  \"prefill\": {{\"tokens\": 1, \"wall_ms\": {:.6}, \"sampling_in_interval\": true, \"full_logits_c0_bitwise_match\": true, \"selected_token\": 201}},\n  \"decode\": {{\"evaluated_positions\": 127, \"wall_ms\": {:.6}, \"evaluated_positions_per_second\": {:.6}}},\n  \"final_logits_c0_bitwise_match\": true,\n  \"integrated_ratio128_rows_c0_bitwise_match\": true,\n  \"captured_initial_state_used\": false,\n  \"paired_protocol_eligible\": false,\n  \"paired_protocol_blocker\": \"native batched prefill and ratio-4 sparse indexed decode are not implemented\"\n}}\n",
        report.prefill_wall_ms,
        report.decode_wall_ms,
        report.decode_tps,
    )?;
    Ok(())
}

pub fn write_prefill_frontier_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillFrontierProbeReport,
) -> Result<()> {
    let (_, batch_logits, decode_logits) = prefill_frontier_2048_fixture()?;
    let batch_logits_mismatch_count = batch_logits
        .iter()
        .zip(&decode_logits)
        .filter(|(batch, decode)| batch.to_bits() != decode.to_bits())
        .count() as u32;
    let batch_logits_max_abs_error = batch_logits
        .iter()
        .zip(&decode_logits)
        .map(|(batch, decode)| (batch - decode).abs())
        .fold(0.0_f32, f32::max);
    if report.fixture_id != PREFILL_FRONTIER_2048_FIXTURE_ID
        || report.context_capacity != 2048
        || report.prompt_tokens != 2048
        || report.final_position != 2047
        || report.raw_cache_capacity_rows != 128
        || report.ratio4_compressed_capacity_rows != 514
        || report.ratio128_compressed_capacity_rows != 18
        || report.selected_token != 15342
        || !report.wall_ms.is_finite()
        || report.wall_ms <= 0.0
        || !report.prefill_tps.is_finite()
        || report.prefill_tps.to_bits() != (2_048_000.0 / report.wall_ms).to_bits()
        || report.decode_logits_checksum != checksum_f32(&decode_logits)
        || report.batch_logits_mismatch_count != batch_logits_mismatch_count
        || report.batch_logits_max_abs_error.to_bits() != batch_logits_max_abs_error.to_bits()
    {
        return Err(Error::invalid(
            "2K prefill frontier report has inconsistent metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_FRONTIER_PROBE_SCHEMA}\",\n  \"classification\": \"diagnostic\",\n  \"fixture\": \"{}\",\n  \"state_initialization\": \"cold-empty-kv-and-compressor-state\",\n  \"context_capacity\": {},\n  \"prefill\": {{\"tokens\": {}, \"final_position\": {}, \"wall_ms\": {:.6}, \"tokens_per_second\": {:.6}, \"sampling_in_interval\": true, \"selected_token\": {}}},\n  \"cache\": {{\"raw_ring_rows_per_layer\": {}, \"ratio4_compressed_rows_per_layer\": {}, \"ratio128_compressed_rows_per_layer\": {}}},\n  \"decode_replay_logits_c0_bitwise_match\": true,\n  \"batched_prefill_logits_c0_bitwise_match\": false,\n  \"batched_prefill_drift\": {{\"mismatch_count\": {}, \"max_abs_error\": {:.9}}},\n  \"decode_logits_checksum\": {},\n  \"captured_initial_state_used\": false,\n  \"paired_protocol_eligible\": false,\n  \"paired_protocol_blocker\": \"native batched prefill arithmetic and ratio-4 sparse indexed decode are required\"\n}}\n",
        report.fixture_id,
        report.context_capacity,
        report.prompt_tokens,
        report.final_position,
        report.wall_ms,
        report.prefill_tps,
        report.selected_token,
        report.raw_cache_capacity_rows,
        report.ratio4_compressed_capacity_rows,
        report.ratio128_compressed_capacity_rows,
        report.batch_logits_mismatch_count,
        report.batch_logits_max_abs_error,
        report.decode_logits_checksum,
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

pub fn write_sparse_indexed_attention_probe_json<W: Write>(
    output: &mut W,
    report: &SparseIndexedAttentionProbeReport,
) -> Result<()> {
    if report.fixture_id != SPARSE_INDEXED_ATTENTION_DEFAULT_FIXTURE_ID
        || report.position != 4099
        || report.compressed_rows != 1025
        || report.raw_rows != 128
        || report.top_k != 512
        || report.diagnostic_threshold_override != 0
        || report.pinned_default_threshold != 1024
        || report.first_default_sparse_rows != 1025
        || report.dispatches != 11
        || report.wrapped_model_ranges != 3
        || report.pointer_matches != 3
        || report.split_count != 12
    {
        return Err(Error::invalid(
            "sparse indexed-attention report has inconsistent boundary metadata",
        ));
    }
    let diagnostic_override = if report.diagnostic_threshold_override == 0 {
        "null".to_owned()
    } else {
        report.diagnostic_threshold_override.to_string()
    };
    write!(
        output,
        "{{\n  \"schema\": \"{SPARSE_INDEXED_ATTENTION_PROBE_SCHEMA}\",\n  \"classification\": \"default-threshold-layer-segment\",\n  \"fixture\": \"{}\",\n  \"boundary\": {{\"layer\": 2, \"position\": {}, \"compressed_rows\": {}, \"raw_rows\": {}, \"top_k\": {}}},\n  \"threshold\": {{\"diagnostic_override\": {}, \"pinned_default\": {}, \"first_default_sparse_rows\": {}}},\n  \"schedule\": {{\"dispatches\": {}, \"split_count\": {}, \"score_kernel\": \"kernel_dsv4_indexer_score_one_direct\", \"topk_kernels\": [\"kernel_argsort_f32_i32_desc\", \"kernel_argsort_merge_f32_i32_desc\"], \"attention_kernels\": [\"kernel_dsv4_indexed_mixed_attention_heads8_split\", \"kernel_dsv4_indexed_mixed_attention_heads8_split_reduce\"]}},\n  \"mapping\": {{\"wrapped_model_ranges\": {}, \"pointer_matches\": {}}},\n  \"timing\": {{\"wall_ms\": {:.6}, \"gpu_ms\": {:.6}}},\n  \"checksums\": {{\"indexer_q\": {}, \"indexer_weights\": {}, \"indexer_scores\": {}, \"indexer_topk\": {}, \"kqv_out\": {}, \"kqv_back\": {}}},\n  \"c0_bitwise_match\": true,\n  \"default_threshold_boundary_claim\": true,\n  \"complete_decode_claim\": false,\n  \"output_logits_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        report.fixture_id,
        report.position,
        report.compressed_rows,
        report.raw_rows,
        report.top_k,
        diagnostic_override,
        report.pinned_default_threshold,
        report.first_default_sparse_rows,
        report.dispatches,
        report.split_count,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.indexer_q_checksum,
        report.indexer_weights_checksum,
        report.indexer_scores_checksum,
        report.indexer_topk_checksum,
        report.kqv_out_checksum,
        report.kqv_back_checksum,
    )?;
    Ok(())
}

pub fn write_retained_sparse_boundary_probe_json<W: Write>(
    output: &mut W,
    report: &RetainedSparseBoundaryProbeReport,
) -> Result<()> {
    if report.fixture_id != RETAINED_SPARSE_BOUNDARY_FIXTURE_ID
        || report.token != 0
        || report.layer != 2
        || report.position != 4099
        || report.raw_rows != 128
        || report.seeded_raw_rows != 127
        || report.compressed_rows != 1025
        || report.seeded_compressed_rows != 1024
        || report.top_k != 512
        || report.sort_blocks != 2
        || report.merge_passes != 1
        || report.topk_work_width != 513
        || report.dispatches != 54
        || report.wrapped_model_ranges != 35
        || report.pointer_matches != 35
        || report.exact_tensor_checks != 16
    {
        return Err(Error::invalid(
            "retained sparse-boundary report has inconsistent metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{RETAINED_SPARSE_BOUNDARY_PROBE_SCHEMA}\",\n  \"classification\": \"retained-state-c0-control\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"boundary\": {{\"layer\": {}, \"position\": {}, \"raw_rows\": {}, \"compressed_rows\": {}, \"top_k\": {}}},\n  \"seed\": {{\"incoming_hc\": true, \"raw_rows\": {}, \"compressed_rows\": {}, \"recurrent_attention_state\": true, \"recurrent_indexer_state\": true}},\n  \"schedule\": {{\"dispatches\": {}, \"sort_blocks\": {}, \"merge_passes\": {}, \"topk_work_width\": {}, \"two_block_topk_merge\": true, \"same_step_compressed_row_commit\": true, \"indexed_attention_splits\": 12}},\n  \"mapping\": {{\"wrapped_model_ranges\": {}, \"pointer_matches\": {}}},\n  \"timing\": {{\"wall_ms\": {:.6}, \"gpu_ms\": {:.6}}},\n  \"checksums\": {{\"q_current\": {}, \"compressed_kv\": {}, \"compressed_indexer\": {}, \"indexer_scores\": {}, \"indexer_topk\": {}, \"kqv_out\": {}, \"kqv_back\": {}, \"attention_hc\": {}, \"selected_experts\": {}, \"final_hc\": {}}},\n  \"exact_tensor_checks\": {},\n  \"c0_bitwise_match\": true,\n  \"retained_layer_execution_claim\": true,\n  \"preceding_layers_execution_claim\": false,\n  \"complete_decoder_claim\": false,\n  \"output_logits_claim\": false,\n  \"complete_layer_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        report.fixture_id,
        report.token,
        report.layer,
        report.position,
        report.raw_rows,
        report.compressed_rows,
        report.top_k,
        report.seeded_raw_rows,
        report.seeded_compressed_rows,
        report.dispatches,
        report.sort_blocks,
        report.merge_passes,
        report.topk_work_width,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.q_current_checksum,
        report.compressed_kv_checksum,
        report.compressed_indexer_checksum,
        report.indexer_scores_checksum,
        report.indexer_topk_checksum,
        report.kqv_out_checksum,
        report.kqv_back_checksum,
        report.attention_hc_checksum,
        report.selected_experts_checksum,
        report.final_hc_checksum,
        report.exact_tensor_checks,
    )?;
    Ok(())
}

pub fn write_retained_sparse_multimerge_probe_json<W: Write>(
    output: &mut W,
    report: &RetainedSparseBoundaryProbeReport,
) -> Result<()> {
    if report.fixture_id != RETAINED_SPARSE_MULTIMERGE_FIXTURE_ID
        || report.token != 381
        || report.layer != 2
        || report.position != 8195
        || report.raw_rows != 128
        || report.seeded_raw_rows != 127
        || report.compressed_rows != 2049
        || report.seeded_compressed_rows != 2048
        || report.top_k != 512
        || report.sort_blocks != 3
        || report.merge_passes != 2
        || report.topk_work_width != 1025
        || report.dispatches != 55
        || report.wrapped_model_ranges != 35
        || report.pointer_matches != 35
        || report.exact_tensor_checks != 40
    {
        return Err(Error::invalid(
            "retained sparse multimerge report has inconsistent metadata",
        ));
    }
    write!(
        output,
        "{{\n  \"schema\": \"{RETAINED_SPARSE_MULTIMERGE_PROBE_SCHEMA}\",\n  \"classification\": \"retained-complete-layer-c0-control\",\n  \"fixture\": \"{}\",\n  \"token\": {},\n  \"boundary\": {{\"layer\": {}, \"position\": {}, \"raw_rows\": {}, \"compressed_rows\": {}, \"top_k\": {}}},\n  \"seed\": {{\"incoming_hc\": true, \"raw_rows\": {}, \"compressed_rows\": {}, \"recurrent_attention_state\": true, \"recurrent_indexer_state\": true}},\n  \"schedule\": {{\"dispatches\": {}, \"sort_blocks\": {}, \"merge_passes\": {}, \"topk_work_width\": {}, \"ping_pong_workspace\": true, \"same_step_compressed_row_commit\": true, \"indexed_attention_splits\": 12}},\n  \"mapping\": {{\"wrapped_model_ranges\": {}, \"pointer_matches\": {}}},\n  \"timing\": {{\"wall_ms\": {:.6}, \"gpu_ms\": {:.6}}},\n  \"checksums\": {{\"q_current\": {}, \"compressed_kv\": {}, \"compressed_indexer\": {}, \"indexer_scores\": {}, \"indexer_topk\": {}, \"kqv_out\": {}, \"kqv_back\": {}, \"attention_hc\": {}, \"selected_experts\": {}, \"final_hc\": {}}},\n  \"exact_tensor_checks\": {},\n  \"c0_bitwise_match\": true,\n  \"retained_layer_execution_claim\": true,\n  \"repeated_merge_boundary_claim\": true,\n  \"complete_layer_claim\": true,\n  \"preceding_layers_execution_claim\": false,\n  \"complete_decoder_claim\": false,\n  \"output_logits_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        report.fixture_id,
        report.token,
        report.layer,
        report.position,
        report.raw_rows,
        report.compressed_rows,
        report.top_k,
        report.seeded_raw_rows,
        report.seeded_compressed_rows,
        report.dispatches,
        report.sort_blocks,
        report.merge_passes,
        report.topk_work_width,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.q_current_checksum,
        report.compressed_kv_checksum,
        report.compressed_indexer_checksum,
        report.indexer_scores_checksum,
        report.indexer_topk_checksum,
        report.kqv_out_checksum,
        report.kqv_back_checksum,
        report.attention_hc_checksum,
        report.selected_experts_checksum,
        report.final_hc_checksum,
        report.exact_tensor_checks,
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
        || report.cache_capacity_rows != 128
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

pub fn write_prefill_q8_boundary_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillQ8BoundaryProbeReport,
) -> Result<()> {
    output.write_all(b"{\n  \"schema\": \"")?;
    output.write_all(PREFILL_Q8_BOUNDARY_PROBE_SCHEMA.as_bytes())?;
    output.write_all(b"\",\n  \"fixture\": ")?;
    crate::artifact::write_json_string(output, report.fixture_id)?;
    output.write_all(b",\n  \"tensor\": ")?;
    crate::artifact::write_json_string(output, &report.tensor_name)?;
    write!(
        output,
        ",\n  \"batch_kernel\": \"kernel_mul_mm_q8_0_f32\",\n  \"decode_control_kernel\": \"kernel_mul_mv_q8_0_f32\",\n  \"shape\": {{\n    \"rows\": {},\n    \"input_elements_per_row\": {},\n    \"output_elements_per_row\": {}\n  }},\n  \"batch_dispatch\": {{\n    \"threads_per_threadgroup\": [{}, 1, 1],\n    \"threadgroups\": [{}, {}, 1],\n    \"threadgroup_memory_bytes\": 6144\n  }},\n  \"mapping\": {{\n    \"no_copy_pointer_match\": {}\n  }},\n  \"timing\": {{\n    \"batch_wall_ms\": {:.6},\n    \"batch_gpu_ms\": {:.6},\n    \"decode_wall_ms\": {:.6},\n    \"decode_gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"input\": {},\n    \"batch_output\": {},\n    \"decode_output\": {}\n  }},\n  \"arithmetic_boundary\": {{\n    \"shared_final_input_row\": true,\n    \"final_row_mismatches\": {},\n    \"final_row_elements\": {},\n    \"max_abs_error\": {:.9}\n  }},\n  \"batch_c0_bitwise_match\": true,\n  \"decode_control_c0_bitwise_match\": true,\n  \"full_prefill_claim\": false\n}}\n",
        report.rows,
        report.input_elements_per_row,
        report.output_elements_per_row,
        report.batch_threads_per_threadgroup,
        report.batch_threadgroups_x,
        report.batch_threadgroups_y,
        report.no_copy_pointer_match,
        report.batch_wall_ms,
        report.batch_gpu_ms,
        report.decode_wall_ms,
        report.decode_gpu_ms,
        report.input_checksum,
        report.batch_output_checksum,
        report.decode_output_checksum,
        report.final_row_mismatches,
        report.output_elements_per_row,
        report.final_row_max_abs_error,
    )?;
    Ok(())
}

pub fn write_prefill_qkv_boundary_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillQkvBoundaryProbeReport,
) -> Result<()> {
    output.write_all(b"{\n  \"schema\": \"")?;
    output.write_all(PREFILL_QKV_BOUNDARY_PROBE_SCHEMA.as_bytes())?;
    output.write_all(b"\",\n  \"fixture\": ")?;
    crate::artifact::write_json_string(output, report.fixture_id)?;
    write!(
        output,
        ",\n  \"shape\": {{\n    \"rows\": {},\n    \"position_start\": {},\n    \"position_end\": {}\n  }},\n  \"schedule\": {{\n    \"dispatches\": {},\n    \"q8_kernel\": \"kernel_mul_mm_q8_0_f32\",\n    \"qkv_norm_kernel\": \"kernel_dsv4_qkv_rms_norm_f32_4\",\n    \"q_head_kernel\": \"kernel_dsv4_head_rms_norm_rope_tail_f32\"\n  }},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"attn_norm\": {},\n    \"q_lora\": {},\n    \"q_lora_norm\": {},\n    \"kv_raw\": {},\n    \"kv_norm\": {},\n    \"q_raw\": {},\n    \"q_current\": {}\n  }},\n  \"c0_bitwise_match\": true,\n  \"native_batch_schedule\": true,\n  \"full_prefill_claim\": false\n}}\n",
        report.rows,
        report.position_start,
        report.position_start + report.rows as u32 - 1,
        report.dispatches,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.checksums[0],
        report.checksums[1],
        report.checksums[2],
        report.checksums[3],
        report.checksums[4],
        report.checksums[5],
        report.checksums[6],
    )?;
    Ok(())
}

pub fn write_prefill_layer0_boundary_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayer0BoundaryProbeReport,
) -> Result<()> {
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYER0_BOUNDARY_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    crate::artifact::write_json_string(output, report.ingress_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.qkv_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.kv_state_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.attention_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.attention_output_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.ffn_output_fixture_id)?;
    write!(
        output,
        "],\n  \"shape\": {{\n    \"rows\": {},\n    \"position_start\": {},\n    \"position_end\": {}\n  }},\n  \"input_boundary\": \"token_ids\",\n  \"output_boundary\": \"layer0_hc_ffn_post\",\n  \"schedule\": {{\n    \"dispatches\": {},\n    \"embedding_kernel\": \"kernel_get_rows_f16\",\n    \"hc_projection_kernel\": \"kernel_mul_mm_f16_f32\",\n    \"hc_norm_kernel\": \"kernel_dsv4_hc_split_weighted_sum_norm4\",\n    \"q8_kernel\": \"kernel_mul_mm_q8_0_f32\",\n    \"qkv_norm_kernel\": \"kernel_dsv4_qkv_rms_norm_f32_4\",\n    \"q_head_kernel\": \"kernel_dsv4_head_rms_norm_rope_tail_f32\",\n    \"kv_rope_kernel\": \"kernel_dsv4_rope_tail_f32\",\n    \"kv_fp8_kernel\": \"kernel_dsv4_fp8_kv_quantize_f32\",\n    \"cache_conversion_kernels\": [\"kernel_cpy_contig_f32_f16_4\", \"kernel_cpy_contig_f16_f32_4\"],\n    \"attention_block_kernel\": \"kernel_flash_attn_ext_blk\",\n    \"attention_kernel\": \"kernel_flash_attn_ext_f16_dk512_dv512\",\n    \"attention_inverse_rope_kernel\": \"kernel_dsv4_rope_tail_f32\",\n    \"attention_output_map_kernel\": \"kernel_mul_mm_id_map0_ne20_8\",\n    \"attention_output_low_kernel\": \"kernel_mul_mm_id_q8_0_f32\",\n    \"attention_output_kernel\": \"kernel_mul_mm_q8_0_f32\",\n    \"attention_hc_post_kernel\": \"kernel_dsv4_hc_expand4\",\n    \"router_matmul_kernel\": \"kernel_mul_mm_f16_f32\",\n    \"router_schedule\": \"decomposed_m1_batch\",\n    \"routed_map_kernel\": \"kernel_mul_mm_id_map0_ne20_6\",\n    \"routed_pair_kernel\": \"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16\",\n    \"routed_down_kernel\": \"kernel_mul_mm_id_q2_K_f16\",\n    \"routed_sum_kernel\": \"kernel_dsv4_moe_sum6_f32\",\n    \"shared_swiglu_kernel\": \"kernel_swiglu_flat_f32\",\n    \"ffn_hc_post_kernel\": \"kernel_dsv4_hc_expand4\"\n  }},\n  \"raw_cache\": {{\n    \"capacity_rows\": {},\n    \"target_row_start\": {},\n    \"target_row_end\": {},\n    \"guard_rows\": [0, {}],\n    \"guard_rows_intact\": true\n  }},\n  \"attention\": {{\n    \"query_rows\": {},\n    \"kv_rows\": {},\n    \"captured_kv_prefix_rows\": {},\n    \"live_kv_rows\": {},\n    \"window\": 128\n  }},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"token_ids\": {},\n    \"hc_collapsed\": {},\n    \"attn_norm\": {},\n    \"q_lora\": {},\n    \"q_lora_norm\": {},\n    \"kv_raw\": {},\n    \"kv_norm\": {},\n    \"q_raw\": {},\n    \"q_current\": {},\n    \"kv_rope\": {},\n    \"kv_current\": {},\n    \"raw_cache\": {},\n    \"full_kv_current\": {},\n    \"kqv_output\": {},\n    \"kqv_back\": {},\n    \"attention_low\": {},\n    \"attention_out\": {},\n    \"attention_hc_post\": {},\n    \"ffn_current\": {},\n    \"ffn_norm\": {},\n    \"router_logits\": {},\n    \"router_probs\": {},\n    \"router_selected\": {},\n    \"router_weights\": {},\n    \"routed_mid\": {},\n    \"routed_out\": {},\n    \"shared_out\": {},\n    \"ffn_hc_post\": {}\n  }},\n  \"c0_bitwise_match\": true,\n  \"continuous_command_buffer\": true,\n  \"native_batch_schedule\": true,\n  \"guarded_cache_mutation\": true,\n  \"rectangular_attention_read\": true,\n  \"full_layer0_final_tile_claim\": true,\n  \"full_prefill_claim\": false\n}}\n",
        report.rows,
        report.position_start,
        report.position_start + report.rows as u32 - 1,
        report.dispatches,
        report.raw_cache_rows,
        report.raw_cache_target_row,
        report.raw_cache_target_row + report.rows as u32 - 1,
        report.raw_cache_guard_rows - 1,
        report.rows,
        report.attention_kv_rows,
        report.attention_kv_prefix_rows,
        report.rows,
        report.wrapped_model_ranges,
        report.pointer_matches,
        report.wall_ms,
        report.gpu_ms,
        report.checksums[0],
        report.checksums[1],
        report.checksums[2],
        report.checksums[3],
        report.checksums[4],
        report.checksums[5],
        report.checksums[6],
        report.checksums[7],
        report.checksums[8],
        report.checksums[9],
        report.checksums[10],
        report.checksums[11],
        report.checksums[12],
        report.checksums[13],
        report.checksums[14],
        report.checksums[15],
        report.checksums[16],
        report.checksums[17],
        report.checksums[18],
        report.checksums[19],
        report.checksums[20],
        report.checksums[21],
        report.checksums[22],
        report.checksums[23],
        report.checksums[24],
        report.checksums[25],
        report.checksums[26],
        report.checksums[27],
    )?;
    Ok(())
}

pub fn write_prefill_layers01_boundary_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers01BoundaryProbeReport,
) -> Result<()> {
    let layer0 = &report.layer0;
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS01_BOUNDARY_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    for (index, fixture) in [
        layer0.ingress_fixture_id,
        layer0.qkv_fixture_id,
        layer0.kv_state_fixture_id,
        layer0.attention_fixture_id,
        layer0.attention_output_fixture_id,
        layer0.ffn_output_fixture_id,
        report.layer1_fixture_id,
    ]
    .iter()
    .enumerate()
    {
        if index != 0 {
            output.write_all(b", ")?;
        }
        crate::artifact::write_json_string(output, fixture)?;
    }
    write!(
        output,
        "],\n  \"shape\": {{\n    \"rows\": {},\n    \"position_start\": {},\n    \"position_end\": {}\n  }},\n  \"input_boundary\": \"token_ids\",\n  \"output_boundary\": \"layer1_q_lora\",\n  \"schedule\": {{\n    \"dispatches\": {},\n    \"layer0_dispatches\": 43,\n    \"layer1_ingress_dispatches\": 4,\n    \"layer1_hc_norm_kernel\": \"kernel_rms_norm_f32_4\",\n    \"layer1_hc_projection_kernel\": \"kernel_mul_mm_f16_f32\",\n    \"layer1_hc_collapse_kernel\": \"kernel_dsv4_hc_split_weighted_sum_norm4\",\n    \"layer1_q_a_kernel\": \"kernel_mul_mm_q8_0_f32\"\n  }},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"checksums\": {{\n    \"layer0_hc_ffn_post\": {},\n    \"layer1_hc_attn_pre\": {},\n    \"layer1_attn_norm\": {},\n    \"layer1_q_lora\": {}\n  }},\n  \"c0_bitwise_match\": true,\n  \"continuous_command_buffer\": true,\n  \"direct_hc_handoff\": true,\n  \"native_batch_schedule\": true,\n  \"full_layer1_claim\": false,\n  \"full_prefill_claim\": false\n}}\n",
        layer0.rows,
        layer0.position_start,
        layer0.position_start + layer0.rows as u32 - 1,
        layer0.dispatches,
        layer0.wrapped_model_ranges,
        layer0.pointer_matches,
        layer0.wall_ms,
        layer0.gpu_ms,
        layer0.checksums[27],
        report.checksums[0],
        report.checksums[1],
        report.checksums[2],
    )?;
    Ok(())
}

pub fn write_prefill_layers01_complete_boundary_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers01CompleteBoundaryProbeReport,
) -> Result<()> {
    let layer0 = &report.layers01.layer0;
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS01_COMPLETE_BOUNDARY_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    for (index, fixture) in [
        layer0.ingress_fixture_id,
        layer0.qkv_fixture_id,
        layer0.kv_state_fixture_id,
        layer0.attention_fixture_id,
        layer0.attention_output_fixture_id,
        layer0.ffn_output_fixture_id,
        report.layers01.layer1_fixture_id,
        report.complete_fixture_id,
    ]
    .iter()
    .enumerate()
    {
        if index != 0 {
            output.write_all(b", ")?;
        }
        crate::artifact::write_json_string(output, fixture)?;
    }
    write!(
        output,
        "],\n  \"shape\": {{\n    \"rows\": {},\n    \"position_start\": {},\n    \"position_end\": {}\n  }},\n  \"input_boundary\": \"token_ids\",\n  \"output_boundary\": \"layer1_hc_ffn_post\",\n  \"schedule\": {{\n    \"dispatches\": {},\n    \"layer0_dispatches\": 43,\n    \"layer1_dispatches\": 41,\n    \"attention_kv_rows\": 2048,\n    \"attention_window\": 128\n  }},\n  \"mapping\": {{\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {}\n  }},\n  \"retained\": {{\n    \"produced_fp32_values\": 12878208,\n    \"selected_expert_ids\": 384\n  }},\n  \"timing\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"layer1_ingress_checksums\": [",
        layer0.rows,
        layer0.position_start,
        layer0.position_start + layer0.rows as u32 - 1,
        layer0.dispatches,
        layer0.wrapped_model_ranges,
        layer0.pointer_matches,
        layer0.wall_ms,
        layer0.gpu_ms,
    )?;
    for (index, checksum) in report.layers01.checksums.iter().enumerate() {
        if index != 0 {
            output.write_all(b", ")?;
        }
        write!(output, "{checksum}")?;
    }
    output.write_all(b"],\n  \"layer1_completion_checksums\": [")?;
    for (index, checksum) in report.checksums.iter().enumerate() {
        if index != 0 {
            output.write_all(b", ")?;
        }
        write!(output, "{checksum}")?;
    }
    output.write_all(
        b"],\n  \"c0_bitwise_match\": true,\n  \"continuous_command_buffer\": true,\n  \"direct_hc_handoff\": true,\n  \"native_batch_schedule\": true,\n  \"full_layer1_claim\": true,\n  \"full_prefill_claim\": false\n}\n",
    )?;
    Ok(())
}

pub fn write_prefill_layers01_row_coverage_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers01RowCoverageProbeReport,
) -> Result<()> {
    let final_layer0 = &report.final_tile.layers01.layer0;
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS01_ROW_COVERAGE_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    crate::artifact::write_json_string(output, report.previous_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.final_tile.complete_fixture_id)?;
    write!(
        output,
        "],\n  \"coverage\": {{\n    \"position_start\": {},\n    \"position_end\": {},\n    \"rows\": 64,\n    \"tile_rows\": 32,\n    \"tiles\": 2\n  }},\n  \"previous_tile\": {{\n    \"position_start\": {},\n    \"position_end\": {},\n    \"dispatches\": {},\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {},\n    \"raw_cache_target_row\": {},\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"checksums\": [",
        report.previous_position_start,
        final_layer0.position_start + final_layer0.rows as u32 - 1,
        report.previous_position_start,
        report.previous_position_end,
        report.previous_dispatches,
        report.previous_wrapped_model_ranges,
        report.previous_pointer_matches,
        report.previous_raw_cache_target_row,
        report.previous_wall_ms,
        report.previous_gpu_ms,
    )?;
    for (index, checksum) in report.previous_checksums.iter().enumerate() {
        if index != 0 {
            output.write_all(b", ")?;
        }
        write!(output, "{checksum}")?;
    }
    write!(
        output,
        "],\n    \"c0_bitwise_match\": true\n  }},\n  \"final_tile\": {{\n    \"position_start\": {},\n    \"position_end\": {},\n    \"dispatches\": {},\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {},\n    \"raw_cache_target_row\": {},\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"c0_bitwise_match\": true\n  }},\n  \"command_buffers\": 2,\n  \"captured_prefix_per_tile\": true,\n  \"chained_live_kv_between_tiles\": false,\n  \"arbitrary_tile_position_claim\": true,\n  \"complete_layers01_tile_claim\": true,\n  \"full_prefill_claim\": false\n}}\n",
        final_layer0.position_start,
        final_layer0.position_start + final_layer0.rows as u32 - 1,
        final_layer0.dispatches,
        final_layer0.wrapped_model_ranges,
        final_layer0.pointer_matches,
        final_layer0.raw_cache_target_row,
        final_layer0.wall_ms,
        final_layer0.gpu_ms,
    )?;
    Ok(())
}

pub fn write_prefill_layers01_live_kv_chain_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers01LiveKvChainProbeReport,
) -> Result<()> {
    let tiles = &report.tiles;
    let final_layer0 = &tiles.final_tile.layers01.layer0;
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS01_LIVE_KV_CHAIN_PROBE_SCHEMA}\",\n  \"coverage\": {{\n    \"position_start\": {},\n    \"position_end\": {},\n    \"rows\": 64,\n    \"tile_rows\": 32,\n    \"tiles\": 2\n  }},\n  \"first_tile\": {{\n    \"position_start\": {},\n    \"position_end\": {},\n    \"dispatches\": {},\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {},\n    \"retained_kv_rows\": {},\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"c0_bitwise_match\": true\n  }},\n  \"final_tile\": {{\n    \"position_start\": {},\n    \"position_end\": {},\n    \"dispatches\": {},\n    \"wrapped_model_ranges\": {},\n    \"pointer_matches\": {},\n    \"retained_kv_rows\": {},\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"c0_bitwise_match\": true\n  }},\n  \"persistent_metal_context\": true,\n  \"command_buffers\": 2,\n  \"inter_tile_host_waits\": 1,\n  \"captured_prefix_first_tile\": true,\n  \"captured_prefix_final_tile\": false,\n  \"retained_prefix_oracle_validation\": true,\n  \"chained_live_kv_between_tiles\": true,\n  \"single_command_buffer\": false,\n  \"complete_layers01_tile_claim\": true,\n  \"full_prefill_claim\": false\n}}\n",
        tiles.previous_position_start,
        final_layer0.position_start + final_layer0.rows as u32 - 1,
        tiles.previous_position_start,
        tiles.previous_position_end,
        tiles.previous_dispatches,
        tiles.previous_wrapped_model_ranges,
        tiles.previous_pointer_matches,
        report.retained_kv_rows_after_first_tile,
        tiles.previous_wall_ms,
        tiles.previous_gpu_ms,
        final_layer0.position_start,
        final_layer0.position_start + final_layer0.rows as u32 - 1,
        final_layer0.dispatches,
        final_layer0.wrapped_model_ranges,
        final_layer0.pointer_matches,
        report.retained_kv_rows_after_final_tile,
        final_layer0.wall_ms,
        final_layer0.gpu_ms,
    )?;
    Ok(())
}

pub fn write_prefill_layers01_live_kv_loop_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers01LiveKvLoopProbeReport,
) -> Result<()> {
    if report.tiles.len() != 64
        || report.tiles.first().map(|tile| tile.position_start) != Some(0)
        || report.tiles.last().map(|tile| tile.position_start) != Some(2016)
        || report.tiles.iter().enumerate().any(|(index, tile)| {
            tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != 84
                || tile.wrapped_model_ranges != 49
                || tile.pointer_matches != 49
        })
        || report.final_tile.layers01.layer0.position_start != 2016
    {
        return Err(Error::invalid(
            "prefill live-KV loop report must contain all 64 contiguous tiles",
        ));
    }
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS01_LIVE_KV_LOOP_PROBE_SCHEMA}\",\n  \"coverage\": {{\n    \"position_start\": 0,\n    \"position_end\": 2047,\n    \"rows\": 2048,\n    \"tile_rows\": 32,\n    \"tiles\": 64,\n    \"layers\": [0, 1]\n  }},\n  \"tiles\": [\n"
    )?;
    for (index, tile) in report.tiles.iter().enumerate() {
        write!(
            output,
            "    {{\"position_start\": {}, \"position_end\": {}, \"retained_kv_rows\": {}, \"dispatches\": {}, \"wrapped_model_ranges\": {}, \"pointer_matches\": {}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"layer01_kv_c0_bitwise_match\": true}}{}\n",
            tile.position_start,
            tile.position_start + tile.rows as u32 - 1,
            tile.position_start + tile.rows as u32,
            tile.dispatches,
            tile.wrapped_model_ranges,
            tile.pointer_matches,
            tile.wall_ms,
            tile.gpu_ms,
            if index + 1 == report.tiles.len() { "" } else { "," },
        )?;
    }
    write!(
        output,
        "  ],\n  \"timing\": {{\"summed_wall_ms\": {:.6}, \"summed_gpu_ms\": {:.6}}},\n  \"persistent_metal_context\": true,\n  \"command_buffers\": 64,\n  \"inter_tile_host_waits\": 63,\n  \"captured_kv_seed_rows\": 0,\n  \"retained_prefix_oracle_validation\": true,\n  \"all_layer01_kv_c0_bitwise_match\": true,\n  \"previous_tile_full_outputs_c0_bitwise_match\": true,\n  \"final_tile_full_outputs_c0_bitwise_match\": true,\n  \"all_tile_full_outputs_c0_bitwise_match\": false,\n  \"single_command_buffer\": false,\n  \"complete_layers01_prefill_claim\": false,\n  \"complete_model_prefill_claim\": false\n}}\n",
        total_wall_ms, total_gpu_ms,
    )?;
    Ok(())
}

pub fn write_prefill_layers012_kvnorm_loop_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers012KvnormLoopProbeReport,
) -> Result<()> {
    if report.tiles.len() != 64
        || report.layer2_kvnorm_checksums.len() != 64
        || report.tiles.first().map(|tile| tile.position_start) != Some(0)
        || report.tiles.last().map(|tile| tile.position_start) != Some(2016)
        || report.tiles.iter().enumerate().any(|(index, tile)| {
            tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != 90
                || tile.wrapped_model_ranges != 57
                || tile.pointer_matches != 57
        })
        || report.final_tile.layers01.layer0.position_start != 2016
    {
        return Err(Error::invalid(
            "prefill layers-0/1/2 KVnorm report must contain all 64 contiguous tiles",
        ));
    }
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS012_KVNORM_LOOP_PROBE_SCHEMA}\",\n  \"fixture\": "
    )?;
    crate::artifact::write_json_string(output, report.layer2_fixture_id)?;
    output.write_all(
        b",\n  \"coverage\": {\n    \"position_start\": 0,\n    \"position_end\": 2047,\n    \"rows\": 2048,\n    \"tile_rows\": 32,\n    \"tiles\": 64,\n    \"complete_layers\": [0, 1],\n    \"downstream_layer\": 2,\n    \"output_boundary\": \"layer2_KVnorm\"\n  },\n  \"tiles\": [\n",
    )?;
    for (index, (tile, checksum)) in report
        .tiles
        .iter()
        .zip(&report.layer2_kvnorm_checksums)
        .enumerate()
    {
        write!(
            output,
            "    {{\"position_start\": {}, \"position_end\": {}, \"retained_layer01_kv_rows\": {}, \"dispatches\": {}, \"wrapped_model_ranges\": {}, \"pointer_matches\": {}, \"layer2_kvnorm_checksum\": {}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"layer01_kv_c0_bitwise_match\": true, \"layer2_kvnorm_c0_bitwise_match\": true}}{}\n",
            tile.position_start,
            tile.position_start + tile.rows as u32 - 1,
            tile.position_start + tile.rows as u32,
            tile.dispatches,
            tile.wrapped_model_ranges,
            tile.pointer_matches,
            checksum,
            tile.wall_ms,
            tile.gpu_ms,
            if index + 1 == report.tiles.len() { "" } else { "," },
        )?;
    }
    write!(
        output,
        "  ],\n  \"timing\": {{\"summed_wall_ms\": {:.6}, \"summed_gpu_ms\": {:.6}}},\n  \"schedule\": {{\n    \"dispatches_per_tile\": 90,\n    \"layer01_dispatches_per_tile\": 84,\n    \"layer2_kvnorm_dispatches_per_tile\": 6,\n    \"wrapped_model_ranges_per_tile\": 57,\n    \"layer2_hc_norm_kernel\": \"kernel_rms_norm_f32_4\",\n    \"layer2_hc_projection_kernel\": \"kernel_mul_mm_f16_f32\",\n    \"layer2_hc_collapse_kernel\": \"kernel_dsv4_hc_split_weighted_sum_norm4\",\n    \"layer2_q_a_kernel\": \"kernel_mul_mm_q8_0_f32\",\n    \"layer2_kv_kernel\": \"kernel_mul_mm_q8_0_f32\",\n    \"layer2_qkv_norm_kernel\": \"kernel_dsv4_qkv_rms_norm_f32_4\"\n  }},\n  \"persistent_metal_context\": true,\n  \"command_buffers\": 64,\n  \"inter_tile_host_waits\": 63,\n  \"captured_kv_seed_rows\": 0,\n  \"retained_prefix_oracle_validation\": true,\n  \"all_layer01_kv_c0_bitwise_match\": true,\n  \"all_layer2_kvnorm_c0_bitwise_match\": true,\n  \"all_layer1_outputs_downstream_validated\": true,\n  \"complete_layers01_prefill_claim\": true,\n  \"complete_layer2_prefill_claim\": false,\n  \"compressed_layer2_attention_claim\": false,\n  \"complete_model_prefill_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        total_wall_ms, total_gpu_ms,
    )?;
    Ok(())
}

pub fn write_prefill_layers012_kv_state_loop_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers012KvStateLoopProbeReport,
) -> Result<()> {
    if report.tiles.len() != 64
        || report.layer2_checksums.len() != 64
        || report.tiles.first().map(|tile| tile.position_start) != Some(0)
        || report.tiles.last().map(|tile| tile.position_start) != Some(2016)
        || report.tiles.iter().enumerate().any(|(index, tile)| {
            tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != 92
                || tile.wrapped_model_ranges != 57
                || tile.pointer_matches != 57
        })
        || report.final_tile.layers01.layer0.position_start != 2016
    {
        return Err(Error::invalid(
            "prefill layers-0/1/2 KV-state report must contain all 64 contiguous tiles",
        ));
    }
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS012_KV_STATE_LOOP_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    crate::artifact::write_json_string(output, report.layer2_kvnorm_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.layer2_kv_state_fixture_id)?;
    output.write_all(
        b"],\n  \"coverage\": {\n    \"position_start\": 0,\n    \"position_end\": 2047,\n    \"rows\": 2048,\n    \"tile_rows\": 32,\n    \"tiles\": 64,\n    \"complete_layers\": [0, 1],\n    \"downstream_layer\": 2,\n    \"output_boundary\": \"layer2_KVcur\"\n  },\n  \"tiles\": [\n",
    )?;
    for (index, (tile, checksums)) in report
        .tiles
        .iter()
        .zip(&report.layer2_checksums)
        .enumerate()
    {
        write!(
            output,
            "    {{\"position_start\": {}, \"position_end\": {}, \"retained_layer012_kv_rows\": {}, \"dispatches\": {}, \"wrapped_model_ranges\": {}, \"pointer_matches\": {}, \"layer2_kvnorm_checksum\": {}, \"layer2_kvrope_checksum\": {}, \"layer2_kvcur_checksum\": {}, \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"layer012_kv_c0_bitwise_match\": true}}{}\n",
            tile.position_start,
            tile.position_start + tile.rows as u32 - 1,
            tile.position_start + tile.rows as u32,
            tile.dispatches,
            tile.wrapped_model_ranges,
            tile.pointer_matches,
            checksums[0],
            checksums[1],
            checksums[2],
            tile.wall_ms,
            tile.gpu_ms,
            if index + 1 == report.tiles.len() { "" } else { "," },
        )?;
    }
    write!(
        output,
        "  ],\n  \"timing\": {{\"summed_wall_ms\": {:.6}, \"summed_gpu_ms\": {:.6}}},\n  \"schedule\": {{\n    \"dispatches_per_tile\": 92,\n    \"layer01_dispatches_per_tile\": 84,\n    \"layer2_dispatches_per_tile\": 8,\n    \"wrapped_model_ranges_per_tile\": 57,\n    \"layer2_kv_rope_kernel\": \"kernel_dsv4_rope_tail_f32\",\n    \"layer2_kv_finalize_kernel\": \"kernel_dsv4_fp8_kv_quantize_f32\"\n  }},\n  \"persistent_metal_context\": true,\n  \"command_buffers\": 64,\n  \"inter_tile_host_waits\": 63,\n  \"captured_kv_seed_rows\": 0,\n  \"retained_prefix_oracle_validation\": true,\n  \"all_layer01_kv_c0_bitwise_match\": true,\n  \"all_layer2_kvnorm_c0_bitwise_match\": true,\n  \"all_layer2_kvrope_c0_bitwise_match\": true,\n  \"all_layer2_kvcur_c0_bitwise_match\": true,\n  \"complete_layer2_raw_kv_state_claim\": true,\n  \"complete_layer2_compressed_kv_state_claim\": false,\n  \"complete_layer2_attention_claim\": false,\n  \"complete_layer2_prefill_claim\": false,\n  \"complete_model_prefill_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        total_wall_ms, total_gpu_ms,
    )?;
    Ok(())
}

pub fn write_prefill_layers012_compressor_loop_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers012CompressorLoopProbeReport,
) -> Result<()> {
    if report.tiles.len() != 64
        || report.layer2_checksums.len() != 64
        || report.layer2_compressor_checksums.len() != 64
        || report.tiles.iter().enumerate().any(|(index, tile)| {
            tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != (if index == 63 { 122 } else { 118 })
                || tile.wrapped_model_ranges != 65
                || tile.pointer_matches != 65
        })
        || report.final_tile.layers01.layer0.position_start != 2016
    {
        return Err(Error::invalid(
            "prefill layers-0/1/2 compressor report must contain all 64 contiguous tiles",
        ));
    }
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS012_COMPRESSOR_LOOP_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    crate::artifact::write_json_string(output, report.layer2_kvnorm_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.layer2_kv_state_fixture_id)?;
    output.write_all(b", ")?;
    crate::artifact::write_json_string(output, report.layer2_compressor_fixture_id)?;
    output.write_all(
        b"],\n  \"coverage\": {\"position_start\": 0, \"position_end\": 2047, \"rows\": 2048, \"tile_rows\": 32, \"tiles\": 64, \"complete_layers\": [0, 1], \"downstream_layer\": 2, \"output_boundary\": \"layer2 paired ratio-4 compressors\"},\n  \"tiles\": [\n",
    )?;
    for (index, ((tile, kv), compressors)) in report
        .tiles
        .iter()
        .zip(&report.layer2_checksums)
        .zip(&report.layer2_compressor_checksums)
        .enumerate()
    {
        write!(
            output,
            "    {{\"position_start\": {}, \"position_end\": {}, \"retained_layer012_kv_rows\": {}, \"retained_layer2_compressed_rows\": {}, \"dispatches\": {}, \"wrapped_model_ranges\": {}, \"pointer_matches\": {}, \"layer2_kv_checksums\": [{}, {}, {}], \"layer2_compressor_checksums\": [{}, {}, {}, {}, {}, {}], \"wall_ms\": {:.6}, \"gpu_ms\": {:.6}, \"layer2_compressor_c0_bitwise_match\": true}}{}\n",
            tile.position_start,
            tile.position_start + tile.rows as u32 - 1,
            tile.position_start + tile.rows as u32,
            (tile.position_start + tile.rows as u32) / 4,
            tile.dispatches,
            tile.wrapped_model_ranges,
            tile.pointer_matches,
            kv[0], kv[1], kv[2],
            compressors[0], compressors[1], compressors[2],
            compressors[3], compressors[4], compressors[5],
            tile.wall_ms,
            tile.gpu_ms,
            if index + 1 == report.tiles.len() { "" } else { "," },
        )?;
    }
    write!(
        output,
        "  ],\n  \"timing\": {{\"summed_wall_ms\": {:.6}, \"summed_gpu_ms\": {:.6}}},\n  \"schedule\": {{\"dispatches_per_regular_tile\": 118, \"dispatches_final_tile\": 122, \"layer01_dispatches_per_tile\": 84, \"layer2_raw_kv_dispatches_per_tile\": 8, \"layer2_compressor_batch_projection_dispatches_per_tile\": 4, \"layer2_compressor_batch_dispatches_per_tile\": 22, \"layer2_compressor_tail_projection_dispatches_final_tile\": 4, \"wrapped_model_ranges_per_tile\": 65, \"projection_kernel\": \"kernel_mul_mm_f16_f32\", \"tail_projection_kernel\": \"kernel_mul_mv_ext_f16_f32_r1_4\", \"pool_kernel\": \"DwarfStar-equivalent fused 8-row softmax pool\", \"store_kernel\": \"kernel_dsv4_compressor_store_one\", \"ratio4_shift_kernel\": \"kernel_dsv4_ratio4_shift_f32\"}},\n  \"persistent_metal_context\": true,\n  \"command_buffers\": 64,\n  \"inter_tile_host_waits\": 63,\n  \"captured_kv_seed_rows\": 0,\n  \"retained_prefix_oracle_validation\": true,\n  \"all_layer2_compressed_rows_c0_bitwise_match\": true,\n  \"final_layer2_compressor_states_c0_bitwise_match\": true,\n  \"complete_layer2_raw_kv_state_claim\": true,\n  \"complete_layer2_paired_compressor_claim\": true,\n  \"complete_layer2_attention_claim\": false,\n  \"complete_layer2_ffn_claim\": false,\n  \"complete_layer2_prefill_claim\": false,\n  \"complete_model_prefill_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        total_wall_ms, total_gpu_ms,
    )?;
    Ok(())
}

pub fn write_prefill_layers012_attention_loop_probe_json<W: Write>(
    output: &mut W,
    report: &PrefillLayers012AttentionLoopProbeReport,
) -> Result<()> {
    let expected = prefill_layer2_attention_fixture()?;
    let layer3_kv_state = prefill_layer3_kv_state_final_tile_fixture()?;
    let _layer3_compressor = prefill_layer3_compressor_fixture()?;
    let _layer3_attention = prefill_layer3_attention_fixture()?;
    let _layer3_complete = prefill_layer3_complete_final_tile_fixture()?;
    let layer4_qkv = prefill_layer4_qkv_final_tile_fixture()?;
    let layer4_compressor = prefill_layer4_compressor_fixture()?;
    let _layer4_attention = prefill_layer4_attention_fixture()?;
    let _layer4_complete = prefill_layer4_complete_final_tile_fixture()?;
    let layer5_qkv = prefill_layer5_qkv_final_tile_fixture()?;
    let layer5_compressor = prefill_layer5_compressor_fixture()?;
    let _layer5_attention = prefill_layer5_attention_fixture()?;
    let _layer5_complete = prefill_layer5_complete_final_tile_fixture()?;
    let layer6_qkv = prefill_layer6_qkv_final_tile_fixture()?;
    let layer6_compressor = prefill_layer6_compressor_fixture()?;
    let _layer6_attention = prefill_layer6_attention_fixture()?;
    let _layer6_complete = prefill_layer6_complete_final_tile_fixture()?;
    let layer7_qkv = prefill_layer7_qkv_final_tile_fixture()?;
    let layer7_compressor = prefill_layer7_compressor_fixture()?;
    let _layer7_attention = prefill_layer7_attention_fixture()?;
    let _layer7_complete = prefill_layer7_complete_final_tile_fixture()?;
    let layer8_qkv = prefill_layer8_qkv_final_tile_fixture()?;
    let layer8_compressor = prefill_layer8_compressor_fixture()?;
    let _layer8_attention = prefill_layer8_attention_fixture()?;
    let _layer8_complete = prefill_layer8_complete_final_tile_fixture()?;
    if report.compressor.tiles.len() != 64
        || report.attention_fixture_id != PREFILL_LAYER2_ATTENTION_FIXTURE_ID
        || report.attention_hc_fixture_id != PREFILL_LAYER2_COMPLETE_FIXTURE_ID
        || report.layer3_ingress_fixture_id != PREFILL_LAYER3_INGRESS_FIXTURE_ID
        || report.layer3_kv_state_fixture_id != PREFILL_LAYER3_KV_STATE_FIXTURE_ID
        || report.layer3_compressor_fixture_id != PREFILL_LAYER3_COMPRESSOR_FIXTURE_ID
        || report.layer3_attention_fixture_id != PREFILL_LAYER3_ATTENTION_FIXTURE_ID
        || report.layer3_complete_fixture_id != PREFILL_LAYER3_COMPLETE_FIXTURE_ID
        || report.layer4_qkv_fixture_id != PREFILL_LAYER4_QKV_FIXTURE_ID
        || report.layer4_compressor_fixture_id != PREFILL_LAYER4_COMPRESSOR_FIXTURE_ID
        || report.layer4_attention_fixture_id != PREFILL_LAYER4_ATTENTION_FIXTURE_ID
        || report.layer4_complete_fixture_id != PREFILL_LAYER4_COMPLETE_FIXTURE_ID
        || report.layer5_qkv_fixture_id != PREFILL_LAYER5_QKV_FIXTURE_ID
        || report.layer5_compressor_fixture_id != PREFILL_LAYER5_COMPRESSOR_FIXTURE_ID
        || report.layer5_attention_fixture_id != PREFILL_LAYER5_ATTENTION_FIXTURE_ID
        || report.layer5_complete_fixture_id != PREFILL_LAYER5_COMPLETE_FIXTURE_ID
        || report.layer6_qkv_fixture_id != PREFILL_LAYER6_QKV_FIXTURE_ID
        || report.layer6_compressor_fixture_id != PREFILL_LAYER6_COMPRESSOR_FIXTURE_ID
        || report.layer6_attention_fixture_id != PREFILL_LAYER6_ATTENTION_FIXTURE_ID
        || report.layer6_complete_fixture_id != PREFILL_LAYER6_COMPLETE_FIXTURE_ID
        || report.layer7_qkv_fixture_id != PREFILL_LAYER7_QKV_FIXTURE_ID
        || report.layer7_compressor_fixture_id != PREFILL_LAYER7_COMPRESSOR_FIXTURE_ID
        || report.layer7_attention_fixture_id != PREFILL_LAYER7_ATTENTION_FIXTURE_ID
        || report.layer7_complete_fixture_id != PREFILL_LAYER7_COMPLETE_FIXTURE_ID
        || report.layer8_qkv_fixture_id != PREFILL_LAYER8_QKV_FIXTURE_ID
        || report.layer8_compressor_fixture_id != PREFILL_LAYER8_COMPRESSOR_FIXTURE_ID
        || report.layer8_attention_fixture_id != PREFILL_LAYER8_ATTENTION_FIXTURE_ID
        || report.layer8_complete_fixture_id != PREFILL_LAYER8_COMPLETE_FIXTURE_ID
        || report.rows != 2048
        || report.raw_kv_rows != 2048
        || report.compressed_kv_rows != 512
        || report.layer3_compressed_kv_rows != 16
        || report.dispatches != 383
        || report.wrapped_model_ranges != 196
        || report.pointer_matches != 196
        || report.output_checksum != checksum_f32(&expected)
        || report.after_attention_hc_checksum != PREFILL_LAYER2_HC_ATTN_POST_FULL_CHECKSUM
        || report.after_ffn_hc_checksum != PREFILL_LAYER2_HC_FFN_POST_FULL_CHECKSUM
        || report.layer3_hc_attn_pre_checksum != PREFILL_LAYER3_HC_ATTN_PRE_FULL_CHECKSUM
        || report.layer3_attn_norm_checksum != PREFILL_LAYER3_ATTN_NORM_FULL_CHECKSUM
        || report.layer3_q_lora_checksum != PREFILL_LAYER3_Q_LORA_FULL_CHECKSUM
        || report.layer3_q_lora_norm_checksum != PREFILL_LAYER3_Q_LORA_NORM_FULL_CHECKSUM
        || report.layer3_kv_raw_checksum != PREFILL_LAYER3_KV_RAW_FULL_CHECKSUM
        || report.layer3_kv_norm_checksum != PREFILL_LAYER3_KV_NORM_FULL_CHECKSUM
        || report.layer3_q_raw_final_tile_checksum != checksum_f32(&layer3_kv_state[3])
        || report.layer3_q_cur_final_tile_checksum != checksum_f32(&layer3_kv_state[4])
        || report.layer3_kv_rope_checksum != PREFILL_LAYER3_KV_ROPE_FULL_CHECKSUM
        || report.layer3_kv_cur_checksum != PREFILL_LAYER3_KV_CUR_FULL_CHECKSUM
        || report.layer3_attn_compressed_checksum != PREFILL_LAYER3_ATTN_COMPRESSED_CHECKSUM
        || report.layer3_attn_state_kv_checksum != PREFILL_LAYER3_ATTN_STATE_KV_CHECKSUM
        || report.layer3_attn_state_score_checksum != PREFILL_LAYER3_ATTN_STATE_SCORE_CHECKSUM
        || report.layer3_attention_output_checksum != PREFILL_LAYER3_ATTENTION_OUTPUT_CHECKSUM
        || report.layer3_after_attention_hc_checksum != PREFILL_LAYER3_HC_ATTN_POST_FULL_CHECKSUM
        || report.layer3_after_ffn_hc_checksum != PREFILL_LAYER3_HC_FFN_POST_FULL_CHECKSUM
        || report.layer4_qkv_checksums != layer4_qkv.each_ref().map(|tensor| checksum_f32(tensor))
        || report.layer4_compressor_checksums
            != layer4_compressor
                .each_ref()
                .map(|tensor| checksum_f32(tensor))
        || report.layer4_attention_output_checksum != PREFILL_LAYER4_ATTENTION_OUTPUT_CHECKSUM
        || report.layer4_after_attention_hc_checksum != PREFILL_LAYER4_HC_ATTN_POST_FULL_CHECKSUM
        || report.layer4_after_ffn_hc_checksum != PREFILL_LAYER4_HC_FFN_POST_FULL_CHECKSUM
        || report.layer5_qkv_checksums != layer5_qkv.each_ref().map(|tensor| checksum_f32(tensor))
        || report.layer5_compressor_checksums
            != [
                checksum_f32(&layer5_compressor.0),
                checksum_f32(&layer5_compressor.1),
                checksum_i32(&layer5_compressor.2),
            ]
        || report.layer5_attention_output_checksum != PREFILL_LAYER5_ATTENTION_OUTPUT_CHECKSUM
        || report.layer5_after_attention_hc_checksum != PREFILL_LAYER5_HC_ATTN_POST_FULL_CHECKSUM
        || report.layer5_after_ffn_hc_checksum != PREFILL_LAYER5_HC_FFN_POST_FULL_CHECKSUM
        || report.layer6_qkv_checksums != layer6_qkv.each_ref().map(|tensor| checksum_f32(tensor))
        || report.layer6_compressor_checksums
            != layer6_compressor
                .each_ref()
                .map(|tensor| checksum_f32(tensor))
        || report.layer6_attention_output_checksum != PREFILL_LAYER6_ATTENTION_OUTPUT_CHECKSUM
        || report.layer6_after_attention_hc_checksum != PREFILL_LAYER6_HC_ATTN_POST_FULL_CHECKSUM
        || report.layer6_after_ffn_hc_checksum != PREFILL_LAYER6_HC_FFN_POST_FULL_CHECKSUM
        || report.layer7_qkv_checksums != layer7_qkv.each_ref().map(|tensor| checksum_f32(tensor))
        || report.layer7_compressor_checksums
            != [
                checksum_f32(&layer7_compressor.0),
                checksum_f32(&layer7_compressor.1),
                checksum_i32(&layer7_compressor.2),
            ]
        || report.layer7_attention_output_checksum != PREFILL_LAYER7_ATTENTION_OUTPUT_CHECKSUM
        || report.layer7_after_attention_hc_checksum != PREFILL_LAYER7_HC_ATTN_POST_FULL_CHECKSUM
        || report.layer7_after_ffn_hc_checksum != PREFILL_LAYER7_HC_FFN_POST_FULL_CHECKSUM
        || report.layer8_qkv_checksums != layer8_qkv.each_ref().map(|tensor| checksum_f32(tensor))
        || report.layer8_compressor_checksums
            != layer8_compressor
                .each_ref()
                .map(|tensor| checksum_f32(tensor))
        || report.layer8_attention_output_checksum != PREFILL_LAYER8_ATTENTION_OUTPUT_CHECKSUM
        || report.layer8_after_attention_hc_checksum != PREFILL_LAYER8_HC_ATTN_POST_FULL_CHECKSUM
        || report.layer8_after_ffn_hc_checksum != PREFILL_LAYER8_HC_FFN_POST_FULL_CHECKSUM
        || !report.wall_ms.is_finite()
        || report.wall_ms <= 0.0
        || !report.gpu_ms.is_finite()
        || report.gpu_ms < 0.0
    {
        return Err(Error::invalid(
            "prefill layers-0/1/2 attention report has inconsistent metadata",
        ));
    }
    let tile_wall_ms: f64 = report
        .compressor
        .tiles
        .iter()
        .map(|tile| tile.wall_ms)
        .sum();
    let tile_gpu_ms: f64 = report.compressor.tiles.iter().map(|tile| tile.gpu_ms).sum();
    write!(
        output,
        "{{\n  \"schema\": \"{PREFILL_LAYERS012_ATTENTION_LOOP_PROBE_SCHEMA}\",\n  \"fixtures\": ["
    )?;
    for (index, fixture) in [
        report.compressor.layer2_kvnorm_fixture_id,
        report.compressor.layer2_kv_state_fixture_id,
        report.compressor.layer2_compressor_fixture_id,
        report.attention_fixture_id,
        report.attention_hc_fixture_id,
        report.layer3_ingress_fixture_id,
        report.layer3_kv_state_fixture_id,
        report.layer3_compressor_fixture_id,
        report.layer3_attention_fixture_id,
        report.layer3_complete_fixture_id,
        report.layer4_qkv_fixture_id,
        report.layer4_compressor_fixture_id,
        report.layer4_attention_fixture_id,
        report.layer4_complete_fixture_id,
        report.layer5_qkv_fixture_id,
        report.layer5_compressor_fixture_id,
        report.layer5_attention_fixture_id,
        report.layer5_complete_fixture_id,
        report.layer6_qkv_fixture_id,
        report.layer6_compressor_fixture_id,
        report.layer6_attention_fixture_id,
        report.layer6_complete_fixture_id,
        report.layer7_qkv_fixture_id,
        report.layer7_compressor_fixture_id,
        report.layer7_attention_fixture_id,
        report.layer7_complete_fixture_id,
        report.layer8_qkv_fixture_id,
        report.layer8_compressor_fixture_id,
        report.layer8_attention_fixture_id,
        report.layer8_complete_fixture_id,
    ]
    .iter()
    .enumerate()
    {
        if index != 0 {
            output.write_all(b", ")?;
        }
        crate::artifact::write_json_string(output, fixture)?;
    }
    let rendered = format!(
        "],\n  \"coverage\": {{\"position_start\": 0, \"position_end\": 2047, \"rows\": {}, \"complete_layers\": [0, 1, 2, 3], \"downstream_layer\": 4, \"output_boundary\": \"layer4_paired_compressors\"}},\n  \"mixed_attention\": {{\"raw_kv_rows\": {}, \"compressed_kv_rows\": {}, \"raw_window\": 128, \"compressor_ratio\": 4, \"sparse_indexer_topk\": false, \"dense_compressed_limit_rows\": 512, \"layer3_compressor_ratio\": 128, \"layer3_compressed_rows\": {}, \"layer7_compressor_ratio\": 128, \"layer7_compressed_rows\": 16}},\n  \"schedule\": {{\"tile_command_buffers\": 64, \"terminal_command_buffers\": 1, \"terminal_dispatches\": {}, \"wrapped_terminal_model_ranges\": {}, \"query_projection_kernel\": \"kernel_mul_mm_q8_0_f32\", \"query_rope_kernel\": \"kernel_dsv4_head_rms_norm_rope_tail_f32\", \"attention_kernel\": \"kernel_flash_attn_ext_f16_dk512_dv512\", \"inverse_rope_kernel\": \"kernel_dsv4_rope_tail_f32\", \"output_low_kernel\": \"kernel_mul_mm_id_q8_0_f32\", \"output_kernel\": \"kernel_mul_mm_q8_0_f32\", \"attention_hc_post_kernel\": \"kernel_dsv4_hc_expand4\", \"ffn_router\": \"token-hash batch\", \"routed_experts\": \"fused IQ2_XXS pair-SwiGLU plus Q2_K down\", \"shared_expert\": \"Q8_0 gate/up/down\", \"ffn_hc_post_kernel\": \"kernel_dsv4_hc_expand4\", \"layer3_attention_ingress_dispatches\": 4, \"layer3_qkv_state_dispatches\": 6, \"layer3_ratio128_compressor_dispatches\": 7, \"layer3_dense_attention_dispatches\": 9, \"layer3_ffn_dispatches\": 21, \"layer4_qkv_dispatches\": 10, \"layer4_paired_compressor_dispatches\": 30, \"layer4_compressor_projection_kernel\": \"kernel_mul_mm_f16_f32 x4\", \"layer4_compressor_ratio4_kernels\": [\"score plus APE\", \"replay pack\", \"softmax pool\", \"weighted RMSNorm\", \"compressed RoPE\", \"E4M3FN/indexer QAT\", \"tail projection and recurrent-state refresh\"]}},\n  \"timing\": {{\"tile_summed_wall_ms\": {:.6}, \"tile_summed_gpu_ms\": {:.6}, \"terminal_wall_ms\": {:.6}, \"terminal_gpu_ms\": {:.6}}},\n  \"checksums\": {{\"layer2_attention_output\": {}, \"layer2_attention_hc_post\": {}, \"layer2_ffn_hc_post\": {}, \"layer3_hc_attn_pre\": {}, \"layer3_attn_norm\": {}, \"layer3_q_lora\": {}, \"layer3_q_lora_norm\": {}, \"layer3_kv_raw\": {}, \"layer3_kv_norm\": {}, \"layer3_q_raw_final_tile\": {}, \"layer3_q_cur_final_tile\": {}, \"layer3_kv_rope\": {}, \"layer3_kv_cur\": {}, \"layer3_attn_compressed\": {}, \"layer3_attn_state_kv\": {}, \"layer3_attn_state_score_bits\": {}, \"layer3_attention_output\": {}, \"layer3_attention_hc_post\": {}, \"layer3_ffn_hc_post\": {}, \"layer4_qkv_final_tiles\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}], \"layer4_paired_compressors\": [{}, {}, {}, {}, {}, {}]}},\n  \"persistent_metal_context\": true,\n  \"all_layer2_compressed_rows_c0_bitwise_match\": true,\n  \"layer2_attention_output_c0_bitwise_match\": true,\n  \"layer2_attention_hc_post_c0_bitwise_match\": true,\n  \"layer2_ffn_hc_post_c0_bitwise_match\": true,\n  \"layer3_attention_ingress_c0_bitwise_match\": true,\n  \"layer3_qkv_state_c0_bitwise_match\": true,\n  \"layer3_ratio128_compressor_c0_bitwise_match\": true,\n  \"layer3_attention_output_c0_bitwise_match\": true,\n  \"layer3_attention_hc_post_c0_bitwise_match\": true,\n  \"layer3_ffn_biased_topk_c0_bitwise_match\": true,\n  \"layer3_ffn_outputs_c0_bitwise_match\": true,\n  \"layer3_ffn_hc_post_c0_bitwise_match\": true,\n  \"layer4_qkv_state_c0_bitwise_match\": true,\n  \"layer4_paired_compressors_c0_bitwise_match\": true,\n  \"complete_layer2_dense_mixed_attention_claim\": true,\n  \"complete_layer2_attention_hc_post_claim\": true,\n  \"complete_layer2_ffn_claim\": true,\n  \"complete_layer2_prefill_claim\": true,\n  \"complete_layer3_qkv_state_claim\": true,\n  \"complete_layer3_ratio128_compressor_claim\": true,\n  \"complete_layer3_dense_mixed_attention_claim\": true,\n  \"complete_layer3_attention_hc_post_claim\": true,\n  \"complete_layer3_ffn_claim\": true,\n  \"complete_layer3_prefill_claim\": true,\n  \"complete_layer4_qkv_state_claim\": true,\n  \"complete_layer4_paired_compressor_claim\": true,\n  \"complete_layer4_prefill_claim\": false,\n  \"sparse_ratio4_decode_claim\": false,\n  \"complete_model_prefill_claim\": false,\n  \"throughput_claim\": false\n}}\n",
        report.rows,
        report.raw_kv_rows,
        report.compressed_kv_rows,
        report.layer3_compressed_kv_rows,
        report.dispatches,
        report.wrapped_model_ranges,
        tile_wall_ms,
        tile_gpu_ms,
        report.wall_ms,
        report.gpu_ms,
        report.output_checksum,
        report.after_attention_hc_checksum,
        report.after_ffn_hc_checksum,
        report.layer3_hc_attn_pre_checksum,
        report.layer3_attn_norm_checksum,
        report.layer3_q_lora_checksum,
        report.layer3_q_lora_norm_checksum,
        report.layer3_kv_raw_checksum,
        report.layer3_kv_norm_checksum,
        report.layer3_q_raw_final_tile_checksum,
        report.layer3_q_cur_final_tile_checksum,
        report.layer3_kv_rope_checksum,
        report.layer3_kv_cur_checksum,
        report.layer3_attn_compressed_checksum,
        report.layer3_attn_state_kv_checksum,
        report.layer3_attn_state_score_checksum,
        report.layer3_attention_output_checksum,
        report.layer3_after_attention_hc_checksum,
        report.layer3_after_ffn_hc_checksum,
        report.layer4_qkv_checksums[0],
        report.layer4_qkv_checksums[1],
        report.layer4_qkv_checksums[2],
        report.layer4_qkv_checksums[3],
        report.layer4_qkv_checksums[4],
        report.layer4_qkv_checksums[5],
        report.layer4_qkv_checksums[6],
        report.layer4_qkv_checksums[7],
        report.layer4_qkv_checksums[8],
        report.layer4_qkv_checksums[9],
        report.layer4_compressor_checksums[0],
        report.layer4_compressor_checksums[1],
        report.layer4_compressor_checksums[2],
        report.layer4_compressor_checksums[3],
        report.layer4_compressor_checksums[4],
        report.layer4_compressor_checksums[5],
    );
    let checksum_tail = format!(
        "], \"layer4_attention_output\": {}, \"layer4_attention_hc_post\": {}, \"layer4_ffn_hc_post\": {}, \"layer5_qkv_final_tiles\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}], \"layer5_ratio128_compressor\": [{}, {}, {}], \"layer5_attention_output\": {}, \"layer5_attention_hc_post\": {}, \"layer5_ffn_hc_post\": {}, \"layer6_qkv_final_tiles\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}], \"layer6_paired_compressors\": [{}, {}, {}, {}, {}, {}], \"layer6_attention_output\": {}, \"layer6_attention_hc_post\": {}, \"layer6_ffn_hc_post\": {}, \"layer7_qkv_final_tiles\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}], \"layer7_ratio128_compressor\": [{}, {}, {}], \"layer7_attention_output\": {}, \"layer7_attention_hc_post\": {}, \"layer7_ffn_hc_post\": {}, \"layer8_qkv_final_tiles\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}], \"layer8_paired_compressors\": [{}, {}, {}, {}, {}, {}], \"layer8_attention_output\": {}, \"layer8_attention_hc_post\": {}, \"layer8_ffn_hc_post\": {}}},\n  \"persistent_metal_context\"",
        report.layer4_attention_output_checksum,
        report.layer4_after_attention_hc_checksum,
        report.layer4_after_ffn_hc_checksum,
        report.layer5_qkv_checksums[0],
        report.layer5_qkv_checksums[1],
        report.layer5_qkv_checksums[2],
        report.layer5_qkv_checksums[3],
        report.layer5_qkv_checksums[4],
        report.layer5_qkv_checksums[5],
        report.layer5_qkv_checksums[6],
        report.layer5_qkv_checksums[7],
        report.layer5_qkv_checksums[8],
        report.layer5_qkv_checksums[9],
        report.layer5_compressor_checksums[0],
        report.layer5_compressor_checksums[1],
        report.layer5_compressor_checksums[2],
        report.layer5_attention_output_checksum,
        report.layer5_after_attention_hc_checksum,
        report.layer5_after_ffn_hc_checksum,
        report.layer6_qkv_checksums[0],
        report.layer6_qkv_checksums[1],
        report.layer6_qkv_checksums[2],
        report.layer6_qkv_checksums[3],
        report.layer6_qkv_checksums[4],
        report.layer6_qkv_checksums[5],
        report.layer6_qkv_checksums[6],
        report.layer6_qkv_checksums[7],
        report.layer6_qkv_checksums[8],
        report.layer6_qkv_checksums[9],
        report.layer6_compressor_checksums[0],
        report.layer6_compressor_checksums[1],
        report.layer6_compressor_checksums[2],
        report.layer6_compressor_checksums[3],
        report.layer6_compressor_checksums[4],
        report.layer6_compressor_checksums[5],
        report.layer6_attention_output_checksum,
        report.layer6_after_attention_hc_checksum,
        report.layer6_after_ffn_hc_checksum,
        report.layer7_qkv_checksums[0],
        report.layer7_qkv_checksums[1],
        report.layer7_qkv_checksums[2],
        report.layer7_qkv_checksums[3],
        report.layer7_qkv_checksums[4],
        report.layer7_qkv_checksums[5],
        report.layer7_qkv_checksums[6],
        report.layer7_qkv_checksums[7],
        report.layer7_qkv_checksums[8],
        report.layer7_qkv_checksums[9],
        report.layer7_compressor_checksums[0],
        report.layer7_compressor_checksums[1],
        report.layer7_compressor_checksums[2],
        report.layer7_attention_output_checksum,
        report.layer7_after_attention_hc_checksum,
        report.layer7_after_ffn_hc_checksum,
        report.layer8_qkv_checksums[0],
        report.layer8_qkv_checksums[1],
        report.layer8_qkv_checksums[2],
        report.layer8_qkv_checksums[3],
        report.layer8_qkv_checksums[4],
        report.layer8_qkv_checksums[5],
        report.layer8_qkv_checksums[6],
        report.layer8_qkv_checksums[7],
        report.layer8_qkv_checksums[8],
        report.layer8_qkv_checksums[9],
        report.layer8_compressor_checksums[0],
        report.layer8_compressor_checksums[1],
        report.layer8_compressor_checksums[2],
        report.layer8_compressor_checksums[3],
        report.layer8_compressor_checksums[4],
        report.layer8_compressor_checksums[5],
        report.layer8_attention_output_checksum,
        report.layer8_after_attention_hc_checksum,
        report.layer8_after_ffn_hc_checksum,
    );
    let rendered = rendered
        .replace(
            "\"complete_layers\": [0, 1, 2, 3], \"downstream_layer\": 4",
            "\"complete_layers\": [0, 1, 2, 3, 4], \"downstream_layer\": 5",
        )
        .replace(
            "\"output_boundary\": \"layer4_paired_compressors\"",
            "\"output_boundary\": \"layer4_attention_hc_post\"",
        )
        .replace(
            "\"output_boundary\": \"layer4_attention_hc_post\"",
            "\"output_boundary\": \"layer4_ffn_hc_post\"",
        )
        .replace(
            "\"layer4_paired_compressor_dispatches\": 30,",
            "\"layer4_paired_compressor_dispatches\": 30, \"layer4_dense_attention_dispatches\": 9,",
        )
        .replace(
            "\"layer4_dense_attention_dispatches\": 9,",
            "\"layer4_dense_attention_dispatches\": 9, \"layer4_ffn_dispatches\": 21, \"layer4_ffn_router\": \"biased top-6 batch\",",
        )
        .replace("]},\n  \"persistent_metal_context\"", &checksum_tail)
        .replace(
            "\"layer4_paired_compressors_c0_bitwise_match\": true,",
            "\"layer4_paired_compressors_c0_bitwise_match\": true,\n  \"layer4_attention_output_c0_bitwise_match\": true,\n  \"layer4_attention_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"layer4_attention_hc_post_c0_bitwise_match\": true,",
            "\"layer4_attention_hc_post_c0_bitwise_match\": true,\n  \"layer4_ffn_biased_topk_c0_bitwise_match\": true,\n  \"layer4_ffn_outputs_c0_bitwise_match\": true,\n  \"layer4_ffn_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer4_paired_compressor_claim\": true,",
            "\"complete_layer4_paired_compressor_claim\": true,\n  \"complete_layer4_dense_mixed_attention_claim\": true,\n  \"complete_layer4_attention_hc_post_claim\": true,",
        )
        .replace(
            "\"complete_layer4_attention_hc_post_claim\": true,",
            "\"complete_layer4_attention_hc_post_claim\": true,\n  \"complete_layer4_ffn_claim\": true,",
        )
        .replace(
            "\"complete_layer4_prefill_claim\": false,",
            "\"complete_layer4_prefill_claim\": true,",
        )
        .replace(
            "\"output_boundary\": \"layer4_ffn_hc_post\"",
            "\"output_boundary\": \"layer5_attention_hc_post\"",
        )
        .replace(
            "\"layer4_ffn_router\": \"biased top-6 batch\",",
            "\"layer4_ffn_router\": \"biased top-6 batch\", \"layer5_qkv_dispatches\": 10, \"layer5_ratio128_compressor_dispatches\": 7, \"layer5_dense_attention_dispatches\": 9,",
        )
        .replace(
            "\"layer4_ffn_hc_post_c0_bitwise_match\": true,",
            "\"layer4_ffn_hc_post_c0_bitwise_match\": true,\n  \"layer5_qkv_state_c0_bitwise_match\": true,\n  \"layer5_ratio128_compressor_c0_bitwise_match\": true,\n  \"layer5_attention_output_c0_bitwise_match\": true,\n  \"layer5_attention_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer4_prefill_claim\": true,",
            "\"complete_layer4_prefill_claim\": true,\n  \"complete_layer5_qkv_state_claim\": true,\n  \"complete_layer5_ratio128_compressor_claim\": true,\n  \"complete_layer5_dense_mixed_attention_claim\": true,\n  \"complete_layer5_attention_hc_post_claim\": true,\n  \"complete_layer5_prefill_claim\": false,",
        )
        .replace(
            "\"complete_layers\": [0, 1, 2, 3, 4], \"downstream_layer\": 5",
            "\"complete_layers\": [0, 1, 2, 3, 4, 5], \"downstream_layer\": null",
        )
        .replace(
            "\"output_boundary\": \"layer5_attention_hc_post\"",
            "\"output_boundary\": \"layer5_ffn_hc_post\"",
        )
        .replace(
            "\"layer5_dense_attention_dispatches\": 9,",
            "\"layer5_dense_attention_dispatches\": 9, \"layer5_ffn_dispatches\": 21, \"layer5_ffn_router\": \"biased top-6 batch\",",
        )
        .replace(
            "\"layer5_attention_hc_post_c0_bitwise_match\": true,",
            "\"layer5_attention_hc_post_c0_bitwise_match\": true,\n  \"layer5_ffn_biased_topk_c0_bitwise_match\": true,\n  \"layer5_ffn_outputs_c0_bitwise_match\": true,\n  \"layer5_ffn_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer5_attention_hc_post_claim\": true,",
            "\"complete_layer5_attention_hc_post_claim\": true,\n  \"complete_layer5_ffn_claim\": true,",
        )
        .replace(
            "\"complete_layer5_prefill_claim\": false,",
            "\"complete_layer5_prefill_claim\": true,",
        )
        .replace(
            "\"complete_layers\": [0, 1, 2, 3, 4, 5], \"downstream_layer\": null",
            "\"complete_layers\": [0, 1, 2, 3, 4, 5], \"downstream_layer\": 6",
        )
        .replace(
            "\"output_boundary\": \"layer5_ffn_hc_post\"",
            "\"output_boundary\": \"layer6_qkv_state\"",
        )
        .replace(
            "\"layer5_ffn_router\": \"biased top-6 batch\",",
            "\"layer5_ffn_router\": \"biased top-6 batch\", \"layer6_qkv_dispatches\": 10,",
        )
        .replace(
            "\"layer5_ffn_hc_post_c0_bitwise_match\": true,",
            "\"layer5_ffn_hc_post_c0_bitwise_match\": true,\n  \"layer6_qkv_state_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer5_prefill_claim\": true,",
            "\"complete_layer5_prefill_claim\": true,\n  \"complete_layer6_qkv_state_claim\": true,\n  \"complete_layer6_prefill_claim\": false,",
        )
        .replace(
            "\"output_boundary\": \"layer6_qkv_state\"",
            "\"output_boundary\": \"layer6_paired_compressors\"",
        )
        .replace(
            "\"layer6_qkv_dispatches\": 10,",
            "\"layer6_qkv_dispatches\": 10, \"layer6_paired_compressor_dispatches\": 30,",
        )
        .replace(
            "\"layer6_qkv_state_c0_bitwise_match\": true,",
            "\"layer6_qkv_state_c0_bitwise_match\": true,\n  \"layer6_paired_compressors_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer6_qkv_state_claim\": true,",
            "\"complete_layer6_qkv_state_claim\": true,\n  \"complete_layer6_paired_compressor_claim\": true,",
        )
        .replace(
            "\"output_boundary\": \"layer6_paired_compressors\"",
            "\"output_boundary\": \"layer6_attention_hc_post\"",
        )
        .replace(
            "\"layer6_paired_compressor_dispatches\": 30,",
            "\"layer6_paired_compressor_dispatches\": 30, \"layer6_dense_attention_dispatches\": 9,",
        )
        .replace(
            "\"layer6_paired_compressors_c0_bitwise_match\": true,",
            "\"layer6_paired_compressors_c0_bitwise_match\": true,\n  \"layer6_attention_output_c0_bitwise_match\": true,\n  \"layer6_attention_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer6_paired_compressor_claim\": true,",
            "\"complete_layer6_paired_compressor_claim\": true,\n  \"complete_layer6_dense_mixed_attention_claim\": true,\n  \"complete_layer6_attention_hc_post_claim\": true,",
        )
        .replace(
            "\"output_boundary\": \"layer6_attention_hc_post\"",
            "\"output_boundary\": \"layer6_ffn_hc_post\"",
        )
        .replace(
            "\"layer6_dense_attention_dispatches\": 9,",
            "\"layer6_dense_attention_dispatches\": 9, \"layer6_ffn_dispatches\": 21, \"layer6_ffn_router\": \"biased top-6 batch\",",
        )
        .replace(
            "\"layer6_attention_hc_post_c0_bitwise_match\": true,",
            "\"layer6_attention_hc_post_c0_bitwise_match\": true,\n  \"layer6_ffn_biased_topk_c0_bitwise_match\": true,\n  \"layer6_ffn_outputs_c0_bitwise_match\": true,\n  \"layer6_ffn_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer6_attention_hc_post_claim\": true,",
            "\"complete_layer6_attention_hc_post_claim\": true,\n  \"complete_layer6_ffn_claim\": true,",
        )
        .replace(
            "\"complete_layers\": [0, 1, 2, 3, 4, 5], \"downstream_layer\": 6",
            "\"complete_layers\": [0, 1, 2, 3, 4, 5, 6], \"downstream_layer\": null",
        )
        .replace(
            "\"complete_layer6_prefill_claim\": false,",
            "\"complete_layer6_prefill_claim\": true,",
        )
        .replace(
            "\"complete_layers\": [0, 1, 2, 3, 4, 5, 6], \"downstream_layer\": null",
            "\"complete_layers\": [0, 1, 2, 3, 4, 5, 6, 7], \"downstream_layer\": null",
        )
        .replace(
            "\"output_boundary\": \"layer6_ffn_hc_post\"",
            "\"output_boundary\": \"layer7_ffn_hc_post\"",
        )
        .replace(
            "\"layer6_ffn_router\": \"biased top-6 batch\",",
            "\"layer6_ffn_router\": \"biased top-6 batch\", \"layer7_qkv_dispatches\": 10, \"layer7_ratio128_compressor_dispatches\": 7, \"layer7_dense_attention_dispatches\": 9, \"layer7_ffn_dispatches\": 21, \"layer7_ffn_router\": \"biased top-6 batch\",",
        )
        .replace(
            "\"layer6_ffn_hc_post_c0_bitwise_match\": true,",
            "\"layer6_ffn_hc_post_c0_bitwise_match\": true,\n  \"layer7_qkv_state_c0_bitwise_match\": true,\n  \"layer7_ratio128_compressor_c0_bitwise_match\": true,\n  \"layer7_attention_output_c0_bitwise_match\": true,\n  \"layer7_attention_hc_post_c0_bitwise_match\": true,\n  \"layer7_ffn_biased_topk_c0_bitwise_match\": true,\n  \"layer7_ffn_outputs_c0_bitwise_match\": true,\n  \"layer7_ffn_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer6_prefill_claim\": true,",
            "\"complete_layer6_prefill_claim\": true,\n  \"complete_layer7_qkv_state_claim\": true,\n  \"complete_layer7_ratio128_compressor_claim\": true,\n  \"complete_layer7_dense_mixed_attention_claim\": true,\n  \"complete_layer7_attention_hc_post_claim\": true,\n  \"complete_layer7_ffn_claim\": true,\n  \"complete_layer7_prefill_claim\": true,",
        )
        .replace(
            "\"complete_layers\": [0, 1, 2, 3, 4, 5, 6, 7], \"downstream_layer\": null",
            "\"complete_layers\": [0, 1, 2, 3, 4, 5, 6, 7, 8], \"downstream_layer\": null",
        )
        .replace(
            "\"output_boundary\": \"layer7_ffn_hc_post\"",
            "\"output_boundary\": \"layer8_ffn_hc_post\"",
        )
        .replace(
            "\"layer7_ffn_router\": \"biased top-6 batch\",",
            "\"layer7_ffn_router\": \"biased top-6 batch\", \"layer8_qkv_dispatches\": 10, \"layer8_paired_compressor_dispatches\": 30, \"layer8_dense_attention_dispatches\": 9, \"layer8_ffn_dispatches\": 21, \"layer8_ffn_router\": \"biased top-6 batch\",",
        )
        .replace(
            "\"layer7_ffn_hc_post_c0_bitwise_match\": true,",
            "\"layer7_ffn_hc_post_c0_bitwise_match\": true,\n  \"layer8_qkv_state_c0_bitwise_match\": true,\n  \"layer8_paired_compressors_c0_bitwise_match\": true,\n  \"layer8_attention_output_c0_bitwise_match\": true,\n  \"layer8_attention_hc_post_c0_bitwise_match\": true,\n  \"layer8_ffn_biased_topk_c0_bitwise_match\": true,\n  \"layer8_ffn_outputs_c0_bitwise_match\": true,\n  \"layer8_ffn_hc_post_c0_bitwise_match\": true,",
        )
        .replace(
            "\"complete_layer7_prefill_claim\": true,",
            "\"complete_layer7_prefill_claim\": true,\n  \"complete_layer8_qkv_state_claim\": true,\n  \"complete_layer8_paired_compressor_claim\": true,\n  \"complete_layer8_dense_mixed_attention_claim\": true,\n  \"complete_layer8_attention_hc_post_claim\": true,\n  \"complete_layer8_ffn_claim\": true,\n  \"complete_layer8_prefill_claim\": true,",
        )
        .replace(
            "\"layer7_compressor_ratio\": 128, \"layer7_compressed_rows\": 16",
            "\"layer7_compressor_ratio\": 128, \"layer7_compressed_rows\": 16, \"layer8_compressor_ratio\": 4, \"layer8_compressed_rows\": 512, \"layer8_sparse_indexer_topk\": false",
        );
    output.write_all(rendered.as_bytes())?;
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

fn retained_sparse_topk_schedule(compressed_rows: u32) -> (u32, u32, u32) {
    const MAX_SORT_THREADS: u32 = 1024;
    const TOP_K: u32 = 512;
    let mut sort_threads = 1;
    while sort_threads < compressed_rows && 2 * sort_threads <= MAX_SORT_THREADS {
        sort_threads *= 2;
    }
    let sort_blocks = compressed_rows.div_ceil(sort_threads);
    let block_top_k = TOP_K.min(sort_threads);
    let work_width = if sort_blocks <= 1 {
        TOP_K.min(compressed_rows)
    } else {
        let last_block = compressed_rows - (sort_blocks - 1) * sort_threads;
        (sort_blocks - 1) * block_top_k + last_block.min(block_top_k)
    };
    let mut merge_passes = 0;
    let mut merge_len = block_top_k;
    while merge_len < work_width {
        merge_passes += 1;
        merge_len *= 2;
    }
    (sort_blocks, merge_passes, work_width)
}

fn checksum_u32(values: &[u32]) -> u64 {
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        checksum ^= u64::from(*value);
        checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
    }
    checksum
}

fn checksum_i32(values: &[i32]) -> u64 {
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        checksum ^= u64::from(*value as u32);
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

fn decode_u32_fixture(bytes: &[u8], label: &str) -> Result<Vec<u32>> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(Error::invalid(format!(
            "{label} fixture must contain nonempty little-endian U32 data"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn position127_decoder_fixture() -> Result<(Vec<u32>, Vec<f32>)> {
    let tokens = decode_u32_fixture(POSITION127_TOKEN_IDS_BYTES, "position-127 token transcript")?;
    let logits = decode_f32_fixture(POSITION127_LOGITS_BYTES, "position-127 logits")?;
    if tokens.len() != 128
        || tokens.first() != Some(&201)
        || tokens.get(1..5) != Some(&[361, 1915, 262, 1554])
        || logits.len() != 129280
        || lowest_id_argmax(&logits)? != *tokens.last().unwrap_or(&u32::MAX)
    {
        return Err(Error::invalid(
            "position-127 decoder fixture shape, prefix, or final selection is invalid",
        ));
    }
    Ok((tokens, logits))
}

fn cold_prefill_fixture() -> Result<Vec<f32>> {
    let logits = decode_f32_fixture(COLD_PREFILL_LOGITS_BYTES, "cold-prefill logits")?;
    if logits.len() != 129280 || lowest_id_argmax(&logits)? != 201 {
        return Err(Error::invalid(
            "cold-prefill fixture shape or selected token is invalid",
        ));
    }
    Ok(logits)
}

fn prefill_frontier_2048_fixture() -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    let tokens = decode_u32_fixture(
        PREFILL_FRONTIER_2048_TOKEN_IDS_BYTES,
        "2K prefill token IDs",
    )?;
    let batch_logits = decode_f32_fixture(
        PREFILL_FRONTIER_2048_BATCH_LOGITS_BYTES,
        "2K batched-prefill frontier logits",
    )?;
    let decode_logits = decode_f32_fixture(
        PREFILL_FRONTIER_2048_DECODE_LOGITS_BYTES,
        "2K decode-replay frontier logits",
    )?;
    if tokens.len() != 2048
        || tokens.first() != Some(&36662)
        || tokens.last() != Some(&895)
        || batch_logits.len() != 129280
        || decode_logits.len() != 129280
        || lowest_id_argmax(&batch_logits)? != 15342
        || lowest_id_argmax(&decode_logits)? != 15342
        || batch_logits == decode_logits
    {
        return Err(Error::invalid(
            "2K prefill fixture shape, token boundary, or selection is invalid",
        ));
    }
    Ok((tokens, batch_logits, decode_logits))
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
        4 => (
            "dwarfstar-oracle-v1-output-head-pos4",
            OUTPUT_HEAD_POS4_PRE_BYTES,
            OUTPUT_HEAD_POS4_WEIGHTS_BYTES,
            OUTPUT_HEAD_POS4_HC_BYTES,
            OUTPUT_HEAD_POS4_NORM_BYTES,
            OUTPUT_HEAD_POS4_LOGITS_BYTES,
            1554,
        ),
        _ => {
            return Err(Error::invalid(
                "output-head fixtures cover positions 1 through 4",
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

fn prefill_q8_boundary_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let input = decode_f32_fixture(PREFILL_Q8_INPUT_BYTES, "prefill Q8 input tile")?;
    let batch_output = decode_f32_fixture(
        PREFILL_Q8_BATCH_OUTPUT_BYTES,
        "prefill Q8 batch output tile",
    )?;
    let decode_output = decode_f32_fixture(
        PREFILL_Q8_DECODE_OUTPUT_BYTES,
        "prefill Q8 sequential output row",
    )?;
    if input.len() != 128 * 4096 || batch_output.len() != 128 * 1024 || decode_output.len() != 1024
    {
        return Err(Error::invalid(
            "prefill Q8 boundary fixture dimensions are invalid",
        ));
    }
    Ok((input, batch_output, decode_output))
}

fn prefill_qkv_boundary_fixture() -> Result<[Vec<f32>; 7]> {
    let tensors = [
        decode_f32_fixture(PREFILL_QKV_ATTN_NORM_BYTES, "prefill Q/KV attention norm")?,
        decode_f32_fixture(PREFILL_QKV_Q_LORA_BYTES, "prefill Q/KV Q-Lora")?,
        decode_f32_fixture(
            PREFILL_QKV_Q_LORA_NORM_BYTES,
            "prefill Q/KV normalized Q-Lora",
        )?,
        decode_f32_fixture(PREFILL_QKV_KV_RAW_BYTES, "prefill Q/KV raw KV")?,
        decode_f32_fixture(PREFILL_QKV_KV_NORM_BYTES, "prefill Q/KV normalized KV")?,
        decode_f32_fixture(PREFILL_QKV_Q_RAW_BYTES, "prefill Q/KV raw Q")?,
        decode_f32_fixture(PREFILL_QKV_Q_CUR_BYTES, "prefill Q/KV current Q")?,
    ];
    let expected = [
        32 * 4096,
        32 * 1024,
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
    ];
    if tensors
        .iter()
        .zip(expected)
        .any(|(tensor, elements)| tensor.len() != elements)
    {
        return Err(Error::invalid(
            "prefill Q/KV boundary fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_kv_state_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(PREFILL_KV_ROPE_BYTES, "prefill KV RoPE final tile")?,
        decode_f32_fixture(PREFILL_KV_CURRENT_BYTES, "prefill current KV final tile")?,
        decode_f32_fixture(PREFILL_RAW_CACHE_BYTES, "prefill raw-cache final tile")?,
    ];
    if tensors.iter().any(|tensor| tensor.len() != 32 * 512) {
        return Err(Error::invalid(
            "prefill KV-state fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_attention_read_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_ATTENTION_KV_PREFIX_BYTES,
            "prefill attention current-KV prefix",
        )?,
        decode_f32_fixture(
            PREFILL_ATTENTION_OUTPUT_BYTES,
            "prefill attention output final tile",
        )?,
        decode_f32_fixture(
            PREFILL_ATTENTION_BACK_BYTES,
            "prefill inverse-RoPE attention final tile",
        )?,
    ];
    if tensors[0].len() != 2016 * 512
        || tensors[1].len() != 32 * 64 * 512
        || tensors[2].len() != 32 * 64 * 512
    {
        return Err(Error::invalid(
            "prefill attention-read fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_attention_output_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_ATTENTION_LOW_BYTES,
            "prefill attention low final tile",
        )?,
        decode_f32_fixture(
            PREFILL_ATTENTION_PROJECTED_BYTES,
            "prefill projected attention final tile",
        )?,
        decode_f32_fixture(
            PREFILL_ATTENTION_HC_POST_BYTES,
            "prefill attention HC-post final tile",
        )?,
    ];
    let expected = [32 * 8192, 32 * 4096, 32 * 16384];
    if tensors
        .iter()
        .zip(expected)
        .any(|(tensor, elements)| tensor.len() != elements)
    {
        return Err(Error::invalid(
            "prefill attention-output fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_ffn_output_fixture() -> Result<([Vec<f32>; 9], Vec<i32>)> {
    let tensors = [
        decode_f32_fixture(PREFILL_FFN_CUR_BYTES, "prefill FFN current final tile")?,
        decode_f32_fixture(PREFILL_FFN_NORM_BYTES, "prefill FFN norm final tile")?,
        decode_f32_fixture(PREFILL_FFN_ROUTER_LOGITS_BYTES, "prefill FFN router logits")?,
        decode_f32_fixture(
            PREFILL_FFN_ROUTER_PROBS_BYTES,
            "prefill FFN router probabilities",
        )?,
        decode_f32_fixture(
            PREFILL_FFN_ROUTER_WEIGHTS_BYTES,
            "prefill FFN router weights",
        )?,
        decode_f32_fixture(PREFILL_FFN_ROUTED_MID_BYTES, "prefill FFN routed mid")?,
        decode_f32_fixture(PREFILL_FFN_ROUTED_OUT_BYTES, "prefill FFN routed output")?,
        decode_f32_fixture(PREFILL_FFN_SHARED_OUT_BYTES, "prefill FFN shared output")?,
        decode_f32_fixture(PREFILL_FFN_HC_POST_BYTES, "prefill FFN HC post")?,
    ];
    let expected = [
        32 * 4096,
        32 * 4096,
        32 * 256,
        32 * 256,
        32 * 6,
        32 * 6 * 2048,
        32 * 4096,
        32 * 4096,
        32 * 16384,
    ];
    if tensors
        .iter()
        .zip(expected)
        .any(|(tensor, elements)| tensor.len() != elements)
    {
        return Err(Error::invalid(
            "prefill FFN-output fixture dimensions are invalid",
        ));
    }
    let selected = decode_i32_fixture(PREFILL_FFN_SELECTED_BYTES, "prefill FFN selected experts")?;
    if selected.len() != 32 * 6 {
        return Err(Error::invalid(
            "prefill FFN selected-expert fixture dimensions are invalid",
        ));
    }
    Ok((tensors, selected))
}

fn prefill_layer1_ingress_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(PREFILL_LAYER1_HC_PRE_BYTES, "prefill layer-1 HC ingress")?,
        decode_f32_fixture(
            PREFILL_LAYER1_ATTN_NORM_BYTES,
            "prefill layer-1 attention norm",
        )?,
        decode_f32_fixture(PREFILL_LAYER1_Q_LORA_BYTES, "prefill layer-1 Q-Lora")?,
    ];
    let expected = [32 * 4096, 32 * 4096, 32 * 1024];
    if tensors
        .iter()
        .zip(expected)
        .any(|(tensor, elements)| tensor.len() != elements)
    {
        return Err(Error::invalid(
            "prefill layer-1 ingress fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_layer1_complete_fixture() -> Result<([Vec<f32>; 20], Vec<i32>)> {
    let tensors = [
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_Q_NORM_BYTES, "layer-1 Q-Lora norm")?,
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_KV_NORM_BYTES, "layer-1 KV norm")?,
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_Q_CUR_BYTES, "layer-1 Q current")?,
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_KV_ROPE_BYTES, "layer-1 KV RoPE")?,
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_KV_CUR_BYTES, "layer-1 KV current")?,
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_KV_PREFIX_BYTES, "layer-1 KV prefix")?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_KQV_OUT_BYTES,
            "layer-1 attention output",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_KQV_BACK_BYTES,
            "layer-1 inverse-RoPE attention",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ATTN_LOW_BYTES,
            "layer-1 attention low",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ATTN_OUT_BYTES,
            "layer-1 projected attention",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_HC_ATTN_POST_BYTES,
            "layer-1 attention HC post",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_HC_FFN_PRE_BYTES,
            "layer-1 FFN current",
        )?,
        decode_f32_fixture(PREFILL_LAYER1_COMPLETE_FFN_NORM_BYTES, "layer-1 FFN norm")?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ROUTER_LOGITS_BYTES,
            "layer-1 router logits",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ROUTER_PROBS_BYTES,
            "layer-1 router probabilities",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ROUTER_WEIGHTS_BYTES,
            "layer-1 router weights",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ROUTED_MID_BYTES,
            "layer-1 routed mid",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_ROUTED_OUT_BYTES,
            "layer-1 routed output",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_SHARED_OUT_BYTES,
            "layer-1 shared output",
        )?,
        decode_f32_fixture(
            PREFILL_LAYER1_COMPLETE_HC_FFN_POST_BYTES,
            "layer-1 FFN HC post",
        )?,
    ];
    let expected = [
        32 * 1024,
        32 * 512,
        32 * 32768,
        32 * 512,
        32 * 512,
        2016 * 512,
        32 * 64 * 512,
        32 * 64 * 512,
        32 * 8192,
        32 * 4096,
        32 * 16384,
        32 * 4096,
        32 * 4096,
        32 * 256,
        32 * 256,
        32 * 6,
        32 * 6 * 2048,
        32 * 4096,
        32 * 4096,
        32 * 16384,
    ];
    if tensors
        .iter()
        .zip(expected)
        .any(|(tensor, elements)| tensor.len() != elements)
    {
        return Err(Error::invalid(
            "prefill complete layer-1 fixture dimensions are invalid",
        ));
    }
    let selected = decode_i32_fixture(
        PREFILL_LAYER1_COMPLETE_SELECTED_BYTES,
        "layer-1 selected experts",
    )?;
    if selected.len() != 32 * 6 {
        return Err(Error::invalid(
            "prefill complete layer-1 selected-expert dimensions are invalid",
        ));
    }
    Ok((tensors, selected))
}

fn prefill_layers01_previous_tile_fixture() -> Result<([Vec<f32>; 4], [Vec<i32>; 2])> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_PREVIOUS_LAYER0_KV_CUR_BYTES,
            "previous layer-0 KV current",
        )?,
        decode_f32_fixture(
            PREFILL_PREVIOUS_LAYER0_HC_POST_BYTES,
            "previous layer-0 HC post",
        )?,
        decode_f32_fixture(
            PREFILL_PREVIOUS_LAYER1_KV_CUR_BYTES,
            "previous layer-1 KV current",
        )?,
        decode_f32_fixture(
            PREFILL_PREVIOUS_LAYER1_HC_POST_BYTES,
            "previous layer-1 HC post",
        )?,
    ];
    let selected = [
        decode_i32_fixture(
            PREFILL_PREVIOUS_LAYER0_SELECTED_BYTES,
            "previous layer-0 experts",
        )?,
        decode_i32_fixture(
            PREFILL_PREVIOUS_LAYER1_SELECTED_BYTES,
            "previous layer-1 experts",
        )?,
    ];
    if tensors[0].len() != 32 * 512
        || tensors[1].len() != 32 * 16384
        || tensors[2].len() != 32 * 512
        || tensors[3].len() != 32 * 16384
        || selected.iter().any(|values| values.len() != 32 * 6)
    {
        return Err(Error::invalid(
            "prefill layers-0/1 previous-tile fixture dimensions are invalid",
        ));
    }
    Ok((tensors, selected))
}

fn prefill_layer2_kvnorm_fixture() -> Result<Vec<f32>> {
    let values = decode_f32_fixture(
        PREFILL_LAYER2_KV_NORM_BYTES,
        "prefill layer-2 normalized KV",
    )?;
    if values.len() != 2048 * 512 {
        return Err(Error::invalid(
            "prefill layer-2 normalized-KV fixture dimensions are invalid",
        ));
    }
    Ok(values)
}

fn prefill_layer2_kv_state_fixture() -> Result<[Vec<f32>; 2]> {
    let rope = decode_f32_fixture(
        PREFILL_LAYER2_KV_ROPE_BYTES,
        "prefill layer-2 KV after compressed RoPE",
    )?;
    let current = decode_f32_fixture(
        PREFILL_LAYER2_KV_CURRENT_BYTES,
        "prefill layer-2 KV after E4M3FN finalization",
    )?;
    if rope.len() != 2048 * 512 || current.len() != 2048 * 512 {
        return Err(Error::invalid(
            "prefill layer-2 KV-state fixture dimensions are invalid",
        ));
    }
    Ok([rope, current])
}

fn prefill_layer2_compressor_fixture() -> Result<[Vec<f32>; 6]> {
    let attention_compressed = decode_f32_fixture(
        PREFILL_LAYER2_ATTN_COMPRESSED_BYTES,
        "prefill layer-2 attention compressed KV",
    )?;
    let attention_state_kv = decode_f32_fixture(
        PREFILL_LAYER2_ATTN_STATE_KV_BYTES,
        "prefill layer-2 attention compressor KV state",
    )?;
    let attention_state_score = decode_i32_fixture(
        PREFILL_LAYER2_ATTN_STATE_SCORE_BITS,
        "prefill layer-2 attention compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    let indexer_compressed = decode_f32_fixture(
        PREFILL_LAYER2_INDEXER_COMPRESSED_BYTES,
        "prefill layer-2 indexer compressed KV",
    )?;
    let indexer_state_kv = decode_f32_fixture(
        PREFILL_LAYER2_INDEXER_STATE_KV_BYTES,
        "prefill layer-2 indexer compressor KV state",
    )?;
    let indexer_state_score = decode_i32_fixture(
        PREFILL_LAYER2_INDEXER_STATE_SCORE_BITS,
        "prefill layer-2 indexer compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    if attention_compressed.len() != 512 * 512
        || attention_state_kv.len() != 8 * 1024
        || attention_state_score.len() != 8 * 1024
        || indexer_compressed.len() != 512 * 128
        || indexer_state_kv.len() != 8 * 256
        || indexer_state_score.len() != 8 * 256
    {
        return Err(Error::invalid(
            "prefill layer-2 compressor fixture dimensions are invalid",
        ));
    }
    Ok([
        attention_compressed,
        attention_state_kv,
        attention_state_score,
        indexer_compressed,
        indexer_state_kv,
        indexer_state_score,
    ])
}

fn prefill_layer4_compressor_fixture() -> Result<[Vec<f32>; 6]> {
    let attention_compressed = decode_f32_fixture(
        PREFILL_LAYER4_ATTN_COMPRESSED_BYTES,
        "prefill layer-4 attention compressed KV",
    )?;
    let attention_state_kv = decode_f32_fixture(
        PREFILL_LAYER4_ATTN_STATE_KV_BYTES,
        "prefill layer-4 attention compressor KV state",
    )?;
    let attention_state_score = decode_i32_fixture(
        PREFILL_LAYER4_ATTN_STATE_SCORE_BITS,
        "prefill layer-4 attention compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    let indexer_compressed = decode_f32_fixture(
        PREFILL_LAYER4_INDEXER_COMPRESSED_BYTES,
        "prefill layer-4 indexer compressed KV",
    )?;
    let indexer_state_kv = decode_f32_fixture(
        PREFILL_LAYER4_INDEXER_STATE_KV_BYTES,
        "prefill layer-4 indexer compressor KV state",
    )?;
    let indexer_state_score = decode_i32_fixture(
        PREFILL_LAYER4_INDEXER_STATE_SCORE_BITS,
        "prefill layer-4 indexer compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    if attention_compressed.len() != 512 * 512
        || attention_state_kv.len() != 8 * 1024
        || attention_state_score.len() != 8 * 1024
        || indexer_compressed.len() != 512 * 128
        || indexer_state_kv.len() != 8 * 256
        || indexer_state_score.len() != 8 * 256
    {
        return Err(Error::invalid(
            "prefill layer-4 compressor fixture dimensions are invalid",
        ));
    }
    Ok([
        attention_compressed,
        attention_state_kv,
        attention_state_score,
        indexer_compressed,
        indexer_state_kv,
        indexer_state_score,
    ])
}

fn prefill_layer6_compressor_fixture() -> Result<[Vec<f32>; 6]> {
    let attention_compressed = decode_f32_fixture(
        PREFILL_LAYER6_ATTN_COMPRESSED_BYTES,
        "prefill layer-6 attention compressed KV",
    )?;
    let attention_state_kv = decode_f32_fixture(
        PREFILL_LAYER6_ATTN_STATE_KV_BYTES,
        "prefill layer-6 attention compressor KV state",
    )?;
    let attention_state_score = decode_i32_fixture(
        PREFILL_LAYER6_ATTN_STATE_SCORE_BITS,
        "prefill layer-6 attention compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    let indexer_compressed = decode_f32_fixture(
        PREFILL_LAYER6_INDEXER_COMPRESSED_BYTES,
        "prefill layer-6 indexer compressed KV",
    )?;
    let indexer_state_kv = decode_f32_fixture(
        PREFILL_LAYER6_INDEXER_STATE_KV_BYTES,
        "prefill layer-6 indexer compressor KV state",
    )?;
    let indexer_state_score = decode_i32_fixture(
        PREFILL_LAYER6_INDEXER_STATE_SCORE_BITS,
        "prefill layer-6 indexer compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    if attention_compressed.len() != 512 * 512
        || attention_state_kv.len() != 8 * 1024
        || attention_state_score.len() != 8 * 1024
        || indexer_compressed.len() != 512 * 128
        || indexer_state_kv.len() != 8 * 256
        || indexer_state_score.len() != 8 * 256
    {
        return Err(Error::invalid(
            "prefill layer-6 compressor fixture dimensions are invalid",
        ));
    }
    Ok([
        attention_compressed,
        attention_state_kv,
        attention_state_score,
        indexer_compressed,
        indexer_state_kv,
        indexer_state_score,
    ])
}

fn prefill_layer6_attention_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let attention = decode_f32_fixture(
        PREFILL_LAYER6_ATTENTION_OUTPUT_BYTES,
        "prefill layer-6 dense mixed-attention output",
    )?;
    let hc_final_tile = decode_f32_fixture(
        PREFILL_LAYER6_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-6 attention HC post final tile",
    )?;
    if attention.len() != 2048 * 4096 || hc_final_tile.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-6 attention fixture dimensions are invalid",
        ));
    }
    Ok((attention, hc_final_tile))
}

fn prefill_layer6_attention_diagnostics_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_LAYER6_KQV_OUT_ROW0_BYTES,
            "layer-6 KQV output row 0",
        )?,
        decode_f32_fixture(PREFILL_LAYER6_KQV_BACK_ROW0_BYTES, "layer-6 KQV back row 0")?,
        decode_f32_fixture(
            PREFILL_LAYER6_ATTN_LOW_ROW0_BYTES,
            "layer-6 attention low row 0",
        )?,
    ];
    if tensors[0].len() != 32_768 || tensors[1].len() != 32_768 || tensors[2].len() != 8_192 {
        return Err(Error::invalid(
            "prefill layer-6 attention diagnostic fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_layer4_attention_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let attention = decode_f32_fixture(
        PREFILL_LAYER4_ATTENTION_OUTPUT_BYTES,
        "prefill layer-4 dense mixed-attention output",
    )?;
    let hc_final_tile = decode_f32_fixture(
        PREFILL_LAYER4_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-4 attention HC post final tile",
    )?;
    if attention.len() != 2048 * 4096 || hc_final_tile.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-4 attention fixture dimensions are invalid",
        ));
    }
    Ok((attention, hc_final_tile))
}

fn prefill_layer4_attention_diagnostics_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_LAYER4_KQV_OUT_ROW0_BYTES,
            "layer-4 KQV output row 0",
        )?,
        decode_f32_fixture(PREFILL_LAYER4_KQV_BACK_ROW0_BYTES, "layer-4 KQV back row 0")?,
        decode_f32_fixture(
            PREFILL_LAYER4_ATTN_LOW_ROW0_BYTES,
            "layer-4 attention low row 0",
        )?,
    ];
    if tensors[0].len() != 32_768 || tensors[1].len() != 32_768 || tensors[2].len() != 8_192 {
        return Err(Error::invalid(
            "prefill layer-4 attention diagnostic fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_layer2_attention_fixture() -> Result<Vec<f32>> {
    let values = decode_f32_fixture(
        PREFILL_LAYER2_ATTENTION_OUTPUT_BYTES,
        "prefill layer-2 dense mixed-attention output",
    )?;
    if values.len() != 2048 * 4096 {
        return Err(Error::invalid(
            "prefill layer-2 attention fixture dimensions are invalid",
        ));
    }
    Ok(values)
}

fn prefill_layer2_hc_attn_post_final_tile_fixture() -> Result<Vec<f32>> {
    let values = decode_f32_fixture(
        PREFILL_LAYER2_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-2 attention HC post final tile",
    )?;
    if values.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-2 attention HC post fixture dimensions are invalid",
        ));
    }
    Ok(values)
}

fn prefill_layer2_hc_ffn_post_final_tile_fixture() -> Result<Vec<f32>> {
    let values = decode_f32_fixture(
        PREFILL_LAYER2_HC_FFN_POST_FINAL_TILE_BYTES,
        "prefill layer-2 FFN HC post final tile",
    )?;
    if values.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-2 FFN HC post fixture dimensions are invalid",
        ));
    }
    Ok(values)
}

fn prefill_layer2_ffn_ingress_final_tile_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let cur = decode_f32_fixture(
        PREFILL_LAYER2_FFN_CUR_FINAL_TILE_BYTES,
        "prefill layer-2 FFN current final tile",
    )?;
    let norm = decode_f32_fixture(
        PREFILL_LAYER2_FFN_NORM_FINAL_TILE_BYTES,
        "prefill layer-2 FFN norm final tile",
    )?;
    if cur.len() != 32 * 4096 || norm.len() != 32 * 4096 {
        return Err(Error::invalid(
            "prefill layer-2 FFN ingress fixture dimensions are invalid",
        ));
    }
    Ok((cur, norm))
}

fn prefill_layer2_ffn_output_final_tile_fixture() -> Result<(Vec<i32>, [Vec<f32>; 3])> {
    let selected = decode_i32_fixture(
        PREFILL_LAYER2_ROUTER_SELECTED_FINAL_TILE_BYTES,
        "prefill layer-2 router selection final tile",
    )?;
    let weights = decode_f32_fixture(
        PREFILL_LAYER2_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
        "prefill layer-2 router weights final tile",
    )?;
    let routed = decode_f32_fixture(
        PREFILL_LAYER2_ROUTED_OUT_FINAL_TILE_BYTES,
        "prefill layer-2 routed output final tile",
    )?;
    let shared = decode_f32_fixture(
        PREFILL_LAYER2_SHARED_OUT_FINAL_TILE_BYTES,
        "prefill layer-2 shared output final tile",
    )?;
    if selected.len() != 32 * 6
        || weights.len() != 32 * 6
        || routed.len() != 32 * 4096
        || shared.len() != 32 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-2 FFN output fixture dimensions are invalid",
        ));
    }
    Ok((selected, [weights, routed, shared]))
}

fn prefill_layer3_ingress_final_tile_fixture() -> Result<[Vec<f32>; 3]> {
    let hc_attn_pre = decode_f32_fixture(
        PREFILL_LAYER3_HC_ATTN_PRE_FINAL_TILE_BYTES,
        "prefill layer-3 HC attention ingress final tile",
    )?;
    let attn_norm = decode_f32_fixture(
        PREFILL_LAYER3_ATTN_NORM_FINAL_TILE_BYTES,
        "prefill layer-3 attention norm final tile",
    )?;
    let q_lora = decode_f32_fixture(
        PREFILL_LAYER3_Q_LORA_FINAL_TILE_BYTES,
        "prefill layer-3 Q-Lora final tile",
    )?;
    if hc_attn_pre.len() != 32 * 4096 || attn_norm.len() != 32 * 4096 || q_lora.len() != 32 * 1024 {
        return Err(Error::invalid(
            "prefill layer-3 attention ingress fixture dimensions are invalid",
        ));
    }
    Ok([hc_attn_pre, attn_norm, q_lora])
}

fn prefill_layer3_kv_state_final_tile_fixture() -> Result<[Vec<f32>; 7]> {
    let bytes = [
        PREFILL_LAYER3_Q_LORA_NORM_FINAL_TILE_BYTES,
        PREFILL_LAYER3_KV_RAW_FINAL_TILE_BYTES,
        PREFILL_LAYER3_KV_NORM_FINAL_TILE_BYTES,
        PREFILL_LAYER3_Q_RAW_FINAL_TILE_BYTES,
        PREFILL_LAYER3_Q_CUR_FINAL_TILE_BYTES,
        PREFILL_LAYER3_KV_ROPE_FINAL_TILE_BYTES,
        PREFILL_LAYER3_KV_CUR_FINAL_TILE_BYTES,
    ];
    let labels = [
        "Q-Lora norm",
        "KV raw",
        "KV norm",
        "Q raw",
        "Q current",
        "KV rope",
        "KV current",
    ];
    let expected_lengths = [
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
        32 * 512,
        32 * 512,
    ];
    let mut values = Vec::with_capacity(bytes.len());
    for ((bytes, label), expected_length) in bytes.into_iter().zip(labels).zip(expected_lengths) {
        let tensor = decode_f32_fixture(bytes, &format!("prefill layer-3 {label} final tile"))?;
        if tensor.len() != expected_length {
            return Err(Error::invalid(format!(
                "prefill layer-3 {label} fixture dimensions are invalid"
            )));
        }
        values.push(tensor);
    }
    values
        .try_into()
        .map_err(|_| Error::invalid("prefill layer-3 KV-state fixture count is invalid"))
}

fn prefill_layer3_compressor_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<i32>)> {
    let compressed = decode_f32_fixture(
        PREFILL_LAYER3_ATTN_COMPRESSED_BYTES,
        "prefill layer-3 ratio-128 compressed KV",
    )?;
    let state_kv = decode_f32_fixture(
        PREFILL_LAYER3_ATTN_STATE_KV_BYTES,
        "prefill layer-3 ratio-128 recurrent KV state",
    )?;
    let state_score = decode_i32_fixture(
        PREFILL_LAYER3_ATTN_STATE_SCORE_BYTES,
        "prefill layer-3 ratio-128 recurrent score-state bits",
    )?;
    if compressed.len() != 16 * 512 || state_kv.len() != 128 * 512 || state_score.len() != 128 * 512
    {
        return Err(Error::invalid(
            "prefill layer-3 ratio-128 compressor fixture dimensions are invalid",
        ));
    }
    Ok((compressed, state_kv, state_score))
}

fn prefill_layer5_compressor_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<i32>)> {
    let compressed = decode_f32_fixture(
        PREFILL_LAYER5_ATTN_COMPRESSED_BYTES,
        "prefill layer-5 ratio-128 compressed KV",
    )?;
    let state_kv = decode_f32_fixture(
        PREFILL_LAYER5_ATTN_STATE_KV_BYTES,
        "prefill layer-5 ratio-128 recurrent KV state",
    )?;
    let state_score = decode_i32_fixture(
        PREFILL_LAYER5_ATTN_STATE_SCORE_BYTES,
        "prefill layer-5 ratio-128 recurrent score-state bits",
    )?;
    if compressed.len() != 16 * 512 || state_kv.len() != 128 * 512 || state_score.len() != 128 * 512
    {
        return Err(Error::invalid(
            "prefill layer-5 ratio-128 compressor fixture dimensions are invalid",
        ));
    }
    Ok((compressed, state_kv, state_score))
}

fn prefill_layer7_compressor_fixture() -> Result<(Vec<f32>, Vec<f32>, Vec<i32>)> {
    let compressed = decode_f32_fixture(
        PREFILL_LAYER7_ATTN_COMPRESSED_BYTES,
        "prefill layer-7 ratio-128 compressed KV",
    )?;
    let state_kv = decode_f32_fixture(
        PREFILL_LAYER7_ATTN_STATE_KV_BYTES,
        "prefill layer-7 ratio-128 recurrent KV state",
    )?;
    let state_score = decode_i32_fixture(
        PREFILL_LAYER7_ATTN_STATE_SCORE_BYTES,
        "prefill layer-7 ratio-128 recurrent score-state bits",
    )?;
    if compressed.len() != 16 * 512 || state_kv.len() != 128 * 512 || state_score.len() != 128 * 512
    {
        return Err(Error::invalid(
            "prefill layer-7 ratio-128 compressor fixture dimensions are invalid",
        ));
    }
    Ok((compressed, state_kv, state_score))
}

fn prefill_layer5_attention_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let attention = decode_f32_fixture(
        PREFILL_LAYER5_ATTENTION_OUTPUT_BYTES,
        "prefill layer-5 dense mixed-attention output",
    )?;
    let hc_final_tile = decode_f32_fixture(
        PREFILL_LAYER5_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-5 attention HC post final tile",
    )?;
    if attention.len() != 2048 * 4096 || hc_final_tile.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-5 attention fixture dimensions are invalid",
        ));
    }
    Ok((attention, hc_final_tile))
}

fn prefill_layer5_attention_diagnostics_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_LAYER5_KQV_OUT_ROW0_BYTES,
            "layer-5 KQV output row 0",
        )?,
        decode_f32_fixture(PREFILL_LAYER5_KQV_BACK_ROW0_BYTES, "layer-5 KQV back row 0")?,
        decode_f32_fixture(
            PREFILL_LAYER5_ATTN_LOW_ROW0_BYTES,
            "layer-5 attention low row 0",
        )?,
    ];
    if tensors[0].len() != 32_768 || tensors[1].len() != 32_768 || tensors[2].len() != 8_192 {
        return Err(Error::invalid(
            "prefill layer-5 attention diagnostic fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_layer7_attention_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let attention = decode_f32_fixture(
        PREFILL_LAYER7_ATTENTION_OUTPUT_BYTES,
        "prefill layer-7 dense mixed-attention output",
    )?;
    let hc_final_tile = decode_f32_fixture(
        PREFILL_LAYER7_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-7 attention HC post final tile",
    )?;
    if attention.len() != 2048 * 4096 || hc_final_tile.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-7 attention fixture dimensions are invalid",
        ));
    }
    Ok((attention, hc_final_tile))
}

fn prefill_layer7_attention_diagnostics_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_LAYER7_KQV_OUT_ROW0_BYTES,
            "layer-7 KQV output row 0",
        )?,
        decode_f32_fixture(PREFILL_LAYER7_KQV_BACK_ROW0_BYTES, "layer-7 KQV back row 0")?,
        decode_f32_fixture(
            PREFILL_LAYER7_ATTN_LOW_ROW0_BYTES,
            "layer-7 attention low row 0",
        )?,
    ];
    if tensors[0].len() != 32_768 || tensors[1].len() != 32_768 || tensors[2].len() != 8_192 {
        return Err(Error::invalid(
            "prefill layer-7 attention diagnostic fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_layer3_attention_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let attention = decode_f32_fixture(
        PREFILL_LAYER3_ATTENTION_OUTPUT_BYTES,
        "prefill layer-3 dense mixed-attention output",
    )?;
    let hc_final_tile = decode_f32_fixture(
        PREFILL_LAYER3_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-3 attention HC post final tile",
    )?;
    if attention.len() != 2048 * 4096 || hc_final_tile.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-3 attention fixture dimensions are invalid",
        ));
    }
    Ok((attention, hc_final_tile))
}

fn prefill_layer3_attention_diagnostics_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_LAYER3_KQV_OUT_ROW0_BYTES,
            "layer-3 KQV output row 0",
        )?,
        decode_f32_fixture(PREFILL_LAYER3_KQV_BACK_ROW0_BYTES, "layer-3 KQV back row 0")?,
        decode_f32_fixture(
            PREFILL_LAYER3_ATTN_LOW_ROW0_BYTES,
            "layer-3 attention low row 0",
        )?,
    ];
    if tensors[0].len() != 32_768 || tensors[1].len() != 32_768 || tensors[2].len() != 8_192 {
        return Err(Error::invalid(
            "prefill layer-3 attention diagnostic fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

struct PrefillLayer3CompleteFixture {
    ffn_cur: Vec<f32>,
    ffn_norm: Vec<f32>,
    router_selected: Vec<i32>,
    router_weights: Vec<f32>,
    routed_out: Vec<f32>,
    shared_out: Vec<f32>,
    hc_post: Vec<f32>,
}

fn prefill_layer3_complete_final_tile_fixture() -> Result<PrefillLayer3CompleteFixture> {
    let fixture = PrefillLayer3CompleteFixture {
        ffn_cur: decode_f32_fixture(
            PREFILL_LAYER3_FFN_CUR_FINAL_TILE_BYTES,
            "prefill layer-3 FFN HC ingress final tile",
        )?,
        ffn_norm: decode_f32_fixture(
            PREFILL_LAYER3_FFN_NORM_FINAL_TILE_BYTES,
            "prefill layer-3 FFN norm final tile",
        )?,
        router_selected: decode_i32_fixture(
            PREFILL_LAYER3_ROUTER_SELECTED_FINAL_TILE_BYTES,
            "prefill layer-3 biased top-k selections final tile",
        )?,
        router_weights: decode_f32_fixture(
            PREFILL_LAYER3_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
            "prefill layer-3 router weights final tile",
        )?,
        routed_out: decode_f32_fixture(
            PREFILL_LAYER3_ROUTED_OUT_FINAL_TILE_BYTES,
            "prefill layer-3 routed output final tile",
        )?,
        shared_out: decode_f32_fixture(
            PREFILL_LAYER3_SHARED_OUT_FINAL_TILE_BYTES,
            "prefill layer-3 shared output final tile",
        )?,
        hc_post: decode_f32_fixture(
            PREFILL_LAYER3_HC_FFN_POST_FINAL_TILE_BYTES,
            "prefill layer-3 FFN HC post final tile",
        )?,
    };
    if fixture.ffn_cur.len() != 32 * 4096
        || fixture.ffn_norm.len() != 32 * 4096
        || fixture.router_selected.len() != 32 * 6
        || fixture.router_weights.len() != 32 * 6
        || fixture.routed_out.len() != 32 * 4096
        || fixture.shared_out.len() != 32 * 4096
        || fixture.hc_post.len() != 32 * 4 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-3 complete fixture dimensions are invalid",
        ));
    }
    Ok(fixture)
}

fn prefill_layer4_complete_final_tile_fixture() -> Result<PrefillLayer3CompleteFixture> {
    let fixture = PrefillLayer3CompleteFixture {
        ffn_cur: decode_f32_fixture(
            PREFILL_LAYER4_FFN_CUR_FINAL_TILE_BYTES,
            "prefill layer-4 FFN HC ingress final tile",
        )?,
        ffn_norm: decode_f32_fixture(
            PREFILL_LAYER4_FFN_NORM_FINAL_TILE_BYTES,
            "prefill layer-4 FFN norm final tile",
        )?,
        router_selected: decode_i32_fixture(
            PREFILL_LAYER4_ROUTER_SELECTED_FINAL_TILE_BYTES,
            "prefill layer-4 biased top-k selections final tile",
        )?,
        router_weights: decode_f32_fixture(
            PREFILL_LAYER4_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
            "prefill layer-4 router weights final tile",
        )?,
        routed_out: decode_f32_fixture(
            PREFILL_LAYER4_ROUTED_OUT_FINAL_TILE_BYTES,
            "prefill layer-4 routed output final tile",
        )?,
        shared_out: decode_f32_fixture(
            PREFILL_LAYER4_SHARED_OUT_FINAL_TILE_BYTES,
            "prefill layer-4 shared output final tile",
        )?,
        hc_post: decode_f32_fixture(
            PREFILL_LAYER4_HC_FFN_POST_FINAL_TILE_BYTES,
            "prefill layer-4 FFN HC post final tile",
        )?,
    };
    if fixture.ffn_cur.len() != 32 * 4096
        || fixture.ffn_norm.len() != 32 * 4096
        || fixture.router_selected.len() != 32 * 6
        || fixture.router_weights.len() != 32 * 6
        || fixture.routed_out.len() != 32 * 4096
        || fixture.shared_out.len() != 32 * 4096
        || fixture.hc_post.len() != 32 * 4 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-4 complete fixture dimensions are invalid",
        ));
    }
    Ok(fixture)
}

fn prefill_layer5_complete_final_tile_fixture() -> Result<PrefillLayer3CompleteFixture> {
    let fixture = PrefillLayer3CompleteFixture {
        ffn_cur: decode_f32_fixture(
            PREFILL_LAYER5_FFN_CUR_FINAL_TILE_BYTES,
            "prefill layer-5 FFN HC ingress final tile",
        )?,
        ffn_norm: decode_f32_fixture(
            PREFILL_LAYER5_FFN_NORM_FINAL_TILE_BYTES,
            "prefill layer-5 FFN norm final tile",
        )?,
        router_selected: decode_i32_fixture(
            PREFILL_LAYER5_ROUTER_SELECTED_FINAL_TILE_BYTES,
            "prefill layer-5 biased top-k selections final tile",
        )?,
        router_weights: decode_f32_fixture(
            PREFILL_LAYER5_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
            "prefill layer-5 router weights final tile",
        )?,
        routed_out: decode_f32_fixture(
            PREFILL_LAYER5_ROUTED_OUT_FINAL_TILE_BYTES,
            "prefill layer-5 routed output final tile",
        )?,
        shared_out: decode_f32_fixture(
            PREFILL_LAYER5_SHARED_OUT_FINAL_TILE_BYTES,
            "prefill layer-5 shared output final tile",
        )?,
        hc_post: decode_f32_fixture(
            PREFILL_LAYER5_HC_FFN_POST_FINAL_TILE_BYTES,
            "prefill layer-5 FFN HC post final tile",
        )?,
    };
    if fixture.ffn_cur.len() != 32 * 4096
        || fixture.ffn_norm.len() != 32 * 4096
        || fixture.router_selected.len() != 32 * 6
        || fixture.router_weights.len() != 32 * 6
        || fixture.routed_out.len() != 32 * 4096
        || fixture.shared_out.len() != 32 * 4096
        || fixture.hc_post.len() != 32 * 4 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-5 complete fixture dimensions are invalid",
        ));
    }
    Ok(fixture)
}

fn prefill_layer6_complete_final_tile_fixture() -> Result<PrefillLayer3CompleteFixture> {
    let fixture = PrefillLayer3CompleteFixture {
        ffn_cur: decode_f32_fixture(
            PREFILL_LAYER6_FFN_CUR_FINAL_TILE_BYTES,
            "prefill layer-6 FFN HC ingress final tile",
        )?,
        ffn_norm: decode_f32_fixture(
            PREFILL_LAYER6_FFN_NORM_FINAL_TILE_BYTES,
            "prefill layer-6 FFN norm final tile",
        )?,
        router_selected: decode_i32_fixture(
            PREFILL_LAYER6_ROUTER_SELECTED_FINAL_TILE_BYTES,
            "prefill layer-6 biased top-k selections final tile",
        )?,
        router_weights: decode_f32_fixture(
            PREFILL_LAYER6_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
            "prefill layer-6 router weights final tile",
        )?,
        routed_out: decode_f32_fixture(
            PREFILL_LAYER6_ROUTED_OUT_FINAL_TILE_BYTES,
            "prefill layer-6 routed output final tile",
        )?,
        shared_out: decode_f32_fixture(
            PREFILL_LAYER6_SHARED_OUT_FINAL_TILE_BYTES,
            "prefill layer-6 shared output final tile",
        )?,
        hc_post: decode_f32_fixture(
            PREFILL_LAYER6_HC_FFN_POST_FINAL_TILE_BYTES,
            "prefill layer-6 FFN HC post final tile",
        )?,
    };
    if fixture.ffn_cur.len() != 32 * 4096
        || fixture.ffn_norm.len() != 32 * 4096
        || fixture.router_selected.len() != 32 * 6
        || fixture.router_weights.len() != 32 * 6
        || fixture.routed_out.len() != 32 * 4096
        || fixture.shared_out.len() != 32 * 4096
        || fixture.hc_post.len() != 32 * 4 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-6 complete fixture dimensions are invalid",
        ));
    }
    Ok(fixture)
}

fn prefill_layer7_complete_final_tile_fixture() -> Result<PrefillLayer3CompleteFixture> {
    let fixture = PrefillLayer3CompleteFixture {
        ffn_cur: decode_f32_fixture(
            PREFILL_LAYER7_FFN_CUR_FINAL_TILE_BYTES,
            "prefill layer-7 FFN HC ingress final tile",
        )?,
        ffn_norm: decode_f32_fixture(
            PREFILL_LAYER7_FFN_NORM_FINAL_TILE_BYTES,
            "prefill layer-7 FFN norm final tile",
        )?,
        router_selected: decode_i32_fixture(
            PREFILL_LAYER7_ROUTER_SELECTED_FINAL_TILE_BYTES,
            "prefill layer-7 biased top-k selections final tile",
        )?,
        router_weights: decode_f32_fixture(
            PREFILL_LAYER7_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
            "prefill layer-7 router weights final tile",
        )?,
        routed_out: decode_f32_fixture(
            PREFILL_LAYER7_ROUTED_OUT_FINAL_TILE_BYTES,
            "prefill layer-7 routed output final tile",
        )?,
        shared_out: decode_f32_fixture(
            PREFILL_LAYER7_SHARED_OUT_FINAL_TILE_BYTES,
            "prefill layer-7 shared output final tile",
        )?,
        hc_post: decode_f32_fixture(
            PREFILL_LAYER7_HC_FFN_POST_FINAL_TILE_BYTES,
            "prefill layer-7 FFN HC post final tile",
        )?,
    };
    if fixture.ffn_cur.len() != 32 * 4096
        || fixture.ffn_norm.len() != 32 * 4096
        || fixture.router_selected.len() != 32 * 6
        || fixture.router_weights.len() != 32 * 6
        || fixture.routed_out.len() != 32 * 4096
        || fixture.shared_out.len() != 32 * 4096
        || fixture.hc_post.len() != 32 * 4 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-7 complete fixture dimensions are invalid",
        ));
    }
    Ok(fixture)
}

fn prefill_layer4_qkv_final_tile_fixture() -> Result<[Vec<f32>; 10]> {
    let labels = [
        "HC attention ingress",
        "attention norm",
        "Q-Lora",
        "Q-Lora norm",
        "KV raw",
        "KV norm",
        "Q raw",
        "Q current",
        "KV rope",
        "KV current",
    ];
    let expected_lengths = [
        32 * 4096,
        32 * 4096,
        32 * 1024,
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
        32 * 512,
        32 * 512,
    ];
    let mut tensors = Vec::with_capacity(PREFILL_LAYER4_QKV_FINAL_TILE_BYTES.len());
    for ((bytes, label), expected_length) in PREFILL_LAYER4_QKV_FINAL_TILE_BYTES
        .into_iter()
        .zip(labels)
        .zip(expected_lengths)
    {
        let tensor = decode_f32_fixture(bytes, &format!("prefill layer-4 {label} final tile"))?;
        if tensor.len() != expected_length {
            return Err(Error::invalid(format!(
                "prefill layer-4 {label} fixture dimensions are invalid"
            )));
        }
        tensors.push(tensor);
    }
    tensors
        .try_into()
        .map_err(|_| Error::invalid("prefill layer-4 Q/KV fixture count is invalid"))
}

fn prefill_layer5_qkv_final_tile_fixture() -> Result<[Vec<f32>; 10]> {
    let labels = [
        "HC attention ingress",
        "attention norm",
        "Q-Lora",
        "Q-Lora norm",
        "KV raw",
        "KV norm",
        "Q raw",
        "Q current",
        "KV rope",
        "KV current",
    ];
    let expected_lengths = [
        32 * 4096,
        32 * 4096,
        32 * 1024,
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
        32 * 512,
        32 * 512,
    ];
    let mut tensors = Vec::with_capacity(PREFILL_LAYER5_QKV_FINAL_TILE_BYTES.len());
    for ((bytes, label), expected_length) in PREFILL_LAYER5_QKV_FINAL_TILE_BYTES
        .into_iter()
        .zip(labels)
        .zip(expected_lengths)
    {
        let tensor = decode_f32_fixture(bytes, &format!("prefill layer-5 {label} final tile"))?;
        if tensor.len() != expected_length {
            return Err(Error::invalid(format!(
                "prefill layer-5 {label} fixture dimensions are invalid"
            )));
        }
        tensors.push(tensor);
    }
    tensors
        .try_into()
        .map_err(|_| Error::invalid("prefill layer-5 Q/KV fixture count is invalid"))
}

fn prefill_layer6_qkv_final_tile_fixture() -> Result<[Vec<f32>; 10]> {
    let labels = [
        "HC attention ingress",
        "attention norm",
        "Q-Lora",
        "Q-Lora norm",
        "KV raw",
        "KV norm",
        "Q raw",
        "Q current",
        "KV rope",
        "KV current",
    ];
    let expected_lengths = [
        32 * 4096,
        32 * 4096,
        32 * 1024,
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
        32 * 512,
        32 * 512,
    ];
    let mut tensors = Vec::with_capacity(PREFILL_LAYER6_QKV_FINAL_TILE_BYTES.len());
    for ((bytes, label), expected_length) in PREFILL_LAYER6_QKV_FINAL_TILE_BYTES
        .into_iter()
        .zip(labels)
        .zip(expected_lengths)
    {
        let tensor = decode_f32_fixture(bytes, &format!("prefill layer-6 {label} final tile"))?;
        if tensor.len() != expected_length {
            return Err(Error::invalid(format!(
                "prefill layer-6 {label} fixture dimensions are invalid"
            )));
        }
        tensors.push(tensor);
    }
    tensors
        .try_into()
        .map_err(|_| Error::invalid("prefill layer-6 Q/KV fixture count is invalid"))
}

fn prefill_layer7_qkv_final_tile_fixture() -> Result<[Vec<f32>; 10]> {
    let labels = [
        "HC attention ingress",
        "attention norm",
        "Q-Lora",
        "Q-Lora norm",
        "KV raw",
        "KV norm",
        "Q raw",
        "Q current",
        "KV rope",
        "KV current",
    ];
    let expected_lengths = [
        32 * 4096,
        32 * 4096,
        32 * 1024,
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
        32 * 512,
        32 * 512,
    ];
    let mut tensors = Vec::with_capacity(PREFILL_LAYER7_QKV_FINAL_TILE_BYTES.len());
    for ((bytes, label), expected_length) in PREFILL_LAYER7_QKV_FINAL_TILE_BYTES
        .into_iter()
        .zip(labels)
        .zip(expected_lengths)
    {
        let tensor = decode_f32_fixture(bytes, &format!("prefill layer-7 {label} final tile"))?;
        if tensor.len() != expected_length {
            return Err(Error::invalid(format!(
                "prefill layer-7 {label} fixture dimensions are invalid"
            )));
        }
        tensors.push(tensor);
    }
    tensors
        .try_into()
        .map_err(|_| Error::invalid("prefill layer-7 Q/KV fixture count is invalid"))
}

fn prefill_layer8_compressor_fixture() -> Result<[Vec<f32>; 6]> {
    let attention_compressed = decode_f32_fixture(
        PREFILL_LAYER8_ATTN_COMPRESSED_BYTES,
        "prefill layer-8 attention compressed KV",
    )?;
    let attention_state_kv = decode_f32_fixture(
        PREFILL_LAYER8_ATTN_STATE_KV_BYTES,
        "prefill layer-8 attention compressor KV state",
    )?;
    let attention_state_score = decode_i32_fixture(
        PREFILL_LAYER8_ATTN_STATE_SCORE_BITS,
        "prefill layer-8 attention compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    let indexer_compressed = decode_f32_fixture(
        PREFILL_LAYER8_INDEXER_COMPRESSED_BYTES,
        "prefill layer-8 indexer compressed KV",
    )?;
    let indexer_state_kv = decode_f32_fixture(
        PREFILL_LAYER8_INDEXER_STATE_KV_BYTES,
        "prefill layer-8 indexer compressor KV state",
    )?;
    let indexer_state_score = decode_i32_fixture(
        PREFILL_LAYER8_INDEXER_STATE_SCORE_BITS,
        "prefill layer-8 indexer compressor score-state bits",
    )?
    .into_iter()
    .map(|bits| f32::from_bits(bits as u32))
    .collect::<Vec<_>>();
    if attention_compressed.len() != 512 * 512
        || attention_state_kv.len() != 8 * 1024
        || attention_state_score.len() != 8 * 1024
        || indexer_compressed.len() != 512 * 128
        || indexer_state_kv.len() != 8 * 256
        || indexer_state_score.len() != 8 * 256
    {
        return Err(Error::invalid(
            "prefill layer-8 compressor fixture dimensions are invalid",
        ));
    }
    Ok([
        attention_compressed,
        attention_state_kv,
        attention_state_score,
        indexer_compressed,
        indexer_state_kv,
        indexer_state_score,
    ])
}

fn prefill_layer8_attention_fixture() -> Result<(Vec<f32>, Vec<f32>)> {
    let attention = decode_f32_fixture(
        PREFILL_LAYER8_ATTENTION_OUTPUT_BYTES,
        "prefill layer-8 dense mixed-attention output",
    )?;
    let hc_final_tile = decode_f32_fixture(
        PREFILL_LAYER8_HC_ATTN_POST_FINAL_TILE_BYTES,
        "prefill layer-8 attention HC post final tile",
    )?;
    if attention.len() != 2048 * 4096 || hc_final_tile.len() != 32 * 4 * 4096 {
        return Err(Error::invalid(
            "prefill layer-8 attention fixture dimensions are invalid",
        ));
    }
    Ok((attention, hc_final_tile))
}

fn prefill_layer8_attention_diagnostics_fixture() -> Result<[Vec<f32>; 3]> {
    let tensors = [
        decode_f32_fixture(
            PREFILL_LAYER8_KQV_OUT_ROW0_BYTES,
            "layer-8 KQV output row 0",
        )?,
        decode_f32_fixture(PREFILL_LAYER8_KQV_BACK_ROW0_BYTES, "layer-8 KQV back row 0")?,
        decode_f32_fixture(
            PREFILL_LAYER8_ATTN_LOW_ROW0_BYTES,
            "layer-8 attention low row 0",
        )?,
    ];
    if tensors[0].len() != 32_768 || tensors[1].len() != 32_768 || tensors[2].len() != 8_192 {
        return Err(Error::invalid(
            "prefill layer-8 attention diagnostic fixture dimensions are invalid",
        ));
    }
    Ok(tensors)
}

fn prefill_layer8_complete_final_tile_fixture() -> Result<PrefillLayer3CompleteFixture> {
    let fixture = PrefillLayer3CompleteFixture {
        ffn_cur: decode_f32_fixture(
            PREFILL_LAYER8_FFN_CUR_FINAL_TILE_BYTES,
            "prefill layer-8 FFN HC ingress final tile",
        )?,
        ffn_norm: decode_f32_fixture(
            PREFILL_LAYER8_FFN_NORM_FINAL_TILE_BYTES,
            "prefill layer-8 FFN norm final tile",
        )?,
        router_selected: decode_i32_fixture(
            PREFILL_LAYER8_ROUTER_SELECTED_FINAL_TILE_BYTES,
            "prefill layer-8 biased top-k selections final tile",
        )?,
        router_weights: decode_f32_fixture(
            PREFILL_LAYER8_ROUTER_WEIGHTS_FINAL_TILE_BYTES,
            "prefill layer-8 router weights final tile",
        )?,
        routed_out: decode_f32_fixture(
            PREFILL_LAYER8_ROUTED_OUT_FINAL_TILE_BYTES,
            "prefill layer-8 routed output final tile",
        )?,
        shared_out: decode_f32_fixture(
            PREFILL_LAYER8_SHARED_OUT_FINAL_TILE_BYTES,
            "prefill layer-8 shared output final tile",
        )?,
        hc_post: decode_f32_fixture(
            PREFILL_LAYER8_HC_FFN_POST_FINAL_TILE_BYTES,
            "prefill layer-8 FFN HC post final tile",
        )?,
    };
    if fixture.ffn_cur.len() != 32 * 4096
        || fixture.ffn_norm.len() != 32 * 4096
        || fixture.router_selected.len() != 32 * 6
        || fixture.router_weights.len() != 32 * 6
        || fixture.routed_out.len() != 32 * 4096
        || fixture.shared_out.len() != 32 * 4096
        || fixture.hc_post.len() != 32 * 4 * 4096
    {
        return Err(Error::invalid(
            "prefill layer-8 complete fixture dimensions are invalid",
        ));
    }
    Ok(fixture)
}

fn prefill_layer8_qkv_final_tile_fixture() -> Result<[Vec<f32>; 10]> {
    let labels = [
        "HC attention ingress",
        "attention norm",
        "Q-Lora",
        "Q-Lora norm",
        "KV raw",
        "KV norm",
        "Q raw",
        "Q current",
        "KV rope",
        "KV current",
    ];
    let expected_lengths = [
        32 * 4096,
        32 * 4096,
        32 * 1024,
        32 * 1024,
        32 * 512,
        32 * 512,
        32 * 32768,
        32 * 32768,
        32 * 512,
        32 * 512,
    ];
    let mut tensors = Vec::with_capacity(PREFILL_LAYER8_QKV_FINAL_TILE_BYTES.len());
    for ((bytes, label), expected_length) in PREFILL_LAYER8_QKV_FINAL_TILE_BYTES
        .into_iter()
        .zip(labels)
        .zip(expected_lengths)
    {
        let tensor = decode_f32_fixture(bytes, &format!("prefill layer-8 {label} final tile"))?;
        if tensor.len() != expected_length {
            return Err(Error::invalid(format!(
                "prefill layer-8 {label} fixture dimensions are invalid"
            )));
        }
        tensors.push(tensor);
    }
    tensors
        .try_into()
        .map_err(|_| Error::invalid("prefill layer-8 Q/KV fixture count is invalid"))
}

fn prefill_hc_ingress_fixture() -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    let tokens = decode_u32_fixture(PREFILL_HC_TOKEN_IDS_BYTES, "prefill HC token IDs")?;
    let collapsed = decode_f32_fixture(
        PREFILL_HC_COLLAPSED_BYTES,
        "prefill HC collapsed final tile",
    )?;
    let attn_norm = decode_f32_fixture(
        PREFILL_HC_ATTN_NORM_BYTES,
        "prefill HC attention norm final tile",
    )?;
    if tokens.len() != 32 || collapsed.len() != 32 * 4096 || attn_norm.len() != 32 * 4096 {
        return Err(Error::invalid(
            "prefill HC ingress fixture dimensions are invalid",
        ));
    }
    Ok((tokens, collapsed, attn_norm))
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

macro_rules! position4_fixture_registry {
    ($($layer:literal),+ $(,)?) => {
        const POS4_BYTES: &[LayerExpectedBytes] = &[
            $(complete_decode_fixture!(@value $layer, 4,
                concat!("dwarfstar-oracle-v1-layer", stringify!($layer), "-pos4-complete"))),+
        ];
    };
}

position4_fixture_registry!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
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
    let bytes = if position == 4 {
        POS4_BYTES.get(layer_index as usize).ok_or_else(|| {
            Error::invalid("position-4 fixture layer is outside the retained decoder")
        })?
    } else if let Some(index) = later_fixture_index(layer_index) {
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
    if (2..=4).contains(&position) {
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

fn validate_prefill_q8_boundary_inputs(
    model: &MappedModel,
    tensor: &TensorInfo,
    input: &[f32],
    batch_output: &[f32],
    decode_output: &[f32],
) -> Result<(u32, u32, u32)> {
    if tensor.name != PROJECTION_TENSOR
        || tensor.tensor_type.id != 8
        || tensor.dimensions != [4096, 1024]
    {
        return Err(Error::invalid(
            "prefill Q8 boundary requires Q8_0 blk.0.attn_q_a.weight [4096, 1024]",
        ));
    }
    const ROWS: u32 = 128;
    const INPUT: u32 = 4096;
    const OUTPUT: u32 = 1024;
    if input.len() != ROWS as usize * INPUT as usize
        || batch_output.len() != ROWS as usize * OUTPUT as usize
        || decode_output.len() != OUTPUT as usize
    {
        return Err(Error::invalid(
            "prefill Q8 boundary fixture does not match its N128 dispatch",
        ));
    }
    let expected_bytes = u64::from(INPUT / 32) * 34 * u64::from(OUTPUT);
    if tensor.bytes != expected_bytes {
        return Err(Error::invalid(format!(
            "prefill Q8 tensor has {} bytes, expected {expected_bytes}",
            tensor.bytes
        )));
    }
    model.tensor_bytes(tensor)?;
    Ok((INPUT, OUTPUT, ROWS))
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr;
    use std::time::Instant;

    const ERROR_BYTES: usize = 1024;
    const COMMAND_SYNCHRONIZED: u32 = 0;
    const COMMAND_CHAINED_ENQUEUE: u32 = 1;
    const COMMAND_CHAINED_FINAL: u32 = 2;
    const COMMAND_CHAINED_COLLECT: u32 = 3;
    const COMMAND_CHAINED_TIMING: u32 = 4;
    const INITIAL_STATE_CAPTURED: u32 = 0;
    const INITIAL_STATE_COLD: u32 = 1;

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
    struct RawPrefillQ8ProbeResult {
        model_bytes: u64,
        tensor_offset: u64,
        tensor_bytes: u64,
        input_elements_per_row: u64,
        output_elements_per_row: u64,
        rows: u64,
        max_buffer_length: u64,
        no_copy_pointer_match: u32,
        batch_threads_per_threadgroup: u32,
        batch_threadgroups_x: u32,
        batch_threadgroups_y: u32,
        batch_wall_ms: f64,
        batch_gpu_ms: f64,
        decode_wall_ms: f64,
        decode_gpu_ms: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawPrefillQkvWeights {
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
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawPrefillQkvProbeResult {
        rows: u64,
        input_elements_per_row: u64,
        q_lora_elements_per_row: u64,
        kv_elements_per_row: u64,
        q_elements_per_row: u64,
        dispatches: u32,
        wrapped_model_ranges: u32,
        pointer_matches: u32,
        position_start: u32,
        wall_ms: f64,
        gpu_ms: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawPrefillFfnWeights {
        hc_fn_offset: u64,
        hc_fn_bytes: u64,
        hc_scale_offset: u64,
        hc_scale_bytes: u64,
        hc_base_offset: u64,
        hc_base_bytes: u64,
        norm_offset: u64,
        norm_bytes: u64,
        router_gate_offset: u64,
        router_gate_bytes: u64,
        router_hash_offset: u64,
        router_hash_bytes: u64,
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
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawPrefillAttentionIngressWeights {
        hc_fn_offset: u64,
        hc_fn_bytes: u64,
        hc_scale_offset: u64,
        hc_scale_bytes: u64,
        hc_base_offset: u64,
        hc_base_bytes: u64,
        norm_offset: u64,
        norm_bytes: u64,
        q_a_offset: u64,
        q_a_bytes: u64,
    }

    #[repr(C)]
    struct RawPrefillLayerWeights {
        ingress: RawPrefillAttentionIngressWeights,
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
        ffn: RawPrefillFfnWeights,
    }

    #[repr(C)]
    struct RawPrefillKvnormWeights {
        ingress: RawPrefillAttentionIngressWeights,
        q_a_norm_offset: u64,
        q_a_norm_bytes: u64,
        kv_offset: u64,
        kv_bytes: u64,
        kv_norm_offset: u64,
        kv_norm_bytes: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawPrefillCompressorWeights {
        attn_ape_offset: u64,
        attn_ape_bytes: u64,
        attn_kv_offset: u64,
        attn_kv_bytes: u64,
        attn_gate_offset: u64,
        attn_gate_bytes: u64,
        attn_norm_offset: u64,
        attn_norm_bytes: u64,
        indexer_ape_offset: u64,
        indexer_ape_bytes: u64,
        indexer_kv_offset: u64,
        indexer_kv_bytes: u64,
        indexer_gate_offset: u64,
        indexer_gate_bytes: u64,
        indexer_norm_offset: u64,
        indexer_norm_bytes: u64,
    }

    #[repr(C)]
    struct RawPrefillLayer2AttentionWeights {
        q_b_offset: u64,
        q_b_bytes: u64,
        attn_sinks_offset: u64,
        attn_sinks_bytes: u64,
        attn_output_a_offset: u64,
        attn_output_a_bytes: u64,
        attn_output_b_offset: u64,
        attn_output_b_bytes: u64,
        ffn: RawPrefillFfnWeights,
        layer3_kvnorm: RawPrefillKvnormWeights,
        layer3_q_b_offset: u64,
        layer3_q_b_bytes: u64,
        layer3_compressor: RawPrefillCompressorWeights,
        layer3_attn_sinks_offset: u64,
        layer3_attn_sinks_bytes: u64,
        layer3_attn_output_a_offset: u64,
        layer3_attn_output_a_bytes: u64,
        layer3_attn_output_b_offset: u64,
        layer3_attn_output_b_bytes: u64,
        layer3_ffn: RawPrefillFfnWeights,
        layer4_kvnorm: RawPrefillKvnormWeights,
        layer4_q_b_offset: u64,
        layer4_q_b_bytes: u64,
        layer4_compressor: RawPrefillCompressorWeights,
        layer4_attn_sinks_offset: u64,
        layer4_attn_sinks_bytes: u64,
        layer4_attn_output_a_offset: u64,
        layer4_attn_output_a_bytes: u64,
        layer4_attn_output_b_offset: u64,
        layer4_attn_output_b_bytes: u64,
        layer4_ffn: RawPrefillFfnWeights,
        layer5_kvnorm: RawPrefillKvnormWeights,
        layer5_q_b_offset: u64,
        layer5_q_b_bytes: u64,
        layer5_compressor: RawPrefillCompressorWeights,
        layer5_attn_sinks_offset: u64,
        layer5_attn_sinks_bytes: u64,
        layer5_attn_output_a_offset: u64,
        layer5_attn_output_a_bytes: u64,
        layer5_attn_output_b_offset: u64,
        layer5_attn_output_b_bytes: u64,
        layer5_ffn: RawPrefillFfnWeights,
        layer6_kvnorm: RawPrefillKvnormWeights,
        layer6_q_b_offset: u64,
        layer6_q_b_bytes: u64,
        layer6_compressor: RawPrefillCompressorWeights,
        layer6_attn_sinks_offset: u64,
        layer6_attn_sinks_bytes: u64,
        layer6_attn_output_a_offset: u64,
        layer6_attn_output_a_bytes: u64,
        layer6_attn_output_b_offset: u64,
        layer6_attn_output_b_bytes: u64,
        layer6_ffn: RawPrefillFfnWeights,
        layer7_kvnorm: RawPrefillKvnormWeights,
        layer7_q_b_offset: u64,
        layer7_q_b_bytes: u64,
        layer7_compressor: RawPrefillCompressorWeights,
        layer7_attn_sinks_offset: u64,
        layer7_attn_sinks_bytes: u64,
        layer7_attn_output_a_offset: u64,
        layer7_attn_output_a_bytes: u64,
        layer7_attn_output_b_offset: u64,
        layer7_attn_output_b_bytes: u64,
        layer7_ffn: RawPrefillFfnWeights,
        layer8_kvnorm: RawPrefillKvnormWeights,
        layer8_q_b_offset: u64,
        layer8_q_b_bytes: u64,
        layer8_compressor: RawPrefillCompressorWeights,
        layer8_attn_sinks_offset: u64,
        layer8_attn_sinks_bytes: u64,
        layer8_attn_output_a_offset: u64,
        layer8_attn_output_a_bytes: u64,
        layer8_attn_output_b_offset: u64,
        layer8_attn_output_b_bytes: u64,
        layer8_ffn: RawPrefillFfnWeights,
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawPrefillLayer2AttentionResult {
        rows: u32,
        raw_kv_rows: u32,
        compressed_kv_rows: u32,
        dispatches: u32,
        wrapped_model_ranges: u32,
        pointer_matches: u32,
        layer3_compressed_kv_rows: u32,
        wall_ms: f64,
        gpu_ms: f64,
    }

    #[repr(C)]
    struct RawPrefillLayerOutputs {
        kv_prefix: *const f32,
        q_lora_norm: *mut f32,
        kv_norm: *mut f32,
        q_cur: *mut f32,
        kv_rope: *mut f32,
        kv_cur: *mut f32,
        attention_output: *mut f32,
        attention_back: *mut f32,
        attention_low: *mut f32,
        attention_out: *mut f32,
        after_attention_hc: *mut f32,
        ffn_cur: *mut f32,
        ffn_norm: *mut f32,
        router_logits: *mut f32,
        router_probs: *mut f32,
        router_selected: *mut i32,
        router_weights: *mut f32,
        routed_mid: *mut f32,
        routed_out: *mut f32,
        shared_out: *mut f32,
        after_ffn_hc: *mut f32,
    }

    #[repr(C)]
    struct RawPrefillLayer0Weights {
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
        qkv: RawPrefillQkvWeights,
        attn_sinks_offset: u64,
        attn_sinks_bytes: u64,
        attn_output_a_offset: u64,
        attn_output_a_bytes: u64,
        attn_output_b_offset: u64,
        attn_output_b_bytes: u64,
        ffn: RawPrefillFfnWeights,
    }

    #[repr(C)]
    #[derive(Default)]
    struct RawPrefillLayer0ProbeResult {
        rows: u64,
        input_elements_per_row: u64,
        q_lora_elements_per_row: u64,
        kv_elements_per_row: u64,
        q_elements_per_row: u64,
        dispatches: u32,
        wrapped_model_ranges: u32,
        pointer_matches: u32,
        position_start: u32,
        raw_cache_rows: u32,
        raw_cache_target_row: u32,
        raw_cache_guard_rows: u32,
        kv_state_mode: u32,
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
    #[derive(Default)]
    struct RawSparseIndexedResult {
        position: u32,
        compressed_rows: u32,
        raw_rows: u32,
        top_k: u32,
        dispatches: u32,
        wrapped_model_ranges: u32,
        pointer_matches: u32,
        split_count: u32,
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
        indexer_q_offset: u64,
        indexer_q_bytes: u64,
        indexer_weight_offset: u64,
        indexer_weight_bytes: u64,
        compressor_prime_attn_norm: *const f32,
        compressed_kv_row: *mut f32,
        compressed_indexer_row: *mut f32,
        kv_norm_pre_rope: *mut f32,
        ffn_mixes: *mut f32,
        ffn_split: *mut f32,
        ffn_cur: *mut f32,
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
        initial_state_mode: u32,
        context_capacity: u32,
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
        fn rust_star_metal_prepare_decoder(
            context: *mut c_void,
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
        fn rust_star_metal_run_prefill_q8_boundary(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            tensor_offset: u64,
            tensor_bytes: u64,
            input_elements_per_row: u32,
            output_elements_per_row: u32,
            rows: u32,
            input: *const f32,
            batch_output: *mut f32,
            decode_output: *mut f32,
            result: *mut RawPrefillQ8ProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_prefill_qkv_boundary(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            weights: *const RawPrefillQkvWeights,
            rows: u32,
            position_start: u32,
            attn_norm: *const f32,
            q_lora: *mut f32,
            q_lora_norm: *mut f32,
            kv_raw: *mut f32,
            kv_norm: *mut f32,
            q_raw: *mut f32,
            q_cur: *mut f32,
            result: *mut RawPrefillQkvProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_prefill_layer0_boundary(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            weights: *const RawPrefillLayer0Weights,
            next_ingress: *const RawPrefillAttentionIngressWeights,
            next_layer: *const RawPrefillLayerWeights,
            next_outputs: *const RawPrefillLayerOutputs,
            layer2_kvnorm: *const RawPrefillKvnormWeights,
            layer2_compressors: *const RawPrefillCompressorWeights,
            layer2_kv_norm_output: *mut f32,
            layer2_kv_rope_output: *mut f32,
            layer2_kv_cur_output: *mut f32,
            layer2_kv_prefix: *const f32,
            layer2_attn_compressed_output: *mut f32,
            layer2_indexer_compressed_output: *mut f32,
            layer2_attn_state_kv_output: *mut f32,
            layer2_attn_state_score_output: *mut f32,
            layer2_indexer_state_kv_output: *mut f32,
            layer2_indexer_state_score_output: *mut f32,
            layer2_attn_compressed_prefix: *const f32,
            layer2_indexer_compressed_prefix: *const f32,
            n_vocab: u32,
            rows: u32,
            position_start: u32,
            kv_state_mode: u32,
            tokens: *const u32,
            hc_collapsed: *mut f32,
            attn_norm: *mut f32,
            q_lora: *mut f32,
            q_lora_norm: *mut f32,
            kv_raw: *mut f32,
            kv_norm: *mut f32,
            q_raw: *mut f32,
            q_cur: *mut f32,
            kv_rope: *mut f32,
            kv_cur: *mut f32,
            raw_cache: *mut f32,
            kv_prefix: *const f32,
            full_kv: *mut f32,
            attention_output: *mut f32,
            attention_back: *mut f32,
            attention_low: *mut f32,
            attention_out: *mut f32,
            after_attention_hc: *mut f32,
            ffn_cur: *mut f32,
            ffn_norm: *mut f32,
            router_logits: *mut f32,
            router_probs: *mut f32,
            router_selected: *mut i32,
            router_weights: *mut f32,
            routed_mid: *mut f32,
            routed_out: *mut f32,
            shared_out: *mut f32,
            after_ffn_hc: *mut f32,
            next_hc_collapsed: *mut f32,
            next_attn_norm: *mut f32,
            next_q_lora: *mut f32,
            result: *mut RawPrefillLayer0ProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_prefill_layer2_attention(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            weights: *const RawPrefillLayer2AttentionWeights,
            attention_output: *mut f32,
            after_attention_hc: *mut f32,
            after_ffn_hc: *mut f32,
            ffn_cur_final_tile: *mut f32,
            ffn_norm_final_tile: *mut f32,
            router_selected_final_tile: *mut i32,
            router_weights_final_tile: *mut f32,
            routed_out_final_tile: *mut f32,
            shared_out_final_tile: *mut f32,
            layer3_hc_attn_pre: *mut f32,
            layer3_attn_norm: *mut f32,
            layer3_q_lora: *mut f32,
            layer3_q_lora_norm: *mut f32,
            layer3_kv_raw: *mut f32,
            layer3_kv_norm: *mut f32,
            layer3_q_raw_final_tile: *mut f32,
            layer3_q_cur_final_tile: *mut f32,
            layer3_kv_rope: *mut f32,
            layer3_kv_cur: *mut f32,
            layer3_attn_compressed: *mut f32,
            layer3_attn_state_kv: *mut f32,
            layer3_attn_state_score: *mut i32,
            layer3_kqv_out_row0: *mut f32,
            layer3_kqv_back_row0: *mut f32,
            layer3_attn_low_row0: *mut f32,
            layer3_attention_output: *mut f32,
            layer3_after_attention_hc: *mut f32,
            layer3_after_ffn_hc: *mut f32,
            layer3_ffn_cur_final_tile: *mut f32,
            layer3_ffn_norm_final_tile: *mut f32,
            layer3_router_selected_final_tile: *mut i32,
            layer3_router_weights_final_tile: *mut f32,
            layer3_routed_out_final_tile: *mut f32,
            layer3_shared_out_final_tile: *mut f32,
            layer4_hc_attn_pre_final_tile: *mut f32,
            layer4_attn_norm_final_tile: *mut f32,
            layer4_q_lora_final_tile: *mut f32,
            layer4_q_lora_norm_final_tile: *mut f32,
            layer4_kv_raw_final_tile: *mut f32,
            layer4_kv_norm_final_tile: *mut f32,
            layer4_q_raw_final_tile: *mut f32,
            layer4_q_cur_final_tile: *mut f32,
            layer4_kv_rope_final_tile: *mut f32,
            layer4_kv_cur_final_tile: *mut f32,
            layer4_attn_compressed: *mut f32,
            layer4_attn_state_kv: *mut f32,
            layer4_attn_state_score: *mut i32,
            layer4_indexer_compressed: *mut f32,
            layer4_indexer_state_kv: *mut f32,
            layer4_indexer_state_score: *mut i32,
            layer4_kqv_out_row0: *mut f32,
            layer4_kqv_back_row0: *mut f32,
            layer4_attn_low_row0: *mut f32,
            layer4_attention_output: *mut f32,
            layer4_after_attention_hc: *mut f32,
            layer4_after_ffn_hc: *mut f32,
            layer4_ffn_cur_final_tile: *mut f32,
            layer4_ffn_norm_final_tile: *mut f32,
            layer4_router_selected_final_tile: *mut i32,
            layer4_router_weights_final_tile: *mut f32,
            layer4_routed_out_final_tile: *mut f32,
            layer4_shared_out_final_tile: *mut f32,
            layer5_hc_attn_pre_final_tile: *mut f32,
            layer5_attn_norm_final_tile: *mut f32,
            layer5_q_lora_final_tile: *mut f32,
            layer5_q_lora_norm_final_tile: *mut f32,
            layer5_kv_raw_final_tile: *mut f32,
            layer5_kv_norm_final_tile: *mut f32,
            layer5_q_raw_final_tile: *mut f32,
            layer5_q_cur_final_tile: *mut f32,
            layer5_kv_rope_final_tile: *mut f32,
            layer5_kv_cur_final_tile: *mut f32,
            layer5_attn_compressed: *mut f32,
            layer5_attn_state_kv: *mut f32,
            layer5_attn_state_score: *mut i32,
            layer5_kqv_out_row0: *mut f32,
            layer5_kqv_back_row0: *mut f32,
            layer5_attn_low_row0: *mut f32,
            layer5_attention_output: *mut f32,
            layer5_after_attention_hc: *mut f32,
            layer5_after_ffn_hc: *mut f32,
            layer5_ffn_cur_final_tile: *mut f32,
            layer5_ffn_norm_final_tile: *mut f32,
            layer5_router_selected_final_tile: *mut i32,
            layer5_router_weights_final_tile: *mut f32,
            layer5_routed_out_final_tile: *mut f32,
            layer5_shared_out_final_tile: *mut f32,
            layer6_hc_attn_pre_final_tile: *mut f32,
            layer6_attn_norm_final_tile: *mut f32,
            layer6_q_lora_final_tile: *mut f32,
            layer6_q_lora_norm_final_tile: *mut f32,
            layer6_kv_raw_final_tile: *mut f32,
            layer6_kv_norm_final_tile: *mut f32,
            layer6_q_raw_final_tile: *mut f32,
            layer6_q_cur_final_tile: *mut f32,
            layer6_kv_rope_final_tile: *mut f32,
            layer6_kv_cur_final_tile: *mut f32,
            layer6_attn_compressed: *mut f32,
            layer6_attn_state_kv: *mut f32,
            layer6_attn_state_score: *mut i32,
            layer6_indexer_compressed: *mut f32,
            layer6_indexer_state_kv: *mut f32,
            layer6_indexer_state_score: *mut i32,
            layer6_kqv_out_row0: *mut f32,
            layer6_kqv_back_row0: *mut f32,
            layer6_attn_low_row0: *mut f32,
            layer6_attention_output: *mut f32,
            layer6_after_attention_hc: *mut f32,
            layer6_after_ffn_hc: *mut f32,
            layer6_ffn_cur_final_tile: *mut f32,
            layer6_ffn_norm_final_tile: *mut f32,
            layer6_router_selected_final_tile: *mut i32,
            layer6_router_weights_final_tile: *mut f32,
            layer6_routed_out_final_tile: *mut f32,
            layer6_shared_out_final_tile: *mut f32,
            layer7_hc_attn_pre_final_tile: *mut f32,
            layer7_attn_norm_final_tile: *mut f32,
            layer7_q_lora_final_tile: *mut f32,
            layer7_q_lora_norm_final_tile: *mut f32,
            layer7_kv_raw_final_tile: *mut f32,
            layer7_kv_norm_final_tile: *mut f32,
            layer7_q_raw_final_tile: *mut f32,
            layer7_q_cur_final_tile: *mut f32,
            layer7_kv_rope_final_tile: *mut f32,
            layer7_kv_cur_final_tile: *mut f32,
            layer7_attn_compressed: *mut f32,
            layer7_attn_state_kv: *mut f32,
            layer7_attn_state_score: *mut i32,
            layer7_kqv_out_row0: *mut f32,
            layer7_kqv_back_row0: *mut f32,
            layer7_attn_low_row0: *mut f32,
            layer7_attention_output: *mut f32,
            layer7_after_attention_hc: *mut f32,
            layer7_after_ffn_hc: *mut f32,
            layer7_ffn_cur_final_tile: *mut f32,
            layer7_ffn_norm_final_tile: *mut f32,
            layer7_router_selected_final_tile: *mut i32,
            layer7_router_weights_final_tile: *mut f32,
            layer7_routed_out_final_tile: *mut f32,
            layer7_shared_out_final_tile: *mut f32,
            layer8_hc_attn_pre_final_tile: *mut f32,
            layer8_attn_norm_final_tile: *mut f32,
            layer8_q_lora_final_tile: *mut f32,
            layer8_q_lora_norm_final_tile: *mut f32,
            layer8_kv_raw_final_tile: *mut f32,
            layer8_kv_norm_final_tile: *mut f32,
            layer8_q_raw_final_tile: *mut f32,
            layer8_q_cur_final_tile: *mut f32,
            layer8_kv_rope_final_tile: *mut f32,
            layer8_kv_cur_final_tile: *mut f32,
            layer8_attn_compressed: *mut f32,
            layer8_attn_state_kv: *mut f32,
            layer8_attn_state_score: *mut i32,
            layer8_indexer_compressed: *mut f32,
            layer8_indexer_state_kv: *mut f32,
            layer8_indexer_state_score: *mut i32,
            layer8_kqv_out_row0: *mut f32,
            layer8_kqv_back_row0: *mut f32,
            layer8_attn_low_row0: *mut f32,
            layer8_attention_output: *mut f32,
            layer8_after_attention_hc: *mut f32,
            layer8_after_ffn_hc: *mut f32,
            layer8_ffn_cur_final_tile: *mut f32,
            layer8_ffn_norm_final_tile: *mut f32,
            layer8_router_selected_final_tile: *mut i32,
            layer8_router_weights_final_tile: *mut f32,
            layer8_routed_out_final_tile: *mut f32,
            layer8_shared_out_final_tile: *mut f32,
            result: *mut RawPrefillLayer2AttentionResult,
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
            collect_intermediates: u32,
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_copy_compressed_kv_row(
            context: *mut c_void,
            layer_index: u32,
            row_index: u32,
            output: *mut f32,
            output_elements: u64,
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
        fn rust_star_metal_run_sparse_indexed_attention(
            context: *mut c_void,
            model_mapping: *const c_void,
            model_bytes: u64,
            indexer_q_offset: u64,
            indexer_q_bytes: u64,
            indexer_weight_offset: u64,
            indexer_weight_bytes: u64,
            sinks_offset: u64,
            sinks_bytes: u64,
            position: u32,
            compressed_rows: u32,
            q_lora_norm: *const f32,
            attn_norm: *const f32,
            q_current: *const f32,
            raw_cache: *const f32,
            attention_comp_cache: *const f32,
            indexer_comp_cache: *const f32,
            indexer_q: *mut f32,
            indexer_weights: *mut f32,
            indexer_scores: *mut f32,
            indexer_topk: *mut i32,
            kqv_out: *mut f32,
            kqv_back: *mut f32,
            result: *mut RawSparseIndexedResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_seed_retained_sparse_layer2_position4099(
            context: *mut c_void,
            input_hc: *const f32,
            raw_cache_prior: *const f32,
            attention_compressed_prior: *const f32,
            indexer_compressed_prior: *const f32,
            attention_state_kv_pre: *const f32,
            attention_state_score_pre: *const f32,
            indexer_state_kv_pre: *const f32,
            indexer_state_score_pre: *const f32,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_copy_retained_sparse_layer2_position4099(
            context: *mut c_void,
            indexer_q: *mut f32,
            indexer_weights: *mut f32,
            indexer_scores: *mut f32,
            indexer_topk: *mut i32,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_seed_retained_sparse_layer2_position8195(
            context: *mut c_void,
            input_hc: *const f32,
            raw_cache_prior: *const f32,
            attention_compressed_prior: *const f32,
            indexer_compressed_prior: *const f32,
            attention_state_kv_pre: *const f32,
            attention_state_score_pre: *const f32,
            indexer_state_kv_pre: *const f32,
            indexer_state_score_pre: *const f32,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_copy_retained_sparse_layer2_position8195(
            context: *mut c_void,
            indexer_q: *mut f32,
            indexer_weights: *mut f32,
            indexer_scores: *mut f32,
            indexer_topk: *mut i32,
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
        indexer_q: Option<ModelSpan>,
        indexer_weight: Option<ModelSpan>,
        validate_expected: bool,
        initial_state_mode: u32,
        context_capacity: u32,
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
        kv_norm_pre_rope: Vec<f32>,
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
        ffn_cur: Vec<f32>,
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

    struct PreparedOutputHead {
        hc_fn: ModelSpan,
        hc_scale: ModelSpan,
        hc_base: ModelSpan,
        output_norm: ModelSpan,
        output: ModelSpan,
        logits: Vec<f32>,
    }

    impl PreparedOutputHead {
        fn new(model: &MappedModel) -> Result<Self> {
            Ok(Self {
                hc_fn: exact_tensor(model, "output_hc_fn.weight", 1, &[16384, 4])?.into(),
                hc_scale: exact_tensor(model, "output_hc_scale.weight", 0, &[1])?.into(),
                hc_base: exact_tensor(model, "output_hc_base.weight", 0, &[4])?.into(),
                output_norm: exact_tensor(model, "output_norm.weight", 0, &[4096])?.into(),
                output: exact_tensor(model, "output.weight", 8, &[4096, 129280])?.into(),
                logits: vec![0.0_f32; 129280],
            })
        }
    }

    impl PreparedLayerExecution {
        fn new(
            model: &MappedModel,
            layer_index: u32,
            position: u32,
            measured_iterations: u32,
        ) -> Result<Self> {
            Self::new_with_initial_state(
                model,
                layer_index,
                position,
                measured_iterations,
                INITIAL_STATE_CAPTURED,
                128,
            )
        }

        fn new_cold(model: &MappedModel, layer_index: u32) -> Result<Self> {
            Self::new_cold_with_capacity(model, layer_index, 128)
        }

        fn new_cold_with_capacity(
            model: &MappedModel,
            layer_index: u32,
            context_capacity: u32,
        ) -> Result<Self> {
            Self::new_with_initial_state(
                model,
                layer_index,
                1,
                1,
                INITIAL_STATE_COLD,
                context_capacity,
            )
        }

        fn new_with_initial_state(
            model: &MappedModel,
            layer_index: u32,
            position: u32,
            measured_iterations: u32,
            initial_state_mode: u32,
            context_capacity: u32,
        ) -> Result<Self> {
            if initial_state_mode > INITIAL_STATE_COLD {
                return Err(Error::invalid("invalid prepared initial-state mode"));
            }
            if context_capacity == 0 || context_capacity > 1_048_576 + 128 {
                return Err(Error::invalid("invalid prepared context capacity"));
            }
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
                let compressor_prime = if initial_state_mode == INITIAL_STATE_CAPTURED {
                    let prime_bytes = compressor_prime_bytes(layer_index).ok_or_else(|| {
                        Error::invalid(format!(
                            "layer-{layer_index} compressor prime is not captured"
                        ))
                    })?;
                    decode_f32_fixture(
                        prime_bytes,
                        &format!("layer-{layer_index} compressor prime"),
                    )?
                } else {
                    Vec::new()
                };
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
                    compressor_prime,
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
            let (indexer_q, indexer_weight) = if layer_index >= 2 && layer_index % 2 == 0 {
                (
                    Some(span(
                        &tensor_name("indexer.attn_q_b.weight"),
                        1,
                        &[1024, 8192],
                    )?),
                    Some(span(&tensor_name("indexer.proj.weight"), 1, &[4096, 64])?),
                )
            } else {
                (None, None)
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
                indexer_q,
                indexer_weight,
                validate_expected: true,
                initial_state_mode,
                context_capacity,
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
                kv_norm_pre_rope: vec![0.0; 512],
                q_raw: vec![0.0; 32768],
                q_cur: vec![0.0; 32768],
                kv_rope: vec![0.0; 512],
                kv_cur: vec![0.0; 512],
                cache_rows: vec![0.0; 128 * 512],
                attention_raw: vec![0.0; 32768],
                attention_back: vec![0.0; 32768],
                attention_low: vec![0.0; 8192],
                attention_out: vec![0.0; 4096],
                after_attention_hc: vec![0.0; 4 * 4096],
                ffn_mixes: vec![0.0; 24],
                ffn_split: vec![0.0; 24],
                ffn_cur: vec![0.0; 4096],
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

        fn prepare_decoder(&self) -> Result<()> {
            let mut error = [0 as c_char; ERROR_BYTES];
            let prepared =
                unsafe { rust_star_metal_prepare_decoder(self.0, error.as_mut_ptr(), error.len()) };
            if prepared == 0 {
                return Err(Error::invalid(format!(
                    "Metal decoder preparation failed: {}",
                    error_text(&error)
                )));
            }
            Ok(())
        }

        fn compressed_kv_row(&self, layer_index: u32, row_index: u32) -> Result<Vec<f32>> {
            let mut output = vec![0.0_f32; 512];
            let mut error = [0 as c_char; ERROR_BYTES];
            let succeeded = unsafe {
                rust_star_metal_copy_compressed_kv_row(
                    self.0,
                    layer_index,
                    row_index,
                    output.as_mut_ptr(),
                    output.len() as u64,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if succeeded == 0 {
                return Err(Error::invalid(format!(
                    "Metal layer-{layer_index} compressed-cache readback failed: {}",
                    error_text(&error)
                )));
            }
            Ok(output)
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

    pub fn run_prefill_q8_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillQ8BoundaryProbeReport> {
        let tensor = model.tensor(PROJECTION_TENSOR)?;
        let (input, expected_batch, expected_decode) = prefill_q8_boundary_fixture()?;
        let (input_elements, output_elements, rows) = validate_prefill_q8_boundary_inputs(
            model,
            tensor,
            &input,
            &expected_batch,
            &expected_decode,
        )?;
        let mut actual_batch = vec![0.0_f32; expected_batch.len()];
        let mut actual_decode = vec![0.0_f32; expected_decode.len()];
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
        let mut raw = RawPrefillQ8ProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_prefill_q8_boundary(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                tensor.absolute_offset,
                tensor.bytes,
                input_elements,
                output_elements,
                rows,
                input.as_ptr(),
                actual_batch.as_mut_ptr(),
                actual_decode.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal prefill Q8 boundary probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.tensor_offset != tensor.absolute_offset
            || raw.tensor_bytes != tensor.bytes
            || raw.input_elements_per_row != u64::from(input_elements)
            || raw.output_elements_per_row != u64::from(output_elements)
            || raw.rows != u64::from(rows)
            || raw.batch_threads_per_threadgroup != 128
            || raw.batch_threadgroups_x != 4
            || raw.batch_threadgroups_y != 16
        {
            return Err(Error::invalid(
                "Metal prefill Q8 boundary returned unexpected dimensions or dispatch",
            ));
        }
        if raw.no_copy_pointer_match == 0 {
            return Err(Error::invalid(
                "Metal prefill Q8 bytes-no-copy buffer did not retain the mmap pointer",
            ));
        }
        for (index, (actual, expected)) in actual_batch.iter().zip(&expected_batch).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "N128 prefill Q8 C0 mismatch at output {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }
        for (index, (actual, expected)) in actual_decode.iter().zip(&expected_decode).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "sequential Q8 control C0 mismatch at output {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }
        for (name, value) in [
            ("batch_wall_ms", raw.batch_wall_ms),
            ("batch_gpu_ms", raw.batch_gpu_ms),
            ("decode_wall_ms", raw.decode_wall_ms),
            ("decode_gpu_ms", raw.decode_gpu_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal prefill Q8 boundary returned invalid {name}"
                )));
            }
        }
        if raw.batch_wall_ms == 0.0 || raw.decode_wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal prefill Q8 boundary returned a zero wall interval",
            ));
        }
        let final_batch = &actual_batch[actual_batch.len() - output_elements as usize..];
        let mut final_row_mismatches = 0_u64;
        let mut final_row_max_abs_error = 0.0_f32;
        for (batch, decode) in final_batch.iter().zip(&actual_decode) {
            if batch.to_bits() != decode.to_bits() {
                final_row_mismatches += 1;
            }
            final_row_max_abs_error = final_row_max_abs_error.max((batch - decode).abs());
        }
        if final_row_mismatches != u64::from(output_elements)
            || final_row_max_abs_error.to_bits() != 3.62396240234375e-05_f32.to_bits()
        {
            return Err(Error::invalid(
                "prefill/decode Q8 arithmetic boundary differs from the oracle classification",
            ));
        }
        Ok(PrefillQ8BoundaryProbeReport {
            fixture_id: PREFILL_Q8_BOUNDARY_FIXTURE_ID,
            tensor_name: tensor.name.clone(),
            rows: raw.rows,
            input_elements_per_row: raw.input_elements_per_row,
            output_elements_per_row: raw.output_elements_per_row,
            no_copy_pointer_match: true,
            batch_threads_per_threadgroup: raw.batch_threads_per_threadgroup,
            batch_threadgroups_x: raw.batch_threadgroups_x,
            batch_threadgroups_y: raw.batch_threadgroups_y,
            batch_wall_ms: raw.batch_wall_ms,
            batch_gpu_ms: raw.batch_gpu_ms,
            decode_wall_ms: raw.decode_wall_ms,
            decode_gpu_ms: raw.decode_gpu_ms,
            input_checksum: checksum_f32(&input),
            batch_output_checksum: checksum_f32(&actual_batch),
            decode_output_checksum: checksum_f32(&actual_decode),
            final_row_mismatches,
            final_row_max_abs_error,
        })
    }

    pub fn run_prefill_qkv_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillQkvBoundaryProbeReport> {
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = exact_tensor(model, "blk.0.attn_q_a_norm.weight", 0, &[1024])?;
        let kv = exact_tensor(model, "blk.0.attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm_weight = exact_tensor(model, "blk.0.attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        let [input, expected_q, expected_q_norm, expected_kv_raw, expected_kv_norm, expected_q_raw, expected_q_cur] =
            prefill_qkv_boundary_fixture()?;
        let mut actual_q = vec![0.0_f32; expected_q.len()];
        let mut actual_q_norm = vec![0.0_f32; expected_q_norm.len()];
        let mut actual_kv_raw = vec![0.0_f32; expected_kv_raw.len()];
        let mut actual_kv_norm = vec![0.0_f32; expected_kv_norm.len()];
        let mut actual_q_raw = vec![0.0_f32; expected_q_raw.len()];
        let mut actual_q_cur = vec![0.0_f32; expected_q_cur.len()];
        let weights = RawPrefillQkvWeights {
            q_a_offset: q_a.absolute_offset,
            q_a_bytes: q_a.bytes,
            q_a_norm_offset: q_a_norm.absolute_offset,
            q_a_norm_bytes: q_a_norm.bytes,
            kv_offset: kv.absolute_offset,
            kv_bytes: kv.bytes,
            kv_norm_offset: kv_norm_weight.absolute_offset,
            kv_norm_bytes: kv_norm_weight.bytes,
            q_b_offset: q_b.absolute_offset,
            q_b_bytes: q_b.bytes,
        };

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
        let mut raw = RawPrefillQkvProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_prefill_qkv_boundary(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &weights,
                32,
                2016,
                input.as_ptr(),
                actual_q.as_mut_ptr(),
                actual_q_norm.as_mut_ptr(),
                actual_kv_raw.as_mut_ptr(),
                actual_kv_norm.as_mut_ptr(),
                actual_q_raw.as_mut_ptr(),
                actual_q_cur.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal prefill Q/KV boundary probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.rows != 32
            || raw.position_start != 2016
            || raw.input_elements_per_row != 4096
            || raw.q_lora_elements_per_row != 1024
            || raw.kv_elements_per_row != 512
            || raw.q_elements_per_row != 32768
            || raw.dispatches != 5
            || raw.wrapped_model_ranges != 5
            || raw.pointer_matches != 5
        {
            return Err(Error::invalid(
                "Metal prefill Q/KV boundary returned an unexpected schedule or mapping",
            ));
        }
        for (label, actual, expected) in [
            ("q_lora", actual_q.as_slice(), expected_q.as_slice()),
            (
                "q_lora_norm",
                actual_q_norm.as_slice(),
                expected_q_norm.as_slice(),
            ),
            (
                "KVraw",
                actual_kv_raw.as_slice(),
                expected_kv_raw.as_slice(),
            ),
            (
                "KVnorm",
                actual_kv_norm.as_slice(),
                expected_kv_norm.as_slice(),
            ),
            ("Qraw", actual_q_raw.as_slice(), expected_q_raw.as_slice()),
            ("Qcur", actual_q_cur.as_slice(), expected_q_cur.as_slice()),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill Q/KV C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits()
                    )));
                }
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal prefill Q/KV boundary returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal prefill Q/KV boundary returned a zero wall interval",
            ));
        }
        Ok(PrefillQkvBoundaryProbeReport {
            fixture_id: PREFILL_QKV_BOUNDARY_FIXTURE_ID,
            rows: raw.rows,
            position_start: raw.position_start,
            dispatches: raw.dispatches,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            checksums: [
                checksum_f32(&input),
                checksum_f32(&actual_q),
                checksum_f32(&actual_q_norm),
                checksum_f32(&actual_kv_raw),
                checksum_f32(&actual_kv_norm),
                checksum_f32(&actual_q_raw),
                checksum_f32(&actual_q_cur),
            ],
        })
    }

    fn run_prefill_boundary_probe(
        model: &MappedModel,
        layer1_mode: u8,
        position_start: u32,
        shared_context: Option<&Context>,
        kv_state_mode: u32,
        include_layer2_kvnorm: bool,
        include_layer2_kv_state: bool,
        include_layer2_compressors: bool,
    ) -> Result<(
        PrefillLayer0BoundaryProbeReport,
        Option<[u64; 3]>,
        Option<[u64; 20]>,
        Option<[u64; 3]>,
        Option<[u64; 6]>,
    )> {
        let include_layer1 = layer1_mode != 0;
        let complete_layer1 = layer1_mode == 2;
        if layer1_mode > 2 {
            return Err(Error::invalid("invalid prefill layer-1 continuation mode"));
        }
        if kv_state_mode > 2 || (kv_state_mode != 0 && !complete_layer1) {
            return Err(Error::invalid("invalid prefill live-KV state mode"));
        }
        if include_layer2_kvnorm && !complete_layer1 {
            return Err(Error::invalid(
                "prefill layer-2 KVnorm requires a complete layer-1 boundary",
            ));
        }
        if include_layer2_kv_state && !include_layer2_kvnorm {
            return Err(Error::invalid(
                "prefill layer-2 KV state requires the layer-2 KVnorm boundary",
            ));
        }
        if include_layer2_compressors && !include_layer2_kv_state {
            return Err(Error::invalid(
                "prefill layer-2 compressors require retained layer-2 KV state",
            ));
        }
        let final_tile = position_start == 2016;
        let previous_fixture_tile = position_start == 1984;
        let live_kv_tile = kv_state_mode != 0 && position_start < 2016;
        if !final_tile && !(complete_layer1 && (previous_fixture_tile || live_kv_tile)) {
            return Err(Error::invalid(
                "prefill boundary supports the final tile, the previous-tile control, or a live-KV tile",
            ));
        }
        let embedding = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let hc_fn = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = exact_tensor(model, "blk.0.hc_attn_scale.weight", 0, &[3])?;
        let hc_base = exact_tensor(model, "blk.0.hc_attn_base.weight", 0, &[24])?;
        let attn_norm_weight = exact_tensor(model, "blk.0.attn_norm.weight", 0, &[4096])?;
        let q_a = exact_tensor(model, "blk.0.attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = exact_tensor(model, "blk.0.attn_q_a_norm.weight", 0, &[1024])?;
        let kv = exact_tensor(model, "blk.0.attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm_weight = exact_tensor(model, "blk.0.attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        let attn_sinks = exact_tensor(model, "blk.0.attn_sinks.weight", 0, &[64])?;
        let attn_output_a = exact_tensor(model, "blk.0.attn_output_a.weight", 8, &[4096, 8192])?;
        let attn_output_b = exact_tensor(model, "blk.0.attn_output_b.weight", 8, &[8192, 4096])?;
        let ffn_hc_fn = exact_tensor(model, "blk.0.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let ffn_hc_scale = exact_tensor(model, "blk.0.hc_ffn_scale.weight", 0, &[3])?;
        let ffn_hc_base = exact_tensor(model, "blk.0.hc_ffn_base.weight", 0, &[24])?;
        let ffn_norm_weight = exact_tensor(model, "blk.0.ffn_norm.weight", 0, &[4096])?;
        let router_gate = exact_tensor(model, "blk.0.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let router_hash = exact_tensor(model, "blk.0.ffn_gate_tid2eid.weight", 26, &[6, 129280])?;
        let routed_gate =
            exact_tensor(model, "blk.0.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_up = exact_tensor(model, "blk.0.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_down =
            exact_tensor(model, "blk.0.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let shared_gate = exact_tensor(model, "blk.0.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let shared_up = exact_tensor(model, "blk.0.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let shared_down = exact_tensor(model, "blk.0.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer1_weights = if include_layer1 {
            Some((
                exact_tensor(model, "blk.1.hc_attn_fn.weight", 1, &[16384, 24])?,
                exact_tensor(model, "blk.1.hc_attn_scale.weight", 0, &[3])?,
                exact_tensor(model, "blk.1.hc_attn_base.weight", 0, &[24])?,
                exact_tensor(model, "blk.1.attn_norm.weight", 0, &[4096])?,
                exact_tensor(model, "blk.1.attn_q_a.weight", 8, &[4096, 1024])?,
            ))
        } else {
            None
        };
        let (mut tokens, expected_collapsed, expected_attn_norm) = prefill_hc_ingress_fixture()?;
        if !final_tile {
            let full_tokens = decode_u32_fixture(
                PREFILL_FRONTIER_2048_TOKEN_IDS_BYTES,
                "prefill frontier token IDs",
            )?;
            tokens = full_tokens[position_start as usize..position_start as usize + 32].to_vec();
        }
        let [qkv_attn_norm, expected_q, expected_q_norm, expected_kv_raw, expected_kv_norm, expected_q_raw, expected_q_cur] =
            prefill_qkv_boundary_fixture()?;
        let [expected_kv_rope, expected_kv_cur, expected_raw_cache_tile] =
            prefill_kv_state_fixture()?;
        let [mut expected_kv_prefix, expected_attention_output, expected_attention_back] =
            prefill_attention_read_fixture()?;
        expected_kv_prefix.truncate(position_start as usize * 512);
        let [expected_attention_low, expected_attention_projected, expected_attention_hc_post] =
            prefill_attention_output_fixture()?;
        let (expected_ffn, expected_selected) = prefill_ffn_output_fixture()?;
        let expected_layer1 = if include_layer1 {
            Some(prefill_layer1_ingress_fixture()?)
        } else {
            None
        };
        let expected_complete_layer1 = if complete_layer1 {
            Some(prefill_layer1_complete_fixture()?)
        } else {
            None
        };
        let expected_previous_tile = if previous_fixture_tile {
            Some(prefill_layers01_previous_tile_fixture()?)
        } else {
            None
        };
        let expected_layer2_kvnorm = if include_layer2_kvnorm {
            let full = prefill_layer2_kvnorm_fixture()?;
            let start = position_start as usize * 512;
            Some(full[start..start + 32 * 512].to_vec())
        } else {
            None
        };
        let expected_layer2_kv_state = if include_layer2_kv_state {
            Some(prefill_layer2_kv_state_fixture()?)
        } else {
            None
        };
        let expected_layer2_compressors = if include_layer2_compressors {
            Some(prefill_layer2_compressor_fixture()?)
        } else {
            None
        };
        if final_tile
            && expected_attn_norm
                .iter()
                .zip(&qkv_attn_norm)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            return Err(Error::invalid(
                "prefill HC ingress and Q/KV fixtures disagree at their attention-norm seam",
            ));
        }

        let weights = RawPrefillLayer0Weights {
            embedding_offset: embedding.absolute_offset,
            embedding_bytes: embedding.bytes,
            hc_fn_offset: hc_fn.absolute_offset,
            hc_fn_bytes: hc_fn.bytes,
            hc_scale_offset: hc_scale.absolute_offset,
            hc_scale_bytes: hc_scale.bytes,
            hc_base_offset: hc_base.absolute_offset,
            hc_base_bytes: hc_base.bytes,
            attn_norm_offset: attn_norm_weight.absolute_offset,
            attn_norm_bytes: attn_norm_weight.bytes,
            qkv: RawPrefillQkvWeights {
                q_a_offset: q_a.absolute_offset,
                q_a_bytes: q_a.bytes,
                q_a_norm_offset: q_a_norm.absolute_offset,
                q_a_norm_bytes: q_a_norm.bytes,
                kv_offset: kv.absolute_offset,
                kv_bytes: kv.bytes,
                kv_norm_offset: kv_norm_weight.absolute_offset,
                kv_norm_bytes: kv_norm_weight.bytes,
                q_b_offset: q_b.absolute_offset,
                q_b_bytes: q_b.bytes,
            },
            attn_sinks_offset: attn_sinks.absolute_offset,
            attn_sinks_bytes: attn_sinks.bytes,
            attn_output_a_offset: attn_output_a.absolute_offset,
            attn_output_a_bytes: attn_output_a.bytes,
            attn_output_b_offset: attn_output_b.absolute_offset,
            attn_output_b_bytes: attn_output_b.bytes,
            ffn: RawPrefillFfnWeights {
                hc_fn_offset: ffn_hc_fn.absolute_offset,
                hc_fn_bytes: ffn_hc_fn.bytes,
                hc_scale_offset: ffn_hc_scale.absolute_offset,
                hc_scale_bytes: ffn_hc_scale.bytes,
                hc_base_offset: ffn_hc_base.absolute_offset,
                hc_base_bytes: ffn_hc_base.bytes,
                norm_offset: ffn_norm_weight.absolute_offset,
                norm_bytes: ffn_norm_weight.bytes,
                router_gate_offset: router_gate.absolute_offset,
                router_gate_bytes: router_gate.bytes,
                router_hash_offset: router_hash.absolute_offset,
                router_hash_bytes: router_hash.bytes,
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
            },
        };
        let raw_layer1_weights =
            layer1_weights
                .as_ref()
                .map(
                    |(hc_fn, hc_scale, hc_base, norm, q_a)| RawPrefillAttentionIngressWeights {
                        hc_fn_offset: hc_fn.absolute_offset,
                        hc_fn_bytes: hc_fn.bytes,
                        hc_scale_offset: hc_scale.absolute_offset,
                        hc_scale_bytes: hc_scale.bytes,
                        hc_base_offset: hc_base.absolute_offset,
                        hc_base_bytes: hc_base.bytes,
                        norm_offset: norm.absolute_offset,
                        norm_bytes: norm.bytes,
                        q_a_offset: q_a.absolute_offset,
                        q_a_bytes: q_a.bytes,
                    },
                );
        let raw_complete_layer1_weights = if complete_layer1 {
            let q_a_norm = exact_tensor(model, "blk.1.attn_q_a_norm.weight", 0, &[1024])?;
            let kv = exact_tensor(model, "blk.1.attn_kv.weight", 8, &[4096, 512])?;
            let kv_norm = exact_tensor(model, "blk.1.attn_kv_a_norm.weight", 0, &[512])?;
            let q_b = exact_tensor(model, "blk.1.attn_q_b.weight", 8, &[1024, 32768])?;
            let sinks = exact_tensor(model, "blk.1.attn_sinks.weight", 0, &[64])?;
            let output_a = exact_tensor(model, "blk.1.attn_output_a.weight", 8, &[4096, 8192])?;
            let output_b = exact_tensor(model, "blk.1.attn_output_b.weight", 8, &[8192, 4096])?;
            let ffn_hc_fn = exact_tensor(model, "blk.1.hc_ffn_fn.weight", 1, &[16384, 24])?;
            let ffn_hc_scale = exact_tensor(model, "blk.1.hc_ffn_scale.weight", 0, &[3])?;
            let ffn_hc_base = exact_tensor(model, "blk.1.hc_ffn_base.weight", 0, &[24])?;
            let ffn_norm = exact_tensor(model, "blk.1.ffn_norm.weight", 0, &[4096])?;
            let router_gate = exact_tensor(model, "blk.1.ffn_gate_inp.weight", 1, &[4096, 256])?;
            let router_hash =
                exact_tensor(model, "blk.1.ffn_gate_tid2eid.weight", 26, &[6, 129280])?;
            let routed_gate =
                exact_tensor(model, "blk.1.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
            let routed_up =
                exact_tensor(model, "blk.1.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
            let routed_down =
                exact_tensor(model, "blk.1.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
            let shared_gate = exact_tensor(model, "blk.1.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
            let shared_up = exact_tensor(model, "blk.1.ffn_up_shexp.weight", 8, &[4096, 2048])?;
            let shared_down = exact_tensor(model, "blk.1.ffn_down_shexp.weight", 8, &[2048, 4096])?;
            Some(RawPrefillLayerWeights {
                ingress: raw_layer1_weights.ok_or_else(|| {
                    Error::invalid("complete layer-1 continuation omitted ingress weights")
                })?,
                q_a_norm_offset: q_a_norm.absolute_offset,
                q_a_norm_bytes: q_a_norm.bytes,
                kv_offset: kv.absolute_offset,
                kv_bytes: kv.bytes,
                kv_norm_offset: kv_norm.absolute_offset,
                kv_norm_bytes: kv_norm.bytes,
                q_b_offset: q_b.absolute_offset,
                q_b_bytes: q_b.bytes,
                attn_sinks_offset: sinks.absolute_offset,
                attn_sinks_bytes: sinks.bytes,
                attn_output_a_offset: output_a.absolute_offset,
                attn_output_a_bytes: output_a.bytes,
                attn_output_b_offset: output_b.absolute_offset,
                attn_output_b_bytes: output_b.bytes,
                ffn: RawPrefillFfnWeights {
                    hc_fn_offset: ffn_hc_fn.absolute_offset,
                    hc_fn_bytes: ffn_hc_fn.bytes,
                    hc_scale_offset: ffn_hc_scale.absolute_offset,
                    hc_scale_bytes: ffn_hc_scale.bytes,
                    hc_base_offset: ffn_hc_base.absolute_offset,
                    hc_base_bytes: ffn_hc_base.bytes,
                    norm_offset: ffn_norm.absolute_offset,
                    norm_bytes: ffn_norm.bytes,
                    router_gate_offset: router_gate.absolute_offset,
                    router_gate_bytes: router_gate.bytes,
                    router_hash_offset: router_hash.absolute_offset,
                    router_hash_bytes: router_hash.bytes,
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
                },
            })
        } else {
            None
        };
        let raw_layer2_kvnorm_weights = if include_layer2_kvnorm {
            let hc_fn = exact_tensor(model, "blk.2.hc_attn_fn.weight", 1, &[16384, 24])?;
            let hc_scale = exact_tensor(model, "blk.2.hc_attn_scale.weight", 0, &[3])?;
            let hc_base = exact_tensor(model, "blk.2.hc_attn_base.weight", 0, &[24])?;
            let norm = exact_tensor(model, "blk.2.attn_norm.weight", 0, &[4096])?;
            let q_a = exact_tensor(model, "blk.2.attn_q_a.weight", 8, &[4096, 1024])?;
            let q_a_norm = exact_tensor(model, "blk.2.attn_q_a_norm.weight", 0, &[1024])?;
            let kv = exact_tensor(model, "blk.2.attn_kv.weight", 8, &[4096, 512])?;
            let kv_norm = exact_tensor(model, "blk.2.attn_kv_a_norm.weight", 0, &[512])?;
            Some(RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: hc_fn.absolute_offset,
                    hc_fn_bytes: hc_fn.bytes,
                    hc_scale_offset: hc_scale.absolute_offset,
                    hc_scale_bytes: hc_scale.bytes,
                    hc_base_offset: hc_base.absolute_offset,
                    hc_base_bytes: hc_base.bytes,
                    norm_offset: norm.absolute_offset,
                    norm_bytes: norm.bytes,
                    q_a_offset: q_a.absolute_offset,
                    q_a_bytes: q_a.bytes,
                },
                q_a_norm_offset: q_a_norm.absolute_offset,
                q_a_norm_bytes: q_a_norm.bytes,
                kv_offset: kv.absolute_offset,
                kv_bytes: kv.bytes,
                kv_norm_offset: kv_norm.absolute_offset,
                kv_norm_bytes: kv_norm.bytes,
            })
        } else {
            None
        };
        let raw_layer2_compressor_weights = if include_layer2_compressors {
            let attn_ape = exact_tensor(model, "blk.2.attn_compressor_ape.weight", 1, &[1024, 4])?;
            let attn_kv = exact_tensor(model, "blk.2.attn_compressor_kv.weight", 1, &[4096, 1024])?;
            let attn_gate =
                exact_tensor(model, "blk.2.attn_compressor_gate.weight", 1, &[4096, 1024])?;
            let attn_norm = exact_tensor(model, "blk.2.attn_compressor_norm.weight", 0, &[512])?;
            let indexer_ape =
                exact_tensor(model, "blk.2.indexer_compressor_ape.weight", 1, &[256, 4])?;
            let indexer_kv =
                exact_tensor(model, "blk.2.indexer_compressor_kv.weight", 1, &[4096, 256])?;
            let indexer_gate = exact_tensor(
                model,
                "blk.2.indexer_compressor_gate.weight",
                1,
                &[4096, 256],
            )?;
            let indexer_norm =
                exact_tensor(model, "blk.2.indexer_compressor_norm.weight", 0, &[128])?;
            Some(RawPrefillCompressorWeights {
                attn_ape_offset: attn_ape.absolute_offset,
                attn_ape_bytes: attn_ape.bytes,
                attn_kv_offset: attn_kv.absolute_offset,
                attn_kv_bytes: attn_kv.bytes,
                attn_gate_offset: attn_gate.absolute_offset,
                attn_gate_bytes: attn_gate.bytes,
                attn_norm_offset: attn_norm.absolute_offset,
                attn_norm_bytes: attn_norm.bytes,
                indexer_ape_offset: indexer_ape.absolute_offset,
                indexer_ape_bytes: indexer_ape.bytes,
                indexer_kv_offset: indexer_kv.absolute_offset,
                indexer_kv_bytes: indexer_kv.bytes,
                indexer_gate_offset: indexer_gate.absolute_offset,
                indexer_gate_bytes: indexer_gate.bytes,
                indexer_norm_offset: indexer_norm.absolute_offset,
                indexer_norm_bytes: indexer_norm.bytes,
            })
        } else {
            None
        };
        let mut actual_collapsed = vec![0.0_f32; expected_collapsed.len()];
        let mut actual_attn_norm = vec![0.0_f32; expected_attn_norm.len()];
        let mut actual_q = vec![0.0_f32; expected_q.len()];
        let mut actual_q_norm = vec![0.0_f32; expected_q_norm.len()];
        let mut actual_kv_raw = vec![0.0_f32; expected_kv_raw.len()];
        let mut actual_kv_norm = vec![0.0_f32; expected_kv_norm.len()];
        let mut actual_q_raw = vec![0.0_f32; expected_q_raw.len()];
        let mut actual_q_cur = vec![0.0_f32; expected_q_cur.len()];
        let mut actual_kv_rope = vec![0.0_f32; expected_kv_rope.len()];
        let mut actual_kv_cur = vec![0.0_f32; expected_kv_cur.len()];
        let mut actual_raw_cache = vec![0.0_f32; 128 * 512];
        let mut actual_full_kv = vec![0.0_f32; 2048 * 512];
        let mut actual_attention_output = vec![0.0_f32; expected_attention_output.len()];
        let mut actual_attention_back = vec![0.0_f32; expected_attention_back.len()];
        let mut actual_attention_low = vec![0.0_f32; expected_attention_low.len()];
        let mut actual_attention_projected = vec![0.0_f32; expected_attention_projected.len()];
        let mut actual_attention_hc_post = vec![0.0_f32; expected_attention_hc_post.len()];
        let mut actual_ffn_cur = vec![0.0_f32; expected_ffn[0].len()];
        let mut actual_ffn_norm = vec![0.0_f32; expected_ffn[1].len()];
        let mut actual_router_logits = vec![0.0_f32; expected_ffn[2].len()];
        let mut actual_router_probs = vec![0.0_f32; expected_ffn[3].len()];
        let mut actual_selected = vec![0_i32; expected_selected.len()];
        let mut actual_router_weights = vec![0.0_f32; expected_ffn[4].len()];
        let mut actual_routed_mid = vec![0.0_f32; expected_ffn[5].len()];
        let mut actual_routed_out = vec![0.0_f32; expected_ffn[6].len()];
        let mut actual_shared_out = vec![0.0_f32; expected_ffn[7].len()];
        let mut actual_ffn_hc_post = vec![0.0_f32; expected_ffn[8].len()];
        let mut actual_layer1_hc = vec![
            0.0_f32;
            expected_layer1
                .as_ref()
                .map_or(0, |tensors| tensors[0].len())
        ];
        let mut actual_layer1_norm = vec![
            0.0_f32;
            expected_layer1
                .as_ref()
                .map_or(0, |tensors| tensors[1].len())
        ];
        let mut actual_layer1_q = vec![
            0.0_f32;
            expected_layer1
                .as_ref()
                .map_or(0, |tensors| tensors[2].len())
        ];
        let mut actual_complete_layer1: Vec<Vec<f32>> = expected_complete_layer1
            .as_ref()
            .map(|(tensors, _)| {
                tensors
                    .iter()
                    .enumerate()
                    .map(|(index, tensor)| {
                        if index == 5 {
                            Vec::new()
                        } else {
                            vec![0.0_f32; tensor.len()]
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut actual_complete_selected = expected_complete_layer1
            .as_ref()
            .map(|(_, selected)| vec![0_i32; selected.len()])
            .unwrap_or_default();
        let mut actual_layer2_kvnorm =
            vec![0.0_f32; expected_layer2_kvnorm.as_ref().map_or(0, Vec::len)];
        let mut actual_layer2_kv_rope =
            vec![0.0_f32; expected_layer2_kv_state.as_ref().map_or(0, |_| 32 * 512)];
        let mut actual_layer2_kv_cur = vec![0.0_f32; actual_layer2_kv_rope.len()];
        let mut actual_layer2_attn_compressed =
            vec![0.0_f32; expected_layer2_compressors.as_ref().map_or(0, |_| 8 * 512)];
        let mut actual_layer2_indexer_compressed =
            vec![0.0_f32; expected_layer2_compressors.as_ref().map_or(0, |_| 8 * 128)];
        let mut actual_layer2_attn_state_kv =
            vec![0.0_f32; expected_layer2_compressors.as_ref().map_or(0, |_| 8 * 1024)];
        let mut actual_layer2_attn_state_score = vec![0.0_f32; actual_layer2_attn_state_kv.len()];
        let mut actual_layer2_indexer_state_kv =
            vec![0.0_f32; expected_layer2_compressors.as_ref().map_or(0, |_| 8 * 256)];
        let mut actual_layer2_indexer_state_score =
            vec![0.0_f32; actual_layer2_indexer_state_kv.len()];
        let mut expected_full_kv = expected_kv_prefix.clone();
        expected_full_kv.extend_from_slice(&expected_kv_cur);
        let raw_complete_layer1_outputs =
            if let Some((expected, _)) = expected_complete_layer1.as_ref() {
                Some(RawPrefillLayerOutputs {
                    kv_prefix: expected[5].as_ptr(),
                    q_lora_norm: actual_complete_layer1[0].as_mut_ptr(),
                    kv_norm: actual_complete_layer1[1].as_mut_ptr(),
                    q_cur: actual_complete_layer1[2].as_mut_ptr(),
                    kv_rope: actual_complete_layer1[3].as_mut_ptr(),
                    kv_cur: actual_complete_layer1[4].as_mut_ptr(),
                    attention_output: actual_complete_layer1[6].as_mut_ptr(),
                    attention_back: actual_complete_layer1[7].as_mut_ptr(),
                    attention_low: actual_complete_layer1[8].as_mut_ptr(),
                    attention_out: actual_complete_layer1[9].as_mut_ptr(),
                    after_attention_hc: actual_complete_layer1[10].as_mut_ptr(),
                    ffn_cur: actual_complete_layer1[11].as_mut_ptr(),
                    ffn_norm: actual_complete_layer1[12].as_mut_ptr(),
                    router_logits: actual_complete_layer1[13].as_mut_ptr(),
                    router_probs: actual_complete_layer1[14].as_mut_ptr(),
                    router_selected: actual_complete_selected.as_mut_ptr(),
                    router_weights: actual_complete_layer1[15].as_mut_ptr(),
                    routed_mid: actual_complete_layer1[16].as_mut_ptr(),
                    routed_out: actual_complete_layer1[17].as_mut_ptr(),
                    shared_out: actual_complete_layer1[18].as_mut_ptr(),
                    after_ffn_hc: actual_complete_layer1[19].as_mut_ptr(),
                })
            } else {
                None
            };

        let owned_context = if shared_context.is_none() {
            Some(Context::new()?)
        } else {
            None
        };
        let context = shared_context
            .or(owned_context.as_ref())
            .ok_or_else(|| Error::invalid("prefill boundary failed to acquire a Metal context"))?;
        let mut error = [0 as c_char; ERROR_BYTES];
        error.fill(0);
        let mut raw = RawPrefillLayer0ProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_prefill_layer0_boundary(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &weights,
                if complete_layer1 {
                    ptr::null()
                } else {
                    raw_layer1_weights
                        .as_ref()
                        .map_or(ptr::null(), |weights| weights as *const _)
                },
                raw_complete_layer1_weights
                    .as_ref()
                    .map_or(ptr::null(), |weights| weights as *const _),
                raw_complete_layer1_outputs
                    .as_ref()
                    .map_or(ptr::null(), |outputs| outputs as *const _),
                raw_layer2_kvnorm_weights
                    .as_ref()
                    .map_or(ptr::null(), |weights| weights as *const _),
                raw_layer2_compressor_weights
                    .as_ref()
                    .map_or(ptr::null(), |weights| weights as *const _),
                if include_layer2_kvnorm {
                    actual_layer2_kvnorm.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_kv_state {
                    actual_layer2_kv_rope.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_kv_state {
                    actual_layer2_kv_cur.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if let Some(expected) = expected_layer2_kv_state.as_ref() {
                    expected[1].as_ptr()
                } else {
                    ptr::null()
                },
                if include_layer2_compressors {
                    actual_layer2_attn_compressed.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_compressors {
                    actual_layer2_indexer_compressed.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_compressors {
                    actual_layer2_attn_state_kv.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_compressors {
                    actual_layer2_attn_state_score.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_compressors {
                    actual_layer2_indexer_state_kv.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer2_compressors {
                    actual_layer2_indexer_state_score.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if let Some(expected) = expected_layer2_compressors.as_ref() {
                    expected[0].as_ptr()
                } else {
                    ptr::null()
                },
                if let Some(expected) = expected_layer2_compressors.as_ref() {
                    expected[3].as_ptr()
                } else {
                    ptr::null()
                },
                129280,
                32,
                position_start,
                kv_state_mode,
                tokens.as_ptr(),
                actual_collapsed.as_mut_ptr(),
                actual_attn_norm.as_mut_ptr(),
                actual_q.as_mut_ptr(),
                actual_q_norm.as_mut_ptr(),
                actual_kv_raw.as_mut_ptr(),
                actual_kv_norm.as_mut_ptr(),
                actual_q_raw.as_mut_ptr(),
                actual_q_cur.as_mut_ptr(),
                actual_kv_rope.as_mut_ptr(),
                actual_kv_cur.as_mut_ptr(),
                actual_raw_cache.as_mut_ptr(),
                expected_kv_prefix.as_ptr(),
                actual_full_kv.as_mut_ptr(),
                actual_attention_output.as_mut_ptr(),
                actual_attention_back.as_mut_ptr(),
                actual_attention_low.as_mut_ptr(),
                actual_attention_projected.as_mut_ptr(),
                actual_attention_hc_post.as_mut_ptr(),
                actual_ffn_cur.as_mut_ptr(),
                actual_ffn_norm.as_mut_ptr(),
                actual_router_logits.as_mut_ptr(),
                actual_router_probs.as_mut_ptr(),
                actual_selected.as_mut_ptr(),
                actual_router_weights.as_mut_ptr(),
                actual_routed_mid.as_mut_ptr(),
                actual_routed_out.as_mut_ptr(),
                actual_shared_out.as_mut_ptr(),
                actual_ffn_hc_post.as_mut_ptr(),
                if include_layer1 {
                    actual_layer1_hc.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer1 {
                    actual_layer1_norm.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if include_layer1 {
                    actual_layer1_q.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal prefill layer-0 boundary probe failed: {}",
                error_text(&error)
            )));
        }
        let expected_dispatches = if include_layer2_compressors {
            if final_tile {
                122
            } else {
                118
            }
        } else if include_layer2_kv_state {
            92
        } else if include_layer2_kvnorm {
            90
        } else if complete_layer1 {
            84
        } else if include_layer1 {
            47
        } else {
            43
        };
        let expected_model_ranges = if include_layer2_compressors {
            65
        } else if include_layer2_kvnorm {
            57
        } else if complete_layer1 {
            49
        } else if include_layer1 {
            30
        } else {
            25
        };
        if raw.rows != 32
            || raw.position_start != position_start
            || raw.input_elements_per_row != 4096
            || raw.q_lora_elements_per_row != 1024
            || raw.kv_elements_per_row != 512
            || raw.q_elements_per_row != 32768
            || raw.dispatches != expected_dispatches
            || raw.wrapped_model_ranges != expected_model_ranges
            || raw.pointer_matches != expected_model_ranges
            || raw.raw_cache_rows != 128
            || raw.raw_cache_target_row != position_start % 128
            || raw.raw_cache_guard_rows != position_start % 128
            || raw.kv_state_mode != kv_state_mode
        {
            return Err(Error::invalid(
                "Metal prefill boundary returned an unexpected schedule or mapping",
            ));
        }
        if final_tile {
            for (label, actual, expected) in [
                (
                    "hc_attn_pre",
                    actual_collapsed.as_slice(),
                    expected_collapsed.as_slice(),
                ),
                (
                    "attn_norm",
                    actual_attn_norm.as_slice(),
                    expected_attn_norm.as_slice(),
                ),
                ("q_lora", actual_q.as_slice(), expected_q.as_slice()),
                (
                    "q_lora_norm",
                    actual_q_norm.as_slice(),
                    expected_q_norm.as_slice(),
                ),
                (
                    "KVraw",
                    actual_kv_raw.as_slice(),
                    expected_kv_raw.as_slice(),
                ),
                (
                    "KVnorm",
                    actual_kv_norm.as_slice(),
                    expected_kv_norm.as_slice(),
                ),
                ("Qraw", actual_q_raw.as_slice(), expected_q_raw.as_slice()),
                ("Qcur", actual_q_cur.as_slice(), expected_q_cur.as_slice()),
                (
                    "KVrope",
                    actual_kv_rope.as_slice(),
                    expected_kv_rope.as_slice(),
                ),
                (
                    "KVcur",
                    actual_kv_cur.as_slice(),
                    expected_kv_cur.as_slice(),
                ),
                (
                    "raw_cache",
                    &actual_raw_cache[96 * 512..128 * 512],
                    expected_raw_cache_tile.as_slice(),
                ),
                (
                    "full_KVcur",
                    actual_full_kv.as_slice(),
                    expected_full_kv.as_slice(),
                ),
                (
                    "kqv_out",
                    actual_attention_output.as_slice(),
                    expected_attention_output.as_slice(),
                ),
                (
                    "kqv_back",
                    actual_attention_back.as_slice(),
                    expected_attention_back.as_slice(),
                ),
                (
                    "attn_low",
                    actual_attention_low.as_slice(),
                    expected_attention_low.as_slice(),
                ),
                (
                    "attn_out",
                    actual_attention_projected.as_slice(),
                    expected_attention_projected.as_slice(),
                ),
                (
                    "hc_attn_post",
                    actual_attention_hc_post.as_slice(),
                    expected_attention_hc_post.as_slice(),
                ),
                (
                    "hc_ffn_pre",
                    actual_ffn_cur.as_slice(),
                    expected_ffn[0].as_slice(),
                ),
                (
                    "ffn_norm",
                    actual_ffn_norm.as_slice(),
                    expected_ffn[1].as_slice(),
                ),
                (
                    "ffn_moe_logits",
                    actual_router_logits.as_slice(),
                    expected_ffn[2].as_slice(),
                ),
                (
                    "ffn_moe_probs",
                    actual_router_probs.as_slice(),
                    expected_ffn[3].as_slice(),
                ),
                (
                    "ffn_moe_weights_scaled",
                    actual_router_weights.as_slice(),
                    expected_ffn[4].as_slice(),
                ),
                (
                    "ffn_moe_weighted_swiglu",
                    actual_routed_mid.as_slice(),
                    expected_ffn[5].as_slice(),
                ),
                (
                    "ffn_moe_out",
                    actual_routed_out.as_slice(),
                    expected_ffn[6].as_slice(),
                ),
                (
                    "ffn_shexp",
                    actual_shared_out.as_slice(),
                    expected_ffn[7].as_slice(),
                ),
                (
                    "hc_ffn_post",
                    actual_ffn_hc_post.as_slice(),
                    expected_ffn[8].as_slice(),
                ),
            ] {
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(Error::invalid(format!(
                            "prefill layer-0 C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                            actual.to_bits(), expected.to_bits()
                        )));
                    }
                }
            }
            for (index, (actual, expected)) in
                actual_selected.iter().zip(&expected_selected).enumerate()
            {
                if actual != expected {
                    return Err(Error::invalid(format!(
                        "prefill layer-0 C0 mismatch in ffn_moe_topk[{index}]: actual={actual} expected={expected}"
                    )));
                }
            }
        }
        let layer1_checksums = if let Some(expected) = expected_layer1.as_ref() {
            if final_tile {
                for (label, actual, expected) in [
                    (
                        "layer1_hc_attn_pre",
                        actual_layer1_hc.as_slice(),
                        expected[0].as_slice(),
                    ),
                    (
                        "layer1_attn_norm",
                        actual_layer1_norm.as_slice(),
                        expected[1].as_slice(),
                    ),
                    (
                        "layer1_q_lora",
                        actual_layer1_q.as_slice(),
                        expected[2].as_slice(),
                    ),
                ] {
                    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        if actual.to_bits() != expected.to_bits() {
                            return Err(Error::invalid(format!(
                                "prefill layers-0/1 C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                                actual.to_bits(), expected.to_bits()
                            )));
                        }
                    }
                }
            }
            Some([
                checksum_f32(&actual_layer1_hc),
                checksum_f32(&actual_layer1_norm),
                checksum_f32(&actual_layer1_q),
            ])
        } else {
            None
        };
        let complete_layer1_checksums = if let Some((expected, expected_selected)) =
            expected_complete_layer1.as_ref()
        {
            if final_tile {
                let labels = [
                    "q_lora_norm",
                    "KVnorm",
                    "Qcur",
                    "KVrope",
                    "KVcur",
                    "kv_prefix_input",
                    "kqv_out",
                    "kqv_back",
                    "attn_low",
                    "attn_out",
                    "hc_attn_post",
                    "hc_ffn_pre",
                    "ffn_norm",
                    "ffn_moe_logits",
                    "ffn_moe_probs",
                    "ffn_moe_weights_scaled",
                    "ffn_moe_weighted_swiglu",
                    "ffn_moe_out",
                    "ffn_shexp",
                    "hc_ffn_post",
                ];
                for index in (0..expected.len()).filter(|index| *index != 5) {
                    for (element, (actual, expected)) in actual_complete_layer1[index]
                        .iter()
                        .zip(&expected[index])
                        .enumerate()
                    {
                        if actual.to_bits() != expected.to_bits() {
                            return Err(Error::invalid(format!(
                                "prefill complete layer-1 C0 mismatch in {}[{element}]: actual={:#010x} expected={:#010x}",
                                labels[index],
                                actual.to_bits(),
                                expected.to_bits()
                            )));
                        }
                    }
                }
                for (index, (actual, expected)) in actual_complete_selected
                    .iter()
                    .zip(expected_selected)
                    .enumerate()
                {
                    if actual != expected {
                        return Err(Error::invalid(format!(
                            "prefill complete layer-1 C0 mismatch in ffn_moe_topk[{index}]: actual={actual} expected={expected}"
                        )));
                    }
                }
            }
            Some([
                checksum_f32(&actual_complete_layer1[0]),
                checksum_f32(&actual_complete_layer1[1]),
                checksum_f32(&actual_complete_layer1[2]),
                checksum_f32(&actual_complete_layer1[3]),
                checksum_f32(&actual_complete_layer1[4]),
                checksum_f32(&actual_complete_layer1[6]),
                checksum_f32(&actual_complete_layer1[7]),
                checksum_f32(&actual_complete_layer1[8]),
                checksum_f32(&actual_complete_layer1[9]),
                checksum_f32(&actual_complete_layer1[10]),
                checksum_f32(&actual_complete_layer1[11]),
                checksum_f32(&actual_complete_layer1[12]),
                checksum_f32(&actual_complete_layer1[13]),
                checksum_f32(&actual_complete_layer1[14]),
                checksum_i32(&actual_complete_selected),
                checksum_f32(&actual_complete_layer1[15]),
                checksum_f32(&actual_complete_layer1[16]),
                checksum_f32(&actual_complete_layer1[17]),
                checksum_f32(&actual_complete_layer1[18]),
                checksum_f32(&actual_complete_layer1[19]),
            ])
        } else {
            None
        };
        if let Some((expected, expected_selected)) = expected_previous_tile.as_ref() {
            for (label, actual, expected) in [
                (
                    "layer0_KVcur",
                    actual_kv_cur.as_slice(),
                    expected[0].as_slice(),
                ),
                (
                    "layer0_hc_ffn_post",
                    actual_ffn_hc_post.as_slice(),
                    expected[1].as_slice(),
                ),
                (
                    "layer1_KVcur",
                    actual_complete_layer1[4].as_slice(),
                    expected[2].as_slice(),
                ),
                (
                    "layer1_hc_ffn_post",
                    actual_complete_layer1[19].as_slice(),
                    expected[3].as_slice(),
                ),
            ] {
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(Error::invalid(format!(
                            "prefill previous-tile C0 mismatch in {label}[{index}]: actual={:#010x} expected={:#010x}",
                            actual.to_bits(),
                            expected.to_bits()
                        )));
                    }
                }
            }
            for (layer, actual, expected) in [
                (
                    0,
                    actual_selected.as_slice(),
                    expected_selected[0].as_slice(),
                ),
                (
                    1,
                    actual_complete_selected.as_slice(),
                    expected_selected[1].as_slice(),
                ),
            ] {
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if actual != expected {
                        return Err(Error::invalid(format!(
                            "prefill previous-tile layer-{layer} selected expert mismatch at {index}: actual={actual} expected={expected}"
                        )));
                    }
                }
            }
        }
        if let Some(expected) = expected_layer2_kvnorm.as_ref() {
            for (index, (actual, expected)) in actual_layer2_kvnorm.iter().zip(expected).enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-2 KVnorm C0 mismatch at position {position_start}, element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits()
                    )));
                }
            }
        }
        if let Some(expected) = expected_layer2_kv_state.as_ref() {
            let start = position_start as usize * 512;
            for (label, actual, expected) in [
                (
                    "KVrope",
                    actual_layer2_kv_rope.as_slice(),
                    &expected[0][start..start + 32 * 512],
                ),
                (
                    "KVcur",
                    actual_layer2_kv_cur.as_slice(),
                    &expected[1][start..start + 32 * 512],
                ),
            ] {
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(Error::invalid(format!(
                            "prefill layer-2 {label} C0 mismatch at position {position_start}, element {index}: actual={:#010x} expected={:#010x}",
                            actual.to_bits(),
                            expected.to_bits()
                        )));
                    }
                }
            }
        }
        if let Some(expected) = expected_layer2_compressors.as_ref() {
            let compressed_start = position_start as usize / 4;
            for (label, actual, expected) in [
                (
                    "attention KVcompress",
                    actual_layer2_attn_compressed.as_slice(),
                    &expected[0][compressed_start * 512..(compressed_start + 8) * 512],
                ),
                (
                    "indexer KVcompress",
                    actual_layer2_indexer_compressed.as_slice(),
                    &expected[3][compressed_start * 128..(compressed_start + 8) * 128],
                ),
            ] {
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(Error::invalid(format!(
                            "prefill layer-2 {label} C0 mismatch at position {position_start}, element {index}: actual={:#010x} expected={:#010x}",
                            actual.to_bits(),
                            expected.to_bits()
                        )));
                    }
                }
            }
            if final_tile {
                for (label, actual, expected) in [
                    (
                        "attention compressor KV state",
                        actual_layer2_attn_state_kv.as_slice(),
                        expected[1].as_slice(),
                    ),
                    (
                        "attention compressor score state",
                        actual_layer2_attn_state_score.as_slice(),
                        expected[2].as_slice(),
                    ),
                    (
                        "indexer compressor KV state",
                        actual_layer2_indexer_state_kv.as_slice(),
                        expected[4].as_slice(),
                    ),
                    (
                        "indexer compressor score state",
                        actual_layer2_indexer_state_score.as_slice(),
                        expected[5].as_slice(),
                    ),
                ] {
                    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        if actual.to_bits() != expected.to_bits() {
                            return Err(Error::invalid(format!(
                                "prefill layer-2 {label} C0 mismatch at final element {index}: actual={:#010x} expected={:#010x}",
                                actual.to_bits(),
                                expected.to_bits()
                            )));
                        }
                    }
                }
            }
        }
        let guard_elements = raw.raw_cache_guard_rows as usize * 512;
        for (index, value) in actual_raw_cache[..guard_elements].iter().enumerate() {
            if value.to_bits() != (-12345.5_f32).to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-0 raw-cache guard changed at element {index}: actual={:#010x}",
                    value.to_bits()
                )));
            }
        }
        for (name, value) in [("wall_ms", raw.wall_ms), ("gpu_ms", raw.gpu_ms)] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal prefill layer-0 boundary returned invalid {name}"
                )));
            }
        }
        if raw.wall_ms == 0.0 {
            return Err(Error::invalid(
                "Metal prefill layer-0 boundary returned a zero wall interval",
            ));
        }
        let layer0 = PrefillLayer0BoundaryProbeReport {
            ingress_fixture_id: PREFILL_HC_INGRESS_FIXTURE_ID,
            qkv_fixture_id: PREFILL_QKV_BOUNDARY_FIXTURE_ID,
            kv_state_fixture_id: PREFILL_KV_STATE_FIXTURE_ID,
            attention_fixture_id: PREFILL_ATTENTION_READ_FIXTURE_ID,
            attention_output_fixture_id: PREFILL_ATTENTION_OUTPUT_FIXTURE_ID,
            ffn_output_fixture_id: PREFILL_FFN_OUTPUT_FIXTURE_ID,
            rows: raw.rows,
            position_start: raw.position_start,
            dispatches: raw.dispatches,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            raw_cache_rows: raw.raw_cache_rows,
            raw_cache_target_row: raw.raw_cache_target_row,
            raw_cache_guard_rows: raw.raw_cache_guard_rows,
            attention_kv_rows: 2048,
            attention_kv_prefix_rows: position_start,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            checksums: [
                checksum_u32(&tokens),
                checksum_f32(&actual_collapsed),
                checksum_f32(&actual_attn_norm),
                checksum_f32(&actual_q),
                checksum_f32(&actual_q_norm),
                checksum_f32(&actual_kv_raw),
                checksum_f32(&actual_kv_norm),
                checksum_f32(&actual_q_raw),
                checksum_f32(&actual_q_cur),
                checksum_f32(&actual_kv_rope),
                checksum_f32(&actual_kv_cur),
                checksum_f32(&actual_raw_cache),
                checksum_f32(&actual_full_kv),
                checksum_f32(&actual_attention_output),
                checksum_f32(&actual_attention_back),
                checksum_f32(&actual_attention_low),
                checksum_f32(&actual_attention_projected),
                checksum_f32(&actual_attention_hc_post),
                checksum_f32(&actual_ffn_cur),
                checksum_f32(&actual_ffn_norm),
                checksum_f32(&actual_router_logits),
                checksum_f32(&actual_router_probs),
                checksum_i32(&actual_selected),
                checksum_f32(&actual_router_weights),
                checksum_f32(&actual_routed_mid),
                checksum_f32(&actual_routed_out),
                checksum_f32(&actual_shared_out),
                checksum_f32(&actual_ffn_hc_post),
            ],
        };
        let layer2_checksums = expected_layer2_kvnorm.as_ref().map(|_| {
            [
                checksum_f32(&actual_layer2_kvnorm),
                checksum_f32(&actual_layer2_kv_rope),
                checksum_f32(&actual_layer2_kv_cur),
            ]
        });
        let layer2_compressor_checksums = expected_layer2_compressors.as_ref().map(|_| {
            [
                checksum_f32(&actual_layer2_attn_compressed),
                checksum_f32(&actual_layer2_indexer_compressed),
                checksum_f32(&actual_layer2_attn_state_kv),
                checksum_f32(&actual_layer2_attn_state_score),
                checksum_f32(&actual_layer2_indexer_state_kv),
                checksum_f32(&actual_layer2_indexer_state_score),
            ]
        });
        Ok((
            layer0,
            layer1_checksums,
            complete_layer1_checksums,
            layer2_checksums,
            layer2_compressor_checksums,
        ))
    }

    pub fn run_prefill_layer0_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayer0BoundaryProbeReport> {
        let (report, layer1, complete, layer2, compressors) =
            run_prefill_boundary_probe(model, 0, 2016, None, 0, false, false, false)?;
        debug_assert!(layer1.is_none());
        debug_assert!(complete.is_none());
        debug_assert!(layer2.is_none());
        debug_assert!(compressors.is_none());
        Ok(report)
    }

    pub fn run_prefill_layers01_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01BoundaryProbeReport> {
        let (layer0, layer1, complete, layer2, compressors) =
            run_prefill_boundary_probe(model, 1, 2016, None, 0, false, false, false)?;
        debug_assert!(complete.is_none());
        debug_assert!(layer2.is_none());
        debug_assert!(compressors.is_none());
        Ok(PrefillLayers01BoundaryProbeReport {
            layer0,
            layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
            checksums: layer1.ok_or_else(|| {
                Error::invalid("prefill layers-0/1 boundary omitted layer-1 checksums")
            })?,
        })
    }

    pub fn run_prefill_layers01_complete_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01CompleteBoundaryProbeReport> {
        let (layer0, layer1, complete, layer2, compressors) =
            run_prefill_boundary_probe(model, 2, 2016, None, 0, false, false, false)?;
        debug_assert!(layer2.is_none());
        debug_assert!(compressors.is_none());
        Ok(PrefillLayers01CompleteBoundaryProbeReport {
            layers01: PrefillLayers01BoundaryProbeReport {
                layer0,
                layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
                checksums: layer1.ok_or_else(|| {
                    Error::invalid("complete prefill boundary omitted layer-1 ingress checksums")
                })?,
            },
            complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
            checksums: complete.ok_or_else(|| {
                Error::invalid("complete prefill boundary omitted layer-1 completion checksums")
            })?,
        })
    }

    pub fn run_prefill_layers01_row_coverage_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01RowCoverageProbeReport> {
        let (previous, previous_ingress, previous_complete, layer2, compressors) =
            run_prefill_boundary_probe(model, 2, 1984, None, 0, false, false, false)?;
        debug_assert!(layer2.is_none());
        debug_assert!(compressors.is_none());
        let previous_ingress = previous_ingress.ok_or_else(|| {
            Error::invalid("previous-tile replay omitted layer-1 ingress checksums")
        })?;
        debug_assert_eq!(previous_ingress.len(), 3);
        let previous_complete = previous_complete.ok_or_else(|| {
            Error::invalid("previous-tile replay omitted layer-1 completion checksums")
        })?;
        let final_tile = run_prefill_layers01_complete_boundary_probe(model)?;
        Ok(PrefillLayers01RowCoverageProbeReport {
            previous_fixture_id: PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE_ID,
            previous_position_start: previous.position_start,
            previous_position_end: previous.position_start + previous.rows as u32 - 1,
            previous_dispatches: previous.dispatches,
            previous_wrapped_model_ranges: previous.wrapped_model_ranges,
            previous_pointer_matches: previous.pointer_matches,
            previous_raw_cache_target_row: previous.raw_cache_target_row,
            previous_wall_ms: previous.wall_ms,
            previous_gpu_ms: previous.gpu_ms,
            previous_checksums: [
                previous.checksums[10],
                previous.checksums[27],
                previous.checksums[22],
                previous_complete[4],
                previous_complete[19],
                previous_complete[14],
            ],
            final_tile,
        })
    }

    pub fn run_prefill_layers01_live_kv_chain_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01LiveKvChainProbeReport> {
        let context = Context::new()?;
        let (previous, previous_ingress, previous_complete, first_layer2, first_compressors) =
            run_prefill_boundary_probe(model, 2, 1984, Some(&context), 1, false, false, false)?;
        debug_assert!(first_layer2.is_none());
        debug_assert!(first_compressors.is_none());
        let previous_ingress = previous_ingress.ok_or_else(|| {
            Error::invalid("live-KV first tile omitted layer-1 ingress checksums")
        })?;
        debug_assert_eq!(previous_ingress.len(), 3);
        let previous_complete = previous_complete.ok_or_else(|| {
            Error::invalid("live-KV first tile omitted layer-1 completion checksums")
        })?;
        let (final_layer0, final_ingress, final_complete, final_layer2, final_compressors) =
            run_prefill_boundary_probe(model, 2, 2016, Some(&context), 2, false, false, false)?;
        debug_assert!(final_layer2.is_none());
        debug_assert!(final_compressors.is_none());
        if previous.position_start + previous.rows as u32 != final_layer0.position_start {
            return Err(Error::invalid("live-KV tiles are not contiguous"));
        }
        let final_tile = PrefillLayers01CompleteBoundaryProbeReport {
            layers01: PrefillLayers01BoundaryProbeReport {
                layer0: final_layer0,
                layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
                checksums: final_ingress.ok_or_else(|| {
                    Error::invalid("live-KV final tile omitted layer-1 ingress checksums")
                })?,
            },
            complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
            checksums: final_complete.ok_or_else(|| {
                Error::invalid("live-KV final tile omitted layer-1 completion checksums")
            })?,
        };
        Ok(PrefillLayers01LiveKvChainProbeReport {
            tiles: PrefillLayers01RowCoverageProbeReport {
                previous_fixture_id: PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE_ID,
                previous_position_start: previous.position_start,
                previous_position_end: previous.position_start + previous.rows as u32 - 1,
                previous_dispatches: previous.dispatches,
                previous_wrapped_model_ranges: previous.wrapped_model_ranges,
                previous_pointer_matches: previous.pointer_matches,
                previous_raw_cache_target_row: previous.raw_cache_target_row,
                previous_wall_ms: previous.wall_ms,
                previous_gpu_ms: previous.gpu_ms,
                previous_checksums: [
                    previous.checksums[10],
                    previous.checksums[27],
                    previous.checksums[22],
                    previous_complete[4],
                    previous_complete[19],
                    previous_complete[14],
                ],
                final_tile,
            },
            retained_kv_rows_after_first_tile: 2016,
            retained_kv_rows_after_final_tile: 2048,
        })
    }

    pub fn run_prefill_layers01_live_kv_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01LiveKvLoopProbeReport> {
        let context = Context::new()?;
        let mut tiles = Vec::with_capacity(64);
        let mut final_tile = None;
        for position_start in (0..2048).step_by(32) {
            let kv_state_mode = if position_start == 0 { 1 } else { 2 };
            let (layer0, layer1, complete, layer2, compressors) = run_prefill_boundary_probe(
                model,
                2,
                position_start,
                Some(&context),
                kv_state_mode,
                false,
                false,
                false,
            )?;
            debug_assert!(layer2.is_none());
            debug_assert!(compressors.is_none());
            let layer1 = layer1.ok_or_else(|| {
                Error::invalid("live-KV loop tile omitted layer-1 ingress checksums")
            })?;
            let complete = complete.ok_or_else(|| {
                Error::invalid("live-KV loop tile omitted layer-1 completion checksums")
            })?;
            if position_start == 2016 {
                final_tile = Some(PrefillLayers01CompleteBoundaryProbeReport {
                    layers01: PrefillLayers01BoundaryProbeReport {
                        layer0: layer0.clone(),
                        layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
                        checksums: layer1,
                    },
                    complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
                    checksums: complete,
                });
            }
            tiles.push(layer0);
        }
        for (index, tile) in tiles.iter().enumerate() {
            let expected_position = index as u32 * 32;
            if tile.position_start != expected_position
                || tile.rows != 32
                || tile.dispatches != 84
                || tile.wrapped_model_ranges != 49
                || tile.pointer_matches != 49
            {
                return Err(Error::invalid(
                    "live-KV loop returned a noncontiguous tile or unexpected schedule",
                ));
            }
        }
        Ok(PrefillLayers01LiveKvLoopProbeReport {
            tiles,
            final_tile: final_tile
                .ok_or_else(|| Error::invalid("live-KV loop omitted the final tile"))?,
        })
    }

    pub fn run_prefill_layers012_kvnorm_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012KvnormLoopProbeReport> {
        let context = Context::new()?;
        let mut tiles = Vec::with_capacity(64);
        let mut layer2_kvnorm_checksums = Vec::with_capacity(64);
        let mut final_tile = None;
        for position_start in (0..2048).step_by(32) {
            let kv_state_mode = if position_start == 0 { 1 } else { 2 };
            let (layer0, layer1, complete, layer2_kvnorm, compressors) =
                run_prefill_boundary_probe(
                    model,
                    2,
                    position_start,
                    Some(&context),
                    kv_state_mode,
                    true,
                    false,
                    false,
                )?;
            debug_assert!(compressors.is_none());
            let layer1 = layer1.ok_or_else(|| {
                Error::invalid("layers-0/1/2 KVnorm loop tile omitted layer-1 ingress checksums")
            })?;
            let complete = complete.ok_or_else(|| {
                Error::invalid("layers-0/1/2 KVnorm loop tile omitted layer-1 completion checksums")
            })?;
            layer2_kvnorm_checksums.push(
                layer2_kvnorm.ok_or_else(|| {
                    Error::invalid("layers-0/1/2 KVnorm loop tile omitted layer-2 KVnorm checksum")
                })?[0],
            );
            if position_start == 2016 {
                final_tile = Some(PrefillLayers01CompleteBoundaryProbeReport {
                    layers01: PrefillLayers01BoundaryProbeReport {
                        layer0: layer0.clone(),
                        layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
                        checksums: layer1,
                    },
                    complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
                    checksums: complete,
                });
            }
            tiles.push(layer0);
        }
        for (index, tile) in tiles.iter().enumerate() {
            if tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != 90
                || tile.wrapped_model_ranges != 57
                || tile.pointer_matches != 57
            {
                return Err(Error::invalid(
                    "layers-0/1/2 KVnorm loop returned a noncontiguous tile or unexpected schedule",
                ));
            }
        }
        Ok(PrefillLayers012KvnormLoopProbeReport {
            tiles,
            layer2_fixture_id: PREFILL_LAYER2_KVNORM_FIXTURE_ID,
            layer2_kvnorm_checksums,
            final_tile: final_tile
                .ok_or_else(|| Error::invalid("layers-0/1/2 KVnorm loop omitted the final tile"))?,
        })
    }

    pub fn run_prefill_layers012_kv_state_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012KvStateLoopProbeReport> {
        let context = Context::new()?;
        let mut tiles = Vec::with_capacity(64);
        let mut layer2_checksums = Vec::with_capacity(64);
        let mut final_tile = None;
        for position_start in (0..2048).step_by(32) {
            let kv_state_mode = if position_start == 0 { 1 } else { 2 };
            let (layer0, layer1, complete, layer2, compressors) = run_prefill_boundary_probe(
                model,
                2,
                position_start,
                Some(&context),
                kv_state_mode,
                true,
                true,
                false,
            )?;
            debug_assert!(compressors.is_none());
            let layer1 = layer1.ok_or_else(|| {
                Error::invalid("layers-0/1/2 KV-state loop omitted layer-1 ingress checksums")
            })?;
            let complete = complete.ok_or_else(|| {
                Error::invalid("layers-0/1/2 KV-state loop omitted layer-1 completion checksums")
            })?;
            layer2_checksums.push(layer2.ok_or_else(|| {
                Error::invalid("layers-0/1/2 KV-state loop omitted layer-2 checksums")
            })?);
            if position_start == 2016 {
                final_tile = Some(PrefillLayers01CompleteBoundaryProbeReport {
                    layers01: PrefillLayers01BoundaryProbeReport {
                        layer0: layer0.clone(),
                        layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
                        checksums: layer1,
                    },
                    complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
                    checksums: complete,
                });
            }
            tiles.push(layer0);
        }
        for (index, tile) in tiles.iter().enumerate() {
            if tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != 92
                || tile.wrapped_model_ranges != 57
                || tile.pointer_matches != 57
            {
                return Err(Error::invalid(
                    "layers-0/1/2 KV-state loop returned a noncontiguous tile or unexpected schedule",
                ));
            }
        }
        Ok(PrefillLayers012KvStateLoopProbeReport {
            tiles,
            layer2_kvnorm_fixture_id: PREFILL_LAYER2_KVNORM_FIXTURE_ID,
            layer2_kv_state_fixture_id: PREFILL_LAYER2_KV_STATE_FIXTURE_ID,
            layer2_checksums,
            final_tile: final_tile.ok_or_else(|| {
                Error::invalid("layers-0/1/2 KV-state loop omitted the final tile")
            })?,
        })
    }

    fn run_prefill_layers012_compressor_loop_probe_in_context(
        model: &MappedModel,
        context: &Context,
    ) -> Result<PrefillLayers012CompressorLoopProbeReport> {
        let mut tiles = Vec::with_capacity(64);
        let mut layer2_checksums = Vec::with_capacity(64);
        let mut layer2_compressor_checksums = Vec::with_capacity(64);
        let mut final_tile = None;
        for position_start in (0..2048).step_by(32) {
            let kv_state_mode = if position_start == 0 { 1 } else { 2 };
            let (layer0, layer1, complete, layer2, compressors) = run_prefill_boundary_probe(
                model,
                2,
                position_start,
                Some(context),
                kv_state_mode,
                true,
                true,
                true,
            )?;
            let layer1 = layer1.ok_or_else(|| {
                Error::invalid("layer-2 compressor loop omitted layer-1 ingress checksums")
            })?;
            let complete = complete.ok_or_else(|| {
                Error::invalid("layer-2 compressor loop omitted layer-1 completion checksums")
            })?;
            layer2_checksums.push(layer2.ok_or_else(|| {
                Error::invalid("layer-2 compressor loop omitted layer-2 KV checksums")
            })?);
            layer2_compressor_checksums.push(compressors.ok_or_else(|| {
                Error::invalid("layer-2 compressor loop omitted compressor checksums")
            })?);
            if position_start == 2016 {
                final_tile = Some(PrefillLayers01CompleteBoundaryProbeReport {
                    layers01: PrefillLayers01BoundaryProbeReport {
                        layer0: layer0.clone(),
                        layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
                        checksums: layer1,
                    },
                    complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
                    checksums: complete,
                });
            }
            tiles.push(layer0);
        }
        for (index, tile) in tiles.iter().enumerate() {
            if tile.position_start != index as u32 * 32
                || tile.rows != 32
                || tile.dispatches != (if index == 63 { 122 } else { 118 })
                || tile.wrapped_model_ranges != 65
                || tile.pointer_matches != 65
            {
                return Err(Error::invalid(
                    "layer-2 compressor loop returned a noncontiguous tile or unexpected schedule",
                ));
            }
        }
        Ok(PrefillLayers012CompressorLoopProbeReport {
            tiles,
            layer2_kvnorm_fixture_id: PREFILL_LAYER2_KVNORM_FIXTURE_ID,
            layer2_kv_state_fixture_id: PREFILL_LAYER2_KV_STATE_FIXTURE_ID,
            layer2_compressor_fixture_id: PREFILL_LAYER2_COMPRESSOR_FIXTURE_ID,
            layer2_checksums,
            layer2_compressor_checksums,
            final_tile: final_tile
                .ok_or_else(|| Error::invalid("layer-2 compressor loop omitted the final tile"))?,
        })
    }

    pub fn run_prefill_layers012_compressor_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012CompressorLoopProbeReport> {
        let context = Context::new()?;
        run_prefill_layers012_compressor_loop_probe_in_context(model, &context)
    }

    pub fn run_prefill_layers012_attention_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012AttentionLoopProbeReport> {
        let context = Context::new()?;
        let compressor = run_prefill_layers012_compressor_loop_probe_in_context(model, &context)?;
        let expected = prefill_layer2_attention_fixture()?;
        let expected_hc_final_tile = prefill_layer2_hc_attn_post_final_tile_fixture()?;
        let expected_ffn_hc_final_tile = prefill_layer2_hc_ffn_post_final_tile_fixture()?;
        let (expected_ffn_cur_final_tile, expected_ffn_norm_final_tile) =
            prefill_layer2_ffn_ingress_final_tile_fixture()?;
        let (expected_router_selected, expected_ffn_outputs) =
            prefill_layer2_ffn_output_final_tile_fixture()?;
        let expected_layer3_ingress = prefill_layer3_ingress_final_tile_fixture()?;
        let expected_layer3_kv_state = prefill_layer3_kv_state_final_tile_fixture()?;
        let expected_layer3_compressor = prefill_layer3_compressor_fixture()?;
        let expected_layer3_attention = prefill_layer3_attention_fixture()?;
        let expected_layer3_attention_diagnostics = prefill_layer3_attention_diagnostics_fixture()?;
        let expected_layer3_complete = prefill_layer3_complete_final_tile_fixture()?;
        let expected_layer4_qkv = prefill_layer4_qkv_final_tile_fixture()?;
        let expected_layer4_compressor = prefill_layer4_compressor_fixture()?;
        let expected_layer4_attention = prefill_layer4_attention_fixture()?;
        let expected_layer4_attention_diagnostics = prefill_layer4_attention_diagnostics_fixture()?;
        let expected_layer4_complete = prefill_layer4_complete_final_tile_fixture()?;
        let expected_layer5_qkv = prefill_layer5_qkv_final_tile_fixture()?;
        let expected_layer5_compressor = prefill_layer5_compressor_fixture()?;
        let expected_layer5_attention = prefill_layer5_attention_fixture()?;
        let expected_layer5_attention_diagnostics = prefill_layer5_attention_diagnostics_fixture()?;
        let expected_layer5_complete = prefill_layer5_complete_final_tile_fixture()?;
        let expected_layer6_qkv = prefill_layer6_qkv_final_tile_fixture()?;
        let expected_layer6_compressor = prefill_layer6_compressor_fixture()?;
        let expected_layer6_attention = prefill_layer6_attention_fixture()?;
        let expected_layer6_attention_diagnostics = prefill_layer6_attention_diagnostics_fixture()?;
        let expected_layer6_complete = prefill_layer6_complete_final_tile_fixture()?;
        let expected_layer7_qkv = prefill_layer7_qkv_final_tile_fixture()?;
        let expected_layer7_compressor = prefill_layer7_compressor_fixture()?;
        let expected_layer7_attention = prefill_layer7_attention_fixture()?;
        let expected_layer7_attention_diagnostics = prefill_layer7_attention_diagnostics_fixture()?;
        let expected_layer7_complete = prefill_layer7_complete_final_tile_fixture()?;
        let expected_layer8_qkv = prefill_layer8_qkv_final_tile_fixture()?;
        let expected_layer8_compressor = prefill_layer8_compressor_fixture()?;
        let expected_layer8_attention = prefill_layer8_attention_fixture()?;
        let expected_layer8_attention_diagnostics = prefill_layer8_attention_diagnostics_fixture()?;
        let expected_layer8_complete = prefill_layer8_complete_final_tile_fixture()?;
        let q_b = exact_tensor(model, "blk.2.attn_q_b.weight", 8, &[1024, 32768])?;
        let sinks = exact_tensor(model, "blk.2.attn_sinks.weight", 0, &[64])?;
        let output_a = exact_tensor(model, "blk.2.attn_output_a.weight", 8, &[4096, 8192])?;
        let output_b = exact_tensor(model, "blk.2.attn_output_b.weight", 8, &[8192, 4096])?;
        let ffn_hc_fn = exact_tensor(model, "blk.2.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let ffn_hc_scale = exact_tensor(model, "blk.2.hc_ffn_scale.weight", 0, &[3])?;
        let ffn_hc_base = exact_tensor(model, "blk.2.hc_ffn_base.weight", 0, &[24])?;
        let ffn_norm = exact_tensor(model, "blk.2.ffn_norm.weight", 0, &[4096])?;
        let router_gate = exact_tensor(model, "blk.2.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let router_hash = exact_tensor(model, "blk.2.ffn_gate_tid2eid.weight", 26, &[6, 129280])?;
        let routed_gate =
            exact_tensor(model, "blk.2.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_up = exact_tensor(model, "blk.2.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_down =
            exact_tensor(model, "blk.2.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let shared_gate = exact_tensor(model, "blk.2.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let shared_up = exact_tensor(model, "blk.2.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let shared_down = exact_tensor(model, "blk.2.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer3_hc_fn = exact_tensor(model, "blk.3.hc_attn_fn.weight", 1, &[16384, 24])?;
        let layer3_hc_scale = exact_tensor(model, "blk.3.hc_attn_scale.weight", 0, &[3])?;
        let layer3_hc_base = exact_tensor(model, "blk.3.hc_attn_base.weight", 0, &[24])?;
        let layer3_norm = exact_tensor(model, "blk.3.attn_norm.weight", 0, &[4096])?;
        let layer3_q_a = exact_tensor(model, "blk.3.attn_q_a.weight", 8, &[4096, 1024])?;
        let layer3_q_a_norm = exact_tensor(model, "blk.3.attn_q_a_norm.weight", 0, &[1024])?;
        let layer3_kv = exact_tensor(model, "blk.3.attn_kv.weight", 8, &[4096, 512])?;
        let layer3_kv_norm = exact_tensor(model, "blk.3.attn_kv_a_norm.weight", 0, &[512])?;
        let layer3_q_b = exact_tensor(model, "blk.3.attn_q_b.weight", 8, &[1024, 32768])?;
        let layer3_attn_ape =
            exact_tensor(model, "blk.3.attn_compressor_ape.weight", 1, &[512, 128])?;
        let layer3_attn_kv =
            exact_tensor(model, "blk.3.attn_compressor_kv.weight", 1, &[4096, 512])?;
        let layer3_attn_gate =
            exact_tensor(model, "blk.3.attn_compressor_gate.weight", 1, &[4096, 512])?;
        let layer3_attn_compressor_norm =
            exact_tensor(model, "blk.3.attn_compressor_norm.weight", 0, &[512])?;
        let layer3_sinks = exact_tensor(model, "blk.3.attn_sinks.weight", 0, &[64])?;
        let layer3_output_a = exact_tensor(model, "blk.3.attn_output_a.weight", 8, &[4096, 8192])?;
        let layer3_output_b = exact_tensor(model, "blk.3.attn_output_b.weight", 8, &[8192, 4096])?;
        let layer3_ffn_hc_fn = exact_tensor(model, "blk.3.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let layer3_ffn_hc_scale = exact_tensor(model, "blk.3.hc_ffn_scale.weight", 0, &[3])?;
        let layer3_ffn_hc_base = exact_tensor(model, "blk.3.hc_ffn_base.weight", 0, &[24])?;
        let layer3_ffn_norm = exact_tensor(model, "blk.3.ffn_norm.weight", 0, &[4096])?;
        let layer3_router_gate = exact_tensor(model, "blk.3.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let layer3_router_bias = exact_tensor(model, "blk.3.exp_probs_b.bias", 0, &[256])?;
        let layer3_routed_gate =
            exact_tensor(model, "blk.3.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let layer3_routed_up =
            exact_tensor(model, "blk.3.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let layer3_routed_down =
            exact_tensor(model, "blk.3.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let layer3_shared_gate =
            exact_tensor(model, "blk.3.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let layer3_shared_up = exact_tensor(model, "blk.3.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let layer3_shared_down =
            exact_tensor(model, "blk.3.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer4_hc_fn = exact_tensor(model, "blk.4.hc_attn_fn.weight", 1, &[16384, 24])?;
        let layer4_hc_scale = exact_tensor(model, "blk.4.hc_attn_scale.weight", 0, &[3])?;
        let layer4_hc_base = exact_tensor(model, "blk.4.hc_attn_base.weight", 0, &[24])?;
        let layer4_norm = exact_tensor(model, "blk.4.attn_norm.weight", 0, &[4096])?;
        let layer4_q_a = exact_tensor(model, "blk.4.attn_q_a.weight", 8, &[4096, 1024])?;
        let layer4_q_a_norm = exact_tensor(model, "blk.4.attn_q_a_norm.weight", 0, &[1024])?;
        let layer4_kv = exact_tensor(model, "blk.4.attn_kv.weight", 8, &[4096, 512])?;
        let layer4_kv_norm = exact_tensor(model, "blk.4.attn_kv_a_norm.weight", 0, &[512])?;
        let layer4_q_b = exact_tensor(model, "blk.4.attn_q_b.weight", 8, &[1024, 32768])?;
        let layer4_attn_ape =
            exact_tensor(model, "blk.4.attn_compressor_ape.weight", 1, &[1024, 4])?;
        let layer4_attn_kv =
            exact_tensor(model, "blk.4.attn_compressor_kv.weight", 1, &[4096, 1024])?;
        let layer4_attn_gate =
            exact_tensor(model, "blk.4.attn_compressor_gate.weight", 1, &[4096, 1024])?;
        let layer4_attn_compressor_norm =
            exact_tensor(model, "blk.4.attn_compressor_norm.weight", 0, &[512])?;
        let layer4_indexer_ape =
            exact_tensor(model, "blk.4.indexer_compressor_ape.weight", 1, &[256, 4])?;
        let layer4_indexer_kv =
            exact_tensor(model, "blk.4.indexer_compressor_kv.weight", 1, &[4096, 256])?;
        let layer4_indexer_gate = exact_tensor(
            model,
            "blk.4.indexer_compressor_gate.weight",
            1,
            &[4096, 256],
        )?;
        let layer4_indexer_compressor_norm =
            exact_tensor(model, "blk.4.indexer_compressor_norm.weight", 0, &[128])?;
        let layer4_sinks = exact_tensor(model, "blk.4.attn_sinks.weight", 0, &[64])?;
        let layer4_output_a = exact_tensor(model, "blk.4.attn_output_a.weight", 8, &[4096, 8192])?;
        let layer4_output_b = exact_tensor(model, "blk.4.attn_output_b.weight", 8, &[8192, 4096])?;
        let layer4_ffn_hc_fn = exact_tensor(model, "blk.4.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let layer4_ffn_hc_scale = exact_tensor(model, "blk.4.hc_ffn_scale.weight", 0, &[3])?;
        let layer4_ffn_hc_base = exact_tensor(model, "blk.4.hc_ffn_base.weight", 0, &[24])?;
        let layer4_ffn_norm = exact_tensor(model, "blk.4.ffn_norm.weight", 0, &[4096])?;
        let layer4_router_gate = exact_tensor(model, "blk.4.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let layer4_router_bias = exact_tensor(model, "blk.4.exp_probs_b.bias", 0, &[256])?;
        let layer4_routed_gate =
            exact_tensor(model, "blk.4.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let layer4_routed_up =
            exact_tensor(model, "blk.4.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let layer4_routed_down =
            exact_tensor(model, "blk.4.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let layer4_shared_gate =
            exact_tensor(model, "blk.4.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let layer4_shared_up = exact_tensor(model, "blk.4.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let layer4_shared_down =
            exact_tensor(model, "blk.4.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer5_hc_fn = exact_tensor(model, "blk.5.hc_attn_fn.weight", 1, &[16384, 24])?;
        let layer5_hc_scale = exact_tensor(model, "blk.5.hc_attn_scale.weight", 0, &[3])?;
        let layer5_hc_base = exact_tensor(model, "blk.5.hc_attn_base.weight", 0, &[24])?;
        let layer5_norm = exact_tensor(model, "blk.5.attn_norm.weight", 0, &[4096])?;
        let layer5_q_a = exact_tensor(model, "blk.5.attn_q_a.weight", 8, &[4096, 1024])?;
        let layer5_q_a_norm = exact_tensor(model, "blk.5.attn_q_a_norm.weight", 0, &[1024])?;
        let layer5_kv = exact_tensor(model, "blk.5.attn_kv.weight", 8, &[4096, 512])?;
        let layer5_kv_norm = exact_tensor(model, "blk.5.attn_kv_a_norm.weight", 0, &[512])?;
        let layer5_q_b = exact_tensor(model, "blk.5.attn_q_b.weight", 8, &[1024, 32768])?;
        let layer5_attn_ape =
            exact_tensor(model, "blk.5.attn_compressor_ape.weight", 1, &[512, 128])?;
        let layer5_attn_kv =
            exact_tensor(model, "blk.5.attn_compressor_kv.weight", 1, &[4096, 512])?;
        let layer5_attn_gate =
            exact_tensor(model, "blk.5.attn_compressor_gate.weight", 1, &[4096, 512])?;
        let layer5_attn_compressor_norm =
            exact_tensor(model, "blk.5.attn_compressor_norm.weight", 0, &[512])?;
        let layer5_sinks = exact_tensor(model, "blk.5.attn_sinks.weight", 0, &[64])?;
        let layer5_output_a = exact_tensor(model, "blk.5.attn_output_a.weight", 8, &[4096, 8192])?;
        let layer5_output_b = exact_tensor(model, "blk.5.attn_output_b.weight", 8, &[8192, 4096])?;
        let layer5_ffn_hc_fn = exact_tensor(model, "blk.5.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let layer5_ffn_hc_scale = exact_tensor(model, "blk.5.hc_ffn_scale.weight", 0, &[3])?;
        let layer5_ffn_hc_base = exact_tensor(model, "blk.5.hc_ffn_base.weight", 0, &[24])?;
        let layer5_ffn_norm = exact_tensor(model, "blk.5.ffn_norm.weight", 0, &[4096])?;
        let layer5_router_gate = exact_tensor(model, "blk.5.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let layer5_router_bias = exact_tensor(model, "blk.5.exp_probs_b.bias", 0, &[256])?;
        let layer5_routed_gate =
            exact_tensor(model, "blk.5.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let layer5_routed_up =
            exact_tensor(model, "blk.5.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let layer5_routed_down =
            exact_tensor(model, "blk.5.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let layer5_shared_gate =
            exact_tensor(model, "blk.5.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let layer5_shared_up = exact_tensor(model, "blk.5.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let layer5_shared_down =
            exact_tensor(model, "blk.5.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer6_hc_fn = exact_tensor(model, "blk.6.hc_attn_fn.weight", 1, &[16384, 24])?;
        let layer6_hc_scale = exact_tensor(model, "blk.6.hc_attn_scale.weight", 0, &[3])?;
        let layer6_hc_base = exact_tensor(model, "blk.6.hc_attn_base.weight", 0, &[24])?;
        let layer6_norm = exact_tensor(model, "blk.6.attn_norm.weight", 0, &[4096])?;
        let layer6_q_a = exact_tensor(model, "blk.6.attn_q_a.weight", 8, &[4096, 1024])?;
        let layer6_q_a_norm = exact_tensor(model, "blk.6.attn_q_a_norm.weight", 0, &[1024])?;
        let layer6_kv = exact_tensor(model, "blk.6.attn_kv.weight", 8, &[4096, 512])?;
        let layer6_kv_norm = exact_tensor(model, "blk.6.attn_kv_a_norm.weight", 0, &[512])?;
        let layer6_q_b = exact_tensor(model, "blk.6.attn_q_b.weight", 8, &[1024, 32768])?;
        let layer6_attn_ape =
            exact_tensor(model, "blk.6.attn_compressor_ape.weight", 1, &[1024, 4])?;
        let layer6_attn_kv =
            exact_tensor(model, "blk.6.attn_compressor_kv.weight", 1, &[4096, 1024])?;
        let layer6_attn_gate =
            exact_tensor(model, "blk.6.attn_compressor_gate.weight", 1, &[4096, 1024])?;
        let layer6_attn_compressor_norm =
            exact_tensor(model, "blk.6.attn_compressor_norm.weight", 0, &[512])?;
        let layer6_indexer_ape =
            exact_tensor(model, "blk.6.indexer_compressor_ape.weight", 1, &[256, 4])?;
        let layer6_indexer_kv =
            exact_tensor(model, "blk.6.indexer_compressor_kv.weight", 1, &[4096, 256])?;
        let layer6_indexer_gate = exact_tensor(
            model,
            "blk.6.indexer_compressor_gate.weight",
            1,
            &[4096, 256],
        )?;
        let layer6_indexer_compressor_norm =
            exact_tensor(model, "blk.6.indexer_compressor_norm.weight", 0, &[128])?;
        let layer6_sinks = exact_tensor(model, "blk.6.attn_sinks.weight", 0, &[64])?;
        let layer6_output_a = exact_tensor(model, "blk.6.attn_output_a.weight", 8, &[4096, 8192])?;
        let layer6_output_b = exact_tensor(model, "blk.6.attn_output_b.weight", 8, &[8192, 4096])?;
        let layer6_ffn_hc_fn = exact_tensor(model, "blk.6.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let layer6_ffn_hc_scale = exact_tensor(model, "blk.6.hc_ffn_scale.weight", 0, &[3])?;
        let layer6_ffn_hc_base = exact_tensor(model, "blk.6.hc_ffn_base.weight", 0, &[24])?;
        let layer6_ffn_norm = exact_tensor(model, "blk.6.ffn_norm.weight", 0, &[4096])?;
        let layer6_router_gate = exact_tensor(model, "blk.6.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let layer6_router_bias = exact_tensor(model, "blk.6.exp_probs_b.bias", 0, &[256])?;
        let layer6_routed_gate =
            exact_tensor(model, "blk.6.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let layer6_routed_up =
            exact_tensor(model, "blk.6.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let layer6_routed_down =
            exact_tensor(model, "blk.6.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let layer6_shared_gate =
            exact_tensor(model, "blk.6.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let layer6_shared_up = exact_tensor(model, "blk.6.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let layer6_shared_down =
            exact_tensor(model, "blk.6.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer7_hc_fn = exact_tensor(model, "blk.7.hc_attn_fn.weight", 1, &[16384, 24])?;
        let layer7_hc_scale = exact_tensor(model, "blk.7.hc_attn_scale.weight", 0, &[3])?;
        let layer7_hc_base = exact_tensor(model, "blk.7.hc_attn_base.weight", 0, &[24])?;
        let layer7_norm = exact_tensor(model, "blk.7.attn_norm.weight", 0, &[4096])?;
        let layer7_q_a = exact_tensor(model, "blk.7.attn_q_a.weight", 8, &[4096, 1024])?;
        let layer7_q_a_norm = exact_tensor(model, "blk.7.attn_q_a_norm.weight", 0, &[1024])?;
        let layer7_kv = exact_tensor(model, "blk.7.attn_kv.weight", 8, &[4096, 512])?;
        let layer7_kv_norm = exact_tensor(model, "blk.7.attn_kv_a_norm.weight", 0, &[512])?;
        let layer7_q_b = exact_tensor(model, "blk.7.attn_q_b.weight", 8, &[1024, 32768])?;
        let layer7_attn_ape =
            exact_tensor(model, "blk.7.attn_compressor_ape.weight", 1, &[512, 128])?;
        let layer7_attn_kv =
            exact_tensor(model, "blk.7.attn_compressor_kv.weight", 1, &[4096, 512])?;
        let layer7_attn_gate =
            exact_tensor(model, "blk.7.attn_compressor_gate.weight", 1, &[4096, 512])?;
        let layer7_attn_compressor_norm =
            exact_tensor(model, "blk.7.attn_compressor_norm.weight", 0, &[512])?;
        let layer7_sinks = exact_tensor(model, "blk.7.attn_sinks.weight", 0, &[64])?;
        let layer7_output_a = exact_tensor(model, "blk.7.attn_output_a.weight", 8, &[4096, 8192])?;
        let layer7_output_b = exact_tensor(model, "blk.7.attn_output_b.weight", 8, &[8192, 4096])?;
        let layer7_ffn_hc_fn = exact_tensor(model, "blk.7.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let layer7_ffn_hc_scale = exact_tensor(model, "blk.7.hc_ffn_scale.weight", 0, &[3])?;
        let layer7_ffn_hc_base = exact_tensor(model, "blk.7.hc_ffn_base.weight", 0, &[24])?;
        let layer7_ffn_norm = exact_tensor(model, "blk.7.ffn_norm.weight", 0, &[4096])?;
        let layer7_router_gate = exact_tensor(model, "blk.7.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let layer7_router_bias = exact_tensor(model, "blk.7.exp_probs_b.bias", 0, &[256])?;
        let layer7_routed_gate =
            exact_tensor(model, "blk.7.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let layer7_routed_up =
            exact_tensor(model, "blk.7.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let layer7_routed_down =
            exact_tensor(model, "blk.7.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let layer7_shared_gate =
            exact_tensor(model, "blk.7.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let layer7_shared_up = exact_tensor(model, "blk.7.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let layer7_shared_down =
            exact_tensor(model, "blk.7.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let layer8_hc_fn = exact_tensor(model, "blk.8.hc_attn_fn.weight", 1, &[16384, 24])?;
        let layer8_hc_scale = exact_tensor(model, "blk.8.hc_attn_scale.weight", 0, &[3])?;
        let layer8_hc_base = exact_tensor(model, "blk.8.hc_attn_base.weight", 0, &[24])?;
        let layer8_norm = exact_tensor(model, "blk.8.attn_norm.weight", 0, &[4096])?;
        let layer8_q_a = exact_tensor(model, "blk.8.attn_q_a.weight", 8, &[4096, 1024])?;
        let layer8_q_a_norm = exact_tensor(model, "blk.8.attn_q_a_norm.weight", 0, &[1024])?;
        let layer8_kv = exact_tensor(model, "blk.8.attn_kv.weight", 8, &[4096, 512])?;
        let layer8_kv_norm = exact_tensor(model, "blk.8.attn_kv_a_norm.weight", 0, &[512])?;
        let layer8_q_b = exact_tensor(model, "blk.8.attn_q_b.weight", 8, &[1024, 32768])?;
        let layer8_attn_ape =
            exact_tensor(model, "blk.8.attn_compressor_ape.weight", 1, &[1024, 4])?;
        let layer8_attn_kv =
            exact_tensor(model, "blk.8.attn_compressor_kv.weight", 1, &[4096, 1024])?;
        let layer8_attn_gate =
            exact_tensor(model, "blk.8.attn_compressor_gate.weight", 1, &[4096, 1024])?;
        let layer8_attn_compressor_norm =
            exact_tensor(model, "blk.8.attn_compressor_norm.weight", 0, &[512])?;
        let layer8_indexer_ape =
            exact_tensor(model, "blk.8.indexer_compressor_ape.weight", 1, &[256, 4])?;
        let layer8_indexer_kv =
            exact_tensor(model, "blk.8.indexer_compressor_kv.weight", 1, &[4096, 256])?;
        let layer8_indexer_gate = exact_tensor(
            model,
            "blk.8.indexer_compressor_gate.weight",
            1,
            &[4096, 256],
        )?;
        let layer8_indexer_compressor_norm =
            exact_tensor(model, "blk.8.indexer_compressor_norm.weight", 0, &[128])?;
        let layer8_sinks = exact_tensor(model, "blk.8.attn_sinks.weight", 0, &[64])?;
        let layer8_output_a = exact_tensor(model, "blk.8.attn_output_a.weight", 8, &[4096, 8192])?;
        let layer8_output_b = exact_tensor(model, "blk.8.attn_output_b.weight", 8, &[8192, 4096])?;
        let layer8_ffn_hc_fn = exact_tensor(model, "blk.8.hc_ffn_fn.weight", 1, &[16384, 24])?;
        let layer8_ffn_hc_scale = exact_tensor(model, "blk.8.hc_ffn_scale.weight", 0, &[3])?;
        let layer8_ffn_hc_base = exact_tensor(model, "blk.8.hc_ffn_base.weight", 0, &[24])?;
        let layer8_ffn_norm = exact_tensor(model, "blk.8.ffn_norm.weight", 0, &[4096])?;
        let layer8_router_gate = exact_tensor(model, "blk.8.ffn_gate_inp.weight", 1, &[4096, 256])?;
        let layer8_router_bias = exact_tensor(model, "blk.8.exp_probs_b.bias", 0, &[256])?;
        let layer8_routed_gate =
            exact_tensor(model, "blk.8.ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let layer8_routed_up =
            exact_tensor(model, "blk.8.ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let layer8_routed_down =
            exact_tensor(model, "blk.8.ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let layer8_shared_gate =
            exact_tensor(model, "blk.8.ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let layer8_shared_up = exact_tensor(model, "blk.8.ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let layer8_shared_down =
            exact_tensor(model, "blk.8.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let weights = RawPrefillLayer2AttentionWeights {
            q_b_offset: q_b.absolute_offset,
            q_b_bytes: q_b.bytes,
            attn_sinks_offset: sinks.absolute_offset,
            attn_sinks_bytes: sinks.bytes,
            attn_output_a_offset: output_a.absolute_offset,
            attn_output_a_bytes: output_a.bytes,
            attn_output_b_offset: output_b.absolute_offset,
            attn_output_b_bytes: output_b.bytes,
            ffn: RawPrefillFfnWeights {
                hc_fn_offset: ffn_hc_fn.absolute_offset,
                hc_fn_bytes: ffn_hc_fn.bytes,
                hc_scale_offset: ffn_hc_scale.absolute_offset,
                hc_scale_bytes: ffn_hc_scale.bytes,
                hc_base_offset: ffn_hc_base.absolute_offset,
                hc_base_bytes: ffn_hc_base.bytes,
                norm_offset: ffn_norm.absolute_offset,
                norm_bytes: ffn_norm.bytes,
                router_gate_offset: router_gate.absolute_offset,
                router_gate_bytes: router_gate.bytes,
                router_hash_offset: router_hash.absolute_offset,
                router_hash_bytes: router_hash.bytes,
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
            },
            layer3_kvnorm: RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: layer3_hc_fn.absolute_offset,
                    hc_fn_bytes: layer3_hc_fn.bytes,
                    hc_scale_offset: layer3_hc_scale.absolute_offset,
                    hc_scale_bytes: layer3_hc_scale.bytes,
                    hc_base_offset: layer3_hc_base.absolute_offset,
                    hc_base_bytes: layer3_hc_base.bytes,
                    norm_offset: layer3_norm.absolute_offset,
                    norm_bytes: layer3_norm.bytes,
                    q_a_offset: layer3_q_a.absolute_offset,
                    q_a_bytes: layer3_q_a.bytes,
                },
                q_a_norm_offset: layer3_q_a_norm.absolute_offset,
                q_a_norm_bytes: layer3_q_a_norm.bytes,
                kv_offset: layer3_kv.absolute_offset,
                kv_bytes: layer3_kv.bytes,
                kv_norm_offset: layer3_kv_norm.absolute_offset,
                kv_norm_bytes: layer3_kv_norm.bytes,
            },
            layer3_q_b_offset: layer3_q_b.absolute_offset,
            layer3_q_b_bytes: layer3_q_b.bytes,
            layer3_compressor: RawPrefillCompressorWeights {
                attn_ape_offset: layer3_attn_ape.absolute_offset,
                attn_ape_bytes: layer3_attn_ape.bytes,
                attn_kv_offset: layer3_attn_kv.absolute_offset,
                attn_kv_bytes: layer3_attn_kv.bytes,
                attn_gate_offset: layer3_attn_gate.absolute_offset,
                attn_gate_bytes: layer3_attn_gate.bytes,
                attn_norm_offset: layer3_attn_compressor_norm.absolute_offset,
                attn_norm_bytes: layer3_attn_compressor_norm.bytes,
                ..RawPrefillCompressorWeights::default()
            },
            layer3_attn_sinks_offset: layer3_sinks.absolute_offset,
            layer3_attn_sinks_bytes: layer3_sinks.bytes,
            layer3_attn_output_a_offset: layer3_output_a.absolute_offset,
            layer3_attn_output_a_bytes: layer3_output_a.bytes,
            layer3_attn_output_b_offset: layer3_output_b.absolute_offset,
            layer3_attn_output_b_bytes: layer3_output_b.bytes,
            layer3_ffn: RawPrefillFfnWeights {
                hc_fn_offset: layer3_ffn_hc_fn.absolute_offset,
                hc_fn_bytes: layer3_ffn_hc_fn.bytes,
                hc_scale_offset: layer3_ffn_hc_scale.absolute_offset,
                hc_scale_bytes: layer3_ffn_hc_scale.bytes,
                hc_base_offset: layer3_ffn_hc_base.absolute_offset,
                hc_base_bytes: layer3_ffn_hc_base.bytes,
                norm_offset: layer3_ffn_norm.absolute_offset,
                norm_bytes: layer3_ffn_norm.bytes,
                router_gate_offset: layer3_router_gate.absolute_offset,
                router_gate_bytes: layer3_router_gate.bytes,
                router_hash_offset: layer3_router_bias.absolute_offset,
                router_hash_bytes: layer3_router_bias.bytes,
                routed_gate_offset: layer3_routed_gate.absolute_offset,
                routed_gate_bytes: layer3_routed_gate.bytes,
                routed_up_offset: layer3_routed_up.absolute_offset,
                routed_up_bytes: layer3_routed_up.bytes,
                routed_down_offset: layer3_routed_down.absolute_offset,
                routed_down_bytes: layer3_routed_down.bytes,
                shared_gate_offset: layer3_shared_gate.absolute_offset,
                shared_gate_bytes: layer3_shared_gate.bytes,
                shared_up_offset: layer3_shared_up.absolute_offset,
                shared_up_bytes: layer3_shared_up.bytes,
                shared_down_offset: layer3_shared_down.absolute_offset,
                shared_down_bytes: layer3_shared_down.bytes,
            },
            layer4_kvnorm: RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: layer4_hc_fn.absolute_offset,
                    hc_fn_bytes: layer4_hc_fn.bytes,
                    hc_scale_offset: layer4_hc_scale.absolute_offset,
                    hc_scale_bytes: layer4_hc_scale.bytes,
                    hc_base_offset: layer4_hc_base.absolute_offset,
                    hc_base_bytes: layer4_hc_base.bytes,
                    norm_offset: layer4_norm.absolute_offset,
                    norm_bytes: layer4_norm.bytes,
                    q_a_offset: layer4_q_a.absolute_offset,
                    q_a_bytes: layer4_q_a.bytes,
                },
                q_a_norm_offset: layer4_q_a_norm.absolute_offset,
                q_a_norm_bytes: layer4_q_a_norm.bytes,
                kv_offset: layer4_kv.absolute_offset,
                kv_bytes: layer4_kv.bytes,
                kv_norm_offset: layer4_kv_norm.absolute_offset,
                kv_norm_bytes: layer4_kv_norm.bytes,
            },
            layer4_q_b_offset: layer4_q_b.absolute_offset,
            layer4_q_b_bytes: layer4_q_b.bytes,
            layer4_compressor: RawPrefillCompressorWeights {
                attn_ape_offset: layer4_attn_ape.absolute_offset,
                attn_ape_bytes: layer4_attn_ape.bytes,
                attn_kv_offset: layer4_attn_kv.absolute_offset,
                attn_kv_bytes: layer4_attn_kv.bytes,
                attn_gate_offset: layer4_attn_gate.absolute_offset,
                attn_gate_bytes: layer4_attn_gate.bytes,
                attn_norm_offset: layer4_attn_compressor_norm.absolute_offset,
                attn_norm_bytes: layer4_attn_compressor_norm.bytes,
                indexer_ape_offset: layer4_indexer_ape.absolute_offset,
                indexer_ape_bytes: layer4_indexer_ape.bytes,
                indexer_kv_offset: layer4_indexer_kv.absolute_offset,
                indexer_kv_bytes: layer4_indexer_kv.bytes,
                indexer_gate_offset: layer4_indexer_gate.absolute_offset,
                indexer_gate_bytes: layer4_indexer_gate.bytes,
                indexer_norm_offset: layer4_indexer_compressor_norm.absolute_offset,
                indexer_norm_bytes: layer4_indexer_compressor_norm.bytes,
            },
            layer4_attn_sinks_offset: layer4_sinks.absolute_offset,
            layer4_attn_sinks_bytes: layer4_sinks.bytes,
            layer4_attn_output_a_offset: layer4_output_a.absolute_offset,
            layer4_attn_output_a_bytes: layer4_output_a.bytes,
            layer4_attn_output_b_offset: layer4_output_b.absolute_offset,
            layer4_attn_output_b_bytes: layer4_output_b.bytes,
            layer4_ffn: RawPrefillFfnWeights {
                hc_fn_offset: layer4_ffn_hc_fn.absolute_offset,
                hc_fn_bytes: layer4_ffn_hc_fn.bytes,
                hc_scale_offset: layer4_ffn_hc_scale.absolute_offset,
                hc_scale_bytes: layer4_ffn_hc_scale.bytes,
                hc_base_offset: layer4_ffn_hc_base.absolute_offset,
                hc_base_bytes: layer4_ffn_hc_base.bytes,
                norm_offset: layer4_ffn_norm.absolute_offset,
                norm_bytes: layer4_ffn_norm.bytes,
                router_gate_offset: layer4_router_gate.absolute_offset,
                router_gate_bytes: layer4_router_gate.bytes,
                router_hash_offset: layer4_router_bias.absolute_offset,
                router_hash_bytes: layer4_router_bias.bytes,
                routed_gate_offset: layer4_routed_gate.absolute_offset,
                routed_gate_bytes: layer4_routed_gate.bytes,
                routed_up_offset: layer4_routed_up.absolute_offset,
                routed_up_bytes: layer4_routed_up.bytes,
                routed_down_offset: layer4_routed_down.absolute_offset,
                routed_down_bytes: layer4_routed_down.bytes,
                shared_gate_offset: layer4_shared_gate.absolute_offset,
                shared_gate_bytes: layer4_shared_gate.bytes,
                shared_up_offset: layer4_shared_up.absolute_offset,
                shared_up_bytes: layer4_shared_up.bytes,
                shared_down_offset: layer4_shared_down.absolute_offset,
                shared_down_bytes: layer4_shared_down.bytes,
            },
            layer5_kvnorm: RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: layer5_hc_fn.absolute_offset,
                    hc_fn_bytes: layer5_hc_fn.bytes,
                    hc_scale_offset: layer5_hc_scale.absolute_offset,
                    hc_scale_bytes: layer5_hc_scale.bytes,
                    hc_base_offset: layer5_hc_base.absolute_offset,
                    hc_base_bytes: layer5_hc_base.bytes,
                    norm_offset: layer5_norm.absolute_offset,
                    norm_bytes: layer5_norm.bytes,
                    q_a_offset: layer5_q_a.absolute_offset,
                    q_a_bytes: layer5_q_a.bytes,
                },
                q_a_norm_offset: layer5_q_a_norm.absolute_offset,
                q_a_norm_bytes: layer5_q_a_norm.bytes,
                kv_offset: layer5_kv.absolute_offset,
                kv_bytes: layer5_kv.bytes,
                kv_norm_offset: layer5_kv_norm.absolute_offset,
                kv_norm_bytes: layer5_kv_norm.bytes,
            },
            layer5_q_b_offset: layer5_q_b.absolute_offset,
            layer5_q_b_bytes: layer5_q_b.bytes,
            layer5_compressor: RawPrefillCompressorWeights {
                attn_ape_offset: layer5_attn_ape.absolute_offset,
                attn_ape_bytes: layer5_attn_ape.bytes,
                attn_kv_offset: layer5_attn_kv.absolute_offset,
                attn_kv_bytes: layer5_attn_kv.bytes,
                attn_gate_offset: layer5_attn_gate.absolute_offset,
                attn_gate_bytes: layer5_attn_gate.bytes,
                attn_norm_offset: layer5_attn_compressor_norm.absolute_offset,
                attn_norm_bytes: layer5_attn_compressor_norm.bytes,
                ..RawPrefillCompressorWeights::default()
            },
            layer5_attn_sinks_offset: layer5_sinks.absolute_offset,
            layer5_attn_sinks_bytes: layer5_sinks.bytes,
            layer5_attn_output_a_offset: layer5_output_a.absolute_offset,
            layer5_attn_output_a_bytes: layer5_output_a.bytes,
            layer5_attn_output_b_offset: layer5_output_b.absolute_offset,
            layer5_attn_output_b_bytes: layer5_output_b.bytes,
            layer5_ffn: RawPrefillFfnWeights {
                hc_fn_offset: layer5_ffn_hc_fn.absolute_offset,
                hc_fn_bytes: layer5_ffn_hc_fn.bytes,
                hc_scale_offset: layer5_ffn_hc_scale.absolute_offset,
                hc_scale_bytes: layer5_ffn_hc_scale.bytes,
                hc_base_offset: layer5_ffn_hc_base.absolute_offset,
                hc_base_bytes: layer5_ffn_hc_base.bytes,
                norm_offset: layer5_ffn_norm.absolute_offset,
                norm_bytes: layer5_ffn_norm.bytes,
                router_gate_offset: layer5_router_gate.absolute_offset,
                router_gate_bytes: layer5_router_gate.bytes,
                router_hash_offset: layer5_router_bias.absolute_offset,
                router_hash_bytes: layer5_router_bias.bytes,
                routed_gate_offset: layer5_routed_gate.absolute_offset,
                routed_gate_bytes: layer5_routed_gate.bytes,
                routed_up_offset: layer5_routed_up.absolute_offset,
                routed_up_bytes: layer5_routed_up.bytes,
                routed_down_offset: layer5_routed_down.absolute_offset,
                routed_down_bytes: layer5_routed_down.bytes,
                shared_gate_offset: layer5_shared_gate.absolute_offset,
                shared_gate_bytes: layer5_shared_gate.bytes,
                shared_up_offset: layer5_shared_up.absolute_offset,
                shared_up_bytes: layer5_shared_up.bytes,
                shared_down_offset: layer5_shared_down.absolute_offset,
                shared_down_bytes: layer5_shared_down.bytes,
            },
            layer6_kvnorm: RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: layer6_hc_fn.absolute_offset,
                    hc_fn_bytes: layer6_hc_fn.bytes,
                    hc_scale_offset: layer6_hc_scale.absolute_offset,
                    hc_scale_bytes: layer6_hc_scale.bytes,
                    hc_base_offset: layer6_hc_base.absolute_offset,
                    hc_base_bytes: layer6_hc_base.bytes,
                    norm_offset: layer6_norm.absolute_offset,
                    norm_bytes: layer6_norm.bytes,
                    q_a_offset: layer6_q_a.absolute_offset,
                    q_a_bytes: layer6_q_a.bytes,
                },
                q_a_norm_offset: layer6_q_a_norm.absolute_offset,
                q_a_norm_bytes: layer6_q_a_norm.bytes,
                kv_offset: layer6_kv.absolute_offset,
                kv_bytes: layer6_kv.bytes,
                kv_norm_offset: layer6_kv_norm.absolute_offset,
                kv_norm_bytes: layer6_kv_norm.bytes,
            },
            layer6_q_b_offset: layer6_q_b.absolute_offset,
            layer6_q_b_bytes: layer6_q_b.bytes,
            layer6_compressor: RawPrefillCompressorWeights {
                attn_ape_offset: layer6_attn_ape.absolute_offset,
                attn_ape_bytes: layer6_attn_ape.bytes,
                attn_kv_offset: layer6_attn_kv.absolute_offset,
                attn_kv_bytes: layer6_attn_kv.bytes,
                attn_gate_offset: layer6_attn_gate.absolute_offset,
                attn_gate_bytes: layer6_attn_gate.bytes,
                attn_norm_offset: layer6_attn_compressor_norm.absolute_offset,
                attn_norm_bytes: layer6_attn_compressor_norm.bytes,
                indexer_ape_offset: layer6_indexer_ape.absolute_offset,
                indexer_ape_bytes: layer6_indexer_ape.bytes,
                indexer_kv_offset: layer6_indexer_kv.absolute_offset,
                indexer_kv_bytes: layer6_indexer_kv.bytes,
                indexer_gate_offset: layer6_indexer_gate.absolute_offset,
                indexer_gate_bytes: layer6_indexer_gate.bytes,
                indexer_norm_offset: layer6_indexer_compressor_norm.absolute_offset,
                indexer_norm_bytes: layer6_indexer_compressor_norm.bytes,
            },
            layer6_attn_sinks_offset: layer6_sinks.absolute_offset,
            layer6_attn_sinks_bytes: layer6_sinks.bytes,
            layer6_attn_output_a_offset: layer6_output_a.absolute_offset,
            layer6_attn_output_a_bytes: layer6_output_a.bytes,
            layer6_attn_output_b_offset: layer6_output_b.absolute_offset,
            layer6_attn_output_b_bytes: layer6_output_b.bytes,
            layer6_ffn: RawPrefillFfnWeights {
                hc_fn_offset: layer6_ffn_hc_fn.absolute_offset,
                hc_fn_bytes: layer6_ffn_hc_fn.bytes,
                hc_scale_offset: layer6_ffn_hc_scale.absolute_offset,
                hc_scale_bytes: layer6_ffn_hc_scale.bytes,
                hc_base_offset: layer6_ffn_hc_base.absolute_offset,
                hc_base_bytes: layer6_ffn_hc_base.bytes,
                norm_offset: layer6_ffn_norm.absolute_offset,
                norm_bytes: layer6_ffn_norm.bytes,
                router_gate_offset: layer6_router_gate.absolute_offset,
                router_gate_bytes: layer6_router_gate.bytes,
                router_hash_offset: layer6_router_bias.absolute_offset,
                router_hash_bytes: layer6_router_bias.bytes,
                routed_gate_offset: layer6_routed_gate.absolute_offset,
                routed_gate_bytes: layer6_routed_gate.bytes,
                routed_up_offset: layer6_routed_up.absolute_offset,
                routed_up_bytes: layer6_routed_up.bytes,
                routed_down_offset: layer6_routed_down.absolute_offset,
                routed_down_bytes: layer6_routed_down.bytes,
                shared_gate_offset: layer6_shared_gate.absolute_offset,
                shared_gate_bytes: layer6_shared_gate.bytes,
                shared_up_offset: layer6_shared_up.absolute_offset,
                shared_up_bytes: layer6_shared_up.bytes,
                shared_down_offset: layer6_shared_down.absolute_offset,
                shared_down_bytes: layer6_shared_down.bytes,
            },
            layer7_kvnorm: RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: layer7_hc_fn.absolute_offset,
                    hc_fn_bytes: layer7_hc_fn.bytes,
                    hc_scale_offset: layer7_hc_scale.absolute_offset,
                    hc_scale_bytes: layer7_hc_scale.bytes,
                    hc_base_offset: layer7_hc_base.absolute_offset,
                    hc_base_bytes: layer7_hc_base.bytes,
                    norm_offset: layer7_norm.absolute_offset,
                    norm_bytes: layer7_norm.bytes,
                    q_a_offset: layer7_q_a.absolute_offset,
                    q_a_bytes: layer7_q_a.bytes,
                },
                q_a_norm_offset: layer7_q_a_norm.absolute_offset,
                q_a_norm_bytes: layer7_q_a_norm.bytes,
                kv_offset: layer7_kv.absolute_offset,
                kv_bytes: layer7_kv.bytes,
                kv_norm_offset: layer7_kv_norm.absolute_offset,
                kv_norm_bytes: layer7_kv_norm.bytes,
            },
            layer7_q_b_offset: layer7_q_b.absolute_offset,
            layer7_q_b_bytes: layer7_q_b.bytes,
            layer7_compressor: RawPrefillCompressorWeights {
                attn_ape_offset: layer7_attn_ape.absolute_offset,
                attn_ape_bytes: layer7_attn_ape.bytes,
                attn_kv_offset: layer7_attn_kv.absolute_offset,
                attn_kv_bytes: layer7_attn_kv.bytes,
                attn_gate_offset: layer7_attn_gate.absolute_offset,
                attn_gate_bytes: layer7_attn_gate.bytes,
                attn_norm_offset: layer7_attn_compressor_norm.absolute_offset,
                attn_norm_bytes: layer7_attn_compressor_norm.bytes,
                ..RawPrefillCompressorWeights::default()
            },
            layer7_attn_sinks_offset: layer7_sinks.absolute_offset,
            layer7_attn_sinks_bytes: layer7_sinks.bytes,
            layer7_attn_output_a_offset: layer7_output_a.absolute_offset,
            layer7_attn_output_a_bytes: layer7_output_a.bytes,
            layer7_attn_output_b_offset: layer7_output_b.absolute_offset,
            layer7_attn_output_b_bytes: layer7_output_b.bytes,
            layer7_ffn: RawPrefillFfnWeights {
                hc_fn_offset: layer7_ffn_hc_fn.absolute_offset,
                hc_fn_bytes: layer7_ffn_hc_fn.bytes,
                hc_scale_offset: layer7_ffn_hc_scale.absolute_offset,
                hc_scale_bytes: layer7_ffn_hc_scale.bytes,
                hc_base_offset: layer7_ffn_hc_base.absolute_offset,
                hc_base_bytes: layer7_ffn_hc_base.bytes,
                norm_offset: layer7_ffn_norm.absolute_offset,
                norm_bytes: layer7_ffn_norm.bytes,
                router_gate_offset: layer7_router_gate.absolute_offset,
                router_gate_bytes: layer7_router_gate.bytes,
                router_hash_offset: layer7_router_bias.absolute_offset,
                router_hash_bytes: layer7_router_bias.bytes,
                routed_gate_offset: layer7_routed_gate.absolute_offset,
                routed_gate_bytes: layer7_routed_gate.bytes,
                routed_up_offset: layer7_routed_up.absolute_offset,
                routed_up_bytes: layer7_routed_up.bytes,
                routed_down_offset: layer7_routed_down.absolute_offset,
                routed_down_bytes: layer7_routed_down.bytes,
                shared_gate_offset: layer7_shared_gate.absolute_offset,
                shared_gate_bytes: layer7_shared_gate.bytes,
                shared_up_offset: layer7_shared_up.absolute_offset,
                shared_up_bytes: layer7_shared_up.bytes,
                shared_down_offset: layer7_shared_down.absolute_offset,
                shared_down_bytes: layer7_shared_down.bytes,
            },
            layer8_kvnorm: RawPrefillKvnormWeights {
                ingress: RawPrefillAttentionIngressWeights {
                    hc_fn_offset: layer8_hc_fn.absolute_offset,
                    hc_fn_bytes: layer8_hc_fn.bytes,
                    hc_scale_offset: layer8_hc_scale.absolute_offset,
                    hc_scale_bytes: layer8_hc_scale.bytes,
                    hc_base_offset: layer8_hc_base.absolute_offset,
                    hc_base_bytes: layer8_hc_base.bytes,
                    norm_offset: layer8_norm.absolute_offset,
                    norm_bytes: layer8_norm.bytes,
                    q_a_offset: layer8_q_a.absolute_offset,
                    q_a_bytes: layer8_q_a.bytes,
                },
                q_a_norm_offset: layer8_q_a_norm.absolute_offset,
                q_a_norm_bytes: layer8_q_a_norm.bytes,
                kv_offset: layer8_kv.absolute_offset,
                kv_bytes: layer8_kv.bytes,
                kv_norm_offset: layer8_kv_norm.absolute_offset,
                kv_norm_bytes: layer8_kv_norm.bytes,
            },
            layer8_q_b_offset: layer8_q_b.absolute_offset,
            layer8_q_b_bytes: layer8_q_b.bytes,
            layer8_compressor: RawPrefillCompressorWeights {
                attn_ape_offset: layer8_attn_ape.absolute_offset,
                attn_ape_bytes: layer8_attn_ape.bytes,
                attn_kv_offset: layer8_attn_kv.absolute_offset,
                attn_kv_bytes: layer8_attn_kv.bytes,
                attn_gate_offset: layer8_attn_gate.absolute_offset,
                attn_gate_bytes: layer8_attn_gate.bytes,
                attn_norm_offset: layer8_attn_compressor_norm.absolute_offset,
                attn_norm_bytes: layer8_attn_compressor_norm.bytes,
                indexer_ape_offset: layer8_indexer_ape.absolute_offset,
                indexer_ape_bytes: layer8_indexer_ape.bytes,
                indexer_kv_offset: layer8_indexer_kv.absolute_offset,
                indexer_kv_bytes: layer8_indexer_kv.bytes,
                indexer_gate_offset: layer8_indexer_gate.absolute_offset,
                indexer_gate_bytes: layer8_indexer_gate.bytes,
                indexer_norm_offset: layer8_indexer_compressor_norm.absolute_offset,
                indexer_norm_bytes: layer8_indexer_compressor_norm.bytes,
            },
            layer8_attn_sinks_offset: layer8_sinks.absolute_offset,
            layer8_attn_sinks_bytes: layer8_sinks.bytes,
            layer8_attn_output_a_offset: layer8_output_a.absolute_offset,
            layer8_attn_output_a_bytes: layer8_output_a.bytes,
            layer8_attn_output_b_offset: layer8_output_b.absolute_offset,
            layer8_attn_output_b_bytes: layer8_output_b.bytes,
            layer8_ffn: RawPrefillFfnWeights {
                hc_fn_offset: layer8_ffn_hc_fn.absolute_offset,
                hc_fn_bytes: layer8_ffn_hc_fn.bytes,
                hc_scale_offset: layer8_ffn_hc_scale.absolute_offset,
                hc_scale_bytes: layer8_ffn_hc_scale.bytes,
                hc_base_offset: layer8_ffn_hc_base.absolute_offset,
                hc_base_bytes: layer8_ffn_hc_base.bytes,
                norm_offset: layer8_ffn_norm.absolute_offset,
                norm_bytes: layer8_ffn_norm.bytes,
                router_gate_offset: layer8_router_gate.absolute_offset,
                router_gate_bytes: layer8_router_gate.bytes,
                router_hash_offset: layer8_router_bias.absolute_offset,
                router_hash_bytes: layer8_router_bias.bytes,
                routed_gate_offset: layer8_routed_gate.absolute_offset,
                routed_gate_bytes: layer8_routed_gate.bytes,
                routed_up_offset: layer8_routed_up.absolute_offset,
                routed_up_bytes: layer8_routed_up.bytes,
                routed_down_offset: layer8_routed_down.absolute_offset,
                routed_down_bytes: layer8_routed_down.bytes,
                shared_gate_offset: layer8_shared_gate.absolute_offset,
                shared_gate_bytes: layer8_shared_gate.bytes,
                shared_up_offset: layer8_shared_up.absolute_offset,
                shared_up_bytes: layer8_shared_up.bytes,
                shared_down_offset: layer8_shared_down.absolute_offset,
                shared_down_bytes: layer8_shared_down.bytes,
            },
        };
        let mut actual = vec![0.0_f32; expected.len()];
        let mut actual_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_router_selected = vec![0_i32; 32 * 6];
        let mut actual_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_shared_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer3_hc_attn_pre = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer3_attn_norm = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer3_q_lora = vec![0.0_f32; 2048 * 1024];
        let mut actual_layer3_q_lora_norm = vec![0.0_f32; 2048 * 1024];
        let mut actual_layer3_kv_raw = vec![0.0_f32; 2048 * 512];
        let mut actual_layer3_kv_norm = vec![0.0_f32; 2048 * 512];
        let mut actual_layer3_q_raw_final_tile = vec![0.0_f32; 32 * 32768];
        let mut actual_layer3_q_cur_final_tile = vec![0.0_f32; 32 * 32768];
        let mut actual_layer3_kv_rope = vec![0.0_f32; 2048 * 512];
        let mut actual_layer3_kv_cur = vec![0.0_f32; 2048 * 512];
        let mut actual_layer3_attn_compressed = vec![0.0_f32; 16 * 512];
        let mut actual_layer3_attn_state_kv = vec![0.0_f32; 128 * 512];
        let mut actual_layer3_attn_state_score = vec![0_i32; 128 * 512];
        let mut actual_layer3_kqv_out_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer3_kqv_back_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer3_attn_low_row0 = vec![0.0_f32; 8_192];
        let mut actual_layer3_attention = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer3_after_attention_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer3_after_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer3_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer3_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer3_router_selected = vec![0_i32; 32 * 6];
        let mut actual_layer3_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_layer3_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer3_shared_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer4_qkv: [Vec<f32>; 10] = [
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
        ];
        let layer4_qkv_pointers = actual_layer4_qkv
            .each_mut()
            .map(|tensor| tensor.as_mut_ptr());
        let mut actual_layer4_compressor: [Vec<f32>; 6] = [
            vec![0.0_f32; 512 * 512],
            vec![0.0_f32; 8 * 1024],
            vec![0.0_f32; 8 * 1024],
            vec![0.0_f32; 512 * 128],
            vec![0.0_f32; 8 * 256],
            vec![0.0_f32; 8 * 256],
        ];
        let mut actual_layer4_kqv_out_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer4_kqv_back_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer4_attn_low_row0 = vec![0.0_f32; 8_192];
        let mut actual_layer4_attention = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer4_after_attention_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer4_after_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer4_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer4_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer4_router_selected = vec![0_i32; 32 * 6];
        let mut actual_layer4_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_layer4_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer4_shared_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer5_qkv: [Vec<f32>; 10] = [
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
        ];
        let layer5_qkv_pointers = actual_layer5_qkv
            .each_mut()
            .map(|tensor| tensor.as_mut_ptr());
        let mut actual_layer5_attn_compressed = vec![0.0_f32; 16 * 512];
        let mut actual_layer5_attn_state_kv = vec![0.0_f32; 128 * 512];
        let mut actual_layer5_attn_state_score = vec![0_i32; 128 * 512];
        let mut actual_layer5_kqv_out_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer5_kqv_back_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer5_attn_low_row0 = vec![0.0_f32; 8_192];
        let mut actual_layer5_attention = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer5_after_attention_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer5_after_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer5_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer5_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer5_router_selected = vec![0_i32; 32 * 6];
        let mut actual_layer5_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_layer5_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer5_shared_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer6_qkv: [Vec<f32>; 10] = [
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
        ];
        let layer6_qkv_pointers = actual_layer6_qkv
            .each_mut()
            .map(|tensor| tensor.as_mut_ptr());
        let mut actual_layer6_compressor: [Vec<f32>; 6] = [
            vec![0.0_f32; 512 * 512],
            vec![0.0_f32; 8 * 1024],
            vec![0.0_f32; 8 * 1024],
            vec![0.0_f32; 512 * 128],
            vec![0.0_f32; 8 * 256],
            vec![0.0_f32; 8 * 256],
        ];
        let mut actual_layer6_kqv_out_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer6_kqv_back_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer6_attn_low_row0 = vec![0.0_f32; 8_192];
        let mut actual_layer6_attention = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer6_after_attention_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer6_after_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer6_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer6_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer6_router_selected = vec![0_i32; 32 * 6];
        let mut actual_layer6_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_layer6_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer6_shared_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer7_qkv: [Vec<f32>; 10] = [
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
        ];
        let layer7_qkv_pointers = actual_layer7_qkv
            .each_mut()
            .map(|tensor| tensor.as_mut_ptr());
        let mut actual_layer7_attn_compressed = vec![0.0_f32; 16 * 512];
        let mut actual_layer7_attn_state_kv = vec![0.0_f32; 128 * 512];
        let mut actual_layer7_attn_state_score = vec![0_i32; 128 * 512];
        let mut actual_layer7_kqv_out_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer7_kqv_back_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer7_attn_low_row0 = vec![0.0_f32; 8_192];
        let mut actual_layer7_attention = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer7_after_attention_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer7_after_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer7_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer7_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer7_router_selected = vec![0_i32; 32 * 6];
        let mut actual_layer7_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_layer7_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer7_shared_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer8_qkv: [Vec<f32>; 10] = [
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 4096],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 1024],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 32768],
            vec![0.0_f32; 32 * 512],
            vec![0.0_f32; 32 * 512],
        ];
        let layer8_qkv_pointers = actual_layer8_qkv
            .each_mut()
            .map(|tensor| tensor.as_mut_ptr());
        let mut actual_layer8_compressor: [Vec<f32>; 6] = [
            vec![0.0_f32; 512 * 512],
            vec![0.0_f32; 8 * 1024],
            vec![0.0_f32; 8 * 1024],
            vec![0.0_f32; 512 * 128],
            vec![0.0_f32; 8 * 256],
            vec![0.0_f32; 8 * 256],
        ];
        let mut actual_layer8_kqv_out_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer8_kqv_back_row0 = vec![0.0_f32; 32_768];
        let mut actual_layer8_attn_low_row0 = vec![0.0_f32; 8_192];
        let mut actual_layer8_attention = vec![0.0_f32; 2048 * 4096];
        let mut actual_layer8_after_attention_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer8_after_ffn_hc = vec![0.0_f32; 2048 * 4 * 4096];
        let mut actual_layer8_ffn_cur_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer8_ffn_norm_final_tile = vec![0.0_f32; 32 * 4096];
        let mut actual_layer8_router_selected = vec![0_i32; 32 * 6];
        let mut actual_layer8_router_weights = vec![0.0_f32; 32 * 6];
        let mut actual_layer8_routed_out = vec![0.0_f32; 32 * 4096];
        let mut actual_layer8_shared_out = vec![0.0_f32; 32 * 4096];
        let mut raw = RawPrefillLayer2AttentionResult::default();
        let mut error = [0 as c_char; ERROR_BYTES];
        let succeeded = unsafe {
            rust_star_metal_run_prefill_layer2_attention(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &weights,
                actual.as_mut_ptr(),
                actual_hc.as_mut_ptr(),
                actual_ffn_hc.as_mut_ptr(),
                actual_ffn_cur_final_tile.as_mut_ptr(),
                actual_ffn_norm_final_tile.as_mut_ptr(),
                actual_router_selected.as_mut_ptr(),
                actual_router_weights.as_mut_ptr(),
                actual_routed_out.as_mut_ptr(),
                actual_shared_out.as_mut_ptr(),
                actual_layer3_hc_attn_pre.as_mut_ptr(),
                actual_layer3_attn_norm.as_mut_ptr(),
                actual_layer3_q_lora.as_mut_ptr(),
                actual_layer3_q_lora_norm.as_mut_ptr(),
                actual_layer3_kv_raw.as_mut_ptr(),
                actual_layer3_kv_norm.as_mut_ptr(),
                actual_layer3_q_raw_final_tile.as_mut_ptr(),
                actual_layer3_q_cur_final_tile.as_mut_ptr(),
                actual_layer3_kv_rope.as_mut_ptr(),
                actual_layer3_kv_cur.as_mut_ptr(),
                actual_layer3_attn_compressed.as_mut_ptr(),
                actual_layer3_attn_state_kv.as_mut_ptr(),
                actual_layer3_attn_state_score.as_mut_ptr(),
                actual_layer3_kqv_out_row0.as_mut_ptr(),
                actual_layer3_kqv_back_row0.as_mut_ptr(),
                actual_layer3_attn_low_row0.as_mut_ptr(),
                actual_layer3_attention.as_mut_ptr(),
                actual_layer3_after_attention_hc.as_mut_ptr(),
                actual_layer3_after_ffn_hc.as_mut_ptr(),
                actual_layer3_ffn_cur_final_tile.as_mut_ptr(),
                actual_layer3_ffn_norm_final_tile.as_mut_ptr(),
                actual_layer3_router_selected.as_mut_ptr(),
                actual_layer3_router_weights.as_mut_ptr(),
                actual_layer3_routed_out.as_mut_ptr(),
                actual_layer3_shared_out.as_mut_ptr(),
                layer4_qkv_pointers[0],
                layer4_qkv_pointers[1],
                layer4_qkv_pointers[2],
                layer4_qkv_pointers[3],
                layer4_qkv_pointers[4],
                layer4_qkv_pointers[5],
                layer4_qkv_pointers[6],
                layer4_qkv_pointers[7],
                layer4_qkv_pointers[8],
                layer4_qkv_pointers[9],
                actual_layer4_compressor[0].as_mut_ptr(),
                actual_layer4_compressor[1].as_mut_ptr(),
                actual_layer4_compressor[2].as_mut_ptr().cast::<i32>(),
                actual_layer4_compressor[3].as_mut_ptr(),
                actual_layer4_compressor[4].as_mut_ptr(),
                actual_layer4_compressor[5].as_mut_ptr().cast::<i32>(),
                actual_layer4_kqv_out_row0.as_mut_ptr(),
                actual_layer4_kqv_back_row0.as_mut_ptr(),
                actual_layer4_attn_low_row0.as_mut_ptr(),
                actual_layer4_attention.as_mut_ptr(),
                actual_layer4_after_attention_hc.as_mut_ptr(),
                actual_layer4_after_ffn_hc.as_mut_ptr(),
                actual_layer4_ffn_cur_final_tile.as_mut_ptr(),
                actual_layer4_ffn_norm_final_tile.as_mut_ptr(),
                actual_layer4_router_selected.as_mut_ptr(),
                actual_layer4_router_weights.as_mut_ptr(),
                actual_layer4_routed_out.as_mut_ptr(),
                actual_layer4_shared_out.as_mut_ptr(),
                layer5_qkv_pointers[0],
                layer5_qkv_pointers[1],
                layer5_qkv_pointers[2],
                layer5_qkv_pointers[3],
                layer5_qkv_pointers[4],
                layer5_qkv_pointers[5],
                layer5_qkv_pointers[6],
                layer5_qkv_pointers[7],
                layer5_qkv_pointers[8],
                layer5_qkv_pointers[9],
                actual_layer5_attn_compressed.as_mut_ptr(),
                actual_layer5_attn_state_kv.as_mut_ptr(),
                actual_layer5_attn_state_score.as_mut_ptr(),
                actual_layer5_kqv_out_row0.as_mut_ptr(),
                actual_layer5_kqv_back_row0.as_mut_ptr(),
                actual_layer5_attn_low_row0.as_mut_ptr(),
                actual_layer5_attention.as_mut_ptr(),
                actual_layer5_after_attention_hc.as_mut_ptr(),
                actual_layer5_after_ffn_hc.as_mut_ptr(),
                actual_layer5_ffn_cur_final_tile.as_mut_ptr(),
                actual_layer5_ffn_norm_final_tile.as_mut_ptr(),
                actual_layer5_router_selected.as_mut_ptr(),
                actual_layer5_router_weights.as_mut_ptr(),
                actual_layer5_routed_out.as_mut_ptr(),
                actual_layer5_shared_out.as_mut_ptr(),
                layer6_qkv_pointers[0],
                layer6_qkv_pointers[1],
                layer6_qkv_pointers[2],
                layer6_qkv_pointers[3],
                layer6_qkv_pointers[4],
                layer6_qkv_pointers[5],
                layer6_qkv_pointers[6],
                layer6_qkv_pointers[7],
                layer6_qkv_pointers[8],
                layer6_qkv_pointers[9],
                actual_layer6_compressor[0].as_mut_ptr(),
                actual_layer6_compressor[1].as_mut_ptr(),
                actual_layer6_compressor[2].as_mut_ptr().cast::<i32>(),
                actual_layer6_compressor[3].as_mut_ptr(),
                actual_layer6_compressor[4].as_mut_ptr(),
                actual_layer6_compressor[5].as_mut_ptr().cast::<i32>(),
                actual_layer6_kqv_out_row0.as_mut_ptr(),
                actual_layer6_kqv_back_row0.as_mut_ptr(),
                actual_layer6_attn_low_row0.as_mut_ptr(),
                actual_layer6_attention.as_mut_ptr(),
                actual_layer6_after_attention_hc.as_mut_ptr(),
                actual_layer6_after_ffn_hc.as_mut_ptr(),
                actual_layer6_ffn_cur_final_tile.as_mut_ptr(),
                actual_layer6_ffn_norm_final_tile.as_mut_ptr(),
                actual_layer6_router_selected.as_mut_ptr(),
                actual_layer6_router_weights.as_mut_ptr(),
                actual_layer6_routed_out.as_mut_ptr(),
                actual_layer6_shared_out.as_mut_ptr(),
                layer7_qkv_pointers[0],
                layer7_qkv_pointers[1],
                layer7_qkv_pointers[2],
                layer7_qkv_pointers[3],
                layer7_qkv_pointers[4],
                layer7_qkv_pointers[5],
                layer7_qkv_pointers[6],
                layer7_qkv_pointers[7],
                layer7_qkv_pointers[8],
                layer7_qkv_pointers[9],
                actual_layer7_attn_compressed.as_mut_ptr(),
                actual_layer7_attn_state_kv.as_mut_ptr(),
                actual_layer7_attn_state_score.as_mut_ptr(),
                actual_layer7_kqv_out_row0.as_mut_ptr(),
                actual_layer7_kqv_back_row0.as_mut_ptr(),
                actual_layer7_attn_low_row0.as_mut_ptr(),
                actual_layer7_attention.as_mut_ptr(),
                actual_layer7_after_attention_hc.as_mut_ptr(),
                actual_layer7_after_ffn_hc.as_mut_ptr(),
                actual_layer7_ffn_cur_final_tile.as_mut_ptr(),
                actual_layer7_ffn_norm_final_tile.as_mut_ptr(),
                actual_layer7_router_selected.as_mut_ptr(),
                actual_layer7_router_weights.as_mut_ptr(),
                actual_layer7_routed_out.as_mut_ptr(),
                actual_layer7_shared_out.as_mut_ptr(),
                layer8_qkv_pointers[0],
                layer8_qkv_pointers[1],
                layer8_qkv_pointers[2],
                layer8_qkv_pointers[3],
                layer8_qkv_pointers[4],
                layer8_qkv_pointers[5],
                layer8_qkv_pointers[6],
                layer8_qkv_pointers[7],
                layer8_qkv_pointers[8],
                layer8_qkv_pointers[9],
                actual_layer8_compressor[0].as_mut_ptr(),
                actual_layer8_compressor[1].as_mut_ptr(),
                actual_layer8_compressor[2].as_mut_ptr().cast::<i32>(),
                actual_layer8_compressor[3].as_mut_ptr(),
                actual_layer8_compressor[4].as_mut_ptr(),
                actual_layer8_compressor[5].as_mut_ptr().cast::<i32>(),
                actual_layer8_kqv_out_row0.as_mut_ptr(),
                actual_layer8_kqv_back_row0.as_mut_ptr(),
                actual_layer8_attn_low_row0.as_mut_ptr(),
                actual_layer8_attention.as_mut_ptr(),
                actual_layer8_after_attention_hc.as_mut_ptr(),
                actual_layer8_after_ffn_hc.as_mut_ptr(),
                actual_layer8_ffn_cur_final_tile.as_mut_ptr(),
                actual_layer8_ffn_norm_final_tile.as_mut_ptr(),
                actual_layer8_router_selected.as_mut_ptr(),
                actual_layer8_router_weights.as_mut_ptr(),
                actual_layer8_routed_out.as_mut_ptr(),
                actual_layer8_shared_out.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal prefill layer-2 attention probe failed: {}",
                error_text(&error)
            )));
        }
        for (label, actual, expected) in [
            (
                "HC collapse",
                &actual_ffn_cur_final_tile,
                &expected_ffn_cur_final_tile,
            ),
            (
                "learned norm",
                &actual_ffn_norm_final_tile,
                &expected_ffn_norm_final_tile,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-2 FFN {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        let actual_layer3_kv_state: [&[f32]; 7] = [
            &actual_layer3_q_lora_norm
                [actual_layer3_q_lora_norm.len() - expected_layer3_kv_state[0].len()..],
            &actual_layer3_kv_raw[actual_layer3_kv_raw.len() - expected_layer3_kv_state[1].len()..],
            &actual_layer3_kv_norm
                [actual_layer3_kv_norm.len() - expected_layer3_kv_state[2].len()..],
            &actual_layer3_q_raw_final_tile,
            &actual_layer3_q_cur_final_tile,
            &actual_layer3_kv_rope
                [actual_layer3_kv_rope.len() - expected_layer3_kv_state[5].len()..],
            &actual_layer3_kv_cur[actual_layer3_kv_cur.len() - expected_layer3_kv_state[6].len()..],
        ];
        let layer3_kv_labels = [
            "Q-Lora norm",
            "KV raw",
            "KV norm",
            "Q raw",
            "Q current",
            "KV rope",
            "KV current",
        ];
        for ((label, actual), expected) in layer3_kv_labels
            .into_iter()
            .zip(actual_layer3_kv_state)
            .zip(&expected_layer3_kv_state)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-3 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "compressed KV",
                &actual_layer3_attn_compressed,
                &expected_layer3_compressor.0,
            ),
            (
                "recurrent KV state",
                &actual_layer3_attn_state_kv,
                &expected_layer3_compressor.1,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-3 ratio-128 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer3_attn_state_score
            .iter()
            .zip(&expected_layer3_compressor.2)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-3 ratio-128 recurrent score state C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    *actual as u32,
                    *expected as u32,
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "KQV output row 0",
                &actual_layer3_kqv_out_row0,
                &expected_layer3_attention_diagnostics[0],
            ),
            (
                "KQV back row 0",
                &actual_layer3_kqv_back_row0,
                &expected_layer3_attention_diagnostics[1],
            ),
            (
                "attention low row 0",
                &actual_layer3_attn_low_row0,
                &expected_layer3_attention_diagnostics[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-3 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer3_attention
            .iter()
            .zip(&expected_layer3_attention.0)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-3 dense mixed-attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits(),
                )));
            }
        }
        let actual_layer3_hc_final_tile = &actual_layer3_after_attention_hc
            [actual_layer3_after_attention_hc.len() - expected_layer3_attention.1.len()..];
        for (index, (actual, expected)) in actual_layer3_hc_final_tile
            .iter()
            .zip(&expected_layer3_attention.1)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-3 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits(),
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "HC ingress",
                &actual_layer3_ffn_cur_final_tile,
                &expected_layer3_complete.ffn_cur,
            ),
            (
                "learned norm",
                &actual_layer3_ffn_norm_final_tile,
                &expected_layer3_complete.ffn_norm,
            ),
            (
                "router weights",
                &actual_layer3_router_weights,
                &expected_layer3_complete.router_weights,
            ),
            (
                "routed output",
                &actual_layer3_routed_out,
                &expected_layer3_complete.routed_out,
            ),
            (
                "shared output",
                &actual_layer3_shared_out,
                &expected_layer3_complete.shared_out,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-3 FFN {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer3_router_selected
            .iter()
            .zip(&expected_layer3_complete.router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-3 biased top-k C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        let actual_layer3_ffn_hc_final_tile = &actual_layer3_after_ffn_hc
            [actual_layer3_after_ffn_hc.len() - expected_layer3_complete.hc_post.len()..];
        for (index, (actual, expected)) in actual_layer3_ffn_hc_final_tile
            .iter()
            .zip(&expected_layer3_complete.hc_post)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-3 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits(),
                )));
            }
        }
        for ((label, actual), expected) in [
            "HC attention ingress",
            "attention norm",
            "Q-Lora",
            "Q-Lora norm",
            "KV raw",
            "KV norm",
            "Q raw",
            "Q current",
            "KV rope",
            "KV current",
        ]
        .into_iter()
        .zip(&actual_layer4_qkv)
        .zip(&expected_layer4_qkv)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-4 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        for ((label, actual), expected) in [
            "attention compressed KV",
            "attention recurrent KV state",
            "attention recurrent score-state bits",
            "indexer compressed KV",
            "indexer recurrent KV state",
            "indexer recurrent score-state bits",
        ]
        .into_iter()
        .zip(&actual_layer4_compressor)
        .zip(&expected_layer4_compressor)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-4 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "KQV output row 0",
                &actual_layer4_kqv_out_row0,
                &expected_layer4_attention_diagnostics[0],
            ),
            (
                "KQV back row 0",
                &actual_layer4_kqv_back_row0,
                &expected_layer4_attention_diagnostics[1],
            ),
            (
                "attention low row 0",
                &actual_layer4_attn_low_row0,
                &expected_layer4_attention_diagnostics[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-4 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer4_attention
            .iter()
            .zip(&expected_layer4_attention.0)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-4 dense mixed-attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        let layer4_hc_final_tile = &actual_layer4_after_attention_hc
            [actual_layer4_after_attention_hc.len() - expected_layer4_attention.1.len()..];
        for (index, (actual, expected)) in layer4_hc_final_tile
            .iter()
            .zip(&expected_layer4_attention.1)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-4 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "HC collapse",
                &actual_layer4_ffn_cur_final_tile,
                &expected_layer4_complete.ffn_cur,
            ),
            (
                "learned norm",
                &actual_layer4_ffn_norm_final_tile,
                &expected_layer4_complete.ffn_norm,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-4 FFN {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer4_router_selected
            .iter()
            .zip(&expected_layer4_complete.router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-4 router selection C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "router weights",
                &actual_layer4_router_weights,
                &expected_layer4_complete.router_weights,
            ),
            (
                "routed output",
                &actual_layer4_routed_out,
                &expected_layer4_complete.routed_out,
            ),
            (
                "shared output",
                &actual_layer4_shared_out,
                &expected_layer4_complete.shared_out,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-4 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        let layer4_ffn_hc_final_tile = &actual_layer4_after_ffn_hc
            [actual_layer4_after_ffn_hc.len() - expected_layer4_complete.hc_post.len()..];
        for (index, (actual, expected)) in layer4_ffn_hc_final_tile
            .iter()
            .zip(&expected_layer4_complete.hc_post)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-4 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer4_after_ffn_hc) != PREFILL_LAYER4_HC_FFN_POST_FULL_CHECKSUM {
            return Err(Error::invalid(
                "prefill layer-4 FFN HC post full-2K checksum mismatch",
            ));
        }
        for ((label, actual), expected) in [
            "HC attention ingress",
            "attention norm",
            "Q-Lora",
            "Q-Lora norm",
            "KV raw",
            "KV norm",
            "Q raw",
            "Q current",
            "KV rope",
            "KV current",
        ]
        .into_iter()
        .zip(&actual_layer5_qkv)
        .zip(&expected_layer5_qkv)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-5 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "compressed KV",
                &actual_layer5_attn_compressed,
                &expected_layer5_compressor.0,
            ),
            (
                "recurrent KV state",
                &actual_layer5_attn_state_kv,
                &expected_layer5_compressor.1,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-5 ratio-128 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer5_attn_state_score
            .iter()
            .zip(&expected_layer5_compressor.2)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-5 ratio-128 recurrent score state C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    *actual as u32, *expected as u32,
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "KQV output row 0",
                &actual_layer5_kqv_out_row0,
                &expected_layer5_attention_diagnostics[0],
            ),
            (
                "KQV back row 0",
                &actual_layer5_kqv_back_row0,
                &expected_layer5_attention_diagnostics[1],
            ),
            (
                "attention low row 0",
                &actual_layer5_attn_low_row0,
                &expected_layer5_attention_diagnostics[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-5 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer5_attention
            .iter()
            .zip(&expected_layer5_attention.0)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-5 dense mixed-attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        let layer5_hc_final_tile = &actual_layer5_after_attention_hc
            [actual_layer5_after_attention_hc.len() - expected_layer5_attention.1.len()..];
        for (index, (actual, expected)) in layer5_hc_final_tile
            .iter()
            .zip(&expected_layer5_attention.1)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-5 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer5_after_attention_hc)
            != PREFILL_LAYER5_HC_ATTN_POST_FULL_CHECKSUM
        {
            return Err(Error::invalid(
                "prefill layer-5 attention HC post full-2K checksum mismatch",
            ));
        }
        for (label, actual, expected) in [
            (
                "HC collapse",
                &actual_layer5_ffn_cur_final_tile,
                &expected_layer5_complete.ffn_cur,
            ),
            (
                "learned norm",
                &actual_layer5_ffn_norm_final_tile,
                &expected_layer5_complete.ffn_norm,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-5 FFN {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer5_router_selected
            .iter()
            .zip(&expected_layer5_complete.router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-5 router selection C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "router weights",
                &actual_layer5_router_weights,
                &expected_layer5_complete.router_weights,
            ),
            (
                "routed output",
                &actual_layer5_routed_out,
                &expected_layer5_complete.routed_out,
            ),
            (
                "shared output",
                &actual_layer5_shared_out,
                &expected_layer5_complete.shared_out,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-5 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        let layer5_ffn_hc_final_tile = &actual_layer5_after_ffn_hc
            [actual_layer5_after_ffn_hc.len() - expected_layer5_complete.hc_post.len()..];
        for (index, (actual, expected)) in layer5_ffn_hc_final_tile
            .iter()
            .zip(&expected_layer5_complete.hc_post)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-5 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer5_after_ffn_hc) != PREFILL_LAYER5_HC_FFN_POST_FULL_CHECKSUM {
            return Err(Error::invalid(
                "prefill layer-5 FFN HC post full-2K checksum mismatch",
            ));
        }
        for ((label, actual), expected) in [
            "HC attention ingress",
            "attention norm",
            "Q-Lora",
            "Q-Lora norm",
            "KV raw",
            "KV norm",
            "Q raw",
            "Q current",
            "KV rope",
            "KV current",
        ]
        .into_iter()
        .zip(&actual_layer6_qkv)
        .zip(&expected_layer6_qkv)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-6 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for ((label, actual), expected) in [
            "attention compressed KV",
            "attention KV state",
            "attention score-state bits",
            "indexer compressed KV",
            "indexer KV state",
            "indexer score-state bits",
        ]
        .into_iter()
        .zip(&actual_layer6_compressor)
        .zip(&expected_layer6_compressor)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-6 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "KQV output row 0",
                &actual_layer6_kqv_out_row0,
                &expected_layer6_attention_diagnostics[0],
            ),
            (
                "KQV back row 0",
                &actual_layer6_kqv_back_row0,
                &expected_layer6_attention_diagnostics[1],
            ),
            (
                "attention low row 0",
                &actual_layer6_attn_low_row0,
                &expected_layer6_attention_diagnostics[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-6 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer6_attention
            .iter()
            .zip(&expected_layer6_attention.0)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-6 dense mixed-attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        let layer6_hc_final_tile = &actual_layer6_after_attention_hc
            [actual_layer6_after_attention_hc.len() - expected_layer6_attention.1.len()..];
        for (index, (actual, expected)) in layer6_hc_final_tile
            .iter()
            .zip(&expected_layer6_attention.1)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-6 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer6_after_attention_hc)
            != PREFILL_LAYER6_HC_ATTN_POST_FULL_CHECKSUM
        {
            return Err(Error::invalid(
                "prefill layer-6 attention HC post full-2K checksum mismatch",
            ));
        }
        for (index, (actual, expected)) in actual_layer6_router_selected
            .iter()
            .zip(&expected_layer6_complete.router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-6 router selection C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "HC collapse",
                &actual_layer6_ffn_cur_final_tile,
                &expected_layer6_complete.ffn_cur,
            ),
            (
                "learned norm",
                &actual_layer6_ffn_norm_final_tile,
                &expected_layer6_complete.ffn_norm,
            ),
            (
                "router weights",
                &actual_layer6_router_weights,
                &expected_layer6_complete.router_weights,
            ),
            (
                "routed output",
                &actual_layer6_routed_out,
                &expected_layer6_complete.routed_out,
            ),
            (
                "shared output",
                &actual_layer6_shared_out,
                &expected_layer6_complete.shared_out,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-6 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        let layer6_ffn_hc_final_tile = &actual_layer6_after_ffn_hc
            [actual_layer6_after_ffn_hc.len() - expected_layer6_complete.hc_post.len()..];
        for (index, (actual, expected)) in layer6_ffn_hc_final_tile
            .iter()
            .zip(&expected_layer6_complete.hc_post)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-6 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer6_after_ffn_hc) != PREFILL_LAYER6_HC_FFN_POST_FULL_CHECKSUM {
            return Err(Error::invalid(
                "prefill layer-6 FFN HC post full-2K checksum mismatch",
            ));
        }
        for ((label, actual), expected) in [
            "HC attention ingress",
            "attention norm",
            "Q-Lora",
            "Q-Lora norm",
            "KV raw",
            "KV norm",
            "Q raw",
            "Q current",
            "KV rope",
            "KV current",
        ]
        .into_iter()
        .zip(&actual_layer7_qkv)
        .zip(&expected_layer7_qkv)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-7 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "compressed KV",
                &actual_layer7_attn_compressed,
                &expected_layer7_compressor.0,
            ),
            (
                "recurrent KV state",
                &actual_layer7_attn_state_kv,
                &expected_layer7_compressor.1,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-7 ratio-128 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer7_attn_state_score
            .iter()
            .zip(&expected_layer7_compressor.2)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-7 ratio-128 recurrent score state C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    *actual as u32, *expected as u32,
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "KQV output row 0",
                &actual_layer7_kqv_out_row0,
                &expected_layer7_attention_diagnostics[0],
            ),
            (
                "KQV back row 0",
                &actual_layer7_kqv_back_row0,
                &expected_layer7_attention_diagnostics[1],
            ),
            (
                "attention low row 0",
                &actual_layer7_attn_low_row0,
                &expected_layer7_attention_diagnostics[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-7 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer7_attention
            .iter()
            .zip(&expected_layer7_attention.0)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-7 dense mixed-attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        let layer7_hc_final_tile = &actual_layer7_after_attention_hc
            [actual_layer7_after_attention_hc.len() - expected_layer7_attention.1.len()..];
        for (index, (actual, expected)) in layer7_hc_final_tile
            .iter()
            .zip(&expected_layer7_attention.1)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-7 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer7_after_attention_hc)
            != PREFILL_LAYER7_HC_ATTN_POST_FULL_CHECKSUM
        {
            return Err(Error::invalid(
                "prefill layer-7 attention HC post full-2K checksum mismatch",
            ));
        }
        for (label, actual, expected) in [
            (
                "HC collapse",
                &actual_layer7_ffn_cur_final_tile,
                &expected_layer7_complete.ffn_cur,
            ),
            (
                "learned norm",
                &actual_layer7_ffn_norm_final_tile,
                &expected_layer7_complete.ffn_norm,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-7 FFN {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer7_router_selected
            .iter()
            .zip(&expected_layer7_complete.router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-7 router selection C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "router weights",
                &actual_layer7_router_weights,
                &expected_layer7_complete.router_weights,
            ),
            (
                "routed output",
                &actual_layer7_routed_out,
                &expected_layer7_complete.routed_out,
            ),
            (
                "shared output",
                &actual_layer7_shared_out,
                &expected_layer7_complete.shared_out,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-7 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        let layer7_ffn_hc_final_tile = &actual_layer7_after_ffn_hc
            [actual_layer7_after_ffn_hc.len() - expected_layer7_complete.hc_post.len()..];
        for (index, (actual, expected)) in layer7_ffn_hc_final_tile
            .iter()
            .zip(&expected_layer7_complete.hc_post)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-7 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer7_after_ffn_hc) != PREFILL_LAYER7_HC_FFN_POST_FULL_CHECKSUM {
            return Err(Error::invalid(
                "prefill layer-7 FFN HC post full-2K checksum mismatch",
            ));
        }
        for ((label, actual), expected) in [
            "HC attention ingress",
            "attention norm",
            "Q-Lora",
            "Q-Lora norm",
            "KV raw",
            "KV norm",
            "Q raw",
            "Q current",
            "KV rope",
            "KV current",
        ]
        .into_iter()
        .zip(&actual_layer8_qkv)
        .zip(&expected_layer8_qkv)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-8 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for ((label, actual), expected) in [
            "attention compressed KV",
            "attention KV state",
            "attention score-state bits",
            "indexer compressed KV",
            "indexer KV state",
            "indexer score-state bits",
        ]
        .into_iter()
        .zip(&actual_layer8_compressor)
        .zip(&expected_layer8_compressor)
        {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-8 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (label, actual, expected) in [
            (
                "KQV output row 0",
                &actual_layer8_kqv_out_row0,
                &expected_layer8_attention_diagnostics[0],
            ),
            (
                "KQV back row 0",
                &actual_layer8_kqv_back_row0,
                &expected_layer8_attention_diagnostics[1],
            ),
            (
                "attention low row 0",
                &actual_layer8_attn_low_row0,
                &expected_layer8_attention_diagnostics[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-8 {label} C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        for (index, (actual, expected)) in actual_layer8_attention
            .iter()
            .zip(&expected_layer8_attention.0)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-8 dense mixed-attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        let layer8_hc_final_tile = &actual_layer8_after_attention_hc
            [actual_layer8_after_attention_hc.len() - expected_layer8_attention.1.len()..];
        for (index, (actual, expected)) in layer8_hc_final_tile
            .iter()
            .zip(&expected_layer8_attention.1)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-8 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer8_after_attention_hc)
            != PREFILL_LAYER8_HC_ATTN_POST_FULL_CHECKSUM
        {
            return Err(Error::invalid(
                "prefill layer-8 attention HC post full-2K checksum mismatch",
            ));
        }
        for (index, (actual, expected)) in actual_layer8_router_selected
            .iter()
            .zip(&expected_layer8_complete.router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-8 router selection C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "HC collapse",
                &actual_layer8_ffn_cur_final_tile,
                &expected_layer8_complete.ffn_cur,
            ),
            (
                "learned norm",
                &actual_layer8_ffn_norm_final_tile,
                &expected_layer8_complete.ffn_norm,
            ),
            (
                "router weights",
                &actual_layer8_router_weights,
                &expected_layer8_complete.router_weights,
            ),
            (
                "routed output",
                &actual_layer8_routed_out,
                &expected_layer8_complete.routed_out,
            ),
            (
                "shared output",
                &actual_layer8_shared_out,
                &expected_layer8_complete.shared_out,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-8 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(), expected.to_bits(),
                    )));
                }
            }
        }
        let layer8_ffn_hc_final_tile = &actual_layer8_after_ffn_hc
            [actual_layer8_after_ffn_hc.len() - expected_layer8_complete.hc_post.len()..];
        for (index, (actual, expected)) in layer8_ffn_hc_final_tile
            .iter()
            .zip(&expected_layer8_complete.hc_post)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-8 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits(),
                )));
            }
        }
        if checksum_f32(&actual_layer8_after_ffn_hc) != PREFILL_LAYER8_HC_FFN_POST_FULL_CHECKSUM {
            return Err(Error::invalid(
                "prefill layer-8 FFN HC post full-2K checksum mismatch",
            ));
        }
        for (index, (actual, expected)) in actual_router_selected
            .iter()
            .zip(&expected_router_selected)
            .enumerate()
        {
            if actual != expected {
                return Err(Error::invalid(format!(
                    "prefill layer-2 router selection C0 mismatch at final-tile element {index}: actual={actual} expected={expected}",
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "router weights",
                &actual_router_weights,
                &expected_ffn_outputs[0],
            ),
            (
                "routed output",
                &actual_routed_out,
                &expected_ffn_outputs[1],
            ),
            (
                "shared output",
                &actual_shared_out,
                &expected_ffn_outputs[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-2 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        let actual_ffn_hc_final_tile =
            &actual_ffn_hc[actual_ffn_hc.len() - expected_ffn_hc_final_tile.len()..];
        for (index, (actual, expected)) in actual_ffn_hc_final_tile
            .iter()
            .zip(&expected_ffn_hc_final_tile)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-2 FFN HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits(),
                )));
            }
        }
        let actual_hc_final_tile = &actual_hc[actual_hc.len() - expected_hc_final_tile.len()..];
        for (index, (actual, expected)) in actual_hc_final_tile
            .iter()
            .zip(&expected_hc_final_tile)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-2 attention HC post C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits(),
                )));
            }
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "prefill layer-2 attention C0 mismatch at element {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits(),
                )));
            }
        }
        for (label, actual, expected) in [
            (
                "HC attention ingress",
                &actual_layer3_hc_attn_pre
                    [actual_layer3_hc_attn_pre.len() - expected_layer3_ingress[0].len()..],
                &expected_layer3_ingress[0],
            ),
            (
                "attention norm",
                &actual_layer3_attn_norm
                    [actual_layer3_attn_norm.len() - expected_layer3_ingress[1].len()..],
                &expected_layer3_ingress[1],
            ),
            (
                "Q-Lora",
                &actual_layer3_q_lora
                    [actual_layer3_q_lora.len() - expected_layer3_ingress[2].len()..],
                &expected_layer3_ingress[2],
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "prefill layer-3 {label} C0 mismatch at final-tile element {index}: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits(),
                    )));
                }
            }
        }
        if raw.rows != 2048
            || raw.raw_kv_rows != 2048
            || raw.compressed_kv_rows != 512
            || raw.layer3_compressed_kv_rows != 16
            || raw.dispatches != 383
            || raw.wrapped_model_ranges != 196
            || raw.pointer_matches != 196
        {
            return Err(Error::invalid(
                "Metal prefill layer-2 attention returned an unexpected schedule or mapping",
            ));
        }
        Ok(PrefillLayers012AttentionLoopProbeReport {
            compressor,
            attention_fixture_id: PREFILL_LAYER2_ATTENTION_FIXTURE_ID,
            attention_hc_fixture_id: PREFILL_LAYER2_COMPLETE_FIXTURE_ID,
            layer3_ingress_fixture_id: PREFILL_LAYER3_INGRESS_FIXTURE_ID,
            layer3_kv_state_fixture_id: PREFILL_LAYER3_KV_STATE_FIXTURE_ID,
            layer3_compressor_fixture_id: PREFILL_LAYER3_COMPRESSOR_FIXTURE_ID,
            layer3_attention_fixture_id: PREFILL_LAYER3_ATTENTION_FIXTURE_ID,
            layer3_complete_fixture_id: PREFILL_LAYER3_COMPLETE_FIXTURE_ID,
            layer4_qkv_fixture_id: PREFILL_LAYER4_QKV_FIXTURE_ID,
            layer4_compressor_fixture_id: PREFILL_LAYER4_COMPRESSOR_FIXTURE_ID,
            layer4_attention_fixture_id: PREFILL_LAYER4_ATTENTION_FIXTURE_ID,
            layer4_complete_fixture_id: PREFILL_LAYER4_COMPLETE_FIXTURE_ID,
            layer5_qkv_fixture_id: PREFILL_LAYER5_QKV_FIXTURE_ID,
            layer5_compressor_fixture_id: PREFILL_LAYER5_COMPRESSOR_FIXTURE_ID,
            layer5_attention_fixture_id: PREFILL_LAYER5_ATTENTION_FIXTURE_ID,
            layer5_complete_fixture_id: PREFILL_LAYER5_COMPLETE_FIXTURE_ID,
            layer6_qkv_fixture_id: PREFILL_LAYER6_QKV_FIXTURE_ID,
            layer6_compressor_fixture_id: PREFILL_LAYER6_COMPRESSOR_FIXTURE_ID,
            layer6_attention_fixture_id: PREFILL_LAYER6_ATTENTION_FIXTURE_ID,
            layer6_complete_fixture_id: PREFILL_LAYER6_COMPLETE_FIXTURE_ID,
            layer7_qkv_fixture_id: PREFILL_LAYER7_QKV_FIXTURE_ID,
            layer7_compressor_fixture_id: PREFILL_LAYER7_COMPRESSOR_FIXTURE_ID,
            layer7_attention_fixture_id: PREFILL_LAYER7_ATTENTION_FIXTURE_ID,
            layer7_complete_fixture_id: PREFILL_LAYER7_COMPLETE_FIXTURE_ID,
            layer8_qkv_fixture_id: PREFILL_LAYER8_QKV_FIXTURE_ID,
            layer8_compressor_fixture_id: PREFILL_LAYER8_COMPRESSOR_FIXTURE_ID,
            layer8_attention_fixture_id: PREFILL_LAYER8_ATTENTION_FIXTURE_ID,
            layer8_complete_fixture_id: PREFILL_LAYER8_COMPLETE_FIXTURE_ID,
            rows: raw.rows,
            raw_kv_rows: raw.raw_kv_rows,
            compressed_kv_rows: raw.compressed_kv_rows,
            layer3_compressed_kv_rows: raw.layer3_compressed_kv_rows,
            dispatches: raw.dispatches,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            output_checksum: checksum_f32(&actual),
            after_attention_hc_checksum: checksum_f32(&actual_hc),
            after_ffn_hc_checksum: checksum_f32(&actual_ffn_hc),
            layer3_hc_attn_pre_checksum: checksum_f32(&actual_layer3_hc_attn_pre),
            layer3_attn_norm_checksum: checksum_f32(&actual_layer3_attn_norm),
            layer3_q_lora_checksum: checksum_f32(&actual_layer3_q_lora),
            layer3_q_lora_norm_checksum: checksum_f32(&actual_layer3_q_lora_norm),
            layer3_kv_raw_checksum: checksum_f32(&actual_layer3_kv_raw),
            layer3_kv_norm_checksum: checksum_f32(&actual_layer3_kv_norm),
            layer3_q_raw_final_tile_checksum: checksum_f32(&actual_layer3_q_raw_final_tile),
            layer3_q_cur_final_tile_checksum: checksum_f32(&actual_layer3_q_cur_final_tile),
            layer3_kv_rope_checksum: checksum_f32(&actual_layer3_kv_rope),
            layer3_kv_cur_checksum: checksum_f32(&actual_layer3_kv_cur),
            layer3_attn_compressed_checksum: checksum_f32(&actual_layer3_attn_compressed),
            layer3_attn_state_kv_checksum: checksum_f32(&actual_layer3_attn_state_kv),
            layer3_attn_state_score_checksum: checksum_i32(&actual_layer3_attn_state_score),
            layer3_attention_output_checksum: checksum_f32(&actual_layer3_attention),
            layer3_after_attention_hc_checksum: checksum_f32(&actual_layer3_after_attention_hc),
            layer3_after_ffn_hc_checksum: checksum_f32(&actual_layer3_after_ffn_hc),
            layer4_qkv_checksums: actual_layer4_qkv
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer4_compressor_checksums: actual_layer4_compressor
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer4_attention_output_checksum: checksum_f32(&actual_layer4_attention),
            layer4_after_attention_hc_checksum: checksum_f32(&actual_layer4_after_attention_hc),
            layer4_after_ffn_hc_checksum: checksum_f32(&actual_layer4_after_ffn_hc),
            layer5_qkv_checksums: actual_layer5_qkv
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer5_compressor_checksums: [
                checksum_f32(&actual_layer5_attn_compressed),
                checksum_f32(&actual_layer5_attn_state_kv),
                checksum_i32(&actual_layer5_attn_state_score),
            ],
            layer5_attention_output_checksum: checksum_f32(&actual_layer5_attention),
            layer5_after_attention_hc_checksum: checksum_f32(&actual_layer5_after_attention_hc),
            layer5_after_ffn_hc_checksum: checksum_f32(&actual_layer5_after_ffn_hc),
            layer6_qkv_checksums: actual_layer6_qkv
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer6_compressor_checksums: actual_layer6_compressor
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer6_attention_output_checksum: checksum_f32(&actual_layer6_attention),
            layer6_after_attention_hc_checksum: checksum_f32(&actual_layer6_after_attention_hc),
            layer6_after_ffn_hc_checksum: checksum_f32(&actual_layer6_after_ffn_hc),
            layer7_qkv_checksums: actual_layer7_qkv
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer7_compressor_checksums: [
                checksum_f32(&actual_layer7_attn_compressed),
                checksum_f32(&actual_layer7_attn_state_kv),
                checksum_i32(&actual_layer7_attn_state_score),
            ],
            layer7_attention_output_checksum: checksum_f32(&actual_layer7_attention),
            layer7_after_attention_hc_checksum: checksum_f32(&actual_layer7_after_attention_hc),
            layer7_after_ffn_hc_checksum: checksum_f32(&actual_layer7_after_ffn_hc),
            layer8_qkv_checksums: actual_layer8_qkv
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer8_compressor_checksums: actual_layer8_compressor
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer8_attention_output_checksum: checksum_f32(&actual_layer8_attention),
            layer8_after_attention_hc_checksum: checksum_f32(&actual_layer8_after_attention_hc),
            layer8_after_ffn_hc_checksum: checksum_f32(&actual_layer8_after_ffn_hc),
        })
    }

    fn run_sparse_indexed_attention_fixture_probe(
        model: &MappedModel,
        default_boundary: bool,
    ) -> Result<SparseIndexedAttentionProbeReport> {
        let q_weight = exact_tensor(model, "blk.2.indexer.attn_q_b.weight", 1, &[1024, 8192])?;
        let indexer_weight = exact_tensor(model, "blk.2.indexer.proj.weight", 1, &[4096, 64])?;
        let sinks = exact_tensor(model, "blk.2.attn_sinks.weight", 0, &[64])?;
        let choose = |diagnostic: &'static [u8], default: &'static [u8]| {
            if default_boundary {
                default
            } else {
                diagnostic
            }
        };
        let position = if default_boundary { 4099 } else { 2051 };
        let compressed_rows = if default_boundary { 1025 } else { 513 };
        let q_lora_norm = decode_f32_fixture(
            choose(SPARSE_Q_LORA_NORM_BYTES, SPARSE_DEFAULT_Q_LORA_NORM_BYTES),
            "sparse Q-Lora norm",
        )?;
        let attn_norm = decode_f32_fixture(
            choose(SPARSE_ATTN_NORM_BYTES, SPARSE_DEFAULT_ATTN_NORM_BYTES),
            "sparse attention norm",
        )?;
        let q_current = decode_f32_fixture(
            choose(SPARSE_Q_CURRENT_BYTES, SPARSE_DEFAULT_Q_CURRENT_BYTES),
            "sparse Q current",
        )?;
        let raw_cache = decode_f32_fixture(
            choose(SPARSE_RAW_CACHE_BYTES, SPARSE_DEFAULT_RAW_CACHE_BYTES),
            "sparse raw cache",
        )?;
        let attention_comp_cache = decode_f32_fixture(
            choose(
                SPARSE_ATTN_COMP_CACHE_BYTES,
                SPARSE_DEFAULT_ATTN_COMP_CACHE_BYTES,
            ),
            "sparse attention compressed cache",
        )?;
        let indexer_comp_cache = decode_f32_fixture(
            choose(
                SPARSE_INDEX_COMP_CACHE_BYTES,
                SPARSE_DEFAULT_INDEX_COMP_CACHE_BYTES,
            ),
            "sparse indexer compressed cache",
        )?;
        let expected_q = decode_f32_fixture(
            choose(SPARSE_INDEXER_Q_BYTES, SPARSE_DEFAULT_INDEXER_Q_BYTES),
            "sparse indexer Q",
        )?;
        let expected_weights = decode_f32_fixture(
            choose(
                SPARSE_INDEXER_WEIGHTS_BYTES,
                SPARSE_DEFAULT_INDEXER_WEIGHTS_BYTES,
            ),
            "sparse indexer weights",
        )?;
        let expected_scores = decode_f32_fixture(
            choose(
                SPARSE_INDEXER_SCORES_BYTES,
                SPARSE_DEFAULT_INDEXER_SCORES_BYTES,
            ),
            "sparse indexer scores",
        )?;
        let expected_topk = decode_i32_fixture(
            choose(SPARSE_INDEXER_TOPK_BYTES, SPARSE_DEFAULT_INDEXER_TOPK_BYTES),
            "sparse indexer top-k",
        )?;
        let expected_out = decode_f32_fixture(
            choose(SPARSE_KQV_OUT_BYTES, SPARSE_DEFAULT_KQV_OUT_BYTES),
            "sparse KQV output",
        )?;
        let expected_back = decode_f32_fixture(
            choose(SPARSE_KQV_BACK_BYTES, SPARSE_DEFAULT_KQV_BACK_BYTES),
            "sparse KQV back",
        )?;
        let mut actual_q = vec![0.0_f32; expected_q.len()];
        let mut actual_weights = vec![0.0_f32; expected_weights.len()];
        let mut actual_scores = vec![0.0_f32; expected_scores.len()];
        let mut actual_topk = vec![0_i32; expected_topk.len()];
        let mut actual_out = vec![0.0_f32; expected_out.len()];
        let mut actual_back = vec![0.0_f32; expected_back.len()];
        let context = Context::new()?;
        let mut raw = RawSparseIndexedResult::default();
        let mut error = [0 as c_char; ERROR_BYTES];
        let succeeded = unsafe {
            rust_star_metal_run_sparse_indexed_attention(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                q_weight.absolute_offset,
                q_weight.bytes,
                indexer_weight.absolute_offset,
                indexer_weight.bytes,
                sinks.absolute_offset,
                sinks.bytes,
                position,
                compressed_rows,
                q_lora_norm.as_ptr(),
                attn_norm.as_ptr(),
                q_current.as_ptr(),
                raw_cache.as_ptr(),
                attention_comp_cache.as_ptr(),
                indexer_comp_cache.as_ptr(),
                actual_q.as_mut_ptr(),
                actual_weights.as_mut_ptr(),
                actual_scores.as_mut_ptr(),
                actual_topk.as_mut_ptr(),
                actual_out.as_mut_ptr(),
                actual_back.as_mut_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal sparse indexed-attention probe failed: {}",
                error_text(&error)
            )));
        }
        for (label, actual, expected) in [
            ("indexer Q", &actual_q[..], &expected_q[..]),
            (
                "indexer weights",
                &actual_weights[..],
                &expected_weights[..],
            ),
            ("indexer scores", &actual_scores[..], &expected_scores[..]),
            ("KQV output", &actual_out[..], &expected_out[..]),
            ("KQV back", &actual_back[..], &expected_back[..]),
        ] {
            if let Some((index, (actual, expected))) = actual
                .iter()
                .zip(expected)
                .enumerate()
                .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
            {
                return Err(Error::invalid(format!(
                    "sparse indexed-attention {label} C0 mismatch at {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits()
                )));
            }
        }
        if let Some((index, (actual, expected))) = actual_topk
            .iter()
            .zip(&expected_topk)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            return Err(Error::invalid(format!(
                "sparse indexed-attention top-k mismatch at {index}: actual={actual} expected={expected}"
            )));
        }
        if raw.position != position
            || raw.compressed_rows != compressed_rows
            || raw.raw_rows != 128
            || raw.top_k != 512
            || raw.dispatches != if default_boundary { 11 } else { 10 }
            || raw.wrapped_model_ranges != 3
            || raw.pointer_matches != 3
            || raw.split_count != 12
            || !raw.wall_ms.is_finite()
            || raw.wall_ms <= 0.0
            || !raw.gpu_ms.is_finite()
            || raw.gpu_ms <= 0.0
        {
            return Err(Error::invalid(
                "Metal sparse indexed-attention result metadata is invalid",
            ));
        }
        Ok(SparseIndexedAttentionProbeReport {
            fixture_id: if default_boundary {
                SPARSE_INDEXED_ATTENTION_DEFAULT_FIXTURE_ID
            } else {
                SPARSE_INDEXED_ATTENTION_FIXTURE_ID
            },
            position: raw.position,
            compressed_rows: raw.compressed_rows,
            raw_rows: raw.raw_rows,
            top_k: raw.top_k,
            diagnostic_threshold_override: if default_boundary { 0 } else { 512 },
            pinned_default_threshold: 1024,
            first_default_sparse_rows: 1025,
            dispatches: raw.dispatches,
            wrapped_model_ranges: raw.wrapped_model_ranges,
            pointer_matches: raw.pointer_matches,
            split_count: raw.split_count,
            wall_ms: raw.wall_ms,
            gpu_ms: raw.gpu_ms,
            indexer_q_checksum: checksum_f32(&actual_q),
            indexer_weights_checksum: checksum_f32(&actual_weights),
            indexer_scores_checksum: checksum_f32(&actual_scores),
            indexer_topk_checksum: checksum_i32(&actual_topk),
            kqv_out_checksum: checksum_f32(&actual_out),
            kqv_back_checksum: checksum_f32(&actual_back),
        })
    }

    pub fn run_sparse_indexed_attention_probe(
        model: &MappedModel,
    ) -> Result<SparseIndexedAttentionProbeReport> {
        let _diagnostic = run_sparse_indexed_attention_fixture_probe(model, false)?;
        run_sparse_indexed_attention_fixture_probe(model, true)
    }

    fn run_retained_sparse_fixture_probe(
        model: &MappedModel,
        multimerge: bool,
    ) -> Result<RetainedSparseBoundaryProbeReport> {
        let position = if multimerge { 8195 } else { 4099 };
        let token = if multimerge { 381 } else { 0 };
        let context_capacity = position + 1;
        let compressed_rows = if multimerge { 2049 } else { 1025 };
        let seeded_compressed_rows = compressed_rows - 1;
        let (sort_blocks, merge_passes, topk_work_width) =
            retained_sparse_topk_schedule(compressed_rows);
        let expected_dispatches = 53 + merge_passes;
        let (
            fixture_id,
            input_hc_bytes,
            raw_prior_bytes,
            attention_prior_bytes,
            indexer_prior_bytes,
            attention_state_kv_bytes,
            attention_state_score_bits,
            indexer_state_kv_bytes,
            indexer_state_score_bits,
            q_lora_norm_bytes,
            attn_norm_bytes,
            q_cur_bytes,
            kv_cur_bytes,
            compressed_kv_bytes,
            compressed_indexer_bytes,
            indexer_q_bytes,
            indexer_weights_bytes,
            indexer_scores_bytes,
            indexer_topk_bytes,
            kqv_out_bytes,
            kqv_back_bytes,
            attn_low_bytes,
            attn_out_bytes,
            hc_attn_post_bytes,
        ) = if multimerge {
            (
                RETAINED_SPARSE_MULTIMERGE_FIXTURE_ID,
                RETAINED_MULTIMERGE_INPUT_HC_BYTES,
                RETAINED_MULTIMERGE_RAW_PRIOR_BYTES,
                RETAINED_MULTIMERGE_ATTN_COMP_PRIOR_BYTES,
                RETAINED_MULTIMERGE_INDEX_COMP_PRIOR_BYTES,
                RETAINED_MULTIMERGE_ATTN_STATE_KV_BYTES,
                RETAINED_MULTIMERGE_ATTN_STATE_SCORE_BITS,
                RETAINED_MULTIMERGE_INDEX_STATE_KV_BYTES,
                RETAINED_MULTIMERGE_INDEX_STATE_SCORE_BITS,
                RETAINED_MULTIMERGE_Q_LORA_NORM_BYTES,
                RETAINED_MULTIMERGE_ATTN_NORM_BYTES,
                RETAINED_MULTIMERGE_Q_CUR_BYTES,
                RETAINED_MULTIMERGE_KV_CUR_BYTES,
                RETAINED_MULTIMERGE_COMPRESSED_KV_BYTES,
                RETAINED_MULTIMERGE_COMPRESSED_INDEXER_BYTES,
                RETAINED_MULTIMERGE_INDEXER_Q_BYTES,
                RETAINED_MULTIMERGE_INDEXER_WEIGHTS_BYTES,
                RETAINED_MULTIMERGE_INDEXER_SCORES_BYTES,
                RETAINED_MULTIMERGE_INDEXER_TOPK_BYTES,
                RETAINED_MULTIMERGE_KQV_OUT_BYTES,
                RETAINED_MULTIMERGE_KQV_BACK_BYTES,
                RETAINED_MULTIMERGE_ATTN_LOW_BYTES,
                RETAINED_MULTIMERGE_ATTN_OUT_BYTES,
                RETAINED_MULTIMERGE_HC_ATTN_POST_BYTES,
            )
        } else {
            (
                RETAINED_SPARSE_BOUNDARY_FIXTURE_ID,
                RETAINED_SPARSE_INPUT_HC_BYTES,
                RETAINED_SPARSE_RAW_PRIOR_BYTES,
                RETAINED_SPARSE_ATTN_COMP_PRIOR_BYTES,
                RETAINED_SPARSE_INDEX_COMP_PRIOR_BYTES,
                RETAINED_SPARSE_ATTN_STATE_KV_BYTES,
                RETAINED_SPARSE_ATTN_STATE_SCORE_BITS,
                RETAINED_SPARSE_INDEX_STATE_KV_BYTES,
                RETAINED_SPARSE_INDEX_STATE_SCORE_BITS,
                RETAINED_SPARSE_Q_LORA_NORM_BYTES,
                RETAINED_SPARSE_ATTN_NORM_BYTES,
                RETAINED_SPARSE_Q_CUR_BYTES,
                RETAINED_SPARSE_KV_CUR_BYTES,
                RETAINED_SPARSE_COMPRESSED_KV_BYTES,
                RETAINED_SPARSE_COMPRESSED_INDEXER_BYTES,
                RETAINED_SPARSE_INDEXER_Q_BYTES,
                RETAINED_SPARSE_INDEXER_WEIGHTS_BYTES,
                RETAINED_SPARSE_INDEXER_SCORES_BYTES,
                RETAINED_SPARSE_INDEXER_TOPK_BYTES,
                RETAINED_SPARSE_KQV_OUT_BYTES,
                RETAINED_SPARSE_KQV_BACK_BYTES,
                RETAINED_SPARSE_ATTN_LOW_BYTES,
                RETAINED_SPARSE_ATTN_OUT_BYTES,
                RETAINED_SPARSE_HC_ATTN_POST_BYTES,
            )
        };
        let input_hc = decode_f32_fixture(input_hc_bytes, "retained input HC")?;
        let raw_prior = decode_f32_fixture(raw_prior_bytes, "retained raw-cache seed")?;
        let attention_prior = decode_f32_fixture(
            attention_prior_bytes,
            "retained attention compressed-cache seed",
        )?;
        let indexer_prior = decode_f32_fixture(
            indexer_prior_bytes,
            "retained indexer compressed-cache seed",
        )?;
        let attention_state_kv = decode_f32_fixture(
            attention_state_kv_bytes,
            "retained attention recurrent KV state",
        )?;
        let attention_state_score_bits = decode_i32_fixture(
            attention_state_score_bits,
            "retained attention recurrent score state bits",
        )?;
        let attention_state_score = attention_state_score_bits
            .iter()
            .map(|bits| f32::from_bits(*bits as u32))
            .collect::<Vec<_>>();
        let indexer_state_kv = decode_f32_fixture(
            indexer_state_kv_bytes,
            "retained indexer recurrent KV state",
        )?;
        let indexer_state_score_bits = decode_i32_fixture(
            indexer_state_score_bits,
            "retained indexer recurrent score state bits",
        )?;
        let indexer_state_score = indexer_state_score_bits
            .iter()
            .map(|bits| f32::from_bits(*bits as u32))
            .collect::<Vec<_>>();

        let context = Context::new()?;
        let mut error = [0 as c_char; ERROR_BYTES];
        let seeded = unsafe {
            let seed = if multimerge {
                rust_star_metal_seed_retained_sparse_layer2_position8195
            } else {
                rust_star_metal_seed_retained_sparse_layer2_position4099
            };
            seed(
                context.0,
                input_hc.as_ptr(),
                raw_prior.as_ptr(),
                attention_prior.as_ptr(),
                indexer_prior.as_ptr(),
                attention_state_kv.as_ptr(),
                attention_state_score.as_ptr(),
                indexer_state_kv.as_ptr(),
                indexer_state_score.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if seeded == 0 {
            return Err(Error::invalid(format!(
                "Metal retained sparse-boundary seed failed: {}",
                error_text(&error)
            )));
        }

        let mut prepared =
            PreparedLayerExecution::new_cold_with_capacity(model, 2, context_capacity)?;
        prepared.validate_expected = false;
        run_prepared_layer_iterations(
            model,
            &context,
            &mut prepared,
            token,
            position,
            0,
            1,
            COMMAND_CHAINED_FINAL,
            2,
        )?;
        let execution = run_prepared_layer_iterations(
            model,
            &context,
            &mut prepared,
            token,
            position,
            0,
            1,
            COMMAND_CHAINED_COLLECT,
            2,
        )?;

        let mut indexer_q = vec![0.0_f32; 64 * 128];
        let mut indexer_weights = vec![0.0_f32; 64];
        let mut indexer_scores = vec![0.0_f32; compressed_rows as usize];
        let mut indexer_topk = vec![0_i32; 512];
        error.fill(0);
        let copied = unsafe {
            let copy = if multimerge {
                rust_star_metal_copy_retained_sparse_layer2_position8195
            } else {
                rust_star_metal_copy_retained_sparse_layer2_position4099
            };
            copy(
                context.0,
                indexer_q.as_mut_ptr(),
                indexer_weights.as_mut_ptr(),
                indexer_scores.as_mut_ptr(),
                indexer_topk.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if copied == 0 {
            return Err(Error::invalid(format!(
                "Metal retained sparse-boundary diagnostic readback failed: {}",
                error_text(&error)
            )));
        }

        let expected_q_lora_norm =
            decode_f32_fixture(q_lora_norm_bytes, "retained sparse Q-Lora norm")?;
        let expected_attn_norm =
            decode_f32_fixture(attn_norm_bytes, "retained sparse attention norm")?;
        let expected_q_cur = decode_f32_fixture(q_cur_bytes, "retained sparse Q current")?;
        let expected_kv_cur = decode_f32_fixture(kv_cur_bytes, "retained sparse KV current")?;
        let expected_compressed_kv =
            decode_f32_fixture(compressed_kv_bytes, "retained sparse compressed KV row")?;
        let expected_compressed_indexer = decode_f32_fixture(
            compressed_indexer_bytes,
            "retained sparse compressed indexer row",
        )?;
        let expected_indexer_q = decode_f32_fixture(indexer_q_bytes, "retained sparse indexer Q")?;
        let expected_indexer_weights =
            decode_f32_fixture(indexer_weights_bytes, "retained sparse indexer weights")?;
        let expected_indexer_scores =
            decode_f32_fixture(indexer_scores_bytes, "retained sparse indexer scores")?;
        let expected_indexer_topk =
            decode_i32_fixture(indexer_topk_bytes, "retained sparse indexer top-k")?;
        let expected_kqv_out = decode_f32_fixture(kqv_out_bytes, "retained sparse KQV output")?;
        let expected_kqv_back = decode_f32_fixture(kqv_back_bytes, "retained sparse KQV back")?;
        let expected_attn_low =
            decode_f32_fixture(attn_low_bytes, "retained sparse attention low projection")?;
        let expected_attn_out =
            decode_f32_fixture(attn_out_bytes, "retained sparse attention projection")?;
        let expected_hc_post =
            decode_f32_fixture(hc_attn_post_bytes, "retained sparse attention HC post")?;

        let check_f32 = |label: &str, actual: &[f32], expected: &[f32]| -> Result<()> {
            if actual.len() != expected.len() {
                return Err(Error::invalid(format!(
                    "retained sparse-boundary {label} length mismatch"
                )));
            }
            if let Some((index, (actual, expected))) = actual
                .iter()
                .zip(expected)
                .enumerate()
                .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
            {
                return Err(Error::invalid(format!(
                    "retained sparse-boundary {label} C0 mismatch at {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(), expected.to_bits()
                )));
            }
            Ok(())
        };
        for (label, actual, expected) in [
            (
                "Q-Lora norm",
                prepared.q_lora_norm.as_slice(),
                expected_q_lora_norm.as_slice(),
            ),
            (
                "attention norm",
                prepared.norm.as_slice(),
                expected_attn_norm.as_slice(),
            ),
            (
                "Q current",
                prepared.q_cur.as_slice(),
                expected_q_cur.as_slice(),
            ),
            (
                "KV current",
                prepared.kv_cur.as_slice(),
                expected_kv_cur.as_slice(),
            ),
            (
                "compressed KV",
                prepared.compressed_kv.as_slice(),
                expected_compressed_kv.as_slice(),
            ),
            (
                "compressed indexer",
                prepared.compressed_indexer.as_slice(),
                expected_compressed_indexer.as_slice(),
            ),
            (
                "indexer Q",
                indexer_q.as_slice(),
                expected_indexer_q.as_slice(),
            ),
            (
                "indexer weights",
                indexer_weights.as_slice(),
                expected_indexer_weights.as_slice(),
            ),
            (
                "indexer scores",
                indexer_scores.as_slice(),
                expected_indexer_scores.as_slice(),
            ),
            (
                "KQV output",
                prepared.attention_raw.as_slice(),
                expected_kqv_out.as_slice(),
            ),
            (
                "KQV back",
                prepared.attention_back.as_slice(),
                expected_kqv_back.as_slice(),
            ),
            (
                "attention low",
                prepared.attention_low.as_slice(),
                expected_attn_low.as_slice(),
            ),
            (
                "attention output",
                prepared.attention_out.as_slice(),
                expected_attn_out.as_slice(),
            ),
            (
                "attention HC post",
                prepared.after_attention_hc.as_slice(),
                expected_hc_post.as_slice(),
            ),
        ] {
            check_f32(label, actual, expected)?;
        }
        let raw_slot = (position as usize % 128) * 512;
        let expected_stored = expected_kv_cur
            .iter()
            .copied()
            .map(f16_round_f32)
            .collect::<Vec<_>>();
        check_f32(
            "raw cache current row",
            &prepared.cache_rows[raw_slot..raw_slot + 512],
            &expected_stored,
        )?;
        if indexer_topk != expected_indexer_topk {
            let mismatch = indexer_topk
                .iter()
                .zip(&expected_indexer_topk)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(0);
            return Err(Error::invalid(format!(
                "retained sparse-boundary top-k mismatch at {mismatch}: actual={} expected={}",
                indexer_topk[mismatch], expected_indexer_topk[mismatch]
            )));
        }
        if multimerge {
            let expected_attention_mixes = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_ATTN_PRE_MIXES_BYTES,
                "retained attention HC pre mixes",
            )?;
            let expected_attention_weights = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_ATTN_PRE_WEIGHTS_BYTES,
                "retained attention HC pre weights",
            )?;
            let expected_attention_post_weights = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_ATTN_PRE_POST_WEIGHTS_BYTES,
                "retained attention HC pre post-weights",
            )?;
            let expected_attention_comb = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_ATTN_PRE_COMB_BYTES,
                "retained attention HC pre combination",
            )?;
            let expected_attention_pre = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_ATTN_PRE_BYTES,
                "retained attention HC pre",
            )?;
            let expected_q_lora =
                decode_f32_fixture(RETAINED_MULTIMERGE_Q_LORA_BYTES, "retained Q-Lora")?;
            let expected_kv_raw =
                decode_f32_fixture(RETAINED_MULTIMERGE_KV_RAW_BYTES, "retained raw KV")?;
            let expected_kv_norm =
                decode_f32_fixture(RETAINED_MULTIMERGE_KV_NORM_BYTES, "retained normalized KV")?;
            let expected_q_raw =
                decode_f32_fixture(RETAINED_MULTIMERGE_Q_RAW_BYTES, "retained raw Q")?;
            let expected_kv_rope = decode_f32_fixture(
                RETAINED_MULTIMERGE_KV_ROPE_BYTES,
                "retained KV before cache store",
            )?;
            let expected_ffn_mixes = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_FFN_PRE_MIXES_BYTES,
                "retained FFN HC pre mixes",
            )?;
            let expected_ffn_weights = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_FFN_PRE_WEIGHTS_BYTES,
                "retained FFN HC pre weights",
            )?;
            let expected_ffn_post_weights = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_FFN_PRE_POST_WEIGHTS_BYTES,
                "retained FFN HC pre post-weights",
            )?;
            let expected_ffn_comb = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_FFN_PRE_COMB_BYTES,
                "retained FFN HC pre combination",
            )?;
            let expected_ffn_pre =
                decode_f32_fixture(RETAINED_MULTIMERGE_HC_FFN_PRE_BYTES, "retained FFN HC pre")?;
            let expected_ffn_norm =
                decode_f32_fixture(RETAINED_MULTIMERGE_FFN_NORM_BYTES, "retained FFN norm")?;
            let expected_router_logits = decode_f32_fixture(
                RETAINED_MULTIMERGE_FFN_MOE_LOGITS_BYTES,
                "retained FFN router logits",
            )?;
            let expected_router_probs = decode_f32_fixture(
                RETAINED_MULTIMERGE_FFN_MOE_PROBS_BYTES,
                "retained FFN router probabilities",
            )?;
            let expected_selected = decode_i32_fixture(
                RETAINED_MULTIMERGE_FFN_MOE_TOPK_BYTES,
                "retained FFN selected experts",
            )?;
            let expected_router_weights = decode_f32_fixture(
                RETAINED_MULTIMERGE_FFN_MOE_WEIGHTS_SCALED_BYTES,
                "retained FFN scaled router weights",
            )?;
            let expected_routed_mid = decode_f32_fixture(
                RETAINED_MULTIMERGE_FFN_MOE_WEIGHTED_SWIGLU_BYTES,
                "retained FFN weighted SwiGLU",
            )?;
            let expected_routed_out = decode_f32_fixture(
                RETAINED_MULTIMERGE_FFN_MOE_OUT_BYTES,
                "retained FFN routed output",
            )?;
            let expected_shared_out = decode_f32_fixture(
                RETAINED_MULTIMERGE_FFN_SHEXP_BYTES,
                "retained FFN shared-expert output",
            )?;
            let expected_final_hc = decode_f32_fixture(
                RETAINED_MULTIMERGE_HC_FFN_POST_BYTES,
                "retained final layer HC",
            )?;

            for (label, actual, expected) in [
                (
                    "attention HC pre mixes",
                    prepared.mixes.as_slice(),
                    expected_attention_mixes.as_slice(),
                ),
                (
                    "attention HC pre weights",
                    &prepared.split[0..4],
                    expected_attention_weights.as_slice(),
                ),
                (
                    "attention HC pre post-weights",
                    &prepared.split[4..8],
                    expected_attention_post_weights.as_slice(),
                ),
                (
                    "attention HC pre combination",
                    &prepared.split[8..24],
                    expected_attention_comb.as_slice(),
                ),
                (
                    "attention HC pre",
                    prepared.collapsed.as_slice(),
                    expected_attention_pre.as_slice(),
                ),
                (
                    "Q-Lora",
                    prepared.q_lora.as_slice(),
                    expected_q_lora.as_slice(),
                ),
                (
                    "raw KV",
                    prepared.kv_raw.as_slice(),
                    expected_kv_raw.as_slice(),
                ),
                (
                    "normalized KV",
                    prepared.kv_norm_pre_rope.as_slice(),
                    expected_kv_norm.as_slice(),
                ),
                (
                    "raw Q",
                    prepared.q_raw.as_slice(),
                    expected_q_raw.as_slice(),
                ),
                (
                    "KV before cache store",
                    prepared.kv_rope.as_slice(),
                    expected_kv_rope.as_slice(),
                ),
                (
                    "FFN HC pre mixes",
                    prepared.ffn_mixes.as_slice(),
                    expected_ffn_mixes.as_slice(),
                ),
                (
                    "FFN HC pre weights",
                    &prepared.ffn_split[0..4],
                    expected_ffn_weights.as_slice(),
                ),
                (
                    "FFN HC pre post-weights",
                    &prepared.ffn_split[4..8],
                    expected_ffn_post_weights.as_slice(),
                ),
                (
                    "FFN HC pre combination",
                    &prepared.ffn_split[8..24],
                    expected_ffn_comb.as_slice(),
                ),
                (
                    "FFN HC pre",
                    prepared.ffn_cur.as_slice(),
                    expected_ffn_pre.as_slice(),
                ),
                (
                    "FFN norm",
                    prepared.ffn_norm.as_slice(),
                    expected_ffn_norm.as_slice(),
                ),
                (
                    "FFN router logits",
                    prepared.router_logits.as_slice(),
                    expected_router_logits.as_slice(),
                ),
                (
                    "FFN router probabilities",
                    prepared.router_probs.as_slice(),
                    expected_router_probs.as_slice(),
                ),
                (
                    "FFN scaled router weights",
                    prepared.router_weights.as_slice(),
                    expected_router_weights.as_slice(),
                ),
                (
                    "FFN weighted SwiGLU",
                    prepared.routed_mid.as_slice(),
                    expected_routed_mid.as_slice(),
                ),
                (
                    "FFN routed output",
                    prepared.routed_out.as_slice(),
                    expected_routed_out.as_slice(),
                ),
                (
                    "FFN shared-expert output",
                    prepared.shared_out.as_slice(),
                    expected_shared_out.as_slice(),
                ),
                (
                    "final layer HC",
                    prepared.after_ffn_hc.as_slice(),
                    expected_final_hc.as_slice(),
                ),
            ] {
                check_f32(label, actual, expected)?;
            }
            if prepared.selected != expected_selected {
                let mismatch = prepared
                    .selected
                    .iter()
                    .zip(&expected_selected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(0);
                return Err(Error::invalid(format!(
                    "retained sparse-boundary FFN selected-expert mismatch at {mismatch}: actual={} expected={}",
                    prepared.selected[mismatch], expected_selected[mismatch]
                )));
            }
        }
        if execution.report.dispatches != expected_dispatches
            || execution.report.wrapped_model_ranges != 35
            || execution.report.pointer_matches != 35
        {
            return Err(Error::invalid(
                "retained sparse-boundary execution metadata is invalid",
            ));
        }

        Ok(RetainedSparseBoundaryProbeReport {
            fixture_id,
            token,
            layer: 2,
            position,
            raw_rows: 128,
            seeded_raw_rows: 127,
            compressed_rows,
            seeded_compressed_rows,
            top_k: 512,
            sort_blocks,
            merge_passes,
            topk_work_width,
            dispatches: execution.report.dispatches,
            wrapped_model_ranges: execution.report.wrapped_model_ranges,
            pointer_matches: execution.report.pointer_matches,
            wall_ms: execution.report.wall_ms,
            gpu_ms: execution.report.gpu_ms,
            exact_tensor_checks: if multimerge { 40 } else { 16 },
            q_current_checksum: checksum_f32(&prepared.q_cur),
            compressed_kv_checksum: checksum_f32(&prepared.compressed_kv),
            compressed_indexer_checksum: checksum_f32(&prepared.compressed_indexer),
            indexer_scores_checksum: checksum_f32(&indexer_scores),
            indexer_topk_checksum: checksum_i32(&indexer_topk),
            kqv_out_checksum: checksum_f32(&prepared.attention_raw),
            kqv_back_checksum: checksum_f32(&prepared.attention_back),
            attention_hc_checksum: checksum_f32(&prepared.after_attention_hc),
            selected_experts_checksum: checksum_i32(&prepared.selected),
            final_hc_checksum: checksum_f32(&prepared.after_ffn_hc),
        })
    }

    pub fn run_retained_sparse_boundary_probe(
        model: &MappedModel,
    ) -> Result<RetainedSparseBoundaryProbeReport> {
        run_retained_sparse_fixture_probe(model, false)
    }

    pub fn run_retained_sparse_multimerge_probe(
        model: &MappedModel,
    ) -> Result<RetainedSparseBoundaryProbeReport> {
        run_retained_sparse_fixture_probe(model, true)
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
                1,
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

    fn run_sampling_output_head(
        model: &MappedModel,
        context: &Context,
        prepared: &mut PreparedOutputHead,
    ) -> Result<(u32, f64)> {
        let mut raw = RawIngressProbeResult::default();
        let mut error = [0 as c_char; ERROR_BYTES];
        let succeeded = unsafe {
            rust_star_metal_run_output_head(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                prepared.hc_fn.absolute_offset,
                prepared.hc_fn.bytes,
                prepared.hc_scale.absolute_offset,
                prepared.hc_scale.bytes,
                prepared.hc_base.absolute_offset,
                prepared.hc_base.bytes,
                prepared.output_norm.absolute_offset,
                prepared.output_norm.bytes,
                prepared.output.absolute_offset,
                prepared.output.bytes,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                prepared.logits.as_mut_ptr(),
                0,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal sampling output head failed: {}",
                error_text(&error)
            )));
        }
        if raw.model_bytes != model.bytes()
            || raw.wrapped_model_ranges != 5
            || raw.pointer_matches != 5
            || !raw.wall_ms.is_finite()
            || raw.wall_ms <= 0.0
            || !raw.gpu_ms.is_finite()
            || raw.gpu_ms < 0.0
        {
            return Err(Error::invalid(
                "Metal sampling output head returned invalid ownership or timing metadata",
            ));
        }
        Ok((lowest_id_argmax(&prepared.logits)?, raw.gpu_ms))
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
            cache_capacity_rows: 128,
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

    fn run_decoder_output_correctness_in_context(
        model: &MappedModel,
        context: &Context,
        layers: &mut [PreparedLayerExecution],
        closed_loop_sampling: bool,
    ) -> Result<DecoderOutputProbeReport> {
        if layers.len() != 43 {
            return Err(Error::invalid(
                "decoder-output correctness requires all forty-three prepared layers",
            ));
        }
        let mut steps = Vec::with_capacity(4);
        let supplied_inputs = [201_u32, 361, 1915, 262];
        let mut next_input = supplied_inputs[0];

        for position in 1_u32..=4 {
            let input_token = if closed_loop_sampling {
                next_input
            } else {
                supplied_inputs[position as usize - 1]
            };
            if position > 1 {
                for (layer_index, layer) in layers.iter_mut().enumerate() {
                    layer.expected = layer_expected(layer_index as u32, position)?;
                }
            }
            submit_prepared_layers(model, context, layers, input_token, position)?;
            let output_head = run_retained_output_head(model, context, position)?;
            let mut reports = Vec::with_capacity(43);
            for layer in layers.iter_mut() {
                reports.push(
                    run_prepared_layer_iterations(
                        model,
                        context,
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
            next_input = output_head.selected_token;
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
            cache_capacity_rows: 128,
            logits_elements: 129280,
            closed_loop_sampling,
            externally_supplied_decode_inputs: !closed_loop_sampling,
        })
    }

    fn run_decoder_output_correctness(
        model: &MappedModel,
        closed_loop_sampling: bool,
    ) -> Result<DecoderOutputProbeReport> {
        let context = Context::new()?;
        let mut layers = (0..43)
            .map(|layer_index| PreparedLayerExecution::new(model, layer_index, 1, 1))
            .collect::<Result<Vec<_>>>()?;
        run_decoder_output_correctness_in_context(
            model,
            &context,
            &mut layers,
            closed_loop_sampling,
        )
    }

    pub fn run_decoder_output_probe(model: &MappedModel) -> Result<DecoderOutputProbeReport> {
        run_decoder_output_correctness(model, false)
    }

    pub fn run_closed_loop_decoder_probe(
        model: &MappedModel,
    ) -> Result<ClosedLoopDecoderProbeReport> {
        let context = Context::new()?;
        let mut layers = (0..43)
            .map(|layer_index| PreparedLayerExecution::new(model, layer_index, 1, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut output_head = PreparedOutputHead::new(model)?;
        let prepare_started = Instant::now();
        context.prepare_decoder()?;
        let pipeline_prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
        let correctness =
            run_decoder_output_correctness_in_context(model, &context, &mut layers, true)?;
        for (layer_index, layer) in layers.iter_mut().enumerate() {
            layer.expected = layer_expected(layer_index as u32, 1)?;
        }
        let expected_inputs = [201_u32, 361, 1915, 262];
        let expected_outputs = [361_u32, 1915, 262, 1554];
        let mut input_token = expected_inputs[0];
        let mut timed_steps = Vec::with_capacity(4);

        for position in 1_u32..=4 {
            if input_token != expected_inputs[position as usize - 1] {
                return Err(Error::invalid(format!(
                    "closed-loop input mismatch at position {position}: actual={input_token} expected={}",
                    expected_inputs[position as usize - 1]
                )));
            }
            if position > 1 {
                for (layer_index, layer) in layers.iter_mut().enumerate() {
                    layer.expected = layer_expected(layer_index as u32, position)?;
                }
            }
            let started = Instant::now();
            submit_prepared_layers(model, &context, &mut layers, input_token, position)?;
            let (selected_token, output_head_gpu_ms) =
                run_sampling_output_head(model, &context, &mut output_head)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            if selected_token != expected_outputs[position as usize - 1] {
                return Err(Error::invalid(format!(
                    "timed closed-loop selection mismatch at position {position}: actual={selected_token} expected={}",
                    expected_outputs[position as usize - 1]
                )));
            }
            timed_steps.push(TimedDecoderStepReport {
                position,
                input_token,
                selected_token,
                wall_ms,
                output_head_gpu_ms,
            });
            input_token = selected_token;
        }

        let generation_wall_ms = timed_steps.iter().map(|step| step.wall_ms).sum::<f64>();
        let first_token_ms = timed_steps[0].wall_ms;
        let steady_wall_ms = timed_steps[1..]
            .iter()
            .map(|step| step.wall_ms)
            .sum::<f64>();
        Ok(ClosedLoopDecoderProbeReport {
            correctness,
            timed_steps,
            pipeline_prepare_ms,
            generation_wall_ms,
            generation_tps: 4000.0 / generation_wall_ms,
            first_token_ms,
            steady_wall_ms,
            steady_tps: 3000.0 / steady_wall_ms,
        })
    }

    pub fn run_position127_decoder_probe(
        model: &MappedModel,
    ) -> Result<Position127DecoderProbeReport> {
        let (expected_tokens, expected_logits) = position127_decoder_fixture()?;
        let context = Context::new()?;
        let mut layers = (0..43)
            .map(|layer_index| PreparedLayerExecution::new(model, layer_index, 1, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut output_head = PreparedOutputHead::new(model)?;
        context.prepare_decoder()?;

        let mut actual_tokens = Vec::with_capacity(128);
        let mut input_token = expected_tokens[0];
        actual_tokens.push(input_token);
        let started = Instant::now();
        for position in 1_u32..=127 {
            submit_prepared_layers(model, &context, &mut layers, input_token, position)?;
            let (selected_token, _) = run_sampling_output_head(model, &context, &mut output_head)?;
            actual_tokens.push(selected_token);
            input_token = selected_token;
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

        if actual_tokens != expected_tokens {
            let mismatch = actual_tokens
                .iter()
                .zip(&expected_tokens)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(actual_tokens.len().min(expected_tokens.len()));
            return Err(Error::invalid(format!(
                "position-127 closed-loop transcript differs at committed token {mismatch}: actual={:?} expected={:?}",
                actual_tokens.get(mismatch),
                expected_tokens.get(mismatch)
            )));
        }
        let layer3_row = context.compressed_kv_row(3, 0)?;
        let layer5_row = context.compressed_kv_row(5, 0)?;
        for (layer, actual, expected_bytes) in [
            (
                3_u32,
                layer3_row.as_slice(),
                LAYER3_POS127_COMPRESSED_KV_BYTES,
            ),
            (
                5_u32,
                layer5_row.as_slice(),
                LAYER5_POS127_COMPRESSED_KV_BYTES,
            ),
        ] {
            let expected = decode_f32_fixture(
                expected_bytes,
                &format!("layer-{layer} position-127 compressed KV row"),
            )?;
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "integrated layer-{layer} ratio-128 C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits()
                    )));
                }
            }
        }
        for (index, (actual, expected)) in
            output_head.logits.iter().zip(&expected_logits).enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "position-127 final-logit C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }

        Ok(Position127DecoderProbeReport {
            fixture_id: POSITION127_DECODER_FIXTURE_ID,
            committed_tokens: actual_tokens,
            evaluated_positions: 127,
            final_position: 127,
            cache_capacity_rows: 128,
            compressed_cache_capacity_rows: 32,
            command_buffers_per_position: 44,
            host_waits_per_position: 2,
            wall_ms,
            eval_tps: 127000.0 / wall_ms,
            final_logits_checksum: checksum_f32(&output_head.logits),
            ratio128_layer3_checksum: checksum_f32(&layer3_row),
            ratio128_layer5_checksum: checksum_f32(&layer5_row),
        })
    }

    pub fn run_cold_prefill_decoder_probe(
        model: &MappedModel,
    ) -> Result<ColdPrefillDecoderProbeReport> {
        let expected_prefill_logits = cold_prefill_fixture()?;
        let (expected_tokens, expected_final_logits) = position127_decoder_fixture()?;
        let context = Context::new()?;
        let mut layers = (0..43)
            .map(|layer_index| PreparedLayerExecution::new_cold(model, layer_index))
            .collect::<Result<Vec<_>>>()?;
        let mut output_head = PreparedOutputHead::new(model)?;
        context.prepare_decoder()?;

        let prefill_started = Instant::now();
        submit_prepared_layers(model, &context, &mut layers, 36662, 0)?;
        let (first_token, _) = run_sampling_output_head(model, &context, &mut output_head)?;
        let prefill_wall_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        if first_token != expected_tokens[0] {
            return Err(Error::invalid(format!(
                "cold prefill selected token {first_token}, expected {}",
                expected_tokens[0]
            )));
        }
        for (index, (actual, expected)) in output_head
            .logits
            .iter()
            .zip(&expected_prefill_logits)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "cold-prefill logit C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }
        let prefill_logits_checksum = checksum_f32(&output_head.logits);

        let mut actual_tokens = Vec::with_capacity(128);
        actual_tokens.push(first_token);
        let mut input_token = first_token;
        let decode_started = Instant::now();
        for position in 1_u32..=127 {
            submit_prepared_layers(model, &context, &mut layers, input_token, position)?;
            let (selected_token, _) = run_sampling_output_head(model, &context, &mut output_head)?;
            actual_tokens.push(selected_token);
            input_token = selected_token;
        }
        let decode_wall_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
        if actual_tokens != expected_tokens {
            let mismatch = actual_tokens
                .iter()
                .zip(&expected_tokens)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(actual_tokens.len().min(expected_tokens.len()));
            return Err(Error::invalid(format!(
                "cold-prefill transcript differs at committed token {mismatch}: actual={:?} expected={:?}",
                actual_tokens.get(mismatch),
                expected_tokens.get(mismatch)
            )));
        }

        let layer3_row = context.compressed_kv_row(3, 0)?;
        let layer5_row = context.compressed_kv_row(5, 0)?;
        for (layer, actual, expected_bytes) in [
            (
                3_u32,
                layer3_row.as_slice(),
                LAYER3_POS127_COMPRESSED_KV_BYTES,
            ),
            (
                5_u32,
                layer5_row.as_slice(),
                LAYER5_POS127_COMPRESSED_KV_BYTES,
            ),
        ] {
            let expected = decode_f32_fixture(
                expected_bytes,
                &format!("layer-{layer} position-127 compressed KV row"),
            )?;
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(Error::invalid(format!(
                        "cold-prefill layer-{layer} ratio-128 C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                        actual.to_bits(),
                        expected.to_bits()
                    )));
                }
            }
        }
        for (index, (actual, expected)) in output_head
            .logits
            .iter()
            .zip(&expected_final_logits)
            .enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "cold-prefill final-logit C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }

        Ok(ColdPrefillDecoderProbeReport {
            fixture_id: COLD_PREFILL_FIXTURE_ID,
            prompt_token: 36662,
            committed_tokens: actual_tokens,
            prefill_wall_ms,
            prefill_logits_checksum,
            decode_wall_ms,
            decode_tps: 127000.0 / decode_wall_ms,
            final_logits_checksum: checksum_f32(&output_head.logits),
            ratio128_layer3_checksum: checksum_f32(&layer3_row),
            ratio128_layer5_checksum: checksum_f32(&layer5_row),
        })
    }

    pub fn run_prefill_frontier_probe(model: &MappedModel) -> Result<PrefillFrontierProbeReport> {
        let (tokens, batch_logits, decode_logits) = prefill_frontier_2048_fixture()?;
        let context_capacity = tokens.len() as u32;
        let context = Context::new()?;
        let mut layers = (0..43)
            .map(|layer_index| {
                PreparedLayerExecution::new_cold_with_capacity(model, layer_index, context_capacity)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output_head = PreparedOutputHead::new(model)?;
        context.prepare_decoder()?;

        let started = Instant::now();
        for (position, token) in tokens.iter().copied().enumerate() {
            submit_prepared_layers(model, &context, &mut layers, token, position as u32)?;
        }
        let (selected_token, _) = run_sampling_output_head(model, &context, &mut output_head)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

        if selected_token != 15342 {
            return Err(Error::invalid(format!(
                "2K prefill selected token {selected_token}, expected 15342"
            )));
        }
        for (index, (actual, expected)) in output_head.logits.iter().zip(&decode_logits).enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "2K decode-replay logit C0 mismatch at [{index}]: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }

        let mut batch_logits_mismatch_count = 0_u32;
        let mut batch_logits_max_abs_error = 0.0_f32;
        for (actual, batch) in output_head.logits.iter().zip(&batch_logits) {
            if actual.to_bits() != batch.to_bits() {
                batch_logits_mismatch_count += 1;
            }
            batch_logits_max_abs_error = batch_logits_max_abs_error.max((actual - batch).abs());
        }
        if batch_logits_mismatch_count == 0 {
            return Err(Error::invalid(
                "2K sequential decode replay unexpectedly matched batched prefill",
            ));
        }

        Ok(PrefillFrontierProbeReport {
            fixture_id: PREFILL_FRONTIER_2048_FIXTURE_ID,
            context_capacity,
            prompt_tokens: context_capacity,
            final_position: context_capacity - 1,
            raw_cache_capacity_rows: 128,
            ratio4_compressed_capacity_rows: context_capacity / 4 + 2,
            ratio128_compressed_capacity_rows: context_capacity / 128 + 2,
            selected_token,
            wall_ms,
            prefill_tps: f64::from(context_capacity) * 1000.0 / wall_ms,
            decode_logits_checksum: checksum_f32(&output_head.logits),
            batch_logits_mismatch_count,
            batch_logits_max_abs_error,
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
            indexer_q,
            indexer_weight,
            validate_expected,
            initial_state_mode,
            context_capacity,
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
            kv_norm_pre_rope,
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
            ffn_cur,
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
        let model_span_fields = |span: &Option<ModelSpan>| {
            span.as_ref()
                .map_or([0_u64; 2], |span| [span.absolute_offset, span.bytes])
        };
        let indexer_q_fields = model_span_fields(indexer_q);
        let indexer_weight_fields = model_span_fields(indexer_weight);

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
            indexer_q_offset: indexer_q_fields[0],
            indexer_q_bytes: indexer_q_fields[1],
            indexer_weight_offset: indexer_weight_fields[0],
            indexer_weight_bytes: indexer_weight_fields[1],
            compressor_prime_attn_norm: if compressor_prime.is_empty() {
                ptr::null()
            } else {
                compressor_prime.as_ptr()
            },
            compressed_kv_row: compressed_kv.as_mut_ptr(),
            compressed_indexer_row: compressed_indexer.as_mut_ptr(),
            kv_norm_pre_rope: kv_norm_pre_rope.as_mut_ptr(),
            ffn_mixes: ffn_mixes.as_mut_ptr(),
            ffn_split: ffn_split.as_mut_ptr(),
            ffn_cur: ffn_cur.as_mut_ptr(),
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
            initial_state_mode: *initial_state_mode,
            context_capacity: *context_capacity,
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
        let sparse_indexed_attention =
            layer_index >= 2 && layer_index % 2 == 0 && (position + 1) / 4 > 1024;
        let expected_wrapped_ranges = if layer_index >= 2 {
            if layer_index % 2 == 0 {
                if sparse_indexed_attention {
                    35
                } else {
                    33
                }
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
            (layer, 0, _) if layer >= 2 && sparse_indexed_attention => {
                let compressed_rows = (position + 1) / 4;
                53 + retained_sparse_topk_schedule(compressed_rows).1
            }
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
        if !*validate_expected {
            if !raw.wall_ms.is_finite()
                || raw.wall_ms <= 0.0
                || !raw.gpu_ms.is_finite()
                || raw.gpu_ms < 0.0
            {
                return Err(Error::invalid(
                    "unchecked retained layer collection returned invalid timing",
                ));
            }
            return Ok(Layer0Execution {
                report: Layer0ProbeReport {
                    fixture_id: RETAINED_SPARSE_BOUNDARY_FIXTURE_ID,
                    token,
                    dispatches,
                    command_buffers: 1,
                    selected_experts: selected.clone(),
                    wrapped_model_ranges: raw.wrapped_model_ranges,
                    pointer_matches: raw.pointer_matches,
                    wall_ms: raw.wall_ms,
                    gpu_ms: raw.gpu_ms,
                    attention_hc_checksum: checksum_f32(after_attention_hc),
                    ffn_norm_checksum: checksum_f32(ffn_norm),
                    router_weights_checksum: checksum_f32(router_weights),
                    routed_mid_checksum: checksum_f32(routed_mid),
                    routed_out_checksum: checksum_f32(routed_out),
                    shared_out_checksum: checksum_f32(shared_out),
                    final_hc_checksum: checksum_f32(after_ffn_hc),
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

    pub fn run_retained_sparse_multimerge_probe(
        model: &MappedModel,
    ) -> Result<RetainedSparseBoundaryProbeReport> {
        let _ = decode_f32_fixture(
            RETAINED_MULTIMERGE_INPUT_HC_BYTES,
            "retained sparse multimerge input HC",
        )?;
        let _ = exact_tensor(model, "blk.2.indexer.attn_q_b.weight", 1, &[1024, 8192])?;
        Err(Error::invalid(
            "the Metal retained sparse multimerge probe is available only on macOS",
        ))
    }

    pub fn run_retained_sparse_boundary_probe(
        model: &MappedModel,
    ) -> Result<RetainedSparseBoundaryProbeReport> {
        let _ = decode_f32_fixture(RETAINED_SPARSE_INPUT_HC_BYTES, "retained sparse input HC")?;
        let _ = exact_tensor(model, "blk.2.indexer.attn_q_b.weight", 1, &[1024, 8192])?;
        Err(Error::invalid(
            "the Metal retained sparse-boundary probe is available only on macOS",
        ))
    }

    pub fn run_sparse_indexed_attention_probe(
        model: &MappedModel,
    ) -> Result<SparseIndexedAttentionProbeReport> {
        let _ = decode_f32_fixture(SPARSE_INDEXER_SCORES_BYTES, "sparse indexer scores")?;
        let _ = exact_tensor(model, "blk.2.indexer.attn_q_b.weight", 1, &[1024, 8192])?;
        Err(Error::invalid(
            "the Metal sparse indexed-attention probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers012_attention_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012AttentionLoopProbeReport> {
        let _ = prefill_layer2_attention_fixture()?;
        let _ = prefill_layer3_ingress_final_tile_fixture()?;
        let _ = prefill_layer3_kv_state_final_tile_fixture()?;
        let _ = prefill_layer3_compressor_fixture()?;
        let _ = exact_tensor(model, "blk.2.attn_q_b.weight", 8, &[1024, 32768])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1/2 attention loop probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers012_compressor_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012CompressorLoopProbeReport> {
        let _ = prefill_layer2_kvnorm_fixture()?;
        let _ = prefill_layer2_kv_state_fixture()?;
        let _ = prefill_layer2_compressor_fixture()?;
        let _ = exact_tensor(model, "blk.2.attn_compressor_kv.weight", 1, &[4096, 1024])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1/2 compressor loop probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers012_kv_state_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012KvStateLoopProbeReport> {
        let _ = prefill_layer2_kvnorm_fixture()?;
        let _ = prefill_layer2_kv_state_fixture()?;
        let _ = exact_tensor(model, "blk.2.attn_kv.weight", 8, &[4096, 512])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1/2 KV-state loop probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers012_kvnorm_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers012KvnormLoopProbeReport> {
        let _ = prefill_frontier_2048_fixture()?;
        let _ = prefill_layer2_kvnorm_fixture()?;
        let _ = exact_tensor(model, "blk.2.hc_attn_fn.weight", 1, &[16384, 24])?;
        let _ = exact_tensor(model, "blk.2.attn_kv.weight", 8, &[4096, 512])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1/2 KVnorm loop probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers01_row_coverage_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01RowCoverageProbeReport> {
        let _ = prefill_layers01_previous_tile_fixture()?;
        let _ = exact_tensor(model, "blk.1.hc_attn_fn.weight", 1, &[16384, 24])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1 row-coverage probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers01_live_kv_chain_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01LiveKvChainProbeReport> {
        let _ = prefill_layers01_previous_tile_fixture()?;
        let _ = exact_tensor(model, "blk.1.hc_attn_fn.weight", 1, &[16384, 24])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1 live-KV chain probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers01_live_kv_loop_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01LiveKvLoopProbeReport> {
        let _ = prefill_frontier_2048_fixture()?;
        let _ = prefill_attention_read_fixture()?;
        let _ = prefill_layer1_complete_fixture()?;
        let _ = exact_tensor(model, "blk.1.hc_attn_fn.weight", 1, &[16384, 24])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1 live-KV loop probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers01_complete_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01CompleteBoundaryProbeReport> {
        let _ = prefill_layer1_ingress_fixture()?;
        let _ = prefill_layer1_complete_fixture()?;
        let _ = exact_tensor(model, "blk.1.hc_attn_fn.weight", 1, &[16384, 24])?;
        let _ = exact_tensor(model, "blk.1.ffn_down_shexp.weight", 8, &[2048, 4096])?;
        Err(Error::invalid(
            "the complete Metal prefill layers-0/1 boundary probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layers01_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayers01BoundaryProbeReport> {
        let _ = prefill_layer1_ingress_fixture()?;
        let _ = exact_tensor(model, "blk.1.hc_attn_fn.weight", 1, &[16384, 24])?;
        let _ = exact_tensor(model, "blk.1.attn_q_a.weight", 8, &[4096, 1024])?;
        Err(Error::invalid(
            "the Metal prefill layers-0/1 boundary probe is available only on macOS",
        ))
    }

    pub fn run_prefill_layer0_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillLayer0BoundaryProbeReport> {
        let _ = prefill_hc_ingress_fixture()?;
        let _ = prefill_qkv_boundary_fixture()?;
        let _ = prefill_kv_state_fixture()?;
        let _ = prefill_attention_read_fixture()?;
        let _ = prefill_attention_output_fixture()?;
        let _ = prefill_ffn_output_fixture()?;
        let _ = exact_tensor(model, "token_embd.weight", 1, &[4096, 129280])?;
        let _ = exact_tensor(model, "blk.0.hc_attn_fn.weight", 1, &[16384, 24])?;
        let _ = exact_tensor(model, "blk.0.attn_output_a.weight", 8, &[4096, 8192])?;
        let _ = exact_tensor(model, "blk.0.attn_output_b.weight", 8, &[8192, 4096])?;
        Err(Error::invalid(
            "the Metal prefill layer-0 boundary probe is available only on macOS",
        ))
    }

    pub fn run_prefill_qkv_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillQkvBoundaryProbeReport> {
        let _ = prefill_qkv_boundary_fixture()?;
        let _ = exact_tensor(model, "blk.0.attn_q_b.weight", 8, &[1024, 32768])?;
        Err(Error::invalid(
            "the Metal prefill Q/KV boundary probe is available only on macOS",
        ))
    }

    pub fn run_prefill_frontier_probe(model: &MappedModel) -> Result<PrefillFrontierProbeReport> {
        let _ = prefill_frontier_2048_fixture()?;
        let _ = exact_tensor(model, "output.weight", 8, &[4096, 129280])?;
        Err(Error::invalid(
            "the Metal 2K prefill frontier probe is available only on macOS",
        ))
    }

    pub fn run_cold_prefill_decoder_probe(
        model: &MappedModel,
    ) -> Result<ColdPrefillDecoderProbeReport> {
        let _ = cold_prefill_fixture()?;
        let _ = position127_decoder_fixture()?;
        let _ = exact_tensor(model, "output.weight", 8, &[4096, 129280])?;
        Err(Error::invalid(
            "the Metal cold-prefill decoder probe is available only on macOS",
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
        for position in 1..=4 {
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

    pub fn run_closed_loop_decoder_probe(
        model: &MappedModel,
    ) -> Result<ClosedLoopDecoderProbeReport> {
        for position in 1..=4 {
            let _ = output_head_expected(position)?;
        }
        let _ = exact_tensor(model, "output.weight", 8, &[4096, 129280])?;
        Err(Error::invalid(
            "the Metal closed-loop decoder probe is available only on macOS",
        ))
    }

    pub fn run_position127_decoder_probe(
        model: &MappedModel,
    ) -> Result<Position127DecoderProbeReport> {
        let _ = position127_decoder_fixture()?;
        let _ = ratio128_compressor_fixture(3)?;
        let _ = ratio128_compressor_fixture(5)?;
        let _ = exact_tensor(model, "output.weight", 8, &[4096, 129280])?;
        Err(Error::invalid(
            "the Metal position-127 decoder frontier probe is available only on macOS",
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

    pub fn run_prefill_q8_boundary_probe(
        model: &MappedModel,
    ) -> Result<PrefillQ8BoundaryProbeReport> {
        let tensor = model.tensor(PROJECTION_TENSOR)?;
        let (input, batch_output, decode_output) = prefill_q8_boundary_fixture()?;
        validate_prefill_q8_boundary_inputs(model, tensor, &input, &batch_output, &decode_output)?;
        Err(Error::invalid(
            "the Metal prefill Q8 boundary probe is available only on macOS",
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
    run_attention_setup_probe, run_closed_loop_decoder_probe, run_cold_prefill_decoder_probe,
    run_decoder_output_probe, run_f16_embedding_probe, run_ffn_router_probe, run_layer0_bench,
    run_layer0_probe, run_layers01234567_decode_probe, run_layers012345_decode_probe,
    run_layers0123_bench, run_layers0123_chained_probe, run_layers0123_decode_probe,
    run_layers0123_probe, run_layers012_chained_probe, run_layers012_probe, run_layers01_probe,
    run_layers0_to_42_decode_probe, run_moe_output_probe, run_position127_decoder_probe,
    run_prefill_frontier_probe, run_prefill_layer0_boundary_probe,
    run_prefill_layers012_attention_loop_probe, run_prefill_layers012_compressor_loop_probe,
    run_prefill_layers012_kv_state_loop_probe, run_prefill_layers012_kvnorm_loop_probe,
    run_prefill_layers01_boundary_probe, run_prefill_layers01_complete_boundary_probe,
    run_prefill_layers01_live_kv_chain_probe, run_prefill_layers01_live_kv_loop_probe,
    run_prefill_layers01_row_coverage_probe, run_prefill_q8_boundary_probe,
    run_prefill_qkv_boundary_probe, run_probe, run_q8_projection_probe,
    run_ratio128_compressor_replay_probe, run_retained_sparse_boundary_probe,
    run_retained_sparse_multimerge_probe, run_rope_kv_store_probe,
    run_sparse_indexed_attention_probe, LayerExecutor,
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

    fn prefill_q8_boundary_report() -> PrefillQ8BoundaryProbeReport {
        PrefillQ8BoundaryProbeReport {
            fixture_id: PREFILL_Q8_BOUNDARY_FIXTURE_ID,
            tensor_name: PROJECTION_TENSOR.to_owned(),
            rows: 128,
            input_elements_per_row: 4096,
            output_elements_per_row: 1024,
            no_copy_pointer_match: true,
            batch_threads_per_threadgroup: 128,
            batch_threadgroups_x: 4,
            batch_threadgroups_y: 16,
            batch_wall_ms: 3.0,
            batch_gpu_ms: 1.0,
            decode_wall_ms: 0.5,
            decode_gpu_ms: 0.25,
            input_checksum: 11,
            batch_output_checksum: 12,
            decode_output_checksum: 13,
            final_row_mismatches: 1024,
            final_row_max_abs_error: 3.6239624e-05,
        }
    }

    fn prefill_qkv_boundary_report() -> PrefillQkvBoundaryProbeReport {
        PrefillQkvBoundaryProbeReport {
            fixture_id: PREFILL_QKV_BOUNDARY_FIXTURE_ID,
            rows: 32,
            position_start: 2016,
            dispatches: 5,
            wrapped_model_ranges: 5,
            pointer_matches: 5,
            wall_ms: 2.0,
            gpu_ms: 1.0,
            checksums: [11, 12, 13, 14, 15, 16, 17],
        }
    }

    fn prefill_layer0_boundary_report() -> PrefillLayer0BoundaryProbeReport {
        PrefillLayer0BoundaryProbeReport {
            ingress_fixture_id: PREFILL_HC_INGRESS_FIXTURE_ID,
            qkv_fixture_id: PREFILL_QKV_BOUNDARY_FIXTURE_ID,
            kv_state_fixture_id: PREFILL_KV_STATE_FIXTURE_ID,
            attention_fixture_id: PREFILL_ATTENTION_READ_FIXTURE_ID,
            attention_output_fixture_id: PREFILL_ATTENTION_OUTPUT_FIXTURE_ID,
            ffn_output_fixture_id: PREFILL_FFN_OUTPUT_FIXTURE_ID,
            rows: 32,
            position_start: 2016,
            dispatches: 43,
            wrapped_model_ranges: 25,
            pointer_matches: 25,
            raw_cache_rows: 128,
            raw_cache_target_row: 96,
            raw_cache_guard_rows: 96,
            attention_kv_rows: 2048,
            attention_kv_prefix_rows: 2016,
            wall_ms: 3.0,
            gpu_ms: 2.0,
            checksums: [
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38,
            ],
        }
    }

    fn prefill_layers01_boundary_report() -> PrefillLayers01BoundaryProbeReport {
        let mut layer0 = prefill_layer0_boundary_report();
        layer0.dispatches = 47;
        layer0.wrapped_model_ranges = 30;
        layer0.pointer_matches = 30;
        PrefillLayers01BoundaryProbeReport {
            layer0,
            layer1_fixture_id: PREFILL_LAYER1_INGRESS_FIXTURE_ID,
            checksums: [39, 40, 41],
        }
    }

    fn prefill_layers01_complete_boundary_report() -> PrefillLayers01CompleteBoundaryProbeReport {
        let mut layers01 = prefill_layers01_boundary_report();
        layers01.layer0.dispatches = 84;
        layers01.layer0.wrapped_model_ranges = 49;
        layers01.layer0.pointer_matches = 49;
        PrefillLayers01CompleteBoundaryProbeReport {
            layers01,
            complete_fixture_id: PREFILL_LAYER1_COMPLETE_FIXTURE_ID,
            checksums: [42; 20],
        }
    }

    fn prefill_layers01_row_coverage_report() -> PrefillLayers01RowCoverageProbeReport {
        PrefillLayers01RowCoverageProbeReport {
            previous_fixture_id: PREFILL_LAYERS01_PREVIOUS_TILE_FIXTURE_ID,
            previous_position_start: 1984,
            previous_position_end: 2015,
            previous_dispatches: 84,
            previous_wrapped_model_ranges: 49,
            previous_pointer_matches: 49,
            previous_raw_cache_target_row: 64,
            previous_wall_ms: 4.0,
            previous_gpu_ms: 3.0,
            previous_checksums: [42; 6],
            final_tile: prefill_layers01_complete_boundary_report(),
        }
    }

    fn prefill_layers01_live_kv_chain_report() -> PrefillLayers01LiveKvChainProbeReport {
        PrefillLayers01LiveKvChainProbeReport {
            tiles: prefill_layers01_row_coverage_report(),
            retained_kv_rows_after_first_tile: 2016,
            retained_kv_rows_after_final_tile: 2048,
        }
    }

    fn prefill_layers01_live_kv_loop_report() -> PrefillLayers01LiveKvLoopProbeReport {
        let tiles = (0..64)
            .map(|index| {
                let mut tile = prefill_layer0_boundary_report();
                tile.position_start = index * 32;
                tile.dispatches = 84;
                tile.wrapped_model_ranges = 49;
                tile.pointer_matches = 49;
                tile.raw_cache_target_row = tile.position_start % 128;
                tile.raw_cache_guard_rows = tile.raw_cache_target_row;
                tile.attention_kv_prefix_rows = tile.position_start;
                tile
            })
            .collect();
        PrefillLayers01LiveKvLoopProbeReport {
            tiles,
            final_tile: prefill_layers01_complete_boundary_report(),
        }
    }

    fn prefill_layers012_kvnorm_loop_report() -> PrefillLayers012KvnormLoopProbeReport {
        let mut base = prefill_layers01_live_kv_loop_report();
        for tile in &mut base.tiles {
            tile.dispatches = 90;
            tile.wrapped_model_ranges = 57;
            tile.pointer_matches = 57;
        }
        base.final_tile.layers01.layer0.dispatches = 90;
        base.final_tile.layers01.layer0.wrapped_model_ranges = 57;
        base.final_tile.layers01.layer0.pointer_matches = 57;
        PrefillLayers012KvnormLoopProbeReport {
            tiles: base.tiles,
            layer2_fixture_id: PREFILL_LAYER2_KVNORM_FIXTURE_ID,
            layer2_kvnorm_checksums: (0..64).map(|index| index + 100).collect(),
            final_tile: base.final_tile,
        }
    }

    fn prefill_layers012_kv_state_loop_report() -> PrefillLayers012KvStateLoopProbeReport {
        let mut base = prefill_layers01_live_kv_loop_report();
        for tile in &mut base.tiles {
            tile.dispatches = 92;
            tile.wrapped_model_ranges = 57;
            tile.pointer_matches = 57;
        }
        base.final_tile.layers01.layer0.dispatches = 92;
        base.final_tile.layers01.layer0.wrapped_model_ranges = 57;
        base.final_tile.layers01.layer0.pointer_matches = 57;
        PrefillLayers012KvStateLoopProbeReport {
            tiles: base.tiles,
            layer2_kvnorm_fixture_id: PREFILL_LAYER2_KVNORM_FIXTURE_ID,
            layer2_kv_state_fixture_id: PREFILL_LAYER2_KV_STATE_FIXTURE_ID,
            layer2_checksums: (0..64)
                .map(|index| [index + 100, index + 200, index + 300])
                .collect(),
            final_tile: base.final_tile,
        }
    }

    fn prefill_layers012_compressor_loop_report() -> PrefillLayers012CompressorLoopProbeReport {
        let mut base = prefill_layers01_live_kv_loop_report();
        for (index, tile) in base.tiles.iter_mut().enumerate() {
            tile.dispatches = if index == 63 { 122 } else { 118 };
            tile.wrapped_model_ranges = 65;
            tile.pointer_matches = 65;
        }
        base.final_tile.layers01.layer0.dispatches = 122;
        base.final_tile.layers01.layer0.wrapped_model_ranges = 65;
        base.final_tile.layers01.layer0.pointer_matches = 65;
        PrefillLayers012CompressorLoopProbeReport {
            tiles: base.tiles,
            layer2_kvnorm_fixture_id: PREFILL_LAYER2_KVNORM_FIXTURE_ID,
            layer2_kv_state_fixture_id: PREFILL_LAYER2_KV_STATE_FIXTURE_ID,
            layer2_compressor_fixture_id: PREFILL_LAYER2_COMPRESSOR_FIXTURE_ID,
            layer2_checksums: (0..64)
                .map(|index| [index + 100, index + 200, index + 300])
                .collect(),
            layer2_compressor_checksums: (0..64)
                .map(|index| {
                    [
                        index + 1,
                        index + 2,
                        index + 3,
                        index + 4,
                        index + 5,
                        index + 6,
                    ]
                })
                .collect(),
            final_tile: base.final_tile,
        }
    }

    fn prefill_layers012_attention_loop_report() -> PrefillLayers012AttentionLoopProbeReport {
        PrefillLayers012AttentionLoopProbeReport {
            compressor: prefill_layers012_compressor_loop_report(),
            attention_fixture_id: PREFILL_LAYER2_ATTENTION_FIXTURE_ID,
            attention_hc_fixture_id: PREFILL_LAYER2_COMPLETE_FIXTURE_ID,
            layer3_ingress_fixture_id: PREFILL_LAYER3_INGRESS_FIXTURE_ID,
            layer3_kv_state_fixture_id: PREFILL_LAYER3_KV_STATE_FIXTURE_ID,
            layer3_compressor_fixture_id: PREFILL_LAYER3_COMPRESSOR_FIXTURE_ID,
            layer3_attention_fixture_id: PREFILL_LAYER3_ATTENTION_FIXTURE_ID,
            layer3_complete_fixture_id: PREFILL_LAYER3_COMPLETE_FIXTURE_ID,
            layer4_qkv_fixture_id: PREFILL_LAYER4_QKV_FIXTURE_ID,
            layer4_compressor_fixture_id: PREFILL_LAYER4_COMPRESSOR_FIXTURE_ID,
            layer4_attention_fixture_id: PREFILL_LAYER4_ATTENTION_FIXTURE_ID,
            layer4_complete_fixture_id: PREFILL_LAYER4_COMPLETE_FIXTURE_ID,
            layer5_qkv_fixture_id: PREFILL_LAYER5_QKV_FIXTURE_ID,
            layer5_compressor_fixture_id: PREFILL_LAYER5_COMPRESSOR_FIXTURE_ID,
            layer5_attention_fixture_id: PREFILL_LAYER5_ATTENTION_FIXTURE_ID,
            layer5_complete_fixture_id: PREFILL_LAYER5_COMPLETE_FIXTURE_ID,
            layer6_qkv_fixture_id: PREFILL_LAYER6_QKV_FIXTURE_ID,
            layer6_compressor_fixture_id: PREFILL_LAYER6_COMPRESSOR_FIXTURE_ID,
            layer6_attention_fixture_id: PREFILL_LAYER6_ATTENTION_FIXTURE_ID,
            layer6_complete_fixture_id: PREFILL_LAYER6_COMPLETE_FIXTURE_ID,
            layer7_qkv_fixture_id: PREFILL_LAYER7_QKV_FIXTURE_ID,
            layer7_compressor_fixture_id: PREFILL_LAYER7_COMPRESSOR_FIXTURE_ID,
            layer7_attention_fixture_id: PREFILL_LAYER7_ATTENTION_FIXTURE_ID,
            layer7_complete_fixture_id: PREFILL_LAYER7_COMPLETE_FIXTURE_ID,
            layer8_qkv_fixture_id: PREFILL_LAYER8_QKV_FIXTURE_ID,
            layer8_compressor_fixture_id: PREFILL_LAYER8_COMPRESSOR_FIXTURE_ID,
            layer8_attention_fixture_id: PREFILL_LAYER8_ATTENTION_FIXTURE_ID,
            layer8_complete_fixture_id: PREFILL_LAYER8_COMPLETE_FIXTURE_ID,
            rows: 2048,
            raw_kv_rows: 2048,
            compressed_kv_rows: 512,
            layer3_compressed_kv_rows: 16,
            dispatches: 383,
            wrapped_model_ranges: 196,
            pointer_matches: 196,
            wall_ms: 196.0,
            gpu_ms: 169.0,
            output_checksum: checksum_f32(&prefill_layer2_attention_fixture().unwrap()),
            after_attention_hc_checksum: PREFILL_LAYER2_HC_ATTN_POST_FULL_CHECKSUM,
            after_ffn_hc_checksum: PREFILL_LAYER2_HC_FFN_POST_FULL_CHECKSUM,
            layer3_hc_attn_pre_checksum: PREFILL_LAYER3_HC_ATTN_PRE_FULL_CHECKSUM,
            layer3_attn_norm_checksum: PREFILL_LAYER3_ATTN_NORM_FULL_CHECKSUM,
            layer3_q_lora_checksum: PREFILL_LAYER3_Q_LORA_FULL_CHECKSUM,
            layer3_q_lora_norm_checksum: PREFILL_LAYER3_Q_LORA_NORM_FULL_CHECKSUM,
            layer3_kv_raw_checksum: PREFILL_LAYER3_KV_RAW_FULL_CHECKSUM,
            layer3_kv_norm_checksum: PREFILL_LAYER3_KV_NORM_FULL_CHECKSUM,
            layer3_q_raw_final_tile_checksum: checksum_f32(
                &prefill_layer3_kv_state_final_tile_fixture().unwrap()[3],
            ),
            layer3_q_cur_final_tile_checksum: checksum_f32(
                &prefill_layer3_kv_state_final_tile_fixture().unwrap()[4],
            ),
            layer3_kv_rope_checksum: PREFILL_LAYER3_KV_ROPE_FULL_CHECKSUM,
            layer3_kv_cur_checksum: PREFILL_LAYER3_KV_CUR_FULL_CHECKSUM,
            layer3_attn_compressed_checksum: PREFILL_LAYER3_ATTN_COMPRESSED_CHECKSUM,
            layer3_attn_state_kv_checksum: PREFILL_LAYER3_ATTN_STATE_KV_CHECKSUM,
            layer3_attn_state_score_checksum: PREFILL_LAYER3_ATTN_STATE_SCORE_CHECKSUM,
            layer3_attention_output_checksum: PREFILL_LAYER3_ATTENTION_OUTPUT_CHECKSUM,
            layer3_after_attention_hc_checksum: PREFILL_LAYER3_HC_ATTN_POST_FULL_CHECKSUM,
            layer3_after_ffn_hc_checksum: PREFILL_LAYER3_HC_FFN_POST_FULL_CHECKSUM,
            layer4_qkv_checksums: prefill_layer4_qkv_final_tile_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer4_compressor_checksums: prefill_layer4_compressor_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer4_attention_output_checksum: PREFILL_LAYER4_ATTENTION_OUTPUT_CHECKSUM,
            layer4_after_attention_hc_checksum: PREFILL_LAYER4_HC_ATTN_POST_FULL_CHECKSUM,
            layer4_after_ffn_hc_checksum: PREFILL_LAYER4_HC_FFN_POST_FULL_CHECKSUM,
            layer5_qkv_checksums: prefill_layer5_qkv_final_tile_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer5_compressor_checksums: {
                let fixture = prefill_layer5_compressor_fixture().unwrap();
                [
                    checksum_f32(&fixture.0),
                    checksum_f32(&fixture.1),
                    checksum_i32(&fixture.2),
                ]
            },
            layer5_attention_output_checksum: PREFILL_LAYER5_ATTENTION_OUTPUT_CHECKSUM,
            layer5_after_attention_hc_checksum: PREFILL_LAYER5_HC_ATTN_POST_FULL_CHECKSUM,
            layer5_after_ffn_hc_checksum: PREFILL_LAYER5_HC_FFN_POST_FULL_CHECKSUM,
            layer6_qkv_checksums: prefill_layer6_qkv_final_tile_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer6_compressor_checksums: prefill_layer6_compressor_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer6_attention_output_checksum: PREFILL_LAYER6_ATTENTION_OUTPUT_CHECKSUM,
            layer6_after_attention_hc_checksum: PREFILL_LAYER6_HC_ATTN_POST_FULL_CHECKSUM,
            layer6_after_ffn_hc_checksum: PREFILL_LAYER6_HC_FFN_POST_FULL_CHECKSUM,
            layer7_qkv_checksums: prefill_layer7_qkv_final_tile_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer7_compressor_checksums: {
                let fixture = prefill_layer7_compressor_fixture().unwrap();
                [
                    checksum_f32(&fixture.0),
                    checksum_f32(&fixture.1),
                    checksum_i32(&fixture.2),
                ]
            },
            layer7_attention_output_checksum: PREFILL_LAYER7_ATTENTION_OUTPUT_CHECKSUM,
            layer7_after_attention_hc_checksum: PREFILL_LAYER7_HC_ATTN_POST_FULL_CHECKSUM,
            layer7_after_ffn_hc_checksum: PREFILL_LAYER7_HC_FFN_POST_FULL_CHECKSUM,
            layer8_qkv_checksums: prefill_layer8_qkv_final_tile_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer8_compressor_checksums: prefill_layer8_compressor_fixture()
                .unwrap()
                .each_ref()
                .map(|tensor| checksum_f32(tensor)),
            layer8_attention_output_checksum: PREFILL_LAYER8_ATTENTION_OUTPUT_CHECKSUM,
            layer8_after_attention_hc_checksum: PREFILL_LAYER8_HC_ATTN_POST_FULL_CHECKSUM,
            layer8_after_ffn_hc_checksum: PREFILL_LAYER8_HC_FFN_POST_FULL_CHECKSUM,
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
            cache_capacity_rows: 128,
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
        let mut transformer = layers0_to_42_decode_report();
        let mut position4 = transformer.steps[2].clone();
        position4.position = 4;
        position4.token = 262;
        position4.cache_rows = 5;
        position4.wall_ms = 8.0;
        position4.gpu_ms = 7.0;
        for (layer_index, layer) in position4.layers.iter_mut().enumerate() {
            layer.fixture_id = POS4_BYTES[layer_index].fixture_id;
            layer.token = 262;
        }
        transformer.steps.push(position4);
        let selected = [361_u32, 1915, 262, 1554];
        let fixtures = [
            "dwarfstar-oracle-v1-output-head-pos1",
            "dwarfstar-oracle-v1-output-head-pos2",
            "dwarfstar-oracle-v1-output-head-pos3",
            "dwarfstar-oracle-v1-output-head-pos4",
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
            cache_capacity_rows: 128,
            logits_elements: 129280,
            closed_loop_sampling: false,
            externally_supplied_decode_inputs: true,
        }
    }

    fn closed_loop_decoder_report() -> ClosedLoopDecoderProbeReport {
        let mut correctness = decoder_output_report();
        correctness.closed_loop_sampling = true;
        correctness.externally_supplied_decode_inputs = false;
        ClosedLoopDecoderProbeReport {
            correctness,
            timed_steps: [
                (1_u32, 201_u32, 361_u32, 12.0_f64),
                (2, 361, 1915, 10.0),
                (3, 1915, 262, 10.0),
                (4, 262, 1554, 10.0),
            ]
            .into_iter()
            .map(
                |(position, input_token, selected_token, wall_ms)| TimedDecoderStepReport {
                    position,
                    input_token,
                    selected_token,
                    wall_ms,
                    output_head_gpu_ms: 1.0,
                },
            )
            .collect(),
            pipeline_prepare_ms: 800.0,
            generation_wall_ms: 42.0,
            generation_tps: 4000.0 / 42.0,
            first_token_ms: 12.0,
            steady_wall_ms: 30.0,
            steady_tps: 100.0,
        }
    }

    fn position127_decoder_report() -> Position127DecoderProbeReport {
        let (tokens, logits) = position127_decoder_fixture().unwrap();
        let layer3 = decode_f32_fixture(
            LAYER3_POS127_COMPRESSED_KV_BYTES,
            "layer-3 position-127 compressed KV row",
        )
        .unwrap();
        let layer5 = decode_f32_fixture(
            LAYER5_POS127_COMPRESSED_KV_BYTES,
            "layer-5 position-127 compressed KV row",
        )
        .unwrap();
        Position127DecoderProbeReport {
            fixture_id: POSITION127_DECODER_FIXTURE_ID,
            committed_tokens: tokens,
            evaluated_positions: 127,
            final_position: 127,
            cache_capacity_rows: 128,
            compressed_cache_capacity_rows: 32,
            command_buffers_per_position: 44,
            host_waits_per_position: 2,
            wall_ms: 7000.0,
            eval_tps: 127000.0 / 7000.0,
            final_logits_checksum: checksum_f32(&logits),
            ratio128_layer3_checksum: checksum_f32(&layer3),
            ratio128_layer5_checksum: checksum_f32(&layer5),
        }
    }

    fn cold_prefill_decoder_report() -> ColdPrefillDecoderProbeReport {
        let prefill_logits = cold_prefill_fixture().unwrap();
        let (tokens, final_logits) = position127_decoder_fixture().unwrap();
        let layer3 = decode_f32_fixture(
            LAYER3_POS127_COMPRESSED_KV_BYTES,
            "layer-3 position-127 compressed KV row",
        )
        .unwrap();
        let layer5 = decode_f32_fixture(
            LAYER5_POS127_COMPRESSED_KV_BYTES,
            "layer-5 position-127 compressed KV row",
        )
        .unwrap();
        ColdPrefillDecoderProbeReport {
            fixture_id: COLD_PREFILL_FIXTURE_ID,
            prompt_token: 36662,
            committed_tokens: tokens,
            prefill_wall_ms: 50.0,
            prefill_logits_checksum: checksum_f32(&prefill_logits),
            decode_wall_ms: 7000.0,
            decode_tps: 127000.0 / 7000.0,
            final_logits_checksum: checksum_f32(&final_logits),
            ratio128_layer3_checksum: checksum_f32(&layer3),
            ratio128_layer5_checksum: checksum_f32(&layer5),
        }
    }

    fn prefill_frontier_report() -> PrefillFrontierProbeReport {
        let (_, batch_logits, decode_logits) = prefill_frontier_2048_fixture().unwrap();
        let batch_logits_max_abs_error = batch_logits
            .iter()
            .zip(&decode_logits)
            .map(|(batch, decode)| (batch - decode).abs())
            .fold(0.0_f32, f32::max);
        PrefillFrontierProbeReport {
            fixture_id: PREFILL_FRONTIER_2048_FIXTURE_ID,
            context_capacity: 2048,
            prompt_tokens: 2048,
            final_position: 2047,
            raw_cache_capacity_rows: 128,
            ratio4_compressed_capacity_rows: 514,
            ratio128_compressed_capacity_rows: 18,
            selected_token: 15342,
            wall_ms: 12000.0,
            prefill_tps: 2_048_000.0 / 12000.0,
            decode_logits_checksum: checksum_f32(&decode_logits),
            batch_logits_mismatch_count: 129280,
            batch_logits_max_abs_error,
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
        for (position, selected) in [(1, 361), (2, 1915), (3, 262), (4, 1554)] {
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
        assert!(text.contains("\"selected_token\": 1554"));
        assert!(text.contains("\"closed_loop_sampling\": false"));
        assert!(text.contains("\"full_logits_c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_closed_loop_decoder_probe_json() {
        let mut output = Vec::new();
        write_closed_loop_decoder_probe_json(&mut output, &closed_loop_decoder_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{CLOSED_LOOP_DECODER_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"generated_tokens\": [361, 1915, 262, 1554]"));
        assert!(text.contains("\"correctness_readback_in_interval\": false"));
        assert!(text.contains("\"gen_steady_tps\": 100.000000"));
        assert!(text.contains("\"paired_protocol_eligible\": false"));
    }

    #[test]
    fn writes_stable_position127_decoder_probe_json() {
        let mut output = Vec::new();
        write_position127_decoder_probe_json(&mut output, &position127_decoder_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{POSITION127_DECODER_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"final_position\": 127"));
        assert!(text.contains("\"committed_tokens\": [201,361,1915,262,1554"));
        assert!(text.contains("\"final_logits_c0_bitwise_match\": true"));
        assert!(text.contains("\"integrated_ratio128_rows_c0_bitwise_match\": true"));
    }

    #[test]
    fn writes_stable_cold_prefill_decoder_probe_json() {
        let mut output = Vec::new();
        write_cold_prefill_decoder_probe_json(&mut output, &cold_prefill_decoder_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{COLD_PREFILL_DECODER_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"prompt_token\": 36662"));
        assert!(text.contains("\"full_logits_c0_bitwise_match\": true"));
        assert!(text.contains("\"captured_initial_state_used\": false"));
    }

    #[test]
    fn writes_stable_prefill_frontier_probe_json() {
        let mut output = Vec::new();
        write_prefill_frontier_probe_json(&mut output, &prefill_frontier_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{PREFILL_FRONTIER_PROBE_SCHEMA}\"")));
        assert!(text.contains("\"context_capacity\": 2048"));
        assert!(text.contains("\"raw_ring_rows_per_layer\": 128"));
        assert!(text.contains("\"decode_replay_logits_c0_bitwise_match\": true"));
        assert!(text.contains("\"batched_prefill_logits_c0_bitwise_match\": false"));
        assert!(text.contains("\"paired_protocol_eligible\": false"));
    }

    #[test]
    fn complete_later_layer_registry_has_target_shapes() {
        assert_eq!(LATER_POS1_BYTES.len(), 39);
        assert_eq!(LATER_POS2_BYTES.len(), 39);
        assert_eq!(LATER_POS3_BYTES.len(), 39);
        assert_eq!(POS4_BYTES.len(), 43);
        assert_eq!(LATER_POS0_COMPRESSOR_PRIME_BYTES.len(), 39);
        assert_eq!(LATER_CACHE_ROW0_BYTES.len(), 39);
        assert_eq!(LATER_POS3_COMPRESSED_KV_BYTES.len(), 20);
        for layer_index in 4..=42 {
            assert_eq!(compressor_prime_bytes(layer_index).unwrap().len(), 4096 * 4);
            for position in 1..=4 {
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
    fn writes_stable_prefill_q8_boundary_probe_json() {
        let mut output = Vec::new();
        write_prefill_q8_boundary_probe_json(&mut output, &prefill_q8_boundary_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_Q8_BOUNDARY_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"batch_kernel\": \"kernel_mul_mm_q8_0_f32\""));
        assert!(text.contains("\"threadgroups\": [4, 16, 1]"));
        assert!(text.contains("\"final_row_mismatches\": 1024"));
        assert!(text.contains("\"batch_c0_bitwise_match\": true"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_qkv_boundary_probe_json() {
        let mut output = Vec::new();
        write_prefill_qkv_boundary_probe_json(&mut output, &prefill_qkv_boundary_report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_QKV_BOUNDARY_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"position_start\": 2016"));
        assert!(text.contains("\"position_end\": 2047"));
        assert!(text.contains("\"dispatches\": 5"));
        assert!(text.contains("\"native_batch_schedule\": true"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layer0_boundary_probe_json() {
        let mut output = Vec::new();
        write_prefill_layer0_boundary_probe_json(&mut output, &prefill_layer0_boundary_report())
            .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYER0_BOUNDARY_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"input_boundary\": \"token_ids\""));
        assert!(text.contains("\"dispatches\": 43"));
        assert!(text.contains("\"attention_output_low_kernel\": \"kernel_mul_mm_id_q8_0_f32\""));
        assert!(text.contains("\"attention_hc_post_kernel\": \"kernel_dsv4_hc_expand4\""));
        assert!(
            text.contains("\"routed_pair_kernel\": \"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16\"")
        );
        assert!(text.contains("\"output_boundary\": \"layer0_hc_ffn_post\""));
        assert!(text.contains("\"target_row_start\": 96"));
        assert!(text.contains("\"kv_rows\": 2048"));
        assert!(text.contains("\"rectangular_attention_read\": true"));
        assert!(text.contains("\"guarded_cache_mutation\": true"));
        assert!(text.contains("\"continuous_command_buffer\": true"));
        assert!(text.contains("\"full_layer0_final_tile_claim\": true"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers01_boundary_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers01_boundary_probe_json(
            &mut output,
            &prefill_layers01_boundary_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS01_BOUNDARY_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"output_boundary\": \"layer1_q_lora\""));
        assert!(text.contains("\"dispatches\": 47"));
        assert!(text.contains("\"wrapped_model_ranges\": 30"));
        assert!(text.contains("\"direct_hc_handoff\": true"));
        assert!(text.contains("\"full_layer1_claim\": false"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers01_complete_boundary_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers01_complete_boundary_probe_json(
            &mut output,
            &prefill_layers01_complete_boundary_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS01_COMPLETE_BOUNDARY_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"output_boundary\": \"layer1_hc_ffn_post\""));
        assert!(text.contains("\"dispatches\": 84"));
        assert!(text.contains("\"wrapped_model_ranges\": 49"));
        assert!(text.contains("\"produced_fp32_values\": 12878208"));
        assert!(text.contains("\"full_layer1_claim\": true"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers01_row_coverage_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers01_row_coverage_probe_json(
            &mut output,
            &prefill_layers01_row_coverage_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS01_ROW_COVERAGE_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"position_start\": 1984"));
        assert!(text.contains("\"position_end\": 2047"));
        assert!(text.contains("\"rows\": 64"));
        assert!(text.contains("\"raw_cache_target_row\": 64"));
        assert!(text.contains("\"chained_live_kv_between_tiles\": false"));
        assert!(text.contains("\"arbitrary_tile_position_claim\": true"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers01_live_kv_chain_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers01_live_kv_chain_probe_json(
            &mut output,
            &prefill_layers01_live_kv_chain_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS01_LIVE_KV_CHAIN_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"position_start\": 1984"));
        assert!(text.contains("\"position_end\": 2047"));
        assert!(text.contains("\"retained_kv_rows\": 2016"));
        assert!(text.contains("\"retained_kv_rows\": 2048"));
        assert!(text.contains("\"persistent_metal_context\": true"));
        assert!(text.contains("\"captured_prefix_final_tile\": false"));
        assert!(text.contains("\"chained_live_kv_between_tiles\": true"));
        assert!(text.contains("\"single_command_buffer\": false"));
        assert!(text.contains("\"full_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers01_live_kv_loop_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers01_live_kv_loop_probe_json(
            &mut output,
            &prefill_layers01_live_kv_loop_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS01_LIVE_KV_LOOP_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"position_start\": 0"));
        assert!(text.contains("\"position_end\": 2047"));
        assert!(text.contains("\"tiles\": 64"));
        assert!(text.contains("\"captured_kv_seed_rows\": 0"));
        assert!(text.contains("\"all_layer01_kv_c0_bitwise_match\": true"));
        assert!(text.contains("\"all_tile_full_outputs_c0_bitwise_match\": false"));
        assert!(text.contains("\"complete_model_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers012_kvnorm_loop_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers012_kvnorm_loop_probe_json(
            &mut output,
            &prefill_layers012_kvnorm_loop_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS012_KVNORM_LOOP_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"output_boundary\": \"layer2_KVnorm\""));
        assert!(text.contains("\"dispatches_per_tile\": 90"));
        assert!(text.contains("\"all_layer1_outputs_downstream_validated\": true"));
        assert!(text.contains("\"complete_layer2_prefill_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers012_kv_state_loop_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers012_kv_state_loop_probe_json(
            &mut output,
            &prefill_layers012_kv_state_loop_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS012_KV_STATE_LOOP_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"output_boundary\": \"layer2_KVcur\""));
        assert!(text.contains("\"dispatches_per_tile\": 92"));
        assert!(text.contains("\"complete_layer2_raw_kv_state_claim\": true"));
        assert!(text.contains("\"complete_layer2_compressed_kv_state_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers012_compressor_loop_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers012_compressor_loop_probe_json(
            &mut output,
            &prefill_layers012_compressor_loop_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS012_COMPRESSOR_LOOP_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"dispatches_per_regular_tile\": 118"));
        assert!(text.contains("\"dispatches_final_tile\": 122"));
        assert!(text.contains("\"complete_layer2_paired_compressor_claim\": true"));
        assert!(text.contains("\"complete_layer2_attention_claim\": false"));
    }

    #[test]
    fn writes_stable_prefill_layers012_attention_loop_probe_json() {
        let mut output = Vec::new();
        write_prefill_layers012_attention_loop_probe_json(
            &mut output,
            &prefill_layers012_attention_loop_report(),
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!(
            "\"schema\": \"{PREFILL_LAYERS012_ATTENTION_LOOP_PROBE_SCHEMA}\""
        )));
        assert!(text.contains("\"compressed_kv_rows\": 512"));
        assert!(text.contains("\"sparse_indexer_topk\": false"));
        assert!(text.contains("\"terminal_dispatches\": 383"));
        assert!(text.contains("\"layer3_compressor_ratio\": 128"));
        assert!(text.contains("\"layer3_compressed_rows\": 16"));
        assert!(text.contains("\"layer7_compressor_ratio\": 128"));
        assert!(text.contains("\"layer7_compressed_rows\": 16"));
        assert!(text.contains("\"layer2_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer2_dense_mixed_attention_claim\": true"));
        assert!(text.contains("\"layer2_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer2_ffn_claim\": true"));
        assert!(text.contains("\"complete_layer2_prefill_claim\": true"));
        assert!(text.contains("\"layer3_attention_ingress_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer3_qkv_state_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer3_qkv_state_claim\": true"));
        assert!(text.contains("\"layer3_ratio128_compressor_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer3_ratio128_compressor_claim\": true"));
        assert!(text.contains("\"layer3_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer3_attention_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer3_dense_mixed_attention_claim\": true"));
        assert!(text.contains("\"complete_layer3_attention_hc_post_claim\": true"));
        assert!(text.contains("\"layer3_ffn_biased_topk_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer3_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer3_ffn_claim\": true"));
        assert!(text.contains("\"complete_layer3_prefill_claim\": true"));
        assert!(text.contains("\"output_boundary\": \"layer8_ffn_hc_post\""));
        assert!(text.contains("\"complete_layers\": [0, 1, 2, 3, 4, 5, 6, 7, 8]"));
        assert!(text.contains("\"downstream_layer\": null"));
        assert!(text.contains("\"layer4_qkv_state_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer4_qkv_state_claim\": true"));
        assert!(text.contains("\"layer4_paired_compressors_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer4_paired_compressor_claim\": true"));
        assert!(text.contains("\"layer4_dense_attention_dispatches\": 9"));
        assert!(text.contains("\"layer4_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer4_attention_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer4_dense_mixed_attention_claim\": true"));
        assert!(text.contains("\"complete_layer4_attention_hc_post_claim\": true"));
        assert!(text.contains("\"layer4_ffn_dispatches\": 21"));
        assert!(text.contains("\"layer4_ffn_router\": \"biased top-6 batch\""));
        assert!(text.contains("\"layer4_ffn_biased_topk_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer4_ffn_outputs_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer4_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer4_ffn_claim\": true"));
        assert!(text.contains("\"complete_layer4_prefill_claim\": true"));
        assert!(text.contains("\"layer5_qkv_dispatches\": 10"));
        assert!(text.contains("\"layer5_qkv_state_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer5_qkv_state_claim\": true"));
        assert!(text.contains("\"layer5_ratio128_compressor_dispatches\": 7"));
        assert!(text.contains("\"layer5_ratio128_compressor_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer5_ratio128_compressor_claim\": true"));
        assert!(text.contains("\"layer5_dense_attention_dispatches\": 9"));
        assert!(text.contains("\"layer5_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer5_attention_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer5_dense_mixed_attention_claim\": true"));
        assert!(text.contains("\"complete_layer5_attention_hc_post_claim\": true"));
        assert!(text.contains("\"layer5_ffn_dispatches\": 21"));
        assert!(text.contains("\"layer5_ffn_router\": \"biased top-6 batch\""));
        assert!(text.contains("\"layer5_ffn_biased_topk_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer5_ffn_outputs_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer5_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer5_ffn_claim\": true"));
        assert!(text.contains("\"complete_layer5_prefill_claim\": true"));
        assert!(text.contains("\"layer6_qkv_dispatches\": 10"));
        assert!(text.contains("\"layer6_qkv_state_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer6_qkv_state_claim\": true"));
        assert!(text.contains("\"layer6_paired_compressor_dispatches\": 30"));
        assert!(text.contains("\"layer6_paired_compressors_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer6_paired_compressor_claim\": true"));
        assert!(text.contains("\"layer6_dense_attention_dispatches\": 9"));
        assert!(text.contains("\"layer6_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer6_attention_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer6_dense_mixed_attention_claim\": true"));
        assert!(text.contains("\"complete_layer6_attention_hc_post_claim\": true"));
        assert!(text.contains("\"layer6_ffn_dispatches\": 21"));
        assert!(text.contains("\"layer6_ffn_router\": \"biased top-6 batch\""));
        assert!(text.contains("\"layer6_ffn_biased_topk_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer6_ffn_outputs_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer6_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer6_ffn_claim\": true"));
        assert!(text.contains("\"complete_layer6_prefill_claim\": true"));
        assert!(text.contains("\"layer7_qkv_dispatches\": 10"));
        assert!(text.contains("\"layer7_ratio128_compressor_dispatches\": 7"));
        assert!(text.contains("\"layer7_dense_attention_dispatches\": 9"));
        assert!(text.contains("\"layer7_ffn_dispatches\": 21"));
        assert!(text.contains("\"layer7_qkv_state_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer7_ratio128_compressor_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer7_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer7_attention_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer7_ffn_biased_topk_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer7_ffn_outputs_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer7_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer7_qkv_state_claim\": true"));
        assert!(text.contains("\"complete_layer7_ratio128_compressor_claim\": true"));
        assert!(text.contains("\"complete_layer7_dense_mixed_attention_claim\": true"));
        assert!(text.contains("\"complete_layer7_attention_hc_post_claim\": true"));
        assert!(text.contains("\"complete_layer7_ffn_claim\": true"));
        assert!(text.contains("\"complete_layer7_prefill_claim\": true"));
        assert!(text.contains("\"layer8_compressor_ratio\": 4"));
        assert!(text.contains("\"layer8_compressed_rows\": 512"));
        assert!(text.contains("\"layer8_sparse_indexer_topk\": false"));
        assert!(text.contains("\"layer8_qkv_dispatches\": 10"));
        assert!(text.contains("\"layer8_paired_compressor_dispatches\": 30"));
        assert!(text.contains("\"layer8_dense_attention_dispatches\": 9"));
        assert!(text.contains("\"layer8_ffn_dispatches\": 21"));
        assert!(text.contains("\"layer8_qkv_state_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer8_paired_compressors_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer8_attention_output_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer8_attention_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"layer8_ffn_hc_post_c0_bitwise_match\": true"));
        assert!(text.contains("\"complete_layer8_prefill_claim\": true"));
        assert!(text.contains("\"sparse_ratio4_decode_claim\": false"));
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
    fn prefill_kv_state_fixture_has_target_shapes() {
        let [kv_rope, kv_current, raw_cache] = prefill_kv_state_fixture().unwrap();
        assert_eq!(kv_rope.len(), 32 * 512);
        assert_eq!(kv_current.len(), 32 * 512);
        assert_eq!(raw_cache.len(), 32 * 512);
        assert!(kv_rope
            .iter()
            .chain(&kv_current)
            .chain(&raw_cache)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_attention_read_fixture_has_target_shapes() {
        let [kv_prefix, attention_output, attention_back] =
            prefill_attention_read_fixture().unwrap();
        assert_eq!(kv_prefix.len(), 2016 * 512);
        assert_eq!(attention_output.len(), 32 * 64 * 512);
        assert_eq!(attention_back.len(), 32 * 64 * 512);
        assert!(kv_prefix
            .iter()
            .chain(&attention_output)
            .chain(&attention_back)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_attention_output_fixture_has_target_shapes() {
        let [low, out, hc_post] = prefill_attention_output_fixture().unwrap();
        assert_eq!(low.len(), 32 * 8192);
        assert_eq!(out.len(), 32 * 4096);
        assert_eq!(hc_post.len(), 32 * 16384);
        assert!(low
            .iter()
            .chain(&out)
            .chain(&hc_post)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_ffn_output_fixture_has_target_shapes() {
        let (tensors, selected) = prefill_ffn_output_fixture().unwrap();
        assert_eq!(tensors[0].len(), 32 * 4096);
        assert_eq!(tensors[1].len(), 32 * 4096);
        assert_eq!(tensors[2].len(), 32 * 256);
        assert_eq!(tensors[3].len(), 32 * 256);
        assert_eq!(selected.len(), 32 * 6);
        assert_eq!(tensors[4].len(), 32 * 6);
        assert_eq!(tensors[5].len(), 32 * 6 * 2048);
        assert_eq!(tensors[6].len(), 32 * 4096);
        assert_eq!(tensors[7].len(), 32 * 4096);
        assert_eq!(tensors[8].len(), 32 * 16384);
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer1_ingress_fixture_has_target_shapes() {
        let [hc, norm, q_lora] = prefill_layer1_ingress_fixture().unwrap();
        assert_eq!(hc.len(), 32 * 4096);
        assert_eq!(norm.len(), 32 * 4096);
        assert_eq!(q_lora.len(), 32 * 1024);
        assert!(hc
            .iter()
            .chain(&norm)
            .chain(&q_lora)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer1_complete_fixture_has_target_shapes() {
        let (tensors, selected) = prefill_layer1_complete_fixture().unwrap();
        assert_eq!(tensors[0].len(), 32 * 1024);
        assert_eq!(tensors[2].len(), 32 * 32768);
        assert_eq!(tensors[5].len(), 2016 * 512);
        assert_eq!(tensors[6].len(), 32 * 64 * 512);
        assert_eq!(tensors[10].len(), 32 * 16384);
        assert_eq!(tensors[16].len(), 32 * 6 * 2048);
        assert_eq!(tensors[19].len(), 32 * 16384);
        assert_eq!(selected.len(), 32 * 6);
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer2_kvnorm_fixture_has_target_shape() {
        let tensor = prefill_layer2_kvnorm_fixture().unwrap();
        assert_eq!(tensor.len(), 2048 * 512);
        assert!(tensor.iter().all(|value| value.is_finite()));
        assert_ne!(checksum_f32(&tensor), 0);
    }

    #[test]
    fn prefill_layer2_kv_state_fixture_has_target_shapes() {
        let tensors = prefill_layer2_kv_state_fixture().unwrap();
        for tensor in tensors {
            assert_eq!(tensor.len(), 2048 * 512);
            assert!(tensor.iter().all(|value| value.is_finite()));
            assert_ne!(checksum_f32(&tensor), 0);
        }
    }

    #[test]
    fn prefill_layer2_compressor_fixture_has_target_shapes() {
        let tensors = prefill_layer2_compressor_fixture().unwrap();
        assert_eq!(tensors[0].len(), 512 * 512);
        assert_eq!(tensors[1].len(), 8 * 1024);
        assert_eq!(tensors[2].len(), 8 * 1024);
        assert_eq!(tensors[3].len(), 512 * 128);
        assert_eq!(tensors[4].len(), 8 * 256);
        assert_eq!(tensors[5].len(), 8 * 256);
        assert!(tensors.iter().flatten().all(|value| !value.is_nan()));
        assert!(tensors[2].iter().any(|value| *value == f32::NEG_INFINITY));
        assert!(tensors[5].iter().any(|value| *value == f32::NEG_INFINITY));
    }

    #[test]
    fn prefill_layer2_attention_fixture_has_target_shape() {
        let tensor = prefill_layer2_attention_fixture().unwrap();
        assert_eq!(tensor.len(), 2048 * 4096);
        assert!(tensor.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer2_complete_fixture_has_target_shapes() {
        assert_eq!(
            prefill_layer2_hc_attn_post_final_tile_fixture()
                .unwrap()
                .len(),
            32 * 4 * 4096
        );
        assert_eq!(
            prefill_layer2_hc_ffn_post_final_tile_fixture()
                .unwrap()
                .len(),
            32 * 4 * 4096
        );
        let (cur, norm) = prefill_layer2_ffn_ingress_final_tile_fixture().unwrap();
        assert_eq!(cur.len(), 32 * 4096);
        assert_eq!(norm.len(), 32 * 4096);
        let (selected, [weights, routed, shared]) =
            prefill_layer2_ffn_output_final_tile_fixture().unwrap();
        assert_eq!(selected.len(), 32 * 6);
        assert_eq!(weights.len(), 32 * 6);
        assert_eq!(routed.len(), 32 * 4096);
        assert_eq!(shared.len(), 32 * 4096);
    }

    #[test]
    fn prefill_layer3_ingress_fixture_has_target_shapes() {
        let [hc_attn_pre, attn_norm, q_lora] = prefill_layer3_ingress_final_tile_fixture().unwrap();
        assert_eq!(hc_attn_pre.len(), 32 * 4096);
        assert_eq!(attn_norm.len(), 32 * 4096);
        assert_eq!(q_lora.len(), 32 * 1024);
        assert!(hc_attn_pre.iter().all(|value| value.is_finite()));
        assert!(attn_norm.iter().all(|value| value.is_finite()));
        assert!(q_lora.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer3_kv_state_fixture_has_target_shapes() {
        let tensors = prefill_layer3_kv_state_final_tile_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768,
                32 * 512,
                32 * 512,
            ]
        );
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer3_attention_fixture_has_target_shapes() {
        let (attention, hc_final_tile) = prefill_layer3_attention_fixture().unwrap();
        assert_eq!(attention.len(), 2048 * 4096);
        assert_eq!(hc_final_tile.len(), 32 * 4 * 4096);
        assert!(attention.iter().all(|value| value.is_finite()));
        assert!(hc_final_tile.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer3_complete_fixture_has_target_shapes() {
        let fixture = prefill_layer3_complete_final_tile_fixture().unwrap();
        assert_eq!(fixture.ffn_cur.len(), 32 * 4096);
        assert_eq!(fixture.ffn_norm.len(), 32 * 4096);
        assert_eq!(fixture.router_selected.len(), 32 * 6);
        assert_eq!(fixture.router_weights.len(), 32 * 6);
        assert_eq!(fixture.routed_out.len(), 32 * 4096);
        assert_eq!(fixture.shared_out.len(), 32 * 4096);
        assert_eq!(fixture.hc_post.len(), 32 * 4 * 4096);
    }

    #[test]
    fn prefill_layer4_qkv_fixture_has_target_shapes() {
        let tensors = prefill_layer4_qkv_final_tile_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 4096,
                32 * 4096,
                32 * 1024,
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768,
                32 * 512,
                32 * 512,
            ]
        );
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer5_qkv_fixture_has_target_shapes() {
        let tensors = prefill_layer5_qkv_final_tile_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 4096,
                32 * 4096,
                32 * 1024,
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768,
                32 * 512,
                32 * 512,
            ]
        );
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer6_qkv_fixture_has_target_shapes() {
        let tensors = prefill_layer6_qkv_final_tile_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 4096,
                32 * 4096,
                32 * 1024,
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768,
                32 * 512,
                32 * 512,
            ]
        );
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer4_compressor_fixture_has_target_shapes() {
        let tensors = prefill_layer4_compressor_fixture().unwrap();
        assert_eq!(tensors[0].len(), 512 * 512);
        assert_eq!(tensors[1].len(), 8 * 1024);
        assert_eq!(tensors[2].len(), 8 * 1024);
        assert_eq!(tensors[3].len(), 512 * 128);
        assert_eq!(tensors[4].len(), 8 * 256);
        assert_eq!(tensors[5].len(), 8 * 256);
        assert!(tensors.iter().flatten().all(|value| !value.is_nan()));
        assert!(tensors[2].iter().any(|value| *value == f32::NEG_INFINITY));
        assert!(tensors[5].iter().any(|value| *value == f32::NEG_INFINITY));
    }

    #[test]
    fn prefill_layer6_compressor_fixture_has_target_shapes() {
        let tensors = prefill_layer6_compressor_fixture().unwrap();
        assert_eq!(tensors[0].len(), 512 * 512);
        assert_eq!(tensors[1].len(), 8 * 1024);
        assert_eq!(tensors[2].len(), 8 * 1024);
        assert_eq!(tensors[3].len(), 512 * 128);
        assert_eq!(tensors[4].len(), 8 * 256);
        assert_eq!(tensors[5].len(), 8 * 256);
        assert!(tensors.iter().flatten().all(|value| !value.is_nan()));
        assert!(tensors[2].iter().any(|value| *value == f32::NEG_INFINITY));
        assert!(tensors[5].iter().any(|value| *value == f32::NEG_INFINITY));
    }

    #[test]
    fn prefill_layer6_attention_fixture_has_target_shapes() {
        let (attention, hc_post) = prefill_layer6_attention_fixture().unwrap();
        assert_eq!(attention.len(), 2048 * 4096);
        assert_eq!(hc_post.len(), 32 * 4 * 4096);
        assert!(attention.iter().all(|value| value.is_finite()));
        assert!(hc_post.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer6_attention_diagnostics_fixture_has_target_shapes() {
        let diagnostics = prefill_layer6_attention_diagnostics_fixture().unwrap();
        assert_eq!(diagnostics[0].len(), 32_768);
        assert_eq!(diagnostics[1].len(), 32_768);
        assert_eq!(diagnostics[2].len(), 8_192);
        assert!(diagnostics.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer6_complete_fixture_has_target_shapes() {
        let fixture = prefill_layer6_complete_final_tile_fixture().unwrap();
        assert_eq!(fixture.ffn_cur.len(), 32 * 4096);
        assert_eq!(fixture.ffn_norm.len(), 32 * 4096);
        assert_eq!(fixture.router_selected.len(), 32 * 6);
        assert_eq!(fixture.router_weights.len(), 32 * 6);
        assert_eq!(fixture.routed_out.len(), 32 * 4096);
        assert_eq!(fixture.shared_out.len(), 32 * 4096);
        assert_eq!(fixture.hc_post.len(), 32 * 4 * 4096);
    }

    #[test]
    fn prefill_layer8_qkv_fixture_has_target_shapes() {
        let tensors = prefill_layer8_qkv_final_tile_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 4096,
                32 * 4096,
                32 * 1024,
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768,
                32 * 512,
                32 * 512,
            ]
        );
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer8_compressor_fixture_has_target_shapes() {
        let tensors = prefill_layer8_compressor_fixture().unwrap();
        assert_eq!(tensors[0].len(), 512 * 512);
        assert_eq!(tensors[1].len(), 8 * 1024);
        assert_eq!(tensors[2].len(), 8 * 1024);
        assert_eq!(tensors[3].len(), 512 * 128);
        assert_eq!(tensors[4].len(), 8 * 256);
        assert_eq!(tensors[5].len(), 8 * 256);
        assert!(tensors.iter().flatten().all(|value| !value.is_nan()));
        assert!(tensors[2].iter().any(|value| *value == f32::NEG_INFINITY));
        assert!(tensors[5].iter().any(|value| *value == f32::NEG_INFINITY));
    }

    #[test]
    fn prefill_layer8_attention_fixture_has_target_shapes() {
        let (attention, hc_post) = prefill_layer8_attention_fixture().unwrap();
        assert_eq!(attention.len(), 2048 * 4096);
        assert_eq!(hc_post.len(), 32 * 4 * 4096);
        assert!(attention.iter().all(|value| value.is_finite()));
        assert!(hc_post.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer8_attention_diagnostics_fixture_has_target_shapes() {
        let diagnostics = prefill_layer8_attention_diagnostics_fixture().unwrap();
        assert_eq!(diagnostics[0].len(), 32_768);
        assert_eq!(diagnostics[1].len(), 32_768);
        assert_eq!(diagnostics[2].len(), 8_192);
        assert!(diagnostics.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer8_complete_fixture_has_target_shapes() {
        let fixture = prefill_layer8_complete_final_tile_fixture().unwrap();
        assert_eq!(fixture.ffn_cur.len(), 32 * 4096);
        assert_eq!(fixture.ffn_norm.len(), 32 * 4096);
        assert_eq!(fixture.router_selected.len(), 32 * 6);
        assert_eq!(fixture.router_weights.len(), 32 * 6);
        assert_eq!(fixture.routed_out.len(), 32 * 4096);
        assert_eq!(fixture.shared_out.len(), 32 * 4096);
        assert_eq!(fixture.hc_post.len(), 32 * 4 * 4096);
    }

    #[test]
    fn prefill_layer7_qkv_fixture_has_target_shapes() {
        let tensors = prefill_layer7_qkv_final_tile_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 4096,
                32 * 4096,
                32 * 1024,
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768,
                32 * 512,
                32 * 512,
            ]
        );
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer7_compressor_fixture_has_target_shapes() {
        let (compressed, state_kv, state_score) = prefill_layer7_compressor_fixture().unwrap();
        assert_eq!(compressed.len(), 16 * 512);
        assert_eq!(state_kv.len(), 128 * 512);
        assert_eq!(state_score.len(), 128 * 512);
        assert!(compressed.iter().all(|value| value.is_finite()));
        assert!(state_kv.iter().all(|value| value.to_bits() == 0));
        assert!(state_score.iter().all(|value| *value as u32 == 0xff80_0000));
    }

    #[test]
    fn prefill_layer7_attention_fixture_has_target_shapes() {
        let (attention, hc_post) = prefill_layer7_attention_fixture().unwrap();
        let diagnostics = prefill_layer7_attention_diagnostics_fixture().unwrap();
        assert_eq!(attention.len(), 2048 * 4096);
        assert_eq!(hc_post.len(), 32 * 4 * 4096);
        assert_eq!(
            diagnostics.each_ref().map(|tensor| tensor.len()),
            [32_768, 32_768, 8_192]
        );
        assert!(attention.iter().all(|value| value.is_finite()));
        assert!(hc_post.iter().all(|value| value.is_finite()));
        assert!(diagnostics.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer7_complete_fixture_has_target_shapes() {
        let fixture = prefill_layer7_complete_final_tile_fixture().unwrap();
        assert_eq!(fixture.ffn_cur.len(), 32 * 4096);
        assert_eq!(fixture.ffn_norm.len(), 32 * 4096);
        assert_eq!(fixture.router_selected.len(), 32 * 6);
        assert_eq!(fixture.router_weights.len(), 32 * 6);
        assert_eq!(fixture.routed_out.len(), 32 * 4096);
        assert_eq!(fixture.shared_out.len(), 32 * 4096);
        assert_eq!(fixture.hc_post.len(), 32 * 4 * 4096);
    }

    #[test]
    fn prefill_layer4_attention_fixture_has_target_shapes() {
        let (attention, hc_post) = prefill_layer4_attention_fixture().unwrap();
        assert_eq!(attention.len(), 2048 * 4096);
        assert_eq!(hc_post.len(), 32 * 4 * 4096);
        assert!(attention.iter().all(|value| value.is_finite()));
        assert!(hc_post.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer5_attention_fixture_has_target_shapes() {
        let (attention, hc_post) = prefill_layer5_attention_fixture().unwrap();
        assert_eq!(attention.len(), 2048 * 4096);
        assert_eq!(hc_post.len(), 32 * 4 * 4096);
        assert!(attention.iter().all(|value| value.is_finite()));
        assert!(hc_post.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer5_complete_fixture_has_target_shapes() {
        let fixture = prefill_layer5_complete_final_tile_fixture().unwrap();
        assert_eq!(fixture.ffn_cur.len(), 32 * 4096);
        assert_eq!(fixture.ffn_norm.len(), 32 * 4096);
        assert_eq!(fixture.router_selected.len(), 32 * 6);
        assert_eq!(fixture.router_weights.len(), 32 * 6);
        assert_eq!(fixture.routed_out.len(), 32 * 4096);
        assert_eq!(fixture.shared_out.len(), 32 * 4096);
        assert_eq!(fixture.hc_post.len(), 32 * 4 * 4096);
    }

    #[test]
    fn prefill_layer5_attention_diagnostics_fixture_has_target_shapes() {
        let diagnostics = prefill_layer5_attention_diagnostics_fixture().unwrap();
        assert_eq!(
            diagnostics.each_ref().map(|tensor| tensor.len()),
            [64 * 512, 64 * 512, 8192]
        );
        assert!(diagnostics.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layer4_complete_fixture_has_target_shapes() {
        let fixture = prefill_layer4_complete_final_tile_fixture().unwrap();
        assert_eq!(fixture.ffn_cur.len(), 32 * 4096);
        assert_eq!(fixture.ffn_norm.len(), 32 * 4096);
        assert_eq!(fixture.router_selected.len(), 32 * 6);
        assert_eq!(fixture.router_weights.len(), 32 * 6);
        assert_eq!(fixture.routed_out.len(), 32 * 4096);
        assert_eq!(fixture.shared_out.len(), 32 * 4096);
        assert_eq!(fixture.hc_post.len(), 32 * 4 * 4096);
    }

    #[test]
    fn prefill_layer4_attention_diagnostics_fixture_has_target_shapes() {
        let diagnostics = prefill_layer4_attention_diagnostics_fixture().unwrap();
        assert_eq!(
            diagnostics.each_ref().map(|tensor| tensor.len()),
            [64 * 512, 64 * 512, 8192]
        );
        assert!(diagnostics.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prefill_layers01_previous_tile_fixture_has_target_shapes() {
        let (tensors, selected) = prefill_layers01_previous_tile_fixture().unwrap();
        assert_eq!(tensors[0].len(), 32 * 512);
        assert_eq!(tensors[1].len(), 32 * 16384);
        assert_eq!(tensors[2].len(), 32 * 512);
        assert_eq!(tensors[3].len(), 32 * 16384);
        assert_eq!(selected[0].len(), 32 * 6);
        assert_eq!(selected[1].len(), 32 * 6);
        assert!(tensors.iter().flatten().all(|value| value.is_finite()));
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
    fn prefill_q8_boundary_fixture_has_target_shapes() {
        let (input, batch_output, decode_output) = prefill_q8_boundary_fixture().unwrap();
        assert_eq!(input.len(), 128 * 4096);
        assert_eq!(batch_output.len(), 128 * 1024);
        assert_eq!(decode_output.len(), 1024);
        let final_batch = &batch_output[batch_output.len() - 1024..];
        assert_eq!(
            final_batch
                .iter()
                .zip(&decode_output)
                .filter(|(batch, decode)| batch.to_bits() != decode.to_bits())
                .count(),
            1024
        );
    }

    #[test]
    fn prefill_layer3_compressor_fixture_has_target_shapes() {
        let (compressed, state_kv, state_score) = prefill_layer3_compressor_fixture().unwrap();
        assert_eq!(compressed.len(), 16 * 512);
        assert_eq!(state_kv.len(), 128 * 512);
        assert_eq!(state_score.len(), 128 * 512);
        assert!(compressed.iter().all(|value| value.is_finite()));
        assert!(state_kv.iter().all(|value| value.to_bits() == 0));
        assert!(state_score.iter().all(|value| *value as u32 == 0xff80_0000));
    }

    #[test]
    fn prefill_layer5_compressor_fixture_has_target_shapes() {
        let (compressed, state_kv, state_score) = prefill_layer5_compressor_fixture().unwrap();
        assert_eq!(compressed.len(), 16 * 512);
        assert_eq!(state_kv.len(), 128 * 512);
        assert_eq!(state_score.len(), 128 * 512);
        assert!(compressed.iter().all(|value| value.is_finite()));
        assert!(state_kv.iter().all(|value| value.to_bits() == 0));
        assert!(state_score.iter().all(|value| *value as u32 == 0xff80_0000));
    }

    #[test]
    fn prefill_qkv_boundary_fixture_has_target_shapes() {
        let tensors = prefill_qkv_boundary_fixture().unwrap();
        assert_eq!(
            tensors.each_ref().map(|tensor| tensor.len()),
            [
                32 * 4096,
                32 * 1024,
                32 * 1024,
                32 * 512,
                32 * 512,
                32 * 32768,
                32 * 32768
            ]
        );
        assert_eq!(
            tensors.each_ref().map(|tensor| checksum_f32(tensor)),
            [
                3_658_078_701_343_054_310,
                12_227_882_723_182_009_333,
                3_802_385_984_011_127_479,
                7_046_193_354_968_420_108,
                17_664_819_937_448_664_622,
                18_441_511_941_582_035_647,
                15_715_312_012_925_293_902,
            ]
        );
    }

    #[test]
    fn prefill_hc_ingress_fixture_has_target_shapes() {
        let (tokens, collapsed, attn_norm) = prefill_hc_ingress_fixture().unwrap();
        assert_eq!(tokens.len(), 32);
        assert_eq!(collapsed.len(), 32 * 4096);
        assert_eq!(attn_norm.len(), 32 * 4096);
        assert_eq!(checksum_u32(&tokens), 4_632_138_271_124_972_668);
        assert_eq!(checksum_f32(&collapsed), 7_852_831_826_961_088_429);
        assert_eq!(checksum_f32(&attn_norm), 3_658_078_701_343_054_310);
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
    fn position127_decoder_fixture_has_target_shapes() {
        let (tokens, logits) = position127_decoder_fixture().unwrap();
        assert_eq!(tokens.len(), 128);
        assert_eq!(&tokens[..5], &[201, 361, 1915, 262, 1554]);
        assert_eq!(tokens.last(), Some(&33148));
        assert_eq!(logits.len(), 129280);
        assert_eq!(lowest_id_argmax(&logits).unwrap(), 33148);
    }

    #[test]
    fn cold_prefill_fixture_has_target_shape_and_selection() {
        let logits = cold_prefill_fixture().unwrap();
        assert_eq!(logits.len(), 129280);
        assert_eq!(lowest_id_argmax(&logits).unwrap(), 201);
    }

    #[test]
    fn prefill_frontier_fixture_has_target_shape_and_selection() {
        let (tokens, batch_logits, decode_logits) = prefill_frontier_2048_fixture().unwrap();
        assert_eq!(tokens.len(), 2048);
        assert_eq!(tokens.first(), Some(&36662));
        assert_eq!(tokens.last(), Some(&895));
        assert_eq!(batch_logits.len(), 129280);
        assert_eq!(decode_logits.len(), 129280);
        assert_eq!(lowest_id_argmax(&batch_logits).unwrap(), 15342);
        assert_eq!(lowest_id_argmax(&decode_logits).unwrap(), 15342);
        assert_ne!(batch_logits, decode_logits);
    }

    #[test]
    fn sparse_indexed_attention_fixture_has_target_shapes() {
        let q = decode_f32_fixture(SPARSE_INDEXER_Q_BYTES, "indexer Q").unwrap();
        let weights = decode_f32_fixture(SPARSE_INDEXER_WEIGHTS_BYTES, "indexer weights").unwrap();
        let scores = decode_f32_fixture(SPARSE_INDEXER_SCORES_BYTES, "indexer scores").unwrap();
        let topk = decode_i32_fixture(SPARSE_INDEXER_TOPK_BYTES, "indexer top-k").unwrap();
        let out = decode_f32_fixture(SPARSE_KQV_OUT_BYTES, "KQV output").unwrap();
        let back = decode_f32_fixture(SPARSE_KQV_BACK_BYTES, "KQV back").unwrap();
        assert_eq!(q.len(), 64 * 128);
        assert_eq!(weights.len(), 64);
        assert_eq!(scores.len(), 513);
        assert_eq!(topk.len(), 512);
        assert_eq!(out.len(), 64 * 512);
        assert_eq!(back.len(), 64 * 512);
        assert!(topk.iter().all(|index| (0..513).contains(index)));
        let mut unique = topk.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 512);
        assert_eq!(checksum_f32(&q), 11_009_500_198_888_317_733);
        assert_eq!(checksum_i32(&topk), 10_522_194_933_279_573_573);
        assert_eq!(checksum_f32(&out), 2_951_582_711_540_827_778);
        assert_eq!(checksum_f32(&back), 10_450_464_586_724_796_497);

        let default_scores = decode_f32_fixture(
            SPARSE_DEFAULT_INDEXER_SCORES_BYTES,
            "default-boundary indexer scores",
        )
        .unwrap();
        let default_topk = decode_i32_fixture(
            SPARSE_DEFAULT_INDEXER_TOPK_BYTES,
            "default-boundary indexer top-k",
        )
        .unwrap();
        assert_eq!(default_scores.len(), 1025);
        assert_eq!(default_topk.len(), 512);
        assert!(default_topk.iter().all(|index| (0..1025).contains(index)));
    }

    #[test]
    fn retained_sparse_boundary_fixture_has_target_shapes() {
        assert_eq!(RETAINED_SPARSE_INPUT_HC_BYTES.len(), 16_384 * 4);
        assert_eq!(RETAINED_SPARSE_RAW_PRIOR_BYTES.len(), 127 * 512 * 4);
        assert_eq!(RETAINED_SPARSE_ATTN_COMP_PRIOR_BYTES.len(), 1_024 * 512 * 4);
        assert_eq!(
            RETAINED_SPARSE_INDEX_COMP_PRIOR_BYTES.len(),
            1_024 * 128 * 4
        );
        assert_eq!(RETAINED_SPARSE_ATTN_STATE_KV_BYTES.len(), 8_192 * 4);
        assert_eq!(RETAINED_SPARSE_ATTN_STATE_SCORE_BITS.len(), 8_192 * 4);
        assert_eq!(RETAINED_SPARSE_INDEX_STATE_KV_BYTES.len(), 2_048 * 4);
        assert_eq!(RETAINED_SPARSE_INDEX_STATE_SCORE_BITS.len(), 2_048 * 4);
        assert_eq!(RETAINED_SPARSE_Q_LORA_NORM_BYTES.len(), 1_024 * 4);
        assert_eq!(RETAINED_SPARSE_ATTN_NORM_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_SPARSE_Q_CUR_BYTES.len(), 32_768 * 4);
        assert_eq!(RETAINED_SPARSE_KV_CUR_BYTES.len(), 512 * 4);
        assert_eq!(RETAINED_SPARSE_COMPRESSED_KV_BYTES.len(), 512 * 4);
        assert_eq!(RETAINED_SPARSE_COMPRESSED_INDEXER_BYTES.len(), 128 * 4);
        assert_eq!(RETAINED_SPARSE_INDEXER_Q_BYTES.len(), 8_192 * 4);
        assert_eq!(RETAINED_SPARSE_INDEXER_WEIGHTS_BYTES.len(), 64 * 4);
        assert_eq!(RETAINED_SPARSE_INDEXER_SCORES_BYTES.len(), 1_025 * 4);
        assert_eq!(RETAINED_SPARSE_INDEXER_TOPK_BYTES.len(), 512 * 4);
        assert_eq!(RETAINED_SPARSE_KQV_OUT_BYTES.len(), 32_768 * 4);
        assert_eq!(RETAINED_SPARSE_KQV_BACK_BYTES.len(), 32_768 * 4);
        assert_eq!(RETAINED_SPARSE_ATTN_LOW_BYTES.len(), 8_192 * 4);
        assert_eq!(RETAINED_SPARSE_ATTN_OUT_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_SPARSE_HC_ATTN_POST_BYTES.len(), 16_384 * 4);
    }

    #[test]
    fn retained_sparse_multimerge_fixture_has_target_shapes() {
        assert_eq!(RETAINED_MULTIMERGE_INPUT_HC_BYTES.len(), 16_384 * 4);
        assert_eq!(RETAINED_MULTIMERGE_RAW_PRIOR_BYTES.len(), 127 * 512 * 4);
        assert_eq!(
            RETAINED_MULTIMERGE_ATTN_COMP_PRIOR_BYTES.len(),
            2_048 * 512 * 4
        );
        assert_eq!(
            RETAINED_MULTIMERGE_INDEX_COMP_PRIOR_BYTES.len(),
            2_048 * 128 * 4
        );
        assert_eq!(RETAINED_MULTIMERGE_ATTN_STATE_KV_BYTES.len(), 8_192 * 4);
        assert_eq!(RETAINED_MULTIMERGE_ATTN_STATE_SCORE_BITS.len(), 8_192 * 4);
        assert_eq!(RETAINED_MULTIMERGE_INDEX_STATE_KV_BYTES.len(), 2_048 * 4);
        assert_eq!(RETAINED_MULTIMERGE_INDEX_STATE_SCORE_BITS.len(), 2_048 * 4);
        assert_eq!(RETAINED_MULTIMERGE_INDEXER_SCORES_BYTES.len(), 2_049 * 4);
        assert_eq!(RETAINED_MULTIMERGE_INDEXER_TOPK_BYTES.len(), 512 * 4);
        assert_eq!(RETAINED_MULTIMERGE_KQV_OUT_BYTES.len(), 32_768 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_ATTN_POST_BYTES.len(), 16_384 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_ATTN_PRE_MIXES_BYTES.len(), 24 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_ATTN_PRE_WEIGHTS_BYTES.len(), 4 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_ATTN_PRE_COMB_BYTES.len(), 16 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_ATTN_PRE_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_MULTIMERGE_Q_LORA_BYTES.len(), 1_024 * 4);
        assert_eq!(RETAINED_MULTIMERGE_KV_RAW_BYTES.len(), 512 * 4);
        assert_eq!(RETAINED_MULTIMERGE_Q_RAW_BYTES.len(), 32_768 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_FFN_PRE_MIXES_BYTES.len(), 24 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_FFN_PRE_WEIGHTS_BYTES.len(), 4 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_FFN_PRE_COMB_BYTES.len(), 16 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_FFN_PRE_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_MULTIMERGE_FFN_NORM_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_MULTIMERGE_FFN_MOE_LOGITS_BYTES.len(), 256 * 4);
        assert_eq!(RETAINED_MULTIMERGE_FFN_MOE_TOPK_BYTES.len(), 6 * 4);
        assert_eq!(
            RETAINED_MULTIMERGE_FFN_MOE_WEIGHTED_SWIGLU_BYTES.len(),
            6 * 2_048 * 4
        );
        assert_eq!(RETAINED_MULTIMERGE_FFN_MOE_OUT_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_MULTIMERGE_FFN_SHEXP_BYTES.len(), 4_096 * 4);
        assert_eq!(RETAINED_MULTIMERGE_HC_FFN_POST_BYTES.len(), 16_384 * 4);
    }

    #[test]
    fn retained_sparse_topk_schedule_generalizes_merge_passes() {
        assert_eq!(retained_sparse_topk_schedule(513), (1, 0, 512));
        assert_eq!(retained_sparse_topk_schedule(1025), (2, 1, 513));
        assert_eq!(retained_sparse_topk_schedule(2048), (2, 1, 1024));
        assert_eq!(retained_sparse_topk_schedule(2049), (3, 2, 1025));
        assert_eq!(retained_sparse_topk_schedule(4097), (5, 3, 2049));
    }

    #[test]
    fn writes_stable_sparse_indexed_attention_probe_json() {
        let report = SparseIndexedAttentionProbeReport {
            fixture_id: SPARSE_INDEXED_ATTENTION_DEFAULT_FIXTURE_ID,
            position: 4099,
            compressed_rows: 1025,
            raw_rows: 128,
            top_k: 512,
            diagnostic_threshold_override: 0,
            pinned_default_threshold: 1024,
            first_default_sparse_rows: 1025,
            dispatches: 11,
            wrapped_model_ranges: 3,
            pointer_matches: 3,
            split_count: 12,
            wall_ms: 1.0,
            gpu_ms: 0.5,
            indexer_q_checksum: 1,
            indexer_weights_checksum: 2,
            indexer_scores_checksum: 3,
            indexer_topk_checksum: 4,
            kqv_out_checksum: 5,
            kqv_back_checksum: 6,
        };
        let mut output = Vec::new();
        write_sparse_indexed_attention_probe_json(&mut output, &report).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"c0_bitwise_match\": true"));
        assert!(text.contains("\"pinned_default\": 1024"));
        assert!(text.contains("\"diagnostic_override\": null"));
        assert!(text.contains("kernel_argsort_merge_f32_i32_desc"));
        assert!(text.contains("\"complete_decode_claim\": false"));
        assert!(text.contains("\"throughput_claim\": false"));
    }

    #[test]
    fn writes_stable_retained_sparse_boundary_probe_json() {
        let report = RetainedSparseBoundaryProbeReport {
            fixture_id: RETAINED_SPARSE_BOUNDARY_FIXTURE_ID,
            token: 0,
            layer: 2,
            position: 4099,
            raw_rows: 128,
            seeded_raw_rows: 127,
            compressed_rows: 1025,
            seeded_compressed_rows: 1024,
            top_k: 512,
            sort_blocks: 2,
            merge_passes: 1,
            topk_work_width: 513,
            dispatches: 54,
            wrapped_model_ranges: 35,
            pointer_matches: 35,
            wall_ms: 1.0,
            gpu_ms: 0.5,
            exact_tensor_checks: 16,
            q_current_checksum: 1,
            compressed_kv_checksum: 2,
            compressed_indexer_checksum: 3,
            indexer_scores_checksum: 4,
            indexer_topk_checksum: 5,
            kqv_out_checksum: 6,
            kqv_back_checksum: 7,
            attention_hc_checksum: 8,
            selected_experts_checksum: 9,
            final_hc_checksum: 10,
        };
        let mut output = Vec::new();
        write_retained_sparse_boundary_probe_json(&mut output, &report).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"schema\": \"rust-star-retained-sparse-boundary-v1\""));
        assert!(text.contains("\"two_block_topk_merge\": true"));
        assert!(text.contains("\"retained_layer_execution_claim\": true"));
        assert!(text.contains("\"complete_layer_claim\": false"));
        assert!(text.contains("\"complete_decoder_claim\": false"));
        assert!(text.contains("\"throughput_claim\": false"));
    }

    #[test]
    fn writes_stable_retained_sparse_multimerge_probe_json() {
        let report = RetainedSparseBoundaryProbeReport {
            fixture_id: RETAINED_SPARSE_MULTIMERGE_FIXTURE_ID,
            token: 381,
            layer: 2,
            position: 8195,
            raw_rows: 128,
            seeded_raw_rows: 127,
            compressed_rows: 2049,
            seeded_compressed_rows: 2048,
            top_k: 512,
            sort_blocks: 3,
            merge_passes: 2,
            topk_work_width: 1025,
            dispatches: 55,
            wrapped_model_ranges: 35,
            pointer_matches: 35,
            wall_ms: 1.0,
            gpu_ms: 0.5,
            exact_tensor_checks: 40,
            q_current_checksum: 1,
            compressed_kv_checksum: 2,
            compressed_indexer_checksum: 3,
            indexer_scores_checksum: 4,
            indexer_topk_checksum: 5,
            kqv_out_checksum: 6,
            kqv_back_checksum: 7,
            attention_hc_checksum: 8,
            selected_experts_checksum: 9,
            final_hc_checksum: 10,
        };
        let mut output = Vec::new();
        write_retained_sparse_multimerge_probe_json(&mut output, &report).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"schema\": \"rust-star-retained-sparse-multimerge-v1\""));
        assert!(text.contains("\"sort_blocks\": 3"));
        assert!(text.contains("\"merge_passes\": 2"));
        assert!(text.contains("\"ping_pong_workspace\": true"));
        assert!(text.contains("\"repeated_merge_boundary_claim\": true"));
        assert!(text.contains("\"token\": 381"));
        assert!(text.contains("\"exact_tensor_checks\": 40"));
        assert!(text.contains("\"complete_layer_claim\": true"));
        assert!(text.contains("\"preceding_layers_execution_claim\": false"));
        assert!(text.contains("\"complete_decoder_claim\": false"));
        assert!(text.contains("\"throughput_claim\": false"));
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
