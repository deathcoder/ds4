//! Exact DeepSeek V4 Flash resident-Q2 target validation.

use crate::gguf::{Gguf, ScalarValue, TensorInfo, ValueType};
use crate::{Error, Result};
use std::collections::BTreeMap;

pub const MODEL_LABEL: &str = "DeepSeek-V4-Flash-0731";
pub const LAYERS: u32 = 43;
pub const EMBEDDING: u64 = 4096;
pub const VOCAB: u64 = 129_280;
pub const EXPERTS: u64 = 256;
pub const ACTIVE_EXPERTS: u64 = 6;

const HEADS: u64 = 64;
const HEAD_DIM: u64 = 512;
const LORA_Q: u64 = 1024;
const LORA_O: u64 = 1024;
const OUTPUT_GROUPS: u64 = 8;
const EXPERT_FF: u64 = 2048;
const INDEXER_HEADS: u64 = 64;
const INDEXER_HEAD_DIM: u64 = 128;
const HC: u64 = 4;

#[derive(Clone, Debug)]
pub struct TargetReport {
    pub model_name: Option<String>,
    pub checkpoint_name_mentions_0731: bool,
    pub model_sha_required_for_checkpoint_identity: bool,
    pub required_tensor_count: usize,
    pub extra_tensor_count: usize,
    pub tensor_type_counts: BTreeMap<&'static str, usize>,
    pub imatrix_entries: u64,
}

/// Validate the exact model shape and quantization layout selected for the
/// resident 128 GiB baseline.
///
/// This establishes shape and recipe identity. It cannot establish checkpoint
/// identity by itself: the completed oracle manifest's whole-file SHA-256 is
/// the authority for `0731` because GGUF metadata names are not immutable.
pub fn validate_resident_q2(gguf: &Gguf) -> Result<TargetReport> {
    expect_string(gguf, "general.architecture", "deepseek4")?;
    expect_u64(gguf, "deepseek4.block_count", LAYERS as u64)?;
    expect_u64(gguf, "deepseek4.embedding_length", EMBEDDING)?;
    expect_u64(gguf, "deepseek4.vocab_size", VOCAB)?;
    expect_u64(gguf, "deepseek4.attention.head_count", HEADS)?;
    expect_u64(gguf, "deepseek4.attention.head_count_kv", 1)?;
    expect_u64(gguf, "deepseek4.attention.key_length", HEAD_DIM)?;
    expect_u64(gguf, "deepseek4.attention.value_length", HEAD_DIM)?;
    expect_u64(gguf, "deepseek4.rope.dimension_count", 64)?;
    expect_u64(gguf, "deepseek4.attention.q_lora_rank", LORA_Q)?;
    expect_u64(gguf, "deepseek4.attention.output_lora_rank", LORA_O)?;
    expect_u64(
        gguf,
        "deepseek4.attention.output_group_count",
        OUTPUT_GROUPS,
    )?;
    expect_u64(gguf, "deepseek4.expert_count", EXPERTS)?;
    expect_u64(gguf, "deepseek4.expert_used_count", ACTIVE_EXPERTS)?;
    expect_u64(gguf, "deepseek4.expert_feed_forward_length", EXPERT_FF)?;
    expect_u64(gguf, "deepseek4.expert_shared_count", 1)?;
    expect_u64(gguf, "deepseek4.hash_layer_count", 3)?;
    expect_optional_u64(gguf, "deepseek4.expert_group_count", 0)?;
    expect_optional_u64(gguf, "deepseek4.expert_group_used_count", 0)?;
    expect_u64(gguf, "deepseek4.attention.sliding_window", 128)?;
    expect_u64(
        gguf,
        "deepseek4.attention.indexer.head_count",
        INDEXER_HEADS,
    )?;
    expect_u64(
        gguf,
        "deepseek4.attention.indexer.key_length",
        INDEXER_HEAD_DIM,
    )?;
    expect_u64(gguf, "deepseek4.attention.indexer.top_k", 512)?;
    expect_u64(gguf, "deepseek4.hyper_connection.count", HC)?;
    expect_u64(gguf, "deepseek4.hyper_connection.sinkhorn_iterations", 20)?;

    expect_f64(gguf, "deepseek4.rope.freq_base", 10_000.0)?;
    expect_optional_f64(gguf, "deepseek4.rope.scaling.factor", 16.0)?;
    expect_optional_f64(gguf, "deepseek4.rope.scaling.yarn_beta_fast", 32.0)?;
    expect_optional_f64(gguf, "deepseek4.rope.scaling.yarn_beta_slow", 1.0)?;
    expect_optional_u64(
        gguf,
        "deepseek4.rope.scaling.original_context_length",
        65_536,
    )?;
    expect_f64(
        gguf,
        "deepseek4.attention.compress_rope_freq_base",
        160_000.0,
    )?;
    expect_f64(gguf, "deepseek4.expert_weights_scale", 1.5)?;
    expect_f64(gguf, "deepseek4.attention.layer_norm_rms_epsilon", 1.0e-6)?;
    expect_f64(gguf, "deepseek4.hyper_connection.epsilon", 1.0e-6)?;
    if !gguf.boolean("deepseek4.expert_weights_norm")? {
        return Err(Error::invalid("deepseek4.expert_weights_norm must be true"));
    }
    validate_compression_ratios(gguf)?;
    validate_swiglu_clamp(gguf)?;

    let imatrix_file = gguf.string("quantize.imatrix.file")?;
    if imatrix_file.trim().is_empty() {
        return Err(Error::invalid("quantize.imatrix.file is empty"));
    }
    let imatrix_entries = gguf.u64("quantize.imatrix.entries_count")?;
    if imatrix_entries == 0 {
        return Err(Error::invalid(
            "quantize.imatrix.entries_count must be nonzero",
        ));
    }

    let expected = expected_tensors();
    for tensor in &expected {
        let actual = gguf.tensors.get(&tensor.name).ok_or_else(|| {
            Error::invalid(format!(
                "required target tensor {:?} is missing",
                tensor.name
            ))
        })?;
        validate_tensor(actual, tensor)?;
    }

    let mut tensor_type_counts = BTreeMap::new();
    for tensor in gguf.tensors.values() {
        *tensor_type_counts
            .entry(tensor.tensor_type.name)
            .or_insert(0) += 1;
    }
    let model_name = optional_string(gguf, "general.name")?.map(str::to_owned);
    let checkpoint_name_mentions_0731 = model_name
        .as_deref()
        .map(|name| name.to_ascii_lowercase().contains("0731"))
        .unwrap_or(false);

    Ok(TargetReport {
        model_name,
        checkpoint_name_mentions_0731,
        model_sha_required_for_checkpoint_identity: true,
        required_tensor_count: expected.len(),
        extra_tensor_count: gguf.tensors.len().saturating_sub(expected.len()),
        tensor_type_counts,
        imatrix_entries,
    })
}

#[derive(Clone, Debug)]
struct ExpectedTensor {
    name: String,
    type_id: u32,
    dimensions: Vec<u64>,
}

fn expected(name: impl Into<String>, type_id: u32, dimensions: &[u64]) -> ExpectedTensor {
    ExpectedTensor {
        name: name.into(),
        type_id,
        dimensions: dimensions.to_vec(),
    }
}

fn expected_tensors() -> Vec<ExpectedTensor> {
    let hc_dim = EMBEDDING * HC;
    let hc_mix_dim = 2 * HC + HC * HC;
    let q_dim = HEADS * HEAD_DIM;
    let out_low_dim = OUTPUT_GROUPS * LORA_O;
    let mut tensors = vec![
        expected("token_embd.weight", 1, &[EMBEDDING, VOCAB]),
        expected("output_hc_base.weight", 0, &[HC]),
        expected("output_hc_fn.weight", 1, &[hc_dim, HC]),
        expected("output_hc_scale.weight", 0, &[1]),
        expected("output_norm.weight", 0, &[EMBEDDING]),
        expected("output.weight", 8, &[EMBEDDING, VOCAB]),
    ];

    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        let ratio = expected_compression_ratio(layer);
        tensors.extend([
            expected(
                format!("{prefix}.hc_attn_fn.weight"),
                1,
                &[hc_dim, hc_mix_dim],
            ),
            expected(format!("{prefix}.hc_attn_scale.weight"), 0, &[3]),
            expected(format!("{prefix}.hc_attn_base.weight"), 0, &[hc_mix_dim]),
            expected(format!("{prefix}.attn_norm.weight"), 0, &[EMBEDDING]),
            expected(format!("{prefix}.attn_q_a.weight"), 8, &[EMBEDDING, LORA_Q]),
            expected(format!("{prefix}.attn_q_a_norm.weight"), 0, &[LORA_Q]),
            expected(format!("{prefix}.attn_q_b.weight"), 8, &[LORA_Q, q_dim]),
            expected(
                format!("{prefix}.attn_kv.weight"),
                8,
                &[EMBEDDING, HEAD_DIM],
            ),
            expected(format!("{prefix}.attn_kv_a_norm.weight"), 0, &[HEAD_DIM]),
            expected(format!("{prefix}.attn_sinks.weight"), 0, &[HEADS]),
            expected(
                format!("{prefix}.attn_output_a.weight"),
                8,
                &[HEAD_DIM * (HEADS / OUTPUT_GROUPS), out_low_dim],
            ),
            expected(
                format!("{prefix}.attn_output_b.weight"),
                8,
                &[out_low_dim, EMBEDDING],
            ),
        ]);
        if ratio != 0 {
            let compressor_width = if ratio == 4 { 2 * HEAD_DIM } else { HEAD_DIM };
            tensors.extend([
                expected(
                    format!("{prefix}.attn_compressor_ape.weight"),
                    1,
                    &[compressor_width, ratio],
                ),
                expected(
                    format!("{prefix}.attn_compressor_kv.weight"),
                    1,
                    &[EMBEDDING, compressor_width],
                ),
                expected(
                    format!("{prefix}.attn_compressor_gate.weight"),
                    1,
                    &[EMBEDDING, compressor_width],
                ),
                expected(
                    format!("{prefix}.attn_compressor_norm.weight"),
                    0,
                    &[HEAD_DIM],
                ),
            ]);
        }
        if ratio == 4 {
            let index_q_dim = INDEXER_HEADS * INDEXER_HEAD_DIM;
            let index_width = 2 * INDEXER_HEAD_DIM;
            tensors.extend([
                expected(
                    format!("{prefix}.indexer.attn_q_b.weight"),
                    1,
                    &[LORA_Q, index_q_dim],
                ),
                expected(
                    format!("{prefix}.indexer.proj.weight"),
                    1,
                    &[EMBEDDING, INDEXER_HEADS],
                ),
                expected(
                    format!("{prefix}.indexer_compressor_ape.weight"),
                    1,
                    &[index_width, ratio],
                ),
                expected(
                    format!("{prefix}.indexer_compressor_kv.weight"),
                    1,
                    &[EMBEDDING, index_width],
                ),
                expected(
                    format!("{prefix}.indexer_compressor_gate.weight"),
                    1,
                    &[EMBEDDING, index_width],
                ),
                expected(
                    format!("{prefix}.indexer_compressor_norm.weight"),
                    0,
                    &[INDEXER_HEAD_DIM],
                ),
            ]);
        }
        tensors.extend([
            expected(
                format!("{prefix}.hc_ffn_fn.weight"),
                1,
                &[hc_dim, hc_mix_dim],
            ),
            expected(format!("{prefix}.hc_ffn_scale.weight"), 0, &[3]),
            expected(format!("{prefix}.hc_ffn_base.weight"), 0, &[hc_mix_dim]),
            expected(format!("{prefix}.ffn_norm.weight"), 0, &[EMBEDDING]),
            expected(
                format!("{prefix}.ffn_gate_inp.weight"),
                1,
                &[EMBEDDING, EXPERTS],
            ),
            expected(
                format!("{prefix}.ffn_gate_exps.weight"),
                16,
                &[EMBEDDING, EXPERT_FF, EXPERTS],
            ),
            expected(
                format!("{prefix}.ffn_up_exps.weight"),
                16,
                &[EMBEDDING, EXPERT_FF, EXPERTS],
            ),
            expected(
                format!("{prefix}.ffn_down_exps.weight"),
                10,
                &[EXPERT_FF, EMBEDDING, EXPERTS],
            ),
            expected(
                format!("{prefix}.ffn_gate_shexp.weight"),
                8,
                &[EMBEDDING, EXPERT_FF],
            ),
            expected(
                format!("{prefix}.ffn_up_shexp.weight"),
                8,
                &[EMBEDDING, EXPERT_FF],
            ),
            expected(
                format!("{prefix}.ffn_down_shexp.weight"),
                8,
                &[EXPERT_FF, EMBEDDING],
            ),
        ]);
        if layer < 3 {
            tensors.push(expected(
                format!("{prefix}.ffn_gate_tid2eid.weight"),
                26,
                &[ACTIVE_EXPERTS, VOCAB],
            ));
        }
    }
    tensors
}

fn validate_tensor(actual: &TensorInfo, expected: &ExpectedTensor) -> Result<()> {
    if actual.tensor_type.id != expected.type_id {
        return Err(Error::invalid(format!(
            "tensor {:?} has type {} ({}), expected GGML type {}",
            expected.name, actual.tensor_type.name, actual.tensor_type.id, expected.type_id
        )));
    }
    if actual.dimensions != expected.dimensions {
        return Err(Error::invalid(format!(
            "tensor {:?} has dimensions {:?}, expected {:?}",
            expected.name, actual.dimensions, expected.dimensions
        )));
    }
    Ok(())
}

fn expected_compression_ratio(layer: u32) -> u64 {
    if layer < 2 {
        0
    } else if layer % 2 == 0 {
        4
    } else {
        128
    }
}

fn validate_compression_ratios(gguf: &Gguf) -> Result<()> {
    let (element_type, values) = gguf.numeric_array("deepseek4.attention.compress_ratios")?;
    if !matches!(element_type, ValueType::Uint32 | ValueType::Int32) {
        return Err(Error::invalid(
            "deepseek4.attention.compress_ratios must contain uint32/int32",
        ));
    }
    if values.len() < LAYERS as usize {
        return Err(Error::invalid(format!(
            "deepseek4.attention.compress_ratios has {} entries, expected at least {LAYERS}",
            values.len()
        )));
    }
    for (layer, value) in values.iter().take(LAYERS as usize).enumerate() {
        let actual = scalar_nonnegative_u64(value).ok_or_else(|| {
            Error::invalid(format!(
                "compression ratio at layer {layer} is not nonnegative"
            ))
        })?;
        let expected = expected_compression_ratio(layer as u32);
        if actual != expected {
            return Err(Error::invalid(format!(
                "compression ratio at layer {layer} is {actual}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_swiglu_clamp(gguf: &Gguf) -> Result<()> {
    let (element_type, values) = gguf.numeric_array("deepseek4.swiglu_clamp_exp")?;
    if !matches!(element_type, ValueType::Float32 | ValueType::Float64) {
        return Err(Error::invalid(
            "deepseek4.swiglu_clamp_exp must contain float32/float64",
        ));
    }
    if values.len() < LAYERS as usize {
        return Err(Error::invalid(format!(
            "deepseek4.swiglu_clamp_exp has {} entries, expected at least {LAYERS}",
            values.len()
        )));
    }
    for (layer, value) in values.iter().take(LAYERS as usize).enumerate() {
        let actual = match value {
            ScalarValue::Float(value) => *value,
            _ => return Err(Error::invalid("SwiGLU clamp array contains a non-float")),
        };
        if !float_matches(actual, 10.0) {
            return Err(Error::invalid(format!(
                "SwiGLU clamp at layer {layer} is {actual}, expected 10"
            )));
        }
    }
    Ok(())
}

fn optional_string<'a>(gguf: &'a Gguf, key: &str) -> Result<Option<&'a str>> {
    if gguf.metadata.contains_key(key) {
        gguf.string(key).map(Some)
    } else {
        Ok(None)
    }
}

fn expect_string(gguf: &Gguf, key: &str, expected: &str) -> Result<()> {
    let actual = gguf.string(key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "metadata {key:?} is {actual:?}, expected {expected:?}"
        )))
    }
}

fn expect_u64(gguf: &Gguf, key: &str, expected: u64) -> Result<()> {
    let actual = gguf.u64(key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "metadata {key:?} is {actual}, expected {expected}"
        )))
    }
}

fn expect_optional_u64(gguf: &Gguf, key: &str, expected: u64) -> Result<()> {
    if gguf.metadata.contains_key(key) {
        expect_u64(gguf, key, expected)
    } else {
        Ok(())
    }
}

fn expect_f64(gguf: &Gguf, key: &str, expected: f64) -> Result<()> {
    let actual = gguf.f64(key)?;
    if float_matches(actual, expected) {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "metadata {key:?} is {actual}, expected {expected}"
        )))
    }
}

fn expect_optional_f64(gguf: &Gguf, key: &str, expected: f64) -> Result<()> {
    if gguf.metadata.contains_key(key) {
        expect_f64(gguf, key, expected)
    } else {
        Ok(())
    }
}

fn float_matches(actual: f64, expected: f64) -> bool {
    actual.is_finite() && (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6
}

fn scalar_nonnegative_u64(value: &ScalarValue) -> Option<u64> {
    match value {
        ScalarValue::Unsigned(value) => Some(*value),
        ScalarValue::Signed(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{MetadataValue, TensorType};

    fn valid_fixture() -> Gguf {
        let mut metadata = BTreeMap::new();
        let mut unsigned = |key: &str, value: u64| {
            metadata.insert(
                key.to_owned(),
                MetadataValue::Scalar {
                    value_type: ValueType::Uint32,
                    value: ScalarValue::Unsigned(value),
                },
            );
        };
        for (key, value) in [
            ("deepseek4.block_count", LAYERS as u64),
            ("deepseek4.embedding_length", EMBEDDING),
            ("deepseek4.vocab_size", VOCAB),
            ("deepseek4.attention.head_count", HEADS),
            ("deepseek4.attention.head_count_kv", 1),
            ("deepseek4.attention.key_length", HEAD_DIM),
            ("deepseek4.attention.value_length", HEAD_DIM),
            ("deepseek4.rope.dimension_count", 64),
            ("deepseek4.attention.q_lora_rank", LORA_Q),
            ("deepseek4.attention.output_lora_rank", LORA_O),
            ("deepseek4.attention.output_group_count", OUTPUT_GROUPS),
            ("deepseek4.expert_count", EXPERTS),
            ("deepseek4.expert_used_count", ACTIVE_EXPERTS),
            ("deepseek4.expert_feed_forward_length", EXPERT_FF),
            ("deepseek4.expert_shared_count", 1),
            ("deepseek4.hash_layer_count", 3),
            ("deepseek4.expert_group_count", 0),
            ("deepseek4.expert_group_used_count", 0),
            ("deepseek4.attention.sliding_window", 128),
            ("deepseek4.attention.indexer.head_count", INDEXER_HEADS),
            ("deepseek4.attention.indexer.key_length", INDEXER_HEAD_DIM),
            ("deepseek4.attention.indexer.top_k", 512),
            ("deepseek4.hyper_connection.count", HC),
            ("deepseek4.hyper_connection.sinkhorn_iterations", 20),
            ("deepseek4.rope.scaling.original_context_length", 65_536),
        ] {
            unsigned(key, value);
        }
        drop(unsigned);

        for (key, value) in [
            ("deepseek4.rope.freq_base", 10_000.0),
            ("deepseek4.rope.scaling.factor", 16.0),
            ("deepseek4.rope.scaling.yarn_beta_fast", 32.0),
            ("deepseek4.rope.scaling.yarn_beta_slow", 1.0),
            ("deepseek4.attention.compress_rope_freq_base", 160_000.0),
            ("deepseek4.expert_weights_scale", 1.5),
            ("deepseek4.attention.layer_norm_rms_epsilon", 1.0e-6),
            ("deepseek4.hyper_connection.epsilon", 1.0e-6),
        ] {
            metadata.insert(
                key.to_owned(),
                MetadataValue::Scalar {
                    value_type: ValueType::Float32,
                    value: ScalarValue::Float(value),
                },
            );
        }
        for (key, value) in [
            ("general.architecture", "deepseek4"),
            ("general.name", "DeepSeek-V4-Flash-0731 fixture"),
            ("quantize.imatrix.file", "fixture.dat"),
        ] {
            metadata.insert(
                key.to_owned(),
                MetadataValue::Scalar {
                    value_type: ValueType::String,
                    value: ScalarValue::String(value.to_owned()),
                },
            );
        }
        metadata.insert(
            "quantize.imatrix.entries_count".to_owned(),
            MetadataValue::Scalar {
                value_type: ValueType::Uint64,
                value: ScalarValue::Unsigned(1),
            },
        );
        metadata.insert(
            "deepseek4.expert_weights_norm".to_owned(),
            MetadataValue::Scalar {
                value_type: ValueType::Bool,
                value: ScalarValue::Bool(true),
            },
        );
        metadata.insert(
            "deepseek4.attention.compress_ratios".to_owned(),
            MetadataValue::Array {
                element_type: ValueType::Uint32,
                len: LAYERS as u64,
                values: Some(
                    (0..LAYERS)
                        .map(|layer| ScalarValue::Unsigned(expected_compression_ratio(layer)))
                        .collect(),
                ),
            },
        );
        metadata.insert(
            "deepseek4.swiglu_clamp_exp".to_owned(),
            MetadataValue::Array {
                element_type: ValueType::Float32,
                len: LAYERS as u64,
                values: Some(vec![ScalarValue::Float(10.0); LAYERS as usize]),
            },
        );

        let tensors = expected_tensors()
            .into_iter()
            .map(|expected| {
                let tensor_type = TensorType::from_id(expected.type_id).unwrap();
                let elements = expected.dimensions.iter().product();
                let name = expected.name;
                (
                    name.clone(),
                    TensorInfo {
                        name,
                        dimensions: expected.dimensions,
                        tensor_type,
                        relative_offset: 0,
                        absolute_offset: 0,
                        elements,
                        bytes: 0,
                    },
                )
            })
            .collect();

        Gguf {
            version: 3,
            file_bytes: 0,
            alignment: 32,
            tensor_data_offset: 0,
            metadata,
            tensors,
        }
    }

    #[test]
    fn flash_compression_schedule_matches_dwarfstar() {
        let actual: Vec<_> = (0..8).map(expected_compression_ratio).collect();
        assert_eq!(actual, [0, 0, 4, 128, 4, 128, 4, 128]);
    }

    #[test]
    fn expected_tensor_names_are_unique_and_cover_all_routed_layers() {
        let tensors = expected_tensors();
        let names: std::collections::BTreeSet<_> =
            tensors.iter().map(|tensor| tensor.name.clone()).collect();
        assert_eq!(names.len(), tensors.len());
        for layer in 0..LAYERS {
            for suffix in ["gate", "up", "down"] {
                assert!(names.contains(&format!("blk.{layer}.ffn_{suffix}_exps.weight")));
            }
        }
    }

    #[test]
    fn q2_recipe_uses_iq2_gate_up_and_q2_down() {
        let tensors = expected_tensors();
        for layer in 0..LAYERS {
            let find = |suffix: &str| {
                tensors
                    .iter()
                    .find(|tensor| tensor.name == format!("blk.{layer}.ffn_{suffix}_exps.weight"))
                    .unwrap()
                    .type_id
            };
            assert_eq!(find("gate"), 16);
            assert_eq!(find("up"), 16);
            assert_eq!(find("down"), 10);
        }
    }

    #[test]
    fn indexer_projection_recipe_is_f16() {
        let tensors = expected_tensors();
        for layer in (2..LAYERS).step_by(2) {
            for suffix in [
                "indexer.attn_q_b.weight",
                "indexer.proj.weight",
                "indexer_compressor_ape.weight",
                "indexer_compressor_kv.weight",
                "indexer_compressor_gate.weight",
            ] {
                let tensor = tensors
                    .iter()
                    .find(|tensor| tensor.name == format!("blk.{layer}.{suffix}"))
                    .unwrap();
                assert_eq!(tensor.type_id, 1, "{}", tensor.name);
            }
        }
    }

    #[test]
    fn complete_fixture_passes_and_wrong_quant_fails() {
        let fixture = valid_fixture();
        let report = validate_resident_q2(&fixture).unwrap();
        assert_eq!(report.extra_tensor_count, 0);
        assert!(report.checkpoint_name_mentions_0731);

        let mut wrong = fixture;
        wrong
            .tensors
            .get_mut("blk.0.ffn_down_exps.weight")
            .unwrap()
            .tensor_type = TensorType::from_id(16).unwrap();
        let error = validate_resident_q2(&wrong).unwrap_err().to_string();
        assert!(error.contains("expected GGML type 10"), "{error}");
    }
}
