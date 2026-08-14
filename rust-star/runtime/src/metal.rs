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
pub const PROJECTION_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attn-q-a";
pub const INGRESS_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attention-ingress";
pub const ATTENTION_SETUP_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-qkv-setup";
pub const ROPE_KV_STORE_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-rope-kv-store";
pub const ATTENTION_READ_FIXTURE_ID: &str = "dwarfstar-oracle-v1-layer0-pos1-attention-read";
pub const DEFAULT_ELEMENTS: u64 = 4096;
pub const DEFAULT_ITERATIONS: u64 = 100;
const MAX_ELEMENTS: u64 = 16 * 1024 * 1024;
const MAX_ITERATIONS: u64 = 100_000;
const MAX_THREAD_INVOCATIONS: u64 = 1_000_000_000;
const MAX_EMBEDDING_TOKENS: usize = 64;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeConfig {
    pub elements: u64,
    pub iterations: u64,
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
            result: *mut RawIngressProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_destroy(context: *mut c_void);
    }

    struct Context(*mut c_void);

    impl Drop for Context {
        fn drop(&mut self) {
            unsafe { rust_star_metal_destroy(self.0) };
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
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
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
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
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
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
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
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
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
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    pub fn run_probe(config: ProbeConfig) -> Result<ProbeReport> {
        config.validate()?;
        Err(Error::invalid(
            "the Metal dispatch probe is available only on macOS",
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
}

pub use imp::{
    run_attention_ingress_probe, run_attention_read_probe, run_attention_setup_probe,
    run_f16_embedding_probe, run_probe, run_q8_projection_probe, run_rope_kv_store_probe,
};

#[cfg(test)]
mod tests {
    use super::*;

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
