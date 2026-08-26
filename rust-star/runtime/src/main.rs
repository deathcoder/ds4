use rust_star_runtime::gguf::Gguf;
use rust_star_runtime::metal::{
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
    run_ratio128_compressor_replay_probe, run_retained_decoder_step_probe,
    run_retained_sparse_boundary_probe, run_retained_sparse_multimerge_probe,
    run_rope_kv_store_probe, run_sparse_indexed_attention_probe, write_attention_output_probe_json,
    write_attention_read_probe_json, write_attention_setup_probe_json,
    write_closed_loop_decoder_probe_json, write_cold_prefill_decoder_probe_json,
    write_decoder_output_probe_json, write_embedding_probe_json, write_ffn_router_probe_json,
    write_ingress_probe_json, write_layer0_bench_json, write_layer0_probe_json,
    write_layers01234567_decode_probe_json, write_layers012345_decode_probe_json,
    write_layers0123_bench_json, write_layers0123_chained_probe_json,
    write_layers0123_decode_probe_json, write_layers0123_probe_json,
    write_layers012_chained_probe_json, write_layers012_probe_json, write_layers01_probe_json,
    write_layers0_to_42_decode_probe_json, write_moe_output_probe_json,
    write_position127_decoder_probe_json, write_prefill_frontier_probe_json,
    write_prefill_layer0_boundary_probe_json, write_prefill_layers012_attention_loop_probe_json,
    write_prefill_layers012_compressor_loop_probe_json,
    write_prefill_layers012_kv_state_loop_probe_json,
    write_prefill_layers012_kvnorm_loop_probe_json, write_prefill_layers01_boundary_probe_json,
    write_prefill_layers01_complete_boundary_probe_json,
    write_prefill_layers01_live_kv_chain_probe_json,
    write_prefill_layers01_live_kv_loop_probe_json, write_prefill_layers01_row_coverage_probe_json,
    write_prefill_q8_boundary_probe_json, write_prefill_qkv_boundary_probe_json, write_probe_json,
    write_projection_probe_json, write_ratio128_compressor_replay_probe_json,
    write_retained_decoder_step_probe_json, write_retained_sparse_boundary_probe_json,
    write_retained_sparse_multimerge_probe_json, write_rope_kv_store_probe_json,
    write_sparse_indexed_attention_probe_json, AttentionOutputProbeReport,
    AttentionReadProbeReport, AttentionSetupProbeReport, ClosedLoopDecoderProbeReport,
    ColdPrefillDecoderProbeReport, DecoderOutputProbeReport, EmbeddingProbeReport,
    FfnRouterProbeReport, IngressProbeReport, Layer0BenchConfig, Layer0BenchReport,
    Layer0ProbeReport, Layers01234567DecodeProbeReport, Layers012345DecodeProbeReport,
    Layers0123BenchConfig, Layers0123BenchReport, Layers0123ChainedProbeReport,
    Layers0123DecodeProbeReport, Layers0123ProbeReport, Layers012ChainedProbeReport,
    Layers012ProbeReport, Layers01ProbeReport, Layers0To42DecodeProbeReport, MoeOutputProbeReport,
    Position127DecoderProbeReport, PrefillFrontierProbeReport, PrefillLayer0BoundaryProbeReport,
    PrefillLayers012AttentionLoopProbeReport, PrefillLayers012CompressorLoopProbeReport,
    PrefillLayers012KvStateLoopProbeReport, PrefillLayers012KvnormLoopProbeReport,
    PrefillLayers01BoundaryProbeReport, PrefillLayers01CompleteBoundaryProbeReport,
    PrefillLayers01LiveKvChainProbeReport, PrefillLayers01LiveKvLoopProbeReport,
    PrefillLayers01RowCoverageProbeReport, PrefillQ8BoundaryProbeReport,
    PrefillQkvBoundaryProbeReport, ProbeConfig, ProjectionProbeReport,
    Ratio128CompressorReplayProbeReport, RetainedDecoderStepProbeReport,
    RetainedSparseBoundaryProbeReport, RopeKvStoreProbeReport, SparseIndexedAttentionProbeReport,
};
use rust_star_runtime::model::MappedModel;
use rust_star_runtime::target::{validate_resident_q2, MODEL_LABEL};
use rust_star_runtime::{Error, Result};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("rust-star: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let command = arguments
        .next()
        .ok_or_else(|| Error::invalid(full_usage()))?;
    if command == "--help" || command == "-h" || command == "help" {
        println!("{}", full_usage());
        return Ok(());
    }
    if command == "--version" || command == "-V" {
        println!("rust-star {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if command == "metal-probe" {
        return run_metal_probe(arguments.collect());
    }
    if command == "embedding-probe" {
        return run_embedding_probe(arguments.collect());
    }
    if command == "projection-probe" {
        return run_projection_probe(arguments.collect());
    }
    if command == "prefill-q8-boundary-probe" {
        return run_prefill_q8_boundary_probe_command(arguments.collect());
    }
    if command == "prefill-qkv-boundary-probe" {
        return run_prefill_qkv_boundary_probe_command(arguments.collect());
    }
    if command == "prefill-layer0-boundary-probe" {
        return run_prefill_layer0_boundary_probe_command(arguments.collect());
    }
    if command == "prefill-layers01-boundary-probe" {
        return run_prefill_layers01_boundary_probe_command(arguments.collect());
    }
    if command == "prefill-layers01-complete-boundary-probe" {
        return run_prefill_layers01_complete_boundary_probe_command(arguments.collect());
    }
    if command == "prefill-layers01-row-coverage-probe" {
        return run_prefill_layers01_row_coverage_probe_command(arguments.collect());
    }
    if command == "prefill-layers01-live-kv-chain-probe" {
        return run_prefill_layers01_live_kv_chain_probe_command(arguments.collect());
    }
    if command == "prefill-layers01-live-kv-loop-probe" {
        return run_prefill_layers01_live_kv_loop_probe_command(arguments.collect());
    }
    if command == "prefill-layers012-kvnorm-loop-probe" {
        return run_prefill_layers012_kvnorm_loop_probe_command(arguments.collect());
    }
    if command == "prefill-layers012-kv-state-loop-probe" {
        return run_prefill_layers012_kv_state_loop_probe_command(arguments.collect());
    }
    if command == "prefill-layers012-compressor-loop-probe" {
        return run_prefill_layers012_compressor_loop_probe_command(arguments.collect());
    }
    if command == "prefill-layers012-attention-loop-probe" {
        return run_prefill_layers012_attention_loop_probe_command(arguments.collect());
    }
    if command == "attention-ingress-probe" {
        return run_ingress_probe(arguments.collect());
    }
    if command == "attention-setup-probe" {
        return run_attention_setup_command(arguments.collect());
    }
    if command == "rope-kv-store-probe" {
        return run_rope_kv_store_command(arguments.collect());
    }
    if command == "attention-read-probe" {
        return run_attention_read_command(arguments.collect());
    }
    if command == "attention-output-probe" {
        return run_attention_output_command(arguments.collect());
    }
    if command == "ffn-router-probe" {
        return run_ffn_router_command(arguments.collect());
    }
    if command == "moe-output-probe" {
        return run_moe_output_command(arguments.collect());
    }
    if command == "layer0-probe" {
        return run_layer0_command(arguments.collect());
    }
    if command == "layer0-bench" {
        return run_layer0_bench_command(arguments.collect());
    }
    if command == "layers01-probe" {
        return run_layers01_command(arguments.collect());
    }
    if command == "layers012-probe" {
        return run_layers012_command(arguments.collect());
    }
    if command == "layers012-chained-probe" {
        return run_layers012_chained_command(arguments.collect());
    }
    if command == "layers0123-probe" {
        return run_layers0123_command(arguments.collect());
    }
    if command == "layers0123-chained-probe" {
        return run_layers0123_chained_command(arguments.collect());
    }
    if command == "layers0123-bench" {
        return run_layers0123_bench_command(arguments.collect());
    }
    if command == "layers0123-decode-probe" {
        return run_layers0123_decode_probe_command(arguments.collect());
    }
    if command == "layers012345-decode-probe" {
        return run_layers012345_decode_probe_command(arguments.collect());
    }
    if command == "layers01234567-decode-probe" {
        return run_layers01234567_decode_probe_command(arguments.collect());
    }
    if command == "layers0-42-decode-probe" {
        return run_layers0_to_42_decode_probe_command(arguments.collect());
    }
    if command == "decoder-output-probe" {
        return run_decoder_output_probe_command(arguments.collect());
    }
    if command == "closed-loop-decoder-probe" {
        return run_closed_loop_decoder_probe_command(arguments.collect());
    }
    if command == "position127-decoder-probe" {
        return run_position127_decoder_probe_command(arguments.collect());
    }
    if command == "cold-prefill-decoder-probe" {
        return run_cold_prefill_decoder_probe_command(arguments.collect());
    }
    if command == "prefill-frontier-probe" {
        return run_prefill_frontier_probe_command(arguments.collect());
    }
    if command == "ratio128-compressor-replay-probe" {
        return run_ratio128_compressor_replay_probe_command(arguments.collect());
    }
    if command == "sparse-indexed-attention-probe" {
        return run_sparse_indexed_attention_probe_command(arguments.collect());
    }
    if command == "retained-sparse-boundary-probe" {
        return run_retained_sparse_boundary_probe_command(arguments.collect());
    }
    if command == "retained-sparse-multimerge-probe" {
        return run_retained_sparse_multimerge_probe_command(arguments.collect());
    }
    if command == "retained-decoder-step-probe" {
        return run_retained_decoder_step_probe_command(arguments.collect());
    }
    run_model_command(&command, arguments.collect())
}

fn run_layers0123_decode_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers0123_decode_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers0123_decode_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers0123_decode_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers0123_decode_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers0123_decode_probe(&model)?;
    println!("position-advancing layers: 0 -> 1 -> 2 -> 3");
    for step in &report.steps {
        println!(
            "position {} token {}: {} cache rows/layer, wall={:.3} ms summed_gpu={:.3} ms, C0 exact",
            step.position, step.token, step.cache_rows, step.wall_ms, step.gpu_ms
        );
    }
    println!(
        "ownership: {} retained per-layer KV caches; {}-element final HC output handoff",
        report.kv_cache_layers, report.output_hc_elements
    );
    if let Some(path) = json_path {
        write_layers0123_decode_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers012345_decode_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers012345_decode_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers012345_decode_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers012345_decode_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers012345_decode_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers012345_decode_probe(&model)?;
    println!("position-advancing layers: 0 -> 1 -> 2 -> 3 -> 4 -> 5");
    for step in &report.steps {
        println!(
            "position {} token {}: {} cache rows/layer, wall={:.3} ms summed_gpu={:.3} ms, C0 exact",
            step.position, step.token, step.cache_rows, step.wall_ms, step.gpu_ms
        );
    }
    println!(
        "ownership: {} retained per-layer KV caches; {}-element final HC output handoff",
        report.kv_cache_layers, report.output_hc_elements
    );
    if let Some(path) = json_path {
        write_layers012345_decode_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers01234567_decode_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers01234567_decode_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers01234567_decode_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers01234567_decode_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers01234567_decode_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers01234567_decode_probe(&model)?;
    println!("position-advancing layers: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7");
    for step in &report.steps {
        println!(
            "position {} token {}: {} cache rows/layer, wall={:.3} ms summed_gpu={:.3} ms, C0 exact",
            step.position, step.token, step.cache_rows, step.wall_ms, step.gpu_ms
        );
    }
    println!(
        "ownership: {} retained per-layer KV caches; {}-element final HC output handoff",
        report.kv_cache_layers, report.output_hc_elements
    );
    if let Some(path) = json_path {
        write_layers01234567_decode_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers0_to_42_decode_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers0_to_42_decode_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers0_to_42_decode_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers0_to_42_decode_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers0_to_42_decode_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers0_to_42_decode_probe(&model)?;
    println!("position-advancing layers: 0 -> ... -> 42");
    for step in &report.steps {
        println!(
            "position {} token {}: {} cache rows/layer, wall={:.3} ms summed_gpu={:.3} ms, C0 exact",
            step.position, step.token, step.cache_rows, step.wall_ms, step.gpu_ms
        );
    }
    println!(
        "ownership: {} retained per-layer KV caches; {}-element final HC output handoff",
        report.kv_cache_layers, report.output_hc_elements
    );
    if let Some(path) = json_path {
        write_layers0_to_42_decode_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_decoder_output_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(decoder_output_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", decoder_output_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", decoder_output_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(decoder_output_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_decoder_output_probe(&model)?;
    println!("decoder output boundary: transformer layers 0 -> 42, output head, full logits");
    for step in &report.steps {
        println!(
            "position {} input token {} -> selected token {}: {} logits, C0 exact",
            step.position,
            step.input_token,
            step.output_head.selected_token,
            report.logits_elements,
        );
    }
    println!(
        "schedule: {} command buffers and {} host waits per correctness step; closed-loop sampling is not yet enabled",
        report.command_buffers_per_step, report.host_waits_per_step,
    );
    if let Some(path) = json_path {
        write_decoder_output_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_closed_loop_decoder_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(closed_loop_decoder_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", closed_loop_decoder_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", closed_loop_decoder_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(closed_loop_decoder_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_closed_loop_decoder_probe(&model)?;
    println!("closed-loop decoder: bootstrap token 201 -> generated 361, 1915, 262, 1554");
    println!("C0: all 43 transformer layers and 129280 logits/position are bit-identical");
    println!(
        "timed diagnostic: {:.3} tok/s complete, {:.3} tok/s steady; correctness readback excluded",
        report.generation_tps, report.steady_tps,
    );
    println!("paired protocol: ineligible captured-state four-position control");
    if let Some(path) = json_path {
        write_closed_loop_decoder_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_position127_decoder_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(position127_decoder_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", position127_decoder_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", position127_decoder_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(position127_decoder_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_position127_decoder_probe(&model)?;
    println!(
        "position-127 decoder frontier: {} committed oracle tokens, final token {}",
        report.committed_tokens.len(),
        report.committed_tokens.last().unwrap_or(&0),
    );
    println!(
        "C0: final full logits and integrated layer-3/layer-5 ratio-128 rows are bit-identical"
    );
    println!(
        "diagnostic execution: {:.3} evaluated positions/s over {} positions",
        report.eval_tps, report.evaluated_positions,
    );
    println!("paired protocol: ineligible captured-state position-127 control");
    if let Some(path) = json_path {
        write_position127_decoder_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_cold_prefill_decoder_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(cold_prefill_decoder_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", cold_prefill_decoder_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", cold_prefill_decoder_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(cold_prefill_decoder_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_cold_prefill_decoder_probe(&model)?;
    println!(
        "cold prefill: prompt token {} -> first committed token {}, full logits C0 exact",
        report.prompt_token,
        report.committed_tokens.first().unwrap_or(&0),
    );
    println!(
        "closed loop: {} committed oracle tokens, final token {}",
        report.committed_tokens.len(),
        report.committed_tokens.last().unwrap_or(&0),
    );
    println!(
        "diagnostic execution: prefill+first selection {:.3} ms; decode {:.3} positions/s",
        report.prefill_wall_ms, report.decode_tps,
    );
    println!("paired protocol: ineligible until native batched prefill and sparse indexed decode");
    if let Some(path) = json_path {
        write_cold_prefill_decoder_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_frontier_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_frontier_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_frontier_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_frontier_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_frontier_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_frontier_probe(&model)?;
    println!(
        "2K sequential initialization: {} canonical tokens -> token {}, decode-replay logits C0 exact",
        report.prompt_tokens, report.selected_token,
    );
    println!(
        "diagnostic execution: {:.3} tokens/s over {:.3} ms",
        report.prefill_tps, report.wall_ms,
    );
    println!(
        "cache ownership: {} raw ring rows, {} ratio-4 compressed rows per layer",
        report.raw_cache_capacity_rows, report.ratio4_compressed_capacity_rows,
    );
    println!(
        "batched-prefill boundary: {} logits differ, max absolute error {:.6}",
        report.batch_logits_mismatch_count, report.batch_logits_max_abs_error,
    );
    println!("paired protocol: ineligible until native batched prefill and sparse indexed decode");
    if let Some(path) = json_path {
        write_prefill_frontier_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_ratio128_compressor_replay_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(ratio128_compressor_replay_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", ratio128_compressor_replay_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", ratio128_compressor_replay_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(ratio128_compressor_replay_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_ratio128_compressor_replay_probe(&model)?;
    println!("ratio-128 compressor activation replay: layers 3 and 5");
    for layer in &report.layers {
        println!(
            "layer {}: {} oracle activation rows, {} dispatches, wall={:.3} ms gpu={:.3} ms, position-127 row C0 exact",
            layer.layer,
            layer.activation_rows,
            layer.dispatches,
            layer.wall_ms,
            layer.gpu_ms,
        );
    }
    println!(
        "boundary: {} externally supplied attn_norm rows; sampling=false; full decoder=false",
        report.externally_supplied_activation_rows
    );
    if let Some(path) = json_path {
        write_ratio128_compressor_replay_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_sparse_indexed_attention_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(sparse_indexed_attention_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", sparse_indexed_attention_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", sparse_indexed_attention_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(sparse_indexed_attention_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_sparse_indexed_attention_probe(&model)?;
    println!(
        "layer-2 position {}: {} compressed rows -> exact top-{}, {}-split indexed attention, C0 exact",
        report.position, report.compressed_rows, report.top_k, report.split_count,
    );
    println!(
        "mapping: {}/{} mmap-backed model ranges preserve pointer identity; {} dispatches",
        report.pointer_matches, report.wrapped_model_ranges, report.dispatches,
    );
    println!(
        "scope: pinned default threshold {} reached at its first sparse row count {}; the 513-row override also passed as an independent control",
        report.pinned_default_threshold, report.first_default_sparse_rows,
    );
    println!("claims: complete-decode=false logits=false throughput=false");
    if let Some(path) = json_path {
        write_sparse_indexed_attention_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_retained_sparse_boundary_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(retained_sparse_boundary_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", retained_sparse_boundary_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", retained_sparse_boundary_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(retained_sparse_boundary_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_retained_sparse_boundary_probe(&model)?;
    println!(
        "retained layer {} position {}: seeded {} prior compressed rows, emitted row {}, and matched {} tensors by bit pattern",
        report.layer,
        report.position,
        report.seeded_compressed_rows,
        report.compressed_rows,
        report.exact_tensor_checks,
    );
    println!(
        "mapping: {}/{} mmap-backed model ranges preserve pointer identity; {} dispatches",
        report.pointer_matches, report.wrapped_model_ranges, report.dispatches,
    );
    println!("claims: retained-layer=true complete-layer=false complete-decoder=false logits=false throughput=false");
    if let Some(path) = json_path {
        write_retained_sparse_boundary_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_retained_sparse_multimerge_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(retained_sparse_multimerge_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", retained_sparse_multimerge_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", retained_sparse_multimerge_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(retained_sparse_multimerge_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_retained_sparse_multimerge_probe(&model)?;
    println!(
        "retained layer {} position {}: {} sort blocks, {} merge passes, row {}, and {} exact tensors",
        report.layer,
        report.position,
        report.sort_blocks,
        report.merge_passes,
        report.compressed_rows,
        report.exact_tensor_checks,
    );
    println!(
        "execution: layers 0-2, {}/{} mmap-backed model ranges preserve pointer identity, {} total dispatches; layer-2 top-k workspace width {}",
        report.total_pointer_matches,
        report.total_wrapped_model_ranges,
        report.total_dispatches,
        report.topk_work_width,
    );
    println!(
        "claims: retained-layer=true repeated-merge=true complete-layer=true preceding-layers=true preceding-history-seeded=true complete-decoder=false logits=false throughput=false"
    );
    if let Some(path) = json_path {
        write_retained_sparse_multimerge_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_retained_decoder_step_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(retained_decoder_step_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", retained_decoder_step_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", retained_decoder_step_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(retained_decoder_step_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_retained_decoder_step_probe(&model)?;
    println!(
        "retained position {}: executed {} layers and selected token {} from exact full logits",
        report.position,
        report.layers.len(),
        report.selected_token,
    );
    println!(
        "mapping: {}/{} mmap-backed model ranges preserve pointer identity; {} transformer dispatches",
        report.total_pointer_matches,
        report.total_wrapped_model_ranges,
        report.total_dispatches,
    );
    println!("claims: complete-decoder-step=true logits=true seeded-history=true native-prefill=false throughput=false");
    if let Some(path) = json_path {
        write_retained_decoder_step_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers0123_bench_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers0123_bench_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers0123_bench_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut config = Layers0123BenchConfig::default();
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--warmup") => {
                config.warmup_iterations =
                    parse_u32_option("--warmup", arguments.next().as_deref())?;
            }
            Some("--iterations") => {
                config.iterations = parse_u32_option("--iterations", arguments.next().as_deref())?;
            }
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers0123_bench_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers0123_bench_usage())),
        }
    }
    let config = config.validate()?;
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers0123_bench(&model, config)?;
    println!(
        "execution: {} warmup + {} measured four-layer chains; {} command buffers and {} tail wait per iteration",
        report.warmup_iterations,
        report.iterations,
        report.command_buffers_per_iteration,
        report.host_waits_per_iteration
    );
    println!(
        "steady state: wall median={:.3} ms MAD={:.3} ms; summed GPU median={:.3} ms MAD={:.3} ms",
        report.wall.median_ms, report.wall.mad_ms, report.gpu.median_ms, report.gpu.mad_ms
    );
    println!(
        "result: final post-measurement readback matched all four pinned DwarfStar layers bit-for-bit"
    );
    if let Some(path) = json_path {
        write_layers0123_bench_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers0123_chained_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers0123_chained_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers0123_chained_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers0123_chained_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers0123_chained_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers0123_chained_probe(&model)?;
    println!("chained layers: 0 -> 1 -> 2 -> 3");
    for (layer_index, layer) in report.layers.iter().enumerate() {
        println!(
            "layer {layer_index}: {} dispatches, experts {:?}, gpu={:.3} ms, C0 exact",
            layer.dispatches, layer.selected_experts, layer.gpu_ms
        );
    }
    println!(
        "scheduler: {} command buffers, {} tail wait, wall={:.3} ms summed_gpu={:.3} ms",
        report.command_buffers, report.host_waits, report.wall_ms, report.gpu_ms
    );
    println!(
        "ownership: {} retained per-layer KV caches; direct HC handoffs",
        report.kv_cache_layers
    );
    if let Some(path) = json_path {
        write_layers0123_chained_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers0123_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers0123_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers0123_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers0123_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers0123_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers0123_probe(&model)?;
    println!("continuous layers: 0 -> 1 -> 2 -> 3");
    for (layer_index, layer) in report.layers.iter().enumerate() {
        println!(
            "layer {layer_index}: {} dispatches, experts {:?}, wall={:.3} ms gpu={:.3} ms, C0 exact",
            layer.dispatches, layer.selected_experts, layer.wall_ms, layer.gpu_ms
        );
    }
    println!(
        "ownership: {} retained per-layer KV caches; direct HC handoffs across {} command buffers",
        report.kv_cache_layers, report.command_buffers
    );
    if let Some(path) = json_path {
        write_layers0123_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers012_chained_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers012_chained_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers012_chained_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers012_chained_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers012_chained_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers012_chained_probe(&model)?;
    println!("chained layers: 0 -> 1 -> 2");
    for (layer_index, layer) in report.layers.iter().enumerate() {
        println!(
            "layer {layer_index}: {} dispatches, experts {:?}, gpu={:.3} ms, C0 exact",
            layer.dispatches, layer.selected_experts, layer.gpu_ms
        );
    }
    println!(
        "scheduler: {} command buffers, {} tail wait, wall={:.3} ms summed_gpu={:.3} ms",
        report.command_buffers, report.host_waits, report.wall_ms, report.gpu_ms
    );
    println!(
        "ownership: {} retained per-layer KV caches; direct HC handoffs",
        report.kv_cache_layers
    );
    if let Some(path) = json_path {
        write_layers012_chained_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers012_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers012_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers012_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers012_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers012_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers012_probe(&model)?;
    println!("continuous layers: 0 -> 1 -> 2");
    for (layer_index, layer) in report.layers.iter().enumerate() {
        println!(
            "layer {layer_index}: {} dispatches, experts {:?}, wall={:.3} ms gpu={:.3} ms, C0 exact",
            layer.dispatches, layer.selected_experts, layer.wall_ms, layer.gpu_ms
        );
    }
    println!(
        "ownership: {} retained per-layer KV caches; direct HC handoffs across {} command buffers",
        report.kv_cache_layers, report.command_buffers
    );
    if let Some(path) = json_path {
        write_layers012_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layers01_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layers01_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layers01_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layers01_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layers01_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layers01_probe(&model)?;
    println!("continuous layers: 0 -> 1");
    for (layer_index, layer) in report.layers.iter().enumerate() {
        println!(
            "layer {layer_index}: {} dispatches, experts {:?}, wall={:.3} ms gpu={:.3} ms, C0 exact",
            layer.dispatches, layer.selected_experts, layer.wall_ms, layer.gpu_ms
        );
    }
    println!(
        "handoff: layer 1 read layer 0's retained HC state directly from Metal memory across {} command buffers",
        report.command_buffers
    );
    if let Some(path) = json_path {
        write_layers01_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layer0_bench_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layer0_bench_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layer0_bench_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut config = Layer0BenchConfig::default();
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--warmup") => {
                config.warmup_iterations =
                    parse_u32_option("--warmup", arguments.next().as_deref())?;
            }
            Some("--iterations") => {
                config.iterations = parse_u32_option("--iterations", arguments.next().as_deref())?;
            }
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layer0_bench_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layer0_bench_usage())),
        }
    }
    let config = config.validate()?;
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layer0_bench(&model, config)?;
    println!("fixture: {}", report.fixture_id);
    println!(
        "persistent setup: {}/{} model views preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!(
        "execution: {} warmup + {} measured iterations; {} dispatches in {} command buffer per iteration",
        report.warmup_iterations,
        report.iterations,
        report.dispatches_per_iteration,
        report.command_buffers_per_iteration
    );
    println!(
        "steady state: wall median={:.3} ms MAD={:.3} ms; gpu median={:.3} ms MAD={:.3} ms",
        report.wall.median_ms, report.wall.mad_ms, report.gpu.median_ms, report.gpu.mad_ms
    );
    println!(
        "result: every measured iteration was bit-identical and the final output matches the pinned DwarfStar layer"
    );
    if let Some(path) = json_path {
        write_layer0_bench_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_layer0_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(layer0_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", layer0_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", layer0_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(layer0_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_layer0_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!(
        "token: {} (complete layer 0, decode position 1)",
        report.token
    );
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!("selected experts: {:?}", report.selected_experts);
    println!(
        "{} dispatches in {} command buffer: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.command_buffers, report.wall_ms, report.gpu_ms
    );
    println!("result: the continuous attention, router, expert, and HC-state chain matches every pinned DwarfStar boundary bit-for-bit");
    if let Some(path) = json_path {
        write_layer0_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_moe_output_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(moe_output_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", moe_output_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", moe_output_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(moe_output_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_moe_output_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!("selected experts: {:?}", report.selected_experts);
    println!(
        "{}-dispatch continuation: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.wall_ms, report.gpu_ms
    );
    println!(
        "result: routed and shared experts plus the FFN HC post-state match DwarfStar bit-for-bit"
    );
    if let Some(path) = json_path {
        write_moe_output_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_ffn_router_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(ffn_router_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", ffn_router_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", ffn_router_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(ffn_router_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_ffn_router_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!("selected experts: {:?}", report.selected_experts);
    println!(
        "{}-dispatch continuation: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.wall_ms, report.gpu_ms
    );
    println!("result: FFN HC ingress, router logits/probabilities, top-k, and scaled weights match DwarfStar bit-for-bit");
    if let Some(path) = json_path {
        write_ffn_router_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_attention_output_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(attention_output_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", attention_output_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", attention_output_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(attention_output_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_attention_output_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!(
        "projection: {} groups with rank {}",
        report.output_groups, report.output_rank
    );
    println!(
        "{}-dispatch chain: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.wall_ms, report.gpu_ms
    );
    println!("result: grouped attention output and fused HC post-state match the pinned DwarfStar boundaries bit-for-bit");
    if let Some(path) = json_path {
        write_attention_output_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_attention_read_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(attention_read_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", attention_read_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", attention_read_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(attention_read_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_attention_read_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!(
        "cache: {} rows read from {} rows; row 0 preserved={} guard row intact={}",
        report.cache_rows_read,
        report.cache_capacity_rows,
        report.cache_row0_preserved,
        report.cache_guard_row_intact
    );
    println!(
        "{}-dispatch chain: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.wall_ms, report.gpu_ms
    );
    println!("result: raw-cache FlashAttention and inverse RoPE match the pinned DwarfStar KQv boundary bit-for-bit");
    if let Some(path) = json_path {
        write_attention_read_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_rope_kv_store_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(rope_kv_store_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", rope_kv_store_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", rope_kv_store_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(rope_kv_store_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_rope_kv_store_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!(
        "cache row: {}/{} with both guard rows intact={}",
        report.cache_target_row, report.cache_capacity_rows, report.cache_guard_rows_intact
    );
    println!(
        "{}-dispatch chain: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.wall_ms, report.gpu_ms
    );
    println!("result: Q/K RoPE, KV FP8 finalization, and the cache-row write match the pinned DwarfStar boundary bit-for-bit");
    if let Some(path) = json_path {
        write_rope_kv_store_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_attention_setup_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(attention_setup_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", attention_setup_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", attention_setup_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(attention_setup_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_attention_setup_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!(
        "{}-dispatch chain: wall={:.3} ms gpu={:.3} ms",
        report.dispatches, report.wall_ms, report.gpu_ms
    );
    println!(
        "result: embedding through Qraw/KVnorm matches every pinned DwarfStar boundary bit-for-bit"
    );
    if let Some(path) = json_path {
        write_attention_setup_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_ingress_probe(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(ingress_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", ingress_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", ingress_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(ingress_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_attention_ingress_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!("token: {} (layer 0, decode position 1)", report.token);
    println!(
        "model views: {}/{} preserve the mmap pointer",
        report.pointer_matches, report.wrapped_model_ranges
    );
    println!(
        "six-dispatch chain: wall={:.3} ms gpu={:.3} ms",
        report.wall_ms, report.gpu_ms
    );
    println!("result: embedding through Q-A projection matches every pinned DwarfStar boundary bit-for-bit");
    if let Some(path) = json_path {
        write_ingress_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_projection_probe(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(projection_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", projection_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help") | Some("-h") => {
                println!("{}", projection_probe_usage());
                return Ok(());
            }
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            _ => return Err(Error::invalid(projection_probe_usage())),
        }
    }

    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let tensor = model.tensor("blk.0.attn_q_a.weight")?;
    let report = run_q8_projection_probe(&model, tensor)?;

    println!("fixture: {}", report.fixture_id);
    println!("kernel: kernel_mul_mv_q8_0_f32 (imported DwarfStar decode matvec)");
    println!(
        "tensor: {} offset={} bytes={}",
        report.tensor_name, report.tensor_offset, report.tensor_bytes
    );
    println!(
        "no-copy view: page_offset={} buffer_bytes={} inner_offset={} pointer_match={}",
        report.page_offset, report.buffer_bytes, report.inner_offset, report.no_copy_pointer_match
    );
    println!(
        "activation: {} FP32 values checksum={}",
        report.input_elements, report.input_checksum
    );
    println!(
        "output: {} FP32 values checksum={}",
        report.output_elements, report.output_checksum
    );
    println!(
        "dispatch: {} simdgroups, {} rows/group, wall={:.3} ms gpu={:.3} ms",
        report.simdgroups, report.rows_per_threadgroup, report.wall_ms, report.gpu_ms
    );
    println!("result: no-copy Q8_0 decode projection matches the DwarfStar fixture bit-for-bit");

    if let Some(path) = json_path {
        write_projection_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_q8_boundary_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_q8_boundary_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_q8_boundary_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_q8_boundary_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_q8_boundary_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_q8_boundary_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!(
        "batch: {} rows through kernel_mul_mm_q8_0_f32, C0 exact",
        report.rows
    );
    println!("decode control: final row through kernel_mul_mv_q8_0_f32, C0 exact");
    println!(
        "dispatch: [{}, 1, 1] threads, [{}, {}, 1] groups; batch wall={:.3} ms gpu={:.3} ms",
        report.batch_threads_per_threadgroup,
        report.batch_threadgroups_x,
        report.batch_threadgroups_y,
        report.batch_wall_ms,
        report.batch_gpu_ms,
    );
    println!(
        "arithmetic boundary: {}/{} final-row values differ, max_abs_error={:.9}",
        report.final_row_mismatches, report.output_elements_per_row, report.final_row_max_abs_error,
    );
    println!("scope: isolated layer-0 projection boundary; no full-prefill claim");
    if let Some(path) = json_path {
        write_prefill_q8_boundary_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_qkv_boundary_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_qkv_boundary_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_qkv_boundary_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_qkv_boundary_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_qkv_boundary_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_qkv_boundary_probe(&model)?;
    println!("fixture: {}", report.fixture_id);
    println!(
        "native batch: {} rows, positions {}..{}, {} dispatches, C0 exact",
        report.rows,
        report.position_start,
        report.position_start + report.rows as u32 - 1,
        report.dispatches,
    );
    println!(
        "mapping: {}/{} no-copy model ranges; wall={:.3} ms gpu={:.3} ms",
        report.pointer_matches, report.wrapped_model_ranges, report.wall_ms, report.gpu_ms,
    );
    println!("scope: layer-0 Q/KV setup through Q head RMSNorm/RoPE; no full-prefill claim");
    if let Some(path) = json_path {
        write_prefill_qkv_boundary_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layer0_boundary_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layer0_boundary_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layer0_boundary_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layer0_boundary_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layer0_boundary_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layer0_boundary_probe(&model)?;
    println!(
        "fixtures: {}, {}, {}, {}, {}, {}",
        report.ingress_fixture_id,
        report.qkv_fixture_id,
        report.kv_state_fixture_id,
        report.attention_fixture_id,
        report.attention_output_fixture_id,
        report.ffn_output_fixture_id,
    );
    println!(
        "native batch: token IDs at positions {}..{} through {} dispatches, C0 exact",
        report.position_start,
        report.position_start + report.rows as u32 - 1,
        report.dispatches,
    );
    println!(
        "mapping: {}/{} no-copy model ranges; wall={:.3} ms gpu={:.3} ms",
        report.pointer_matches, report.wrapped_model_ranges, report.wall_ms, report.gpu_ms,
    );
    println!("scope: continuous final layer-0 tile from token IDs through the FFN HC post-state; no full-prefill claim");
    if let Some(path) = json_path {
        write_prefill_layer0_boundary_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers01_boundary_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layers01_boundary_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers01_boundary_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers01_boundary_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layers01_boundary_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers01_boundary_probe(&model)?;
    println!(
        "fixtures: {}, {}",
        report.layer0.ffn_output_fixture_id, report.layer1_fixture_id,
    );
    println!(
        "native batch: token IDs at positions {}..{} through {} dispatches, C0 exact",
        report.layer0.position_start,
        report.layer0.position_start + report.layer0.rows as u32 - 1,
        report.layer0.dispatches,
    );
    println!(
        "mapping: {}/{} no-copy model ranges; wall={:.3} ms gpu={:.3} ms",
        report.layer0.pointer_matches,
        report.layer0.wrapped_model_ranges,
        report.layer0.wall_ms,
        report.layer0.gpu_ms,
    );
    println!(
        "scope: direct layer-0 FFN HC to layer-1 Q-Lora handoff; no complete-layer-1 or full-prefill claim"
    );
    if let Some(path) = json_path {
        write_prefill_layers01_boundary_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers01_complete_boundary_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(
            prefill_layers01_complete_boundary_probe_usage(),
        ));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers01_complete_boundary_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers01_complete_boundary_probe_usage());
                return Ok(());
            }
            _ => {
                return Err(Error::invalid(
                    prefill_layers01_complete_boundary_probe_usage(),
                ))
            }
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers01_complete_boundary_probe(&model)?;
    println!(
        "fixtures: {}, {}",
        report.layers01.layer1_fixture_id, report.complete_fixture_id,
    );
    println!(
        "native batch: token IDs at positions {}..{} through {} dispatches, C0 exact",
        report.layers01.layer0.position_start,
        report.layers01.layer0.position_start + report.layers01.layer0.rows as u32 - 1,
        report.layers01.layer0.dispatches,
    );
    println!(
        "mapping: {}/{} no-copy model ranges; wall={:.3} ms gpu={:.3} ms",
        report.layers01.layer0.pointer_matches,
        report.layers01.layer0.wrapped_model_ranges,
        report.layers01.layer0.wall_ms,
        report.layers01.layer0.gpu_ms,
    );
    println!(
        "scope: direct token-ID through complete layer-1 FFN HC final tile; no full-prefill claim"
    );
    if let Some(path) = json_path {
        write_prefill_layers01_complete_boundary_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers01_row_coverage_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layers01_row_coverage_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers01_row_coverage_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers01_row_coverage_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layers01_row_coverage_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers01_row_coverage_probe(&model)?;
    println!(
        "native coverage: positions {}..{} across two complete 32-row layers-0/1 tiles, C0 exact",
        report.previous_position_start,
        report.final_tile.layers01.layer0.position_start
            + report.final_tile.layers01.layer0.rows as u32
            - 1,
    );
    println!(
        "previous tile: {}/{} no-copy model ranges; wall={:.3} ms gpu={:.3} ms",
        report.previous_pointer_matches,
        report.previous_wrapped_model_ranges,
        report.previous_wall_ms,
        report.previous_gpu_ms,
    );
    println!(
        "final tile: {}/{} no-copy model ranges; wall={:.3} ms gpu={:.3} ms",
        report.final_tile.layers01.layer0.pointer_matches,
        report.final_tile.layers01.layer0.wrapped_model_ranges,
        report.final_tile.layers01.layer0.wall_ms,
        report.final_tile.layers01.layer0.gpu_ms,
    );
    println!("scope: arbitrary-position exact tile replay with captured per-tile KV prefixes; no live inter-tile KV chain or full-prefill claim");
    if let Some(path) = json_path {
        write_prefill_layers01_row_coverage_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers01_live_kv_chain_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layers01_live_kv_chain_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers01_live_kv_chain_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers01_live_kv_chain_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layers01_live_kv_chain_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers01_live_kv_chain_probe(&model)?;
    let tiles = &report.tiles;
    println!(
        "live KV chain: positions {}..{} across two complete 32-row layers-0/1 tiles, C0 exact",
        tiles.previous_position_start,
        tiles.final_tile.layers01.layer0.position_start
            + tiles.final_tile.layers01.layer0.rows as u32
            - 1,
    );
    println!(
        "retained state: {} rows after tile one, {} rows after tile two; one persistent Metal context",
        report.retained_kv_rows_after_first_tile,
        report.retained_kv_rows_after_final_tile,
    );
    println!(
        "mapping: tile one {}/{} and tile two {}/{} no-copy model ranges",
        tiles.previous_pointer_matches,
        tiles.previous_wrapped_model_ranges,
        tiles.final_tile.layers01.layer0.pointer_matches,
        tiles.final_tile.layers01.layer0.wrapped_model_ranges,
    );
    println!("scope: live layer-0/layer-1 KV append and consume across two command buffers with one inter-tile host wait; no single-command-buffer or full-prefill claim");
    if let Some(path) = json_path {
        write_prefill_layers01_live_kv_chain_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers01_live_kv_loop_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layers01_live_kv_loop_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers01_live_kv_loop_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers01_live_kv_loop_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layers01_live_kv_loop_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers01_live_kv_loop_probe(&model)?;
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    println!("live KV loop: positions 0..2047 across 64 complete 32-row layers-0/1 schedules");
    println!(
        "KV ownership: empty seed -> 2048 retained rows/layer, every accumulated prefix C0 exact"
    );
    println!(
        "mapping: every tile preserved 49/49 no-copy model ranges; summed wall={total_wall_ms:.3} ms gpu={total_gpu_ms:.3} ms"
    );
    println!("scope: exact full-2K layer-0/layer-1 KV chain and exact final tile; non-final full outputs are not all retained, no complete-model-prefill or throughput claim");
    if let Some(path) = json_path {
        write_prefill_layers01_live_kv_loop_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers012_kvnorm_loop_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layers012_kvnorm_loop_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers012_kvnorm_loop_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers012_kvnorm_loop_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layers012_kvnorm_loop_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers012_kvnorm_loop_probe(&model)?;
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    println!(
        "native layer-2 KVnorm loop: positions 0..2047 across 64 complete layers-0/1 schedules"
    );
    println!("downstream validation: every live layer-1 output row matched the captured layer-2 KVnorm boundary bit-for-bit");
    println!("mapping: every tile preserved 57/57 no-copy model ranges; 90 dispatches/tile; summed wall={total_wall_ms:.3} ms gpu={total_gpu_ms:.3} ms");
    println!("scope: complete native layers 0/1 plus layer-2 HC ingress and normalized Q/KV projections; no compressed layer-2 attention, complete-model-prefill, or throughput claim");
    if let Some(path) = json_path {
        write_prefill_layers012_kvnorm_loop_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers012_kv_state_loop_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(prefill_layers012_kv_state_loop_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers012_kv_state_loop_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers012_kv_state_loop_probe_usage());
                return Ok(());
            }
            _ => return Err(Error::invalid(prefill_layers012_kv_state_loop_probe_usage())),
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers012_kv_state_loop_probe(&model)?;
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    println!(
        "native layer-2 KV-state loop: positions 0..2047 across 64 complete layers-0/1 schedules"
    );
    println!("layer-2 ownership: empty seed -> 2048 exact KV rows after compressed RoPE and E4M3FN finalization");
    println!("mapping: every tile preserved 57/57 no-copy model ranges; 92 dispatches/tile; summed wall={total_wall_ms:.3} ms gpu={total_gpu_ms:.3} ms");
    println!("scope: complete native layers 0/1 plus exact layer-2 normalized, rotated, and finalized raw KV state; no layer-2 compressor, attention, FFN, complete-model-prefill, or throughput claim");
    if let Some(path) = json_path {
        write_prefill_layers012_kv_state_loop_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers012_compressor_loop_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(
            prefill_layers012_compressor_loop_probe_usage(),
        ));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers012_compressor_loop_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers012_compressor_loop_probe_usage());
                return Ok(());
            }
            _ => {
                return Err(Error::invalid(
                    prefill_layers012_compressor_loop_probe_usage(),
                ));
            }
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers012_compressor_loop_probe(&model)?;
    let total_wall_ms: f64 = report.tiles.iter().map(|tile| tile.wall_ms).sum();
    let total_gpu_ms: f64 = report.tiles.iter().map(|tile| tile.gpu_ms).sum();
    println!(
        "native layer-2 paired-compressor loop: positions 0..2047 across 64 complete layers-0/1 schedules"
    );
    println!("compressor ownership: empty seed -> 512 exact attention and indexer compressed rows plus exact final recurrent states");
    println!("mapping: every tile preserved 65/65 no-copy model ranges; 118 dispatches/regular tile and 122 on the final tail refresh; summed wall={total_wall_ms:.3} ms gpu={total_gpu_ms:.3} ms");
    println!("scope: complete native layers 0/1 plus exact layer-2 raw KV and paired ratio-4 compressors; no layer-2 mixed attention, FFN, complete-model-prefill, or throughput claim");
    if let Some(path) = json_path {
        write_prefill_layers012_compressor_loop_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_prefill_layers012_attention_loop_probe_command(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(
            prefill_layers012_attention_loop_probe_usage(),
        ));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", prefill_layers012_attention_loop_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--help") | Some("-h") => {
                println!("{}", prefill_layers012_attention_loop_probe_usage());
                return Ok(());
            }
            _ => {
                return Err(Error::invalid(
                    prefill_layers012_attention_loop_probe_usage(),
                ));
            }
        }
    }
    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let report = run_prefill_layers012_attention_loop_probe(&model)?;
    println!("native complete layers 0-31 at 2K: 2048 raw rows + 512 dense layer-2/layer-4/layer-6/layer-8/layer-10/layer-12/layer-14/layer-16/layer-18/layer-20/layer-22/layer-24/layer-26/layer-28/layer-30 compressed rows + 16 layer-3/layer-5/layer-7/layer-9/layer-11/layer-13/layer-15/layer-17/layer-19/layer-21/layer-23/layer-25/layer-27/layer-29/layer-31 compressed rows");
    println!(
        "terminal attention/FFN schedule: {} dispatches, {}/{} no-copy model ranges, wall={:.3} ms gpu={:.3} ms",
        report.dispatches,
        report.pointer_matches,
        report.wrapped_model_ranges,
        report.wall_ms,
        report.gpu_ms,
    );
    println!("scope: complete native layers 0-31, including exact paired ratio-4 and ratio-128 compressors, dense mixed attention, biased top-6 routed/shared FFNs, and additive final HC updates at the 2K prompt boundary; no layer-32 prefill, sparse post-prompt top-k, complete-model-prefill, output-logit, or throughput claim");
    if let Some(path) = json_path {
        write_prefill_layers012_attention_loop_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn run_embedding_probe(arguments: Vec<OsString>) -> Result<()> {
    if arguments.is_empty() {
        return Err(Error::invalid(embedding_probe_usage()));
    }
    if matches!(arguments[0].to_str(), Some("--help") | Some("-h")) {
        println!("{}", embedding_probe_usage());
        return Ok(());
    }
    let model_path = PathBuf::from(&arguments[0]);
    let mut json_path: Option<PathBuf> = None;
    let mut requested_tokens: Option<Vec<u32>> = None;
    let mut arguments = arguments.into_iter().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help") | Some("-h") => {
                println!("{}", embedding_probe_usage());
                return Ok(());
            }
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            Some("--tokens") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--tokens requires a comma-separated list"))?;
                if requested_tokens.is_some() {
                    return Err(Error::invalid("--tokens may be specified only once"));
                }
                requested_tokens = Some(parse_tokens(&value)?);
            }
            _ => return Err(Error::invalid(embedding_probe_usage())),
        }
    }

    let model = MappedModel::open(&model_path)?;
    validate_resident_q2(model.gguf())?;
    let tensor = model.tensor("token_embd.weight")?;
    let n_vocab = u32::try_from(tensor.dimensions[1])
        .map_err(|_| Error::invalid("embedding vocabulary exceeds u32"))?;
    let tokens =
        requested_tokens.unwrap_or_else(|| vec![0, 1, 17, n_vocab / 2, n_vocab.saturating_sub(1)]);
    let report = run_f16_embedding_probe(&model, tensor, &tokens)?;

    println!("kernel: kernel_get_rows_f16 (imported DwarfStar embedding gather)");
    println!(
        "tensor: {} offset={} bytes={}",
        report.tensor_name, report.tensor_offset, report.tensor_bytes
    );
    println!(
        "no-copy view: page_offset={} buffer_bytes={} inner_offset={} pointer_match={}",
        report.page_offset, report.buffer_bytes, report.inner_offset, report.no_copy_pointer_match
    );
    println!(
        "tokens: {:?}, output={} FP32 values checksum={}",
        report.tokens, report.output_elements, report.checksum
    );
    println!(
        "dispatch: wall={:.3} ms gpu={:.3} ms",
        report.wall_ms, report.gpu_ms
    );
    println!("result: no-copy F16 embedding gather is C0 bit-identical");

    if let Some(path) = json_path {
        write_embedding_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn parse_tokens(value: &OsStr) -> Result<Vec<u32>> {
    let text = value
        .to_str()
        .ok_or_else(|| Error::invalid("--tokens must be UTF-8"))?;
    let mut tokens = Vec::new();
    for item in text.split(',') {
        if item.is_empty() {
            return Err(Error::invalid("--tokens contains an empty item"));
        }
        let token = item
            .parse::<u32>()
            .map_err(|_| Error::invalid("--tokens values must be unsigned integers"))?;
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err(Error::invalid("--tokens must not be empty"));
    }
    Ok(tokens)
}

fn run_model_command(command: &OsStr, arguments: Vec<OsString>) -> Result<()> {
    if arguments.len() != 1 {
        return Err(Error::invalid(usage()));
    }
    let model = &arguments[0];
    let model = Path::new(model);
    let file = File::open(model).map_err(|error| {
        Error::invalid(format!("cannot open model {}: {error}", model.display()))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let gguf = Gguf::parse(&mut reader)?;

    println!("file: {}", model.display());
    println!(
        "gguf: v{}, {} metadata keys, {} tensors, {} bytes",
        gguf.version,
        gguf.metadata.len(),
        gguf.tensors.len(),
        gguf.file_bytes
    );
    println!(
        "layout: alignment={} tensor_data_offset={}",
        gguf.alignment, gguf.tensor_data_offset
    );

    if command == "gguf" {
        print_type_counts(&gguf);
        println!("result: structurally valid GGUF v3 directory");
        return Ok(());
    }
    if command != "inspect" {
        return Err(Error::invalid(usage()));
    }

    let report = validate_resident_q2(&gguf)?;
    println!("target: {MODEL_LABEL} resident imatrix Q2");
    println!(
        "model metadata name: {}",
        report.model_name.as_deref().unwrap_or("<missing>")
    );
    println!(
        "tensor contract: {} required tensors validated, {} additional tensors",
        report.required_tensor_count, report.extra_tensor_count
    );
    println!("imatrix entries: {}", report.imatrix_entries);
    println!("tensor types:");
    for (tensor_type, count) in &report.tensor_type_counts {
        println!("  {tensor_type:<8} {count}");
    }
    if !report.checkpoint_name_mentions_0731 {
        println!(
            "checkpoint note: metadata name does not assert 0731; this is not a failure because the oracle model SHA-256 is authoritative"
        );
    }
    if report.model_sha_required_for_checkpoint_identity {
        println!(
            "checkpoint identity: pending comparison with the completed oracle-v1 model SHA-256"
        );
    }
    println!("result: target shape and Q2 recipe valid");
    Ok(())
}

fn run_metal_probe(arguments: Vec<OsString>) -> Result<()> {
    let mut config = ProbeConfig::default();
    let mut json_path: Option<PathBuf> = None;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help") | Some("-h") => {
                println!("{}", metal_probe_usage());
                return Ok(());
            }
            Some("--elements") => {
                config.elements = parse_u64_option("--elements", arguments.next().as_deref())?;
            }
            Some("--iterations") => {
                config.iterations = parse_u64_option("--iterations", arguments.next().as_deref())?;
            }
            Some("--json") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::invalid("--json requires a path"))?;
                if json_path.is_some() {
                    return Err(Error::invalid("--json may be specified only once"));
                }
                json_path = Some(PathBuf::from(value));
            }
            _ => return Err(Error::invalid(metal_probe_usage())),
        }
    }

    let report = run_probe(config.validate()?)?;
    println!("device: {}", report.device_name);
    println!("unified memory: {}", report.has_unified_memory);
    println!(
        "recommended working set: {} bytes",
        report.recommended_max_working_set_bytes
    );
    println!(
        "pipeline: max_threads={} setup={:.3} ms compile={:.3} ms",
        report.max_total_threads_per_threadgroup, report.setup_ms, report.compile_ms
    );
    println!(
        "warmup: wall={:.3} ms gpu={:.3} ms",
        report.warmup_wall_ms, report.warmup_gpu_ms
    );
    println!(
        "roundtrip: {} dispatches wall={:.3} ms gpu={:.3} ms rate={:.1} dispatches/s",
        report.iterations,
        report.roundtrip_wall_ms,
        report.roundtrip_gpu_ms,
        report.roundtrip_dispatches_per_second()
    );
    println!(
        "batched: {} dispatches wall={:.3} ms gpu={:.3} ms rate={:.1} dispatches/s",
        report.iterations,
        report.batched_wall_ms,
        report.batched_gpu_ms,
        report.batched_dispatches_per_second()
    );
    println!(
        "batching ratio: {:.3}x",
        report.batched_dispatches_per_second() / report.roundtrip_dispatches_per_second()
    );
    println!("result: Metal dispatch and shared-buffer validation passed");

    if let Some(path) = json_path {
        write_probe_file(&path, &report)?;
        println!("json: {}", path.display());
    }
    Ok(())
}

fn parse_u64_option(option: &str, value: Option<&OsStr>) -> Result<u64> {
    let value = value.ok_or_else(|| Error::invalid(format!("{option} requires a value")))?;
    let text = value
        .to_str()
        .ok_or_else(|| Error::invalid(format!("{option} must be UTF-8")))?;
    text.parse::<u64>()
        .map_err(|_| Error::invalid(format!("{option} must be an unsigned integer")))
}

fn parse_u32_option(option: &str, value: Option<&OsStr>) -> Result<u32> {
    let parsed = parse_u64_option(option, value)?;
    u32::try_from(parsed)
        .map_err(|_| Error::invalid(format!("{option} exceeds the supported integer range")))
}

fn write_probe_file(path: &Path, report: &rust_star_runtime::metal::ProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_embedding_probe_file(path: &Path, report: &EmbeddingProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create embedding probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_embedding_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install embedding probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_projection_probe_file(path: &Path, report: &ProjectionProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create projection probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_projection_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install projection probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_q8_boundary_probe_file(
    path: &Path,
    report: &PrefillQ8BoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill Q8 boundary JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_q8_boundary_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill Q8 boundary JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_qkv_boundary_probe_file(
    path: &Path,
    report: &PrefillQkvBoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill Q/KV boundary JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_qkv_boundary_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill Q/KV boundary JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layer0_boundary_probe_file(
    path: &Path,
    report: &PrefillLayer0BoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layer-0 boundary JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layer0_boundary_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layer-0 boundary JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers01_boundary_probe_file(
    path: &Path,
    report: &PrefillLayers01BoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1 boundary JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers01_boundary_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1 boundary JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers01_complete_boundary_probe_file(
    path: &Path,
    report: &PrefillLayers01CompleteBoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create complete prefill layers-0/1 boundary JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers01_complete_boundary_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install complete prefill layers-0/1 boundary JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers01_row_coverage_probe_file(
    path: &Path,
    report: &PrefillLayers01RowCoverageProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1 row-coverage JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers01_row_coverage_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1 row-coverage JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers01_live_kv_chain_probe_file(
    path: &Path,
    report: &PrefillLayers01LiveKvChainProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1 live-KV-chain JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers01_live_kv_chain_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1 live-KV-chain JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers01_live_kv_loop_probe_file(
    path: &Path,
    report: &PrefillLayers01LiveKvLoopProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1 live-KV-loop JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers01_live_kv_loop_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1 live-KV-loop JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers012_kvnorm_loop_probe_file(
    path: &Path,
    report: &PrefillLayers012KvnormLoopProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1/2 KVnorm-loop JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers012_kvnorm_loop_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1/2 KVnorm-loop JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers012_kv_state_loop_probe_file(
    path: &Path,
    report: &PrefillLayers012KvStateLoopProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1/2 KV-state-loop JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers012_kv_state_loop_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1/2 KV-state-loop JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers012_compressor_loop_probe_file(
    path: &Path,
    report: &PrefillLayers012CompressorLoopProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1/2 compressor-loop JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers012_compressor_loop_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1/2 compressor-loop JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_layers012_attention_loop_probe_file(
    path: &Path,
    report: &PrefillLayers012AttentionLoopProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create prefill layers-0/1/2 attention-loop JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_layers012_attention_loop_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install prefill layers-0/1/2 attention-loop JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_ingress_probe_file(path: &Path, report: &IngressProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create ingress probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_ingress_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install ingress probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_attention_setup_probe_file(path: &Path, report: &AttentionSetupProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create attention setup probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_attention_setup_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install attention setup probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_rope_kv_store_probe_file(path: &Path, report: &RopeKvStoreProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create RoPE/KV-store probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_rope_kv_store_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install RoPE/KV-store probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_attention_read_probe_file(path: &Path, report: &AttentionReadProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create attention-read probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_attention_read_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install attention-read probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_attention_output_probe_file(
    path: &Path,
    report: &AttentionOutputProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create attention-output probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_attention_output_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install attention-output probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_ffn_router_probe_file(path: &Path, report: &FfnRouterProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create FFN router probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_ffn_router_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install FFN router probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layer0_probe_file(path: &Path, report: &Layer0ProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create complete layer-0 probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layer0_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install complete layer-0 probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layer0_bench_file(path: &Path, report: &Layer0BenchReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create layer-0 steady-state JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layer0_bench_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install layer-0 steady-state JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers0123_bench_file(path: &Path, report: &Layers0123BenchReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create layers-0/1/2/3 steady-state JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers0123_bench_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install layers-0/1/2/3 steady-state JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers0123_decode_probe_file(
    path: &Path,
    report: &Layers0123DecodeProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create position-advancing JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers0123_decode_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install position-advancing JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers012345_decode_probe_file(
    path: &Path,
    report: &Layers012345DecodeProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create six-layer position-advancing JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers012345_decode_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install six-layer position-advancing JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers01234567_decode_probe_file(
    path: &Path,
    report: &Layers01234567DecodeProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create eight-layer position-advancing JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers01234567_decode_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install eight-layer position-advancing JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers0_to_42_decode_probe_file(
    path: &Path,
    report: &Layers0To42DecodeProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create forty-three-layer position-advancing JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers0_to_42_decode_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install forty-three-layer position-advancing JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_decoder_output_probe_file(path: &Path, report: &DecoderOutputProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create decoder-output JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_decoder_output_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install decoder-output JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_closed_loop_decoder_probe_file(
    path: &Path,
    report: &ClosedLoopDecoderProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create closed-loop decoder JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_closed_loop_decoder_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install closed-loop decoder JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_position127_decoder_probe_file(
    path: &Path,
    report: &Position127DecoderProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create position-127 decoder JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_position127_decoder_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install position-127 decoder JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_cold_prefill_decoder_probe_file(
    path: &Path,
    report: &ColdPrefillDecoderProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create cold-prefill decoder JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_cold_prefill_decoder_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install cold-prefill decoder JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_prefill_frontier_probe_file(
    path: &Path,
    report: &PrefillFrontierProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create 2K prefill frontier JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_prefill_frontier_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install 2K prefill frontier JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_ratio128_compressor_replay_probe_file(
    path: &Path,
    report: &Ratio128CompressorReplayProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create ratio-128 compressor replay JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_ratio128_compressor_replay_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install ratio-128 compressor replay JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_sparse_indexed_attention_probe_file(
    path: &Path,
    report: &SparseIndexedAttentionProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create sparse indexed-attention JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_sparse_indexed_attention_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install sparse indexed-attention JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_retained_sparse_boundary_probe_file(
    path: &Path,
    report: &RetainedSparseBoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create retained sparse-boundary JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_retained_sparse_boundary_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install retained sparse-boundary JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_retained_sparse_multimerge_probe_file(
    path: &Path,
    report: &RetainedSparseBoundaryProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create retained sparse multimerge JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_retained_sparse_multimerge_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install retained sparse multimerge JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_retained_decoder_step_probe_file(
    path: &Path,
    report: &RetainedDecoderStepProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create retained decoder-step JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_retained_decoder_step_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install retained decoder-step JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers01_probe_file(path: &Path, report: &Layers01ProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create layers-0/1 probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers01_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install layers-0/1 probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers012_probe_file(path: &Path, report: &Layers012ProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create layers-0/1/2 probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers012_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install layers-0/1/2 probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers012_chained_probe_file(
    path: &Path,
    report: &Layers012ChainedProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create chained layers-0/1/2 probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers012_chained_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install chained layers-0/1/2 probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers0123_probe_file(path: &Path, report: &Layers0123ProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create layers-0/1/2/3 probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers0123_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install layers-0/1/2/3 probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_layers0123_chained_probe_file(
    path: &Path,
    report: &Layers0123ChainedProbeReport,
) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create chained layers-0/1/2/3 probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_layers0123_chained_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install chained layers-0/1/2/3 probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn write_moe_output_probe_file(path: &Path, report: &MoeOutputProbeReport) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(OsStr::to_str)
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    let file = File::create(&temporary).map_err(|error| {
        Error::invalid(format!(
            "cannot create MoE output probe JSON {}: {error}",
            temporary.display()
        ))
    })?;
    let mut output = BufWriter::new(file);
    write_moe_output_probe_json(&mut output, report)?;
    output.flush()?;
    drop(output);
    std::fs::rename(&temporary, path).map_err(|error| {
        Error::invalid(format!(
            "cannot install MoE output probe JSON {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn print_type_counts(gguf: &Gguf) {
    let mut counts = std::collections::BTreeMap::new();
    for tensor in gguf.tensors.values() {
        *counts.entry(tensor.tensor_type.name).or_insert(0_usize) += 1;
    }
    println!("tensor types:");
    for (tensor_type, count) in counts {
        println!("  {tensor_type:<8} {count}");
    }
}

fn usage() -> &'static str {
    "usage:\n  rust-star inspect MODEL.gguf  # strict Flash-0731 resident-Q2 validation\n  rust-star gguf MODEL.gguf     # structural GGUF v3 validation only\n  rust-star metal-probe [OPTIONS]\n  rust-star embedding-probe MODEL.gguf [OPTIONS]\n  rust-star projection-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-q8-boundary-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-qkv-boundary-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layer0-boundary-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers01-boundary-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers01-complete-boundary-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers01-row-coverage-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers01-live-kv-chain-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers01-live-kv-loop-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers012-kvnorm-loop-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers012-kv-state-loop-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers012-compressor-loop-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-layers012-attention-loop-probe MODEL.gguf [OPTIONS]\n  rust-star attention-ingress-probe MODEL.gguf [OPTIONS]\n  rust-star attention-setup-probe MODEL.gguf [OPTIONS]\n  rust-star rope-kv-store-probe MODEL.gguf [OPTIONS]\n  rust-star attention-read-probe MODEL.gguf [OPTIONS]\n  rust-star attention-output-probe MODEL.gguf [OPTIONS]\n  rust-star ffn-router-probe MODEL.gguf [OPTIONS]\n  rust-star moe-output-probe MODEL.gguf [OPTIONS]\n  rust-star layer0-probe MODEL.gguf [OPTIONS]\n  rust-star layer0-bench MODEL.gguf [OPTIONS]\n  rust-star layers01-probe MODEL.gguf [OPTIONS]\n  rust-star layers012-probe MODEL.gguf [OPTIONS]\n  rust-star layers012-chained-probe MODEL.gguf [OPTIONS]\n  rust-star layers0123-probe MODEL.gguf [OPTIONS]\n  rust-star layers0123-chained-probe MODEL.gguf [OPTIONS]\n  rust-star layers0123-bench MODEL.gguf [OPTIONS]\n  rust-star layers0123-decode-probe MODEL.gguf [OPTIONS]\n  rust-star layers012345-decode-probe MODEL.gguf [OPTIONS]\n  rust-star layers01234567-decode-probe MODEL.gguf [OPTIONS]\n  rust-star layers0-42-decode-probe MODEL.gguf [OPTIONS]\n  rust-star decoder-output-probe MODEL.gguf [OPTIONS]\n  rust-star closed-loop-decoder-probe MODEL.gguf [OPTIONS]\n  rust-star position127-decoder-probe MODEL.gguf [OPTIONS]\n  rust-star cold-prefill-decoder-probe MODEL.gguf [OPTIONS]\n  rust-star prefill-frontier-probe MODEL.gguf [OPTIONS]\n  rust-star ratio128-compressor-replay-probe MODEL.gguf [OPTIONS]\n  rust-star sparse-indexed-attention-probe MODEL.gguf [OPTIONS]\n  rust-star retained-sparse-boundary-probe MODEL.gguf [OPTIONS]"
}

fn full_usage() -> String {
    format!(
        "{}\n  rust-star retained-sparse-multimerge-probe MODEL.gguf [OPTIONS]\n  rust-star retained-decoder-step-probe MODEL.gguf [OPTIONS]",
        usage()
    )
}

fn metal_probe_usage() -> &'static str {
    "usage: rust-star metal-probe [--elements N] [--iterations N] [--json PATH]\n\nCompares synchronized per-dispatch command buffers with one batched command buffer."
}

fn embedding_probe_usage() -> &'static str {
    "usage: rust-star embedding-probe MODEL.gguf [--tokens ID,ID,...] [--json PATH]\n\nWraps the F16 token embedding as a page-aligned bytes-no-copy Metal buffer and validates DwarfStar kernel_get_rows_f16 bit-for-bit."
}

fn projection_probe_usage() -> &'static str {
    "usage: rust-star projection-probe MODEL.gguf [--json PATH]\n\nWraps blk.0.attn_q_a.weight without copying and requires DwarfStar kernel_mul_mv_q8_0_f32 to reproduce a pinned layer-0 decode fixture bit-for-bit."
}

fn prefill_q8_boundary_probe_usage() -> &'static str {
    "usage: rust-star prefill-q8-boundary-probe MODEL.gguf [--json PATH]\n\nRuns DwarfStar's native M1 Q8 prefill matmul over the captured final 128-row layer-0 tile, requires a bitwise oracle match, and independently checks the final row with the sequential decode kernel. This is an isolated arithmetic-boundary probe, not a full-prefill claim."
}

fn prefill_qkv_boundary_probe_usage() -> &'static str {
    "usage: rust-star prefill-qkv-boundary-probe MODEL.gguf [--json PATH]\n\nRuns the final 32-row native M1 prefill tile from layer-0 attention normalization through Q-Lora, KV, fused Q/KV RMSNorm, Q-B, and Q head RMSNorm/RoPE. Every retained boundary must match repeated DwarfStar captures bit-for-bit; this is not a full-prefill claim."
}

fn prefill_layer0_boundary_probe_usage() -> &'static str {
    "usage: rust-star prefill-layer0-boundary-probe MODEL.gguf [--json PATH]\n\nRuns the final 32-row native M1 prefill tile continuously from token IDs through embedding, four-stream HC ingress, Q/KV setup, guarded raw-cache storage, zero-prefix batched FlashAttention, grouped Q8 attention output, FFN routing, routed and shared experts, and the additive FFN HC post-update. Every retained boundary must match repeated DwarfStar captures bit-for-bit; this is not a full-prefill claim."
}

fn prefill_layers01_boundary_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers01-boundary-probe MODEL.gguf [--json PATH]\n\nRuns the exact final 32-row native M1 layer-0 tile and hands its live four-stream FFN HC state directly into layer 1's HC ingress, learned attention norm, and Q-A projection. Every retained boundary must match repeated DwarfStar captures bit-for-bit; this is not a complete layer-1 or full-prefill claim."
}

fn prefill_layers01_complete_boundary_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers01-complete-boundary-probe MODEL.gguf [--json PATH]\n\nRuns the exact final 32-row native M1 tile continuously from token IDs through complete layers 0 and 1, ending at layer 1's additive FFN HC post-state. Every retained boundary must match repeated DwarfStar captures bit-for-bit; this is not a full-prefill claim."
}

fn prefill_layers01_row_coverage_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers01-row-coverage-probe MODEL.gguf [--json PATH]\n\nRuns complete layers 0 and 1 over positions 1984--2015 and the preserved final tile at 2016--2047. Both native 32-row tiles must match repeated DwarfStar captures bit-for-bit with direct mmap-backed model views. Each tile uses its captured KV prefix; this is an arbitrary-position tile-coverage checkpoint, not a live inter-tile KV chain or full-prefill claim."
}

fn prefill_layers01_live_kv_chain_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers01-live-kv-chain-probe MODEL.gguf [--json PATH]\n\nRuns complete layers 0 and 1 over positions 1984--2047 in one persistent Metal context. The first exact tile seeds and retains both layers' KV buffers; the second exact tile appends to and consumes those live buffers without assembling its execution prefix from a capture. Both tiles and the retained prefix must match the DwarfStar oracle bit-for-bit. This uses two command buffers with one inter-tile host wait and is not a full-prefill claim."
}

fn prefill_layers01_live_kv_loop_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers01-live-kv-loop-probe MODEL.gguf [--json PATH]\n\nRuns all 64 native 32-row layers-0/1 schedules over positions 0--2047 in one persistent Metal context. It starts with empty KV buffers, validates every accumulated layer-0/layer-1 prefix against the DwarfStar oracle before each continuation, and retains an exact final 2K KV state. The final tile keeps its exhaustive output comparison; non-final full outputs are not all retained. This is not a complete-model-prefill or throughput claim."
}

fn prefill_layers012_kvnorm_loop_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers012-kvnorm-loop-probe MODEL.gguf [--json PATH]\n\nRuns all 64 native 32-row schedules over positions 0--2047 in one persistent Metal context. Layers 0 and 1 execute completely with live KV ownership; each resulting layer-1 tile feeds native layer-2 HC ingress, attention normalization, Q-A/KV projections, and fused QKV normalization. Every layer-2 KVnorm tile is compared bit-for-bit with the captured DwarfStar oracle. This does not claim compressed layer-2 attention, complete-model prefill, or throughput."
}

fn prefill_layers012_kv_state_loop_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers012-kv-state-loop-probe MODEL.gguf [--json PATH]\n\nRuns all 64 native 32-row schedules over positions 0--2047 in one persistent Metal context. Complete live layers 0 and 1 feed layer 2 through normalized KV, compressed-attention YaRN RoPE, E4M3FN finalization, and persistent raw-KV ownership. Every layer-2 KVnorm, KVrope, KVcur tile and retained prefix must match the repeated DwarfStar oracle bit-for-bit. This does not claim layer-2 compression, attention, FFN, complete-model prefill, or throughput."
}

fn prefill_layers012_compressor_loop_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers012-compressor-loop-probe MODEL.gguf [--json PATH]\n\nRuns all 64 native 32-row schedules over positions 0--2047 in one persistent Metal context. Complete live layers 0 and 1 feed layer 2 through exact raw KV and the paired ratio-4 attention/indexer compressors. Every one of the 512 compressed rows and all four final recurrent-state tensors must match repeated DwarfStar captures bit-for-bit. This does not claim layer-2 mixed attention, FFN, complete-model prefill, or throughput."
}

fn prefill_layers012_attention_loop_probe_usage() -> &'static str {
    "usage: rust-star prefill-layers012-attention-loop-probe MODEL.gguf [--json PATH]\n\nRuns all 64 native 32-row schedules over positions 0--2047 in one persistent Metal context and completes layers 2 through 31. Even layers 2/4/6/8/10/12/14/16/18/20/22/24/26/28/30 include paired ratio-4 attention/indexer compressors; odd layers 3/5/7/9/11/13/15/17/19/21/23/25/27/29/31 use ratio-128 attention compressors. Each layer continues through dense mixed attention, biased top-6 routed/shared experts, and its additive final HC update. Every retained boundary must match repeated DwarfStar captures bit-for-bit. The 2,064 logical odd-layer keys use a 2,112-row masked physical extent required by the 64-row FlashAttention block contract. Exactly 512 ratio-4 rows remain on the dense path at this prompt boundary. This does not claim layer-32 prefill, sparse post-prompt ratio-4 attention, complete-model prefill, output logits, or throughput."
}

fn ingress_probe_usage() -> &'static str {
    "usage: rust-star attention-ingress-probe MODEL.gguf [--json PATH]\n\nRuns the six imported DwarfStar Metal operations from a real token embedding through the layer-0 Q-A projection and checks every pinned boundary bit-for-bit."
}

fn attention_setup_probe_usage() -> &'static str {
    "usage: rust-star attention-setup-probe MODEL.gguf [--json PATH]\n\nExtends the connected layer-0 path through KV projection, fused Q-Lora/KV RMSNorm, and Q-B, then checks every pinned DwarfStar boundary bit-for-bit."
}

fn rope_kv_store_probe_usage() -> &'static str {
    "usage: rust-star rope-kv-store-probe MODEL.gguf [--json PATH]\n\nExtends the connected layer-0 path through fused Q head RMSNorm/RoPE, KV RoPE, FP8 finalization, and an exact guarded cache-row write."
}

fn attention_read_probe_usage() -> &'static str {
    "usage: rust-star attention-read-probe MODEL.gguf [--json PATH]\n\nExtends the connected layer-0 path through raw-cache F32-to-F16 staging, DwarfStar FlashAttention over rows 0-1, reduction, and inverse RoPE."
}

fn attention_output_probe_usage() -> &'static str {
    "usage: rust-star attention-output-probe MODEL.gguf [--json PATH]\n\nExtends the connected layer-0 path through DwarfStar's grouped Q8 attention output projection and fused four-stream HC post-update."
}

fn ffn_router_probe_usage() -> &'static str {
    "usage: rust-star ffn-router-probe MODEL.gguf [--json PATH]\n\nContinues from the pinned layer-0 attention HC state through FFN HC ingress and DwarfStar's exact one-token router selection."
}

fn moe_output_probe_usage() -> &'static str {
    "usage: rust-star moe-output-probe MODEL.gguf [--json PATH]\n\nContinues from the pinned layer-0 router outputs through the fused routed and shared experts and FFN HC post-update."
}

fn layer0_probe_usage() -> &'static str {
    "usage: rust-star layer0-probe MODEL.gguf [--json PATH]\n\nRuns the complete 30-dispatch layer-0 decode path in one Metal command buffer and checks every retained attention, router, expert, and HC-state boundary."
}

fn layer0_bench_usage() -> &'static str {
    "usage: rust-star layer0-bench MODEL.gguf [--warmup N] [--iterations N] [--json PATH]\n\nCreates model views and activation buffers once, excludes warmups, then measures repeated bit-exact executions of the complete layer-0 command chain."
}

fn layers01_probe_usage() -> &'static str {
    "usage: rust-star layers01-probe MODEL.gguf [--json PATH]\n\nRuns complete layers 0 and 1 under one Rust-owned Metal executor, handing layer 0's final HC state directly to layer 1 without a host roundtrip, and requires both layers to match their pinned DwarfStar boundaries bit-for-bit."
}

fn layers012_probe_usage() -> &'static str {
    "usage: rust-star layers012-probe MODEL.gguf [--json PATH]\n\nRuns complete layers 0, 1, and 2 under one Rust-owned Metal executor with one retained KV cache per layer and direct GPU-resident HC handoffs, requiring all three layers to match pinned DwarfStar boundaries bit-for-bit."
}

fn layers012_chained_probe_usage() -> &'static str {
    "usage: rust-star layers012-chained-probe MODEL.gguf [--json PATH]\n\nCommits complete layers 0, 1, and 2 to one Metal queue without inter-layer host waits, waits once at the tail, and requires all retained DwarfStar boundaries to remain bit-identical."
}

fn layers0123_probe_usage() -> &'static str {
    "usage: rust-star layers0123-probe MODEL.gguf [--json PATH]\n\nRuns complete layers 0 through 3 under one Rust-owned Metal executor with one retained KV cache per layer and direct GPU-resident HC handoffs, requiring all four layers to match pinned DwarfStar boundaries bit-for-bit."
}

fn layers0123_chained_probe_usage() -> &'static str {
    "usage: rust-star layers0123-chained-probe MODEL.gguf [--json PATH]\n\nCommits complete layers 0 through 3 to one Metal queue without inter-layer host waits, waits once at the tail, and requires all retained DwarfStar boundaries to remain bit-identical."
}

fn layers0123_bench_usage() -> &'static str {
    "usage: rust-star layers0123-bench MODEL.gguf [--warmup N] [--iterations N] [--json PATH]\n\nPrepares layers 0 through 3 once, measures repeated four-command-buffer chains with one tail wait and no in-interval boundary readback, then performs one exhaustive final C0 collection."
}

fn layers0123_decode_probe_usage() -> &'static str {
    "usage: rust-star layers0123-decode-probe MODEL.gguf [--json PATH]\n\nExecutes tokens 201, 361, and 1915 at positions 1 through 3 across layers 0 through 3, preserves four raw KV rows per layer, advances the layer-2 ratio-4 and layer-3 ratio-128 compressor states, validates layer 2's first compressed KV emission, and requires every retained DwarfStar boundary plus the final HC output handoff to remain bit-identical."
}

fn layers012345_decode_probe_usage() -> &'static str {
    "usage: rust-star layers012345-decode-probe MODEL.gguf [--json PATH]\n\nExecutes tokens 201, 361, and 1915 at positions 1 through 3 across layers 0 through 5, preserves six raw KV caches, advances the alternating ratio-4/ratio-128 compressor schedule, validates the layer-2 and layer-4 compressed KV emissions, and requires every retained DwarfStar boundary plus layer 5's final HC handoff to remain bit-identical."
}

fn layers01234567_decode_probe_usage() -> &'static str {
    "usage: rust-star layers01234567-decode-probe MODEL.gguf [--json PATH]\n\nExecutes tokens 201, 361, and 1915 at positions 1 through 3 across layers 0 through 7, preserves eight raw KV caches, advances the alternating ratio-4/ratio-128 compressor schedule, validates the layer-2, layer-4, and layer-6 compressed KV emissions, and requires every retained DwarfStar boundary plus layer 7's final HC handoff to remain bit-identical."
}

fn layers0_to_42_decode_probe_usage() -> &'static str {
    "usage: rust-star layers0-42-decode-probe MODEL.gguf [--json PATH]\n\nExecutes tokens 201, 361, and 1915 at positions 1 through 3 across all 43 transformer layers, preserves one raw KV cache per layer, advances the alternating ratio-4/ratio-128 compressor schedule, validates every even-layer compressed KV emission through layer 42, and requires every retained DwarfStar boundary plus layer 42's final HC handoff to remain bit-identical. This stops before output normalization, logits, and sampling."
}

fn decoder_output_probe_usage() -> &'static str {
    "usage: rust-star decoder-output-probe MODEL.gguf [--json PATH]\n\nExecutes the fixed input tokens 201, 361, 1915, and 262 at positions 1 through 4 across all 43 transformer layers, then applies the exact DwarfStar HC collapse, learned output normalization, full 129280-token Q8_0 vocabulary projection, and lowest-token-ID argmax. Position 4 consumes the persistent compressed rows emitted at position 3. Every retained transformer boundary and all five output-head tensors must remain bit-identical to the captured oracle. Input tokens are externally supplied; this is not yet a closed-loop generator."
}

fn closed_loop_decoder_probe_usage() -> &'static str {
    "usage: rust-star closed-loop-decoder-probe MODEL.gguf [--json PATH]\n\nRuns an exhaustive four-position C0 pass in which each selected token becomes the next input, including the first step that reuses persistent compressed memory, then repeats the same closed loop through a diagnostic timed path. Timed intervals include Metal submission, synchronized execution, logits transfer, and lowest-ID argmax while excluding fixture tensor readback. This remains an ineligible captured-state four-position regression control."
}

fn position127_decoder_probe_usage() -> &'static str {
    "usage: rust-star position127-decoder-probe MODEL.gguf [--json PATH]\n\nAdvances one Rust-owned 43-layer decoder through positions 1-127 using closed-loop greedy selection. It requires the complete 128-token transcript, final full logits, and the integrated layer-3/layer-5 ratio-128 compressed rows to match independently repeated DwarfStar evidence. This is a frontier diagnostic, not a paired-protocol measurement."
}

fn cold_prefill_decoder_probe_usage() -> &'static str {
    "usage: rust-star cold-prefill-decoder-probe MODEL.gguf [--json PATH]\n\nStarts from empty Rust-owned raw and compressed cache state, evaluates the one-token raw oracle prompt at position 0, and requires its full logits to match DwarfStar bit-for-bit before committing token 201. It then reproduces the complete 128-token transcript, final logits, and live layer-3/layer-5 ratio-128 rows. This removes captured initial state but remains diagnostic until native batched prefill and sparse indexed decode exist."
}

fn prefill_frontier_probe_usage() -> &'static str {
    "usage: rust-star prefill-frontier-probe MODEL.gguf [--json PATH]\n\nStarts from empty Rust-owned state, sequentially evaluates the canonical 2048-token oracle prefix through all 43 layers, retains a 128-row raw-KV ring plus context-sized compressed memory, and requires the final logits to match two fresh DwarfStar one-token decode replays bit-for-bit. It also preserves and reports the expected divergence from DwarfStar's batched-prefill logits, so this remains ineligible until native batched prefill and sparse indexed attention are implemented."
}

fn ratio128_compressor_replay_probe_usage() -> &'static str {
    "usage: rust-star ratio128-compressor-replay-probe MODEL.gguf [--json PATH]\n\nReplays 128 independently captured DwarfStar attn_norm rows through the layer-3 and layer-5 ratio-128 attention compressors, validates both first emitted KV rows bit-for-bit, and explicitly performs neither token sampling nor a complete decoder pass."
}

fn sparse_indexed_attention_probe_usage() -> &'static str {
    "usage: rust-star sparse-indexed-attention-probe MODEL.gguf [--json PATH]\n\nRuns two isolated layer-2 sparse-indexed-attention controls against repeated DwarfStar captures: the position-2051 diagnostic threshold override and the production-default position-4099 switch at 1,025 compressed rows. The latter includes the exact two-block argsort merge. This is not a complete-decode, logits, or throughput claim."
}

fn retained_sparse_boundary_probe_usage() -> &'static str {
    "usage: rust-star retained-sparse-boundary-probe MODEL.gguf [--json PATH]\n\nSeeds the independently repeated incoming layer-2 HC, 127 raw rows, 1,024 compressed rows, and recurrent compressor state, then executes the production retained layer path at position 4099. It requires the emitted row 1,025 and every sparse-attention boundary through HC post to match DwarfStar bit-for-bit. FFN, complete-decoder, logits, and throughput claims remain false."
}

fn retained_sparse_multimerge_probe_usage() -> &'static str {
    "usage: rust-star retained-sparse-multimerge-probe MODEL.gguf [--json PATH]\n\nSeeds the independently repeated layer-2 state immediately before position 8195, then executes the production retained path through compressed row 2,049. It requires three initial sort blocks, two ping-pong merge passes, and every sparse-attention boundary through HC post to match DwarfStar bit-for-bit. FFN, complete-decoder, logits, and throughput claims remain false."
}

fn retained_decoder_step_probe_usage() -> &'static str {
    "usage: rust-star retained-decoder-step-probe MODEL.gguf [--json PATH]\n\nSeeds independently repeated retained state immediately before position 8195, executes layers 0 through 42 with live HC handoffs, and runs the output head. Every layer HC and all 129,280 logits must match DwarfStar bit-for-bit. This is a complete retained decoder step with seeded history; native prefill and throughput claims remain false."
}
