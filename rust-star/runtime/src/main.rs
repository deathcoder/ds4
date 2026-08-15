use rust_star_runtime::gguf::Gguf;
use rust_star_runtime::metal::{
    run_attention_ingress_probe, run_attention_output_probe, run_attention_read_probe,
    run_attention_setup_probe, run_f16_embedding_probe, run_ffn_router_probe, run_layer0_bench,
    run_layer0_probe, run_layers01234567_decode_probe, run_layers012345_decode_probe,
    run_layers0123_bench, run_layers0123_chained_probe, run_layers0123_decode_probe,
    run_layers0123_probe, run_layers012_chained_probe, run_layers012_probe, run_layers01_probe,
    run_layers0_to_42_decode_probe, run_moe_output_probe, run_probe, run_q8_projection_probe,
    run_ratio128_compressor_replay_probe, run_rope_kv_store_probe,
    write_attention_output_probe_json, write_attention_read_probe_json,
    write_attention_setup_probe_json, write_embedding_probe_json, write_ffn_router_probe_json,
    write_ingress_probe_json, write_layer0_bench_json, write_layer0_probe_json,
    write_layers01234567_decode_probe_json, write_layers012345_decode_probe_json,
    write_layers0123_bench_json, write_layers0123_chained_probe_json,
    write_layers0123_decode_probe_json, write_layers0123_probe_json,
    write_layers012_chained_probe_json, write_layers012_probe_json, write_layers01_probe_json,
    write_layers0_to_42_decode_probe_json, write_moe_output_probe_json, write_probe_json,
    write_projection_probe_json, write_ratio128_compressor_replay_probe_json,
    write_rope_kv_store_probe_json, AttentionOutputProbeReport, AttentionReadProbeReport,
    AttentionSetupProbeReport, EmbeddingProbeReport, FfnRouterProbeReport, IngressProbeReport,
    Layer0BenchConfig, Layer0BenchReport, Layer0ProbeReport, Layers01234567DecodeProbeReport,
    Layers012345DecodeProbeReport, Layers0123BenchConfig, Layers0123BenchReport,
    Layers0123ChainedProbeReport, Layers0123DecodeProbeReport, Layers0123ProbeReport,
    Layers012ChainedProbeReport, Layers012ProbeReport, Layers01ProbeReport,
    Layers0To42DecodeProbeReport, MoeOutputProbeReport, ProbeConfig, ProjectionProbeReport,
    Ratio128CompressorReplayProbeReport, RopeKvStoreProbeReport,
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
    let command = arguments.next().ok_or_else(|| Error::invalid(usage()))?;
    if command == "--help" || command == "-h" || command == "help" {
        println!("{}", usage());
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
    if command == "ratio128-compressor-replay-probe" {
        return run_ratio128_compressor_replay_probe_command(arguments.collect());
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
    "usage:\n  rust-star inspect MODEL.gguf  # strict Flash-0731 resident-Q2 validation\n  rust-star gguf MODEL.gguf     # structural GGUF v3 validation only\n  rust-star metal-probe [OPTIONS]\n  rust-star embedding-probe MODEL.gguf [OPTIONS]\n  rust-star projection-probe MODEL.gguf [OPTIONS]\n  rust-star attention-ingress-probe MODEL.gguf [OPTIONS]\n  rust-star attention-setup-probe MODEL.gguf [OPTIONS]\n  rust-star rope-kv-store-probe MODEL.gguf [OPTIONS]\n  rust-star attention-read-probe MODEL.gguf [OPTIONS]\n  rust-star attention-output-probe MODEL.gguf [OPTIONS]\n  rust-star ffn-router-probe MODEL.gguf [OPTIONS]\n  rust-star moe-output-probe MODEL.gguf [OPTIONS]\n  rust-star layer0-probe MODEL.gguf [OPTIONS]\n  rust-star layer0-bench MODEL.gguf [OPTIONS]\n  rust-star layers01-probe MODEL.gguf [OPTIONS]\n  rust-star layers012-probe MODEL.gguf [OPTIONS]\n  rust-star layers012-chained-probe MODEL.gguf [OPTIONS]\n  rust-star layers0123-probe MODEL.gguf [OPTIONS]\n  rust-star layers0123-chained-probe MODEL.gguf [OPTIONS]\n  rust-star layers0123-bench MODEL.gguf [OPTIONS]\n  rust-star layers0123-decode-probe MODEL.gguf [OPTIONS]\n  rust-star layers012345-decode-probe MODEL.gguf [OPTIONS]\n  rust-star layers01234567-decode-probe MODEL.gguf [OPTIONS]\n  rust-star layers0-42-decode-probe MODEL.gguf [OPTIONS]\n  rust-star ratio128-compressor-replay-probe MODEL.gguf [OPTIONS]"
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

fn ratio128_compressor_replay_probe_usage() -> &'static str {
    "usage: rust-star ratio128-compressor-replay-probe MODEL.gguf [--json PATH]\n\nReplays 128 independently captured DwarfStar attn_norm rows through the layer-3 and layer-5 ratio-128 attention compressors, validates both first emitted KV rows bit-for-bit, and explicitly performs neither token sampling nor a complete decoder pass."
}
