use rust_star_runtime::gguf::Gguf;
use rust_star_runtime::metal::{
    run_attention_ingress_probe, run_attention_output_probe, run_attention_read_probe,
    run_attention_setup_probe, run_f16_embedding_probe, run_probe, run_q8_projection_probe,
    run_rope_kv_store_probe, write_attention_output_probe_json, write_attention_read_probe_json,
    write_attention_setup_probe_json, write_embedding_probe_json, write_ingress_probe_json,
    write_probe_json, write_projection_probe_json, write_rope_kv_store_probe_json,
    AttentionOutputProbeReport, AttentionReadProbeReport, AttentionSetupProbeReport,
    EmbeddingProbeReport, IngressProbeReport, ProbeConfig, ProjectionProbeReport,
    RopeKvStoreProbeReport,
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
    run_model_command(&command, arguments.collect())
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
    "usage:\n  rust-star inspect MODEL.gguf  # strict Flash-0731 resident-Q2 validation\n  rust-star gguf MODEL.gguf     # structural GGUF v3 validation only\n  rust-star metal-probe [OPTIONS]\n  rust-star embedding-probe MODEL.gguf [OPTIONS]\n  rust-star projection-probe MODEL.gguf [OPTIONS]\n  rust-star attention-ingress-probe MODEL.gguf [OPTIONS]\n  rust-star attention-setup-probe MODEL.gguf [OPTIONS]\n  rust-star rope-kv-store-probe MODEL.gguf [OPTIONS]\n  rust-star attention-read-probe MODEL.gguf [OPTIONS]\n  rust-star attention-output-probe MODEL.gguf [OPTIONS]"
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
