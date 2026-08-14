use rust_star_runtime::gguf::Gguf;
use rust_star_runtime::metal::{
    run_f16_embedding_probe, run_probe, write_embedding_probe_json, write_probe_json,
    EmbeddingProbeReport, ProbeConfig,
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
    run_model_command(&command, arguments.collect())
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
    "usage:\n  rust-star inspect MODEL.gguf  # strict Flash-0731 resident-Q2 validation\n  rust-star gguf MODEL.gguf     # structural GGUF v3 validation only\n  rust-star metal-probe [OPTIONS]\n  rust-star embedding-probe MODEL.gguf [OPTIONS]"
}

fn metal_probe_usage() -> &'static str {
    "usage: rust-star metal-probe [--elements N] [--iterations N] [--json PATH]\n\nCompares synchronized per-dispatch command buffers with one batched command buffer."
}

fn embedding_probe_usage() -> &'static str {
    "usage: rust-star embedding-probe MODEL.gguf [--tokens ID,ID,...] [--json PATH]\n\nWraps the F16 token embedding as a page-aligned bytes-no-copy Metal buffer and validates DwarfStar kernel_get_rows_f16 bit-for-bit."
}
