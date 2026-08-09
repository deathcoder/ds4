use rust_star_runtime::gguf::Gguf;
use rust_star_runtime::target::{validate_resident_q2, MODEL_LABEL};
use rust_star_runtime::{Error, Result};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

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
    let model = arguments.next().ok_or_else(|| Error::invalid(usage()))?;
    if arguments.next().is_some() {
        return Err(Error::invalid(usage()));
    }
    let model = Path::new(&model);
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
    "usage:\n  rust-star inspect MODEL.gguf  # strict Flash-0731 resident-Q2 validation\n  rust-star gguf MODEL.gguf     # structural GGUF v3 validation only"
}
