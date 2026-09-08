use rust_star_runtime::artifact::{write_full_logits, LogitMetadata};
use std::env;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let path = arguments
        .next()
        .ok_or("usage: write_logit_fixture OUTPUT.json")?;
    if arguments.next().is_some() {
        return Err("usage: write_logit_fixture OUTPUT.json".into());
    }
    let mut output = File::create(path)?;
    let metadata = LogitMetadata {
        source: "rust-star",
        backend: "host-contract-test",
        model: "synthetic-fixture",
        prompt_tokens: 3,
        frontier_tokens: 3,
        context: 256,
        quant_bits: 2,
        quality: true,
    };
    write_full_logits(
        &mut output,
        &metadata,
        &[-0.0, 1.25, 1.25, f32::MIN_POSITIVE, -3.5],
    )?;
    Ok(())
}
