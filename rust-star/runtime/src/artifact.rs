//! Candidate output writer for `rust-star/ARTIFACT_FORMAT.md`.

use crate::{Error, Result};
use std::io::Write;

#[derive(Clone, Debug)]
pub struct LogitMetadata<'a> {
    pub source: &'a str,
    pub backend: &'a str,
    pub model: &'a str,
    pub prompt_tokens: u64,
    pub frontier_tokens: u64,
    pub context: u64,
    pub quant_bits: u32,
    pub quality: bool,
}

/// Write full FP32 logits in the stable oracle/candidate JSON contract.
///
/// Nine significant decimal digits are emitted for every value, which is
/// sufficient to round-trip any finite IEEE-754 binary32 value. Signed zero is
/// preserved and the recorded argmax uses the lowest token ID on ties.
pub fn write_full_logits<W: Write>(
    output: &mut W,
    metadata: &LogitMetadata<'_>,
    logits: &[f32],
) -> Result<()> {
    if logits.is_empty() {
        return Err(Error::invalid("cannot write an empty logit vector"));
    }
    if metadata.context == 0 {
        return Err(Error::invalid("logit artifact context must be nonzero"));
    }
    if metadata.prompt_tokens > metadata.frontier_tokens {
        return Err(Error::invalid(
            "prompt_tokens cannot exceed frontier_tokens in a post-prefill artifact",
        ));
    }
    if metadata.frontier_tokens > metadata.context {
        return Err(Error::invalid(
            "frontier_tokens cannot exceed the allocated context",
        ));
    }
    if metadata.quant_bits == 0 {
        return Err(Error::invalid("quant_bits must be nonzero"));
    }

    let mut argmax_id = 0_usize;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(Error::invalid(format!(
                "logit at token ID {index} is not finite"
            )));
        }
        if value > logits[argmax_id] {
            argmax_id = index;
        }
    }

    output.write_all(b"{\n  \"source\": ")?;
    write_json_string(output, metadata.source)?;
    output.write_all(b",\n  \"backend\": ")?;
    write_json_string(output, metadata.backend)?;
    output.write_all(b",\n  \"model\": ")?;
    write_json_string(output, metadata.model)?;
    write!(
        output,
        ",\n  \"vocab\": {},\n  \"prompt_tokens\": {},\n  \"frontier_tokens\": {},\n  \"ctx\": {},\n  \"quant_bits\": {},\n  \"quality\": {},\n  \"argmax_id\": {},\n  \"logits\": [",
        logits.len(),
        metadata.prompt_tokens,
        metadata.frontier_tokens,
        metadata.context,
        metadata.quant_bits,
        metadata.quality,
        argmax_id,
    )?;
    for (index, value) in logits.iter().enumerate() {
        if index != 0 {
            output.write_all(b", ")?;
        }
        // One digit before the decimal plus eight after it = nine significant
        // digits. Scientific notation also keeps the representation uniform.
        write!(output, "{value:.8e}")?;
    }
    output.write_all(b"]\n}\n")?;
    Ok(())
}

fn write_json_string<W: Write>(output: &mut W, value: &str) -> Result<()> {
    output.write_all(b"\"")?;
    let mut plain_start = 0;
    for (offset, character) in value.char_indices() {
        let escape = match character {
            '"' => Some("\\\""),
            '\\' => Some("\\\\"),
            '\u{08}' => Some("\\b"),
            '\u{0c}' => Some("\\f"),
            '\n' => Some("\\n"),
            '\r' => Some("\\r"),
            '\t' => Some("\\t"),
            _ => None,
        };
        if let Some(escape) = escape {
            output.write_all(value[plain_start..offset].as_bytes())?;
            output.write_all(escape.as_bytes())?;
            plain_start = offset + character.len_utf8();
        } else if character <= '\u{1f}' {
            output.write_all(value[plain_start..offset].as_bytes())?;
            write!(output, "\\u{:04x}", character as u32)?;
            plain_start = offset + character.len_utf8();
        }
    }
    output.write_all(value[plain_start..].as_bytes())?;
    output.write_all(b"\"")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata<'a>() -> LogitMetadata<'a> {
        LogitMetadata {
            source: "rust-star",
            backend: "host-test",
            model: "fixture",
            prompt_tokens: 2,
            frontier_tokens: 2,
            context: 256,
            quant_bits: 2,
            quality: true,
        }
    }

    #[test]
    fn preserves_signed_zero_and_lowest_argmax_tie() {
        let mut output = Vec::new();
        write_full_logits(&mut output, &metadata(), &[-0.0, 3.0, 3.0]).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("-0.00000000e0"), "{text}");
        assert!(text.contains("\"argmax_id\": 1"), "{text}");
    }

    #[test]
    fn escapes_informational_strings_as_json() {
        let mut metadata = metadata();
        metadata.model = "model \"x\"\n";
        let mut output = Vec::new();
        write_full_logits(&mut output, &metadata, &[1.0]).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("model \\\"x\\\"\\n"), "{text}");
    }

    #[test]
    fn rejects_nonfinite_values() {
        let error = write_full_logits(&mut Vec::new(), &metadata(), &[f32::INFINITY])
            .unwrap_err()
            .to_string();
        assert!(error.contains("not finite"), "{error}");
    }

    #[test]
    fn rejects_impossible_frontier_metadata() {
        let mut metadata = metadata();
        metadata.frontier_tokens = 300;
        let error = write_full_logits(&mut Vec::new(), &metadata, &[0.0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("allocated context"), "{error}");
    }
}
