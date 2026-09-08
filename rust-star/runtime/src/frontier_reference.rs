//! Validated external vectors for diagnostic frontier comparison.
//! Authenticating their oracle/model provenance belongs to the evidence runner.
use crate::{Error, Result};

pub struct FrontierReference {
    pub(crate) tokens: Vec<u32>,
    pub(crate) selected: Vec<u32>,
    pub(crate) logits: Vec<f32>,
}
impl FrontierReference {
    pub fn from_bytes(tokens: &[u8], selected: &[u8], logits: &[u8]) -> Result<Self> {
        fn words(bytes: &[u8]) -> Result<Vec<u32>> {
            if bytes.len() % 4 != 0 {
                return Err(Error::invalid("reference has a partial 32-bit word"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        }
        let tokens = words(tokens)?;
        let selected = words(selected)?;
        let n = tokens.len();
        if !(32768..=131072).contains(&n) || n % 4096 != 0 {
            return Err(Error::invalid(
                "external frontier must be aligned 32K through 128K",
            ));
        }
        if !matches!(selected.len(), 1 | 129) || (n == 131072 && selected.len() != 129) {
            return Err(Error::invalid(
                "reference requires one prefill row or 129 full rows; 128K requires 129",
            ));
        }
        if tokens.iter().chain(&selected).any(|&t| t >= 129280) {
            return Err(Error::invalid("reference token is outside the vocabulary"));
        }
        if logits.len() != selected.len() * 129280 * 4 {
            return Err(Error::invalid(
                "reference logit extent differs from its transcript",
            ));
        }
        let logits: Vec<f32> = words(logits)?.into_iter().map(f32::from_bits).collect();
        for (row, &token) in logits.chunks_exact(129280).zip(&selected) {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid("reference contains non-finite logits"));
            }
            let mut best = 0;
            for i in 1..row.len() {
                if row[i] > row[best] {
                    best = i;
                }
            }
            if best as u32 != token {
                return Err(Error::invalid(
                    "reference transcript differs from greedy logits",
                ));
            }
        }
        Ok(Self {
            tokens,
            selected,
            logits,
        })
    }
    pub fn frontier(&self) -> u32 {
        self.tokens.len() as u32
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measurement_128k_fixture_extends_exact_32k_prefix() {
        let prompt = include_bytes!("../../fixtures/measurement-131072-v5/prompt.u32le.bin");
        let selected = include_bytes!("../../fixtures/measurement-131072-v5/selected.u32le.bin");
        assert_eq!(prompt.len(), 131072 * 4);
        assert_eq!(
            &prompt[..32768 * 4],
            include_bytes!("../../fixtures/prefill-frontier-32768-v5/token-ids.u32le.bin")
        );
        assert_eq!(selected.len(), 129 * 4);
        assert_eq!(u32::from_le_bytes(selected[..4].try_into().unwrap()), 1110);
        assert_eq!(
            u32::from_le_bytes(selected[128 * 4..].try_into().unwrap()),
            2945
        );
        assert!(prompt
            .chunks_exact(4)
            .chain(selected.chunks_exact(4))
            .all(|v| u32::from_le_bytes(v.try_into().unwrap()) < 129280));
    }

    #[test]
    fn rejects_invalid_frontier_and_partial_words() {
        assert!(FrontierReference::from_bytes(&[0], &[], &[]).is_err());
        for n in [0, 32767, 32769, 131073] {
            assert!(FrontierReference::from_bytes(&vec![0; n * 4], &[0; 4], &[]).is_err());
        }
    }
    #[test]
    fn checks_full_rows_greedy_selection_and_finiteness() {
        let tokens = vec![0; 32768 * 4];
        let mut logits = vec![0; 129280 * 4];
        assert!(FrontierReference::from_bytes(&tokens, &[0; 4], &logits).is_ok());
        assert!(FrontierReference::from_bytes(&tokens, &1u32.to_le_bytes(), &logits).is_err());
        logits[4..8].copy_from_slice(&1f32.to_le_bytes());
        assert!(FrontierReference::from_bytes(&tokens, &1u32.to_le_bytes(), &logits).is_ok());
        logits[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(FrontierReference::from_bytes(&tokens, &1u32.to_le_bytes(), &logits).is_err());
        assert!(FrontierReference::from_bytes(&vec![0; 131072 * 4], &[0; 4], &logits).is_err());
    }
}
