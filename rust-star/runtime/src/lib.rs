//! Model-specific host/runtime contracts for Rust Star.
//!
//! This crate deliberately has no third-party dependencies. The first target is
//! the resident imatrix-Q2 DeepSeek-V4-Flash-0731 GGUF used by DwarfStar, not a
//! general-purpose inference API.

pub mod artifact;
pub mod gguf;
pub mod metal;
pub mod target;

use std::fmt;
use std::io;

/// A parse, validation, or artifact-contract failure.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
