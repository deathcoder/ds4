//! Strict, bounded GGUF v3 directory parser.
//!
//! Tensor payloads are never read. The parser retains only small metadata
//! arrays needed by target validation, so the tokenizer tables do not get
//! duplicated in host memory.

use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Seek, SeekFrom};

const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_ALIGNMENT: u64 = 1 << 20;
const MAX_STRING_BYTES: u64 = 16 << 20;
const MAX_METADATA: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_DIMS: u32 = 8;
const MAX_CAPTURED_ARRAY_ITEMS: u64 = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for ValueType {
    type Error = Error;

    fn try_from(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return Err(Error::invalid(format!("unknown GGUF metadata type {raw}"))),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Scalar {
        value_type: ValueType,
        value: ScalarValue,
    },
    Array {
        element_type: ValueType,
        len: u64,
        /// Present only for small numeric arrays needed by model validation.
        values: Option<Vec<ScalarValue>>,
    },
}

impl MetadataValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::Scalar { value_type, .. } => *value_type,
            Self::Array { .. } => ValueType::Array,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorType {
    pub id: u32,
    pub name: &'static str,
    pub block_elements: u64,
    pub block_bytes: u64,
}

impl TensorType {
    pub fn from_id(id: u32) -> Option<Self> {
        let (name, block_elements, block_bytes) = match id {
            0 => ("f32", 1, 4),
            1 => ("f16", 1, 2),
            2 => ("q4_0", 32, 18),
            3 => ("q4_1", 32, 20),
            6 => ("q5_0", 32, 22),
            7 => ("q5_1", 32, 24),
            8 => ("q8_0", 32, 34),
            9 => ("q8_1", 32, 40),
            10 => ("q2_k", 256, 84),
            11 => ("q3_k", 256, 110),
            12 => ("q4_k", 256, 144),
            13 => ("q5_k", 256, 176),
            14 => ("q6_k", 256, 210),
            15 => ("q8_k", 256, 292),
            16 => ("iq2_xxs", 256, 66),
            17 => ("iq2_xs", 256, 74),
            18 => ("iq3_xxs", 256, 98),
            19 => ("iq1_s", 256, 110),
            20 => ("iq4_nl", 256, 50),
            21 => ("iq3_s", 256, 110),
            22 => ("iq2_s", 256, 82),
            23 => ("iq4_xs", 256, 136),
            24 => ("i8", 1, 1),
            25 => ("i16", 1, 2),
            26 => ("i32", 1, 4),
            27 => ("i64", 1, 8),
            28 => ("f64", 1, 8),
            29 => ("iq1_m", 256, 56),
            30 => ("bf16", 1, 2),
            39 => ("mxfp4", 32, 17),
            _ => return None,
        };
        Some(Self { id, name, block_elements, block_bytes })
    }

    fn byte_len(&self, elements: u64) -> Result<u64> {
        let rounded = elements
            .checked_add(self.block_elements - 1)
            .ok_or_else(|| Error::invalid("tensor block count overflow"))?;
        let blocks = rounded / self.block_elements;
        blocks
            .checked_mul(self.block_bytes)
            .ok_or_else(|| Error::invalid("tensor byte count overflow"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: TensorType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Gguf {
    pub version: u32,
    pub file_bytes: u64,
    pub alignment: u64,
    pub tensor_data_offset: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl Gguf {
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let file_bytes = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        if file_bytes < 24 {
            return Err(Error::invalid("file is too small to contain a GGUF header"));
        }

        let mut input = Input { reader, file_bytes };
        let mut magic = [0_u8; 4];
        input.read_exact(&mut magic, "GGUF magic")?;
        if &magic != b"GGUF" {
            return Err(Error::invalid("file does not begin with GGUF magic"));
        }
        let version = input.u32("GGUF version")?;
        if version != 3 {
            return Err(Error::invalid(format!("only GGUF v3 is supported, found v{version}")));
        }
        let tensor_count = input.u64("tensor count")?;
        let metadata_count = input.u64("metadata count")?;
        if tensor_count > MAX_TENSORS {
            return Err(Error::invalid(format!("tensor count {tensor_count} exceeds safety limit")));
        }
        if metadata_count > MAX_METADATA {
            return Err(Error::invalid(format!("metadata count {metadata_count} exceeds safety limit")));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = input.string("metadata key")?;
            if metadata.contains_key(&key) {
                return Err(Error::invalid(format!("duplicate GGUF metadata key {key:?}")));
            }
            let value_type = ValueType::try_from(input.u32("metadata value type")?)?;
            let capture = should_capture_array(&key);
            let value = input.value(value_type, capture)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(MetadataValue::Scalar {
                value_type: ValueType::Uint32,
                value: ScalarValue::Unsigned(value),
            }) => *value,
            Some(_) => {
                return Err(Error::invalid(
                    "general.alignment must be a GGUF uint32",
                ))
            }
        };
        if alignment == 0 || !alignment.is_power_of_two() || alignment > MAX_ALIGNMENT {
            return Err(Error::invalid(format!(
                "invalid GGUF alignment {alignment}; expected a power of two up to {MAX_ALIGNMENT}"
            )));
        }

        struct PendingTensor {
            name: String,
            dimensions: Vec<u64>,
            tensor_type: TensorType,
            relative_offset: u64,
            elements: u64,
            bytes: u64,
        }

        let mut pending = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
        let mut names = HashSet::with_capacity(pending.capacity());
        for _ in 0..tensor_count {
            let name = input.string("tensor name")?;
            if !names.insert(name.clone()) {
                return Err(Error::invalid(format!("duplicate GGUF tensor name {name:?}")));
            }
            let rank = input.u32("tensor rank")?;
            if rank == 0 || rank > MAX_DIMS {
                return Err(Error::invalid(format!(
                    "tensor {name:?} has unsupported rank {rank}"
                )));
            }
            let mut dimensions = Vec::with_capacity(rank as usize);
            let mut elements = 1_u64;
            for _ in 0..rank {
                let dimension = input.u64("tensor dimension")?;
                if dimension == 0 {
                    return Err(Error::invalid(format!("tensor {name:?} has a zero dimension")));
                }
                elements = elements.checked_mul(dimension).ok_or_else(|| {
                    Error::invalid(format!("tensor {name:?} element count overflows u64"))
                })?;
                dimensions.push(dimension);
            }
            let type_id = input.u32("tensor type")?;
            let tensor_type = TensorType::from_id(type_id).ok_or_else(|| {
                Error::invalid(format!("tensor {name:?} uses unsupported GGML type {type_id}"))
            })?;
            let relative_offset = input.u64("tensor relative offset")?;
            if relative_offset % alignment != 0 {
                return Err(Error::invalid(format!(
                    "tensor {name:?} offset {relative_offset} is not aligned to {alignment}"
                )));
            }
            let bytes = tensor_type.byte_len(elements)?;
            pending.push(PendingTensor {
                name,
                dimensions,
                tensor_type,
                relative_offset,
                elements,
                bytes,
            });
        }

        let directory_end = input.position()?;
        let tensor_data_offset = align_up(directory_end, alignment)?;
        if tensor_data_offset > file_bytes {
            return Err(Error::invalid("aligned tensor-data offset is outside the file"));
        }

        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::with_capacity(pending.len());
        for tensor in pending {
            let absolute_offset = tensor_data_offset
                .checked_add(tensor.relative_offset)
                .ok_or_else(|| Error::invalid("absolute tensor offset overflow"))?;
            let end = absolute_offset
                .checked_add(tensor.bytes)
                .ok_or_else(|| Error::invalid("tensor end offset overflow"))?;
            if end > file_bytes {
                return Err(Error::invalid(format!(
                    "tensor {:?} range {absolute_offset}..{end} exceeds file size {file_bytes}",
                    tensor.name
                )));
            }
            ranges.push((absolute_offset, end, tensor.name.clone()));
            let info = TensorInfo {
                name: tensor.name.clone(),
                dimensions: tensor.dimensions,
                tensor_type: tensor.tensor_type,
                relative_offset: tensor.relative_offset,
                absolute_offset,
                elements: tensor.elements,
                bytes: tensor.bytes,
            };
            tensors.insert(tensor.name, info);
        }
        ranges.sort_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(Error::invalid(format!(
                    "tensor payloads overlap: {:?} ends at {}, {:?} begins at {}",
                    pair[0].2, pair[0].1, pair[1].2, pair[1].0
                )));
            }
        }

        Ok(Self {
            version,
            file_bytes,
            alignment,
            tensor_data_offset,
            metadata,
            tensors,
        })
    }

    pub fn string(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::Scalar {
                value_type: ValueType::String,
                value: ScalarValue::String(value),
            }) => Ok(value),
            Some(other) => Err(Error::invalid(format!(
                "metadata {key:?} has type {:?}, expected string",
                other.value_type()
            ))),
            None => Err(Error::invalid(format!("required metadata {key:?} is missing"))),
        }
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        match self.metadata.get(key) {
            Some(MetadataValue::Scalar {
                value_type: ValueType::Uint32 | ValueType::Uint64,
                value: ScalarValue::Unsigned(value),
            }) => Ok(*value),
            Some(other) => Err(Error::invalid(format!(
                "metadata {key:?} has type {:?}, expected uint32/uint64",
                other.value_type()
            ))),
            None => Err(Error::invalid(format!("required metadata {key:?} is missing"))),
        }
    }

    pub fn f64(&self, key: &str) -> Result<f64> {
        match self.metadata.get(key) {
            Some(MetadataValue::Scalar {
                value_type: ValueType::Float32 | ValueType::Float64,
                value: ScalarValue::Float(value),
            }) => Ok(*value),
            Some(other) => Err(Error::invalid(format!(
                "metadata {key:?} has type {:?}, expected float32/float64",
                other.value_type()
            ))),
            None => Err(Error::invalid(format!("required metadata {key:?} is missing"))),
        }
    }

    pub fn boolean(&self, key: &str) -> Result<bool> {
        match self.metadata.get(key) {
            Some(MetadataValue::Scalar {
                value_type: ValueType::Bool,
                value: ScalarValue::Bool(value),
            }) => Ok(*value),
            Some(other) => Err(Error::invalid(format!(
                "metadata {key:?} has type {:?}, expected bool",
                other.value_type()
            ))),
            None => Err(Error::invalid(format!("required metadata {key:?} is missing"))),
        }
    }

    pub fn numeric_array(&self, key: &str) -> Result<(ValueType, &[ScalarValue])> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array {
                element_type,
                values: Some(values),
                ..
            }) => Ok((*element_type, values)),
            Some(MetadataValue::Array { len, .. }) => Err(Error::invalid(format!(
                "metadata array {key:?} ({len} elements) was not retained"
            ))),
            Some(other) => Err(Error::invalid(format!(
                "metadata {key:?} has type {:?}, expected array",
                other.value_type()
            ))),
            None => Err(Error::invalid(format!("required metadata {key:?} is missing"))),
        }
    }
}

fn should_capture_array(key: &str) -> bool {
    matches!(
        key,
        "deepseek4.attention.compress_ratios" | "deepseek4.swiglu_clamp_exp"
    )
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let add = alignment - 1;
    value
        .checked_add(add)
        .map(|sum| sum & !add)
        .ok_or_else(|| Error::invalid("aligned offset overflow"))
}

struct Input<'a, R> {
    reader: &'a mut R,
    file_bytes: u64,
}

impl<R: Read + Seek> Input<'_, R> {
    fn position(&mut self) -> Result<u64> {
        Ok(self.reader.stream_position()?)
    }

    fn read_exact(&mut self, bytes: &mut [u8], what: &str) -> Result<()> {
        let start = self.position()?;
        let len = bytes.len() as u64;
        if start > self.file_bytes || len > self.file_bytes - start {
            return Err(Error::invalid(format!("short read while reading {what}")));
        }
        self.reader.read_exact(bytes).map_err(|error| {
            Error::invalid(format!("failed to read {what} at byte {start}: {error}"))
        })
    }

    fn bytes<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let mut bytes = [0_u8; N];
        self.read_exact(&mut bytes, what)?;
        Ok(bytes)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.bytes::<1>(what)?[0])
    }

    fn u16(&mut self, what: &str) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(what)?))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(what)?))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let len = self.u64(&format!("{what} length"))?;
        if len > MAX_STRING_BYTES {
            return Err(Error::invalid(format!("{what} length {len} exceeds safety limit")));
        }
        let size = usize::try_from(len)
            .map_err(|_| Error::invalid(format!("{what} does not fit host address space")))?;
        let mut bytes = vec![0_u8; size];
        self.read_exact(&mut bytes, what)?;
        String::from_utf8(bytes)
            .map_err(|_| Error::invalid(format!("{what} is not valid UTF-8")))
    }

    fn skip(&mut self, bytes: u64, what: &str) -> Result<()> {
        let start = self.position()?;
        if start > self.file_bytes || bytes > self.file_bytes - start {
            return Err(Error::invalid(format!("{what} exceeds file bounds")));
        }
        self.reader.seek(SeekFrom::Start(start + bytes))?;
        Ok(())
    }

    fn value(&mut self, value_type: ValueType, capture_array: bool) -> Result<MetadataValue> {
        if value_type == ValueType::Array {
            let element_type = ValueType::try_from(self.u32("metadata array element type")?)?;
            if matches!(element_type, ValueType::Array) {
                return Err(Error::invalid("nested GGUF metadata arrays are forbidden"));
            }
            let len = self.u64("metadata array length")?;
            let capture = capture_array && len <= MAX_CAPTURED_ARRAY_ITEMS;
            let values = if capture {
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.scalar(element_type)?);
                }
                Some(values)
            } else {
                self.skip_array(element_type, len)?;
                None
            };
            return Ok(MetadataValue::Array { element_type, len, values });
        }
        Ok(MetadataValue::Scalar {
            value_type,
            value: self.scalar(value_type)?,
        })
    }

    fn scalar(&mut self, value_type: ValueType) -> Result<ScalarValue> {
        Ok(match value_type {
            ValueType::Uint8 => ScalarValue::Unsigned(self.u8("uint8 metadata")? as u64),
            ValueType::Int8 => ScalarValue::Signed(i8::from_le_bytes(self.bytes("int8 metadata")?) as i64),
            ValueType::Uint16 => ScalarValue::Unsigned(self.u16("uint16 metadata")? as u64),
            ValueType::Int16 => ScalarValue::Signed(i16::from_le_bytes(self.bytes("int16 metadata")?) as i64),
            ValueType::Uint32 => ScalarValue::Unsigned(self.u32("uint32 metadata")? as u64),
            ValueType::Int32 => ScalarValue::Signed(i32::from_le_bytes(self.bytes("int32 metadata")?) as i64),
            ValueType::Float32 => ScalarValue::Float(f32::from_le_bytes(self.bytes("float32 metadata")?) as f64),
            ValueType::Bool => match self.u8("bool metadata")? {
                0 => ScalarValue::Bool(false),
                1 => ScalarValue::Bool(true),
                value => return Err(Error::invalid(format!("invalid GGUF bool byte {value}"))),
            },
            ValueType::String => ScalarValue::String(self.string("metadata string")?),
            ValueType::Uint64 => ScalarValue::Unsigned(self.u64("uint64 metadata")?),
            ValueType::Int64 => ScalarValue::Signed(i64::from_le_bytes(self.bytes("int64 metadata")?)),
            ValueType::Float64 => ScalarValue::Float(f64::from_le_bytes(self.bytes("float64 metadata")?)),
            ValueType::Array => return Err(Error::invalid("nested GGUF metadata arrays are forbidden")),
        })
    }

    fn skip_array(&mut self, element_type: ValueType, len: u64) -> Result<()> {
        if element_type == ValueType::String {
            for _ in 0..len {
                let string_len = self.u64("array string length")?;
                if string_len > MAX_STRING_BYTES {
                    return Err(Error::invalid(format!(
                        "array string length {string_len} exceeds safety limit"
                    )));
                }
                self.skip(string_len, "array string")?;
            }
            return Ok(());
        }
        let width = match element_type {
            ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => 1,
            ValueType::Uint16 | ValueType::Int16 => 2,
            ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => 4,
            ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => 8,
            ValueType::String | ValueType::Array => unreachable!(),
        };
        let bytes = len
            .checked_mul(width)
            .ok_or_else(|| Error::invalid("metadata array byte count overflow"))?;
        self.skip(bytes, "metadata array")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_u32(output: &mut Vec<u8>, value: u32) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(output: &mut Vec<u8>, value: u64) {
        output.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(output: &mut Vec<u8>, value: &str) {
        push_u64(output, value.len() as u64);
        output.extend_from_slice(value.as_bytes());
    }

    fn minimal_gguf(relative_offset: u64, data_bytes: usize) -> Vec<u8> {
        let mut output = b"GGUF".to_vec();
        push_u32(&mut output, 3);
        push_u64(&mut output, 1);
        push_u64(&mut output, 2);

        push_string(&mut output, "general.architecture");
        push_u32(&mut output, ValueType::String as u32);
        push_string(&mut output, "deepseek4");
        push_string(&mut output, "general.alignment");
        push_u32(&mut output, ValueType::Uint32 as u32);
        push_u32(&mut output, 32);

        push_string(&mut output, "token_embd.weight");
        push_u32(&mut output, 1);
        push_u64(&mut output, 1);
        push_u32(&mut output, 0);
        push_u64(&mut output, relative_offset);

        while output.len() % 32 != 0 {
            output.push(0);
        }
        output.resize(output.len() + relative_offset as usize + data_bytes, 0);
        output
    }

    #[test]
    fn parses_directory_without_reading_payload() {
        let bytes = minimal_gguf(32, 4);
        let mut cursor = Cursor::new(bytes);
        let gguf = Gguf::parse(&mut cursor).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.alignment, 32);
        assert_eq!(gguf.string("general.architecture").unwrap(), "deepseek4");
        let tensor = gguf.tensors.get("token_embd.weight").unwrap();
        assert_eq!(tensor.bytes, 4);
        assert_eq!(tensor.relative_offset, 32);
    }

    #[test]
    fn rejects_unaligned_tensor_offsets() {
        let bytes = minimal_gguf(1, 4);
        let mut cursor = Cursor::new(bytes);
        let error = Gguf::parse(&mut cursor).unwrap_err().to_string();
        assert!(error.contains("not aligned"), "{error}");
    }

    #[test]
    fn rejects_tensor_ranges_outside_file() {
        let bytes = minimal_gguf(32, 3);
        let mut cursor = Cursor::new(bytes);
        let error = Gguf::parse(&mut cursor).unwrap_err().to_string();
        assert!(error.contains("exceeds file size"), "{error}");
    }

    #[test]
    fn rejects_noncanonical_bool_values() {
        let mut bytes = b"GGUF".to_vec();
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 1);
        push_string(&mut bytes, "test.bool");
        push_u32(&mut bytes, ValueType::Bool as u32);
        bytes.push(2);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let mut cursor = Cursor::new(bytes);
        let error = Gguf::parse(&mut cursor).unwrap_err().to_string();
        assert!(error.contains("invalid GGUF bool"), "{error}");
    }
}
