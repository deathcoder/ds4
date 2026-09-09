//! Read-only, file-backed model ownership.
//!
//! Metal wraps page-aligned ranges of this mapping with bytes-no-copy buffers.
//! The mapping therefore remains alive for the entire GPU use and is never
//! exposed as mutable Rust memory.

use crate::gguf::{Gguf, TensorInfo};
use crate::{Error, Result};
use std::ffi::c_void;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::ptr::NonNull;
use std::time::Instant;

#[cfg(unix)]
use std::os::fd::AsRawFd;

pub struct MappedModel {
    gguf: Gguf,
    mapping: NonNull<c_void>,
    mapping_bytes: usize,
    // Keep the descriptor alive with the mapping. POSIX does not require this,
    // but the explicit ownership mirrors the model lifetime and simplifies
    // future residency/prefetch operations.
    _file: File,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelWarmReport {
    pub bytes: u64,
    pub pages: u64,
    pub checksum: u64,
    pub wall_ms: f64,
}

impl MappedModel {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|error| {
            Error::invalid(format!("cannot open model {}: {error}", path.display()))
        })?;
        let file_bytes = file.metadata()?.len();
        if file_bytes == 0 {
            return Err(Error::invalid("cannot map an empty model file"));
        }
        let mapping_bytes = usize::try_from(file_bytes)
            .map_err(|_| Error::invalid("model size does not fit the host address space"))?;

        let parser_file = file.try_clone()?;
        let mut reader = BufReader::with_capacity(1024 * 1024, parser_file);
        let gguf = Gguf::parse(&mut reader)?;
        if gguf.file_bytes != file_bytes {
            return Err(Error::invalid("model size changed while opening it"));
        }

        #[cfg(unix)]
        let mapping = {
            const PROT_READ: i32 = 0x1;
            const MAP_SHARED: i32 = 0x1;
            let pointer = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    mapping_bytes,
                    PROT_READ,
                    MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if pointer as isize == -1 {
                return Err(Error::invalid(format!(
                    "cannot mmap model {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                )));
            }
            NonNull::new(pointer).ok_or_else(|| Error::invalid("mmap returned a null pointer"))?
        };

        #[cfg(not(unix))]
        let mapping = {
            return Err(Error::invalid(
                "file-backed model mapping is supported only on Unix hosts",
            ));
        };

        Ok(Self {
            gguf,
            mapping,
            mapping_bytes,
            _file: file,
        })
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    pub fn bytes(&self) -> u64 {
        self.mapping_bytes as u64
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.gguf
            .tensors
            .get(name)
            .ok_or_else(|| Error::invalid(format!("required tensor {name:?} is missing")))
    }

    pub fn tensor_bytes(&self, tensor: &TensorInfo) -> Result<&[u8]> {
        let offset = usize::try_from(tensor.absolute_offset)
            .map_err(|_| Error::invalid("tensor offset does not fit the host address space"))?;
        let bytes = usize::try_from(tensor.bytes)
            .map_err(|_| Error::invalid("tensor size does not fit the host address space"))?;
        if offset > self.mapping_bytes || bytes > self.mapping_bytes - offset {
            return Err(Error::invalid("tensor range is outside the mapped model"));
        }
        let pointer = unsafe { self.mapping.as_ptr().cast::<u8>().add(offset) };
        Ok(unsafe { std::slice::from_raw_parts(pointer, bytes) })
    }

    /// Match DwarfStar's `--warm-weights` policy before internal inference
    /// timers: advise and touch one byte from every mapped tensor-data page.
    pub fn warm_tensor_pages(&self) -> Result<ModelWarmReport> {
        let start = usize::try_from(self.gguf.tensor_data_offset).map_err(|_| {
            Error::invalid("tensor-data offset does not fit the host address space")
        })?;
        if start >= self.mapping_bytes {
            return Err(Error::invalid("mapped model has no tensor payload to warm"));
        }
        let page_size = unsafe { getpagesize() };
        if page_size <= 0 {
            return Err(Error::invalid("host page size is invalid"));
        }
        let page_size = page_size as usize;
        let bytes = self.mapping_bytes - start;
        let pointer = unsafe { self.mapping.as_ptr().cast::<u8>().add(start) };
        let started = Instant::now();
        unsafe {
            const POSIX_MADV_WILLNEED: i32 = 3;
            let _ = posix_madvise(pointer.cast(), bytes, POSIX_MADV_WILLNEED);
        }
        let mut checksum = 0_u64;
        for offset in (0..bytes).step_by(page_size) {
            checksum = checksum.wrapping_add(u64::from(unsafe {
                std::ptr::read_volatile(pointer.add(offset))
            }));
        }
        checksum = checksum.wrapping_add(u64::from(unsafe {
            std::ptr::read_volatile(pointer.add(bytes - 1))
        }));
        Ok(ModelWarmReport {
            bytes: bytes as u64,
            pages: bytes.div_ceil(page_size) as u64,
            checksum,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(crate) fn mapping_pointer(&self) -> *const c_void {
        self.mapping.as_ptr()
    }
}

impl Drop for MappedModel {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let _ = munmap(self.mapping.as_ptr(), self.mapping_bytes);
        }
    }
}

#[cfg(unix)]
extern "C" {
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: i32,
        flags: i32,
        file_descriptor: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> i32;
    fn getpagesize() -> i32;
    fn posix_madvise(address: *mut c_void, length: usize, advice: i32) -> i32;
}
