//! Minimal Metal ownership and command-dispatch probe.

use crate::{Error, Result};
use std::io::Write;

pub const PROBE_SCHEMA: &str = "rust-star-metal-dispatch-probe-v1";
pub const DEFAULT_ELEMENTS: u64 = 4096;
pub const DEFAULT_ITERATIONS: u64 = 100;
const MAX_ELEMENTS: u64 = 16 * 1024 * 1024;
const MAX_ITERATIONS: u64 = 100_000;
const MAX_THREAD_INVOCATIONS: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeConfig {
    pub elements: u64,
    pub iterations: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            elements: DEFAULT_ELEMENTS,
            iterations: DEFAULT_ITERATIONS,
        }
    }
}

impl ProbeConfig {
    pub fn validate(self) -> Result<Self> {
        if self.elements == 0 || self.elements > MAX_ELEMENTS {
            return Err(Error::invalid(format!(
                "Metal probe elements must be in 1..={MAX_ELEMENTS}"
            )));
        }
        if self.iterations == 0 || self.iterations > MAX_ITERATIONS {
            return Err(Error::invalid(format!(
                "Metal probe iterations must be in 1..={MAX_ITERATIONS}"
            )));
        }
        let work = self
            .elements
            .checked_mul(self.iterations)
            .ok_or_else(|| Error::invalid("Metal probe work size overflows"))?;
        if work > MAX_THREAD_INVOCATIONS {
            return Err(Error::invalid(format!(
                "Metal probe elements*iterations must not exceed {MAX_THREAD_INVOCATIONS}"
            )));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub device_name: String,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_bytes: u64,
    pub max_total_threads_per_threadgroup: u64,
    pub elements: u64,
    pub iterations: u64,
    pub buffer_bytes: u64,
    pub checksum: u64,
    pub setup_ms: f64,
    pub compile_ms: f64,
    pub warmup_wall_ms: f64,
    pub warmup_gpu_ms: f64,
    pub roundtrip_wall_ms: f64,
    pub roundtrip_gpu_ms: f64,
    pub batched_wall_ms: f64,
    pub batched_gpu_ms: f64,
}

impl ProbeReport {
    pub fn roundtrip_dispatches_per_second(&self) -> f64 {
        rate(self.iterations, self.roundtrip_wall_ms)
    }

    pub fn batched_dispatches_per_second(&self) -> f64 {
        rate(self.iterations, self.batched_wall_ms)
    }

    pub fn roundtrip_thread_invocations_per_second(&self) -> f64 {
        work_rate(self.elements, self.iterations, self.roundtrip_wall_ms)
    }

    pub fn batched_thread_invocations_per_second(&self) -> f64 {
        work_rate(self.elements, self.iterations, self.batched_wall_ms)
    }
}

fn rate(iterations: u64, milliseconds: f64) -> f64 {
    if milliseconds > 0.0 {
        iterations as f64 * 1000.0 / milliseconds
    } else {
        0.0
    }
}

fn work_rate(elements: u64, iterations: u64, milliseconds: f64) -> f64 {
    if milliseconds > 0.0 {
        elements as f64 * iterations as f64 * 1000.0 / milliseconds
    } else {
        0.0
    }
}

pub fn write_probe_json<W: Write>(output: &mut W, report: &ProbeReport) -> Result<()> {
    output.write_all(b"{\n  \"schema\": \"")?;
    output.write_all(PROBE_SCHEMA.as_bytes())?;
    output.write_all(b"\",\n  \"device\": {\n    \"name\": ")?;
    crate::artifact::write_json_string(output, &report.device_name)?;
    write!(
        output,
        ",\n    \"has_unified_memory\": {},\n    \"recommended_max_working_set_bytes\": {},\n    \"max_total_threads_per_threadgroup\": {}\n  }},\n  \"configuration\": {{\n    \"elements\": {},\n    \"iterations\": {},\n    \"buffer_bytes\": {}\n  }},\n  \"setup_ms\": {:.6},\n  \"compile_ms\": {:.6},\n  \"warmup\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6}\n  }},\n  \"roundtrip\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"dispatches_per_second\": {:.6},\n    \"thread_invocations_per_second\": {:.6}\n  }},\n  \"batched\": {{\n    \"wall_ms\": {:.6},\n    \"gpu_ms\": {:.6},\n    \"dispatches_per_second\": {:.6},\n    \"thread_invocations_per_second\": {:.6}\n  }},\n  \"checksum\": {}\n}}\n",
        report.has_unified_memory,
        report.recommended_max_working_set_bytes,
        report.max_total_threads_per_threadgroup,
        report.elements,
        report.iterations,
        report.buffer_bytes,
        report.setup_ms,
        report.compile_ms,
        report.warmup_wall_ms,
        report.warmup_gpu_ms,
        report.roundtrip_wall_ms,
        report.roundtrip_gpu_ms,
        report.roundtrip_dispatches_per_second(),
        report.roundtrip_thread_invocations_per_second(),
        report.batched_wall_ms,
        report.batched_gpu_ms,
        report.batched_dispatches_per_second(),
        report.batched_thread_invocations_per_second(),
        report.checksum,
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr;

    const ERROR_BYTES: usize = 1024;

    #[repr(C)]
    struct RawProbeResult {
        elements: u64,
        iterations: u64,
        recommended_max_working_set_bytes: u64,
        buffer_bytes: u64,
        max_total_threads_per_threadgroup: u64,
        checksum: u64,
        has_unified_memory: u32,
        reserved: u32,
        setup_ms: f64,
        compile_ms: f64,
        warmup_wall_ms: f64,
        warmup_gpu_ms: f64,
        roundtrip_wall_ms: f64,
        roundtrip_gpu_ms: f64,
        batched_wall_ms: f64,
        batched_gpu_ms: f64,
        device_name: [c_char; 256],
    }

    impl Default for RawProbeResult {
        fn default() -> Self {
            Self {
                elements: 0,
                iterations: 0,
                recommended_max_working_set_bytes: 0,
                buffer_bytes: 0,
                max_total_threads_per_threadgroup: 0,
                checksum: 0,
                has_unified_memory: 0,
                reserved: 0,
                setup_ms: 0.0,
                compile_ms: 0.0,
                warmup_wall_ms: 0.0,
                warmup_gpu_ms: 0.0,
                roundtrip_wall_ms: 0.0,
                roundtrip_gpu_ms: 0.0,
                batched_wall_ms: 0.0,
                batched_gpu_ms: 0.0,
                device_name: [0; 256],
            }
        }
    }

    extern "C" {
        fn rust_star_metal_create(
            context_out: *mut *mut c_void,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_run_probe(
            context: *mut c_void,
            elements: u64,
            iterations: u64,
            result: *mut RawProbeResult,
            error: *mut c_char,
            error_bytes: usize,
        ) -> i32;
        fn rust_star_metal_destroy(context: *mut c_void);
    }

    struct Context(*mut c_void);

    impl Drop for Context {
        fn drop(&mut self) {
            unsafe { rust_star_metal_destroy(self.0) };
        }
    }

    fn error_text(buffer: &[c_char; ERROR_BYTES]) -> String {
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn validate_times(raw: &RawProbeResult) -> Result<()> {
        for (name, value) in [
            ("setup_ms", raw.setup_ms),
            ("compile_ms", raw.compile_ms),
            ("warmup_wall_ms", raw.warmup_wall_ms),
            ("warmup_gpu_ms", raw.warmup_gpu_ms),
            ("roundtrip_wall_ms", raw.roundtrip_wall_ms),
            ("roundtrip_gpu_ms", raw.roundtrip_gpu_ms),
            ("batched_wall_ms", raw.batched_wall_ms),
            ("batched_gpu_ms", raw.batched_gpu_ms),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "Metal probe returned invalid {name}"
                )));
            }
        }
        if raw.roundtrip_wall_ms == 0.0 || raw.batched_wall_ms == 0.0 {
            return Err(Error::invalid("Metal probe returned a zero timed interval"));
        }
        Ok(())
    }

    pub fn run_probe(config: ProbeConfig) -> Result<ProbeReport> {
        let config = config.validate()?;
        let mut error = [0 as c_char; ERROR_BYTES];
        let mut pointer = ptr::null_mut();
        let created =
            unsafe { rust_star_metal_create(&mut pointer, error.as_mut_ptr(), error.len()) };
        if created == 0 || pointer.is_null() {
            return Err(Error::invalid(format!(
                "Metal initialization failed: {}",
                error_text(&error)
            )));
        }
        let context = Context(pointer);
        error.fill(0);
        let mut raw = RawProbeResult::default();
        let succeeded = unsafe {
            rust_star_metal_run_probe(
                context.0,
                config.elements,
                config.iterations,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal dispatch probe failed: {}",
                error_text(&error)
            )));
        }
        if raw.elements != config.elements || raw.iterations != config.iterations {
            return Err(Error::invalid("Metal probe returned different dimensions"));
        }
        validate_times(&raw)?;
        let device_name = unsafe { CStr::from_ptr(raw.device_name.as_ptr()) }
            .to_str()
            .map_err(|_| Error::invalid("Metal device name is not UTF-8"))?
            .to_owned();
        if device_name.is_empty() {
            return Err(Error::invalid("Metal device name is empty"));
        }
        Ok(ProbeReport {
            device_name,
            has_unified_memory: raw.has_unified_memory != 0,
            recommended_max_working_set_bytes: raw.recommended_max_working_set_bytes,
            max_total_threads_per_threadgroup: raw.max_total_threads_per_threadgroup,
            elements: raw.elements,
            iterations: raw.iterations,
            buffer_bytes: raw.buffer_bytes,
            checksum: raw.checksum,
            setup_ms: raw.setup_ms,
            compile_ms: raw.compile_ms,
            warmup_wall_ms: raw.warmup_wall_ms,
            warmup_gpu_ms: raw.warmup_gpu_ms,
            roundtrip_wall_ms: raw.roundtrip_wall_ms,
            roundtrip_gpu_ms: raw.roundtrip_gpu_ms,
            batched_wall_ms: raw.batched_wall_ms,
            batched_gpu_ms: raw.batched_gpu_ms,
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    pub fn run_probe(config: ProbeConfig) -> Result<ProbeReport> {
        config.validate()?;
        Err(Error::invalid(
            "the Metal dispatch probe is available only on macOS",
        ))
    }
}

pub use imp::run_probe;

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> ProbeReport {
        ProbeReport {
            device_name: "Apple GPU \"fixture\"".to_owned(),
            has_unified_memory: true,
            recommended_max_working_set_bytes: 100,
            max_total_threads_per_threadgroup: 1024,
            elements: 4,
            iterations: 2,
            buffer_bytes: 16,
            checksum: 42,
            setup_ms: 1.0,
            compile_ms: 2.0,
            warmup_wall_ms: 3.0,
            warmup_gpu_ms: 1.0,
            roundtrip_wall_ms: 4.0,
            roundtrip_gpu_ms: 2.0,
            batched_wall_ms: 1.0,
            batched_gpu_ms: 0.5,
        }
    }

    #[test]
    fn validates_probe_work_bounds() {
        assert!(ProbeConfig::default().validate().is_ok());
        assert!(ProbeConfig {
            elements: 0,
            iterations: 1,
        }
        .validate()
        .is_err());
        assert!(ProbeConfig {
            elements: MAX_ELEMENTS,
            iterations: MAX_ITERATIONS,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn writes_stable_probe_json() {
        let mut output = Vec::new();
        write_probe_json(&mut output, &report()).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(&format!("\"schema\": \"{PROBE_SCHEMA}\"")));
        assert!(text.contains("Apple GPU \\\"fixture\\\""));
        assert!(text.contains("\"dispatches_per_second\": 2000"));
        assert!(text.contains("\"checksum\": 42"));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_probe_is_explicitly_unsupported() {
        let error = run_probe(ProbeConfig::default()).unwrap_err().to_string();
        assert!(error.contains("only on macOS"), "{error}");
    }
}
