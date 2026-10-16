from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.h');s=p.read_text().replace('    /* 0: full logits, 1: full logits plus intermediates, 2: GPU top-1 only. */','    /* 0: logits, 1: intermediates, 2: top-1, 3: checked top-1 plus logits. */');m='int rust_star_metal_run_output_head(';i=s.index(m);s=s[:i]+'''/* Failure diagnostics only. Caller must have completed the output-head wait. */
int rust_star_metal_copy_completed_output_logits(
    void *context, float *logits, uint32_t count,
    char *error, size_t error_bytes);

'''+s[i:];p.write_text(s)
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();i=s.index('int rust_star_metal_run_output_head(');s=s[:i]+'''int rust_star_metal_copy_completed_output_logits(
    void *opaque_context, float *logits, uint32_t count,
    char *error, size_t error_bytes)
{
    if (!opaque_context || !logits || count != 129280u)
        return fail_with_message(error, error_bytes, @"invalid completed-logit copy");
    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;
    id<MTLBuffer> buffer = context.activationBufferCache[@"output_logits"];
    if (!buffer || buffer.length != (NSUInteger)count*sizeof(float))
        return fail_with_message(error, error_bytes, @"completed logits unavailable");
    memcpy(logits, buffer.contents, buffer.length);
    return 1;
}

'''+s[i:];p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();i=s.index('        fn rust_star_metal_run_output_head(');s=s[:i]+'''        fn rust_star_metal_copy_completed_output_logits(
            context: *mut c_void, logits: *mut f32, count: u32,
            error: *mut c_char, error_bytes: usize,
        ) -> i32;
'''+s[i:]
i=s.index('    fn run_top1_output_head(');s=s[:i]+'''    // Read only on a failed transcript, after the output head has synchronized.
    // Preserve the actual failing row rather than recomputing it.
    fn completed_output_failure_details(context: &Context, context_tokens: u32, step: u32) -> Result<String> {
        let mut logits = vec![0.0; 129280];
        let mut error = [0 as c_char; ERROR_BYTES];
        let copied = unsafe { rust_star_metal_copy_completed_output_logits(
            context.0, logits.as_mut_ptr(), logits.len() as u32,
            error.as_mut_ptr(), error.len()) };
        if copied == 0 { return Err(Error::invalid(error_text(&error))); }
        let cpu_token = lowest_id_argmax(&logits)?;
        let mut details = format!("cpu_argmax={cpu_token} actual_logits_checksum={:#018x}", checksum_f32(&logits));
        if context_tokens == 32768 {
            let (_, expected) = prefill_decode_32768_fixture()?;
            let row = &expected[step as usize*129280..(step as usize+1)*129280];
            let differences = logits.iter().zip(row).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
            details.push_str(&format!(" differing_fp32_words={differences}"));
            if let Some((index,(actual,expected))) = logits.iter().zip(row).enumerate().find(|(_, (a,b))|a.to_bits()!=b.to_bits()) {
                details.push_str(&format!(" first_difference={index} actual_bits={:#010x} expected_bits={:#010x}",actual.to_bits(),expected.to_bits()));
            }
        }
        Ok(details)
    }

'''+s[i:]
m='''            step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            if step == 0 || collect_steady_layer_times {''';assert s.count(m)==1
s=s.replace(m,'''            step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            // Failure-only readback occurs after this step's measured interval.
            // Successful runs still collect only the eight-byte GPU selection.
            if selected_token != oracle_inputs[step as usize + 1] {
                let details = completed_output_failure_details(&context, context_tokens, step)
                    .unwrap_or_else(|error| format!("failure_diagnostic_error={error}"));
                return Err(Error::invalid(format!(
                    "timed engine transcript mismatch after generation step {}: actual={} expected={} {details}",
                    step + 1, selected_token, oracle_inputs[step as usize + 1]
                )));
            }
            if step == 0 || collect_steady_layer_times {''');p.write_text(s)
