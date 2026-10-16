from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text();i=s.index('static int encode_projected_compressor_batch_ratio4(');helper='''/* Match oracle-v3's host-waited M1 compressor dependency boundary. An
 * asynchronous split alone was insufficient in the reference's 32K audit.
 * Keep both command buffers in the caller's wall/GPU timing totals. */
static int finish_compressor_projection(
    RustStarMetalContext *context,
    id<MTLCommandBuffer> __strong *command,
    id<MTLComputeCommandEncoder> __strong *encoder,
    double *projection_gpu_ms,
    char *error, size_t error_bytes)
{
    [*encoder endEncoding];
    [*command commit];
    if (!command_succeeded(*command, error, error_bytes)) return 0;
    *projection_gpu_ms += gpu_elapsed_ms(*command);
    *command = [context.queue commandBuffer];
    *encoder = [*command computeCommandEncoder];
    if (!*command || !*encoder)
        return fail_with_message(error, error_bytes,
            @"failed to reopen compressor consumer command");
    return 1;
}

''';s=s[:i]+helper+s[i:]
for name,macro in [('rust_star_metal_run_prefill_layer3_continuation_batch_compressor','RUST_STAR_ENCODE_L3_BATCH_PROJECTION'),('rust_star_metal_run_prefill_layer4_continuation_boundary','RUST_STAR_ENCODE_L4_PROJECTION')]:
 a=s.index('int '+name+'(');b=s.index('\nint ',a+1);part=s[a:b];assert part.count('const double wall_start = monotonic_ms();')==1
 part=part.replace('        const double wall_start = monotonic_ms();\n','')
 marker='#undef '+macro+'\n';assert part.count(marker)==1
 part=part.replace(marker,marker+'''        const double wall_start = monotonic_ms();
        double projection_gpu_ms = 0.0;
        if (!finish_compressor_projection(context, &command, &encoder,
                &projection_gpu_ms, error, error_bytes)) return 0;
''')
 assert part.count('result->gpu_ms += gpu_elapsed_ms(command);')==1
 part=part.replace('result->gpu_ms += gpu_elapsed_ms(command);','result->gpu_ms += projection_gpu_ms + gpu_elapsed_ms(command);');s=s[:a]+part+s[b:]
p.write_text(s)
