from pathlib import Path
p=Path('rust-star/runtime/src/metal_shim.m');s=p.read_text()
start=s.index('int rust_star_metal_adopt_prefill_decoder_state(')
bs=s.index('            id<MTLBuffer> source_raw',start)
be=s.index('\n        [blit endEncoding];\n        const double wall_start',bs)
body=s[bs:be];assert body.endswith('\n        }')
body=body[:-len('\n        }')]
body='\n'.join(line[8:] if line.startswith('        ') else line for line in body.splitlines())
body=body.replace('        [blit endEncoding];\n','').replace('if (layer < 2u) continue;','if (layer < 2u) return 1;').replace('if (ratio != 4u) continue;','if (ratio != 4u) return 1;').replace('copies++;','(*copies)++;').replace('copies += 3u;','*copies += 3u;')
helper='''/* Encode the same exact KV/state copies for both handoff schedules. */
static int encode_prefill_layer_handoff(
    RustStarMetalContext *context, id<MTLBlitCommandEncoder> blit,
    uint32_t layer, uint32_t prefill_rows, uint32_t decoder_context_capacity,
    uint32_t *copies, char *error, size_t error_bytes)
{
    const uint32_t raw_rows = 128u;
    const NSUInteger kv_row_bytes = 512u*sizeof(float);
    const uint32_t ratio4_rows = prefill_rows/4u;
    const uint32_t ratio128_rows = prefill_rows/128u;
'''+body+'''
    return 1;
}

static void release_completed_prefill_layer_history(
    RustStarMetalContext *context, uint32_t layer)
{
    NSDictionary *state = context.prefillLayerChunkStates[@(layer)];
    /* Executor slots can still own the final histories. Clear only aliases of
     * this completed layer; other layers keep their dictionary ownership. */
    for (NSString *role in state) {
        for (uint32_t slot = 0u; slot < 43u; slot++) {
            NSString *key = [NSString stringWithFormat:@"prefillLayer%u%@", slot, role];
            if ([context respondsToSelector:NSSelectorFromString(key)] &&
                [context valueForKey:key] == state[role]) {
                [context setValue:nil forKey:key];
            }
        }
    }
    [context.prefillLayerChunkStates removeObjectForKey:@(layer)];
}

'''
s=s[:start]+helper+s[start:]
start=s.index('int rust_star_metal_adopt_prefill_decoder_state(')
a=s.index('        id<MTLCommandBuffer> command =',start)
b=s.index('        [context.chainedCommands removeAllObjects];',a)
s=s[:a]+'''        /* Above 128K, one command retaining every source history plus the
         * attached model residency set can exceed the GPU working set on a
         * repeated request. Bound each command to one layer and retire that
         * history after completion. Keep the established <=128K schedule. */
        const uint32_t layers_per_command = prefill_rows > 131072u ? 1u : layers;
        uint32_t copies = 0u, commands = 0u;
        double wall_ms = 0.0, gpu_ms = 0.0;
        for (uint32_t first = 0u; first < layers; first += layers_per_command) {
            @autoreleasepool {
                id<MTLCommandBuffer> command = [context.queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                if (!command || !blit) {
                    return fail_with_message(error, error_bytes,
                        @"prefill-to-decode handoff could not create its blit command");
                }
                for (uint32_t layer = first; layer < first+layers_per_command; layer++) {
                    if (!encode_prefill_layer_handoff(context, blit, layer,
                            prefill_rows, decoder_context_capacity, &copies, error, error_bytes)) {
                        [blit endEncoding];
                        return 0;
                    }
                }
                [blit endEncoding];
                const double wall_start = monotonic_ms();
                [command commit];
                if (!command_succeeded(command, error, error_bytes)) return 0;
                wall_ms += monotonic_ms()-wall_start;
                gpu_ms += gpu_elapsed_ms(command);
                commands++;
                if (layers_per_command == 1u)
                    release_completed_prefill_layer_history(context, first);
            }
        }
'''+s[b:]
a=s.index('        result->command_buffers = 1u;',start)
b=s.index('        return 1;',a)
s=s[:a]+'''        result->command_buffers = commands;
        result->host_waits = commands;
        result->wall_ms = wall_ms;
        result->gpu_ms = gpu_ms;
'''+s[b:]
s=s.replace('    const NSUInteger kv_row_bytes = 512u*sizeof(float);\n    if (!opaque_context || !result)', '    if (!opaque_context || !result)',1)
p.write_text(s)
p=Path('rust-star/runtime/src/metal.rs');s=p.read_text();a=s.index('        fn adopt_prefill_decoder_state(');b=s.index('        fn compressed_kv_row(',a)
part=s[a:b].replace('            if !(matches!(prefill_rows,', '            let expected_commands = if prefill_rows > 131_072 { 43 } else { 1 };\n            if !(matches!(prefill_rows,').replace('raw.command_buffers != 1','raw.command_buffers != expected_commands').replace('raw.host_waits != 1','raw.host_waits != expected_commands');s=s[:a]+part+s[b:];p.write_text(s)
