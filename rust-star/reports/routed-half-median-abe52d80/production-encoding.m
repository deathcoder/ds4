        [encoder setComputePipelineState:context.routedBatchMapPipeline];
        [encoder setBytes:&routed_map_args length:sizeof(routed_map_args) atIndex:0];
        [encoder setBuffer:router_selected_buffer offset:0 atIndex:1];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:2];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:3];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:4];
        [encoder setThreadgroupMemoryLength:256u*6u*sizeof(uint16_t) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        if (tail_samples) {
            encoder = next_tail_profile_encoder(encoder, command, tail_samples, tail_fence, 4u);
            if (!encoder) return fail_with_message(error, error_bytes, @"failed to create tail profile encoder");
        }
        // The native 4K shader transposes only grid axes. Keep the legacy
        // 32-row diagnostic and initial-prefix pipeline/geometry unchanged.
        [encoder setComputePipelineState:native_chunk
            ? context.routedNativePairSwigluPipeline
            : context.routedBatchPairSwigluPipeline];
        [encoder setBytes:&routed_gate_args length:sizeof(routed_gate_args) atIndex:0];
        [encoder setBytes:&routed_activation_args length:sizeof(routed_activation_args) atIndex:1];
        [encoder setBuffer:model_buffers[8] offset:inner[8] atIndex:2];
        [encoder setBuffer:model_buffers[9] offset:inner[9] atIndex:3];
        [encoder setBuffer:ffn_norm_buffer offset:0 atIndex:4];
        [encoder setBuffer:routed_map_buffer offset:0 atIndex:5];
        [encoder setBuffer:routed_map_buffer offset:routed_map_tpe_bytes atIndex:6];
        [encoder setBuffer:routed_mid_buffer offset:0 atIndex:7];
        [encoder setBuffer:router_weights_buffer offset:0 atIndex:8];
        [encoder setBuffer:routed_map_buffer offset:routed_map_work_offset atIndex:9];
        [encoder setThreadgroupMemoryLength:16384u atIndex:0];
        [encoder dispatchThreadgroups:native_chunk
            ? MTLSizeMake(32,routed_map_work_cap,1)
            : MTLSizeMake(routed_map_work_cap,32,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        if (tail_samples) {
            encoder = next_tail_profile_encoder(encoder, command, tail_samples, tail_fence, 5u);
            if (!encoder) return fail_with_message(error, error_bytes, @"failed to create tail profile encoder");
        }
