        [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline];
        [encoder setBytes:&map_args length:sizeof(map_args) atIndex:0];
        [encoder setBuffer:group_ids_buffer offset:0 atIndex:1];
        [encoder setBuffer:group_map_buffer offset:0 atIndex:2];
        [encoder setBuffer:group_map_buffer offset:tpe_bytes atIndex:3];
        [encoder setBuffer:group_map_buffer offset:work_offset atIndex:4];
        [encoder setThreadgroupMemoryLength:128u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
             threadsPerThreadgroup:MTLSizeMake(8,1,1)];

        [encoder setComputePipelineState:context.attentionOutputBatchLowPipeline];
        [encoder setBytes:&low_args length:sizeof(low_args) atIndex:0];
        [encoder setBuffer:model_buffers[0] offset:inner[0] atIndex:1];
        [encoder setBuffer:heads_buffer offset:heads_offset atIndex:2];
        [encoder setBuffer:group_map_buffer offset:0 atIndex:3];
        [encoder setBuffer:group_map_buffer offset:tpe_bytes atIndex:4];
        [encoder setBuffer:attention_low_buffer offset:0 atIndex:5];
        [encoder setBuffer:group_map_buffer offset:work_offset atIndex:6];
        [encoder setThreadgroupMemoryLength:8192u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(work_cap,output_rank/64u,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.q8PrefillPipeline];
        [encoder setBytes:&output_args length:sizeof(output_args) atIndex:0];
        [encoder setBuffer:model_buffers[1] offset:inner[1] atIndex:1];
        [encoder setBuffer:attention_low_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setThreadgroupMemoryLength:6144u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows/32u,n_embd/64u,1)
             threadsPerThreadgroup:MTLSizeMake(128,1,1)];

        [encoder setComputePipelineState:context.hcExpand4Pipeline];
        [encoder setBytes:&attention_hc_args length:sizeof(attention_hc_args) atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:input_hc offset:input_hc_offset atIndex:2];
        [encoder setBuffer:attn_split
                     offset:split_offset+4u*sizeof(float) atIndex:3];
        [encoder setBuffer:attn_split
                     offset:split_offset+8u*sizeof(float) atIndex:4];
        [encoder setBuffer:output_buffer offset:0 atIndex:5];
        [encoder setBuffer:after_attention_hc_buffer offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((rows*n_embd+255u)/256u,1,1)
             threadsPerThreadgroup:MTLSizeMake(256,1,1)];

