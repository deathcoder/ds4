        rust_star_q8_mm_id_map_args routed_map_args = {
            .ne02=routed_experts, .ne10=n_embd, .ne11=1,
            .nb11=n_embd*sizeof(float), .nb12=n_embd*sizeof(float),
            .ne21=rows, .ne20=used_experts,
            .nb21=used_experts*sizeof(int32_t),
        };
        rust_star_q8_mm_id_args routed_gate_args = {
            .ne00=n_embd, .ne02=routed_experts,
            .nb01=1056, .nb02=2048ull*1056ull,
            .nb03=256ull*2048ull*1056ull,
            .ne11=1, .nb10=sizeof(float), .nb11=n_embd*sizeof(float),
            .nb12=n_embd*sizeof(float),
            .nb13=(uint64_t)rows*n_embd*sizeof(float),
            .ne20=used_experts, .ne21=rows, .ne0=routed_width,
            .ne1=used_experts, .r2=1, .r3=1,
            .tp_rank=0, .tp_world=0, .tp_expert_base=0,
        };
        rust_star_moe_swiglu_weight_args routed_activation_args = {
            .width=routed_width, .rows=rows*used_experts,
            .gate_row_stride=routed_width*sizeof(float),
            .up_row_stride=routed_width*sizeof(float),
            .mid_row_stride=routed_width*sizeof(uint16_t),
            .weight_stride=sizeof(float), .write_clamped=0, .clamp_value=10.0f,
        };
