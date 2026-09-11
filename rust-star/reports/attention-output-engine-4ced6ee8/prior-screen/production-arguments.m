        rust_star_q8_mm_id_map_args map_args = {
            .ne02=8, .ne10=n_embd, .ne11=8,
            .nb11=n_embd*sizeof(float), .nb12=8u*n_embd*sizeof(float),
            .ne21=rows, .ne20=8, .nb21=8u*sizeof(int32_t),
        };
        rust_star_q8_mm_id_args low_args = {
            .ne00=n_embd, .ne02=8,
            .nb01=(n_embd/32u)*34u,
            .nb02=(uint64_t)output_rank*((n_embd/32u)*34u),
            .nb03=weights->attn_output_a_bytes,
            .ne11=8, .nb10=sizeof(float),
            .nb11=n_embd*sizeof(float), .nb12=8u*n_embd*sizeof(float),
            .nb13=(uint64_t)rows*8u*n_embd*sizeof(float),
            .ne20=8, .ne21=rows, .ne0=output_rank, .ne1=8,
            .r2=1, .r3=1,
        };
        rust_star_q8_mm_args output_args = {
            .ne00=8u*output_rank, .ne02=1,
            .nb01=((8u*output_rank)/32u)*34u,
            .nb02=weights->attn_output_b_bytes,
            .nb03=weights->attn_output_b_bytes,
            .ne12=1, .nb10=sizeof(float),
            .nb11=8u*output_rank*sizeof(float),
            .nb12=(uint64_t)rows*8u*output_rank*sizeof(float),
            .nb13=(uint64_t)rows*8u*output_rank*sizeof(float),
            .ne0=n_embd, .ne1=rows, .r2=1, .r3=1,
        };
        rust_star_hc_expand_args attention_hc_args = {
            .n_embd=n_embd, .n_hc=4, .n_tokens=rows,
            .nb_block0=sizeof(float), .nb_block1=n_embd*sizeof(float),
            .nb_add0=sizeof(float), .nb_add1=n_embd*sizeof(float),
            .nb_res0=sizeof(float), .nb_res1=n_embd*sizeof(float),
            .nb_res2=4u*n_embd*sizeof(float),
            .nb_post0=sizeof(float), .nb_post1=24u*sizeof(float),
            .nb_comb0=sizeof(float), .nb_comb1=4u*sizeof(float),
            .nb_comb2=24u*sizeof(float),
            .nb0=sizeof(float), .nb1=n_embd*sizeof(float),
            .nb2=4u*n_embd*sizeof(float), .has_add=0,
        };

