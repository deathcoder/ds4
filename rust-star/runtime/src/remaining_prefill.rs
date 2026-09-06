// Shared execution for the remaining fixed-model transformer layers.
// The arithmetic and dispatch order follow the validated layer-13/14 paths.
fn run_remaining_prefill_layers(
    model: &MappedModel,
    context: &Context,
) -> Result<Vec<RemainingPrefillLayerReport>> {
    let release_completed = |layer| -> Result<()> {
        let mut error = [0 as c_char; ERROR_BYTES];
        let ok = unsafe {
            rust_star_metal_release_remaining_prefill_transients(
                context.0,
                layer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ok == 0 {
            return Err(Error::invalid(error_text(&error)));
        }
        Ok(())
    };
    release_completed(14)?;
    let mut reports = Vec::new();
    for &(layer, prefix, continuation, prefix_tail, continuation_tail) in
        REMAINING_PREFILL_CHECKSUMS
    {
        let result = if layer % 2 == 1 {
            run_remaining_odd(
                model,
                context,
                layer,
                prefix,
                continuation,
                prefix_tail,
                continuation_tail,
            )
        } else {
            run_remaining_even(
                model,
                context,
                layer,
                prefix,
                continuation,
                prefix_tail,
                continuation_tail,
            )
        }
        .map_err(|error| Error::invalid(format!("remaining layer {layer}: {error}")))?;
        eprintln!("rust-star: complete retained layer {layer} exact");
        reports.push(result);
        release_completed(layer)?;
    }
    Ok(reports)
}

fn run_remaining_odd(
    model: &MappedModel,
    context: &Context,
    layer: u32,
    prefix: &[u64],
    continuation: &[u64],
    prefix_tail: [u64; 3],
    continuation_tail: [u64; 3],
) -> Result<RemainingPrefillLayerReport> {
    let mut error = [0 as c_char; 1024];
    let mut layer23_loop = RawLayer23ContinuationLoopResult::default();
    let prefill_layer13_prefix_boundary_checksums: [u64; 5] = prefix
        .try_into()
        .map_err(|_| Error::invalid("invalid oracle boundary width"))?;
    let prefill_layer13_prefix_tail_checksums: [u64; 3] = prefix_tail;
    let prefill_layer13_continuation_tail_checksums: [u64; 3] = continuation_tail;
    let prefill_layer13_continuation_ingress_checksums: [u64; 4] =
        continuation[..4].try_into().unwrap();
    let prefill_layer13_continuation_compressed_checksum: u64 = continuation[4];
    let layer13_hc_fn = exact_tensor(
        model,
        &format!("blk.{layer}.hc_attn_fn.weight"),
        1,
        &[16384, 24],
    )?;
    let layer13_hc_scale =
        exact_tensor(model, &format!("blk.{layer}.hc_attn_scale.weight"), 0, &[3])?;
    let layer13_hc_base =
        exact_tensor(model, &format!("blk.{layer}.hc_attn_base.weight"), 0, &[24])?;
    let layer13_norm = exact_tensor(model, &format!("blk.{layer}.attn_norm.weight"), 0, &[4096])?;
    let layer13_q_a = exact_tensor(
        model,
        &format!("blk.{layer}.attn_q_a.weight"),
        8,
        &[4096, 1024],
    )?;
    let layer13_q_a_norm = exact_tensor(
        model,
        &format!("blk.{layer}.attn_q_a_norm.weight"),
        0,
        &[1024],
    )?;
    let layer13_kv = exact_tensor(
        model,
        &format!("blk.{layer}.attn_kv.weight"),
        8,
        &[4096, 512],
    )?;
    let layer13_kv_norm = exact_tensor(
        model,
        &format!("blk.{layer}.attn_kv_a_norm.weight"),
        0,
        &[512],
    )?;
    let layer13_q_b = exact_tensor(
        model,
        &format!("blk.{layer}.attn_q_b.weight"),
        8,
        &[1024, 32768],
    )?;
    let layer13_attn_ape = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_ape.weight"),
        1,
        &[512, 128],
    )?;
    let layer13_attn_kv = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_kv.weight"),
        1,
        &[4096, 512],
    )?;
    let layer13_attn_gate = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_gate.weight"),
        1,
        &[4096, 512],
    )?;
    let layer13_attn_compressor_norm = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_norm.weight"),
        0,
        &[512],
    )?;
    let layer13_sinks = exact_tensor(model, &format!("blk.{layer}.attn_sinks.weight"), 0, &[64])?;
    let layer13_output_a = exact_tensor(
        model,
        &format!("blk.{layer}.attn_output_a.weight"),
        8,
        &[4096, 8192],
    )?;
    let layer13_output_b = exact_tensor(
        model,
        &format!("blk.{layer}.attn_output_b.weight"),
        8,
        &[8192, 4096],
    )?;
    let layer13_ffn_hc_fn = exact_tensor(
        model,
        &format!("blk.{layer}.hc_ffn_fn.weight"),
        1,
        &[16384, 24],
    )?;
    let layer13_ffn_hc_scale =
        exact_tensor(model, &format!("blk.{layer}.hc_ffn_scale.weight"), 0, &[3])?;
    let layer13_ffn_hc_base =
        exact_tensor(model, &format!("blk.{layer}.hc_ffn_base.weight"), 0, &[24])?;
    let layer13_ffn_norm =
        exact_tensor(model, &format!("blk.{layer}.ffn_norm.weight"), 0, &[4096])?;
    let layer13_router_gate = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        1,
        &[4096, 256],
    )?;
    let layer13_router_bias =
        exact_tensor(model, &format!("blk.{layer}.exp_probs_b.bias"), 0, &[256])?;
    let layer13_routed_gate = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_gate_exps.weight"),
        16,
        &[4096, 2048, 256],
    )?;
    let layer13_routed_up = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_up_exps.weight"),
        16,
        &[4096, 2048, 256],
    )?;
    let layer13_routed_down = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_down_exps.weight"),
        10,
        &[2048, 4096, 256],
    )?;
    let layer13_shared_gate = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_gate_shexp.weight"),
        8,
        &[4096, 2048],
    )?;
    let layer13_shared_up = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_up_shexp.weight"),
        8,
        &[4096, 2048],
    )?;
    let layer13_shared_down = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_down_shexp.weight"),
        8,
        &[2048, 4096],
    )?;
    let layer13_weights = RawPrefillLayerWeights {
        ingress: RawPrefillAttentionIngressWeights {
            hc_fn_offset: layer13_hc_fn.absolute_offset,
            hc_fn_bytes: layer13_hc_fn.bytes,
            hc_scale_offset: layer13_hc_scale.absolute_offset,
            hc_scale_bytes: layer13_hc_scale.bytes,
            hc_base_offset: layer13_hc_base.absolute_offset,
            hc_base_bytes: layer13_hc_base.bytes,
            norm_offset: layer13_norm.absolute_offset,
            norm_bytes: layer13_norm.bytes,
            q_a_offset: layer13_q_a.absolute_offset,
            q_a_bytes: layer13_q_a.bytes,
        },
        q_a_norm_offset: layer13_q_a_norm.absolute_offset,
        q_a_norm_bytes: layer13_q_a_norm.bytes,
        kv_offset: layer13_kv.absolute_offset,
        kv_bytes: layer13_kv.bytes,
        kv_norm_offset: layer13_kv_norm.absolute_offset,
        kv_norm_bytes: layer13_kv_norm.bytes,
        q_b_offset: layer13_q_b.absolute_offset,
        q_b_bytes: layer13_q_b.bytes,
        attn_sinks_offset: layer13_sinks.absolute_offset,
        attn_sinks_bytes: layer13_sinks.bytes,
        attn_output_a_offset: layer13_output_a.absolute_offset,
        attn_output_a_bytes: layer13_output_a.bytes,
        attn_output_b_offset: layer13_output_b.absolute_offset,
        attn_output_b_bytes: layer13_output_b.bytes,
        ffn: RawPrefillFfnWeights {
            hc_fn_offset: layer13_ffn_hc_fn.absolute_offset,
            hc_fn_bytes: layer13_ffn_hc_fn.bytes,
            hc_scale_offset: layer13_ffn_hc_scale.absolute_offset,
            hc_scale_bytes: layer13_ffn_hc_scale.bytes,
            hc_base_offset: layer13_ffn_hc_base.absolute_offset,
            hc_base_bytes: layer13_ffn_hc_base.bytes,
            norm_offset: layer13_ffn_norm.absolute_offset,
            norm_bytes: layer13_ffn_norm.bytes,
            router_gate_offset: layer13_router_gate.absolute_offset,
            router_gate_bytes: layer13_router_gate.bytes,
            router_hash_offset: layer13_router_bias.absolute_offset,
            router_hash_bytes: layer13_router_bias.bytes,
            routed_gate_offset: layer13_routed_gate.absolute_offset,
            routed_gate_bytes: layer13_routed_gate.bytes,
            routed_up_offset: layer13_routed_up.absolute_offset,
            routed_up_bytes: layer13_routed_up.bytes,
            routed_down_offset: layer13_routed_down.absolute_offset,
            routed_down_bytes: layer13_routed_down.bytes,
            shared_gate_offset: layer13_shared_gate.absolute_offset,
            shared_gate_bytes: layer13_shared_gate.bytes,
            shared_up_offset: layer13_shared_up.absolute_offset,
            shared_up_bytes: layer13_shared_up.bytes,
            shared_down_offset: layer13_shared_down.absolute_offset,
            shared_down_bytes: layer13_shared_down.bytes,
        },
    };
    let layer13_compressor = RawPrefillCompressorWeights {
        attn_ape_offset: layer13_attn_ape.absolute_offset,
        attn_ape_bytes: layer13_attn_ape.bytes,
        attn_kv_offset: layer13_attn_kv.absolute_offset,
        attn_kv_bytes: layer13_attn_kv.bytes,
        attn_gate_offset: layer13_attn_gate.absolute_offset,
        attn_gate_bytes: layer13_attn_gate.bytes,
        attn_norm_offset: layer13_attn_compressor_norm.absolute_offset,
        attn_norm_bytes: layer13_attn_compressor_norm.bytes,
        ..RawPrefillCompressorWeights::default()
    };
    let mut layer5_hc_attn_pre = vec![0.0_f32; 32 * 4_096];
    let mut layer5_attn_norm = vec![0.0_f32; 32 * 4_096];
    let mut layer5_q_lora = vec![0.0_f32; 32 * 1_024];
    let mut layer5_q_lora_norm = vec![0.0_f32; 32 * 1_024];
    let mut layer5_kv_raw = vec![0.0_f32; 32 * 512];
    let mut layer5_kv_norm = vec![0.0_f32; 32 * 512];
    let mut layer5_q_current = vec![0.0_f32; 32 * 32_768];
    let mut layer5_kv_rope = vec![0.0_f32; 32 * 512];
    let mut layer5_kv_current = vec![0.0_f32; 32 * 512];
    let layer13_prefix_begin_succeeded = unsafe {
        rust_star_metal_begin_prefill_remaining_odd_prefix_rebuild(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_begin_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal odd-layer prefix binding failed: {}",
            error_text(&error)
        )));
    }
    let mut layer13_prefix_raw = RawSparseIndexedResult::default();
    let layer13_prefix_ingress_succeeded = unsafe {
        rust_star_metal_run_prefill_layer5_prefix_ingress(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer13_weights,
            &mut layer13_prefix_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_ingress_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer prefix ingress failed: {}",
            error_text(&error)
        )));
    }
    let layer13_prefix_compressor_succeeded = unsafe {
        rust_star_metal_run_prefill_layer3_continuation_batch_compressor(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer13_compressor,
            0,
            &mut layer13_prefix_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_compressor_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer prefix compressor failed: {}",
            error_text(&error)
        )));
    }
    let layer13_prefix_checksum_succeeded = unsafe {
        rust_star_metal_checksum_prefill_layer7_prefix_boundary(
            context.0,
            layer23_loop.layer13_prefix_boundary_checksums.as_mut_ptr(),
            layer23_loop.layer13_prefix_boundary_checksums.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_checksum_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer prefix checksum failed: {}",
            error_text(&error)
        )));
    }
    if layer13_prefix_raw.dispatches != 17
        || layer13_prefix_raw.wrapped_model_ranges != 13
        || layer13_prefix_raw.pointer_matches != 13
        || layer23_loop.layer13_prefix_boundary_checksums
            != prefill_layer13_prefix_boundary_checksums
    {
        return Err(Error::invalid(format!(
                "Metal production-batch odd-layer prefix boundary is invalid: dispatches={}, mappings={}/{}, checksums={:?} expected={:?}",
                layer13_prefix_raw.dispatches,
                layer13_prefix_raw.pointer_matches,
                layer13_prefix_raw.wrapped_model_ranges,
                layer23_loop.layer13_prefix_boundary_checksums,
                prefill_layer13_prefix_boundary_checksums,
            )));
    }
    layer23_loop.layer13_prefix_boundary_dispatches = layer13_prefix_raw.dispatches;
    layer23_loop.layer13_prefix_boundary_wrapped_model_ranges =
        layer13_prefix_raw.wrapped_model_ranges;
    layer23_loop.layer13_prefix_boundary_pointer_matches = layer13_prefix_raw.pointer_matches;
    let layer13_prefix_attention_succeeded = unsafe {
        rust_star_metal_run_prefill_layer3_continuation_batch_attention(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            layer13_sinks.absolute_offset,
            layer13_sinks.bytes,
            0,
            &mut layer13_prefix_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_attention_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer prefix attention failed: {}",
            error_text(&error)
        )));
    }
    let layer13_prefix_tail_succeeded = unsafe {
        rust_star_metal_run_prefill_layer2_continuation_tail(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer13_weights,
            5,
            0,
            4_096,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut layer13_prefix_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_tail_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer prefix attention/FFN tail failed: {}",
            error_text(&error)
        )));
    }
    let layer13_prefix_tail_checksum_succeeded = unsafe {
        rust_star_metal_checksum_prefill_layer5_prefix_tail(
            context.0,
            layer23_loop.layer13_prefix_tail_checksums.as_mut_ptr(),
            layer23_loop.layer13_prefix_tail_checksums.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_prefix_tail_checksum_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer prefix tail checksum failed: {}",
            error_text(&error)
        )));
    }
    if layer13_prefix_raw.dispatches != 49
        || layer13_prefix_raw.wrapped_model_ranges != 28
        || layer13_prefix_raw.pointer_matches != 28
        || layer23_loop.layer13_prefix_tail_checksums != prefill_layer13_prefix_tail_checksums
    {
        return Err(Error::invalid(format!(
                "Metal complete odd-layer prefix is invalid: dispatches={}, mappings={}/{}, checksums={:?} expected={:?}",
                layer13_prefix_raw.dispatches,
                layer13_prefix_raw.pointer_matches,
                layer13_prefix_raw.wrapped_model_ranges,
                layer23_loop.layer13_prefix_tail_checksums,
                prefill_layer13_prefix_tail_checksums,
            )));
    }
    layer23_loop.layer13_prefix_complete_dispatches = layer13_prefix_raw.dispatches;
    layer23_loop.layer13_prefix_complete_wrapped_model_ranges =
        layer13_prefix_raw.wrapped_model_ranges;
    layer23_loop.layer13_prefix_complete_pointer_matches = layer13_prefix_raw.pointer_matches;

    let layer13_continuation_begin_succeeded = unsafe {
        rust_star_metal_begin_prefill_remaining_odd_continuation(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_continuation_begin_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal odd-layer continuation binding failed: {}",
            error_text(&error)
        )));
    }
    layer23_loop.layer13_continuation_ingress_checksums = [0xcbf2_9ce4_8422_2325_u64; 4];
    for tile_start in (4_096_u32..=8_160_u32).step_by(32) {
        let prepared = unsafe {
            rust_star_metal_prepare_prefill_layer5_continuation_tile(
                context.0,
                tile_start,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if prepared == 0 {
            return Err(Error::invalid(format!(
                "Metal odd-layer continuation tile {tile_start} staging failed: {}",
                error_text(&error)
            )));
        }
        let mut tile = RawSparseIndexedResult::default();
        let ingress_succeeded = unsafe {
            rust_star_metal_run_prefill_layer3_continuation_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &layer13_weights,
                tile_start,
                layer5_hc_attn_pre.as_mut_ptr(),
                layer5_attn_norm.as_mut_ptr(),
                layer5_q_lora.as_mut_ptr(),
                layer5_q_lora_norm.as_mut_ptr(),
                layer5_kv_raw.as_mut_ptr(),
                layer5_kv_norm.as_mut_ptr(),
                layer5_q_current.as_mut_ptr(),
                layer5_kv_rope.as_mut_ptr(),
                layer5_kv_current.as_mut_ptr(),
                &mut tile,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ingress_succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal odd-layer continuation ingress tile {tile_start} failed: {}",
                error_text(&error)
            )));
        }
        let checksums = &mut layer23_loop.layer13_continuation_ingress_checksums;
        checksums[0] = extend_checksum_f32(checksums[0], &layer5_hc_attn_pre);
        checksums[1] = extend_checksum_f32(checksums[1], &layer5_attn_norm);
        checksums[2] = extend_checksum_f32(checksums[2], &layer5_q_current);
        checksums[3] = extend_checksum_f32(checksums[3], &layer5_kv_current);
        layer23_loop.layer13_continuation_tiles += 1;
        layer23_loop.layer13_continuation_dispatches += tile.dispatches;
        layer23_loop.layer13_continuation_wrapped_model_ranges += tile.wrapped_model_ranges;
        layer23_loop.layer13_continuation_pointer_matches += tile.pointer_matches;
        layer23_loop.wall_ms += tile.wall_ms;
        layer23_loop.gpu_ms += tile.gpu_ms;
    }
    let mut layer13_production = RawSparseIndexedResult::default();
    let layer13_compressor_succeeded = unsafe {
        rust_star_metal_run_prefill_layer3_continuation_batch_compressor(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer13_compressor,
            4_096,
            &mut layer13_production,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_compressor_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer compressor failed: {}",
            error_text(&error)
        )));
    }
    let layer13_compressor_finish_succeeded = unsafe {
        rust_star_metal_finish_prefill_layer5_continuation(
            context.0,
            &mut layer23_loop.layer13_continuation_compressed_checksum,
            1,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_compressor_finish_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal complete odd-layer continuation checksum failed: {}",
            error_text(&error)
        )));
    }
    let layer13_attention_succeeded = unsafe {
        rust_star_metal_run_prefill_layer3_continuation_batch_attention(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            layer13_sinks.absolute_offset,
            layer13_sinks.bytes,
            4_096,
            &mut layer13_production,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_attention_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer continuation attention failed: {}",
            error_text(&error)
        )));
    }
    let layer13_tail_succeeded = unsafe {
        rust_star_metal_run_prefill_layer2_continuation_tail(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer13_weights,
            5,
            4_096,
            4_096,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut layer13_production,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_tail_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer continuation attention/FFN tail failed: {}",
            error_text(&error)
        )));
    }
    let layer13_tail_checksum_succeeded = unsafe {
        rust_star_metal_checksum_prefill_layer5_continuation_tail(
            context.0,
            layer23_loop
                .layer13_continuation_tail_checksums
                .as_mut_ptr(),
            layer23_loop.layer13_continuation_tail_checksums.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_tail_checksum_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch odd-layer continuation tail checksum failed: {}",
            error_text(&error)
        )));
    }
    if layer13_production.dispatches != 38
        || layer13_production.wrapped_model_ranges != 19
        || layer13_production.pointer_matches != 19
        || layer23_loop.layer13_continuation_tail_checksums
            != prefill_layer13_continuation_tail_checksums
    {
        return Err(Error::invalid(format!(
                "Metal complete odd-layer continuation is invalid: dispatches={}, mappings={}/{}, checksums={:?} expected={:?}",
                layer13_production.dispatches,
                layer13_production.pointer_matches,
                layer13_production.wrapped_model_ranges,
                layer23_loop.layer13_continuation_tail_checksums,
                prefill_layer13_continuation_tail_checksums,
            )));
    }
    layer23_loop.layer13_continuation_complete_dispatches = layer13_production.dispatches;
    layer23_loop.layer13_continuation_complete_wrapped_model_ranges =
        layer13_production.wrapped_model_ranges;
    layer23_loop.layer13_continuation_complete_pointer_matches = layer13_production.pointer_matches;
    layer23_loop.layer13_continuation_dispatches += layer13_production.dispatches;
    layer23_loop.layer13_continuation_wrapped_model_ranges +=
        layer13_production.wrapped_model_ranges;
    layer23_loop.layer13_continuation_pointer_matches += layer13_production.pointer_matches;
    layer23_loop.wall_ms += layer13_production.wall_ms;
    layer23_loop.gpu_ms += layer13_production.gpu_ms;
    if layer23_loop.layer13_continuation_tiles != 128
        || layer23_loop.layer13_continuation_dispatches != 1_318
        || layer23_loop.layer13_continuation_wrapped_model_ranges != 1_171
        || layer23_loop.layer13_continuation_pointer_matches != 1_171
        || layer23_loop.layer13_continuation_ingress_checksums
            != prefill_layer13_continuation_ingress_checksums
        || layer23_loop.layer13_continuation_compressed_checksum
            != prefill_layer13_continuation_compressed_checksum
    {
        return Err(Error::invalid(format!(
                "Metal odd-layer continuation boundary is invalid: tiles={}, dispatches={}, mappings={}/{}, ingress={:?} expected={:?}, compressed={} expected={}",
                layer23_loop.layer13_continuation_tiles,
                layer23_loop.layer13_continuation_dispatches,
                layer23_loop.layer13_continuation_pointer_matches,
                layer23_loop.layer13_continuation_wrapped_model_ranges,
                layer23_loop.layer13_continuation_ingress_checksums,
                prefill_layer13_continuation_ingress_checksums,
                layer23_loop.layer13_continuation_compressed_checksum,
                prefill_layer13_continuation_compressed_checksum,
            )));
    }
    let layer13_finish_succeeded = unsafe {
        rust_star_metal_finish_prefill_remaining_odd_rebuild(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer13_finish_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal odd-layer rebuild finalization failed: {}",
            error_text(&error)
        )));
    }
    Ok(RemainingPrefillLayerReport {
        layer,
        prefix_boundary: layer23_loop.layer13_prefix_boundary_checksums.to_vec(),
        continuation_boundary: [
            layer23_loop.layer13_continuation_ingress_checksums.to_vec(),
            vec![layer23_loop.layer13_continuation_compressed_checksum],
        ]
        .concat(),
        prefix_tail: layer23_loop.layer13_prefix_tail_checksums,
        continuation_tail: layer23_loop.layer13_continuation_tail_checksums,
        dispatches: (layer13_prefix_raw.dispatches + layer23_loop.layer13_continuation_dispatches)
            as u64,
        model_mappings: (layer13_prefix_raw.wrapped_model_ranges
            + layer23_loop.layer13_continuation_wrapped_model_ranges)
            as u64,
        pointer_matches: (layer13_prefix_raw.pointer_matches
            + layer23_loop.layer13_continuation_pointer_matches) as u64,
    })
}

fn run_remaining_even(
    model: &MappedModel,
    context: &Context,
    layer: u32,
    prefix: &[u64],
    continuation: &[u64],
    prefix_tail: [u64; 3],
    continuation_tail: [u64; 3],
) -> Result<RemainingPrefillLayerReport> {
    let mut error = [0 as c_char; 1024];
    let mut layer23_loop = RawLayer23ContinuationLoopResult::default();
    let prefill_layer14_prefix_boundary_checksums: [u64; 10] = prefix
        .try_into()
        .map_err(|_| Error::invalid("invalid oracle boundary width"))?;
    let prefill_layer14_continuation_boundary_checksums: [u64; 10] = continuation
        .try_into()
        .map_err(|_| Error::invalid("invalid oracle boundary width"))?;
    let prefill_layer14_prefix_tail_checksums: [u64; 3] = prefix_tail;
    let prefill_layer14_continuation_tail_checksums: [u64; 3] = continuation_tail;
    let layer14_hc_fn = exact_tensor(
        model,
        &format!("blk.{layer}.hc_attn_fn.weight"),
        1,
        &[16384, 24],
    )?;
    let layer14_hc_scale =
        exact_tensor(model, &format!("blk.{layer}.hc_attn_scale.weight"), 0, &[3])?;
    let layer14_hc_base =
        exact_tensor(model, &format!("blk.{layer}.hc_attn_base.weight"), 0, &[24])?;
    let layer14_norm = exact_tensor(model, &format!("blk.{layer}.attn_norm.weight"), 0, &[4096])?;
    let layer14_q_a = exact_tensor(
        model,
        &format!("blk.{layer}.attn_q_a.weight"),
        8,
        &[4096, 1024],
    )?;
    let layer14_q_a_norm = exact_tensor(
        model,
        &format!("blk.{layer}.attn_q_a_norm.weight"),
        0,
        &[1024],
    )?;
    let layer14_kv = exact_tensor(
        model,
        &format!("blk.{layer}.attn_kv.weight"),
        8,
        &[4096, 512],
    )?;
    let layer14_kv_norm = exact_tensor(
        model,
        &format!("blk.{layer}.attn_kv_a_norm.weight"),
        0,
        &[512],
    )?;
    let layer14_q_b = exact_tensor(
        model,
        &format!("blk.{layer}.attn_q_b.weight"),
        8,
        &[1024, 32768],
    )?;
    let layer14_attn_ape = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_ape.weight"),
        1,
        &[1024, 4],
    )?;
    let layer14_attn_kv = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_kv.weight"),
        1,
        &[4096, 1024],
    )?;
    let layer14_attn_gate = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_gate.weight"),
        1,
        &[4096, 1024],
    )?;
    let layer14_attn_compressor_norm = exact_tensor(
        model,
        &format!("blk.{layer}.attn_compressor_norm.weight"),
        0,
        &[512],
    )?;
    let layer14_indexer_ape = exact_tensor(
        model,
        &format!("blk.{layer}.indexer_compressor_ape.weight"),
        1,
        &[256, 4],
    )?;
    let layer14_indexer_kv = exact_tensor(
        model,
        &format!("blk.{layer}.indexer_compressor_kv.weight"),
        1,
        &[4096, 256],
    )?;
    let layer14_indexer_gate = exact_tensor(
        model,
        &format!("blk.{layer}.indexer_compressor_gate.weight"),
        1,
        &[4096, 256],
    )?;
    let layer14_indexer_compressor_norm = exact_tensor(
        model,
        &format!("blk.{layer}.indexer_compressor_norm.weight"),
        0,
        &[128],
    )?;
    let layer14_indexer_q = exact_tensor(
        model,
        &format!("blk.{layer}.indexer.attn_q_b.weight"),
        1,
        &[1024, 8192],
    )?;
    let layer14_indexer_weight = exact_tensor(
        model,
        &format!("blk.{layer}.indexer.proj.weight"),
        1,
        &[4096, 64],
    )?;
    let layer14_sinks = exact_tensor(model, &format!("blk.{layer}.attn_sinks.weight"), 0, &[64])?;
    let layer14_output_a = exact_tensor(
        model,
        &format!("blk.{layer}.attn_output_a.weight"),
        8,
        &[4096, 8192],
    )?;
    let layer14_output_b = exact_tensor(
        model,
        &format!("blk.{layer}.attn_output_b.weight"),
        8,
        &[8192, 4096],
    )?;
    let layer14_ffn_hc_fn = exact_tensor(
        model,
        &format!("blk.{layer}.hc_ffn_fn.weight"),
        1,
        &[16384, 24],
    )?;
    let layer14_ffn_hc_scale =
        exact_tensor(model, &format!("blk.{layer}.hc_ffn_scale.weight"), 0, &[3])?;
    let layer14_ffn_hc_base =
        exact_tensor(model, &format!("blk.{layer}.hc_ffn_base.weight"), 0, &[24])?;
    let layer14_ffn_norm =
        exact_tensor(model, &format!("blk.{layer}.ffn_norm.weight"), 0, &[4096])?;
    let layer14_router_gate = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        1,
        &[4096, 256],
    )?;
    let layer14_router_bias =
        exact_tensor(model, &format!("blk.{layer}.exp_probs_b.bias"), 0, &[256])?;
    let layer14_routed_gate = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_gate_exps.weight"),
        16,
        &[4096, 2048, 256],
    )?;
    let layer14_routed_up = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_up_exps.weight"),
        16,
        &[4096, 2048, 256],
    )?;
    let layer14_routed_down = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_down_exps.weight"),
        10,
        &[2048, 4096, 256],
    )?;
    let layer14_shared_gate = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_gate_shexp.weight"),
        8,
        &[4096, 2048],
    )?;
    let layer14_shared_up = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_up_shexp.weight"),
        8,
        &[4096, 2048],
    )?;
    let layer14_shared_down = exact_tensor(
        model,
        &format!("blk.{layer}.ffn_down_shexp.weight"),
        8,
        &[2048, 4096],
    )?;
    let layer14_weights = RawPrefillLayerWeights {
        ingress: RawPrefillAttentionIngressWeights {
            hc_fn_offset: layer14_hc_fn.absolute_offset,
            hc_fn_bytes: layer14_hc_fn.bytes,
            hc_scale_offset: layer14_hc_scale.absolute_offset,
            hc_scale_bytes: layer14_hc_scale.bytes,
            hc_base_offset: layer14_hc_base.absolute_offset,
            hc_base_bytes: layer14_hc_base.bytes,
            norm_offset: layer14_norm.absolute_offset,
            norm_bytes: layer14_norm.bytes,
            q_a_offset: layer14_q_a.absolute_offset,
            q_a_bytes: layer14_q_a.bytes,
        },
        q_a_norm_offset: layer14_q_a_norm.absolute_offset,
        q_a_norm_bytes: layer14_q_a_norm.bytes,
        kv_offset: layer14_kv.absolute_offset,
        kv_bytes: layer14_kv.bytes,
        kv_norm_offset: layer14_kv_norm.absolute_offset,
        kv_norm_bytes: layer14_kv_norm.bytes,
        q_b_offset: layer14_q_b.absolute_offset,
        q_b_bytes: layer14_q_b.bytes,
        attn_sinks_offset: layer14_sinks.absolute_offset,
        attn_sinks_bytes: layer14_sinks.bytes,
        attn_output_a_offset: layer14_output_a.absolute_offset,
        attn_output_a_bytes: layer14_output_a.bytes,
        attn_output_b_offset: layer14_output_b.absolute_offset,
        attn_output_b_bytes: layer14_output_b.bytes,
        ffn: RawPrefillFfnWeights {
            hc_fn_offset: layer14_ffn_hc_fn.absolute_offset,
            hc_fn_bytes: layer14_ffn_hc_fn.bytes,
            hc_scale_offset: layer14_ffn_hc_scale.absolute_offset,
            hc_scale_bytes: layer14_ffn_hc_scale.bytes,
            hc_base_offset: layer14_ffn_hc_base.absolute_offset,
            hc_base_bytes: layer14_ffn_hc_base.bytes,
            norm_offset: layer14_ffn_norm.absolute_offset,
            norm_bytes: layer14_ffn_norm.bytes,
            router_gate_offset: layer14_router_gate.absolute_offset,
            router_gate_bytes: layer14_router_gate.bytes,
            router_hash_offset: layer14_router_bias.absolute_offset,
            router_hash_bytes: layer14_router_bias.bytes,
            routed_gate_offset: layer14_routed_gate.absolute_offset,
            routed_gate_bytes: layer14_routed_gate.bytes,
            routed_up_offset: layer14_routed_up.absolute_offset,
            routed_up_bytes: layer14_routed_up.bytes,
            routed_down_offset: layer14_routed_down.absolute_offset,
            routed_down_bytes: layer14_routed_down.bytes,
            shared_gate_offset: layer14_shared_gate.absolute_offset,
            shared_gate_bytes: layer14_shared_gate.bytes,
            shared_up_offset: layer14_shared_up.absolute_offset,
            shared_up_bytes: layer14_shared_up.bytes,
            shared_down_offset: layer14_shared_down.absolute_offset,
            shared_down_bytes: layer14_shared_down.bytes,
        },
    };
    let layer14_compressor = RawPrefillCompressorWeights {
        attn_ape_offset: layer14_attn_ape.absolute_offset,
        attn_ape_bytes: layer14_attn_ape.bytes,
        attn_kv_offset: layer14_attn_kv.absolute_offset,
        attn_kv_bytes: layer14_attn_kv.bytes,
        attn_gate_offset: layer14_attn_gate.absolute_offset,
        attn_gate_bytes: layer14_attn_gate.bytes,
        attn_norm_offset: layer14_attn_compressor_norm.absolute_offset,
        attn_norm_bytes: layer14_attn_compressor_norm.bytes,
        indexer_ape_offset: layer14_indexer_ape.absolute_offset,
        indexer_ape_bytes: layer14_indexer_ape.bytes,
        indexer_kv_offset: layer14_indexer_kv.absolute_offset,
        indexer_kv_bytes: layer14_indexer_kv.bytes,
        indexer_gate_offset: layer14_indexer_gate.absolute_offset,
        indexer_gate_bytes: layer14_indexer_gate.bytes,
        indexer_norm_offset: layer14_indexer_compressor_norm.absolute_offset,
        indexer_norm_bytes: layer14_indexer_compressor_norm.bytes,
    };
    let mut layer4_indexer_scores = vec![0.0_f32; 32 * 2_048];
    let mut layer4_indexer_topk = vec![0_i32; 32 * 512];
    let mut layer4_kqv_back = vec![0.0_f32; 32 * 32_768];
    let mut layer4_attention_low = vec![0.0_f32; 32 * 8_192];
    let mut layer4_attention_output = vec![0.0_f32; 32 * 4_096];
    let mut layer4_after_attention_hc = vec![0.0_f32; 32 * 16_384];
    let mut layer4_ffn_current = vec![0.0_f32; 32 * 4_096];
    let mut layer4_ffn_norm = vec![0.0_f32; 32 * 4_096];
    let mut layer4_router_logits = vec![0.0_f32; 32 * 256];
    let mut layer4_router_probs = vec![0.0_f32; 32 * 256];
    let mut layer4_router_selected = vec![0_i32; 32 * 6];
    let mut layer4_router_weights = vec![0.0_f32; 32 * 6];
    let mut layer4_routed_mid = vec![0.0_f32; 32 * 6 * 2_048];
    let mut layer4_routed_output = vec![0.0_f32; 32 * 4_096];
    let mut layer4_shared_output = vec![0.0_f32; 32 * 4_096];
    let mut layer4_after_ffn_hc = vec![0.0_f32; 32 * 16_384];
    let layer14_begin_succeeded = unsafe {
        rust_star_metal_begin_prefill_remaining_even_rebuild(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_begin_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal even-layer rebuild binding failed: {}",
            error_text(&error)
        )));
    }
    let mut layer14_prefix_raw = RawSparseIndexedResult::default();
    let layer14_prefix_succeeded = unsafe {
        rust_star_metal_run_prefill_layer4_continuation_boundary(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer14_weights,
            &layer14_compressor,
            0,
            layer23_loop.layer14_prefix_boundary_checksums.as_mut_ptr(),
            layer23_loop.layer14_prefix_boundary_checksums.len(),
            &mut layer14_prefix_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_prefix_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch even-layer prefix boundary failed: {}",
            error_text(&error)
        )));
    }
    layer23_loop.layer14_prefix_boundary_dispatches = layer14_prefix_raw.dispatches;
    layer23_loop.layer14_prefix_boundary_wrapped_model_ranges =
        layer14_prefix_raw.wrapped_model_ranges;
    layer23_loop.layer14_prefix_boundary_pointer_matches = layer14_prefix_raw.pointer_matches;
    if layer14_prefix_raw.dispatches != 40
        || layer14_prefix_raw.wrapped_model_ranges != 17
        || layer14_prefix_raw.pointer_matches != 17
        || layer23_loop.layer14_prefix_boundary_checksums
            != prefill_layer14_prefix_boundary_checksums
    {
        return Err(Error::invalid(format!(
                "Metal production-batch even-layer prefix boundary is invalid: dispatches={}, mappings={}/{}, checksums={:?} expected={:?}",
                layer14_prefix_raw.dispatches,
                layer14_prefix_raw.pointer_matches,
                layer14_prefix_raw.wrapped_model_ranges,
                layer23_loop.layer14_prefix_boundary_checksums,
                prefill_layer14_prefix_boundary_checksums,
            )));
    }
    let mut layer14_prefix_sparse_checksums = [0_u64; 5];
    let mut layer14_prefix_complete_raw = RawSparseIndexedResult::default();
    let layer14_prefix_sparse_succeeded = unsafe {
        rust_star_metal_run_prefill_layer4_prefix_sparse_batch(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            layer14_indexer_q.absolute_offset,
            layer14_indexer_q.bytes,
            layer14_indexer_weight.absolute_offset,
            layer14_indexer_weight.bytes,
            layer14_sinks.absolute_offset,
            layer14_sinks.bytes,
            layer14_prefix_sparse_checksums.as_mut_ptr(),
            layer14_prefix_sparse_checksums.len(),
            &mut layer14_prefix_complete_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_prefix_sparse_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch even-layer prefix sparse attention failed: {}",
            error_text(&error)
        )));
    }
    let layer14_prefix_tail_succeeded = unsafe {
        rust_star_metal_run_prefill_layer2_continuation_tail(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer14_weights,
            4,
            0,
            4_096,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut layer14_prefix_complete_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_prefix_tail_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch even-layer prefix attention/FFN tail failed: {}",
            error_text(&error)
        )));
    }
    let mut layer14_prefix_hc_checksums = [0_u64; 2];
    let layer14_prefix_hc_succeeded = unsafe {
        rust_star_metal_checksum_prefill_layer4_prefix_hc(
            context.0,
            layer14_prefix_hc_checksums.as_mut_ptr(),
            layer14_prefix_hc_checksums.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_prefix_hc_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch even-layer prefix HC checksum failed: {}",
            error_text(&error)
        )));
    }
    let layer14_prefix_tail_checksums = [
        layer14_prefix_sparse_checksums[4],
        layer14_prefix_hc_checksums[0],
        layer14_prefix_hc_checksums[1],
    ];
    if layer14_prefix_complete_raw.dispatches != 36
        || layer14_prefix_complete_raw.wrapped_model_ranges != 17
        || layer14_prefix_complete_raw.pointer_matches != 17
        || layer14_prefix_tail_checksums != prefill_layer14_prefix_tail_checksums
    {
        return Err(Error::invalid(format!(
                "Metal complete even-layer prefix is invalid: dispatches={}, mappings={}/{}, checksums={:?} expected={:?}",
                layer14_prefix_complete_raw.dispatches,
                layer14_prefix_complete_raw.pointer_matches,
                layer14_prefix_complete_raw.wrapped_model_ranges,
                layer14_prefix_tail_checksums,
                prefill_layer14_prefix_tail_checksums,
            )));
    }
    layer23_loop.layer14_prefix_complete_dispatches = layer14_prefix_complete_raw.dispatches;
    layer23_loop.layer14_prefix_complete_wrapped_model_ranges =
        layer14_prefix_complete_raw.wrapped_model_ranges;
    layer23_loop.layer14_prefix_complete_pointer_matches =
        layer14_prefix_complete_raw.pointer_matches;
    layer23_loop.layer14_prefix_tail_checksums = layer14_prefix_tail_checksums;
    let layer14_prefix_retain_succeeded = unsafe {
        rust_star_metal_retain_prefill_remaining_even_prefix_hc(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_prefix_retain_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal complete even-layer prefix retention failed: {}",
            error_text(&error)
        )));
    }
    let mut layer14_continuation_raw = RawSparseIndexedResult::default();
    let layer14_continuation_succeeded = unsafe {
        rust_star_metal_run_prefill_layer4_continuation_boundary(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &layer14_weights,
            &layer14_compressor,
            4_096,
            layer23_loop
                .layer14_continuation_boundary_checksums
                .as_mut_ptr(),
            layer23_loop.layer14_continuation_boundary_checksums.len(),
            &mut layer14_continuation_raw,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_continuation_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal production-batch even-layer continuation boundary failed: {}",
            error_text(&error)
        )));
    }
    layer23_loop.layer14_continuation_boundary_dispatches = layer14_continuation_raw.dispatches;
    layer23_loop.layer14_continuation_boundary_wrapped_model_ranges =
        layer14_continuation_raw.wrapped_model_ranges;
    layer23_loop.layer14_continuation_boundary_pointer_matches =
        layer14_continuation_raw.pointer_matches;
    if layer14_continuation_raw.dispatches != 40
        || layer14_continuation_raw.wrapped_model_ranges != 17
        || layer14_continuation_raw.pointer_matches != 17
        || layer23_loop.layer14_continuation_boundary_checksums
            != prefill_layer14_continuation_boundary_checksums
    {
        return Err(Error::invalid(format!(
                "Metal production-batch even-layer continuation boundary is invalid: dispatches={}, mappings={}/{}, checksums={:?} expected={:?}",
                layer14_continuation_raw.dispatches,
                layer14_continuation_raw.pointer_matches,
                layer14_continuation_raw.wrapped_model_ranges,
                layer23_loop.layer14_continuation_boundary_checksums,
                prefill_layer14_continuation_boundary_checksums,
            )));
    }
    layer23_loop.layer14_continuation_tail_checksums = [0xcbf2_9ce4_8422_2325_u64; 3];
    for tile_start in (4_096_u32..=8_160_u32).step_by(32) {
        let mut tile = RawSparseIndexedResult::default();
        let layer14_sparse_succeeded = unsafe {
            rust_star_metal_run_prefill_layer4_continuation_sparse_tile(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                layer14_indexer_q.absolute_offset,
                layer14_indexer_q.bytes,
                layer14_indexer_weight.absolute_offset,
                layer14_indexer_weight.bytes,
                layer14_sinks.absolute_offset,
                layer14_sinks.bytes,
                tile_start,
                layer4_indexer_scores.as_mut_ptr(),
                layer4_indexer_topk.as_mut_ptr(),
                layer4_kqv_back.as_mut_ptr(),
                &mut tile,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if layer14_sparse_succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal even-layer continuation sparse tile {tile_start} failed: {}",
                error_text(&error)
            )));
        }
        let layer14_tail_succeeded = unsafe {
            rust_star_metal_run_prefill_layer2_continuation_tail(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &layer14_weights,
                4,
                tile_start,
                32,
                layer4_attention_low.as_mut_ptr(),
                layer4_attention_output.as_mut_ptr(),
                layer4_after_attention_hc.as_mut_ptr(),
                layer4_ffn_current.as_mut_ptr(),
                layer4_ffn_norm.as_mut_ptr(),
                layer4_router_logits.as_mut_ptr(),
                layer4_router_probs.as_mut_ptr(),
                layer4_router_selected.as_mut_ptr(),
                layer4_router_weights.as_mut_ptr(),
                layer4_routed_mid.as_mut_ptr(),
                layer4_routed_output.as_mut_ptr(),
                layer4_shared_output.as_mut_ptr(),
                layer4_after_ffn_hc.as_mut_ptr(),
                &mut tile,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if layer14_tail_succeeded == 0 {
            return Err(Error::invalid(format!(
                "Metal even-layer continuation tail {tile_start} failed: {}",
                error_text(&error)
            )));
        }
        if tile.dispatches != 36 || tile.wrapped_model_ranges != 17 || tile.pointer_matches != 17 {
            return Err(Error::invalid(format!(
                    "Metal even-layer continuation tile {tile_start} metadata is invalid: dispatches={}, mappings={}/{}",
                    tile.dispatches, tile.pointer_matches, tile.wrapped_model_ranges,
                )));
        }
        let checksums = &mut layer23_loop.layer14_continuation_tail_checksums;
        checksums[0] = extend_checksum_f32(checksums[0], &layer4_kqv_back);
        checksums[1] = extend_checksum_f32(checksums[1], &layer4_after_attention_hc);
        checksums[2] = extend_checksum_f32(checksums[2], &layer4_after_ffn_hc);
        layer23_loop.layer14_continuation_tiles += 1;
        layer23_loop.layer14_continuation_dispatches += tile.dispatches;
        layer23_loop.layer14_continuation_wrapped_model_ranges += tile.wrapped_model_ranges;
        layer23_loop.layer14_continuation_pointer_matches += tile.pointer_matches;
        layer23_loop.wall_ms += tile.wall_ms;
        layer23_loop.gpu_ms += tile.gpu_ms;
    }
    let mut layer14_continuation_full_hc_checksum = 0_u64;
    let layer14_continuation_hc_succeeded = unsafe {
        rust_star_metal_checksum_prefill_layer4_continuation_hc(
            context.0,
            &mut layer14_continuation_full_hc_checksum,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_continuation_hc_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal complete even-layer continuation HC checksum failed: {}",
            error_text(&error)
        )));
    }
    if layer23_loop.layer14_continuation_tiles != 128
        || layer23_loop.layer14_continuation_dispatches != 4_608
        || layer23_loop.layer14_continuation_wrapped_model_ranges != 2_176
        || layer23_loop.layer14_continuation_pointer_matches != 2_176
        || layer23_loop.layer14_continuation_tail_checksums
            != prefill_layer14_continuation_tail_checksums
        || layer14_continuation_full_hc_checksum != prefill_layer14_continuation_tail_checksums[2]
    {
        return Err(Error::invalid(format!(
                "Metal complete even-layer continuation is invalid: tiles={}, dispatches={}, mappings={}/{}, checksums={:?} expected={:?}, full_hc={} expected={}",
                layer23_loop.layer14_continuation_tiles,
                layer23_loop.layer14_continuation_dispatches,
                layer23_loop.layer14_continuation_pointer_matches,
                layer23_loop.layer14_continuation_wrapped_model_ranges,
                layer23_loop.layer14_continuation_tail_checksums,
                prefill_layer14_continuation_tail_checksums,
                layer14_continuation_full_hc_checksum,
                prefill_layer14_continuation_tail_checksums[2],
            )));
    }
    let layer14_finish_succeeded = unsafe {
        rust_star_metal_finish_prefill_remaining_even_rebuild(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if layer14_finish_succeeded == 0 {
        return Err(Error::invalid(format!(
            "Metal even-layer rebuild finalization failed: {}",
            error_text(&error)
        )));
    }
    Ok(RemainingPrefillLayerReport {
        layer,
        prefix_boundary: layer23_loop.layer14_prefix_boundary_checksums.to_vec(),
        continuation_boundary: layer23_loop
            .layer14_continuation_boundary_checksums
            .to_vec(),
        prefix_tail: layer23_loop.layer14_prefix_tail_checksums,
        continuation_tail: layer23_loop.layer14_continuation_tail_checksums,
        dispatches: (layer14_prefix_raw.dispatches
            + layer14_prefix_complete_raw.dispatches
            + layer14_continuation_raw.dispatches
            + layer23_loop.layer14_continuation_dispatches) as u64,
        model_mappings: (layer14_prefix_raw.wrapped_model_ranges
            + layer14_prefix_complete_raw.wrapped_model_ranges
            + layer14_continuation_raw.wrapped_model_ranges
            + layer23_loop.layer14_continuation_wrapped_model_ranges)
            as u64,
        pointer_matches: (layer14_prefix_raw.pointer_matches
            + layer14_prefix_complete_raw.pointer_matches
            + layer14_continuation_raw.pointer_matches
            + layer23_loop.layer14_continuation_pointer_matches) as u64,
    })
}

fn run_remaining_prefill_output(
    model: &MappedModel,
    context: &Context,
) -> Result<RemainingPrefillOutputReport> {
    let (_, expected) = prefill_frontier_8192_fixture()?;
    let mut prepared = PreparedOutputHead::new(model)?;
    let (selected_token, _) = run_sampling_output_head(model, context, &mut prepared)?;
    if prepared.logits.len() != expected.len() {
        return Err(Error::invalid(
            "8K output vocabulary length differs from oracle",
        ));
    }
    for (index, (actual, expected)) in prepared.logits.iter().zip(&expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(Error::invalid(format!(
                "8K full-logit C0 mismatch at {index}: actual={:#010x} expected={:#010x}",
                actual.to_bits(),
                expected.to_bits()
            )));
        }
    }
    Ok(RemainingPrefillOutputReport {
        checked_logits: expected.len() as u32,
        logits_checksum: checksum_f32(&prepared.logits),
        selected_token,
    })
}
