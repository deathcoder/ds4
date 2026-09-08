// Stage totals observe existing completed command buffers; they add no GPU waits.
// Bootstrap ranges are explicit because their shared scheduler spans several layers.
struct LongPrefillStage {
    chunk: u32,
    first: u32,
    last: u32,
    stage: &'static str,
    calls: u32,
    elapsed: f64,
    wall: f64,
    gpu: f64,
}

#[derive(Default)]
struct LongPrefillProfile {
    enabled: bool,
    rows: Vec<LongPrefillStage>,
    tail_components: Vec<String>,
}
impl LongPrefillProfile {
    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }
    fn record(
        &mut self,
        chunk: u32,
        first: u32,
        last: u32,
        stage: &'static str,
        start: Option<Instant>,
        before: (f64, f64),
        after: (f64, f64),
    ) {
        if let Some(start) = start {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let wall = after.0 - before.0;
            let gpu = after.1 - before.1;
            if let Some(row) = self.rows.iter_mut().rev().find(|r| {
                r.chunk == chunk && r.first == first && r.last == last && r.stage == stage
            }) {
                row.calls += 1;
                row.elapsed += elapsed;
                row.wall += wall;
                row.gpu += gpu;
            } else {
                self.rows.push(LongPrefillStage {
                    chunk,
                    first,
                    last,
                    stage,
                    calls: 1,
                    elapsed,
                    wall,
                    gpu,
                });
            }
        }
    }
    fn record_tail(
        &mut self,
        context: &Context,
        chunk: u32,
        layer: u32,
        gpu_ms: f64,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let mut ticks = [0u64; 18];
        let mut error = [0 as c_char; ERROR_BYTES];
        if unsafe {
            rust_star_metal_copy_prefill_tail_ticks(
                context.0,
                ticks.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        } == 0
        {
            return Err(Error::invalid(format!(
                "tail profile chunk={chunk} layer={layer}: {}",
                error_text(&error)
            )));
        }
        self.record_tail_ticks(chunk, layer, gpu_ms, &ticks)
    }
    fn record_tail_ticks(
        &mut self,
        chunk: u32,
        layer: u32,
        gpu_ms: f64,
        ticks: &[u64; 18],
    ) -> Result<()> {
        if ticks.contains(&u64::MAX) || ticks.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(Error::invalid(format!(
                "invalid tail timestamps chunk={chunk} layer={layer}: {ticks:x?}"
            )));
        }
        let names = [
            "attention_output",
            "ffn_ingress",
            "router",
            "routed_map",
            "routed_gate_up",
            "routed_down",
            "routed_sum",
            "shared_expert",
            "final_hc",
        ];
        if !gpu_ms.is_finite() || gpu_ms <= 0.0 {
            return Err(Error::invalid("invalid tail command GPU duration"));
        }
        let spans = ticks
            .chunks_exact(2)
            .map(|p| p[1] - p[0])
            .collect::<Vec<_>>();
        let total = spans.iter().map(|x| u128::from(*x)).sum::<u128>() as f64;
        for (i, name) in names.iter().enumerate() {
            self.tail_components.push(format!(
                "{{\"chunk_start\":{chunk},\"layer\":{layer},\"component\":\"{name}\",\"start_tick\":{},\"end_tick\":{},\"attributed_gpu_ms\":{:.9}}}",
                ticks[2*i], ticks[2*i+1], gpu_ms * spans[i] as f64 / total));
        }
        Ok(())
    }
    fn json(self) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let rows = self.rows.iter().map(|r| format!(
            "{{\"chunk_start\":{},\"first_layer\":{},\"last_layer\":{},\"stage\":\"{}\",\"calls\":{},\"host_elapsed_ms\":{:.9},\"command_wall_ms\":{:.9},\"gpu_ms\":{:.9}}}",
            r.chunk,r.first,r.last,r.stage,r.calls,r.elapsed,r.wall,r.gpu)).collect::<Vec<_>>().join(",");
        let components = self.tail_components.join(",");
        Some(format!("{{\"method\":\"existing-command-completion-and-host-stage-timers\",\"paired_ineligible\":true,\"rows\":[{rows}],\"tail_components\":{{\"method\":\"metal-compute-stage-boundary-timestamp-counters\",\"profile_perturbed\":true,\"encoder_splitting\":true,\"scaling\":\"counter shares scaled to each completed tail command GPU interval; do not add to parent rows\",\"rows\":[{components}]}}}}"))
    }
}

fn prefill_output_ptr<T>(output: &mut [T], collect: bool) -> *mut T {
    if collect {
        output.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    }
}

// Fixed-model 4096-token continuation chunks after the exact 4K prefix.
struct PreparedPrefillChunkLayer {
    weights: RawPrefillLayerWeights,
    compressor: RawPrefillCompressorWeights,
    indexer_q: (u64, u64),
    indexer_weight: (u64, u64),
}
impl PreparedPrefillChunkLayer {
    fn new(model: &MappedModel, layer: u32) -> Result<Self> {
        let tensor = |suffix: &str, ty, dims: &[u64]| -> Result<(u64, u64)> {
            let tensor = exact_tensor(model, &format!("blk.{layer}.{suffix}"), ty, dims)?;
            Ok((tensor.absolute_offset, tensor.bytes))
        };
        let hc_fn = tensor("hc_attn_fn.weight", 1, &[16384, 24])?;
        let hc_scale = tensor("hc_attn_scale.weight", 0, &[3])?;
        let hc_base = tensor("hc_attn_base.weight", 0, &[24])?;
        let norm = tensor("attn_norm.weight", 0, &[4096])?;
        let q_a = tensor("attn_q_a.weight", 8, &[4096, 1024])?;
        let q_a_norm = tensor("attn_q_a_norm.weight", 0, &[1024])?;
        let kv = tensor("attn_kv.weight", 8, &[4096, 512])?;
        let kv_norm = tensor("attn_kv_a_norm.weight", 0, &[512])?;
        let q_b = tensor("attn_q_b.weight", 8, &[1024, 32768])?;
        let attn_ape = if layer % 2 == 0 {
            tensor("attn_compressor_ape.weight", 1, &[1024, 4])?
        } else {
            tensor("attn_compressor_ape.weight", 1, &[512, 128])?
        };
        let attn_kv = if layer % 2 == 0 {
            tensor("attn_compressor_kv.weight", 1, &[4096, 1024])?
        } else {
            tensor("attn_compressor_kv.weight", 1, &[4096, 512])?
        };
        let attn_gate = if layer % 2 == 0 {
            tensor("attn_compressor_gate.weight", 1, &[4096, 1024])?
        } else {
            tensor("attn_compressor_gate.weight", 1, &[4096, 512])?
        };
        let attn_compressor_norm = tensor("attn_compressor_norm.weight", 0, &[512])?;
        let indexer_ape = if layer % 2 == 0 {
            tensor("indexer_compressor_ape.weight", 1, &[256, 4])?
        } else {
            (0, 0)
        };
        let indexer_kv = if layer % 2 == 0 {
            tensor("indexer_compressor_kv.weight", 1, &[4096, 256])?
        } else {
            (0, 0)
        };
        let indexer_gate = if layer % 2 == 0 {
            tensor("indexer_compressor_gate.weight", 1, &[4096, 256])?
        } else {
            (0, 0)
        };
        let indexer_compressor_norm = if layer % 2 == 0 {
            tensor("indexer_compressor_norm.weight", 0, &[128])?
        } else {
            (0, 0)
        };
        let indexer_q = if layer % 2 == 0 {
            tensor("indexer.attn_q_b.weight", 1, &[1024, 8192])?
        } else {
            (0, 0)
        };
        let indexer_weight = if layer % 2 == 0 {
            tensor("indexer.proj.weight", 1, &[4096, 64])?
        } else {
            (0, 0)
        };
        let sinks = tensor("attn_sinks.weight", 0, &[64])?;
        let output_a = tensor("attn_output_a.weight", 8, &[4096, 8192])?;
        let output_b = tensor("attn_output_b.weight", 8, &[8192, 4096])?;
        let ffn_hc_fn = tensor("hc_ffn_fn.weight", 1, &[16384, 24])?;
        let ffn_hc_scale = tensor("hc_ffn_scale.weight", 0, &[3])?;
        let ffn_hc_base = tensor("hc_ffn_base.weight", 0, &[24])?;
        let ffn_norm = tensor("ffn_norm.weight", 0, &[4096])?;
        let router_gate = tensor("ffn_gate_inp.weight", 1, &[4096, 256])?;
        let router_bias = if layer == 2 {
            tensor("ffn_gate_tid2eid.weight", 26, &[6, 129280])?
        } else {
            tensor("exp_probs_b.bias", 0, &[256])?
        };
        let routed_gate = tensor("ffn_gate_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_up = tensor("ffn_up_exps.weight", 16, &[4096, 2048, 256])?;
        let routed_down = tensor("ffn_down_exps.weight", 10, &[2048, 4096, 256])?;
        let shared_gate = tensor("ffn_gate_shexp.weight", 8, &[4096, 2048])?;
        let shared_up = tensor("ffn_up_shexp.weight", 8, &[4096, 2048])?;
        let shared_down = tensor("ffn_down_shexp.weight", 8, &[2048, 4096])?;
        let weights = RawPrefillLayerWeights {
            ingress: RawPrefillAttentionIngressWeights {
                hc_fn_offset: hc_fn.0,
                hc_fn_bytes: hc_fn.1,
                hc_scale_offset: hc_scale.0,
                hc_scale_bytes: hc_scale.1,
                hc_base_offset: hc_base.0,
                hc_base_bytes: hc_base.1,
                norm_offset: norm.0,
                norm_bytes: norm.1,
                q_a_offset: q_a.0,
                q_a_bytes: q_a.1,
            },
            q_a_norm_offset: q_a_norm.0,
            q_a_norm_bytes: q_a_norm.1,
            kv_offset: kv.0,
            kv_bytes: kv.1,
            kv_norm_offset: kv_norm.0,
            kv_norm_bytes: kv_norm.1,
            q_b_offset: q_b.0,
            q_b_bytes: q_b.1,
            attn_sinks_offset: sinks.0,
            attn_sinks_bytes: sinks.1,
            attn_output_a_offset: output_a.0,
            attn_output_a_bytes: output_a.1,
            attn_output_b_offset: output_b.0,
            attn_output_b_bytes: output_b.1,
            ffn: RawPrefillFfnWeights {
                hc_fn_offset: ffn_hc_fn.0,
                hc_fn_bytes: ffn_hc_fn.1,
                hc_scale_offset: ffn_hc_scale.0,
                hc_scale_bytes: ffn_hc_scale.1,
                hc_base_offset: ffn_hc_base.0,
                hc_base_bytes: ffn_hc_base.1,
                norm_offset: ffn_norm.0,
                norm_bytes: ffn_norm.1,
                router_gate_offset: router_gate.0,
                router_gate_bytes: router_gate.1,
                router_hash_offset: router_bias.0,
                router_hash_bytes: router_bias.1,
                routed_gate_offset: routed_gate.0,
                routed_gate_bytes: routed_gate.1,
                routed_up_offset: routed_up.0,
                routed_up_bytes: routed_up.1,
                routed_down_offset: routed_down.0,
                routed_down_bytes: routed_down.1,
                shared_gate_offset: shared_gate.0,
                shared_gate_bytes: shared_gate.1,
                shared_up_offset: shared_up.0,
                shared_up_bytes: shared_up.1,
                shared_down_offset: shared_down.0,
                shared_down_bytes: shared_down.1,
            },
        };
        let compressor = RawPrefillCompressorWeights {
            attn_ape_offset: attn_ape.0,
            attn_ape_bytes: attn_ape.1,
            attn_kv_offset: attn_kv.0,
            attn_kv_bytes: attn_kv.1,
            attn_gate_offset: attn_gate.0,
            attn_gate_bytes: attn_gate.1,
            attn_norm_offset: attn_compressor_norm.0,
            attn_norm_bytes: attn_compressor_norm.1,
            indexer_ape_offset: indexer_ape.0,
            indexer_ape_bytes: indexer_ape.1,
            indexer_kv_offset: indexer_kv.0,
            indexer_kv_bytes: indexer_kv.1,
            indexer_gate_offset: indexer_gate.0,
            indexer_gate_bytes: indexer_gate.1,
            indexer_norm_offset: indexer_compressor_norm.0,
            indexer_norm_bytes: indexer_compressor_norm.1,
        };
        Ok(Self {
            weights,
            compressor,
            indexer_q,
            indexer_weight,
        })
    }
}

fn run_prefill_chunk_layer(
    model: &MappedModel,
    context: &Context,
    layer: u32,
    start: u32,
    prepared: &PreparedPrefillChunkLayer,
    collect_checks: bool,
    profile: &mut LongPrefillProfile,
) -> Result<(f64, f64)> {
    let mut error = [0 as c_char; ERROR_BYTES];
    macro_rules! checked {
        ($call:expr) => {{
            if unsafe { $call } == 0 {
                return Err(Error::invalid(format!(
                    "chunk {start} layer {layer}: {}",
                    error_text(&error)
                )));
            }
        }};
    }
    let mut raw = RawSparseIndexedResult::default();
    let bind_start = profile.start();
    if layer >= 3 {
        checked!(rust_star_metal_bind_long_prefill_layer(
            context.0,
            layer,
            error.as_mut_ptr(),
            error.len()
        ));
    }
    profile.record(
        start,
        layer,
        layer,
        "bind",
        bind_start,
        (0.0, 0.0),
        (0.0, 0.0),
    );
    let stage_start = profile.start();
    if layer % 2 == 1 {
        let mut inputs: Vec<Vec<f32>> = [4096, 4096, 1024, 1024, 512, 512, 32768, 512, 512]
            .iter()
            .map(|width| {
                if collect_checks {
                    vec![0.0; 4096 * width]
                } else {
                    Vec::new()
                }
            })
            .collect();
        for position in (start..start + 4096).step_by(4096) {
            checked!(rust_star_metal_prepare_prefill_layer5_continuation_tile(
                context.0,
                position,
                error.as_mut_ptr(),
                error.len()
            ));
            checked!(rust_star_metal_run_prefill_layer3_continuation_ingress(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &prepared.weights,
                position,
                prefill_output_ptr(&mut inputs[0], collect_checks),
                prefill_output_ptr(&mut inputs[1], collect_checks),
                prefill_output_ptr(&mut inputs[2], collect_checks),
                prefill_output_ptr(&mut inputs[3], collect_checks),
                prefill_output_ptr(&mut inputs[4], collect_checks),
                prefill_output_ptr(&mut inputs[5], collect_checks),
                prefill_output_ptr(&mut inputs[6], collect_checks),
                prefill_output_ptr(&mut inputs[7], collect_checks),
                prefill_output_ptr(&mut inputs[8], collect_checks),
                &mut raw,
                error.as_mut_ptr(),
                error.len()
            ));
        }
        profile.record(
            start,
            layer,
            layer,
            "ingress",
            stage_start,
            (0.0, 0.0),
            (raw.wall_ms, raw.gpu_ms),
        );
        let stage_start = profile.start();
        let before = (raw.wall_ms, raw.gpu_ms);
        checked!(
            rust_star_metal_run_prefill_layer3_continuation_batch_compressor(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &prepared.compressor,
                start,
                &mut raw,
                error.as_mut_ptr(),
                error.len()
            )
        );
        let mut compressed = 0;
        checked!(rust_star_metal_finish_prefill_layer5_continuation(
            context.0,
            prefill_output_ptr(std::slice::from_mut(&mut compressed), collect_checks),
            1,
            error.as_mut_ptr(),
            error.len()
        ));
        profile.record(
            start,
            layer,
            layer,
            "compressor",
            stage_start,
            before,
            (raw.wall_ms, raw.gpu_ms),
        );
        let stage_start = profile.start();
        let before = (raw.wall_ms, raw.gpu_ms);
        checked!(
            rust_star_metal_run_prefill_layer3_continuation_batch_attention(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                prepared.weights.attn_sinks_offset,
                prepared.weights.attn_sinks_bytes,
                start,
                &mut raw,
                error.as_mut_ptr(),
                error.len()
            )
        );
        profile.record(
            start,
            layer,
            layer,
            "attention",
            stage_start,
            before,
            (raw.wall_ms, raw.gpu_ms),
        );
        let stage_start = profile.start();
        let before = (raw.wall_ms, raw.gpu_ms);
        checked!(rust_star_metal_run_prefill_layer2_continuation_tail(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &prepared.weights,
            5,
            start,
            4096,
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
            &mut raw,
            error.as_mut_ptr(),
            error.len()
        ));
        profile.record(
            start,
            layer,
            layer,
            "tail",
            stage_start,
            before,
            (raw.wall_ms, raw.gpu_ms),
        );
        profile.record_tail(context, start, layer, raw.gpu_ms - before.1)?;
        let padding_dispatch = u32::from((4224 + (start + 4096) / 128) % 64 != 0);
        // One ingress call contributes 10 dispatches and nine model views.
        if raw.dispatches != 47 + padding_dispatch
            || raw.wrapped_model_ranges != 28
            || raw.pointer_matches != 28
        {
            return Err(Error::invalid(format!(
                "chunk {start} odd layer {layer} schedule mismatch: {}/{}/{}",
                raw.dispatches, raw.wrapped_model_ranges, raw.pointer_matches
            )));
        }
    } else {
        if layer == 2 {
            checked!(rust_star_metal_prepare_long_prefill_layer2_q(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                prepared.weights.q_b_offset,
                prepared.weights.q_b_bytes,
                &mut raw,
                error.as_mut_ptr(),
                error.len()
            ));
        } else {
            let mut boundary = [0; 10];
            checked!(rust_star_metal_run_prefill_layer4_continuation_boundary(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &prepared.weights,
                &prepared.compressor,
                start,
                prefill_output_ptr(&mut boundary, collect_checks),
                boundary.len(),
                &mut raw,
                error.as_mut_ptr(),
                error.len()
            ));
            if raw.dispatches != 40 || raw.wrapped_model_ranges != 17 || raw.pointer_matches != 17 {
                return Err(Error::invalid("even chunk compressor schedule mismatch"));
            }
        }
        profile.record(
            start,
            layer,
            layer,
            "boundary",
            stage_start,
            (0.0, 0.0),
            (raw.wall_ms, raw.gpu_ms),
        );
        let scratch_start = profile.start();
        let mut scores = if collect_checks {
            vec![0.0; 4096 * ((start + 4096) / 4) as usize]
        } else {
            Vec::new()
        };
        let mut topk = if collect_checks {
            vec![0; 4096 * 512]
        } else {
            Vec::new()
        };
        let mut heads = if collect_checks {
            vec![0.0; 4096 * 32768]
        } else {
            Vec::new()
        };
        profile.record(
            start,
            layer,
            layer,
            "host_scratch",
            scratch_start,
            (0.0, 0.0),
            (0.0, 0.0),
        );
        checked!(rust_star_metal_begin_long_prefill_even_attention(
            context.0,
            error.as_mut_ptr(),
            error.len()
        ));
        for position in (start..start + 4096).step_by(4096) {
            let mut tile = RawSparseIndexedResult::default();
            let sparse = if layer == 2 {
                rust_star_metal_run_prefill_layer2_continuation_sparse_tile
            } else {
                rust_star_metal_run_prefill_layer4_continuation_sparse_tile
            };
            let stage_start = profile.start();
            checked!(sparse(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                prepared.indexer_q.0,
                prepared.indexer_q.1,
                prepared.indexer_weight.0,
                prepared.indexer_weight.1,
                prepared.weights.attn_sinks_offset,
                prepared.weights.attn_sinks_bytes,
                position,
                prefill_output_ptr(&mut scores, collect_checks),
                prefill_output_ptr(&mut topk, collect_checks),
                prefill_output_ptr(&mut heads, collect_checks),
                &mut tile,
                error.as_mut_ptr(),
                error.len()
            ));
            profile.record(
                start,
                layer,
                layer,
                "attention",
                stage_start,
                (0.0, 0.0),
                (tile.wall_ms, tile.gpu_ms),
            );
            // 1024-column sorted blocks retain 512 entries; each merge doubles
            // their width until the final 512-index selection is produced.
            let blocks = ((start + 4096) / 4).div_ceil(1024);
            let merges = blocks.next_power_of_two().trailing_zeros();
            if tile.dispatches != 10 + merges
                || tile.wrapped_model_ranges != 3
                || tile.pointer_matches != 3
            {
                return Err(Error::invalid("even chunk attention schedule mismatch"));
            }
            raw.wall_ms += tile.wall_ms;
            raw.gpu_ms += tile.gpu_ms;
        }
        let stage_start = profile.start();
        let mut tail = RawSparseIndexedResult::default();
        checked!(rust_star_metal_run_prefill_layer2_continuation_tail(
            context.0,
            model.mapping_pointer(),
            model.bytes(),
            &prepared.weights,
            if layer == 2 { 2 } else { 4 },
            start,
            4096,
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
            &mut tail,
            error.as_mut_ptr(),
            error.len()
        ));
        profile.record(
            start,
            layer,
            layer,
            "tail",
            stage_start,
            (0.0, 0.0),
            (tail.wall_ms, tail.gpu_ms),
        );
        profile.record_tail(context, start, layer, tail.gpu_ms)?;
        if tail.dispatches != 25 || tail.wrapped_model_ranges != 14 || tail.pointer_matches != 14 {
            return Err(Error::invalid("even chunk batched tail schedule mismatch"));
        }
        raw.wall_ms += tail.wall_ms;
        raw.gpu_ms += tail.gpu_ms;
    }
    Ok((raw.wall_ms, raw.gpu_ms))
}

fn rebuild_long_prefill_prefix(
    model: &MappedModel,
    context: &Context,
    prepared: &[PreparedPrefillChunkLayer],
    collect_checks: bool,
    profile: &mut LongPrefillProfile,
) -> Result<(f64, f64)> {
    let mut error = [0 as c_char; ERROR_BYTES];
    macro_rules! checked {
        ($call:expr) => {
            if unsafe { $call } == 0 {
                return Err(Error::invalid(error_text(&error)));
            }
        };
    }
    let mut total = (0.0, 0.0);
    let mut actual = [0; 43];
    checked!(rust_star_metal_begin_long_prefill_prefix(
        context.0,
        prefill_output_ptr(&mut actual, collect_checks),
        error.as_mut_ptr(),
        error.len()
    ));
    for layer in 0..43 {
        if layer >= 3 {
            let stage_start = profile.start();
            let p = &prepared[layer as usize - 2];
            checked!(rust_star_metal_bind_long_prefill_prefix(
                context.0,
                layer,
                error.as_mut_ptr(),
                error.len()
            ));
            let mut raw = RawSparseIndexedResult::default();
            let mut preceding = (0.0, 0.0);
            if layer % 2 == 1 {
                checked!(rust_star_metal_run_prefill_layer5_prefix_ingress(
                    context.0,
                    model.mapping_pointer(),
                    model.bytes(),
                    &p.weights,
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len()
                ));
                checked!(
                    rust_star_metal_run_prefill_layer3_continuation_batch_compressor(
                        context.0,
                        model.mapping_pointer(),
                        model.bytes(),
                        &p.compressor,
                        0,
                        &mut raw,
                        error.as_mut_ptr(),
                        error.len()
                    )
                );
                checked!(
                    rust_star_metal_run_prefill_layer3_continuation_batch_attention(
                        context.0,
                        model.mapping_pointer(),
                        model.bytes(),
                        p.weights.attn_sinks_offset,
                        p.weights.attn_sinks_bytes,
                        0,
                        &mut raw,
                        error.as_mut_ptr(),
                        error.len()
                    )
                );
            } else {
                let mut boundary = [0; 10];
                checked!(rust_star_metal_run_prefill_layer4_continuation_boundary(
                    context.0,
                    model.mapping_pointer(),
                    model.bytes(),
                    &p.weights,
                    &p.compressor,
                    0,
                    prefill_output_ptr(&mut boundary, collect_checks),
                    boundary.len(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len()
                ));
                preceding = (raw.wall_ms, raw.gpu_ms);
                let mut sparse = [0; 5];
                checked!(rust_star_metal_run_prefill_layer4_prefix_sparse_batch(
                    context.0,
                    model.mapping_pointer(),
                    model.bytes(),
                    p.indexer_q.0,
                    p.indexer_q.1,
                    p.indexer_weight.0,
                    p.indexer_weight.1,
                    p.weights.attn_sinks_offset,
                    p.weights.attn_sinks_bytes,
                    prefill_output_ptr(&mut sparse, collect_checks),
                    sparse.len(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len()
                ));
            }
            checked!(rust_star_metal_run_prefill_layer2_continuation_tail(
                context.0,
                model.mapping_pointer(),
                model.bytes(),
                &p.weights,
                if layer % 2 == 1 { 5 } else { 4 },
                0,
                4096,
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
                &mut raw,
                error.as_mut_ptr(),
                error.len()
            ));
            checked!(rust_star_metal_finish_long_prefill_prefix(
                context.0,
                layer,
                prefill_output_ptr(
                    std::slice::from_mut(&mut actual[layer as usize]),
                    collect_checks
                ),
                error.as_mut_ptr(),
                error.len()
            ));
            profile.record(
                0,
                layer,
                layer,
                "prefix_layer",
                stage_start,
                (0.0, 0.0),
                (preceding.0 + raw.wall_ms, preceding.1 + raw.gpu_ms),
            );
            total.0 += preceding.0 + raw.wall_ms;
            total.1 += preceding.1 + raw.gpu_ms;
        }
        if collect_checks && actual[layer as usize] != LONG_PREFILL_PREFIX_CHECKSUMS[layer as usize]
        {
            return Err(Error::invalid(format!(
                "4K prefix layer {layer} HC differs: actual={:#018x} expected={:#018x}",
                actual[layer as usize], LONG_PREFILL_PREFIX_CHECKSUMS[layer as usize]
            )));
        }
        if collect_checks {
            eprintln!("rust-star: prefix layer {layer} exact");
        }
    }
    Ok(total)
}

fn run_long_prefill_with_context(
    model: &MappedModel,
    frontier: u32,
    retained_context: Option<Context>,
    collect_checks: bool,
    validate_outputs: bool,
    collect_profile: bool,
) -> Result<(Context, LongPrefillChunkProbeReport, NativePrefillTiming)> {
    if frontier < 12288 || frontier > 32768 || frontier % 4096 != 0 {
        return Err(Error::invalid(
            "long prefill requires a 4096-token frontier from 12288 through 32768",
        ));
    }
    let (tokens, expected_logits) = if validate_outputs {
        prefill_frontier_32768_fixture()?
    } else {
        (
            decode_u32_fixture(PREFILL_FRONTIER_32768_TOKEN_IDS_BYTES, "32K prompt tokens")?,
            Vec::new(),
        )
    };
    let (context, context_ms) = match retained_context {
        Some(context) => (context, 0.0),
        None => {
            let started = Instant::now();
            let context = Context::new()?;
            (context, started.elapsed().as_secs_f64() * 1000.0)
        }
    };
    let mut profile = LongPrefillProfile {
        enabled: collect_profile,
        ..Default::default()
    };
    let stage_start = profile.start();
    let (context, mut timing) =
        run_long_prefill_initial_bootstrap(model, context, context_ms, &tokens)?;
    profile.record(
        0,
        0,
        2,
        "initial_bootstrap",
        stage_start,
        (0.0, 0.0),
        (
            timing.tile_wall_ms + timing.transformer_wall_ms,
            timing.tile_gpu_ms + timing.transformer_gpu_ms,
        ),
    );
    if collect_checks {
        eprintln!("rust-star: native 4096-token bootstrap ready for exact prefix rebuild");
    }
    let prefix_started = Instant::now();
    let prepared = (2..43)
        .map(|layer| PreparedPrefillChunkLayer::new(model, layer))
        .collect::<Result<Vec<_>>>()?;
    let prefix =
        rebuild_long_prefill_prefix(model, &context, &prepared, collect_checks, &mut profile)?;
    timing.transformer_wall_ms += prefix.0;
    timing.transformer_gpu_ms += prefix.1;
    timing.transformer_setup_ms +=
        (prefix_started.elapsed().as_secs_f64() * 1000.0 - prefix.0).max(0.0);
    if validate_outputs && !collect_checks {
        eprintln!("rust-star: output-only gate completed the native 4K prefix");
    }
    if collect_profile {
        let mut error = [0 as c_char; ERROR_BYTES];
        if unsafe {
            rust_star_metal_enable_prefill_tail_profiling(
                context.0,
                error.as_mut_ptr(),
                error.len(),
            )
        } == 0
        {
            return Err(Error::invalid(error_text(&error)));
        }
    }
    let mut chunks = Vec::new();
    for start in (4096..frontier).step_by(4096) {
        let end = start + 4096;
        let expected = if end == 8192 {
            Some(LONG_PREFILL_BOOTSTRAP_CHECKSUMS)
        } else {
            LONG_PREFILL_CHUNK_CHECKSUMS
                .iter()
                .find(|(n, _)| *n == end)
                .map(|(_, checksums)| *checksums)
        };
        let mut error = [0 as c_char; ERROR_BYTES];
        if unsafe {
            rust_star_metal_begin_long_prefill_chunk(
                context.0,
                start,
                error.as_mut_ptr(),
                error.len(),
            )
        } == 0
        {
            return Err(Error::invalid(error_text(&error)));
        }
        let stage_start = profile.start();
        let (_, bootstrap_timing) = run_prefill_layers012_compressor_loop_probe_in_context(
            model,
            &context,
            false,
            start,
            end,
            Some(&tokens[..end as usize]),
        )?;
        profile.record(
            start,
            0,
            2,
            "bootstrap_ingress",
            stage_start,
            (0.0, 0.0),
            (bootstrap_timing.tile_wall_ms, bootstrap_timing.tile_gpu_ms),
        );
        timing.bootstrap_setup_ms += bootstrap_timing.bootstrap_setup_ms;
        timing.tile_wall_ms += bootstrap_timing.tile_wall_ms;
        timing.tile_gpu_ms += bootstrap_timing.tile_gpu_ms;
        let mut actual = [0; 43];
        for layer in 0..43 {
            if layer >= 2 {
                let started = Instant::now();
                let layer_timing = run_prefill_chunk_layer(
                    model,
                    &context,
                    layer,
                    start,
                    &prepared[layer as usize - 2],
                    collect_checks,
                    &mut profile,
                )?;
                timing.transformer_wall_ms += layer_timing.0;
                timing.transformer_gpu_ms += layer_timing.1;
                timing.transformer_setup_ms +=
                    (started.elapsed().as_secs_f64() * 1000.0 - layer_timing.0).max(0.0);
            }
            if unsafe {
                rust_star_metal_finish_long_prefill_layer(
                    context.0,
                    layer,
                    prefill_output_ptr(
                        std::slice::from_mut(&mut actual[layer as usize]),
                        collect_checks,
                    ),
                    error.as_mut_ptr(),
                    error.len(),
                )
            } == 0
            {
                return Err(Error::invalid(error_text(&error)));
            }
            if collect_checks {
                if let Some(expected) = expected {
                    if actual[layer as usize] != expected[layer as usize] {
                        return Err(Error::invalid(format!("chunk {start} layer {layer} final HC mismatch: actual={:#018x} expected={:#018x}",actual[layer as usize],expected[layer as usize])));
                    }
                    eprintln!("rust-star: chunk {start} layer {layer} exact");
                } else {
                    eprintln!("rust-star: chunk {start} layer {layer} executed, diagnostic HC={:#018x} (no repeated internal target)", actual[layer as usize]);
                }
            }
        }
        if end == 8192 && collect_checks {
            // Keep the previously accepted full vocabulary as an independent control.
            let mut head = PreparedOutputHead::new(model)?;
            run_sampling_output_head(model, &context, &mut head)?;
            let (_, expected) = prefill_frontier_8192_fixture()?;
            if head
                .logits
                .iter()
                .map(|v| v.to_bits())
                .ne(expected.iter().map(|v| v.to_bits()))
            {
                return Err(Error::invalid(
                    "uniform long-prefill 8K full-logit control differs",
                ));
            }
            eprintln!("rust-star: uniform 8192-token full-logit control exact");
        } else if end > 8192 && collect_checks {
            chunks.push((end, actual));
        }
        if validate_outputs && !collect_checks {
            eprintln!("rust-star: output-only gate completed native prefill through {end} tokens");
        }
    }
    let head_started = Instant::now();
    let mut head = PreparedOutputHead::new(model)?;
    timing.output_head_setup_ms += head_started.elapsed().as_secs_f64() * 1000.0;
    let head_started = Instant::now();
    let (selected_token, gpu_ms) = if validate_outputs {
        run_checked_top1_output_head(model, &context, &mut head)?
    } else {
        run_top1_output_head(model, &context, &head)?
    };
    timing.output_head_wall_ms += head_started.elapsed().as_secs_f64() * 1000.0;
    timing.output_head_gpu_ms += gpu_ms;
    let mut checked_logits = 0;
    if frontier == 32768 && validate_outputs {
        if head.logits.len() != expected_logits.len() {
            return Err(Error::invalid("32K vocabulary length mismatch"));
        }
        for (index, (actual, expected)) in head.logits.iter().zip(&expected_logits).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!(
                    "32K full-logit C0 mismatch at {index}: actual={:#010x} expected={:#010x}",
                    actual.to_bits(),
                    expected.to_bits()
                )));
            }
        }
        checked_logits = expected_logits.len() as u32;
    }
    let (checked_decode_steps, decode_logits_checksum, decode_tokens_checksum) =
        if frontier == 32768 && validate_outputs {
            run_long_prefill_decode(model, &context, selected_token)?
        } else {
            (0, 0, 0)
        };
    timing.long_prefill_profile = profile.json();
    Ok((
        context,
        LongPrefillChunkProbeReport {
            frontier,
            internal_hc_collection: collect_checks,
            chunks,
            checked_decode_steps,
            decode_logits_checksum,
            decode_tokens_checksum,
            selected_token,
            checked_logits,
            logits_checksum: if validate_outputs {
                checksum_f32(&head.logits)
            } else {
                0
            },
        },
        timing,
    ))
}

pub fn run_long_prefill_chunk_probe(
    model: &MappedModel,
    frontier: u32,
) -> Result<LongPrefillChunkProbeReport> {
    Ok(run_long_prefill_with_context(model, frontier, None, true, true, false)?.1)
}

fn run_long_prefill_decode(
    model: &MappedModel,
    context: &Context,
    first_token: u32,
) -> Result<(u32, u64, u64)> {
    let (tokens, expected) = prefill_decode_32768_fixture()?;
    if first_token != tokens[0] {
        return Err(Error::invalid(
            "native 32K prefill selection differs from the decode reference",
        ));
    }
    context.adopt_prefill_decoder_state(32768, 32896)?;
    context.prepare_model_residency()?;
    let mut layers = (0..43)
        .map(|layer| PreparedLayerExecution::new_cold_with_capacity(model, layer, 32896))
        .collect::<Result<Vec<_>>>()?;
    for layer in &mut layers {
        layer.validate_expected = false;
        layer.collect_outputs = false;
    }
    context.prepare_decoder()?;
    let mut head = PreparedOutputHead::new(model)?;
    let mut input = first_token;
    let mut actual_logits = Vec::with_capacity(expected.len());
    let mut selected = Vec::with_capacity(128);
    for step in 0..128 {
        let position = 32768 + step;
        submit_prepared_layers(model, context, &mut layers, input, position)?;
        run_prepared_layer_iterations(
            model,
            context,
            &mut layers[0],
            input,
            position,
            0,
            1,
            COMMAND_CHAINED_TIMING,
            42,
        )?;
        let (token, _) = run_checked_top1_output_head(model, context, &mut head)?;
        let row = &expected[step as usize * 129280..(step as usize + 1) * 129280];
        if head.logits.len() != row.len() {
            return Err(Error::invalid("32K decode vocabulary length differs"));
        }
        for (index, (actual, expected)) in head.logits.iter().zip(row).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::invalid(format!("32K decode position {position} logit {index} differs: actual={:#010x} expected={:#010x}",actual.to_bits(),expected.to_bits())));
            }
        }
        if token != tokens[step as usize + 1] {
            return Err(Error::invalid(format!(
                "32K decode token differs at position {position}"
            )));
        }
        actual_logits.extend_from_slice(&head.logits);
        selected.push(token);
        input = token;
        eprintln!("rust-star: decode position {position} all 129280 logits exact, token {token}");
    }
    Ok((128, checksum_f32(&actual_logits), checksum_u32(&selected)))
}

// This gate exercises the timing path's prefill without internal readbacks,
// then verifies every output word before making any measurement claim.
pub fn run_long_prefill_output_probe(model: &MappedModel) -> Result<LongPrefillChunkProbeReport> {
    Ok(run_long_prefill_with_context(model, 32768, None, false, true, false)?.1)
}

// Encoder-split diagnostics must satisfy the same complete output gate.
pub fn run_long_prefill_profile_probe(
    model: &MappedModel,
) -> Result<(LongPrefillChunkProbeReport, String)> {
    let (_, report, timing) = run_long_prefill_with_context(model, 32768, None, false, true, true)?;
    let profile = timing
        .long_prefill_profile
        .ok_or_else(|| Error::invalid("missing long-prefill profile"))?;
    Ok((report, profile))
}

#[cfg(test)]
mod tail_profile_tests {
    use super::*;

    #[test]
    fn attributes_each_tail_component_without_adding_parent_rows() {
        let mut profile = LongPrefillProfile {
            enabled: true,
            ..Default::default()
        };
        let ticks = std::array::from_fn(|i| 100 + i as u64 * 10);
        profile.record_tail_ticks(4096, 2, 90.0, &ticks).unwrap();
        assert_eq!(profile.tail_components.len(), 9);
        assert!(profile.rows.is_empty());
        assert!(profile
            .tail_components
            .iter()
            .all(|s| s.contains("\"attributed_gpu_ms\":10.000000000")));
        let json = profile.json().unwrap();
        assert!(json.contains("\"profile_perturbed\":true"));
        assert!(json.contains("\"paired_ineligible\":true"));
        assert!(json.contains("\"start_tick\":100,\"end_tick\":110"));
    }

    #[test]
    fn rejects_invalid_tail_timing_before_recording() {
        let mut profile = LongPrefillProfile::default();
        let ticks = std::array::from_fn(|i| i as u64 + 1);
        for duration in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(profile
                .record_tail_ticks(4096, 2, duration, &ticks)
                .is_err());
        }
        let mut bad = ticks;
        bad[5] = bad[4];
        assert!(profile.record_tail_ticks(4096, 2, 90.0, &bad).is_err());
        bad[5] = 0;
        assert!(profile.record_tail_ticks(4096, 2, 90.0, &bad).is_err());
        bad = ticks;
        bad[17] = u64::MAX;
        let error = profile.record_tail_ticks(4096, 2, 90.0, &bad).unwrap_err();
        assert!(error.to_string().contains("chunk=4096 layer=2"));
        assert!(error.to_string().contains("ffffffffffffffff"));
        assert!(profile.tail_components.is_empty());
        assert!(profile.json().is_none());
    }
}
