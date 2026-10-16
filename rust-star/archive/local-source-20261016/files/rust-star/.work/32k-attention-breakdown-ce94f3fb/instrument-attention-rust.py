from pathlib import Path
root=Path(__file__).resolve().parents[3];p=root/'rust-star/runtime/src/long_prefill.rs';s=p.read_text()
s=s.replace('    tail_components: Vec<String>,','    tail_components: Vec<String>,\n    attention_components: Vec<String>,',1)
anchor='    fn json(self) -> Option<String> {'
assert s.count(anchor)==1
methods='''    fn record_attention(
        &mut self,
        context: &Context,
        chunk: u32,
        layer: u32,
        gpu_ms: f64,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let sample_count = if layer % 2 == 0 { 18 } else { 8 };
        let mut ticks = [0u64; 18];
        let mut error = [0 as c_char; ERROR_BYTES];
        if unsafe {
            rust_star_metal_copy_prefill_attention_ticks(
                context.0,
                sample_count as u32,
                ticks.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        } == 0
        {
            return Err(Error::invalid(format!(
                "attention profile chunk={chunk} layer={layer}: {}",
                error_text(&error)
            )));
        }
        self.record_attention_ticks(chunk, layer, gpu_ms, &ticks[..sample_count])
    }
    fn record_attention_ticks(
        &mut self,
        chunk: u32,
        layer: u32,
        gpu_ms: f64,
        ticks: &[u64],
    ) -> Result<()> {
        let indexed = layer % 2 == 0;
        let (path, names): (&str, &[&str]) = if indexed {
            ("indexed", &[
                "indexer_q_projection", "indexer_q_transform", "indexer_weight_projection",
                "indexer_scores", "topk_selection", "topk_ordering", "kv_conversion",
                "indexed_attention", "inverse_rope",
            ])
        } else {
            ("compressed", &["kv_conversion", "mask_blocks", "compressed_attention", "inverse_rope"])
        };
        if chunk < 4096 || chunk % 4096 != 0 || !(2..=42).contains(&layer)
            || ticks.len() != 2 * names.len()
            || ticks.contains(&u64::MAX)
            || ticks.windows(2).any(|p| p[1] <= p[0])
            || !gpu_ms.is_finite() || gpu_ms <= 0.0
        {
            return Err(Error::invalid(format!(
                "invalid attention profile chunk={chunk} layer={layer} gpu_ms={gpu_ms}: {ticks:x?}"
            )));
        }
        let spans = ticks.chunks_exact(2).map(|p| p[1] - p[0]).collect::<Vec<_>>();
        let total = spans.iter().map(|x| u128::from(*x)).sum::<u128>() as f64;
        for (i, name) in names.iter().enumerate() {
            self.attention_components.push(format!(
                "{{\\"chunk_start\\":{chunk},\\"layer\\":{layer},\\"path\\":\\"{path}\\",\\"component\\":\\"{name}\\",\\"start_tick\\":{},\\"end_tick\\":{},\\"attributed_gpu_ms\\":{:.9}}}",
                ticks[2*i], ticks[2*i+1], gpu_ms * spans[i] as f64 / total));
        }
        Ok(())
    }
'''
s=s.replace(anchor,methods+anchor)
anchor='        let components = self.tail_components.join(",");'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n        let attention = self.attention_components.join(",");')
anchor=r'\"rows\":[{components}]}}}}"))'
assert s.count(anchor)==1
s=s.replace(anchor,r'\"rows\":[{components}]}},\"attention_components\":{{\"schema\":\"rust-star-attention-components-v1\",\"method\":\"metal-compute-stage-boundary-timestamp-counters\",\"profile_perturbed\":true,\"encoder_splitting\":true,\"encoder_serialization\":\"MTLFence\",\"scaling\":\"counter shares scaled to each completed attention command GPU interval; do not add to parent rows\",\"rows\":[{attention}]}}}}"))')
anchor='''            "attention",
            stage_start,
            before,
            (raw.wall_ms, raw.gpu_ms),
        );'''
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n        profile.record_attention(context, start, layer, raw.gpu_ms - before.1)?;')
anchor='''                "attention",
                stage_start,
                (0.0, 0.0),
                (tile.wall_ms, tile.gpu_ms),
            );'''
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n            profile.record_attention(context, start, layer, tile.gpu_ms)?;')
start=s.index('    if collect_profile {\n        let mut error =',s.index('let prefix_started ='))
end=s.index('    let mut chunks = Vec::new();',start)
f=s[start:end];anchor='''        {
            return Err(Error::invalid(error_text(&error)));
        }
'''
assert f.count(anchor)==1
f=f.replace(anchor,anchor+'''        if unsafe {
            rust_star_metal_enable_prefill_attention_profiling(
                context.0,
                error.as_mut_ptr(),
                error.len(),
            )
        } == 0
        {
            return Err(Error::invalid(error_text(&error)));
        }
''');s=s[:start]+f+s[end:]
p.write_text(s)
