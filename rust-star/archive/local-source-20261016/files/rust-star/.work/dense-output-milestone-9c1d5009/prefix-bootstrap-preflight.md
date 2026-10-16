Read-only next-optimization preflight (2026-09-08). No runtime change while 92ca3ef3
is frozen. Resolve the current candidate before implementing this hypothesis.

The aggregate profile includes initial_bootstrap (about 29 GPU seconds in the tail
baseline), followed by prefix_layer (about 22 GPU seconds). Source suggests avoidable
work: run_long_prefill_with_context first invokes the monolithic full 4K path, then
rebuild_long_prefill_prefix recomputes layers 3..42 using the accepted C0 schedules.

Evidence:
- Journal 2026-09-06 notes the monolithic 4K path was not a complete C0 prefix;
  exact per-layer rebuilding was required after the layer 5 failure.
- metal.rs run_prefill_layers012_attention_loop_with_context calls the giant
  rust_star_metal_run_prefill_layer2_attention and then runs the output head even
  when used only as the initial long-prefill bootstrap.
- metal_shim.m rust_star_metal_begin_long_prefill_prefix requires only 4096 rows
  and context.prefillLayer3InputHc (completed layers 0..2). It initializes a new
  history dictionary and saves states only for layers 0..2.
- The monolithic C function sets context.prefillLayer3InputHc=after_ffn_hc_buffer
  at its final completion section. That buffer is the layer 2 result. Later layer
  results appear to be overwritten by the explicit prefix rebuild.

Potential milestone: provide an internal layers 0..2-only bootstrap scope,
retain the exact layer 2 HC/state, and enter the existing prefix rebuild directly.
Do not replace the correct rebuild with the rejected monolithic 4K result.

Before implementation, audit all remaining state dependencies. The giant C helper
allocates/wraps later-layer resources upfront, so an early stop may initially
remove GPU work without removing all setup allocations. Track actual dispatch and
model-view counts, timestamp accounting and ownership; do not return fake full-model
metadata or run an output head from incomplete state. Preserve 2K/legacy controls.

Require native 12K and full 32K C0, retained lifecycle, separate profile and paired
measurement. The expected benefit is a hypothesis; no speedup is established.

Concrete boundary identified in current source (line numbers are only navigation):
- C entry rust_star_metal_run_prefill_layer2_attention begins near7024; its final
  prefill_rows/collect_outputs arguments are near8098. Matching C header and Rust
  extern declaration exist. There is one Rust FFI call near35182.
- First layer2 phase ends after the final HC dispatch immediately before
  RUST_STAR_PREFILL_LAYER_BOUNDARY(1u), near18595. An explicit bootstrap-only scope
  could end/commit/wait this command, retain after_ffn_hc_buffer as
  context.prefillLayer3InputHc and return real scope metadata.
- That4K layer2 phase has36 explicit/macro dispatches plus the return count from
  encode_indexer_topk, whose count depends on the compiled pipeline thread limit:
  37 total only if its1024 columns fit one sort block, versus the full helper
  reporting2396. Query/record the actual helper count; do not assume one. Capture the helper's returned dispatch count
  instead of silently assuming it. This does not include earlier layers0/1 and
  layer2 ingress/compressor work, already charged by the bootstrap helper.
- All1218 views are presently wrapped upfront even if the later GPU phases are
  skipped; report that real count until setup is narrowed. Layer3 compressed-row
  result must be0 in a layers0..2 scope, not a fake completed-layer value.
- Rust near36269 validates2396/1218 and then runs a top1 output head. A partial
  scope must validate its own metadata, return no selected token, and skip the
  output head. Prefer an explicit private scope/optional selection or narrow
  bootstrap wrapper; keep existing complete callers' contracts intact.
- LongPrefillProfile initial_bootstrap range should become0..2, not0..42.

These are source-derived implementation notes, not applied changes or proof that
all state dependencies are dispensable. Full exact/lifecycle gates remain required.

Private API design to consider after this milestone:
- Keep existing complete-prefill wrapper signatures unchanged; delegate them to
  an internal scope-aware implementation that returns an optional selected token.
  Complete scope requires Some(token); a narrow bootstrap helper requires None
  and returns only Context plus NativePrefillTiming to the long-prefill driver.
- Add a checked C scope (completed layer2 or42), and truthful result fields for
  last completed layer and actual layer2 indexer dispatch count. Do not label a
  partial run with complete2396-dispatch metadata or manufacture a selected token.
- Count36 non-selector dispatches plus encode_indexer_topk's real return value
  in bootstrap mode. Validate this result separately from legacy full modes.
- Preserve all current complete-mode returns (the second one is near44311 in
  metal.rs). No source edit or build for this hypothesis has been performed.

Additional read-only review after warmups:
- save_prefill_chunk_state captures only FullKv, attention compressor state and
  indexer compressor state from slots 0..2 at prefix entry. Those fields are
  produced before the proposed early return, while the layer 2 HC must be retained
  explicitly as drafted. Later-layer full outputs are not saved at this boundary.
- release_prefill_layer_buffers clears all prefillLayer object properties and
  transient activation cache entries at the decode/reset ownership boundary.
  Retained lifecycle must still be tested after the partial bootstrap change.
- The profile analyzer currently checks continuation layer coverage only. For the
  bootstrap milestone, extend it to require initial_bootstrap range 0..2 and
  prefix_layer ranges exactly 3..42; do not merely keep the old 1483 row count.
- Compare the new profile with the dense-output component-enabled profile if
  retained. The older 89 aggregate tail profile did not split nine encoders, so
  its GPU component comparisons have an instrumentation confound.

Follow-on sequencing after a successful early-return milestone:
The shared C setup still wraps all 1218 views and allocates later-layer buffers
before command encoding. Keep that behavior for the first change to narrow the
correctness surface, then use measured host setup/peak-memory evidence to decide
whether a second scope-specific setup path is worthwhile. Do not claim that the
early return has removed those allocations merely because later results are no
longer retained. No layer-2 property writes occur in the full helper's final
completion block; its history is already retained by preceding calls.
